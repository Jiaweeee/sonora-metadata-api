import argparse
import asyncio
from datetime import timedelta
import logging
from timeit import default_timer as timer

import aiohttp

from lidarrmetadata.config import get_config
from lidarrmetadata import provider
from lidarrmetadata.provider import ProviderUnavailableException
from lidarrmetadata import util
from lidarrmetadata import limit
from lidarrmetadata.api import get_artist_info_multi

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info('Have crawler logger')

CONFIG = get_config()

# Global shared instances of providers
_SPOTIFY_PROVIDER = None
_FANART_PROVIDER = None

# Global shared session
_SESSION = None

async def get_shared_session():
    """
    Get or create a shared aiohttp session
    
    Returns:
        aiohttp.ClientSession: A shared session instance
    """
    global _SESSION
    if _SESSION is None:
        logger.info("Creating global shared aiohttp session")
        _SESSION = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=CONFIG.EXTERNAL_TIMEOUT / 1000),
            trust_env=True
        )
    return _SESSION

async def get_spotify_provider():
    """
    获取或初始化全局共享的 SpotifyProvider 实例
    
    Returns:
        SpotifyProvider: 全局共享的 SpotifyProvider 实例
    """
    global _SPOTIFY_PROVIDER
    if _SPOTIFY_PROVIDER is None:
        logger.info("Initializing global SpotifyProvider")
        credentials = CONFIG.SPOTIFY_CREDENTIALS
        if not credentials:
            logger.warning("No Spotify credentials provided")
            return None
            
        try:
            session = await get_shared_session()
            # 使用共享的 session 初始化 SpotifyProvider
            _SPOTIFY_PROVIDER = provider.SpotifyProvider(
                credentials=credentials,
                session=session
            )
        except Exception as e:
            logger.error(f"Failed to initialize SpotifyProvider: {str(e)}")
            return None
    
    return _SPOTIFY_PROVIDER

async def get_fanart_provider():
    """
    获取或初始化全局共享的 FanArtTvProvider 实例
    
    Returns:
        FanArtTvProvider: 全局共享的 FanArtTvProvider 实例
    """
    global _FANART_PROVIDER
    if _FANART_PROVIDER is None:
        logger.info("Initializing global FanArtTvProvider")
        try:
            session = await get_shared_session()
            # 使用共享的 session 初始化 FanArtTvProvider
            _FANART_PROVIDER = provider.FanArtTvProvider(
                api_key=CONFIG.FANART_KEY,
                session=session
            )
        except Exception as e:
            logger.error(f"Failed to initialize FanArtTvProvider: {str(e)}")
            return None
    
    return _FANART_PROVIDER

async def update_wikipedia(count = 50, max_ttl = 60 * 60):
    
    # Use an aiohttp session which only allows a single concurrent connection per host to be nice
    # https://www.mediawiki.org/wiki/API:Etiquette
    # Only put timeout on sock_read - otherwise we can get timed out waiting for a connection from the pool.
    # Don't make these count towards rate limiting.
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(sock_read = 2), connector = aiohttp.TCPConnector(limit_per_host=1)) as session:
        wikipedia_provider = provider.WikipediaProvider(session, limit.NullRateLimiter())

        while True:
            keys = await util.WIKI_CACHE.get_stale(count, provider.utcnow() + timedelta(seconds = max_ttl))
            logger.debug(f"Got {len(keys)} stale wikipedia items to refresh")

            start = timer()
            await asyncio.gather(*(wikipedia_provider.get_artist_overview(url, ignore_cache=True) for url in keys))
            logger.debug(f"Refreshed {len(keys)} wikipedia overviews in {timer() - start:.1f}s")

            # If there weren't any to update sleep, otherwise continue
            if not keys:
                await asyncio.sleep(60)
            
def format_elapsed_time(elapsed, count=None):
    """
    Format elapsed time, optionally showing processing speed
    """
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    
    time_str = f"{hours}h {minutes}m {seconds:.1f}s" if hours > 0 else \
               f"{minutes}m {seconds:.1f}s" if minutes > 0 else \
               f"{seconds:.1f}s"
               
    if count is not None:
        return f"{time_str} ({count/elapsed:.1f} items/s)"
    return time_str

async def init_artists():
    id_provider = provider.get_providers_implementing(provider.ArtistIdListMixin)[0]
    
    start = timer()
    
    # Execute multiple requests in parallel
    page_size = 2000
    concurrent_requests = 10
    all_ids = []
    
    async def fetch_page(offset):
        ids = await id_provider.get_artist_ids_paged(limit=page_size, offset=offset)
        if ids:
            logger.debug(f"Retrieved {len(ids)} artist IDs from offset {offset}")
            return ids
        return []
    
    offset = 0
    while True:
        # Create multiple concurrent tasks
        tasks = []
        for _ in range(concurrent_requests):
            tasks.append(fetch_page(offset))
            offset += page_size
            
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)
        
        # Process results
        new_ids = [id for page_ids in results if page_ids for id in page_ids]
        if not new_ids:
            break
            
        all_ids.extend(new_ids)
        logger.debug(f"Total retrieved {len(all_ids)} artist IDs so far...")
    
    pairs = [(id, None) for id in all_ids]
    
    await util.ARTIST_CACHE.clear()
    await util.ARTIST_CACHE.multi_set(pairs, ttl=0, timeout=None)
    
    elapsed = timer() - start
    logger.info(f"Initialized {len(all_ids)} artists in {format_elapsed_time(elapsed, len(all_ids))}")

async def init_spotify():
    """
    Initialize Spotify ID mapping cache using concurrent paged loading for all mapping data
    
    Retrieves MusicBrainz ID to Spotify ID mappings from the database in batches,
    using 10 concurrent tasks, requesting 1000 records at a time, until all data is retrieved,
    and stores it in the cache
    """
    link_provider = provider.get_providers_implementing(provider.ArtistByIdMixin)[0]
    
    # Basic parameter settings
    page_size = 1000
    concurrency = 10  # Number of concurrent tasks
    offset = 0  # Start from page 0
    all_pairs = []  # Store all results
    
    # Create a global dictionary to merge results from all pages
    global_id_map = {}
    
    async def fetch_page(page_offset):
        """
        Fetch a single page of Spotify mapping data, with retry on failure
        
        Args:
            page_offset: Page offset
            
        Returns:
            A dictionary containing (spotifyid, mbid) mappings
        """
        max_retries = 3
        retry_count = 0
        backoff_time = 1.0  # Initial retry wait time (seconds)
        
        while retry_count <= max_retries:
            try:
                maps = await link_provider.get_spotify_mappings(limit=page_size, offset=page_offset)
                page_id_map = {}
                for item in maps:
                    if item['mbid'] not in page_id_map:
                        page_id_map[item['mbid']] = []
                    page_id_map[item['mbid']].append(item['spotifyid'])
                logger.debug(f"Successfully retrieved Spotify mappings, offset={page_offset}, count={len(maps)}")
                return page_id_map  # Return a dictionary instead of tuple list
            
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    # Calculate exponential backoff time
                    wait_time = backoff_time * (2 ** (retry_count - 1))
                    logger.warning(f"Failed to get Spotify mappings, offset={page_offset}, retry {retry_count}, waiting {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to get Spotify mappings, offset={page_offset}, gave up after {max_retries} retries: {str(e)}")
                    return {}
    
    while True:
        # Create tasks for the current batch
        tasks = []
        for i in range(concurrency):
            current_offset = offset + i * page_size
            tasks.append(fetch_page(current_offset))
        
        # Execute the current batch of tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results, check if there's more data
        has_more_data = False
        total_records_in_batch = 0
        
        for page_id_map in results:
            if page_id_map:  # If the page has data
                # Merge the current page's mappings to the global map
                for mbid, spotifyids in page_id_map.items():
                    if mbid not in global_id_map:
                        global_id_map[mbid] = []
                    # Add Spotify IDs without duplicates
                    for spotify_id in spotifyids:
                        if spotify_id not in global_id_map[mbid]:
                            global_id_map[mbid].append(spotify_id)
                            
                total_records_in_batch += len(page_id_map)
                if len(page_id_map) == page_size:
                    has_more_data = True  # At least one page is full, there might be more data
        
        # If no data was retrieved in this batch, we're done
        if total_records_in_batch == 0:
            logger.info("No more Spotify mapping data")
            break
            
        # If none of the pages were full, we've reached the last batch
        if not has_more_data:
            break
            
        # Update offset for the next batch
        offset += concurrency * page_size
    
    # Convert the merged global map to a list of tuples for caching
    all_pairs = [(mbid, spotifyids) for mbid, spotifyids in global_id_map.items()]
    
    logger.info(f"Loaded {len(all_pairs)} Spotify ID mappings, containing {sum(len(spotifyids) for _, spotifyids in all_pairs)} Spotify IDs")
    
    # Clear and repopulate the cache
    if all_pairs:
        await util.SPOTIFY_CACHE.clear()
        await util.SPOTIFY_CACHE.multi_set(all_pairs, ttl=None, timeout=None)
    else:
        logger.warning("No Spotify mapping data found, cache not updated")

async def filter_artists_without_images():
    """
    Get all artist IDs from SPOTIFY_CACHE and filter out artists that already have images in ARTIST_IMAGE_CACHE
    
    Returns:
        list: List of artist IDs without images
    """
    logger.info("Starting to retrieve and filter artist IDs...")
    start = timer()
    
    try:
        # Use paged method to get all artist IDs
        all_artists = []
        page_size = 10000  # Records per page
        offset = 0
        
        while True:
            # Get one page of artist IDs
            result = await util.SPOTIFY_CACHE.get_all_keys_paged(limit=page_size, offset=offset)
            
            if not result['keys']:
                break
                
            all_artists.extend(result['keys'])
            logger.debug(f"Retrieved {len(all_artists)}/{result['total']} artist IDs...")
            
            # If there's no more data, exit the loop
            if not result['has_more']:
                break
                
            # Update offset
            offset += page_size
            
        if not all_artists:
            logger.warning("No artist IDs found, please ensure initialization commands have been run")
            return []
            
        logger.info(f"Found {len(all_artists)} artist IDs")
        
        # Filter out artists that already have images in ARTIST_IMAGE_CACHE
        filtered_artists = []
        artists_with_images = 0
        
        # Check cache in batches to avoid checking too many at once
        batch_size = 5000
        for i in range(0, len(all_artists), batch_size):
            batch = all_artists[i:i+batch_size]
            logger.debug(f"Checking ARTIST_IMAGE_CACHE for artist batch {i//batch_size + 1}/{len(all_artists)//batch_size + 1}...")
            
            # Check if artists in this batch already have images
            results = await util.ARTIST_IMAGE_CACHE.multi_get(batch)
            
            # Only keep artist IDs without images
            for j, result in enumerate(results):
                if result is None or result[0] is None:  # If result is None or the first element (image data) is None
                    filtered_artists.append(batch[j])
                else:
                    artists_with_images += 1
        
        # Update artist list, keeping only those without images
        logger.info(f"After filtering, {len(filtered_artists)}/{len(all_artists)} artists need processing (skipped {artists_with_images} artists with existing images)")
        
        elapsed = timer() - start
        logger.info(f"Artist ID filtering complete, took: {format_elapsed_time(elapsed)}")
        
        return filtered_artists
    except Exception as e:
        logger.error(f"Error retrieving and filtering artist IDs: {str(e)}")
        import traceback
        logger.debug(f"Error stack: {traceback.format_exc()}")
        return []

async def get_images_from_spotify(mbid):
    """
    Attempt to get artist images from Spotify, including initializing Provider and getting Spotify ID from cache
    
    Args:
        mbid (str): The artist's MusicBrainz ID
        
    Returns:
        tuple: (images, expiry, success, api_request_count)
            - images: Dictionary of retrieved images, empty dict if failed
            - expiry: Expiry time, None if failed
            - success: Whether images were successfully retrieved
            - api_request_count: Number of API requests sent
    """
    images = {}
    expiry = None
    api_requests_count = 0
    
    try:
        # 获取全局共享的 SpotifyProvider
        spotify_provider = await get_spotify_provider()
        if not spotify_provider:
            logger.error("No valid SpotifyProvider available, cannot retrieve Spotify images")
            return {}, None, False, 0
            
        # Get spotify_ids from SPOTIFY_CACHE
        spotify_ids_result = await util.SPOTIFY_CACHE.get(mbid)
        
        # Handle possible None return value
        if spotify_ids_result is None:
            logger.debug(f"Artist {mbid} not found in Spotify cache")
            return {}, None, False, 0
            
        spotify_ids, _ = spotify_ids_result
        
        if not spotify_ids:
            logger.debug(f"Artist {mbid} has no associated Spotify IDs")
            return {}, None, False, 0
        
        logger.debug(f"Artist {mbid} found {len(spotify_ids)} Spotify IDs: {spotify_ids}")
        
        # Try each spotify_id serially until images are found
        for spotify_id in spotify_ids:
            try:
                logger.debug(f"Retrieving images for artist {mbid} from Spotify (spotify_id={spotify_id})")
                
                # Record request start time
                start_request = timer()
                logger.debug(f"Spotify API request start time: {start_request}")
                
                # Increment API request count
                api_requests_count += 1
                
                # 使用全局共享的 SpotifyProvider 获取图片
                images, expiry = await spotify_provider.get_artist_images(spotify_id)
                
                logger.debug(f"Spotify API request completed, took: {timer() - start_request:.2f}s")
                
                if images:
                    logger.debug(f"Successfully retrieved images for artist {mbid} from Spotify: found {len(images)} image URLs")
                    return images, expiry, True, api_requests_count
                else:
                    logger.debug(f"Artist {mbid}'s Spotify ID {spotify_id} did not return any images")
                
            except ProviderUnavailableException as pu:
                # Handle ProviderUnavailableException
                if "429" in str(pu):
                    # Handle rate limiting error
                    logger.warning(f"Spotify API rate limit error (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                    logger.info(f"Waiting 5 seconds before continuing...")
                    await asyncio.sleep(5)  # Simple backoff wait
                elif "401" in str(pu):
                    # Handle authentication error
                    logger.warning(f"Spotify API authentication error (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                    logger.info(f"Possibly invalid Spotify ID, skipping...")
                    # Skip this ID directly
                    break
                else:
                    # Handle other API errors
                    logger.warning(f"Spotify API error (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                    # Try the next ID
                    continue
            except Exception as e:
                # Handle other types of errors
                logger.warning(f"Failed to get Spotify artist images (MBID={mbid}, spotify_id={spotify_id}): {str(e)}, error type: {type(e)}")
                import traceback
                logger.debug(f"Error stack: {traceback.format_exc()}")
                continue
    except Exception as e:
        logger.error(f"Error getting Spotify ID or images (MBID={mbid}): {str(e)}")
        import traceback
        logger.debug(f"Error stack: {traceback.format_exc()}")
    
    # If we reach here, all spotify_ids have been tried but no images were found
    return {}, None, False, api_requests_count

async def get_images_from_fanart(mbid):
    """
    Attempt to get artist images from FanartTV, including initializing Provider
    
    Args:
        mbid (str): The artist's MusicBrainz ID
        
    Returns:
        tuple: (images, expiry, success, api_request_count)
            - images: Dictionary of retrieved images, empty dict if failed
            - expiry: Expiry time, None if failed
            - success: Whether images were successfully retrieved
            - api_request_count: Number of API requests sent
    """
    try:
        # 使用全局共享的 FanArtTvProvider
        fanart_provider = await get_fanart_provider()
        if not fanart_provider:
            logger.warning("No valid FanartTvProvider available, cannot retrieve FanartTV images")
            return {}, None, False, 0
            
        logger.debug(f"Retrieving images for artist {mbid} from FanartTV")
        
        # Record request start time
        start_request = timer()
        logger.debug(f"FanartTV API request start time: {start_request}")
        
        # Use the artist's MusicBrainz ID directly to request from FanartTV
        images, expiry = await fanart_provider.get_artist_images(mbid)
        
        # Count just one API request
        api_requests_count = 1
        
        logger.debug(f"FanartTV API request completed, took: {timer() - start_request:.2f}s")
        
        if images:
            logger.debug(f"Successfully retrieved images for artist {mbid} from FanartTV: found {len(images)} image URLs")
            return images, expiry, True, api_requests_count
        else:
            logger.debug(f"No images found for artist {mbid} on FanartTV")
            return {}, None, False, api_requests_count
    except ProviderUnavailableException as pu:
        # Handle API errors
        logger.warning(f"FanartTV API error (MBID={mbid}): {str(pu)}")
        # Simple backoff wait
        if "429" in str(pu):
            logger.info(f"Waiting 5 seconds before continuing...")
            await asyncio.sleep(5)
        return {}, None, False, 1
    except Exception as e:
        # Handle other types of errors
        logger.warning(f"Failed to get FanartTV artist images (MBID={mbid}): {str(e)}, error type: {type(e)}")
        import traceback
        logger.debug(f"Error stack: {traceback.format_exc()}")
        return {}, None, False, 1

async def retrieve_artist_images(artist_ids):
    """
    Retrieve and cache images for the specified list of artist IDs
    
    First tries to get artist images from Spotify, and if that fails,
    tries to use FanartTV. Processes each artist serially,
    with a 3-second interval between requests to avoid API rate limiting.
    Pauses for 5 minutes after every 1000 API requests to prevent stricter rate limiting during long runs.
    
    Args:
        artist_ids (list): List of artist IDs to process
    """
    if not artist_ids:
        logger.info("No artists to process, skipping image retrieval")
        return
        
    logger.info(f"Starting to retrieve images for {len(artist_ids)} artists...")
    start = timer()
    
    # Set expiry time to 100 years from now
    expiry_time = provider.utcnow() + timedelta(days=365*100)
    
    # Statistics counters
    total_processed = 0
    total_with_images = 0
    total_errors = 0
    total_spotify_images = 0
    total_fanart_images = 0
    api_requests_count = 0  # API request counter
    
    # Process artists serially, waiting 3 seconds between each request
    for i, mbid in enumerate(artist_ids):
        # Check if we need to pause
        if api_requests_count >= 1000:
            pause_minutes = 1
            logger.info(f"Sent {api_requests_count} API requests, pausing for {pause_minutes} minutes before continuing...")
            await asyncio.sleep(pause_minutes * 60)  # Convert to seconds
            api_requests_count = 0  # Reset counter
            logger.info("Resuming processing...")
            
        # Check cache again, in case another process has added images since filtering
        image, _ = await util.ARTIST_IMAGE_CACHE.get(mbid)
        if image:
            logger.debug(f"Artist {mbid} already has images, skipping")
            continue
        
        total_processed += 1
        logger.debug(f"Starting to process artist MBID: {mbid} ({total_processed}/{len(artist_ids)})")
        
        # If not the first request, wait 2 seconds
        if i > 0:
            await asyncio.sleep(1)
        
        try:
            images = {}
            expiry = None
            image_source = None
            
            # 1. First, try to get images from Spotify
            spotify_images, spotify_expiry, spotify_success, spotify_requests = await get_images_from_spotify(mbid)
            api_requests_count += spotify_requests
            
            if spotify_success:
                images = spotify_images
                expiry = spotify_expiry
                image_source = "Spotify"
                total_spotify_images += 1
            
            # 2. If Spotify doesn't have images, try FanartTV
            if not images:
                fanart_images, fanart_expiry, fanart_success, fanart_requests = await get_images_from_fanart(mbid)
                api_requests_count += fanart_requests
                
                if fanart_success:
                    images = fanart_images
                    expiry = fanart_expiry
                    image_source = "FanartTV"
                    total_fanart_images += 1
            
            # 3. Save the found images
            if images:
                # Store found images in cache
                await util.ARTIST_IMAGE_CACHE.set(mbid, images, ttl=(expiry_time - provider.utcnow()).total_seconds())
                total_with_images += 1
                            
                logger.debug(f"Successfully retrieved images for artist {mbid} (source: {image_source}): found {len(images)} image URLs")
                
                # Log progress every 10 artists or at the end
                if total_processed % 10 == 0 or total_processed == len(artist_ids):
                    logger.info(f"Progress: {total_processed}/{len(artist_ids)} artists ({total_with_images} with images [Spotify: {total_spotify_images}, FanartTV: {total_fanart_images}], {total_errors} errors), "
                              f"processed {format_elapsed_time(timer() - start, total_processed)}, API request count: {api_requests_count}")
            
        except Exception as e:
            logger.error(f"Error processing MusicBrainz artist ID {mbid}: {str(e)}")
            total_errors += 1
            logger.debug(f"Current error count: {total_errors}")
    
    elapsed = timer() - start
    logger.info(f"Artist image retrieval complete. Total: {total_processed} artists, {total_with_images} with images "
               f"[Spotify: {total_spotify_images}, FanartTV: {total_fanart_images}], "
               f"{total_errors} errors. Total time: {format_elapsed_time(elapsed)}")

async def crawl_artist_images():
    """
    Crawl and cache images for all artists
    
    Gets all artist IDs from ARTIST_CACHE, looks up corresponding Spotify IDs via SPOTIFY_CACHE,
    then uses SpotifyProvider to get artist images, and stores results in ARTIST_IMAGE_CACHE,
    with expiry time set to 100 years from now. Includes comprehensive error handling and logging.
    
    All requests are executed serially with a 3-second interval between requests to avoid API rate limiting.
    """
    logger.info("Starting artist image retrieval...")
    
    # 1. Filter artist IDs that need processing
    artist_ids = await filter_artists_without_images()
    
    # 2. If there are artists to process, retrieve and cache their images
    if artist_ids:
        await retrieve_artist_images(artist_ids)
    else:
        logger.info("No artists to process, task complete")

async def update_items(multi_function, cache, name, count = 100, max_ttl = 60 * 60):
    while True:
        keys = await cache.get_stale(count, provider.utcnow() + timedelta(seconds = max_ttl))
        logger.debug(f"Got {len(keys)} stale {name}s to refresh")
        
        if keys:
            start = timer()
            results = await multi_function(keys)
            
            if not results:
                missing = keys
            else:
                missing = set(keys) - set(item['id'] for item, _ in results)
                
            if missing:
                logger.debug(f"Removing deleted {name}s:\n{missing}")
                await asyncio.gather(*(cache.delete(id) for id in missing))
                
            await asyncio.gather(*(cache.set(result['id'], result, ttl=(expiry - provider.utcnow()).total_seconds()) for result, expiry in results))
                
            logger.debug(f"Refreshed {len(keys)} {name}s in {timer() - start:.1f}s")

        else:
            # If there weren't any to update sleep, otherwise continue
            await asyncio.sleep(60)
    
async def init_releases():
    """
    Initialize all release IDs in the RELEASE_CACHE
    
    Retrieves all release IDs from the database and stores them in the RELEASE_CACHE 
    with None values and expiry time set to now. Uses concurrent requests to efficiently
    fetch large numbers of IDs.
    """
    logger.info("Starting to initialize release IDs...")
    start = timer()
    
    # Get provider that can retrieve release IDs
    release_provider = provider.get_providers_implementing(provider.ReleaseByIdMixin)[0]
    
    # File for SQL doesn't exist in the provider yet, so let's add the method
    # This will get all release IDs from the release table
    async def get_all_release_ids():
        results = await release_provider.query_from_file('all_release_ids.sql')
        return [item['gid'] for item in results]
    
    # Get all release IDs
    all_ids = await get_all_release_ids()
    
    if not all_ids:
        logger.warning("No release IDs found in database")
        return
    
    # Create pairs for cache storage - using None as value as requested
    pairs = [(id, None) for id in all_ids]
    
    # Clear cache before updating
    await util.RELEASE_CACHE.clear()
    
    # Set expiry time to now (0 seconds from now)
    await util.RELEASE_CACHE.multi_set(pairs, ttl=0, timeout=None)
    
    elapsed = timer() - start
    logger.info(f"Initialized {len(all_ids)} releases in {format_elapsed_time(elapsed, len(all_ids))}")

async def crawl_release_images():
    """
    Retrieve and cache images for all cached releases in RELEASE_IMAGE_CACHE
    
    Gets release IDs from RELEASE_CACHE in batches, filters out those that already have images,
    then processes each batch using get_release_images from ReleaseArtworkMixin in parallel.
    Stores results in RELEASE_IMAGE_CACHE with expiry time set to 100 years from now.
    Uses high concurrency processing while also implementing rate limiting to avoid service disruption.
    """
    def get_ttl(is_success):
        days = 30
        if is_success:
            # Set expiry time to 100 years from now if images were found
            days = 365 * 100
        expiry_time = provider.utcnow() + timedelta(days=days)
        return (expiry_time - provider.utcnow()).total_seconds()

    # Async function to get images for a single release
    async def process_release(release_id, image_provider):
        try:
            # Use get_release_images directly rather than get_release_images_multi
            images, _ = await image_provider.get_release_images(release_id)
            
            if images:
                # Save to cache with 100-year expiry time
                await util.RELEASE_IMAGE_CACHE.set(release_id, images, ttl=get_ttl(is_success=True))
                return True
            else:
                await util.RELEASE_IMAGE_CACHE.set(release_id, None, ttl=get_ttl(is_success=False))
                return False
                
        except ProviderUnavailableException as pu:
            # Handle API rate limiting or service unavailability
            if "429" in str(pu):
                await asyncio.sleep(5)  # Simple backoff wait
            return False
            
        except Exception as e:
            logger.error(f"Error processing release {release_id} images: {str(e)}")
            return False


    logger.info("Starting release image retrieval...")
    
    # Initialize image provider
    try:
        release_image_provider = provider.get_providers_implementing(provider.CoverArtArchiveProvider)[0]
        if not release_image_provider:
            logger.error("No valid ReleaseArtworkMixin found, cannot retrieve images")
            return
    except Exception as e:
        logger.error(f"Error initializing ReleaseArtworkMixin: {str(e)}")
        return

    # Process releases in batches
    page_size = 100000  # Records per page
    batch_size = 100   # Number of releases to process per batch
    offset = 0
    total_processed = 0
    total_found = 0
    total_time_elapsed = 0
    
    while True:
        try:
            # Get one page of release IDs
            result = await util.RELEASE_CACHE.get_all_keys_paged(limit=page_size, offset=offset)
            
            if not result['keys']:  # result is a dictionary with 'keys' key
                break
            
            keys_list = result['keys']
            logger.info(f"Retrieved {len(keys_list)} release IDs from offset {offset}")
            
            batch_index = 0
            total_batches = len(keys_list) // batch_size
            # Process this page in smaller batches
            for i in range(0, len(keys_list), batch_size):
                start = timer()
                batch = keys_list[i:i+batch_size]    

                # Filter out releases that already have images
                filtered_batch = []
                batch_results = await util.RELEASE_IMAGE_CACHE.multi_get(batch)
                for release_id, images_result in zip(batch, batch_results):
                    value, expiry = images_result
                    # If the release has no images and no expiry, it means it's not in the cache
                    if not value and not expiry:
                        filtered_batch.append(release_id)
                    
                if not filtered_batch:
                    logger.info(f"No releases to process, skipping batch {batch_index}/{total_batches}")
                    batch_index += 1
                    continue
                else:
                    logger.info(f"Found {len(filtered_batch)} releases without images")
                
                # Process the filtered batch with a progress bar
                tasks = [process_release(release_id, release_image_provider) for release_id in filtered_batch]
                release_image_results = await asyncio.gather(*tasks)

                # Update stats
                images_found = sum(1 for result in release_image_results if result)
                total_processed += len(filtered_batch)
                total_found += images_found
                elapsed = timer() - start
                total_time_elapsed += elapsed
                # Log progress
                logger.info(f"Batch {batch_index}/{total_batches} complete: Processed {len(filtered_batch)}, Found {images_found} images, {format_elapsed_time(elapsed, count=len(filtered_batch))}")
                logger.info(f"Overall progress: {total_found}/{total_processed}, {format_elapsed_time(total_time_elapsed, count=total_processed)}")
                logger.info("===============================================")
                batch_index += 1
            # If there's no more data, exit the loop
            if not result['has_more']:
                break
                
            # Update offset for next page
            offset += page_size
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            import traceback
            logger.debug(f"Error stack: {traceback.format_exc()}")
            continue
    
async def cleanup_resources():
    """
    清理全局资源，确保在程序退出时正确关闭所有连接
    """
    global _SESSION
    
    if _SESSION:
        logger.info("Closing shared aiohttp session")
        try:
            await _SESSION.close()
            _SESSION = None
        except Exception as e:
            logger.error(f"Error closing shared session: {str(e)}")

async def run_with_cleanup(coro_func):
    """
    运行协程函数并确保在完成后清理资源
    
    Args:
        coro_func: 要运行的协程函数
    """
    try:
        return await coro_func
    finally:
        await cleanup_resources()

async def crawl():
    await asyncio.gather(
        update_wikipedia(count = CONFIG.CRAWLER_BATCH_SIZE['wikipedia'], max_ttl = 60 * 60 * 2),
        update_items(get_artist_info_multi, util.ARTIST_CACHE, "artist", count = CONFIG.CRAWLER_BATCH_SIZE['artist']),
    )
    
async def init():
    await asyncio.gather(
        init_artists(),
        init_spotify(),
        init_releases(),
    )
    
def main():
    
    parser = argparse.ArgumentParser(prog="lidarr-metadata-crawler")
    parser.add_argument("--init-artists", action="store_true")
    parser.add_argument("--init-spotify", action="store_true")
    parser.add_argument("--init-releases", action="store_true")
    parser.add_argument("--crawl-release-images", action="store_true")
    parser.add_argument("--crawl-artist-images", action="store_true")
    args = parser.parse_args()
    
    async def run_task():
        if args.init_artists:
            await init_artists()
        elif args.init_spotify:
            await init_spotify()
        elif args.init_releases:
            await init_releases()
        elif args.crawl_release_images:
            await crawl_release_images()
        elif args.crawl_artist_images:
            await crawl_artist_images()
        else:
            await crawl()
    
    asyncio.run(run_with_cleanup(run_task()))
    
if __name__ == "__main__":
    main()
