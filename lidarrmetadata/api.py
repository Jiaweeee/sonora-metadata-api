import uuid
import functools
import asyncio
import logging
import re

from datetime import datetime, date, timedelta
from timeit import default_timer as timer
from aiocache import cached


from lidarrmetadata import config
from lidarrmetadata import provider
from lidarrmetadata import util
from lidarrmetadata.chart import charts, ChartException

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info('Have api logger')

CONFIG = config.get_config()

# Set up providers
for provider_name, (args, kwargs) in CONFIG.PROVIDERS.items():
    provider_key = list(filter(lambda k: k.upper() == provider_name,
                               provider.PROVIDER_CLASSES.keys()))[0]
    lower_kwargs = {k.lower(): v for k, v in kwargs.items()}
    provider.PROVIDER_CLASSES[provider_key](*args, **lower_kwargs)

def validate_mbid(mbid, check_blacklist=True):
    """
    Validates Musicbrainz ID and returns flask response in case of error
    :param mbid: Musicbrainz ID to verify
    :param check_blacklist: Checks blacklist for blacklisted ids. Defaults to True
    :return: Flask response if error, None if valid
    """
    try:
        uuid.UUID(mbid, version=4)
    except ValueError:
        return jsonify(error='Invalid UUID'), 400

    if check_blacklist and mbid in config.get_config().BLACKLISTED_ARTISTS:
        return jsonify(error='Blacklisted artist'), 403

async def get_overview(links, mbid=None):

    overview = ''
    expiry = provider.utcnow() + timedelta(days=365)
    overview_providers = provider.get_providers_implementing(provider.ArtistOverviewMixin)    

    if overview_providers:
        wikidata_link = next(filter(
            lambda link: 'wikidata' in link.get('type', ''),
            links), None)
        wikipedia_link = next(filter(
            lambda link: 'wikipedia' in link.get('type', ''),
            links), None)

        if wikipedia_link:
            overview, expiry = await overview_providers[0].get_artist_overview(wikipedia_link['target'])
        elif wikidata_link:
            overview, expiry = await overview_providers[0].get_artist_overview(wikidata_link['target'])

        if len(overview_providers) > 1 and mbid and not overview:
            overview, expiry = await overview_providers[1].get_artist_overview(mbid)

    return overview, expiry

# Decorator to cache in redis and postgres
def postgres_cache(cache):
    def decorator(function):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            
            mbid = args[0]
            
            now = provider.utcnow()

            cached, expiry = await cache.get(mbid)
            if cached and expiry > now:
                logger.debug(f"Cache hit for {mbid} in {cache.__class__.__name__}")
                return cached, expiry
            
            result, expiry = await function(*args, **kwargs)
            
            # 只有当结果不为 None 时才缓存
            if result is not None:
                ttl = (expiry - now).total_seconds()
                await cache.set(mbid, result, ttl=ttl)
                
            return result, expiry

        wrapper.__cache__ = cache
        return wrapper
    return decorator

class ArtistNotFoundException(Exception):
    def __init__(self, mbid):
        super().__init__(f"Artist not found: {mbid}")
        self.mbid = mbid

class MissingProviderException(Exception):
    """ Thown when we can't cope without a provider """

@postgres_cache(util.ARTIST_CACHE)
async def get_artist_info(mbid):
    artists = await get_artist_info_multi([mbid])
    if not artists:
        artist_provider = provider.get_providers_implementing(provider.ArtistByIdMixin)[0]
        new_id = await artist_provider.redirect_old_artist_id(mbid)
        artists = await get_artist_info_multi([new_id])
        
        if not artists:
            raise ArtistNotFoundException(mbid)
            
    artist, expiry = artists[0]  # 解构元组，获取 data 和 expiry
    if 'links' in artist:
        artist['streaming_links'] = get_artist_streaming_links(artist['links'])
        del artist['links']
    return artist, expiry  # 返回处理后的 artist 数据和过期时间

async def get_artist_info_multi(mbids):
    
    start = timer()

    artist_providers = provider.get_providers_implementing(provider.ArtistByIdMixin)
    artist_art_providers = provider.get_providers_implementing(provider.ArtistArtworkMixin)
    
    if not artist_providers:
        # 500 error if we don't have an artist provider since it's essential
        raise MissingProviderException('No artist provider available')
    
    expiry = provider.utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['cloudflare'])
    
    # Do the main DB query
    artists = await artist_providers[0].get_artists_by_id(mbids)
    if not artists:
        return None
    
    # Add in default expiry
    artists = [{'data': artist, 'expiry': expiry} for artist in artists]
    
    # Start overviews
    overviews_task = asyncio.gather(*[get_overview(artist['data']['links'], artist['data']['id']) for artist in artists])
    
    if artist_art_providers:
        results = await asyncio.gather(*[artist_art_providers[0].get_artist_images(x['data']['id']) for x in artists])
        
        for i, artist in enumerate(artists):
            images, expiry = results[i]
            artist['data']['images'] = images
            artist['expiry'] = min(artist['expiry'], expiry)

        if len(artist_art_providers) > 1:
            image_types = {'Banner', 'Fanart', 'Logo', 'Poster'}
            artists_without_images = [x for x in artists if not x['data']['images'] or not image_types.issubset({i['CoverType'] for i in x['data']['images']})]
            results = await asyncio.gather(*[artist_art_providers[1].get_artist_images(x['data']['id']) for x in artists_without_images])

            for i, artist in enumerate(artists_without_images):
                images, expiry = results[i]
                artist['data']['images'] = combine_images(artist['data']['images'], images)
                artist['expiry'] = min(artist['expiry'], expiry)
    else:
        for artist in artists:
            artist['images'] = []

    # Get overview results
    results = await overviews_task
    for i, artist in enumerate(artists):
        overview, expiry = results[i]
        artist['data']['overview'] = overview
        artist['expiry'] = min(artist['expiry'], expiry)
            
    logger.debug(f"Got basic artist info for {len(mbids)} artists in {(timer() - start) * 1000:.0f}ms ")

    return [(item['data'], item['expiry']) for item in artists]

def combine_images(a, b):
    result = a
    extra_types = {i['CoverType'] for i in b} - {i['CoverType'] for i in a}
    extra_images = [i for i in b if i['CoverType'] in extra_types]
    result.extend(extra_images)

    return result

async def get_artist_release_groups(mbid):
    release_group_providers = provider.get_providers_implementing(
        provider.ReleaseGroupByArtistMixin)
    if release_group_providers and not mbid in CONFIG.BLACKLISTED_ARTISTS:
        return await release_group_providers[0].get_release_groups_by_artist(mbid)
    else:
        return []

async def get_artist_releases(mbid):
    """
    获取艺术家的发行版信息，将 release group 的 primary_release 中的 id 和 date 字段提取到外层
    
    :param mbid: 艺术家的 MusicBrainz ID
    :return: 按 primary_type 分组的发行版信息，每个发行版包含 id、title、date 和 images 字段
    """
    release_groups = await get_artist_release_groups(mbid)
    
    if not release_groups:
        return {}
        
    result = {}
    all_releases = []  # 用于收集所有需要获取封面的 release
    release_map = {}  # 用于映射 release_id 到其在结果中的位置
    
    # 第一步：处理所有 release groups 并收集 release ids
    for type_name, groups in release_groups.items():
        if not groups:  # 如果这个类型没有数据，保持为 null
            result[type_name] = groups
            continue
            
        processed_groups = []
        for group in groups:
            if not group.get('primary_release'):
                continue
                
            processed_group = {
                'release_group_id': group['id'],
                'id': group['primary_release']['id'],  # 使用 primary_release 的 id
                'title': group['title'],
                'date': group['primary_release'].get('date')  # 提取 date 字段
            }
            processed_groups.append(processed_group)
            all_releases.append(processed_group)
            release_map[processed_group['id']] = processed_group
            
        result[type_name] = processed_groups
    
    # 第二步：批量获取所有 release 的封面
    if all_releases:
        art_provider = provider.get_providers_implementing(provider.ReleaseArtworkMixin)[0]
        release_ids = [release['id'] for release in all_releases]
        image_results = await art_provider.get_release_images_multi(release_ids)
        
        # 将封面信息添加到对应的 release 中
        for i, release_id in enumerate(release_ids):
            images, _ = image_results[i]
            if images and release_id in release_map:
                release_map[release_id]['images'] = images
    
    return result

async def get_release_group_artists(release_group):
    
    start = timer()
    
    results = await asyncio.gather(*[get_artist_info(gid) for gid in release_group['artistids']])
                                   
    artists = [result[0] for result in results]
    expiry = min([result[1] for result in results])
    
    logger.debug(f"Got album artists in {(timer() - start) * 1000:.0f}ms ")
    
    return artists, expiry

class ReleaseGroupNotFoundException(Exception):
    def __init__(self, mbid):
        super().__init__(f"Album not found: {mbid}")
        self.mbid = mbid

@postgres_cache(util.ALBUM_CACHE)
async def get_release_group_info_basic(mbid):
    
    release_groups = await get_release_group_info_multi([mbid])
    if not release_groups:

        album_provider = provider.get_providers_implementing(provider.ReleaseGroupByIdMixin)[0]
        new_id = await album_provider.redirect_old_release_group_id(mbid)
        release_groups = await get_release_group_info_multi([new_id])

        if not release_groups:
            raise ReleaseGroupNotFoundException(mbid)
    
    return release_groups[0]

async def get_release_group_info_multi(mbids):
    
    start = timer()
    
    release_group_providers = provider.get_providers_implementing(provider.ReleaseGroupByIdMixin)
    album_art_providers = provider.get_providers_implementing(provider.AlbumArtworkMixin)
    
    if not release_group_providers:
        raise MissingProviderException('No album provider available')

    expiry = provider.utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['cloudflare'])

    # Do the main DB query
    release_groups = await release_group_providers[0].get_release_groups_by_id(mbids)
    if not release_groups:
        return None

    # Add in default expiry
    release_groups = [{'data': rg, 'expiry': expiry} for rg in release_groups]
    
    # Start overviews
    overviews_task = asyncio.gather(*[get_overview(rg['data']['links']) for rg in release_groups])
    
    # Get fanart images (and prefer those if possible)
    if album_art_providers:
        results = await asyncio.gather(*[album_art_providers[0].get_album_images(x['data']['id']) for x in release_groups])
        
        for i, rg in enumerate(release_groups):
            images, expiry = results[i]
            rg['data']['images'] = combine_images(images, rg['data']['images'])
            rg['expiry'] = min(rg['expiry'], expiry)

    # Get overview results
    results = await overviews_task
    for i, rg in enumerate(release_groups):
        overview, expiry = results[i]
        rg['data']['overview'] = overview
        rg['expiry'] = min(rg['expiry'], expiry)
    
    logger.debug(f"Got basic album info for {len(mbids)} albums in {(timer() - start) * 1000:.0f}ms ")

    return [(item['data'], item['expiry']) for item in release_groups]

async def get_release_group_info(mbid):

    release_group, rg_expiry = await get_release_group_info_basic(mbid)
    artists, artist_expiry = await get_release_group_artists(release_group)
    
    release_group['artists'] = artists
    del release_group['artistids']
    
    return release_group, min(rg_expiry, artist_expiry)

async def get_release_info_basic(mbid):
    release_provider = provider.get_providers_implementing(provider.ReleaseByIdMixin)[0]
    releases = await release_provider.get_release_by_id([mbid])
    
    if not releases:
        return None
    
    for release in releases:
        # 提取 streaming links
        if 'links' in release:
            streaming_links = get_streaming_links(release['links'])
            if streaming_links:
                release['streaming_links'] = streaming_links
        # 删除无用字段
        if 'oldids' in release:
            del release['oldids']
        if 'media' in release:
            del release['media']
        if 'status' in release:
            del release['status']
    return releases[0]

def get_streaming_links(links):
    """
    从发行版的链接列表中提取流媒体服务的链接
    :param links: 链接列表，每个链接包含 'type' 和 'url' 字段
    :return: 流媒体服务的链接列表。当前只支持 Spotify 和 Apple Music
    """
    if not links:
        return None
    streaming_links = []
    for link in links:
        if 'streaming' in link.get('type', ''):
            url = link.get('url', link.get('target', '')).lower()
            if 'spotify' in url:
                streaming_links.append({
                    'source': 'spotify',
                    'url': link.get('url', link.get('target'))
                })
            elif 'music.apple' in url:
                streaming_links.append({
                    'source': 'apple_music',
                    'url': link.get('url', link.get('target'))
                })
    return streaming_links if streaming_links else None

def get_artist_streaming_links(links):
    """
    从艺术家的链接列表中提取流媒体服务的链接，每个服务类型只保留一个链接
    :param links: 链接列表，每个链接包含 'type' 和 'target' 字段
    :return: 流媒体服务的链接列表。当前只支持 Spotify 和 Apple Music，每个服务只返回一个链接
    """
    if not links:
        return None
        
    # 使用字典记录已添加的服务类型
    service_links = {}
    
    for link in links:
        link_type = link.get('type', '').lower()
        url = link.get('url', link.get('target', ''))
        
        if link_type == 'spotify' and 'spotify' not in service_links:
            service_links['spotify'] = {
                'source': 'spotify',
                'url': url
            }
        elif link_type == 'apple' and 'apple_music' not in service_links and 'artist' in url.lower():
            service_links['apple_music'] = {
                'source': 'apple_music',
                'url': url
            }
            
    streaming_links = list(service_links.values())
    return streaming_links if streaming_links else None

async def get_release_images(mbid):
    image_provider = provider.get_providers_implementing(provider.ReleaseArtworkMixin)[0]
    return await image_provider.get_release_images(mbid)


class ReleaseNotFoundException(Exception):
    def __init__(self, mbid):
        super().__init__(f"Release not found: {mbid}")
        self.mbid = mbid

@postgres_cache(util.RELEASE_CACHE)
async def get_release_info(mbid):
    release = await get_release_info_basic(mbid)
    if not release:
        return None, provider.utcnow()
    
    # Get overview
    overview, overview_expiry = await get_overview(release.get('wiki_links', []))
    release['overview'] = overview
    
    # Get images
    images, image_expiry = await get_release_images(mbid)
    release['images'] = images

    # Use the earliest expiry
    expiry = provider.utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['release'])
    expiry = min(expiry, overview_expiry, image_expiry)
    return release, expiry

async def get_release_search_results(query, limit, artist_name=''):
    search_providers = provider.get_providers_implementing(provider.ReleaseNameSearchMixin)
    if search_providers:
        response = await search_providers[0].search_release_name(query, artist_name=artist_name, limit=limit)
        releases = response.get('releases', None)

        if not releases:
            return []
            
        search_result = []
        for release in releases:
            def get_artists(artist_credit):
                if not artist_credit:
                    return []
                return [artist['name'] for artist in artist_credit]

            def get_type(release_group):
                if not release_group or 'primary-type' not in release_group:
                    return 'Unknown'
                return release_group['primary-type']

            search_result.append({
                'id': release['id'],
                'artists': get_artists(release.get('artist-credit', None)),
                'title': release['title'],
                'type': get_type(release.get('release-group', None)),
                'score': release['score'],
                'release_date': release.get('date', None),
            })
            
        # 批量获取封面
        art_providers = provider.get_providers_implementing(provider.ReleaseArtworkMixin)
        if art_providers:
            release_ids = [r['id'] for r in search_result]
            image_results = await art_providers[0].get_release_images_multi(release_ids)
            
            # 将封面信息添加到搜索结果中
            for i, (images, _) in enumerate(image_results):
                if images:
                    search_result[i]['images'] = images
                    
        return search_result
    return []
        

class TrackNotFoundException(Exception):
    def __init__(self, mbid):
        super().__init__(f"Track not found: {mbid}")
        self.mbid = mbid

class DiscoverContentException(Exception):
    """Raised when there is an error fetching hot songs data"""
    def __init__(self, message="Failed to fetch hot songs data"):
        super().__init__(message)
        self.message = message

@postgres_cache(util.TRACK_CACHE)
async def get_track_info(mbid):
    """
    获取曲目详细信息
    """
    track_provider = provider.get_providers_implementing(provider.TrackByIdMixin)[0]
    tracks = await track_provider.get_track_by_id([mbid])
    
    if not tracks or len(tracks) == 0:
        raise TrackNotFoundException(mbid)
    
    track = tracks[0]
    # 提取流媒体 links
    if 'urls' in track:
        streaming_links = get_streaming_links(track['urls'])
        if streaming_links:
            track['streaming_links'] = streaming_links
        del track['urls']
    
    # 获取 release image
    release_id = track['release']['id']
    release_images, image_expiry = await get_release_images(release_id)  # Unpack the tuple, ignoring expiry
    track['release']['images'] = release_images

    # Use the earlist expiry
    expiry = provider.utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['track'])
    expiry = min(expiry, image_expiry)

    return track, expiry

async def get_track_search_results(query, limit, artist_name=''):
    search_providers = provider.get_providers_implementing(provider.RecordingNameSearchMixin)
    if search_providers:
        recording_results = await search_providers[0].search_recording_name(
            query,
            limit,
            artist_name
        )
        if not recording_results:
            return []
            
        recordings = recording_results['recordings']
        tracks = []
        release_ids = set()  # 使用 set 去重
        
        # 先收集所有的 release_id 和构建基础 track 信息
        track_map = {}  # release_id 到 track 的映射
        for recording in recordings:
            # 检查 recording 是否包含 releases
            if 'releases' not in recording:
                continue
            for release in recording['releases']:
                if 'media' not in release:
                    continue
                    
                track_id = release['media'][0]['track'][0].get('id', None)
                if not track_id:
                    continue
                    
                track = {
                    'id': track_id,
                    'title': recording['title'],
                    'score': recording['score'],
                }
                
                if 'artist-credit' in recording:
                    track['artists'] = [artist['name'] for artist in recording['artist-credit']]
                
                release_ids.add(release['id'])
                track_map[release['id']] = track
        
        # 批量获取图片
        if release_ids:
            art_provider = provider.get_providers_implementing(provider.ReleaseArtworkMixin)[0]
            image_results = await art_provider.get_release_images_multi(list(release_ids))
            
            # 将图片信息添加到对应的 track 中
            for i, release_id in enumerate(release_ids):
                images, _ = image_results[i]
                if images and release_id in track_map:
                    track_map[release_id]['images'] = images
            
            # 构建最终的 tracks 列表
            tracks = list(track_map.values())
            
        return tracks
    return []

async def get_artist_search_results(query, limit):
    search_providers = provider.get_providers_implementing(
        provider.ArtistNameSearchMixin)

    if not search_providers:
        return []
    
    artists = await search_providers[0].search_artist_name(query, limit=limit)
    artist_ids = [item['id'] for item in artists]
    
    # Get artwork provider
    artwork_providers = provider.get_providers_implementing(provider.ArtistArtworkMixin)
    if artwork_providers:
        # Get images for each artist
        artwork_provider = artwork_providers[0]
        artist_images_results = await asyncio.gather(*[artwork_provider.get_artist_images(aid) for aid in artist_ids])
        
        # Add images to the original search results - only take the first element of the tuple (images list)
        for artist, image_result in zip(artists, artist_images_results):
            artist['images'] = image_result[0] if image_result else []
    
    return [{
        'id': artist['id'],
        'name': artist['name'],
        'images': artist['images'],
        'score': artist['score']
    } for artist in artists]


@cached(ttl=60 * 60 * 24 * 7, key='new-releases', alias='default')
async def get_new_releases(months_threshold=2):
    """Get new releases from Billboard 200 chart within the specified time threshold.
    
    This function fetches the Billboard 200 chart data and filters it to only include
    releases that came out within the specified number of months.
    
    Args:
        months_threshold (int, optional): Number of months to look back. Defaults to 2.
    
    Returns:
        dict: A dictionary containing:
            - releases: A list of recent releases, each containing:
                - id: The release's MusicBrainz ID
                - title: The release's title
                - artists: List of artist names
                - type: Release type (e.g. 'Album')
                - release_date: Release date (YYYY-MM-DD format)
                - images: List of cover art images (optional)
            - cache_info: Cache information containing:
                - expired_at: ISO format timestamp when the cache will expire
                - cached_at: Timestamp when the data was cached
    """
    def is_recent_release(release_date_str, threshold=months_threshold):
        if not release_date_str:
            return False
        
        today = date.today()
        try:
            # Handle full date format (2001-01-01)
            if '-' in str(release_date_str):
                release_date = datetime.strptime(str(release_date_str), '%Y-%m-%d').date()
            # Handle year only format (2001)
            elif re.match(r'^\d{4}$', str(release_date_str)):
                release_date = date(int(release_date_str), 1, 1)
            else:
                return False
            
            # Calculate months between dates
            months_diff = (today.year - release_date.year) * 12 + (today.month - release_date.month)
            return months_diff <= threshold
        except (ValueError, TypeError):
            return False

    try:
        chart_function = charts.get('billboard-200-albums')
        chart_releases = await chart_function()
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days
        
        if not chart_releases:    
            raise ChartException('No chart releases found')
            
        filtered_releases = list(filter(lambda release: is_recent_release(release.get('release_date')), chart_releases))
        return {
            'releases': filtered_releases,
            'cache_info': {
                'expired_at': expired_at.isoformat(),
                'cached_at': now.isoformat()
            }
        }
    except ChartException as e:
        raise DiscoverContentException(str(e))
    except Exception as e:
        logger.error(f"Error in get_new_releases: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch new releases data: {str(e)}")

@cached(ttl=60 * 60 * 24 * 7, key='hot-songs', alias='default')
async def get_hot_songs():
    """Get hot songs from Apple Music charts
    
    This function fetches the Apple Music top songs chart data and returns the top 24 songs.
    Results are cached for 7 days to reduce API calls.
    
    Returns:
        dict: A dictionary containing:
            - songs: A list of hot songs, each containing:
                - id: The song's MusicBrainz ID
                - title: The song's title
                - artists: List of artist names
                - images: List of cover art images (optional)
            - cache_info: Cache information containing:
                - expired_at: ISO format timestamp when the cache will expire
                - cached_at: Timestamp when the data was cached
                
    Raises:
        HotSongsException: If there is an error fetching the hot songs data
    """
    try:
        chart_function = charts.get('apple-music-top-songs')
        chart_songs = await chart_function()
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days
        
        # 取前24首歌曲
        top_songs = chart_songs[:24] if len(chart_songs) > 24 else chart_songs
        
        return {
            'songs': top_songs,
            'cache_info': {
                'expired_at': expired_at.isoformat(),
                'cached_at': now.isoformat()
            }
        }
    except ChartException as e:
        raise DiscoverContentException(str(e))
    except Exception as e:
        # 处理其他异常
        logger.error(f"Error in get_hot_songs: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch hot songs data: {str(e)}")
    
async def get_release_chart(chart_name):
    def format_release(release):
        images = release.get('images', None)
        if 'images' in release:
            del release['images']
        image = images.get('small') if images else None
        release['image'] = image
        return release
    
    try:
        chart_function = charts.get(chart_name)
        releases = await chart_function()
        releases = [format_release(release) for release in releases]

        # get chart image
        image = None
        for release in releases:
            if release.get('image'):
                image = release['image']
                break

        return {
            'image': image,
            'item_type': 'release',
            'items': releases,
        }
    except ChartException as e:
        raise DiscoverContentException(str(e))
    except Exception as e:  
        logger.error(f"Error in get_release_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch release chart data: {str(e)}")

async def get_track_chart(chart_name):
    def format_track(track):
        images = track.get('images', None)
        if 'images' in track:
            del track['images']
        image = images.get('small') if images else None
        track['image'] = image
        return track
    
    try:
        chart_function = charts.get(chart_name)
        tracks = await chart_function()
        tracks = [format_track(track) for track in tracks]

        # get chart image
        image = None
        for track in tracks:
            if track.get('image'):
                image = track['image']
                break

        return {
            'image': image,
            'item_type': 'track',
            'items': tracks,
        }
    except ChartException as e:
        raise DiscoverContentException(str(e))
    except Exception as e:  
        logger.error(f"Error in get_track_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch track chart data: {str(e)}")

async def get_artist_chart(chart_name):
    def format_artist(artist):
        images = artist.get('images', None)
        image = None
        if images:
            image_map = {}
            for img in images:
                image_map[img['CoverType']] = img['Url']
            if image_map.get('Poster'):
                image = image_map['Poster']
            elif image_map.get('Fanart'):
                image = image_map['Fanart']
        return {
            'id': artist['id'],
            'title': artist['name'],
            'image': image,
        }

    try:
        chart_function = charts.get(chart_name)
        artists = await chart_function()
        artists = [format_artist(artist) for artist in artists]

        # get chart image
        image = None
        for artist in artists:
            if artist.get('image'):
                image = artist['image']
                break   
        
        return {
            'image': image,
            'item_type': 'artist',
            'items': artists,
        }   
    except ChartException as e:
        raise DiscoverContentException(str(e))
    except Exception as e:
        logger.error(f"Error in get_artist_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch artist chart data: {str(e)}")


@cached(ttl=60 * 60 * 24 * 7, key='taste-picks-chart', alias='default')
async def get_taste_picks_chart():
    try:
        chart = await get_release_chart(
            chart_name='billboard-tastemaker-albums'
        )
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days

        # add update date
        chart['updated_at'] = now.isoformat()

        # add cache info
        chart['cache_info'] = {
            'expired_at': expired_at.isoformat(),
            'cached_at': now.isoformat()
        }

        # add more info
        chart['id'] = 'taste-picks'
        chart['title'] = 'Taste Picks'
        
        return chart
    except Exception as e:
        logger.error(f"Error in get_taste_picks_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch taste picks chart data: {str(e)}")

@cached(ttl=60 * 60 * 24 * 7, key='on-air-chart', alias='default')
async def get_on_air_chart():
    try:
        chart = await get_track_chart(
            chart_name='billboard-radio-songs',
        )
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days
        
        # add update date
        chart['updated_at'] = now.isoformat()

        # add cache info
        chart['cache_info'] = {
            'expired_at': expired_at.isoformat(),
            'cached_at': now.isoformat()
        }

        # add more info
        chart['id'] = 'on-air'
        chart['title'] = 'On Air'
            
        return chart
    except Exception as e:
        logger.error(f"Error in get_on_air_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch on air chart data: {str(e)}")
    
@cached(ttl=60 * 60 * 24 * 7, key='stream-hits-chart', alias='default')
async def get_stream_hits_chart():
    try:
        chart = await get_track_chart(
            chart_name='billboard-streaming-songs'
        )
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days

        # add update date
        chart['updated_at'] = now.isoformat()

        # add cache info
        chart['cache_info'] = {
            'expired_at': expired_at.isoformat(),
            'cached_at': now.isoformat()
        }

        # add more info
        chart['id'] = 'stream-hits'
        chart['title'] = 'Stream Hits'

        return chart
    except Exception as e:
        logger.error(f"Error in get_stream_hits_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch stream hits chart data: {str(e)}")
    
@cached(ttl=60 * 60 * 24 * 7, key='indie-gems-chart', alias='default')
async def get_indie_gems_chart():
    try:
        chart = await get_release_chart(
            chart_name='billboard-independent-albums'
        )
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days

        # add update date
        chart['updated_at'] = now.isoformat()

        # add cache info
        chart['cache_info'] = {
            'expired_at': expired_at.isoformat(),
            'cached_at': now.isoformat()
        }

        # add more info
        chart['id'] = 'indie-gems'
        chart['title'] = 'Indie Gems'   

        return chart
    except Exception as e:
        logger.error(f"Error in get_indie_gems_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch indie gems chart data: {str(e)}")
    
@cached(ttl=60 * 60 * 24 * 7, key='rising-stars-chart', alias='default')
async def get_rising_stars_chart():
    try:
        chart = await get_artist_chart(
            chart_name='billboard-emerging-artists'
        )
        now = datetime.now()
        expired_at = now + timedelta(hours=168)  # 7 days

        # add update date
        chart['updated_at'] = now.isoformat()

        # add cache info
        chart['cache_info'] = {
            'expired_at': expired_at.isoformat(),
            'cached_at': now.isoformat()
        }
        
        # add more info
        chart['id'] = 'rising-stars'
        chart['title'] = 'Rising Stars'

        return chart
    except Exception as e:
        logger.error(f"Error in get_rising_stars_chart: {str(e)}")
        raise DiscoverContentException(f"Failed to fetch rising stars chart data: {str(e)}")

chart_map = {
    'taste-picks': get_taste_picks_chart,
    'on-air': get_on_air_chart,
    'stream-hits': get_stream_hits_chart,
    'indie-gems': get_indie_gems_chart,
    'rising-stars': get_rising_stars_chart,
}

async def get_all_charts():
    """
    Get all charts.

    Returns:
        JSON: A JSON response containing the list of charts
    """
    charts = []
    for chart_function in chart_map.values():
        chart = await chart_function()
        charts.append(chart)
    return charts
