"""
Spider for crawling Spotify artist data using Playwright.
"""

import scrapy
from urllib.parse import quote
from scrapy.http import Request
from scrapy_playwright.page import PageMethod

from lidarrmetadata import util, provider
from crawler.items import ArtistImageItem


class SpotifyArtistImageSpider(scrapy.Spider):
    """
    Spider for crawling Spotify artist data.
    
    This spider uses Playwright to handle dynamic content loading.
    It fetches artist images from Spotify based on cached Spotify IDs.
    """
    
    name = 'spotify-artist-image'
    allowed_domains = ['open.spotify.com']
    custom_settings = {
        'CONCURRENT_REQUESTS': 32,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 32,
        'DOWNLOAD_DELAY': 0.5,
    }
    
    def __init__(self, batch_size=100, offset=0, *args, **kwargs):
        """
        Initialize the spider.
        
        Args:
            batch_size: Number of artists to process in each batch
            offset: Offset to start processing from
        """
        super(SpotifyArtistImageSpider, self).__init__(*args, **kwargs)
        self.batch_size = int(batch_size)
        self.offset = int(offset)
        self.spotify_cache = util.SPOTIFY_CACHE
        self.artist_image_cache = util.ARTIST_IMAGE_CACHE
        
        # 添加统计计数器
        self.stats = {
            'total_processed': 0,
            'image_found': 0,
            'skipped_cached': 0,
            'no_spotify_ids': 0,
            'no_artist_name': 0
        }
        
        # Initialize the MusicBrainz provider for artist name lookup
        self.mb_provider = None
        self.setup_mb_provider()
        
        self.logger.info(f"Initialized SpotifyArtistImageSpider with batch size {self.batch_size}, offset {self.offset}")
    
    def setup_mb_provider(self):
        """Set up the MusicBrainz database provider"""
        try:
            config = util.CONFIG
            provider_args = config.PROVIDERS['MUSICBRAINZDBPROVIDER']
            self.mb_provider = provider.MusicbrainzDbProvider()
            self.logger.info("Successfully initialized MusicBrainzDbProvider")
        except Exception as e:
            self.logger.error(f"Failed to initialize MusicBrainzDbProvider: {e}")
    
    def async_to_sync(self, coro):
        """
        Helper method to run an async coroutine in a synchronous context.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        try:
            import nest_asyncio
            import asyncio
            
            # Patch asyncio to allow nested event loops
            nest_asyncio.apply()
            
            # Get or create an event loop
            loop = asyncio.get_event_loop()
            
            # Run the coroutine and return its result
            return loop.run_until_complete(coro)
        except Exception as e:
            self.logger.error(f"Error in async_to_sync: {e}")
            return None
    
    async def get_spotify_cache_keys(self, limit, offset):
        """
        Get the Spotify cache keys.
        
        Args:
            limit: Maximum number of keys to retrieve
            offset: Offset to start from
            
        Returns:
            Dictionary with cache keys and metadata
        """
        try:
            return await self.spotify_cache.get_all_keys_paged(limit=limit, offset=offset)
        except Exception as e:
            self.logger.error(f"Error getting Spotify cache keys: {e}")
            return {"keys": [], "total": 0, "has_more": False}
    
    async def get_artist_names(self, mbids):
        """
        Get artist names for the given MusicBrainz IDs.
        
        Args:
            mbids: List of MusicBrainz IDs
            
        Returns:
            Dictionary mapping MusicBrainz IDs to artist names
        """
        try:
            artists = await self.mb_provider.get_artists_by_id(mbids)
            return {artist['id']: artist.get('artist_name', '') for artist in artists if 'id' in artist}
        except Exception as e:
            self.logger.error(f"Error getting artist names: {e}")
            return {}
    
    async def get_spotify_ids(self, mbid):
        """
        Get Spotify IDs for the given MusicBrainz ID.
        
        Args:
            mbid: MusicBrainz ID
            
        Returns:
            List of Spotify IDs for the artist
        """
        try:
            spotify_ids, _ = await self.spotify_cache.get(mbid)
            if spotify_ids:
                if isinstance(spotify_ids, str):
                    # Handle case where it's stored as a single string
                    return [spotify_ids]
                elif isinstance(spotify_ids, list):
                    return spotify_ids
            return []
        except Exception as e:
            self.logger.error(f"Error getting Spotify IDs for {mbid}: {e}")
            return []
    
    async def check_artist_image_cache(self, mbid):
        """
        Check if the artist image is already in the cache.
        
        Args:
            mbid: MusicBrainz ID
            
        Returns:
            True if the image is already cached, False otherwise
        """
        try:
            cached_result = await self.artist_image_cache.get(mbid)
            
            # 处理元组结果 - 第一个元素是实际值
            if isinstance(cached_result, tuple) and len(cached_result) >= 1:
                cached_image = cached_result[0]
            else:
                cached_image = cached_result
            
            return cached_image is not None and cached_image != ""
        except Exception as e:
            self.logger.error(f"Error checking artist image cache for {mbid}: {e}")
            return False
    
    def start_requests(self):
        """
        Start the crawling process by fetching artist data from the cache.
        """
        self.logger.info("Starting to fetch artist data from cache")
        
        try:
            # Get keys from the SPOTIFY_CACHE
            current_offset = self.offset
            
            # 首先获取第一批数据来确定总数
            initial_cache_data = self.async_to_sync(self.get_spotify_cache_keys(1, 0))
            if not initial_cache_data:
                self.logger.error("No data found in Spotify cache")
                return
            
            total_count = initial_cache_data.get('total', 0)
            self.logger.info(f"Total artists in Spotify cache: {total_count}")
            
            # 循环处理所有数据，每次处理一个批次
            while current_offset < total_count:
                # Process artist data directly here
                cache_data = self.async_to_sync(self.get_spotify_cache_keys(self.batch_size, current_offset))
                
                if not cache_data or not cache_data.get('keys'):
                    self.logger.error(f"No data found in Spotify cache at offset {current_offset}")
                    break
                
                mbids = cache_data['keys']
                self.logger.info(f"Processing batch of {len(mbids)} artists starting at offset {current_offset}")
                
                # Get artist names
                artist_names = self.async_to_sync(self.get_artist_names(mbids))
                
                # Process each artist
                for mbid in mbids:
                    # Skip if already in ARTIST_IMAGE_CACHE
                    if self.async_to_sync(self.check_artist_image_cache(mbid)):
                        self.logger.info(f"Skipping artist {mbid} - already in cache")
                        self.stats['skipped_cached'] += 1
                        continue
                    
                    # Get Spotify IDs
                    spotify_ids = self.async_to_sync(self.get_spotify_ids(mbid))
                    
                    if not spotify_ids:
                        self.logger.warning(f"No Spotify IDs found for artist {mbid}")
                        self.stats['no_spotify_ids'] += 1
                        continue
                    
                    # Get artist name
                    artist_name = artist_names.get(mbid)
                    
                    if not artist_name:
                        self.logger.warning(f"No name found for artist {mbid}")
                        self.stats['no_artist_name'] += 1
                        continue
                    
                    # Generate search request
                    yield self.generate_search_request(mbid, artist_name, spotify_ids)
                    # 增加处理计数
                    self.stats['total_processed'] += 1
                    
                    # 每批次结束时打印当前统计
                    if self.stats['total_processed'] % 10 == 0:
                        self._log_stats()
                
                # 更新offset为下一批次
                current_offset += self.batch_size
                self.logger.info(f"Moving to next batch at offset {current_offset}")
                # 每批次结束打印统计
                self._log_stats()
        
        except Exception as e:
            self.logger.error(f"Error fetching data from cache: {e}")
            # 异常情况下，也打印统计信息
            self._log_stats()
    
    def generate_search_request(self, mbid, artist_name, spotify_ids):
        """
        Generate a search request for the given artist.
        
        Args:
            mbid: The MusicBrainz ID of the artist
            artist_name: The name of the artist
            spotify_ids: List of Spotify IDs for the artist
            
        Returns:
            A Request object for the search URL
        """
        # URL encode the artist name for the search
        encoded_query = quote(artist_name)
        url = f"https://open.spotify.com/search/{encoded_query}/artists"
        
        self.logger.info(f"Generating search request for {artist_name} (MBID: {mbid})")
        
        # Use Playwright for this request to handle JavaScript rendering
        return Request(
            url=url,
            callback=self.parse_search_results,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_methods": [
                    # Wait until network is idle - good indicator that dynamic content has loaded
                    # PageMethod("wait_for_load_state", "networkidle")
                ],
                "errback": self.errback,
                "mbid": mbid,
                "artist_name": artist_name,
                "spotify_ids": spotify_ids
            }
        )
    
    async def parse_search_results(self, response):
        """
        Parse the search results page to find the artist card with matching Spotify ID.
        
        Args:
            response: The response object containing the page and its content
        """
        page = response.meta["playwright_page"]
        mbid = response.meta["mbid"]
        artist_name = response.meta["artist_name"]
        spotify_ids = response.meta["spotify_ids"]
        
        self.logger.info(f"Parsing search results for {artist_name} (MBID: {mbid})")
        
        try:
            # For each Spotify ID, try to find a matching card
            image_url = None
            
            for spotify_id in spotify_ids:
                self.logger.info(f"Looking for Spotify ID: {spotify_id}")
                
                # First, try the data-encore-id approach (as seen in the example HTML)
                # Look for any element with attributes containing the Spotify ID
                selectors = [
                    f"[data-testid*='{spotify_id}']",
                    f"[aria-labelledby*='{spotify_id}']",
                    f"[id*='{spotify_id}']",
                    f"[data-encore-id='card'][aria-labelledby*='{spotify_id}']",
                    f"[role='group'][aria-labelledby*='{spotify_id}']"
                ]
                
                for selector in selectors:
                    self.logger.info(f"Trying selector: {selector}")
                    elements = await page.query_selector_all(selector)
                    
                    if elements:
                        self.logger.info(f"Found {len(elements)} elements with selector: {selector}")
                        
                        # Process each element to find an image
                        for element in elements:
                            # First, try to find an image directly within this element
                            img = await element.query_selector("img")
                            
                            if img:
                                src = await img.get_attribute('src')
                                if src:
                                    self.logger.info(f"Found image with src: {src}")
                                    image_url = src
                                    break
                
                # If no image found yet, try a page-wide search
                # if not image_url:
                #     # Dump the page HTML for debugging (only in development)
                #     if not image_url:
                #         self.logger.info("No image found with direct selectors, trying page content analysis")
                        
                #         # Take a screenshot for debugging
                #         await page.screenshot(path=f"debug_{spotify_id}.png")
                        
                #         # Try to find any card with an image
                #         cards = await page.query_selector_all("[data-encore-id='card']")
                #         self.logger.info(f"Found {len(cards)} cards on the page")
                        
                #         for card in cards:
                #             card_html = await page.evaluate("(element) => element.outerHTML", card)
                            
                #             # Check if this card contains the Spotify ID
                #             if spotify_id in card_html:
                #                 self.logger.info(f"Found card containing Spotify ID: {spotify_id}")
                #                 img = await card.query_selector("img")
                #                 if img:
                #                     src = await img.get_attribute('src')
                #                     if src:
                #                         self.logger.info(f"Found image with src: {src}")
                #                         image_url = src
                #                         break
                
                if image_url:
                    break
            
            # If we found an image URL, yield an item
            if image_url:
                # 增加成功找到图片的计数
                self.stats['image_found'] += 1
                
                item = ArtistImageItem()
                item['mbid'] = mbid
                item['spotify_ids'] = spotify_ids
                item['artist_name'] = artist_name
                item['image_url'] = image_url
                
                # 打印当前成功率
                success_rate = self._percentage(self.stats['image_found'], self.stats['total_processed'])
                self.logger.info(f"Image found for {artist_name}! Current success rate: {success_rate}%")
                
                yield item
            else:
                self.logger.warning(f"No image found for artist {artist_name} (MBID: {mbid})")
            
        except Exception as e:
            self.logger.error(f"Error parsing search results: {e}")
        finally:
            # Always close the page to avoid memory leaks
            await page.close()

    async def errback(self, failure):
        """
        Handle errors that occur during the request.
        
        Args:
            failure: The failure information
        """
        page = failure.request.meta.get("playwright_page")
        if page:
            await page.close()
            
        mbid = failure.request.meta.get("mbid", "unknown")
        artist_name = failure.request.meta.get("artist_name", "unknown")
        
        self.logger.error(f"Error processing artist {artist_name} (MBID: {mbid}): {failure}") 

    def _log_stats(self):
        """输出当前统计信息到日志"""
        self.logger.info("="*50)
        self.logger.info("CURRENT PROCESSING STATISTICS:")
        self.logger.info(f"Total requests processed: {self.stats['total_processed']}")
        self.logger.info(f"Successfully found images: {self.stats['image_found']} ({self._percentage(self.stats['image_found'], self.stats['total_processed'])}%)")
        self.logger.info(f"Skipped (already cached): {self.stats['skipped_cached']}")
        self.logger.info(f"No Spotify IDs found: {self.stats['no_spotify_ids']}")
        self.logger.info(f"No artist name found: {self.stats['no_artist_name']}")
        self.logger.info("="*50)

    def _percentage(self, part, total):
        """计算百分比，避免除以零的错误"""
        return round((part / total) * 100, 2) if total > 0 else 0

    # 添加爬虫关闭时的统计报告
    def closed(self, reason):
        """
        当爬虫关闭时调用，输出最终的统计信息
        
        Args:
            reason: 关闭原因
        """
        self.logger.info("="*70)
        self.logger.info("FINAL STATISTICS REPORT")
        self.logger.info("="*70)
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total requests processed: {self.stats['total_processed']}")
        self.logger.info(f"Successfully found images: {self.stats['image_found']} ({self._percentage(self.stats['image_found'], self.stats['total_processed'])}%)")
        self.logger.info(f"Failed to find images: {self.stats['total_processed'] - self.stats['image_found']}")
        self.logger.info(f"Skipped (already cached): {self.stats['skipped_cached']}")
        self.logger.info(f"No Spotify IDs found: {self.stats['no_spotify_ids']}")
        self.logger.info(f"No artist name found: {self.stats['no_artist_name']}")
        self.logger.info("="*70) 