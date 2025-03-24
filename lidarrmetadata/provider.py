import abc
import collections
import contextlib
import datetime
from datetime import timedelta
import time
import pytz
import imp
import logging
import pkg_resources
import re
import six
from timeit import default_timer as timer
from urllib.parse import urlparse
from urllib.parse import quote as url_quote

import asyncio
import aiohttp
import asyncpg
import json
import base64

import dateutil.parser

from lidarrmetadata.config import get_config
from lidarrmetadata import limit
from lidarrmetadata import stats
from lidarrmetadata import util
from lidarrmetadata.cache import conn
from lidarrmetadata.util import deprecated

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info('Have provider logger')

CONFIG = get_config()

# Provider class dictionary
PROVIDER_CLASSES = {}

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)

def get_providers_implementing(cls):
    """
    Gets list of providers implementing mixin
    :param cls: Mixin class for implementation
    :return: List of providers inheriting from cls
    """
    return [p for p in Provider.providers if isinstance(p, cls)]


def _get_rate_limiter(key=None):
    """
    Builds a rate limiter from config values
    :return: RateLimiter appropriate to config
    """
    try:
        limit_class = getattr(limit, CONFIG.EXTERNAL_LIMIT_CLASS)
    except AttributeError:
        logger.error('Limit class "{}" does not exist. Defaulting to NullRateLimiter'.format(
            CONFIG.EXTERNAL_LIMIT_CLASS))
        return limit.NullRateLimiter()

    logger.info('Initializing rate limiter class {} with key {}'.format(limit_class, key))
    if limit_class == limit.NullRateLimiter:
        return limit.NullRateLimiter()
    elif limit_class == limit.RedisRateLimiter:
        return limit.RedisRateLimiter(key=key,
                                      redis_host=CONFIG.EXTERNAL_LIMIT_REDIS_HOST,
                                      redis_port=CONFIG.EXTERNAL_LIMIT_REDIS_PORT,
                                      redis_db=CONFIG.EXTERNAL_LIMIT_REDIS_DB,
                                      queue_size=CONFIG.EXTERNAL_LIMIT_QUEUE_SIZE,
                                      time_delta=CONFIG.EXTERNAL_LIMIT_TIME_DELTA)
    elif limit_class == limit.SimpleRateLimiter:
        return limit.SimpleRateLimiter(queue_size=CONFIG.EXTERNAL_LIMIT_QUEUE_SIZE,
                                       time_delta=CONFIG.EXTERNAL_LIMIT_TIME_DELTA)
    else:
        logger.warning(
            "Don't know how to instantiate {}. Defaulting to NullRateLimiter".format(limit_class))
        return limit.NullRateLimiter()

def response_url(url: str) -> str:
    """
    Transforms a URL to a response URL, which can take into account things such as a hosted cache
    for third-party services.
    """
    parsed = urlparse(url)
    if parsed.netloc.endswith("theaudiodb.com"):
        new_path = re.sub("^/images/media/", "", parsed.path)
        old_parsed = parsed
        parsed = parsed._replace(
            netloc=CONFIG.IMAGE_CACHE_HOST,
            path=f"v1/tadb/{new_path}"
        )
        logger.debug(f"Transformed {old_parsed.geturl()} to {parsed.geturl()}")
    else:
        logger.debug(f"Leaving {parsed.geturl()} as is")
    return parsed.geturl()

class MixinBase(six.with_metaclass(abc.ABCMeta, object)):
    pass


class ArtistByIdMixin(MixinBase):
    """
    Gets artist by id
    """

    @abc.abstractmethod
    def get_artists_by_id(self, artist_id):
        """
        Gets artist by id
        :param artist_id: ID of artist
        :return: Artist matching ID or None
        """
        pass
    
    @abc.abstractmethod
    def redirect_old_artist_id(self, artist_id):
        """
        Given an id that isn't found, see if it's an old id that needs redirecting.  If so, return new id.
        """
        pass

    @abc.abstractmethod
    def get_artist_id_from_spotify_id(self, spotify_id):
        """
        Gets artist id from spotify id
        :param spotify_id: Spotify ID of artist
        :return: Artist matching ID or None
        """
        pass

    @abc.abstractmethod
    def get_spotify_mappings(self, limit=100, offset=0):
        """
        Grabs link entities from database and parses for spotify maps with pagination support
        
        Args:
            limit (int): 每页返回的最大记录数量，默认100条
            offset (int): 分页偏移量，默认从0开始
            
        Returns:
            A list of (mbid, spotify_id) tuples for the requested page
        """
        pass


class ArtistIdListMixin(MixinBase):
    """
    Returns a list of all artist ids we should cache
    """
    
    @deprecated('Use get_artist_ids_paged instead.')
    @abc.abstractmethod
    def get_all_artist_ids(self):
        pass

    @abc.abstractmethod
    async def get_artist_ids_paged(self, limit=1000, offset=0):
        """
        Gets artist ids with pagination
        :param limit: Number of results to return per page
        :param offset: Number of results to skip
        :return: List of artist ids for the requested page
        """
        pass

class MusicBrainzCoverArtMixin(MixinBase):
    """
    从 MusicBrainz 数据库获取 Cover Art Archive 的封面信息。
    
    Cover Art Archive 是 MusicBrainz 的官方封面数据库,存储了专辑发行版和专辑组的封面图片。
    该 Mixin 通过查询 MusicBrainz 数据库获取封面信息,包括:
    - 封面图片 ID
    - 封面类型(前封面、后封面、CD等)
    - 图片尺寸
    - 上传时间等元数据
    """

    @abc.abstractmethod
    async def get_release_cover_art(self, release_ids):
        """
        获取一个或多个发行版(Release)的封面信息
        
        :param release_ids: 单个 MusicBrainz Release ID 或 ID 列表
        :return: 如果传入单个 ID，返回单个封面信息；如果传入 ID 列表，返回字典 {release_id: cover_art_data}
            cover_art_data 包含:
            {
                'id': 封面图片ID,
                'type': 封面类型(front/back/medium等),
                'approved': 是否已审核通过,
                'edit': 最后编辑ID,
                'comment': 备注信息,
                'uploaded': 上传时间
            }
        """
        pass

class ArtistNameSearchMixin(MixinBase):
    """
    Searches for artist with artist name
    """

    @abc.abstractmethod
    def search_artist_name(self, name, limit=None):
        """
        Searches for artist with name
        :param name: Name to search for
        :param limit: Limit of number of results to return. Defaults to None, indicating no limit
        :return: List of possible matches
        """
        pass

class SpotifyAuthMixin(MixinBase):
    """
    Provides/renews tokens for spotify API
    """

    @abc.abstractmethod
    def get_token(self, code):
        """
        Gets spotify token as part of oAuth callback
        :param code: Auth code from spotify
        :return: Spotify access and refresh code
        """
        pass

    def renew_token(self, refresh_token):
        """
        Renews access code given refresh token
        :param refresh_token: refresh token
        :return: Spotify access and refresh code
        """
        pass

class ReleaseGroupByArtistMixin(MixinBase):
    """
    Gets release groups for artist
    """

    @abc.abstractmethod
    def get_release_groups_by_artist(self, artist_id):
        """
        Gets release groups by artist by ID
        :param artist_id: ID of artist
        :return: List of release groups by artist
        """
        pass


class ReleaseGroupByIdMixin(MixinBase):
    """
    Gets release group by ID
    """

    @abc.abstractmethod
    def get_release_groups_by_id(self, rgids):
        """
        Gets release group by ID
        :param rgid: List of release group IDs
        :return: Release Group corresponding to rgid
        """
        pass
    
    @abc.abstractmethod
    def redirect_old_release_group_id(self, artist_id):
        """
        Given an id that isn't found, see if it's an old id that needs redirecting.  If so, return new id.
        """
        pass

    @abc.abstractmethod
    def get_release_group_id_from_spotify_id(self, spotify_id):
        """
        Gets artist id from spotify id
        :param spotify_id: Spotify ID of album
        :return: Release group matching ID or None
        """
        pass


class ReleaseGroupIdListMixin(MixinBase):
    """
    Returns a list of all artist ids we should cache
    """
    
    @abc.abstractmethod
    def get_all_release_group_ids(self):
        pass


class ReleasesByReleaseGroupIdMixin(MixinBase):
    """
    Gets releases by ReleaseGroup ID
    """

    @abc.abstractmethod
    def get_releases_by_rgid(self, rgid):
        """
        Gets releases by release group ID
        :param rgid: Release group ID
        :return: Releases corresponding to rgid or rid
        """
        pass

class ReleaseByIdMixin(MixinBase):
    """
    Gets release by ID
    """
    @abc.abstractmethod
    def get_release_by_id(self, rids):
        """
        Gets release by ID
        :param rids: List of release IDs
        :return: Release corresponding to rid
        """
        pass

class TrackByIdMixin(MixinBase):
    """
    Gets track by ID
    """
    @abc.abstractmethod
    def get_track_by_id(self, track_ids):
        """
        Gets track by ID
        :param track_ids: List of track IDs
        :return: Track corresponding to track_id
        """
        pass

class SeriesMixin(MixinBase):
    """
    Musicbrainz series
    """

    @abc.abstractmethod
    def get_series(self, mbid):
        pass


class TrackSearchMixin(MixinBase):
    """
    Search for tracks by name
    """

    @abc.abstractmethod
    def search_track(self, query, artist_name=None, album_name=None, limit=10):
        """
        Searches for tracks matching query
        :param query: Search query
        :param artist_name: Artist name. Defaults to None, in which case tracks from all artists are returned
        :param album_name: Album name. Defaults to None, in which case tracks from all albums are returned
        :param limit: Maximum number of results to return. Defaults to 10. Returns all results if negative
        :return: List of track results
        """
        pass


class ArtistOverviewMixin(MixinBase):
    """
    Gets overview for artist
    """

    @abc.abstractmethod
    def get_artist_overview(self, artist_id):
        pass


class ArtistArtworkMixin(MixinBase):
    """
    Gets art for artist
    """

    @abc.abstractmethod
    def get_artist_images(self, artist_id):
        """
        Gets images for artist with ID
        :param artist_id: ID of artist
        :return: List of results
        """
        pass


class ReleaseArtworkMixin(MixinBase):
    """
    Gets art for release
    """
    @abc.abstractmethod
    def get_release_images(self, release_id):
        """
        Gets images for release with ID
        :param release_id: ID of release
        :return: image links for release with different qualities, e.g.
        {
            'small': 'http://xxx',
            'mid': 'http://xxx',
            'large': 'http://xxx',
            'original': 'http://xxx'
        }
        """
        pass

    @abc.abstractmethod
    def get_release_images_multi(self, release_ids):
        """
        Gets images for multiple releases with IDs
        :param release_ids: List of release IDs
        :return: List of tuples (images, expiry) where images is a dict of image links with different qualities:
        {
            'small': 'http://xxx',
            'mid': 'http://xxx',
            'large': 'http://xxx',
            'original': 'http://xxx'
        }
        """
        pass


class AlbumNameSearchMixin(MixinBase):
    """
    Searches for album by name
    """

    @abc.abstractmethod
    def search_album_name(self, name, limit=None, artist_name=''):
        """
        Searches for album with name
        :param name: Name of album
        :param limit: Limit of number of results to return. Defaults to None, indicating no limit
        :param artist_name: Artist name restriction
        :return: List of albums
        """
        pass

class ReleaseNameSearchMixin(MixinBase):
    """
    Searches for release by name
    """
    @abc.abstractmethod
    def search_release_name(self, name, limit=None, artist_name=''):
        """
        Searches for release with name
        :param name: Name of release
        :param limit: Limit of number of results to return. Defaults to None, indicating no limit
        :param artist_name: Artist name restriction
        :return: List of releases
        """
        pass

class RecordingNameSearchMixin(MixinBase):
    """
    Searches for recording by name
    """
    @abc.abstractmethod
    def search_recording_name(self, name, limit=None, artist_name=''):
        """
        Searches for recording with name
        :param name: Name of recording
        :param limit: Limit of number of results to return. Defaults to None, indicating no limit
        :param artist_name: Artist name restriction
        :return: List of recordings
        """
        pass

class DataVintageMixin(MixinBase):
    """
    Returns vintage of data in use
    """
    
    @abc.abstractmethod
    def data_vintage(self):
        pass

class InvalidateCacheMixin(MixinBase):
    """
    Invalidates cache for updated items
    """
    
    @abc.abstractmethod
    def invalidate_cache(self, prefix, since):
        """
        Invalidates any internal cache as appropriate and returns entities that need invalidating at API level
        :param prefix: URL prefix for the instance we are clearing cache for
        :return: Dict {"artists":[ids], "albums":[ids]} of artists/albums that need to be updated
        """
        pass
    
class AsyncDel(MixinBase):
    """
    Hook to finalize async items
    """
    
    @abc.abstractmethod
    async def _del(self, prefix):
        """
        Run at initialization using await
        """
        pass

    
class ProviderMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace):
        """
        Creates class and registers it to PROVIDER_CLASSES
        :param mcls: Parent metaclass
        :param name: Name of class
        :param bases: Base classes
        :param namespace: Class dictionary
        :return: Newly created class
        """
        cls = super(ProviderMeta, mcls).__new__(mcls, name, bases, namespace)
        PROVIDER_CLASSES[name] = cls
        return cls


class Provider(six.with_metaclass(ProviderMeta, object)):
    """
    Provider base class
    """

    # List of provider instances
    providers = []

    def __init__(self):
        logger.info('Initializing provider {}'.format(self.__class__))
        self.providers.append(self)
        
class ProviderException(Exception):
    """
    Base exception for providers
    """
    pass

class ProviderNotFoundException(ProviderException):
    """
    Exception for when searching yields no results
    """
    pass

class ProviderUnavailableException(ProviderException):
    """
    Exception for when the provider is temporarily unavailable
    """
    pass

class NonRetryableError(ProviderException):
    """
    Exception indicating an error response that should not be retried
    """
    def __init__(self, response):
        self.response = response
        super(NonRetryableError, self).__init__(f"Non-retryable error: {response.status}")

class HttpProvider(Provider,
                   AsyncDel):
    """
    Generic provider which makes external HTTP queries
    """
    
    def __init__(self, name, session = None, limiter = None):
        super(HttpProvider, self).__init__()
        
        self._name = name
        self._stats = stats.TelegrafStatsClient(CONFIG.STATS_HOST,
                                                CONFIG.STATS_PORT) if CONFIG.ENABLE_STATS else None
        
        self._session = session
        self.__session_lock = None
            
        if limiter:
            self._limiter = limiter
        else:
            self._limiter = _get_rate_limiter(key=name)
        
    @property
    def _session_lock(self):
        if self.__session_lock is None:
            self.__session_lock = asyncio.Lock()
        return self.__session_lock
    
    async def _get_session(self):
        if self._session is None:
            async with self._session_lock:
                logger.debug("Initializing AIOHTTP Session")
                self._session = aiohttp.ClientSession(
                    timeout = aiohttp.ClientTimeout(total=CONFIG.EXTERNAL_TIMEOUT / 1000),
                    trust_env=True
                )
                
        return self._session
            
    async def _del(self):
        session = await self._get_session()
        if session:
            await session.close()
        
    def _count_request(self, result_type):
        if self._stats:
            self._stats.metric('external_request', {result_type: 1}, tags={'provider': self._name})

    def _record_response_result(self, response, elapsed):
        if self._stats:
            self._stats.metric('external',
                               {
                                   'response_time': elapsed
                               },
                               tags={
                                   'provider': self._name,
                                   'response_status_code': response.status
                               })

    # 定义不应该重试的状态码
    NON_RETRYABLE_STATUS_CODES = {
        400, 401, 403, 404, 405, 409, 410, 
        413, 414, 415, 422, 
        501, 505, 508
    }

    async def get(self, url, raise_on_http_error=True, max_retries=0, retry_delay=1.0, **kwargs):
        """
        发送 GET 请求
        
        :param url: 要请求的 URL
        :param raise_on_http_error: 是否在 HTTP 错误时抛出异常
        :param max_retries: 最大重试次数
        :param retry_delay: 重试延迟时间（秒）
        :param kwargs: 传递给 aiohttp 的其他参数
        :return: 响应内容（通常是 JSON）
        """
        start = timer()
        retry_count = 0
        timeout = aiohttp.ClientTimeout(total=10)
        
        async def _try_request():
            session = await self._get_session()
            try:
                async with session.get(url, timeout=timeout, **kwargs) as response:
                    content_length = int(response.headers.get('Content-Length', 0))
                    self._record_response_result(response, timer() - start)
                    logger.debug(f"Response status: {response.status}")
                    
                    if not response.ok and raise_on_http_error:
                        # 检查是否是不应该重试的状态码
                        if response.status in self.NON_RETRYABLE_STATUS_CODES:
                            logger.debug(f"Non-retryable status code {response.status} for URL {url}")
                            # 抛出一个特殊异常，表示这是不可重试的错误
                            raise NonRetryableError(response)
                            
                        # 对于可重试的错误状态码，记录更多信息
                        if response.status == 429:
                            retry_after = response.headers.get('Retry-After', None)
                            logger.warning(f"Rate limited (429) for URL {url}. Retry-After: {retry_after}")
                        if response.status >= 500:
                            logger.warning(f"Server error ({response.status}) for URL {url}")
                        
                        # 如果还有重试次数，返回None触发重试
                        if retry_count < max_retries:
                            return None
                        
                        # 已达最大重试次数，抛出标准异常
                        response.raise_for_status()
                    return await response.json()
            
            except NonRetryableError as e:
                # 对于不可重试的错误，直接让原始异常传播
                e.response.raise_for_status()
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # 对于连接错误，记录日志并根据重试策略决定是否重试
                logger.warning(f"HTTP error ({type(e).__name__}) for URL {url}: {str(e)}")
                if retry_count < max_retries:
                    return None
                raise ProviderUnavailableException(f"HTTP error after {retry_count + 1} attempts: {str(e)}")
        
        # 主重试循环
        while True:
            try:
                result = await _try_request()
                if result is not None:
                    return result
                
                # 如果请求失败且还有重试次数，则继续重试
                retry_count += 1
                logger.debug(f"Retrying request ({retry_count}/{max_retries}): {url}")
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                # 捕获所有异常并抛出自定义异常
                if isinstance(e, ProviderUnavailableException):
                    raise
                raise ProviderUnavailableException(f"Error accessing {url}: {str(e)}")

    async def get_with_limit(self, url, raise_on_http_error=True, max_retries=0, retry_delay=1.0, **kwargs):
        """
        Performs an HTTP GET request with rate limiting and optional retry mechanism
        
        :param url: URL to request
        :param raise_on_http_error: Whether to raise an exception on HTTP error
        :param max_retries: Maximum number of retry attempts (default: 0, no retries)
        :param retry_delay: Delay between retries in seconds (default: 1.0)
        :param kwargs: Additional arguments to pass to session.get
        :return: JSON response
        """
        try:
            with self._limiter.limited():
                return await self.get(url, raise_on_http_error, max_retries=max_retries, retry_delay=retry_delay, **kwargs)
        except limit.RateLimitedError:
            logger.debug(f'{self._name} request rate limited')
            self._count_request('ratelimit')

class CacheReleaseImageProvider(Provider, ReleaseArtworkMixin):
    """
    Only get images from cache
    """
    def __init__(self):
        super(CacheReleaseImageProvider, self).__init__()
        self._cache = util.RELEASE_IMAGE_CACHE

    async def get_release_images(self, release_id):
        """
        Get release images from cache
        """
        logger.debug(f"CacheReleaseImageProvider: getting images for release_id={release_id}")
        result = await self._cache.get(release_id)
        logger.debug(f"CacheReleaseImageProvider: got images for release_id={release_id}, found: {result[0] is not None}")
        return result
    
    async def get_release_images_multi(self, release_ids):
        """
        Get release images from cache
        """
        logger.debug(f"CacheReleaseImageProvider: getting images for {len(release_ids)} releases")
        results = await self._cache.multi_get(release_ids)
        found_count = sum(1 for res in results if res[0] is not None)
        logger.debug(f"CacheReleaseImageProvider: found {found_count}/{len(release_ids)} release images in cache")
        return results

# TODO: Use this for artist image retrieval when crawling is complete
class CacheArtistImageProvider(Provider, ArtistArtworkMixin):
    """
    Only get images from cache
    """
    def __init__(self):
        super(CacheArtistImageProvider, self).__init__()
        self._cache = util.ARTIST_IMAGE_CACHE

    async def get_artist_images(self, artist_id):
        """
        Get artist images from cache
        """
        logger.debug(f"CacheArtistImageProvider: getting images for artist_id={artist_id}")
        result = await self._cache.get(artist_id)
        logger.debug(f"CacheArtistImageProvider: got images for artist_id={artist_id}, result: {result}")
        return result

class CoverArtArchiveProvider(HttpProvider, ReleaseArtworkMixin):
    """
    从 Cover Art Archive 获取专辑封面的 Provider
    """
    def __init__(self, session=None, limiter=None):
        """
        初始化 Provider
        """
        super(CoverArtArchiveProvider, self).__init__('coverart', session, limiter)
        self._db_provider = get_providers_implementing(MusicBrainzCoverArtMixin)[0]
        self._base_url = 'http://coverartarchive.org'
        
    async def get_release_images(self, release_id):
        """
        获取单个专辑封面
        :param release_id: MusicBrainz Release ID
        :return: (images, expiry) 元组，其中 images 包含不同尺寸的图片 URL
        """
        results = await self.get_release_images_multi([release_id])
        return results[0] if results else (None, utcnow())

    async def get_release_images_multi(self, release_ids):
        """
        批量获取多个专辑的封面
        :param release_ids: MusicBrainz Release ID 列表
        :return: 封面信息列表的列表
        """
        now = utcnow()
        logger.debug(f"Fetching images for {len(release_ids)} releases")
        
        # 先检查缓存
        cached_results = await asyncio.gather(*[util.RELEASE_IMAGE_CACHE.get(rid) for rid in release_ids])
        results = []
        uncached_ids = []
        uncached_indices = {}  # 使用字典来保存 release_id 到结果索引的映射
        
        # 处理缓存结果
        for i, (cached, expires) in enumerate(cached_results):
            if cached and expires > now:
                logger.debug(f"Cache hit for release {release_ids[i]}")
                results.append((cached, expires))
            else:
                results.append(None)
                rid = release_ids[i]
                uncached_ids.append(rid)
                uncached_indices[rid] = i
                
        if not uncached_ids:
            logger.debug("All release images were found in cache")
            return results
            
        logger.debug(f"Fetching {len(uncached_ids)} uncached release images")
        
        try:
            # 批量获取未缓存的封面
            cover_art_data_dict = await self._db_provider.get_release_cover_art(uncached_ids)
            
            ttl = CONFIG.CACHE_TTL['release_image']
            expiry = now + timedelta(seconds=ttl)
            
            # 处理每个未缓存的结果
            for release_id in uncached_ids:
                result_index = uncached_indices[release_id]
                
                # 如果没有找到任何封面数据，或者这个 release_id 没有对应的封面
                if not cover_art_data_dict or release_id not in cover_art_data_dict:
                    logger.debug(f'No cover art data found for release {release_id}, trying release group')
                    
                    # 获取 release 对应的 release group id
                    release_provider = get_providers_implementing(ReleaseByIdMixin)[0]
                    releases = await release_provider.get_release_by_id([release_id])
                    
                    if releases and releases[0].get('release_group_id'):
                        rg_id = releases[0]['release_group_id']
                        logger.debug(f"Found release group ID {rg_id} for release {release_id}")
                        
                        # 尝试从 release group 获取封面
                        try:
                            url = f"{self._base_url}/release-group/{rg_id}"
                            logger.debug(f"Requesting cover art from {url}")
                            response = await self.get(url, max_retries=3)
                            
                            if response and 'images' in response:
                                logger.debug(f"Found {len(response['images'])} images for release group {rg_id}")
                                front_image = next((img for img in response['images'] 
                                                  if img.get('front', False)), None)
                                if front_image:
                                    logger.debug(f"Using front image from release group {rg_id} for release {release_id}")
                                    
                                    if 'thumbnails' in front_image and 'image' in front_image:
                                        # 从 URL 中提取 release_id 和 image_id
                                        # URL 格式: https://coverartarchive.org/release/{release_id}/{image_id}.jpg
                                        image_url = front_image['image']
                                        logger.debug(f"Parsing image URL: {image_url}")
                                        
                                        try:
                                            # 正则表达式提取 release_id 和 image_id
                                            pattern = r'coverartarchive\.org/release/([a-f0-9-]+)/(\d+)\.jpg'
                                            match = re.search(pattern, image_url)
                                            
                                            if match:
                                                rid, image_id = match.groups()
                                                logger.debug(f"Extracted release_id={rid}, image_id={image_id}")
                                                
                                                # 使用 build_caa_url 构建一致的 URL
                                                images = {
                                                    'small': self.build_caa_url(rid, image_id, size=250),
                                                    'mid': self.build_caa_url(rid, image_id, size=500),
                                                    'large': self.build_caa_url(rid, image_id, size=1200),
                                                    'original': self.build_caa_url(rid, image_id)
                                                }
                                                
                                                # 缓存结果
                                                await util.RELEASE_IMAGE_CACHE.set(release_id, images, ttl=ttl)
                                                results[result_index] = (images, expiry)
                                                continue
                                            else:
                                                # 如果无法提取，只记录日志
                                                logger.warning(f"Could not extract IDs from URL: {image_url}")
                                        except Exception as e:
                                            # 如果解析出错，只记录日志
                                            logger.warning(f"Error parsing image URL: {e}")
                                    else:
                                        # 如果没有缩略图信息，只记录日志
                                        logger.warning("No thumbnails or image found in response")
                                    
                                    # 如果无法从 release group 获取有效的封面，继续尝试其他方法
                                    logger.debug(f"Failed to use release group cover art for release {release_id}")
                                else:
                                    logger.debug(f"No front image found for release group {rg_id}")
                            else:
                                logger.debug(f"No valid response or images found for release group {rg_id}")
                                
                        except Exception as e:
                            logger.error(f"Error getting release group cover art for {rg_id}: {e}")
                    else:
                        logger.debug(f"No release group found for release {release_id}")
                    
                    # 如果所有尝试都失败，设置为空结果
                    logger.debug(f"No cover art found for release {release_id} after all attempts")
                    results[result_index] = (None, now)
                    continue
                else:
                    logger.debug(f"Found cover art data for release {release_id}")
                    cover_art_data = cover_art_data_dict[release_id]
                    if not cover_art_data or 'id' not in cover_art_data:
                        logger.debug(f'Invalid cover art data for {release_id}')
                        results[result_index] = (None, now)
                        continue
                        
                    logger.debug(f"Found cover art ID {cover_art_data['id']} for release {release_id}")
                    image_id = cover_art_data['id']
                    images = {
                        'small': self.build_caa_url(release_id, image_id, size=250),
                        'mid': self.build_caa_url(release_id, image_id, size=500),
                        'large': self.build_caa_url(release_id, image_id, size=1200),
                        'original': self.build_caa_url(release_id, image_id)
                    }
                
                # 缓存结果
                await util.RELEASE_IMAGE_CACHE.set(release_id, images, ttl=ttl)
                results[result_index] = (images, expiry)
                
            logger.debug(f"Finished processing {len(uncached_ids)} release images")
            return results
            
        except ProviderUnavailableException as e:
            # 如果服务不可用，对所有未缓存的结果返回空列表
            logger.error(f"Provider unavailable: {e}")
            error_expiry = now + timedelta(seconds=CONFIG.CACHE_TTL['provider_error'])
            for i in uncached_indices.values():
                results[i] = (None, error_expiry)
            return results

    @staticmethod
    def build_caa_url(release_id, image_id, size=None):
        """
        Builds the cover art archive url for a given release and image id
        :param release_id: Musicbrainz release id
        :param image_id: Cover Art Archive image id 
        :param size: Size of image to return. Should be one of: 250, 500, 1200 or None.
        """
        base_url = f'https://imagecache.lidarr.audio/v1/caa/{release_id}/{image_id}'
        return f'{base_url}-{size}.jpg' if size else f'{base_url}.jpg'

class FanArtTvProvider(HttpProvider,
                       ArtistArtworkMixin,
                       InvalidateCacheMixin):
    def __init__(self,
                 api_key,
                 base_url='webservice.fanart.tv/v3/music/',
                 use_https=True,
                 session=None,
                 limiter=None):
        """
        Class initialization

        :param api_key: fanart.tv API key
        :param base_url: Base URL of API. Defaults to
                         webservice.fanart.tv/v3/music
        :param use_https: Whether or not to use https. Defaults to True.
        """
        super(FanArtTvProvider, self).__init__('fanart', session, limiter)

        self._api_key = api_key
        self._base_url = base_url
        self.use_https = use_https
        
        ## dummy value for initialization, will be picked up from redis later on
        self._last_cache_invalidation = time.time() - 60 * 60 * 24

    async def get_artist_images(self, artist_id):
        
        return await self.get_images(artist_id, self.parse_artist_images)
        
        
    async def get_images(self, mbid, handler):

        now = utcnow()
        cached, expires = await util.FANART_CACHE.get(mbid)

        if cached is not None and expires > now:
            return handler(cached), expires
        
        try:
            results = await self.get_by_mbid(mbid)
            results, ttl = await self.cache_results(mbid, results)
            
            return handler(results), now + timedelta(seconds=ttl)

        except (ProviderUnavailableException, ValueError):
            return handler(cached or {}), now + timedelta(seconds=CONFIG.CACHE_TTL['provider_error'])
        
    async def refresh_images(self, mbid):
        try:
            results = await self.get_by_mbid(mbid)
            await self.cache_results(mbid, results)

        except (ProviderUnavailableException, ValueError):
            logger.debug("Fanart unavailable")
            await util.FANART_CACHE.expire(mbid, CONFIG.CACHE_TTL['provider_error'])
        
    async def get_by_mbid(self, mbid):
        """
        Gets the fanart.tv response for resource with Musicbrainz id mbid
        :param mbid: Musicbrainz ID
        :return: fanart.tv response for mbid
        """
        url = self.build_url(mbid)
        response = await self.get_with_limit(url, raise_on_http_error=False)
        if response.get('status', None):
            return {}
        else:
            return response
        
    async def cache_results(self, mbid, results):
        ttl = CONFIG.CACHE_TTL['fanart']

        if results.get('mbid_id', None) == mbid:
            # This was a successful artist request, so cache albums also
            await util.FANART_CACHE.set(mbid, results, ttl=ttl)
            for id, album_result in results.get('albums', {}).items():
                await util.FANART_CACHE.set(id, album_result, ttl=ttl)

        else:
            # This was an album request or an unsuccessful artist request
            
            results = results.get('albums', {}).get(mbid, {})

            # There seems to be a bug in the fanart api whereby querying by album id sometimes returns not found
            # Don't overwrite a good cached value with not found.
            if not results:
                cached, expires = await util.FANART_CACHE.get(mbid)
                if cached:
                    results = cached
                
            await util.FANART_CACHE.set(mbid, results, ttl=ttl)

        return results, ttl
        
    async def invalidate_cache(self, prefix, since):
        logger.debug('Invalidating fanart cache')
        
        result = {'artists': [], 'albums': [], 'spotify_artists': [], 'spotify_albums': []}
        
        last_invalidation_key = prefix + 'FanartProviderLastCacheInvalidation'
        if since:
            since = int(since.timestamp())
        self._last_cache_invalidation = since or await util.CACHE.get(last_invalidation_key) or self._last_cache_invalidation
        current_cache_invalidation = int(time.time())
        
        # Since we don't have a fanart personal key we can only see things with a lag
        all_updates = await self.get_fanart_updates(self._last_cache_invalidation - CONFIG.FANART_API_DELAY_SECONDS)
        invisible_updates = await self.get_fanart_updates(current_cache_invalidation - CONFIG.FANART_API_DELAY_SECONDS)
        
        # Remove the updates we can't see
        artist_ids = self.diff_fanart_updates(all_updates, invisible_updates)
        logger.info('Invalidating artists given fanart updates:\n{}'.format('\n'.join(artist_ids)))

        # Mark artists as expired
        for id in artist_ids:
            await util.FANART_CACHE.expire(id, ttl=-1)
                
        await util.CACHE.set(last_invalidation_key, current_cache_invalidation)
        
        result['artists'] = artist_ids
        return result
    
    async def get_fanart_updates(self, time):
        url = self.build_url('latest') + '&date={}'.format(int(time))
        logger.debug(url)
        
        try:
            return await self.get(url, timeout=aiohttp.ClientTimeout(total=5))

        except Exception as error:
            logger.error(f'Error getting fanart updates: {error}')
            return []
        
    @staticmethod
    def diff_fanart_updates(long, short):
        """
        Unpicks the fanart api lag so we can see which have been updated
        """
        
        long_ids = collections.Counter([x['id'] for x in long])
        short_ids = collections.Counter([x['id'] for x in short])

        long_ids.subtract(short_ids)
        return set(long_ids.elements())

    def build_url(self, mbid):
        """
        Builds query url
        :param mbid: Musicbrainz ID of resource
        :return: URL to query
        """
        scheme = 'https://' if self.use_https else 'http://'
        url = scheme + self._base_url
        if url[-1] != '/':
            url += '/'
        url += mbid
        url += '/?api_key={api_key}'.format(api_key=self._api_key)
        return url

    @staticmethod
    def parse_artist_images(response):
        """
        解析艺术家图片为统一格式
        :param response: API 响应数据
        :return: 包含 small 字段的字典: {'small': url 或 None}
        """
        # 从响应获取缩略图
        thumb = util.first_key_item(response, 'artistthumb')
        url = thumb['url'] if thumb else None
        
        # 只返回 small 字段
        return {
            'small': url,
            'mid': url,
            'large': url,
            'original': url
        }

class SpotifyAuthProvider(HttpProvider,
                          SpotifyAuthMixin):
    
    """
    Provider to handle OAuth redirect from spotify
    """
    def __init__(self,
                 token_url='https://accounts.spotify.com/api/token',
                 redirect_uri='',
                 client_id='',
                 client_secret=''):
        """
        Class initialization
        """
        super(SpotifyAuthProvider, self).__init__('spotify')
        self._token_url = token_url
        self._redirect_uri = redirect_uri
        self._client_id = client_id
        self._client_secret = client_secret

    async def get_token(self, code):

        body = {'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': self._redirect_uri,
                'client_id': self._client_id,
                'client_secret': self._client_secret}

        session = await self._get_session()
        async with session.post(self._token_url, data=body) as resp:
            resp.raise_for_status()
            json = await resp.json()

            access_token = json.get('access_token', '')
            expires_in = json.get('expires_in', '')
            refresh_token = json.get('refresh_token', '')

            return access_token, expires_in, refresh_token
            
    async def refresh_token(self, refresh_token):

        body = {'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self._client_id,
                'client_secret': self._client_secret}

        session = await self._get_session()
        async with session.post(self._token_url, data=body) as resp:
            resp.raise_for_status()
            return await resp.json()
        
class SolrSearchProvider(HttpProvider,
                         ArtistNameSearchMixin,
                         ReleaseNameSearchMixin,
                         RecordingNameSearchMixin):
    
    """
    Provider that uses a solr indexed search
    """
    def __init__(self,
                 search_server=f'http://{CONFIG.MB_DB_HOST}:8983/solr'):
        """
        Class initialization

        Defaults to the offical musicbrainz webservice but in principle we could
        host our own mirror using https://github.com/metabrainz/musicbrainz-docker

        :param search_server: URL for the search server.  Note that using HTTPS adds around 100ms to search time.
        """
        super(SolrSearchProvider, self).__init__('solr_search')

        self._search_server = search_server
        
    async def get_with_limit(self, url, max_retries=0, retry_delay=1.0):
        """
        Performs an HTTP GET request with rate limiting and optional retry mechanism
        
        :param url: URL to request
        :param max_retries: Maximum number of retry attempts (default: 0, no retries)
        :param retry_delay: Delay between retries in seconds (default: 1.0)
        :return: JSON response
        """
        return await super().get_with_limit(url, timeout=aiohttp.ClientTimeout(total=5), max_retries=max_retries, retry_delay=retry_delay)
            
    async def search_artist_name(self, name, limit=None):
        
        # Note that when using a dismax query we shouldn't apply lucene escaping
        # See https://github.com/metabrainz/musicbrainz-server/blob/master/lib/MusicBrainz/Server/Data/WebService.pm
        url = u'{server}/artist/select?wt=mbjson&q={query}'.format(
            server=self._search_server,
            query=url_quote(name.encode('utf-8'))
        )
        
        if limit:
            url += u'&rows={}'.format(limit)
        
        response = await self.get(url, max_retries=3)
        
        if not response:
            return {}
        
        return self.parse_artist_search(response)

    async def search_release_name(self, name, limit=None, artist_name=''):
        """
        Searches for release with name
        :param name: Name of release
        :param limit: Limit of number of results to return. Defaults to None, indicating no limit
        :param artist_name: Artist name restriction
        :return: List of releases
        """
        if artist_name:
            query = f"({self.escape_lucene_query(name)}) AND (artist:{self.escape_lucene_query(artist_name)} OR artistname:{self.escape_lucene_query(artist_name)} OR creditname:{self.escape_lucene_query(artist_name)})"
            url = u'{server}/release/advanced?wt=mbjson&q={query}'.format(
                server=self._search_server,
                query=url_quote(query.encode('utf-8'))
            )
        else:
            url = u'{server}/release/select?wt=mbjson&q={query}'.format(
                server=self._search_server,
                query=url_quote(name.encode('utf-8'))
            )
        
        if limit:
            url += u'&rows={}'.format(limit)
        logger.debug(f'search_release_name, url = {url}')

        response = await self.get(url, max_retries=3)
        
        if not response:
            return {}
        
        return response
    
    async def search_recording_name(self, name, limit=None, artist_name=''):
        """
        搜索录音作品
        :param name: 录音作品名称
        :param limit: 返回结果数量限制,默认为 None 表示无限制
        :param artist_name: 艺术家名称限制
        :return: 录音作品列表
        """
        if artist_name:
            query = f"({self.escape_lucene_query(name)}) AND (artist:{self.escape_lucene_query(artist_name)} OR artistname:{self.escape_lucene_query(artist_name)} OR creditname:{self.escape_lucene_query(artist_name)})"
            url = u'{server}/recording/advanced?wt=mbjson&q={query}'.format(
                server=self._search_server,
                query=url_quote(query.encode('utf-8'))
            )
        else:
            url = u'{server}/recording/select?wt=mbjson&q={query}'.format(
                server=self._search_server,
                query=url_quote(name.encode('utf-8'))
            )
        
        if limit:
            url += u'&rows={}'.format(limit)
            
        logger.debug(f'search_recording_name, url = {url}')
        
        # Use the retry mechanism with 3 retries and 1 second delay
        response = await self.get(url, max_retries=3, retry_delay=1.0)
        
        if not response:
            return {}
        
        return response

    @staticmethod
    def escape_lucene_query(text):
        return re.sub(r'([+\-&|!(){}\[\]\^"~*?:\\/])', r'\\\1', text)
        
    @staticmethod
    def parse_artist_search(response):
        if not response:
            return []
        return response['artists']
    
    # @staticmethod
    # def parse_album_search(response):
        
    #     if not 'count' in response or response['count'] == 0:
    #         return []
        
    #     result = [{'Id': result['id'],
    #                'Title': result['title'],
    #                'Type': result['primary-type'] if 'primary-type' in result else 'Unknown',
    #                'Score': result['score']}
    #             for result in response['release-groups']]

    #     return result
    
class MusicbrainzDbProvider(Provider,
                            DataVintageMixin,
                            InvalidateCacheMixin,
                            ArtistIdListMixin,
                            ArtistByIdMixin,
                            ReleaseGroupByArtistMixin,
                            ReleaseGroupByIdMixin,
                            ReleaseGroupIdListMixin,
                            ReleaseByIdMixin,
                            TrackByIdMixin,
                            MusicBrainzCoverArtMixin,
                            SeriesMixin):
    """
    Provider for directly querying musicbrainz database
    """

    def __init__(self,
                 db_host=CONFIG.MB_DB_HOST,
                 db_port=5432,
                 db_name='musicbrainz_db',
                 db_user='musicbrainz',
                 db_password='musicbrainz'):
        """
        Class initialization

        Note that these defaults are reasonable if the linuxserverio/musicbrainz
        docker image is running locally with port 5432 exposed.

        :param db_host: Host of musicbrainz db. Defaults to localhost
        :param db_port: Port of musicbrainz db. Defaults to 5432
        :param db_name: Name of musicbrainz db. Defaults to musicbrainz_db
        :param db_user: User for musicbrainz db. Defaults to abc
        :param db_password: Password for musicbrainz db. Defaults to abc
        """
        super(MusicbrainzDbProvider, self).__init__()

        self._db_host = db_host
        self._db_port = db_port
        self._db_name = db_name
        self._db_user = db_user
        self._db_password = db_password
        self._pool = None
        self.__pool_lock = None
        
        ## dummy value for initialization, will be picked up from redis later on
        self._last_cache_invalidation = datetime.datetime.now(pytz.utc) - datetime.timedelta(hours = 2)
        
    async def uuid_as_str(self, con):
        await con.set_type_codec(
            'uuid', encoder=str, decoder=str,
            schema='pg_catalog', format='text'
        )
        
    @property
    def _pool_lock(self):
        if self.__pool_lock is None:
            self.__pool_lock = asyncio.Lock()
        return self.__pool_lock
    
    async def _get_pool(self):
        async with self._pool_lock:
            if self._pool is None:
                logger.debug("Initializing MB DB pool")
                
                start = timer()
                # Initialize pool
                self._pool = await asyncpg.create_pool(host = self._db_host,
                                                       port = self._db_port,
                                                       user = self._db_user,
                                                       password = self._db_password,
                                                       database = self._db_name,
                                                       init = self.uuid_as_str,
                                                       statement_cache_size=0)
                elapsed = int((timer() - start) * 1000)
                logger.info(f"MB DB pool initialized in {elapsed}ms")
                
            return self._pool
        
    async def data_vintage(self):
        data = await self.query_from_file('data_vintage.sql')
        return data[0]['vintage']
    
    async def invalidate_cache(self, prefix, since):

        last_invalidation_key = prefix + 'MBProviderLastCacheInvalidation'
        self._last_cache_invalidation = since or await util.CACHE.get(last_invalidation_key) or self._last_cache_invalidation

        result = {'artists': [], 'albums': [], 'spotify_artists': [], 'spotify_albums': []}
        
        vintage = await self.data_vintage()
        if vintage > self._last_cache_invalidation:
            logger.debug('Invalidating musicbrainz cache')

            result['artists'] = await self._invalidate_queries_by_entity_id('updated_artists.sql')
            result['albums'] = await self._invalidate_queries_by_entity_id('updated_albums.sql')
            result['spotify_artists'] = await self._invalidate_spotify_ids('updated_spotify_artists.sql')
            result['spotify_albums'] = await self._invalidate_spotify_ids('updated_spotify_albums.sql')
            
            logger.info('Invalidating these artists given musicbrainz updates:\n{}'.format('\n'.join(result['artists'])))
            logger.info('Invalidating these albums given musicbrainz updates:\n{}'.format('\n'.join(result['albums'])))
            logger.info('Invalidating these spotify artists given musicbrainz updates:\n{}'.format('\n'.join(result['spotify_artists'])))
            logger.info('Invalidating these spotify albums given musicbrainz updates:\n{}'.format('\n'.join(result['spotify_albums'])))

            await util.CACHE.set(last_invalidation_key, vintage)
        else:
            logger.debug('Musicbrainz invalidation not required')
            
        return result
    
    async def _invalidate_queries_by_entity_id(self, changed_query):
        entities = await self.query_from_file(changed_query, self._last_cache_invalidation)
        return [entity['gid'] for entity in entities]

    async def _invalidate_spotify_ids(self, changed_query):
        entities = await self.query_from_file(changed_query, self._last_cache_invalidation)
        return [entity['spotifyid'] for entity in entities]
    
    async def get_artists_by_id(self, artist_ids):
        artists = await self.query_from_file('artist_by_id.sql', artist_ids)
        
        logger.debug("got artists")
        
        if not artists:
            return None
        
        artists = [self._load_artist(item['artist']) for item in artists]

        return artists
        
    async def redirect_old_artist_id(self, artist_id):
        results = await self.query_from_file('artist_redirect.sql', artist_id)
        if results:
            return results[0]['gid']
        return None

    async def get_artist_id_from_spotify_id(self, spotify_id):
        results = await self.query_from_file('artist_id_from_spotify_id.sql', spotify_id)
        if results:
            return results[0]['gid']
        return None

    async def get_spotify_mappings(self, limit=100, offset=0):
        """
        获取Spotify ID与MusicBrainz ID的映射关系，支持分页
        
        Args:
            limit (int): 每页返回的最大记录数量，默认100条
            offset (int): 分页偏移量，默认从0开始
            
        Returns:
            A list of mappings with mbid and spotifyid for the requested page
        """
        results = await self.query_from_file('artist_spotify_maps_paged.sql', limit, offset)
        return results
    
    @deprecated('Use get_artist_ids_paged instead.')
    async def get_all_artist_ids(self):
        results = await self.query_from_file('all_artist_ids.sql')
        return [item['gid'] for item in results]
    
    async def get_artist_ids_paged(self, limit=1000, offset=0):
        """
        实现分页获取艺术家 ID
        """
        results = await self.query_from_file('artist_ids_paged.sql', limit, offset)
        return [item['gid'] for item in results]
    
    @classmethod
    def _load_artist(cls, data):
        # Load the json from postgres
        artist = json.loads(data)
        
        # parse the links
        artist['links'] = [{
            'target': link,
            'type': cls.parse_url_source(link)
        } for link in artist['links']]
        
        return artist

    @classmethod
    def _load_release_group(cls, data):
        # Load the json from postgres
        release_group = json.loads(data)
        
        # parse the links
        release_group['links'] = [{
            'target': link,
            'type': cls.parse_url_source(link)
        } for link in release_group['links']]
        
        # parse caa images
        if release_group['images']:
            type_mapping = {'Front': 'Cover', 'Medium': 'Disc'}

            art = {}
            for result in release_group['images']:
                cover_type = type_mapping.get(result['type'], None)
                if cover_type is not None and cover_type not in art:
                    art[cover_type] = CoverArtArchiveProvider.build_caa_url(result['release_gid'], result['image_id'])
            release_group['images'] = [{'CoverType': art_type, 'Url': url} for art_type, url in art.items()]
        else:
            release_group['images'] = []
            
        return release_group

    async def get_release_groups_by_id(self, rgids):
        release_groups = await self.query_from_file('release_group_by_id.sql', rgids)
        
        logger.debug("got release groups")
        
        if not release_groups:
            return None
        
        release_groups = [self._load_release_group(item['album']) for item in release_groups]

        return release_groups
    
    async def get_release_by_id(self, rids):
        releases = await self.query_from_file('release_by_id.sql', rids)
        
        logger.debug("got releases: ")
        
        if not releases:
            return None
        
        results = []
        for release_data in releases:
            release = json.loads(release_data['release'])    
            # 处理 release group 信息
            if release.get('releasegroup'):
                release_group = release['releasegroup']
                release['type'] = release_group['Type']
                # 提取 wiki 相关链接
                if release_group.get('Links'):
                    wiki_links = [
                        link for link in release_group['Links'] 
                        if any(wiki in link['type'].lower() for wiki in ['wikidata', 'wikipedia'])
                    ]
                    if wiki_links:
                        release['wiki_links'] = wiki_links
                del release['releasegroup']
            results.append(release)
        return results
    
    async def get_track_by_id(self, track_ids):
        """
        获取曲目信息
        """
        tracks = await self.query_from_file('track_by_id.sql', track_ids)
        
        if not tracks:
            return None
            
        results = []
        for track_data in tracks:
            track = json.loads(track_data['track'])
            results.append(track)
            
        return results

    async def get_release_groups_by_recording_ids(self, rids):
        results = await self.query_from_file('release_group_by_recording_ids.sql', len(rids), rids)

        return [item['rgid'] for item in results]
    
    async def redirect_old_release_group_id(self, id):
        results = await self.query_from_file('release_group_redirect.sql', id)
        if results:
            return results[0]['gid']
        return None

    async def get_release_group_id_from_spotify_id(self, spotify_id):
        results = await self.query_from_file('release_group_id_from_spotify_id.sql', spotify_id)
        if results:
            return results[0]['gid']
        return None
    
    async def get_all_release_group_ids(self):
        results = await self.query_from_file('all_release_group_ids.sql')
        return [item['gid'] for item in results]

    async def get_release_groups_by_artist(self, artist_id):
        results = await self.query_from_file('release_group_search_artist_mbid.sql', artist_id)
        
        logger.debug("got artist release groups")

        if not results or not results[0]['result']:
            return {}
            
        return json.loads(results[0]['result'])

    async def get_series(self, mbid):
        series = await self.query_from_file('release_group_series.sql', mbid)
        return [json.loads(x['item']) for x in series]

    async def get_release_cover_art(self, release_ids):
        """
        获取一个或多个发行版(Release)的封面信息
        
        :param release_ids: 单个 MusicBrainz Release ID 或 ID 列表
        :return: 如果传入单个 ID，返回单个封面信息；如果传入 ID 列表，返回字典 {release_id: cover_art_data}
            cover_art_data 包含:
            {
                'id': 封面图片ID,
                'type': 封面类型(front/back/medium等),
                'approved': 是否已审核通过,
                'edit': 最后编辑ID,
                'comment': 备注信息,
                'uploaded': 上传时间
            }
        """
        results = await self.query_from_file('release_cover_art.sql', release_ids)
        
        if not results:
            return None
            
        # 将结果转换为字典，每个 release_id 对应其最新的封面
        result_dict = {}
        for row in results:
            release_id = row['release_id']
            # 由于结果按 release_id, date_uploaded DESC 排序
            # 所以每个 release_id 的第一条记录就是最新的封面
            if release_id not in result_dict:
                result_dict[release_id] = {
                    'id': row['id'],
                    'uploaded': row['uploaded']
                }
            
        return result_dict

    async def query_from_file(self, sql_file, *args):
        """
        Executes query from sql file
        :param sql_file: Filename of sql file
        :param args: Positional args to pass to cursor.execute
        :param kwargs: Keyword args to pass to cursor.execute
        :return: List of dict with column: value results
        """
        filename = pkg_resources.resource_filename('lidarrmetadata.sql', sql_file)

        with open(filename, 'r') as sql:
            return await self.map_query(sql.read(), *args)

    @conn
    async def map_query(self, sql, *args, _conn=None):
        """
        Maps a SQL query to a list of dicts of column name: value
        :param args: Args to pass to cursor.execute
        :param kwargs: Keyword args to pass to cursor.execute
        :return: List of dict with column: value
        """

        data = await _conn.fetch(sql, *args, timeout=120)
            
        results = [dict(row.items()) for row in data]

        return results

    @staticmethod
    def parse_url_source(url):
        """
        Parses URL for name
        :param url: URL to parse
        :return: Website name of url
        """
        domain = url.split('/')[2]
        split_domain = domain.split('.')
        try:
            return split_domain[-2] if split_domain[-2] != 'co' else split_domain[-3]
        except IndexError:
            return domain

class SpotifyProvider(HttpProvider, ArtistArtworkMixin):
    """
    Provider to get details for a spotify id
    
    使用直接HTTP请求而非spotipy库与Spotify API交互
    实现了自动获取和刷新访问令牌的功能
    """

    def __init__(self, client_id, client_secret):
        """
        初始化Spotify提供程序
        
        Args:
            client_id: Spotify应用客户端ID
            client_secret: Spotify应用客户端密钥
        """
        super(SpotifyProvider, self).__init__('spotify')
        
        self._client_id = client_id
        self._client_secret = client_secret
        self._token = None
        self._token_expires = 0
        self._token_lock = asyncio.Lock()
        self._api_base_url = "https://api.spotify.com/v1"
        
    async def _ensure_token(self):
        """
        确保有一个有效的访问令牌，如果令牌过期则刷新
        
        Returns:
            str: 有效的访问令牌
        """
        now = datetime.datetime.now().timestamp()
        
        # 使用锁确保只有一个协程可以刷新令牌
        async with self._token_lock:
            # 检查令牌是否已过期或即将过期(留5分钟的缓冲时间)
            if not self._token or now > (self._token_expires - 300):
                logger.debug("获取新的Spotify访问令牌")
                await self._get_new_token()
                
        return self._token
    
    async def _get_new_token(self):
        """
        从Spotify获取新的访问令牌
        """
        auth = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        try:
            session = await self._get_session()
            async with session.post("https://accounts.spotify.com/api/token", 
                                    headers=headers, 
                                    data=data) as response:
                
                response.raise_for_status()
                token_data = await response.json()
                
                self._token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)  # 默认1小时
                
                # 设置过期时间
                self._token_expires = datetime.datetime.now().timestamp() + expires_in
                logger.debug(f"获取到新的Spotify令牌，将在{expires_in}秒后过期")
                
        except Exception as e:
            logger.error(f"获取Spotify令牌失败: {str(e)}")
            raise ProviderUnavailableException(f"Failed to get Spotify token: {str(e)}")
    
    async def _api_request(self, endpoint, params=None, retries=2):
        """
        向Spotify API发送请求
        
        Args:
            endpoint: API端点路径（不含基础URL）
            params: 请求参数字典
            retries: 重试次数
            
        Returns:
            dict: API响应数据
        """
        url = f"{self._api_base_url}/{endpoint}"
        
        for attempt in range(retries + 1):
            # 获取令牌
            token = await self._ensure_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            try:
                session = await self._get_session()
                async with session.get(url, headers=headers, params=params) as response:
                    # 处理401错误（令牌失效）
                    if response.status == 401 and attempt < retries:
                        logger.debug("Spotify令牌过期，正在刷新")
                        async with self._token_lock:
                            self._token_expires = 0  # 强制刷新令牌
                            await self._ensure_token()
                        continue  # 重试请求
                        
                    # 处理429错误（限流）
                    if response.status == 429 and attempt < retries:
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        logger.warning(f"Spotify API限流，等待 {retry_after} 秒后重试")
                        await asyncio.sleep(retry_after)
                        continue  # 重试请求
                    
                    # 其他错误状态码
                    if not response.ok:
                        response.raise_for_status()  # 这会引发ClientResponseError异常
                        
                    # 正常情况，返回JSON响应
                    return await response.json()
                        
            except aiohttp.ClientResponseError as e:
                if attempt < retries:
                    logger.warning(f"Spotify API响应错误，尝试重试 ({attempt+1}/{retries}): {str(e)}")
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Spotify API响应错误: {str(e)}")
                raise ProviderUnavailableException(f"Spotify API response error: {str(e)}")
                
            except aiohttp.ClientError as e:
                if attempt < retries:
                    logger.warning(f"Spotify API连接错误，尝试重试 ({attempt+1}/{retries}): {str(e)}")
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Spotify API连接错误: {str(e)}")
                raise ProviderUnavailableException(f"Spotify API connection error: {str(e)}")
                
            except Exception as e:
                logger.error(f"Spotify API请求未预期错误: {str(e)}")
                raise ProviderUnavailableException(f"Spotify API unexpected error: {str(e)}")
                
        # 如果达到这里，说明所有重试都失败了
        raise ProviderUnavailableException(f"Spotify API request failed after {retries+1} attempts")
    
    async def get_artist_images(self, artist_id):
        """
        获取艺术家的图片信息
        
        从Spotify API获取艺术家图片，并按相对大小分类
        
        Args:
            artist_id (str): Spotify艺术家ID
            
        Returns:
            图片字典包含不同尺寸的图片URL:
            {
                'small': <最小分辨率图片的url>,
                'mid': <中等分辨率图片的url>,
                'large': <最大分辨率图片的url>
            }
        """
        try:
            logger.debug(f"请求Spotify艺术家图片: {artist_id}")
            start_time = timer()
            
            # 获取艺术家数据
            artist_data = await self._api_request(f"artists/{artist_id}")
            
            logger.debug(f"Spotify API请求完成，耗时: {timer() - start_time:.2f}秒")
            
            if not artist_data or 'images' not in artist_data or not artist_data['images']:
                logger.debug(f"未找到艺术家图片: {artist_id}")
                return {}, utcnow() + timedelta(seconds=CONFIG.CACHE_TTL['provider_error'])
            
            # 获取所有图片并按尺寸排序
            images_list = sorted(
                artist_data['images'], 
                key=lambda img: img.get('height', 0) * img.get('width', 0),  # 按面积排序
                reverse=True  # 从大到小排序
            )
            
            result = {}
            
            # 如果有图片，将最大的作为large
            if len(images_list) > 0:
                result['large'] = images_list[0]['url']
            
            # 如果有至少2张图片，将中间的作为mid
            if len(images_list) > 2:
                result['mid'] = images_list[len(images_list) // 2]['url']
            elif len(images_list) > 1:
                result['mid'] = images_list[1]['url']
            elif 'large' in result:
                # 如果只有一张图片，将large作为mid
                result['mid'] = result['large']
            
            # 如果有至少3张图片，将最小的作为small
            if len(images_list) >= 3:
                result['small'] = images_list[-1]['url']
            elif len(images_list) > 1:
                # 如果只有两张图片，将较小的作为small
                result['small'] = images_list[-1]['url']
            elif 'mid' in result:
                # 如果只有一张图片，将mid作为small
                result['small'] = result['mid']
            
            cache_ttl = CONFIG.CACHE_TTL.get('spotify', 60 * 60 * 24)  # 默认1天
            expiry = utcnow() + timedelta(seconds=cache_ttl)
            
            return result, expiry
            
        except Exception as e:
            logger.error(f"获取Spotify艺术家图片失败: {artist_id}, 错误: {str(e)}")
            error_ttl = CONFIG.CACHE_TTL.get('provider_error', 60 * 10)  # 默认10分钟
            return {}, utcnow() + timedelta(seconds=error_ttl)


class WikipediaProvider(HttpProvider, ArtistOverviewMixin):
    """
    Provider for querying wikipedia
    """

    WIKIPEDIA_REGEX = re.compile(r'https?://(?:(?P<language>\w+)\.)?wikipedia\.org/wiki/(?P<title>.+)')
    WIKIDATA_REGEX = re.compile(r'https?://www.wikidata.org/(wiki|entity)/(?P<entity>.+)')

    def __init__(self, session=None, limiter=None):
        """
        Class initialization
        """
        super(WikipediaProvider, self).__init__('wikipedia', session, limiter)

        # https://github.com/metabrainz/musicbrainz-server/blob/v-2019-05-13-schema-change/lib/MusicBrainz/Server/Data/WikipediaExtract.pm#L61
        self.language_preference = (
            'en', 'ja', 'de', 'fr', 'fi', 'it', 'sv', 'es', 'ru', 'pl',
            'nl', 'pt', 'et', 'da', 'ko', 'ca', 'cs', 'cy', 'el', 'he',
            'hu', 'id', 'lt', 'lv', 'no', 'ro', 'sk', 'sl', 'tr', 'uk',
            'vi', 'zh'
        )
        
    async def get_artist_overview(self, url, ignore_cache=False):
        
        if not ignore_cache:
            cached, expires = await util.WIKI_CACHE.get(url) or (None, True)

            if cached and expires > utcnow():
                return cached, expires
        else:
            cached = None
        
        logger.debug(f"getting overview, url = {url}")
        
        try:
            summary = await self.wikidata_get_summary_from_url(url) if 'wikidata' in url else await self.wikipedia_get_summary_from_url(url)
            ttl = CONFIG.CACHE_TTL['wikipedia']
            await util.WIKI_CACHE.set(url, summary, ttl=ttl)
            return summary, utcnow() + timedelta(seconds = ttl)
        
        except ProviderUnavailableException:
            return (cached or '', utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['provider_error']))

        except ValueError as error:
            logger.error(f'Could not get summary', extra = dict(url=url, error = repr(error)))
            return '', utcnow() + timedelta(seconds = CONFIG.CACHE_TTL['provider_error'])
            
    async def wikidata_get_summary_from_url(self, url):
        data = await self.wikidata_get_entity_data_from_url(url)
        return await self.wikidata_get_summary_from_entity_data(data)
            
    async def wikidata_get_summary_from_entity_data(self, data):
        
        sites = { item['site']: url_quote(item['title'].encode('utf-8')) for item in data.get('sitelinks', {}).values() }

        # return english wiki if possible
        if 'enwiki' in sites:
            return await self.wikipedia_get_summary_from_title(sites['enwiki'], 'en')
        
        # if not, return english entity description
        description = data.get('descriptions', {}).get('en', {}).get('value', '')
        if description:
            return description
        
        # otherwise fall back to most common language available
        language = next((x for x in self.language_preference if sites.get('{}wiki'.format(x), '')), None)
        
        if language:
            title = sites['{}wiki'.format(language)]
            return await self.wikipedia_get_summary_from_title(title, language)
        return ''
    
    async def wikidata_get_entity_data_from_url(self, url):
        entity = self.wikidata_entity_from_url(url)
        wikidata_url = (
            'https://www.wikidata.org/w/api.php'
            '?action=wbgetentities'
            '&ids={}'
            '&props=sitelinks|descriptions'
            '&format=json'
        ).format(entity)
        logger.debug(f"getting summary from wikidata, url = {wikidata_url}")
        
        data = await self.get_with_limit(wikidata_url)
        return (
            data
            .get('entities', {})
            .get(entity, {})
        )
    
    async def wikidata_get_entity_data_from_language_title(self, title, language):
        title = title.split("#", 1)[0]
        wikidata_url = (
            'https://www.wikidata.org/w/api.php'
            '?action=wbgetentities'
            '&sites={language}wiki'
            '&titles={title}'
            '&props=sitelinks|descriptions'
            '&format=json'
        ).format(language=language, title=title)
        logger.debug(f"getting summary from wikidata, url = {wikidata_url}")
        data = await self.get_with_limit(wikidata_url)
        entities = data.get('entities', {})
        return entities[next(iter(entities))]
    
    async def wikipedia_get_summary_from_url(self, url):
        url_title, url_language = self.wikipedia_title_from_url(url)
        
        # if English link, just use that
        if url_language == 'en':
            return await self.wikipedia_get_summary_from_title(url_title, url_language)
        
        # Otherwise go via wikidata to try to get something in English or best other language
        data = await self.wikidata_get_entity_data_from_language_title(url_title, url_language)
        return await self.wikidata_get_summary_from_entity_data(data)
        
    async def wikipedia_get_summary_from_title(self, title, language):
        """
        Gets summary of a wikipedia page
        :param url: URL of wikipedia page
        :return: Summary String
        """
        
        wiki_url = (
            'https://{language}.wikipedia.org/w/api.php'
            '?action=query'
            '&prop=extracts'
            '&exintro'
            '&explaintext'
            '&format=json'
            '&formatversion=2'
            '&titles={title}'
        ).format(language = language, title = title)
        logger.debug(f"getting summary from wikipedia, url = {wiki_url}")
        
        data = await self.get_with_limit(wiki_url)
        return data.get('query', {}).get('pages', [{}])[0].get('extract', '')

    @classmethod
    def wikipedia_title_from_url(cls, url):
        """
        Gets the wikipedia page title from url. This may not work for URLs with
        certain special characters
        :param url: URL of wikipedia page
        :return: Title of page at URL
        """
        match = cls.WIKIPEDIA_REGEX.match(url)

        if not match:
            raise ValueError(f'URL {url} does not match regex `{cls.WIKIPEDIA_REGEX.pattern}`')

        title = match.group('title')
        language = match.group('language')
        if language is None or language == 'www':
            language = 'en'
        
        return title, language

    @classmethod
    def wikidata_entity_from_url(cls, url):
        """
        Gets the wikidata entity id from the url. This may not work for URLs with
        certain special characters
        :param url: URL of wikidata page
        :return: Entity referred to
        """
        match = cls.WIKIDATA_REGEX.match(url)

        if not match:
            raise ValueError(u'URL {} does not match regex `{}`'.format(url, cls.WIKIDATA_REGEX.pattern))

        id = match.group('entity')
        return id
