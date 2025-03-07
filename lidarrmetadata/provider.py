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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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
    def get_all_spotify_mappings(self):
        """
        Grabs all link entities from database and parses for spotify maps
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


class AlbumArtworkMixin(MixinBase):
    """
    Gets art for album
    """

    @abc.abstractmethod
    def get_album_images(self, album_id):
        """
        Gets images for album with ID
        :param album_id: ID of album
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

class SpotifyIdMixin(MixinBase):
    """
    Details for a spotify id
    """
    
    @abc.abstractmethod
    def album_from_artist(self, artist_id):
        pass

    @abc.abstractmethod
    def album(self, album_id):
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
        
class ProviderUnavailableException(Exception):
    """ Thown on error for providers we can cope without """
    pass

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

    async def get(self, url, raise_on_http_error=True, **kwargs):
        try:
            self._count_request('request')
            start = timer()
            session = await self._get_session()
            async with session.get(url, **kwargs) as resp:
                end = timer()
                elapsed = int((end - start) * 1000)
                logger.debug(f"Got response [{resp.status}] for URL: {url} in {elapsed}ms ")
                self._record_response_result(resp, elapsed)

                if raise_on_http_error:
                    resp.raise_for_status()

                json = await resp.json()
                return json
        except ValueError as error:
            logger.error(f'Response from {self._name} not valid json', extra=dict(error=error))
            raise
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as error:
            logger.error(f'aiohttp exception {getattr(error, "status", None)}',
                         extra = dict(error_message=getattr(error, "message", None), error=repr(error)))
            raise ProviderUnavailableException(f'{self._name} aiohttp exception')
        except asyncio.CancelledError:
            logger.debug(f'Task cancelled {url}')
            raise
        except asyncio.TimeoutError:
            logger.debug(f'Timeout for {self._name}', extra=dict(url=url))
            self._count_request('timeout')
            raise ProviderUnavailableException(f'{self._name} timeout')
        except Exception as error:
            logger.error(f'Non-aiohttp exceptions occured: {getattr(error, "__dict__", {})}', extra=dict(error = repr(error)))
            raise
        
    async def get_with_limit(self, url, raise_on_http_error=True, **kwargs):
        try:
            with self._limiter.limited():
                return await self.get(url, raise_on_http_error, **kwargs)
        except limit.RateLimitedError:
            logger.debug(f'{self._name} request rate limited')
            self._count_request('ratelimit')

class CoverArtArchiveProvider(Provider, ReleaseArtworkMixin):
    """
    从 Cover Art Archive 获取专辑封面的 Provider
    """
    def __init__(self):
        """
        初始化 Provider
        """
        super(CoverArtArchiveProvider, self).__init__()
        self._db_provider = get_providers_implementing(MusicBrainzCoverArtMixin)[0]
        self._base_url = 'http://coverartarchive.org/release/'
        
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
        
        # 先检查缓存
        cached_results = await asyncio.gather(*[util.RELEASE_IMAGE_CACHE.get(rid) for rid in release_ids])
        results = []
        uncached_ids = []
        uncached_indices = {}  # 使用字典来保存 release_id 到结果索引的映射
        
        # 处理缓存结果
        for i, (cached, expires) in enumerate(cached_results):
            if cached and expires > now:
                results.append((cached, expires))
            else:
                results.append(None)
                rid = release_ids[i]
                uncached_ids.append(rid)
                uncached_indices[rid] = i
                
        if not uncached_ids:
            return results
            
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
                    logger.debug(f'No cover art data found for {release_id}')
                    results[result_index] = (None, now)
                    continue
                    
                cover_art_data = cover_art_data_dict[release_id]
                if not cover_art_data or 'id' not in cover_art_data:
                    logger.debug(f'Invalid cover art data for {release_id}')
                    results[result_index] = (None, now)
                    continue
                    
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
                
            return results
            
        except ProviderUnavailableException:
            # 如果服务不可用，对所有未缓存的结果返回空列表
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
        base_url = f'http://imagecache.lidarr.audio/v1/caa/{release_id}/{image_id}'
        return f'{base_url}-{size}.jpg' if size else f'{base_url}.jpg'

class FanArtTvProvider(HttpProvider, 
                       AlbumArtworkMixin, 
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
        
    async def get_album_images(self, album_id):
        
        return await self.get_images(album_id, self.parse_album_images)
        
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
    def parse_album_images(response):
        """
        Parses album images to our expected format
        :param response: API response
        :return: List of images in our expected format
        """
        images = {'Cover': util.first_key_item(response, 'albumcover'),
                  'Disc': util.first_key_item(response, 'cdart')}
        return [{'CoverType': key, 'Url': value['url'].replace('https', 'http')}
                for key, value in images.items() if value]

    @staticmethod
    def parse_artist_images(response):
        """
        Parses artist images to our expected format
        :param response: API response
        :return: List of images in our expected format
        """
        images = {'Banner': util.first_key_item(response, 'musicbanner'),
                  'Fanart': util.first_key_item(response, 'artistbackground'),
                  'Logo': util.first_key_item(response, 'hdmusiclogo'),
                  'Poster': util.first_key_item(response, 'artistthumb')}
        return [{'CoverType': key, 'Url': value['url'].replace('https', 'http')}
                for key, value in images.items() if value]

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
                         AlbumNameSearchMixin,
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
        
    async def get_with_limit(self, url):
        return await super().get_with_limit(url, timeout=aiohttp.ClientTimeout(total=5))
            
    async def search_artist_name(self, name, limit=None):
        
        # Note that when using a dismax query we shouldn't apply lucene escaping
        # See https://github.com/metabrainz/musicbrainz-server/blob/master/lib/MusicBrainz/Server/Data/WebService.pm
        url = u'{server}/artist/select?wt=mbjson&q={query}'.format(
            server=self._search_server,
            query=url_quote(name.encode('utf-8'))
        )
        
        if limit:
            url += u'&rows={}'.format(limit)
        
        response = await self.get(url)
        
        if not response:
            return {}
        
        return self.parse_artist_search(response)
    
    async def search_albums_with_artist(self, artist, albums, handler, limit=None):
        
        album_query = u" ".join(albums)
        query = u"({album_query}) AND (artist:{artist} OR artistname:{artist} OR creditname:{artist})".format(
            album_query=url_quote(self.escape_lucene_query(album_query).encode('utf-8')),
            artist=url_quote(self.escape_lucene_query(artist).encode('utf-8'))
        )
        
        url = u'{server}/release-group/advanced?wt=mbjson&q={query}'.format(
            server=self._search_server,
            query=query
        )
        
        if limit:
            url += u'&rows={}'.format(limit)
            
        response = await self.get_with_limit(url)
        
        if not response:
            return {}

        return handler(response)
    
    async def search_album_name(self, name, limit=None, artist_name=''):
        
        if artist_name:
            return await self.search_albums_with_artist(artist_name, [name], self.parse_album_search, limit)

        # Note that when using a dismax query we shouldn't apply lucene escaping
        # See https://github.com/metabrainz/musicbrainz-server/blob/master/lib/MusicBrainz/Server/Data/WebService.pm
        url = u'{server}/release-group/select?wt=mbjson&q={query}'.format(
            server=self._search_server,
            query=url_quote(name.encode('utf-8'))
        )
        
        if limit:
            url += u'&rows={}'.format(limit)
        
        response = await self.get_with_limit(url)
        
        if not response:
            return {}
        
        return self.parse_album_search(response)

    async def search_release_name(self, name, limit=None, artist_name=''):
        url = u'{server}/release/select?wt=mbjson&q={query}'.format(
            server=self._search_server,
            query=url_quote(name.encode('utf-8'))
        )
        
        if limit:
            url += u'&rows={}'.format(limit)
        logger.debug(f'search_release_name, url = {url}')

        response = await self.get(url)
        
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
        
        response = await self.get(url)
        
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
    
    @staticmethod
    def parse_album_search(response):
        
        if not 'count' in response or response['count'] == 0:
            return []
        
        result = [{'Id': result['id'],
                   'Title': result['title'],
                   'Type': result['primary-type'] if 'primary-type' in result else 'Unknown',
                   'Score': result['score']}
                for result in response['release-groups']]

        return result
    
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
                
                # Initialize pool
                self._pool = await asyncpg.create_pool(host = self._db_host,
                                                       port = self._db_port,
                                                       user = self._db_user,
                                                       password = self._db_password,
                                                       database = self._db_name,
                                                       init = self.uuid_as_str,
                                                       statement_cache_size=0)
                
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

    async def get_all_spotify_mappings(self):
        return await self.query_from_file('all_spotify_maps.sql')
        # return results
        # return [{'mbid': item['gid'], 'spotifyid': item['spotifyid']} for item in results]
    
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

class SpotifyProvider(Provider,
                      SpotifyIdMixin):
    """
    Provider to get details for a spotify id
    """

    def __init__(self, client_id, client_secret):
        super(SpotifyProvider, self).__init__()
        
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def album_from_artist(self, artist_id):
        top_tracks = self.spotify.artist_top_tracks(artist_id, country='US')

        if not top_tracks['tracks']:
            return None

        album = top_tracks['tracks'][0]['album']
        artist = album['artists'][0]

        return {'Artist': artist['name'],
                'ArtistSpotifyId': artist_id,
                'Album': album['name'],
                'AlbumSpotifyId': album['id']}

    def album(self, album_id):
        album = self.spotify.album(album_id)
        artist = album['artists'][0]
        
        return {'Artist': artist['name'],
                'ArtistSpotifyId': artist['id'],
                'Album': album['name'],
                'AlbumSpotifyId': album_id}
        
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
