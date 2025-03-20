"""
Utility functionality that isn't specific to a given module
"""

import abc
import time
import logging

import warnings
import functools
import inspect
import redis
from aiocache import caches
from aiocache.backends.redis import RedisCache

from lidarrmetadata import config
from lidarrmetadata import cache

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info('Have util logger')


CONFIG = config.get_config()
if CONFIG.USE_CACHE:
    logger.debug('using cache')
    caches.set_config(CONFIG.CACHE_CONFIG)
else:
    logger.debug('null cache')
    caches.set_config(CONFIG.NULL_CACHE_CONFIG)

# Cache for application
CACHE = caches.get('default')
FANART_CACHE = caches.get('fanart')
WIKI_CACHE = caches.get('wikipedia')
ARTIST_CACHE = caches.get('artist')
SPOTIFY_CACHE = caches.get('spotify')
# New
RELEASE_IMAGE_CACHE = caches.get('release_image')
RELEASE_CACHE = caches.get('release')
TRACK_CACHE = caches.get('track')
ARTIST_IMAGE_CACHE = caches.get('artist_image')

def first_key_item(dictionary, key, default=None):
    """
    Gets the first item from a dictionary key that returns a list
    :param dictionary: Dictionary to get item from
    :param key: Key to get
    :param default: Default value to use
    :return: First item or default
    """
    if not dictionary:
        return default
    
    value = dictionary.get(key, default)

    if value and value != default and hasattr(value, '__getitem__'):
        return value[0]

    return value


class SentryProcessor(object):
    @abc.abstractmethod
    def _allowed(self):
        raise NotImplementedError()

    def create_event(self, event, hint):
        return event if self._allowed() else None

class SentryTtlProcessor(SentryProcessor):
    def __init__(self, ttl=1):
        """
        :param ttl: TTL in seconds
        """
        self._allowed_time = None
        self.ttl = ttl

    def _allowed(self):
        current_time = time.time()
        if self._allowed_time is None or current_time > self._allowed_time:
            self._allowed_time = current_time + self.ttl
            return True

        return False

class SentryRedisTtlProcessor(SentryProcessor):
    """
    Processor for TTL handled by redis, which should work better for multiple server processes or versions
    """
    _KEY = 'SENTRY_TTL'
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=1):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.ttl = ttl

    def _allowed(self):
        # TODO Check on a per-exception basis
        if self.redis.exists(self._KEY):
            return False
        else:
            self.redis.set(self._KEY, "rate_limited", ex=self.ttl)

        return True

def deprecated(reason):
    """
    带原因说明的废弃装饰器
    可以用两种方式使用：
    @deprecated("请使用其他函数") - 带原因
    @deprecated - 不带原因
    """
    if isinstance(reason, str):
        # 带原因使用装饰器的情况
        def decorator(func):
            if inspect.isclass(func):
                fmt = "调用已废弃的类 {name} ({reason})."
            else:
                fmt = "调用已废弃的函数 {name} ({reason})."
                
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt.format(name=func.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    elif inspect.isclass(reason) or inspect.isfunction(reason):
        # 不带原因直接使用装饰器的情况
        func = reason
        if inspect.isclass(func):
            fmt = "调用已废弃的类 {name}."
        else:
            fmt = "调用已废弃的函数 {name}."
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt.format(name=func.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return wrapper
    else:
        raise TypeError(repr(type(reason)))