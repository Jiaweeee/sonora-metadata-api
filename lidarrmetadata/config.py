"""
Lidarr metadata config
"""

import os
import six
import re

# Environment key to use for configuration setting. This environment variable
# may be set to override the default config if no CLI argument is given
ENV_KEY = 'LIDARR_METADATA_CONFIG'

# Dictionary to get a config class based on key. This is automatically
# populated on class creation by ConfigMeta. Keys are the name of the class
# before Config suffix in uppercase
CONFIGS = {}


# TODO Move these functions to util once circular dependency is resolved

def first_key(d):
    """
    Gets the first key of a dictionary
    :param d: Dictionary
    :return: First key in dictionary
    """
    return list(d.keys())[0]


def get_index_type(iterable):
    """
    Gets the index type of an iterable. Note that iterables with multiple
    index types are not supported

    :param iterable: Iterable to get index type of
    :return: Type of index of iterable
    """
    if isinstance(iterable, (tuple, list)):
        return int
    elif isinstance(iterable, dict):
        return type(first_key(iterable)) if iterable else None
    else:
        raise ValueError()


def get_value_type(iterable):
    """
    Gets the value type of an iterable. Note that iterables with multiple
    value types are not supported

    :param iterable: Iterable to get value type of
    :return: Value type of iterable
    """
    if isinstance(iterable, (tuple, list)):
        return type(iterable[0])
    elif isinstance(iterable, dict):
        return type(iterable.get(first_key(iterable)))
    else:
        raise ValueError()


def get_nested(iterable, indices, fail_return_first=False):
    """
    Gets a nested value of a series of indices

    :param iterable: Iterable to get value from
    :param indices: Sequence of indices to follow
    :param fail_return_first: Returns first key if an index doesn't exist.
            This is a somewhat dirty way of getting what we need for config.
            Defaults to False
    :return: Value at sequence of indices
    """
    index = get_index_type(iterable)(indices[0])
    if len(indices) > 1:
        return get_nested(iterable[index], indices[1:])
    else:
        if fail_return_first:
            try:
                return iterable[index]
            except IndexError:
                return iterable[0]
            except KeyError:
                return iterable[first_key(iterable)]
        else:
            return iterable[index]


def set_nested(iterable, indices, value):
    """
    Sets a nested value of a series of indices. Note that this will
    edit the iterable in-place since all iterables should be references

    :param iterable: Iterable to set value in
    :param indices: Indices to follow
    :param value: Value to set
    :return:
    """
    index = get_index_type(iterable)(indices[0])
    if len(indices) > 1:
        set_nested(iterable[index], indices[1:], value)
    else:
        if isinstance(iterable, dict):
            iterable.update({index: value})
        elif isinstance(iterable, list):
            if index < len(iterable):
                iterable[index] = value
            else:
                # Add Nones if we have a list. Note that we can't do
                # this for tuples since they're immutable
                set_nested(iterable.append(None), indices, value)
        else:
            iterable[index] = value


def split_camel_case(string):
    """
    Splits camel case string into list of strings
    :param string: String to split
    :returns: List of substrings in CamelCase
    """
    return re.sub('([a-z])([A-Z])', r'\1 \2', string).split()


def split_escaped(string, split_char=' ', escape_char='\\'):
    """
    Splits escaped string

    :param string: String to split
    :param split_char: Character to split on. Defaults to single space
    :param escape_char: Character to escape with. Defaults to \
    """
    ret = []
    current = ''
    skip = False
    for char in string:
        if skip:
            skip = False
            continue
        elif char == escape_char:
            current += split_char
            skip = True
        elif char == split_char:
            if current:
                ret.append(current)

            current = ''
        else:
            current += char

    if current:
        ret.append(current)

    return ret


class ConfigMeta(type):
    """
    Config metaclass to register config classes
    """

    def __init__(cls, name, bases, attr):
        """
        Called upon the creation of a new class
        """
        # Parent initialization
        type.__init__(cls, name, bases, attr)

        # Get key for name
        substrings = split_camel_case(name)
        substrings = substrings[
                     :-1] if substrings[-1].lower() == 'config' else substrings
        key = '_'.join([s.upper() for s in substrings])

        # Register class
        CONFIGS[key] = cls


class ConfigBase(object):
    # Instance so we don't create a new config every time (singleton)
    __instance = None

    def __init__(self):
        """
        Initialization. Uses instance if there is one, otherwise replaces class variables with environment variables
        """
        if self.__instance:
            self = self.__instance
        else:
            for var in dir(self):
                # Make sure it isn't a python special attribute
                if var.upper() != var:
                    continue

                self._set_env_override(var, getattr(self, var))

            self.__instance = self

    @staticmethod
    def _search_env(name):
        """
        Searches env variables for variables matching name and returns a list of indices
        :param name: Name to match
        :return: List of (var, value, [indices]) tuples
        """
        envs = filter(lambda k: k.split('__')[0] == name, os.environ.keys())
        return [{'config_var': var.split('__')[0],
                 'env_var': var,
                 'env_setting': os.getenv(var),
                 'indices': var.split('__')[1:]}
                for var in envs]

    def _set_env_override(self, var, original):
        """
        Gets the environment variable override value for a variable if it exists or returns the original value
        :param var: Name of variable to check. It will check the environment variable of the same name
        :return: Environment variable of object or original value if environment variable does not exist
        """
        original_type = type(original) if original is not None else str

        envs = self._search_env(var)
        override = None
        for env in envs:
            if env['indices']:
                original_value = get_nested(original, env['indices'], True)
                set_nested(original, env['indices'],
                           self._parse_env_value(env['env_setting'], type(original_value), original_value))
            else:
                setting = self._parse_env_value(env['env_setting'], original_type, original)
                setattr(self, var, setting)

        return override

    @classmethod
    def _parse_env_value(cls, env_setting, original_type, original_value):
        """
        Parses the value of an environment variable according to the type of the original variable
        :param env_setting: Environment setting as string
        :param original_type: Type of original variable
        :param original_value: Value of original variable
        :return:
        """
        # No override if there is no env setting
        if not env_setting:
            return original_value

        if isinstance(original_value, (list, tuple)):
            # Lists are separated with colons such as a:b:c -> ['a', 'b', 'c']
            list_item_type = type(original_value[0]) if original_value else str
            items = split_escaped(env_setting, split_char=':')
            override = original_type(map(list_item_type, items))
        elif isinstance(original_value, bool):
            return env_setting.lower() == 'true'
        else:
            override = original_type(env_setting)
        return override


class DefaultConfig(six.with_metaclass(ConfigMeta, ConfigBase)):
    """
    Base configuration class to define default values. All possible config
    values should be defined in this class to avoid KeyErrors or unexpected
    missing values. Explanation for the functionality of each configuration
    value should be provided above the variable and options should be listed
    in alphabetical order.

    Note that for the environmental override system to work correctly, keys
    in dictionary config variables should be capitalized.
    """

    ROOT_PATH = ''

    BLACKLISTED_ARTISTS = [
        'f731ccc4-e22a-43af-a747-64213329e088',  # [anonymous]
        '33cf029c-63b0-41a0-9855-be2a3665fb3b',  # [data]
        '314e1c25-dde7-4e4d-b2f4-0a7b9f7c56dc',  # [dialogue]
        'eec63d3c-3b81-4ad4-b1e4-7c147d4d2b61',  # [no artist]
        '9be7f096-97ec-4615-8957-8d40b5dcbc41',  # [traditional]
        '125ec42a-7229-4250-afc5-e057484327fe',  # [unknown]
        '89ad4ac3-39f7-470e-963a-56509c546377',  # Various Artists
    ]
    
    # Host definitions used elsewhere
    REDIS_HOST = 'redis'
    REDIS_PORT = 6379
    POSTGRES_CACHE_HOST = 'cache-db'
    POSTGRES_CACHE_PORT = 5432
    MB_DB_HOST = '82.29.153.93'

    # TTL set in Cache-Control headers.  Use 0 to disable caching.
    # The GOOD value is used if we got info from all providers
    # The BAD value is used if some providers were unavailable but
    # there was enough information to return a useful response
    # (e.g. we are missing overviews or images)
    DAYS = 60 * 60 * 24
    WEEKS = 7 * DAYS
    MONTHS = 30 * DAYS
    YEARS = 365 * DAYS
    
    USE_CACHE = True
    CACHE_TTL = {
        'cloudflare': DAYS * 30,
        'changes': 60,
        'chart': DAYS * 1,
        'provider_error': 60 * 30,
        'redis': DAYS * 7,
        'fanart': DAYS * 30,
        'wikipedia': DAYS * 7,
        'release_image': MONTHS * 1,
        'release': MONTHS * 3,
        'track': MONTHS * 3,
    }
    
    @property
    def CACHE_CONFIG(self):
        return {
                'default': {
                    'cache': 'aiocache.RedisCache',
                    'endpoint': self.REDIS_HOST,
                    'port': self.REDIS_PORT,
                    'namespace': 'lm3.7',
                    'serializer': {
                    'class': 'lidarrmetadata.cache.CompressionSerializer'
                },
            },
            'fanart': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'fanart',
                'timeout': 0,
            },
            'wikipedia': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'wikipedia',
                'timeout': 0,
            },
            'artist': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'artist',
                'timeout': 0,
            },
            'album': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'album',
                'timeout': 0,
            },
            'spotify': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'spotify',
                'timeout': 0,
            },
            'release_image': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'release_image',
                'timeout': 0,
            },
            'release': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'release',
                'timeout': 0,
            },
            'track': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'track',
                'timeout': 0,
            },
            'artist_image': {
                'cache': 'lidarrmetadata.cache.PostgresCache',
                'endpoint': self.POSTGRES_CACHE_HOST,
                'port': self.POSTGRES_CACHE_PORT,
                'db_table': 'artist_image',
                'timeout': 0,
            }
        }

    CRAWLER_BATCH_SIZE = {
        'wikipedia': 50,
        'fanart': 500,
        'artist': 100,
        'album': 100
    }
    
    NULL_CACHE_CONFIG = {
        'default': {
            'cache': 'lidarrmetadata.cache.NullCache',
        },
        'fanart': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            },
        },
        'wikipedia': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'artist': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }

        },
        'album': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'spotify': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'release_image': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'release': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'track': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        },
        'artist_image': {
            'cache': 'lidarrmetadata.cache.NullCache',
            'serializer': {
                'class': 'lidarrmetadata.cache.ExpirySerializer'
            }
        }
    }
    
    # Debug mode
    DEBUG = False

    # Enable sending stats
    ENABLE_STATS = True

    # External request parameters
    # Class of limiter
    EXTERNAL_LIMIT_CLASS = 'NullRateLimiter'
    # Size of rate limit queue
    EXTERNAL_LIMIT_QUEUE_SIZE = 60
    # Rate limit time delta in ms
    EXTERNAL_LIMIT_TIME_DELTA = 1000
    # Request timeout in ms
    EXTERNAL_TIMEOUT = 5000

    # Redis db if using RedisRateLimiter
    EXTERNAL_LIMIT_REDIS_DB = 10
    # Redis host if using RedisRateLimiter
    EXTERNAL_LIMIT_REDIS_HOST = REDIS_HOST
    # Redis port if using RedisRateLimiter
    EXTERNAL_LIMIT_REDIS_PORT = REDIS_PORT

    # Fanart.tv API credentials
    FANART_KEY = 'e730454add1a72eed2bcc4dbc73d9775'
    # The API for standard keys is supposed to be delayed by 7 days but
    # in practise it appears the lag is slightly more
    FANART_API_DELAY_SECONDS = 8 * 24 * 60 * 60

    # Port to use
    HTTP_PORT = 5001

    # LastFM API connection details
    LASTFM_KEY = '8212faf2c2b44eb8a7a2a9353730b282'
    LASTFM_SECRET = '27c14aedca1394cec9426695f85aa6e0'

    # Spotify app credentials
    SPOTIFY_CREDENTIALS = [
        {
            'id': '2db54d025bc648d1882ac0467ecf48fd',
            'secret': '35f3659dca7045baa1073d9f749bdd88'
        },
        {
            'id': '7c800d690a8e43c88e58fd6c96acebac',
            'secret': '36407648935d4a07957e2766e208e619'
        },
        {
            'id': '722f70f56f494929b763cc353e1d8176',
            'secret': 'f046502b3e0943d285167841ca09c5a3'
        },
        {
            'id': '51ed1448c037447abe95cd02d413f450',
            'secret': '9c41734dab8f48a19acdfca3f5ca2e44'
        },
        {
            'id': 'df132b90db674bd0aeab4178d705cf21',
            'secret': '5ec6a6259f0045e5b19786827e655abb'
        },
        {
            'id': '22c26d33aacf444182da150afb2ce233',
            'secret': '1d91eb195b784ec7a0b5b082873abed3'
        },
        {
            'id': 'c4ce48a117b2497f8d768c8b35fb7fe9',
            'secret': '97c8d98581794b29b9c486eecd4c3822'
        },
        {
            'id': '98da0e80e9d148c8a8631931ce9ad925',
            'secret': '910a646310f44741997b7e10ac902451'
        },
        {
            'id': '784b6fadd9e74bcca77068998483a2da',
            'secret': '6059f8d06e2f416b87d8e3e503b7727e'
        },
        {
            'id': 'f358eb20c52b48ff89439890ddaaf560',
            'secret': '353caca2ebb04108a86194c07d3d911b'
        },
        {
            'id': '6c838517cc5846bd9a282ee99e714fb0',
            'secret': '904c86baaa594c1bad73b7eff62dd44d'
        },
        {
            'id': '5e7a432bfdbc4db6a46e5283b4a5fe3a',
            'secret': '30fa601c23964aec8098a1b284d7c4b9'
        },
        {
            'id': '053fda57923b4fb9901ca1ca8ee848f6',
            'secret': '2de133dde26046e595364672d44d49b1'
        },
        {
            'id': '4d5ac0e3ff334017953bcbd804ad16cf',
            'secret': '51477fbe5d39474087c0aa7519cdb8bc'
        },
        {
            'id': '093ae8fb1ced46e79bf7697486c7bd6c',
            'secret': '94592e6339574f16bd33c9e50d8200e8'
        },
        {
            'id': 'da029c33d3cb49ba92ef60135ab153d3',
            'secret': '2c8c626e738f4b94bb039a054b6212e4'
        },
        {
            'id': '8da9406c55694de8b2c5960055297500',
            'secret': '3b5f4253ab5c44e689e40fdecd0a62cb'
        },
        {
            'id': '3e4cd3a8790340cab824ef912b8edba0',
            'secret': 'd8402e14a8a844d48f68c8054e08a06c'
        },
        {
            'id': 'a792e09b6a2a45c5927152cf6de38668',
            'secret': 'ecf864be49f14dd5b8f3d3a27a3f2ec7'
        },
        {
            'id': 'e331680ae5c04403811ae3a4fc49f413',
            'secret': '2ae91e23831d4aa9976e8e16e6e4397d'
        },
        {
            'id': '07ccc6614446455a91fceb7bb0ddea67',
            'secret': 'b59efcf5739c4ad985c71aa8629ea19c'
        },
        {
            'id': '7e79848dd64041c3a8a6a65c8729d3ae',
            'secret': '498b306af04b41c1a4066389ff4127da'
        },
        {
            'id': 'e3eab5f5d8574f3881959f3bc09ae09b',
            'secret': 'edf74597ba224f61b3f950c32a8117a8'
        },
        {
            'id': '5c4a8efb13c24fb6b495e4368e4751f3',
            'secret': 'ebe870f593bd4f3ea2ca027e8347b9a9'
        },
        {
            'id': '7d6909af7acf44b183d1a9ce6bdb0f1c',
            'secret': 'dfebc35a46c24d79b9e0bd96fa66c1ee'
        },
        {
            'id': 'a71d674f01de49ab97cad7b868265938',
            'secret': '7ba9323a2cf14a2aacfd307f0b0b555f'
        },
        {
            'id': '738147af919742b0b7d082bea591a38f',
            'secret': '6d78833b12334f1aa1730421af14406b'
        }
    ]

    # Whether or not running in production
    PRODUCTION = False

    # Provider -> (args, kwargs) dictionaries
    PROVIDERS = {
        'MUSICBRAINZDBPROVIDER': ([], {
            'DB_HOST': MB_DB_HOST,
            'DB_PORT': 5432,
            'DB_USER': 'musicbrainz',
            'DB_PASSWORD': 'musicbrainz'
        }),
        'SOLRSEARCHPROVIDER': ([], {'SEARCH_SERVER': f'http://{MB_DB_HOST}:8983/solr'}),
        'FANARTTVPROVIDER': ([FANART_KEY], {}),
        'WIKIPEDIAPROVIDER': ([], {}),
        'SPOTIFYPROVIDER': ([], {'CREDENTIALS': SPOTIFY_CREDENTIALS}),
        'COVERARTARCHIVEPROVIDER': ([], {}),
        'CACHERELEASEIMAGEPROVIDER': ([], {}),
        'CACHEARTISTIMAGEPROVIDER': ([], {}),
    }

    # Stats server
    STATS_HOST = 'telegraf'
    STATS_PORT = 8092
    
    # Cloudflare details for invalidating cache on update
    CLOUDFLARE_ZONE_ID = ''
    CLOUDFLARE_AUTH_EMAIL = ''
    CLOUDFLARE_AUTH_KEY = ''
    CLOUDFLARE_URL_BASE = ''
    INVALIDATE_APIKEY = '78085ee343295addf2282cfc929abd4d121d75050fce8d6eb2bae9c0a81cb20d'

    # Testing mode
    TESTING = False

    # Hosted cache for third-party images
    IMAGE_CACHE_HOST = "imagecache.lidarr.audio"


class TestConfig(DefaultConfig):
    USE_CACHE = False
    ENABLE_STATS = False
    EXTERNAL_LIMIT_CLASS = 'NullRateLimiter'
    Testing = True


__config = None


def get_config():
    global __config
    if not __config:
        config_key = os.environ.get(ENV_KEY, 'DEFAULT').upper()
        __config = CONFIGS[config_key]()

        # TODO Replace with logging functions when we improve logging
        print('Initializing config {} from environment {}={}'.format(__config.__class__.__name__, ENV_KEY, config_key))
        for key in dir(__config):
            if key == key.upper():
                value = getattr(__config, key)
                print('\t{:24s}{:30s}{}'.format(key, str(type(value)), value))
    return __config


if __name__ == '__main__':
    get_config()
    get_config()
