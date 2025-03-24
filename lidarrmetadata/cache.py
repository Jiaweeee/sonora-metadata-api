"""
Defines the custom redis cache backend which compresses pickle dumps
"""
import functools
import hashlib
import logging
import contextlib
import zlib
import asyncio
import asyncpg
import datetime
from timeit import default_timer as timer

from aiocache.serializers import BaseSerializer, PickleSerializer
from aiocache.base import BaseCache

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.info('Have cache logger')

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle
    
class CompressionSerializer(BaseSerializer):

    # This is needed because zlib works with bytes.
    # this way the underlying backend knows how to
    # store/retrieve values
    DEFAULT_ENCODING = None

    def dumps(self, value):
        return zlib.compress(pickle.dumps(value))

    def loads(self, value):
        if value is None:
            return None
        return pickle.loads(zlib.decompress(value))
    
class ExpirySerializer(PickleSerializer):
    """
    Allows an expiry time to be returned at the same time as the value
    """
    
    DEFAULT_ENCODING = None

    def loads(self, value):
        if value is None:
            return None, datetime.datetime.now(datetime.timezone.utc)
        return super().loads(value[0]), value[1]

def conn(func):
    @functools.wraps(func)
    async def wrapper(self, *args, _conn=None, **kwargs):
        if _conn is None:
            pool = await self._get_pool()
            async with pool.acquire() as _conn:
                return await func(self, *args, _conn=_conn, **kwargs)

        return await func(self, *args, _conn=_conn, **kwargs)

    return wrapper

class PostgresBackend:
    """
    Simple postgres cache.
    """

    def __init__(self,
                 endpoint='cache-db',
                 port=5432,
                 user='abc',
                 password='abc',
                 db_name='lm_cache_db',
                 db_table='cache',
                 keep_expired=True,
                 loop=None,
                 **kwargs):
        
        self._db_host = endpoint
        self._db_port = port
        self._db_user = user
        self._db_password = password
        self._db_name = db_name
        self._db_table = db_table
        self._keep_expired = keep_expired

        self._pool = None
        self.__pool_lock = None
        self._loop = loop
        
        super().__init__(**kwargs)
        
    @property
    def _pool_lock(self):
        if self.__pool_lock is None:
            self.__pool_lock = asyncio.Lock()
        return self.__pool_lock
    
    async def _get_pool(self):
        async with self._pool_lock:
            if self._pool is None:
                
                # Initialize pool
                self._pool = await asyncpg.create_pool(host = self._db_host,
                                                       port = self._db_port,
                                                       user = self._db_user,
                                                       password = self._db_password,
                                                       database = self._db_name,
                                                       loop = self._loop,
                                                       statement_cache_size=0)
                
                # Make sure table is created
                async with self._pool.acquire() as _conn:
                    await self._create_table(_conn)
                
            return self._pool

    async def _close(self, *args, **kwargs):
        if self._pool is not None:
            await self._pool.close()
    
    async def _create_table(self, _conn=None):

        logger.debug("checking table")
        result = await _conn.fetchrow(
                f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '{self._db_table}');",
            )
        table_exists = result['exists']
        logger.debug(f"exists: {table_exists}")

        if table_exists:
            logger.debug("table exists")
        else:
            logger.debug("table doesn't exist")
            await _conn.execute("""CREATE OR REPLACE FUNCTION cache_updated() RETURNS TRIGGER
AS
$$
BEGIN
    NEW.updated = current_timestamp;
    RETURN NEW;
END;
$$
language 'plpgsql';"""
            )

            await _conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._db_table} (key varchar PRIMARY KEY, expires timestamp with time zone, updated timestamp with time zone default current_timestamp, value bytea);"
                f"CREATE INDEX IF NOT EXISTS {self._db_table}_expires_idx ON {self._db_table}(expires);"
                f"CREATE INDEX IF NOT EXISTS {self._db_table}_updated_idx ON {self._db_table}(updated DESC) INCLUDE (key);"
                f"CREATE TRIGGER {self._db_table}_updated_trigger BEFORE UPDATE ON {self._db_table} FOR EACH ROW WHEN (OLD.value IS DISTINCT FROM NEW.value) EXECUTE PROCEDURE cache_updated();"
            )
            
    @conn
    async def _get(self, key, encoding="utf-8", _conn=None):
        try:
            logger.debug(f"getting {key}")
            start = timer()
            result = await _conn.fetchrow(
                f"SELECT value, expires FROM {self._db_table} WHERE key = $1;",
                key
            )
            end = timer()
            elapsed = int((end - start) * 1000)
            logger.debug(f"got {key} in {elapsed}ms")
            if result is not None:
                return result
        except KeyError:
            return None
        return None

    @conn
    async def _multi_get(self, keys, encoding="utf-8", _conn=None):
        try:
            start = timer()
            result = await _conn.fetch(
                f"""
select value, expires from {self._db_table}
right join
(
    select key, row_number() over() as key_sorter
    from (select unnest($1::text[]) as key) as y
) x on x.key = {self._db_table}.key
order by x.key_sorter""",
                keys
            )
            end = timer()
            elapsed = int((end - start) * 1000)
            logger.debug(f"got {len(keys)} keys in {elapsed}ms")
            if result is not None:
                return result
        except KeyError:
            return None
        return None

    @conn
    async def _set(self, key, value, ttl=None, _cas_token=None, _conn=None):
        if ttl is not None:
            expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds = ttl)
        else:
            expiry = None
            
        await _conn.execute(
            f"INSERT INTO {self._db_table} (key, expires, value) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT(key) DO UPDATE "
            "SET expires = EXCLUDED.expires, "
            "value = EXCLUDED.value;",
            key, expiry, value
        )

        return True
    
    @conn
    async def _multi_set(self, pairs, ttl=None, _conn=None):
        if ttl is not None:
            expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds = ttl)
        else:
            expiry = None
        
        records = [(key, expiry, value) for key, value in pairs]
        logger.debug(records[1:10])
        
        async with _conn.transaction():
            await _conn.execute(
                f"CREATE TEMP TABLE tmp_table ON COMMIT DROP AS SELECT key, expires, value FROM {self._db_table} WITH NO DATA;"
            );
            
            result = await _conn.copy_records_to_table("tmp_table", records=records)
            logger.debug(result)
            
            await _conn.execute(
                f"INSERT INTO {self._db_table} (key, expires, value) "
                "SELECT key, expires, value FROM tmp_table "
                "ON CONFLICT(key) DO UPDATE "
                "SET expires = EXCLUDED.expires, "
                "value = EXCLUDED.value;",
            )
            
        return True

    @conn
    async def _delete(self, key, _conn=None):
        await _conn.execute(
            f"DELETE FROM {self._db_table} WHERE key = $1;",
            key
        )
        return True
    
    @conn
    async def _expire(self, key, ttl, _conn=None):
        if ttl != 0:
            expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds = ttl)
        else:
            expiry = None
        
        await _conn.execute(
            f"UPDATE {self._db_table} "
            "SET expires = $2 "
            "WHERE key = $1",
            key, expiry
        )
    
    @conn
    async def _clear(self, namespace=None, _conn=None):
        await _conn.execute(f"TRUNCATE {self._db_table};")
        return True
    
    @conn
    async  def _get_stale(self, count, expires_before, _conn=None):
        results = await _conn.fetch(
            f"SELECT key FROM {self._db_table} "
            "WHERE expires < $1 "
            "ORDER by expires "
            "LIMIT $2;",
            expires_before, count
        )
        return [item['key'] for item in results] if results else []

    @conn
    async def _get_recently_updated(self, updated_since, limit, _conn=None):
        results = await _conn.fetch(
            f"SELECT key, updated FROM {self._db_table} "
            "WHERE updated > $1 "
            "ORDER by updated DESC "
            "LIMIT $2;",
            updated_since,
            limit
        )

        if not results:
            return {'Since': updated_since.isoformat(),
                    'Count': 0,
                    'Limited': False,
                    'Items': []}

        return {'Since': results[-1]['updated'],
                'Count': len(results),
                'Limited': len(results) == limit,
                'Items': [item['key'] for item in results]}

    @conn
    async def _get_all_keys_paged(self, limit, offset, _conn=None):
        """
        获取所有键，支持分页
        
        Args:
            limit (int): 每页返回的最大记录数
            offset (int): 分页偏移量
            _conn: 数据库连接对象
            
        Returns:
            dict: 包含键列表、总记录数和是否有更多数据的字典
        """
        start = timer()
        
        # 先获取总记录数
        count_result = await _conn.fetchrow(
            f"SELECT COUNT(*) as total FROM {self._db_table}"
        )
        total_count = count_result['total']
        
        # 获取指定页的键
        results = await _conn.fetch(
            f"SELECT key FROM {self._db_table} "
            "ORDER BY key "  # 按键名排序，确保分页结果一致
            "LIMIT $1 OFFSET $2",
            limit, offset
        )
        
        keys = [item['key'] for item in results] if results else []
        
        end = timer()
        elapsed = int((end - start) * 1000)
        
        # 计算是否有更多数据
        has_more = offset + len(keys) < total_count
        
        logger.debug(f"获取了 {len(keys)} 个键，偏移量={offset}，总数={total_count}，耗时={elapsed}ms")
        
        return {
            'keys': keys,
            'total': total_count,
            'offset': offset,
            'limit': limit,
            'has_more': has_more
        }

class PostgresCache(PostgresBackend, BaseCache):
    """
    Cache implementation using postgres table
    """
    
    NAME = "postgres"
    
    def __init__(self, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self.serializer = serializer or ExpirySerializer()
        
    async def get_stale(self, count, expires_before, _conn=None):
        return await self._get_stale(count, expires_before, _conn=_conn)

    async def get_recently_updated(self, updated_since, limit, _conn=None):
        return await self._get_recently_updated(updated_since, limit, _conn=_conn)
        
    async def get_all_keys_paged(self, limit=1000, offset=0, _conn=None):
        """
        获取所有缓存键，支持分页
        
        Args:
            limit (int): 每页返回的最大记录数，默认1000
            offset (int): 分页偏移量，默认0
            _conn: 数据库连接对象
            
        Returns:
            dict: 包含键列表、总记录数和是否有更多数据的字典
        """
        return await self._get_all_keys_paged(limit, offset, _conn=_conn)


class NullCache(BaseCache):
    """
    Dummy cache that doesn't store any data
    """
    
    NAME = "dummy"
    
    def __init__(self, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self.serializer = serializer or PickleSerializer()
    
    async def _get(self, key, encoding="utf-8", _conn=None):
        return None
    
    async def _set(self, key, value, ttl=None, _cas_token=None, _conn=None):
        return True
    
    async def get_stale(self, count, expires_before, _conn=None):
        return []
        
    async def get_all_keys_paged(self, limit=1000, offset=0, _conn=None):
        """
        获取所有缓存键，支持分页 (空实现)
        
        Args:
            limit (int): 每页返回的最大记录数，默认1000
            offset (int): 分页偏移量，默认0
            _conn: 数据库连接对象
            
        Returns:
            dict: 包含空键列表和元数据的字典
        """
        return {
            'keys': [],
            'total': 0,
            'offset': offset,
            'limit': limit,
            'has_more': False
        }
