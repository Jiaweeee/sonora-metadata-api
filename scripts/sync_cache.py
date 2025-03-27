#!/usr/bin/env python3


import os
import sys
import asyncio
import logging
import argparse
import time
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidarrmetadata.cache import PostgresCache
from lidarrmetadata import config
from lidarrmetadata import util
from lidarrmetadata import provider
# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sync_cache')

# 全局参数
BATCH_SIZE = 10000
MAX_RETRIES = 3
RETRY_INTERVAL = 5

# 远程数据库连接参数
REMOTE_HOST = '82.29.153.93'  # 替换为实际的远程主机地址
REMOTE_PORT = 54321
REMOTE_USER = 'abc'  # 替换为实际的远程用户名
REMOTE_PASSWORD = 'abc'  # 替换为实际的远程密码
REMOTE_DB = 'lm_cache_db'  # 替换为实际的远程数据库名

# 获取所有的可用缓存名称（从util模块）
AVAILABLE_CACHES = {
    'spotify': util.SPOTIFY_CACHE,
    'artist': util.ARTIST_CACHE,
    'fanart': util.FANART_CACHE,
    'wikipedia': util.WIKI_CACHE,
    'release_image': util.RELEASE_IMAGE_CACHE,
    'release': util.RELEASE_CACHE,
    'track': util.TRACK_CACHE,
    'artist_image': util.ARTIST_IMAGE_CACHE
}

async def get_cache_instance(cache_name, config_obj, is_remote=False):
    """获取缓存实例，可以是本地或远程的"""
    try:
        cache_config = config_obj.CACHE_CONFIG[cache_name].copy()
        logger.debug(f"原始缓存配置: {cache_config}")
        
        # 移除 PostgresCache 构造函数不支持的参数
        if 'cache' in cache_config:
            del cache_config['cache']
        
        # 移除其他不支持的参数
        for key in ['ttl', 'timeout', 'namespace', 'key_builder']:
            if key in cache_config:
                del cache_config[key]
        
        if is_remote:
            # 将远程连接信息替换到配置中
            cache_config['endpoint'] = REMOTE_HOST
            cache_config['port'] = REMOTE_PORT
            cache_config['user'] = REMOTE_USER
            cache_config['password'] = REMOTE_PASSWORD
            cache_config['db_name'] = REMOTE_DB
        
        logger.debug(f"处理后的缓存配置: {cache_config}")
        cache = PostgresCache(**cache_config)
        
        # 测试连接
        pool = await cache._get_pool()
        logger.info(f"成功连接到{'远程' if is_remote else '本地'} {cache_name} 缓存")
        
        return cache
    except Exception as e:
        location = "远程" if is_remote else "本地"
        logger.error(f"连接到{location} {cache_name} 缓存失败: {e}")
        raise

async def retry_operation(operation, *args, **kwargs):
    """使用重试机制执行操作"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"操作失败，已达到最大重试次数: {e}")
                raise
            
            logger.warning(f"操作失败，正在重试 ({retries}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(RETRY_INTERVAL)

async def get_cache_count(cache, cache_name):
    """获取缓存中的记录总数"""
    try:
        # 尝试使用SQL直接查询count
        async with (await cache._get_pool()).acquire() as conn:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {cache._db_table};")
            return result
    except Exception as e:
        logger.warning(f"获取{cache_name}缓存记录数失败: {e}")
        # 如果直接查询失败，返回None
        return None

async def get_all_keys_paged(source_cache, limit, offset):
    """
    获取缓存中所有键的通用方法，处理不同返回类型
    """
    try:
        result = await source_cache.get_all_keys_paged(limit=limit, offset=offset)
        logger.debug(f"get_all_keys_paged 返回类型: {type(result)}")
        
        # 如果是字典，按照预期处理
        if isinstance(result, dict):
            return {
                'keys': result.get('keys', []),
                'has_more': result.get('has_more', False),
                'total': result.get('total', 0),
                'offset': result.get('offset', offset)
            }
        # 如果是列表，将其转换为我们期望的格式
        elif isinstance(result, list):
            return {
                'keys': result,
                'has_more': len(result) == limit,  # 如果获取了满页数据，假设还有更多
                'total': len(result) + offset,  # 这是一个估计值
                'offset': offset
            }
        # 如果是None或其他类型，返回空字典
        else:
            logger.warning(f"get_all_keys_paged 返回了意外的类型: {type(result)}")
            return {
                'keys': [],
                'has_more': False,
                'total': 0,
                'offset': offset
            }
    except Exception as e:
        logger.error(f"获取键时出错: {e}")
        return {
            'keys': [],
            'has_more': False,
            'total': 0,
            'offset': offset
        }

async def sync_cache(source_cache, target_cache, cache_name):
    """从源缓存同步数据到目标缓存"""
    offset = 0
    total_synced = 0
    start_time = time.time()
    loop_count = 0  # 添加循环计数器
    max_loop_count = args.max_loop_count  # 获取最大循环次数
    
    logger.info(f"开始同步 {cache_name} 缓存")
    
    try:
        # 尝试获取源缓存的记录总数
        source_count = await get_cache_count(source_cache, cache_name)
        if source_count is not None:
            logger.info(f"{cache_name} 缓存中包含约 {source_count} 条记录")
        
        # 获取第一页以确定是否有数据
        initial_result = await get_all_keys_paged(source_cache, limit=1, offset=0)
        logger.debug(f"初始查询结果: {initial_result}")
        
        if not initial_result or not initial_result.get('keys', []):
            logger.info(f"{cache_name} 缓存为空，无需同步")
            return
        
        # 获取所有键直到没有更多数据
        with tqdm(total=source_count, desc=f"同步 {cache_name}", unit="记录") as pbar:
            while True:
                loop_count += 1
                logger.debug(f"循环次数: {loop_count}, 当前偏移量: {offset}")
                
                try:
                    keys_result = await get_all_keys_paged(source_cache, limit=BATCH_SIZE, offset=offset)
                    logger.debug(f"获取键结果: 有结果={bool(keys_result)}, 键数量={len(keys_result.get('keys', []) if keys_result else [])}")
                    logger.debug(f"分页信息: offset={keys_result.get('offset') if keys_result else 'N/A'}, has_more={keys_result.get('has_more') if keys_result else 'N/A'}")
                except Exception as e:
                    if args.ignore_errors:
                        logger.error(f"获取键时发生错误，但继续处理: {e}")
                        # 尝试跳到下一页
                        offset += BATCH_SIZE
                        continue
                    else:
                        raise
                
                if not keys_result or not keys_result.get('keys', []):
                    logger.info(f"没有更多的键，终止循环")
                    break
                
                keys = keys_result.get('keys', [])
                logger.info(f"正在处理 {cache_name} 缓存的 {offset} 到 {offset + len(keys)} 条记录")
                
                # 如果三次循环后仍然没有获取到任何键，则终止循环
                if loop_count > 3 and total_synced == 0 and len(keys) == 0:
                    logger.warning(f"连续 {loop_count} 次循环未获取到任何有效键，终止循环")
                    break
                
                # 获取当前批次的所有键值对
                pairs = []
                failed_keys = []

                source_results = await source_cache.multi_get(keys)
                for key, result in zip(keys, source_results):
                    if result is None:
                        failed_keys.append(key)
                    else:
                        value, _ = result
                        pairs.append((key, value))
                
                # Log the number of failed keys
                if failed_keys:
                    logger.warning(f"共有 {len(failed_keys)} 个键获取失败")
                
                if pairs:
                    # 批量写入目标缓存
                    try:
                        logger.debug(f"准备写入 {len(pairs)} 对键值对")
                        await retry_operation(target_cache.multi_set, pairs, ttl=None, timeout=None)
                        total_synced += len(pairs)
                        elapsed = time.time() - start_time
                        pbar.update(len(pairs))
                        logger.info(f"已同步 {total_synced} 条记录，耗时: {elapsed:.2f}秒，速率: {total_synced/elapsed:.2f}条/秒")
                    except Exception as e:
                        if args.ignore_errors:
                            logger.error(f"批量写入目标缓存失败，但继续处理: {e}")
                        else:
                            logger.error(f"批量写入目标缓存失败: {e}")
                            raise
                else:
                    logger.warning(f"本批次未找到有效的键值对，无数据写入")
                
                has_more = keys_result.get('has_more', False)
                logger.debug(f"has_more 标志: {has_more}")
                
                if not has_more:
                    logger.info(f"服务器表示没有更多数据，终止循环")
                    break
                
                # 为了避免无限循环，添加额外的检查
                if len(keys) == 0:
                    logger.warning(f"返回了0个键但 has_more={has_more}，终止循环")
                    break
                    
                # 确保 offset 正确递增
                new_offset = offset + len(keys)
                if new_offset <= offset:
                    logger.warning(f"偏移量未增加 (offset={offset}, new_offset={new_offset})，可能导致无限循环，终止循环")
                    break
                    
                offset = new_offset
                logger.debug(f"更新偏移量为 {offset}")
                
                # 设置一个最大循环次数以防止无限循环
                if loop_count >= max_loop_count:
                    logger.warning(f"达到最大循环次数 {loop_count}，终止循环")
                    break
        
        elapsed = time.time() - start_time
        logger.info(f"{cache_name} 缓存同步完成，共同步 {total_synced} 条记录，总耗时: {elapsed:.2f}秒，平均速率: {total_synced/elapsed if elapsed else 0:.2f}条/秒")
        
    except Exception as e:
        logger.error(f"同步 {cache_name} 缓存时发生错误: {e}")
        raise

async def main():
    """主函数：同步所有可用的缓存"""
    global BATCH_SIZE, MAX_RETRIES, RETRY_INTERVAL
    
    # 更新全局参数
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.max_retries:
        MAX_RETRIES = args.max_retries
    if args.retry_interval:
        RETRY_INTERVAL = args.retry_interval
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    
    try:
        config_obj = config.get_config()
        logger.info("成功加载配置")
        
        # 决定要同步的缓存
        cache_names = []
        if args.all:
            # 使用所有可用的缓存（除了default）
            cache_names = list(AVAILABLE_CACHES.keys())
        else:
            # 处理指定的缓存
            for cache_name in args.caches:
                if cache_name in AVAILABLE_CACHES:
                    cache_names.append(cache_name)
                else:
                    logger.warning(f"未知的缓存名称: {cache_name}，将被忽略")
            
            # 如果没有指定任何缓存，默认同步所有
            if not cache_names:
                cache_names = list(AVAILABLE_CACHES.keys())
        
        logger.info(f"将同步以下缓存: {', '.join(cache_names)}")
        
        # 创建源缓存和目标缓存实例
        for cache_name in cache_names:
            logger.info(f"===== 开始同步 {cache_name} 缓存 =====")
            source_cache = None
            target_cache = None
            
            try:
                source_cache = await get_cache_instance(cache_name, config_obj)
                target_cache = await get_cache_instance(cache_name, config_obj, is_remote=True)
                
                await sync_cache(source_cache, target_cache, cache_name)
                logger.info(f"===== {cache_name} 缓存同步成功 =====\n")
                
            except Exception as e:
                logger.error(f"同步 {cache_name} 缓存失败: {e}")
            finally:
                # 关闭连接池
                if source_cache:
                    await source_cache._close()
                if target_cache:
                    await target_cache._close()
        
        logger.info("所有缓存同步完成")
        
    except Exception as e:
        logger.error(f"同步过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='同步缓存数据到远程服务器')
    
    # 性能和执行控制参数
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='每批处理的记录数')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help='操作失败时的最大重试次数')
    parser.add_argument('--retry-interval', type=int, default=RETRY_INTERVAL, help='重试之间的间隔秒数')
    parser.add_argument('--ignore-errors', action='store_true', help='忽略错误并继续处理')
    parser.add_argument('--max-loop-count', type=int, default=1000, help='最大循环次数，防止无限循环')
    
    # 缓存选择参数
    parser.add_argument('--all', action='store_true', help='同步所有可用缓存（默认）')
    parser.add_argument('--caches', nargs='+', help='指定要同步的缓存名称，如 spotify artist fanart wikipedia release_image release track artist_image')
    
    # 输出控制参数
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--verbose', '-v', action='store_true', help='显示详细输出')
    output_group.add_argument('--quiet', '-q', action='store_true', help='只显示警告和错误')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断，同步终止")
        sys.exit(130)
    except Exception as e:
        logger.error(f"执行同步任务时发生错误: {e}")
        sys.exit(1) 