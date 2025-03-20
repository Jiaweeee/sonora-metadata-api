import argparse
import asyncio
import datetime
from datetime import timedelta
import logging
from timeit import default_timer as timer
import sys

import aiohttp
import sentry_sdk

import lidarrmetadata
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

if CONFIG.SENTRY_DSN:
    if CONFIG.SENTRY_REDIS_HOST is not None:
        processor = util.SentryRedisTtlProcessor(redis_host=CONFIG.SENTRY_REDIS_HOST,
                                                 redis_port=CONFIG.SENTRY_REDIS_PORT,
                                                 ttl=CONFIG.SENTRY_TTL)
    else:
        processor = util.SentryTtlProcessor(ttl=CONFIG.SENTRY_TTL)
        
    sentry_sdk.init(dsn=CONFIG.SENTRY_DSN,
                    before_send=processor.create_event,
                    send_default_pii=True)

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
    格式化运行时间,可选显示处理速度
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
    
    # 并行执行多个请求
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
        # 创建多个并发任务
        tasks = []
        for _ in range(concurrent_requests):
            tasks.append(fetch_page(offset))
            offset += page_size
            
        # 并行执行所有任务
        results = await asyncio.gather(*tasks)
        
        # 处理结果
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
    初始化Spotify ID映射缓存，使用并发分页加载所有映射数据
    
    从数据库分批获取MusicBrainz ID与Spotify ID的映射关系，
    使用10个并发任务，每次请求1000条，直到获取所有数据，
    并将其存入缓存
    """
    link_provider = provider.get_providers_implementing(provider.ArtistByIdMixin)[0]
    
    # 基本参数设置
    page_size = 1000
    concurrency = 10  # 并发任务数
    offset = 0  # 从第0页开始
    all_pairs = []  # 存储所有结果
    
    # 创建一个全局字典来合并所有页面的结果
    global_id_map = {}
    
    async def fetch_page(page_offset):
        """
        获取单页Spotify映射数据，失败时进行重试
        
        Args:
            page_offset: 分页偏移量
            
        Returns:
            包含(spotifyid, mbid)元组的列表
        """
        max_retries = 3
        retry_count = 0
        backoff_time = 1.0  # 初始重试等待时间（秒）
        
        while retry_count <= max_retries:
            try:
                maps = await link_provider.get_spotify_mappings(limit=page_size, offset=page_offset)
                page_id_map = {}
                for item in maps:
                    if item['mbid'] not in page_id_map:
                        page_id_map[item['mbid']] = []
                    page_id_map[item['mbid']].append(item['spotifyid'])
                logger.debug(f"成功获取Spotify映射，偏移量={page_offset}，数量={len(maps)}")
                return page_id_map  # 返回字典而不是元组列表
            
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    # 计算指数退避时间
                    wait_time = backoff_time * (2 ** (retry_count - 1))
                    logger.warning(f"获取Spotify映射失败，偏移量={page_offset}，正在进行第{retry_count}次重试，等待{wait_time}秒: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"获取Spotify映射失败，偏移量={page_offset}，已重试{max_retries}次，放弃: {str(e)}")
                    return {}
    
    while True:
        # 创建当前批次的任务
        tasks = []
        for i in range(concurrency):
            current_offset = offset + i * page_size
            tasks.append(fetch_page(current_offset))
        
        # 并发执行当前批次的任务
        results = await asyncio.gather(*tasks)
        
        # 处理结果，检查是否还有更多数据
        has_more_data = False
        total_records_in_batch = 0
        
        for page_id_map in results:
            if page_id_map:  # 如果页面有数据
                # 合并当前页面的映射到全局映射
                for mbid, spotifyids in page_id_map.items():
                    if mbid not in global_id_map:
                        global_id_map[mbid] = []
                    # 去重添加Spotify IDs
                    for spotify_id in spotifyids:
                        if spotify_id not in global_id_map[mbid]:
                            global_id_map[mbid].append(spotify_id)
                            
                total_records_in_batch += len(page_id_map)
                if len(page_id_map) == page_size:
                    has_more_data = True  # 至少有一页是满的，可能还有更多数据
        
        # 如果该批次没有获取到任何数据，说明已经完成
        if total_records_in_batch == 0:
            logger.info("没有更多Spotify映射数据")
            break
            
        # 如果所有页都不是满的，说明已经到达最后一批
        if not has_more_data:
            break
            
        # 更新偏移量，准备获取下一批
        offset += concurrency * page_size
    
    # 将合并后的全局映射转换为元组列表，用于缓存
    all_pairs = [(mbid, spotifyids) for mbid, spotifyids in global_id_map.items()]
    
    logger.info(f"共加载了 {len(all_pairs)} 条Spotify ID映射，包含 {sum(len(spotifyids) for _, spotifyids in all_pairs)} 个Spotify ID")
    
    # 清除并重新填充缓存
    if all_pairs:
        await util.SPOTIFY_CACHE.clear()
        await util.SPOTIFY_CACHE.multi_set(all_pairs, ttl=None, timeout=None)
    else:
        logger.warning("未找到任何Spotify映射数据，缓存未更新")

# TODO: 需要解决 API 限流问题
async def crawl_artist_images():
    """
    爬取所有艺术家的图片信息并缓存
    
    从ARTIST_CACHE获取所有艺术家IDs，通过SPOTIFY_CACHE查找对应的Spotify IDs，
    然后使用SpotifyProvider获取艺术家图片，并将结果存入ARTIST_IMAGE_CACHE，
    过期时间设为100年后。包含完善的错误处理和日志记录。
    """
    logger.info("开始获取艺术家图片...")
    start = timer()
    
    # 获取所有艺术家IDs，使用分页方法
    try:
        # 使用新的分页方法获取所有键
        all_artists = []
        page_size = 10000  # 每页获取的记录数
        offset = 0
        
        while True:
            # 获取一页艺术家ID
            result = await util.SPOTIFY_CACHE.get_all_keys_paged(limit=page_size, offset=offset)
            
            if not result['keys']:
                break
                
            all_artists.extend(result['keys'])
            logger.debug(f"已获取 {len(all_artists)}/{result['total']} 个艺术家ID...")
            
            # 如果没有更多数据，退出循环
            if not result['has_more']:
                break
                
            # 更新偏移量
            offset += page_size
            
        if not all_artists:
            logger.warning("未找到任何艺术家ID，请确保先运行初始化命令")
            return
            
        logger.info(f"共找到 {len(all_artists)} 个艺术家ID")
    except Exception as e:
        logger.error(f"获取艺术家IDs时出错: {str(e)}")
        return
    
    # 初始化SpotifyProvider
    try:
        artist_image_provider = provider.get_providers_implementing(provider.SpotifyProvider)[0]
        if not artist_image_provider:
            logger.error("未找到有效的SpotifyProvider，无法获取图片")
            return
    except Exception as e:
        logger.error(f"初始化SpotifyProvider时出错: {str(e)}")
        return
    
    # 设置过期时间为100年后
    expiry_time = provider.utcnow() + timedelta(days=365*100)
    
    # 统计计数器
    total_processed = 0
    total_with_images = 0
    total_errors = 0
    batch_size = 100  # 每次处理的批次大小
    
    # 分批处理艺术家
    for i in range(0, len(all_artists), batch_size):
        batch = all_artists[i:i+batch_size]
        logger.debug(f"正在处理艺术家批次 {i//batch_size + 1}/{len(all_artists)//batch_size + 1}, "
                    f"共 {len(batch)} 个艺术家")
        
        for mbid in batch:
            total_processed += 1
            logger.debug(f"开始处理艺术家 MBID: {mbid} ({total_processed}/{len(all_artists)})")
            
            try:
                # 从SPOTIFY_CACHE获取spotify_ids
                result = await util.SPOTIFY_CACHE.get(mbid)
                
                # 处理可能的None返回值
                if result is None:
                    logger.debug(f"艺术家 {mbid} 在Spotify缓存中未找到对应记录")
                    continue
                    
                spotify_ids, _ = result
                
                if not spotify_ids:
                    logger.debug(f"艺术家 {mbid} 没有关联的Spotify ID")
                    continue
                
                logger.debug(f"艺术家 {mbid} 找到 {len(spotify_ids)} 个Spotify ID: {spotify_ids}")
                images = {}
                # 尝试每个spotify_id直到找到图片
                for spotify_id in spotify_ids:
                    try:
                        logger.debug(f"正在从Spotify获取艺术家 {mbid} 的图片 (spotify_id={spotify_id})")
                        
                        # 记录请求开始时间
                        start_request = timer()
                        logger.debug(f"开始Spotify API请求时间: {start_request}")
                        
                        # 直接使用异步API请求
                        images, expiry = await artist_image_provider.get_artist_images(spotify_id)
                        
                        logger.debug(f"Spotify API请求完成，耗时: {timer() - start_request:.2f}秒")
                        
                        if images:
                            logger.debug(f"成功获取艺术家 {mbid} 的图片: 找到 {len(images)} 个图片URL")
                            # 找到图片后存入缓存并跳出循环
                            await util.ARTIST_IMAGE_CACHE.set(mbid, images, ttl=(expiry_time - provider.utcnow()).total_seconds())
                            total_with_images += 1
                            
                            if total_processed % 100 == 0 or total_processed == len(all_artists):
                                logger.info(f"进度: {total_processed}/{len(all_artists)} 艺术家 ({total_with_images} 有图片, {total_errors} 错误), "
                                          f"已处理 {format_elapsed_time(timer() - start, total_processed)}")
                            break
                        else:
                            logger.debug(f"艺术家 {mbid} 的Spotify ID {spotify_id} 未返回任何图片")
                    except ProviderUnavailableException as pu:
                        # 处理ProviderUnavailableException
                        if "429" in str(pu):
                            # 处理限流错误
                            logger.warning(f"Spotify API限流错误 (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                            logger.info(f"等待5秒后继续...")
                            await asyncio.sleep(5)  # 简单的退避等待
                        elif "401" in str(pu):
                            # 处理认证错误
                            logger.warning(f"Spotify API认证错误 (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                            logger.info(f"可能是无效的Spotify ID，跳过...")
                            # 直接跳过这个ID
                            break
                        else:
                            # 处理其他API错误
                            logger.warning(f"Spotify API错误 (MBID={mbid}, spotify_id={spotify_id}): {str(pu)}")
                            # 尝试下一个ID
                            continue
                    except Exception as e:
                        # 处理其它类型的错误
                        logger.warning(f"获取Spotify艺术家图片失败 (MBID={mbid}, spotify_id={spotify_id}): {str(e)}, 错误类型: {type(e)}")
                        import traceback
                        logger.debug(f"错误堆栈: {traceback.format_exc()}")
                        continue
            
            except Exception as e:
                logger.error(f"处理MusicBrainz艺术家ID {mbid}时出错: {str(e)}")
                total_errors += 1
                logger.debug(f"当前错误总数: {total_errors}")
                
        # 每批次后短暂休息，避免过于频繁的API调用
        await asyncio.sleep(1)
    
    elapsed = timer() - start
    logger.info(f"艺术家图片获取完成。总计: {total_processed} 艺术家, {total_with_images} 有图片, "
               f"{total_errors} 出错. 总耗时: {format_elapsed_time(elapsed)}")

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
    
async def crawl():
    await asyncio.gather(
        update_wikipedia(count = CONFIG.CRAWLER_BATCH_SIZE['wikipedia'], max_ttl = 60 * 60 * 2),
        update_items(get_artist_info_multi, util.ARTIST_CACHE, "artist", count = CONFIG.CRAWLER_BATCH_SIZE['artist']),
    )
    
async def init():
    await asyncio.gather(
        init_artists(),
        init_spotify(),
    )
    
def main():
    
    parser = argparse.ArgumentParser(prog="lidarr-metadata-crawler")
    parser.add_argument("--init-artists", action="store_true")
    parser.add_argument("--init-spotify", action="store_true")
    parser.add_argument("--crawl-artist-images", action="store_true")
    args = parser.parse_args()
    
    if args.init_artists:
        asyncio.run(init_artists())
        sys.exit()

    if args.init_spotify:
        asyncio.run(init_spotify())
        sys.exit()
    
    if args.crawl_artist_images:
        asyncio.run(crawl_artist_images())
        sys.exit()
    
    asyncio.run(crawl())
    
if __name__ == "__main__":
    main()
