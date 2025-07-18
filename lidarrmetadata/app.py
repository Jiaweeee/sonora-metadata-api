import os
import uuid
import functools
import asyncio

from quart import Quart, abort, make_response, request, jsonify
from quart.exceptions import HTTPStatusException

import redis
from datetime import timedelta
import logging
import aiohttp
from dateutil import parser

# 导入 Prometheus 相关库
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from lidarrmetadata.metrics import metrics

import lidarrmetadata
from lidarrmetadata import api
from lidarrmetadata import config
from lidarrmetadata import provider
from lidarrmetadata import util
from lidarrmetadata.util import deprecated

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.info('Have app logger')

# 创建一个 Blueprint
from quart import Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 创建 Quart 应用
app = Quart(__name__)
app.config.from_object(config.get_config())

# 初始化 Prometheus 指标
metrics.init_app(app)

# 添加 metrics 路由到 API 蓝图中
@api_bp.route('/metrics')
async def api_metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# Allow all endpoints to be cached by default
@app.after_request
async def add_cache_control_header(response, expiry = provider.utcnow() + timedelta(seconds=app.config['CACHE_TTL']['cloudflare'])):
    if response.status_code not in set([200, 301, 400, 403, 404]):
        response.cache_control.no_cache = True
    # This is a bodge to figure out if we have already set any cache control headers
    elif not response.cache_control:
        if expiry:
            now = provider.utcnow()
            response.cache_control.public = True
            # We want to allow caching on cloudflare (which we can invalidate)
            # but disallow caching for local users (which we cannot invalidate)
            response.cache_control.s_maxage = int((expiry - now).total_seconds())
            response.cache_control.max_age = 0
            response.expires = now - timedelta(days=1)
        else:
            response.cache_control.no_cache = True
    return response
    
# Decorator to disable caching by endpoint
def no_cache(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        response = await make_response(result)
        response.cache_control.no_cache = True
        return response
    return wrapper

def get_search_query():
    """
    Search for a track
    """
    query = request.args.get('query', '')
    query = query.strip().strip('\x00')
    if not query:
        abort(400, 'No query provided')
    
    logger.debug(f"Search query: {query}")
    
    # These are invalid search queries for lucene
    if query == "+":
        return "plus"
    elif query == "-":
        return "minus"

    return query

@app.errorhandler(500)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Internal server error'), 500

@app.errorhandler(HTTPStatusException)
async def handle_http_error(e):
    return jsonify(error = e.description), e.status_code

@app.errorhandler(api.ArtistNotFoundException)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Artist not found'), 404

@app.errorhandler(api.ReleaseNotFoundException)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Release not found'), 404

@app.errorhandler(api.TrackNotFoundException)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Track not found'), 404

@app.errorhandler(api.DiscoverContentException)
async def handle_discover_content_error(e):
    metrics.record_exception(e)
    return jsonify(error=e.message), 503

@app.errorhandler(redis.ConnectionError)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Could not connect to redis'), 503

@app.errorhandler(redis.BusyLoadingError)
async def handle_error(e):
    metrics.record_exception(e)
    return jsonify(error='Redis not ready'), 503

def validate_mbid(mbid):
    """
    Validates Musicbrainz ID and returns flask response in case of error
    :param mbid: Musicbrainz ID to verify
    :return: Flask response if error, None if valid
    """
    try:
        uuid.UUID(mbid, version=4)
    except ValueError:
        return jsonify(error='Invalid UUID'), 400

@api_bp.route('/')
@no_cache
async def default_route():
    """
    Default route with API information
    :return:
    """
    vintage_providers = provider.get_providers_implementing(
        provider.DataVintageMixin)
    
    data = await vintage_providers[0].data_vintage()

    info = {
        'branch': os.getenv('GIT_BRANCH'),
        'commit': os.getenv('COMMIT_HASH'),
        'version': lidarrmetadata.__version__,
        'replication_date': data
    }
    return jsonify(info)


@api_bp.route('/artist/<mbid>', methods=['GET'])
async def get_artist_info_route(mbid):
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response

    artist, expiry = await api.get_artist_info(mbid)

    releases = await api.get_artist_releases(mbid)

    artist['works'] = releases

    return await add_cache_control_header(jsonify(artist), expiry)

@api_bp.route('/release/<mbid>', methods=['GET'])
async def get_release_info_route(mbid):
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response
    release, _ = await api.get_release_info(mbid)
    return jsonify(release)

@api_bp.route('/track/<mbid>', methods=['GET'])
async def get_track_info_route(mbid):
    """
    获取曲目详细信息
    """
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response
        
    track, _ = await api.get_track_info(mbid)
    return jsonify(track)

@api_bp.route('/search')
async def search():
    """Unified search endpoint that can search for artists, releases, tracks or all
    
    Query Parameters:
    - query: Search query (required)
    - type: Type of search - one of: artist, release, track, all (required)
    - artist: Artist name filter (optional)
    - limit: Maximum number of results to return (optional, default 10)
    
    Returns:
    {
        "results": [
            {
                "type": "artist|release|track",
                "score": float,
                "data": {
                    "id": string,              # MusicBrainz ID
                    "title": string,           # Name of artist/release/track
                    "entity_type": string,     # "Artist", "Album", or "Song"
                    "artist_name": string,     # Artist name(s) joined by " & " (for release/track only)
                    "image": string           # URL to cover image, can be null
                }
            },
            ...
        ]
    }
    """
    query = get_search_query()
    search_type = request.args.get('type')
    artist_name = request.args.get('artist', '')
    limit = request.args.get('limit', default=10, type=int)
    limit = None if limit < 1 else limit

    if not search_type:
        return jsonify(error='Search type not provided'), 400

    if search_type not in ['artist', 'release', 'track', 'all']:
        return jsonify(error=f'Invalid search type: {search_type}'), 400

    try:
        if search_type == 'artist':
            results = await api.get_artist_search_results(query, limit)
            formatted_results = [format_artist_result(item) for item in results]
            
        elif search_type == 'release':
            results = await api.get_release_search_results(query, limit, artist_name)
            formatted_results = [format_release_result(item) for item in results]
            
        elif search_type == 'track':
            results = await api.get_track_search_results(query, limit, artist_name)
            formatted_results = [format_track_result(item) for item in results]
            
        else:  # search_type == 'all'
            results = await asyncio.gather(
                api.get_artist_search_results(query, limit),
                api.get_release_search_results(query, limit, artist_name),
                api.get_track_search_results(query, limit, artist_name),
                return_exceptions=True
            )
            
            formatted_results = []
            
            # Process artists
            if not isinstance(results[0], Exception):
                formatted_results.extend([format_artist_result(item) for item in results[0]])
            
            # Process releases
            if not isinstance(results[1], Exception):
                formatted_results.extend([format_release_result(item) for item in results[1]])
            
            # Process tracks
            if not isinstance(results[2], Exception):
                formatted_results.extend([format_track_result(item) for item in results[2]])
            
            # Sort by score in descending order
            formatted_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Log any errors that occurred
            for i, result_type in enumerate(['artists', 'releases', 'tracks']):
                if isinstance(results[i], Exception):
                    logger.error(f"Error in {result_type} search: {str(results[i])}")

        return jsonify({'results': formatted_results})
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify(error='Internal server error'), 500

def format_search_result(item, entity_type):
    """Format search result into standardized structure
    
    Args:
        item: The search result item to format
        entity_type: One of 'artist', 'release', or 'track'
    
    Returns:
        Formatted search result
    """
    images = item.get('images')
    image = images.get('small', None) if images and isinstance(images, dict) else None

    # 处理艺术家信息
    artist_name = None
    if entity_type in ['release', 'track']:
        artists = item.get('artists', [])
        artist_names = []
        for artist in artists:
            artist_names.append(artist)
        artist_name = ' & '.join(artist_names) if artist_names else None

    # 确定实体类型显示名称
    entity_type_display = {
        'artist': 'Artist',
        'release': item.get('type', 'Album'),
        'track': 'Song'
    }[entity_type]

    # 获取标题
    title = item.get('name') if entity_type == 'artist' else item.get('title')

    result = {
        'type': entity_type,
        'score': item.get('score', 0),
        'data': {
            'id': item.get('id'),
            'title': title,
            'entity_type': entity_type_display,
            'image': image
        }
    }

    # 只为 release 和 track 添加 artist_name
    if entity_type in ['release', 'track']:
        result['data']['artist_name'] = artist_name

    return result

def format_artist_result(item):
    """Format artist search result into standardized structure"""
    return format_search_result(item, 'artist')

def format_release_result(item):
    """Format release search result into standardized structure"""
    return format_search_result(item, 'release')

def format_track_result(item):
    """Format track search result into standardized structure"""
    return format_search_result(item, 'track')


@deprecated(reason="Need to be re-implemented")
@api_bp.route('/invalidate')
@no_cache
async def invalidate_cache():

    if request.headers.get('authorization') != app.config['INVALIDATE_APIKEY']:
        return jsonify('Unauthorized'), 401

    since = request.args.get('since')
    if since:
        since = parser.parse(since)
    
    ## this is used as a prefix in various places to make sure
    ## we keep cache for different metadata versions separate
    base_url = app.config['CLOUDFLARE_URL_BASE'] + '/' +  app.config['ROOT_PATH'].lstrip('/').rstrip('/')
    
    ## Use a cache key to make sure we don't trigger this in parallel
    invalidation_in_progress_key = base_url + 'CacheInvalidationInProgress'
    in_progress = await util.CACHE.get(invalidation_in_progress_key)
    if in_progress:
        return jsonify('Invalidation already in progress'), 500
    
    try:
        await util.CACHE.set(invalidation_in_progress_key, True, timeout=60*5)
        logger.info('Invalidating cache')

        ## clear cache for all providers, aggregating a list of artists/albums
        ## that we need to invalidate the final responses for
        artists = set()
        albums = set()
        spotify_artists = set()
        spotify_albums = set()

        ## Get all the artists/albums that need updating
        cache_users = provider.get_providers_implementing(provider.InvalidateCacheMixin)
        for cache_user in cache_users:
            result = await cache_user.invalidate_cache(base_url, since)

            artists = artists.union(result['artists'])
            albums = albums.union(result['albums'])
            spotify_artists = spotify_artists.union(result['spotify_artists'])
            spotify_albums = spotify_albums.union(result['spotify_albums'])

        ## Invalidate all the local caches
        ## Use set rather than expires so that we add entries for new items also
        await asyncio.gather(
            util.ARTIST_CACHE.multi_set([(artist, None) for artist in artists], ttl=0, timeout=None),
            util.SPOTIFY_CACHE.multi_set([(spotify_artist, None) for spotify_artist in spotify_artists], ttl=0, timeout=None),
            util.SPOTIFY_CACHE.multi_set([(spotify_album, None) for spotify_album in spotify_albums], ttl=0, timeout=None)
        )

        ## Invalidate cloudflare cache
        invalidated = ([f'{base_url}/artist/{artist}' for artist in artists] + 
            [f'{base_url}/album/{album}' for album in albums] +
            [f'{base_url}/spotify/artist/{spotify_artist}' for spotify_artist in spotify_artists] +
            [f'{base_url}/spotify/album/{spotify_album}' for spotify_album in spotify_albums]
        )
        await invalidate_cloudflare(invalidated)
    
    finally:
        await util.CACHE.delete(invalidation_in_progress_key)
        # make sure any exceptions are not swallowed
        pass
        
    logger.info('Invalidation complete')
    
    return jsonify(invalidated)

async def invalidate_cloudflare(files):
    
    zoneid = app.config['CLOUDFLARE_ZONE_ID']
    if not zoneid:
        return
    
    url = f'https://api.cloudflare.com/client/v4/zones/{zoneid}/purge_cache'
    headers = {'X-Auth-Email': app.config['CLOUDFLARE_AUTH_EMAIL'],
               'X-Auth-Key': app.config['CLOUDFLARE_AUTH_KEY'],
               'Content-Type': 'application/json'}
    
    async with aiohttp.ClientSession() as session:
        # cloudflare only accepts 30 files at a time
        for i in range(0, len(files), 30):
            data = {'files': files[i:i+30]}
            retries = 2
            
            while retries > 0:
                async with session.post(url, headers=headers, json=data) as r:
                    logger.info(await r.text())
                    json = await r.json()

                    if json.get('success', False):
                        break
                    
                    retries -= 1


@api_bp.route('/discover/new-releases')
async def discover_new_releases():
    """Get new releases from Billboard 200 chart.
    
    This endpoint returns a list of recently released albums from the Billboard 200 chart,
    filtered to only include releases from the past 2 months.
    """
    result = await api.get_new_releases()
    return jsonify(result)

@api_bp.route('/discover/hot-songs')
async def discover_hot_songs():
    """Get hot songs from Apple Music charts
    
    This endpoint returns the top 24 songs from Apple Music's hot songs chart.
    Results are cached for 7 days to reduce API calls.
    
    Returns:
        JSON: A JSON response containing the list of hot songs and cache information
        
    Raises:
        HotSongsException: If there is an error fetching the hot songs data
    """
    # Call api.get_hot_songs to retrieve hot songs data
    results = await api.get_hot_songs()
    return jsonify(results)

@api_bp.route('/discover/charts')
async def discover_charts():
    """
    Get all charts.

    Returns:
        JSON: A JSON response containing the list of charts
    """
    charts = await api.get_all_charts()

    result = {
        'charts': charts
    }
    return jsonify(result)

@api_bp.route('/discover/cache/invalidate', methods=['POST'])
@no_cache
async def invalidate_discover_cache():
    """Invalidate one or more discover cache entries
    
    This endpoint allows invalidating the cache for specific discover endpoints.
    It requires proper authorization and accepts a key parameter that identifies
    which cache(s) to invalidate.
    
    Args:
        key: The cache key(s) to invalidate (e.g., 'hot-songs', 'new-releases')
             Can be a single key, a comma-separated list of keys, or multiple key parameters
        
    Returns:
        JSON response indicating success or failure
    """
    # Verify authorization
    if request.headers.get('authorization') != app.config['INVALIDATE_APIKEY']:
        return jsonify(error='Unauthorized'), 401
    
    # Get all keys from the request arguments
    # This handles both comma-separated values and multiple key parameters
    keys = []
    
    # Handle multiple key parameters
    if request.args.getlist('key'):
        keys.extend(request.args.getlist('key'))
    
    # If no keys were found, return an error
    if not keys:
        return jsonify(error='No keys provided for cache invalidation'), 400
    
    # Process comma-separated values in each key parameter
    processed_keys = []
    for key_param in keys:
        if ',' in key_param:
            processed_keys.extend([k.strip() for k in key_param.split(',') if k.strip()])
        else:
            processed_keys.append(key_param.strip())
    
    # Remove duplicates while preserving order
    unique_keys = []
    for key in processed_keys:
        if key not in unique_keys:
            unique_keys.append(key)
    
    results = []
    failed_keys = []
    
    # Delete each cache entry
    for key in unique_keys:
        try:
            await util.CACHE.delete(key)
            results.append(key)
        except Exception as e:
            logger.error(f'Error invalidating discover cache for {key}: {str(e)}')
            failed_keys.append({"key": key, "error": str(e)})
    
    # Prepare response
    response = {
        'success': len(results) > 0,
        'message': f'Successfully invalidated {len(results)} cache entries',
        'invalidated_keys': results,
    }
    
    if failed_keys:
        response['failed_keys'] = failed_keys
        
    status_code = 200 if len(results) > 0 else 500
    return jsonify(response), status_code

@api_bp.route('/discover/chart/<chart_id>')
async def get_chart(chart_id):
    """
    Get a specific chart.

    Returns:
        JSON: A JSON response containing the chart data
    """
    if chart_id not in api.chart_map:
        return jsonify(error='Invalid chart ID'), 400
    
    chart_function = api.chart_map[chart_id]
    chart = await chart_function()
    return jsonify(chart)

@api_bp.route('/discover')
async def get_discover_data():
    """
    Get discover data.

    Returns:
        JSON: A JSON response containing the discover data
    """
    new_releases = await api.get_new_releases()
    hot_songs = await api.get_hot_songs()
    charts = await api.get_all_charts()
        
    return jsonify({
        'new_releases': new_releases,
        'hot_songs': hot_songs,
        'charts': charts,
    })

@app.after_serving
async def run_async_del():
    async_providers = provider.get_providers_implementing(provider.AsyncDel)
    for prov in async_providers:
        await prov._del()
        
# 在文件末尾注册 Blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=config.get_config().HTTP_PORT, use_reloader=True)
