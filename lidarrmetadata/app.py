import os
import uuid
import functools
import asyncio

from quart import Quart, abort, make_response, request, jsonify, redirect, url_for
from quart.exceptions import HTTPStatusException

import redis
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import datetime
from datetime import timedelta
from dateutil import parser
import time
import logging
import aiohttp
from timeit import default_timer as timer
from spotipy import SpotifyException
import Levenshtein

import lidarrmetadata
from lidarrmetadata import api
from lidarrmetadata import chart
from lidarrmetadata import config
from lidarrmetadata import provider
from lidarrmetadata import util
from lidarrmetadata.util import deprecated

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.info('Have app logger')

app = Quart(__name__)
app.config.from_object(config.get_config())

if app.config['SENTRY_DSN']:
    if app.config['SENTRY_REDIS_HOST'] is not None:
        processor = util.SentryRedisTtlProcessor(redis_host=app.config['SENTRY_REDIS_HOST'],
                                                redis_port=app.config['SENTRY_REDIS_PORT'],
                                                ttl=app.config['SENTRY_TTL'])
    else:
        processor = util.SentryTtlProcessor(ttl=app.config['SENTRY_TTL'])

    sentry_sdk.init(dsn=app.config['SENTRY_DSN'],
                    integrations=[FlaskIntegration()],
                    release=f"lidarr-metadata-{lidarrmetadata.__version__}",
                    before_send=processor.create_event,
                    send_default_pii=True)

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
def handle_error(e):
    sentry_sdk.capture_exception(e)
    return jsonify(error='Internal server error'), 500

@app.errorhandler(HTTPStatusException)
async def handle_http_error(e):
    return jsonify(error = e.description), e.status_code

@app.errorhandler(api.ArtistNotFoundException)
async def handle_error(e):
    return jsonify(error='Artist not found'), 404

@app.errorhandler(api.ReleaseNotFoundException)
async def handle_error(e):
    return jsonify(error='Release not found'), 404

@app.errorhandler(api.TrackNotFoundException)
async def handle_error(e):
    return jsonify(error='Track not found'), 404

@app.errorhandler(redis.ConnectionError)
def handle_error(e):
    return jsonify(error='Could not connect to redis'), 503

@app.errorhandler(redis.BusyLoadingError)
def handle_error(e):
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

@app.route('/')
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


@app.route('/artist/<mbid>', methods=['GET'])
async def get_artist_info_route(mbid):
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response

    artist, expiry = await api.get_artist_info(mbid)

    releases = await api.get_artist_releases(mbid)

    artist['works'] = releases

    return await add_cache_control_header(jsonify(artist), expiry)

@app.route('/artist/<mbid>/refresh', methods=['POST'])
async def refresh_artist_route(mbid):
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response

    await util.ARTIST_CACHE.set(mbid, None)
    base_url = app.config['CLOUDFLARE_URL_BASE'] + '/' +  app.config['ROOT_PATH'].lstrip('/').rstrip('/')
    await invalidate_cloudflare([f'{base_url}/artist/{mbid}'])
    return jsonify(success=True)

@app.route('/release/<mbid>', methods=['GET'])
async def get_release_info_route(mbid):
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response
    release, _ = await api.get_release_info(mbid)
    return jsonify(release)

@app.route('/track/<mbid>', methods=['GET'])
async def get_track_info_route(mbid):
    """
    获取曲目详细信息
    """
    uuid_validation_response = validate_mbid(mbid)
    if uuid_validation_response:
        return uuid_validation_response
        
    track, _ = await api.get_track_info(mbid)
    return jsonify(track)

@app.route('/chart/<name>/<type_>/<selection>')
async def chart_route(name, type_, selection):
    """
    Gets chart
    :param name: Name of chart. 404 if name invalid
    """
    name = name.lower()
    count = request.args.get('count', 10, type=int)

    # Get remaining chart-dependent args
    chart_kwargs = request.args.to_dict()
    if 'count' in chart_kwargs:
        del chart_kwargs['count']

    key = (name, type_, selection)

    # Function to get each chart. Use lower case for keys
    charts = {
        ('apple-music', 'album', 'top'): chart.get_apple_music_top_albums_chart,
        ('apple-music', 'album', 'new'): chart.get_apple_music_top_albums_chart,
        ('billboard', 'album', 'top'): chart.get_billboard_200_albums_chart,
        ('billboard', 'artist', 'top'): chart.get_billboard_100_artists_chart,
    }

    if key not in charts.keys():
        return jsonify(error='Chart {}/{}/{} not found'.format(*key)), 404
    else:
        result = await charts[key](count, **chart_kwargs)
        expiry = provider.utcnow() + timedelta(seconds=app.config['CACHE_TTL']['chart'])
        return await add_cache_control_header(jsonify(result), expiry)

@app.route('/search')
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
    # 处理 image
    image = None
    images = item.get('images', [])
    if images:
        if entity_type == 'artist':
            # Artist 的图片处理逻辑
            image_map = {}
            for img in images:
                image_map[img['CoverType']] = img['Url']
            if image_map.get('Poster'):
                image = image_map['Poster']
            elif image_map.get('Fanart'):
                image = image_map['Fanart']
        else:
            # Release 和 Track 的图片处理逻辑
            if isinstance(images, dict) and 'small' in images:
                image = images['small']

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

@app.route('/spotify/artist/<spotify_id>', methods=['GET'])
async def spotify_lookup_artist(spotify_id):
    mbid, expires = await util.SPOTIFY_CACHE.get(spotify_id)

    if mbid == 0 and expires > provider.utcnow():
        return jsonify(error='Not found'), 404
    if mbid is not None:
        return redirect(app.config['ROOT_PATH'] + url_for('get_artist_info_route', mbid=mbid), 301)

    # Search on links in musicbrainz db
    link_provider = provider.get_providers_implementing(provider.ArtistByIdMixin)[0]
    artistid = await link_provider.get_artist_id_from_spotify_id(spotify_id)
    logger.debug(f"Got match from musicbrainz db: {artistid}")
    if artistid:
        await util.SPOTIFY_CACHE.set(spotify_id, artistid, ttl=None)
        return redirect(app.config['ROOT_PATH'] + url_for('get_artist_info_route', mbid=artistid), 301)

    # Fall back to a text search for a popular album
    try:
        spotify_provider = provider.get_providers_implementing(provider.SpotifyIdMixin)[0]
        spotifyalbum = spotify_provider.album_from_artist(spotify_id)
    except SpotifyException:
        await util.SPOTIFY_CACHE.set(spotify_id, 0, ttl=app.config['CACHE_TTL']['cloudflare'])
        return jsonify(error='Not found'), 404

    if spotifyalbum is None:
        await util.SPOTIFY_CACHE.set(spotify_id, 0, ttl=app.config['CACHE_TTL']['cloudflare'])
        return jsonify(error='Not found'), 404
    
    spotifyalbum = await spotify_lookup_by_text_search(spotifyalbum)

    if spotifyalbum is None:
        await util.SPOTIFY_CACHE.set(spotify_id, 0, ttl=app.config['CACHE_TTL']['cloudflare'])
        return jsonify(error='Not found'), 404

    await util.SPOTIFY_CACHE.set(spotifyalbum['AlbumSpotifyId'], spotifyalbum['AlbumMusicBrainzId'], ttl=None)
    await util.SPOTIFY_CACHE.set(spotifyalbum['ArtistSpotifyId'], spotifyalbum['ArtistMusicBrainzId'], ttl=None)

    return redirect(app.config['ROOT_PATH'] + url_for('get_artist_info_route', mbid=spotifyalbum['ArtistMusicBrainzId']), 301)

@app.route('/spotify/album/<spotify_id>', methods=['GET'])
async def spotify_lookup_album(spotify_id):
    mbid, expires = await util.SPOTIFY_CACHE.get(spotify_id)

    if mbid == 0 and expires > provider.utcnow():
        return jsonify(error='Not found'), 404
    if mbid is not None:
        return redirect(app.config['ROOT_PATH'] + url_for('get_release_group_info_route', mbid=mbid), 301)

    # Search on links in musicbrainz db
    link_provider = provider.get_providers_implementing(provider.ReleaseGroupByIdMixin)[0]
    albumid = await link_provider.get_release_group_id_from_spotify_id(spotify_id)
    logger.debug(f"Got match from musicbrainz db: {albumid}")
    if albumid:
        await util.SPOTIFY_CACHE.set(spotify_id, 0, ttl=app.config['CACHE_TTL']['cloudflare'])
        return redirect(app.config['ROOT_PATH'] + url_for('get_release_group_info_route', mbid=albumid), 301)

    # Fall back to a text search
    try:
        spotify_provider = provider.get_providers_implementing(provider.SpotifyIdMixin)[0]
        spotifyalbum = spotify_provider.album(spotify_id)
    except SpotifyException:
        await util.SPOTIFY_CACHE.set(spotify_id, 0, ttl=None)
        return jsonify(error='Not found'), 404

    spotifyalbum = await spotify_lookup_by_text_search(spotifyalbum)
    if spotifyalbum is None:
        return jsonify(error='Not found'), 404

    await util.SPOTIFY_CACHE.set(spotifyalbum['AlbumSpotifyId'], spotifyalbum['AlbumMusicBrainzId'], ttl=None)

    return redirect(app.config['ROOT_PATH'] + url_for('get_release_group_info_route', mbid=spotifyalbum['AlbumMusicBrainzId']), 301)

async def spotify_lookup_by_text_search(spotifyalbum):
    logger.debug(f"Looking for album corresponding to Artist: {spotifyalbum['Artist']} Album: {spotifyalbum['Album']}")    
    
    # do search
    search_provider = provider.get_providers_implementing(provider.AlbumNameSearchMixin)[0]
    result = await search_provider.search_album_name(spotifyalbum['Album'], artist_name=spotifyalbum['Artist'], limit=1)

    if not result:
        ttl = app.config['CACHE_TTL']['cloudflare']
        await util.SPOTIFY_CACHE.set(spotifyalbum['AlbumSpotifyId'], 0, ttl=ttl)
        await util.SPOTIFY_CACHE.set(spotifyalbum['ArtistSpotifyId'], 0, ttl=ttl)
        return None

    # Map back to an artist
    albumid = result[0]['Id']
    album, validity = await api.get_release_group_info(result[0]['Id'])
    artistid = album['artistid']

    found_title = album['title']
    found_artist = next(filter(lambda a: a['id'] == artistid, album['artists']))['artistname']

    title_dist = Levenshtein.ratio(found_title, spotifyalbum['Album'])
    artist_dist = Levenshtein.ratio(found_artist, spotifyalbum['Artist'])
    min_ratio = app.config['SPOTIFY_MATCH_MIN_RATIO']

    if title_dist < min_ratio or artist_dist < min_ratio:
        ttl = app.config['CACHE_TTL']['cloudflare']
        await util.SPOTIFY_CACHE.set(spotifyalbum['AlbumSpotifyId'], 0, ttl=ttl)
        await util.SPOTIFY_CACHE.set(spotifyalbum['ArtistSpotifyId'], 0, ttl=ttl)
        return None

    logger.info(f"Mapped Spotify Album: '{spotifyalbum['Album']}' by '{spotifyalbum['Artist']}' to Musicbrainz Album: '{found_title}' ({title_dist}) by '{found_artist}' ({artist_dist})")

    spotifyalbum['AlbumMusicBrainzId'] = albumid
    spotifyalbum['ArtistMusicBrainzId'] = artistid

    return spotifyalbum

@app.route('/spotify/lookup', methods=['POST'])
async def spotify_lookup():
    ids = await request.json

    if ids is None:
        return jsonify(error='Bad Request - expected JSON list of spotify IDs as post body'), 400
    
    results = await util.SPOTIFY_CACHE.multi_get(ids)
    output = [{'spotifyid': ids[x], 'musicbrainzid': results[x][0]} for x in range(len(ids))]

    return jsonify(output)
    
@app.route('/invalidate')
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
            util.ALBUM_CACHE.multi_set([(album, None) for album in albums], ttl=0, timeout=None),
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

@app.route('/spotify/auth')
@no_cache
async def handle_spotify_auth_redirect():
    code = request.args.get('code', '')
    state = request.args.get('state', '')

    if not code:
        abort(400, 'No auth code provided')

    if not state:
        abort(400, 'No state provided')

    if not state.endswith('/oauth.html'):
        abort(400, 'Illegal state value')

    spotify_provider = provider.get_providers_implementing(provider.SpotifyAuthMixin)[0]

    try:
        access_token, expires_in, refresh_token = await spotify_provider.get_token(code)
        newurl = f"{state}?access_token={access_token}&expires_in={expires_in}&refresh_token={refresh_token}";

        return redirect(newurl, 302)
    except aiohttp.ClientResponseError as error:
        abort(error.status, f"spotify: {error.message}")

@app.route('/spotify/renew')
@no_cache
async def handle_spotify_token_renew():
    refresh_token = request.args.get('refresh_token', '')

    if not refresh_token:
        abort(400, 'No refresh token provided')

    spotify_provider = provider.get_providers_implementing(provider.SpotifyAuthMixin)[0]

    try:
        json = await spotify_provider.refresh_token(refresh_token)
        return jsonify(json)
    except aiohttp.ClientResponseError as error:
        abort(error.status, error.message)

@app.after_serving
async def run_async_del():
    async_providers = provider.get_providers_implementing(provider.AsyncDel)
    for prov in async_providers:
        await prov._del()
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=config.get_config().HTTP_PORT, use_reloader=True)
