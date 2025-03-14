"""
Code for getting and parsing music charts (Billboard, itunes, etc)
"""
import asyncio
import billboard
import pylast
import aiohttp
from aiocache import cached
import logging

from lidarrmetadata import api
from lidarrmetadata import config
from lidarrmetadata import provider
from lidarrmetadata import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ChartException(Exception):
    """Raised when there is an error fetching chart data"""
    def __init__(self, message="Failed to fetch chart data"):
        super().__init__(message)
        self.message = message

async def get_apple_music_top_songs_chart():
    """
    Gets and parses itunes chart
    :param count: Number of results to return
    :return: Chart response for itunes
    :raises: ChartException if there is an error fetching or parsing the chart data
    """
    URL = 'https://rss.applemarketingtools.com/api/v2/us/music/most-played/50/songs.json'
    try:
        async with aiohttp.ClientSession() as session:
            # Allow redirects to handle the URL change
            async with session.get(URL, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=True) as response:
                if response.status != 200:
                    error_msg = f"Error fetching Apple Music top songs chart: HTTP {response.status}"
                    logger.error(f"{error_msg}, {await response.text()[:100]}")
                    raise ChartException(error_msg)
                
                try:
                    json = await response.json()
                except Exception as e:
                    error_msg = f"Error parsing JSON from Apple Music: {str(e)}"
                    logger.error(f"{error_msg}, Content-Type: {response.headers.get('Content-Type')}")
                    raise ChartException(error_msg)
                
                results = filter(lambda r: r.get('kind', '') == 'songs', json['feed']['results'])
                
                # Create a semaphore to limit concurrent tasks
                semaphore = asyncio.Semaphore(10)
                
                async def process_song(result):
                    try:
                        async with semaphore:
                            search_result = await api.get_track_search_results(query=result['name'], limit=1, artist_name=result['artistName'])
                            if search_result:
                                track = search_result[0]
                                # Use the artworkUrl100 from the apple music chart if track has no images
                                if not track.get('images') and result.get('artworkUrl100'):
                                    track['images'] = {
                                        'small': result.get('artworkUrl100')
                                    }
                                return track
                            return None
                    except Exception as e:
                        logger.error(f"Error processing song {result['name']}: {str(e)}")
                        return None
                
                # Create tasks for all songs with semaphore
                tasks = []
                for result in results:
                    tasks.append(process_song(result))
                
                # Execute all tasks with concurrency control
                search_results = await asyncio.gather(*tasks)
                
                # Filter out None results and limit to count
                valid_results = [result for result in search_results if result is not None]
                
                if not valid_results:
                    raise ChartException("No valid songs found in Apple Music chart")
                    
                return valid_results
    except ChartException:
        # Re-raise ChartException to be handled by the caller
        raise
    except Exception as e:
        error_msg = f"Error fetching Apple Music top songs chart: {str(e)}"
        logger.error(error_msg)
        raise ChartException(error_msg)

async def get_billboard_albums_chart(chart_name=None):
    """
    Gets billboard albums chart
    :param chart_name: Name of the billboard chart.
    :return: Chart response for the specified billboard chart
    """
    try:
        results = billboard.ChartData(chart_name)
        
        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(20)  # 限制最多10个并发任务
        
        async def process_album(result):
            try:
                async with semaphore:
                    # search_result = await search_provider.search_release_name(result.title, artist_name=result.artist)
                    search_result = await api.get_release_search_results(query=result.title, limit=1, artist_name=result.artist)
                    if search_result:
                        release = search_result[0]
                        if 'score' in release:
                            del release['score']
                        return release
                    return None
            except Exception as e:
                logger.error(f"Error processing album {result.title}: {str(e)}")
                return None
        
        # Create tasks for all albums with semaphore
        tasks = [process_album(result) for result in results]
        
        # Execute all tasks with concurrency control
        search_results = await asyncio.gather(*tasks)
        
        # Filter out None results and return
        return [result for result in search_results if result is not None]
        
    except Exception as e:
        error_msg = f"Error fetching Billboard chart {chart_name}: {str(e)}"
        logger.error(error_msg)
        raise ChartException(error_msg)

async def get_billboard_artists_chart(chart_name=None):
    """
    Gets billboard artists chart
    :param chart_name: Name of the billboard chart.
    :return: Chart response for the specified billboard chart
    """
    try:
        results = billboard.ChartData(chart_name)
        search_provider = provider.get_providers_implementing(provider.ArtistNameSearchMixin)[0]
        
        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(10)
        
        async def process_artist(result):
            try:
                async with semaphore:
                    search_result = await api.get_artist_search_results(query=result.artist, limit=1)
                    if search_result:
                        artist = search_result[0]
                        if 'score' in artist:
                            del artist['score']
                        return artist
                    return None
            except Exception as e:
                logger.error(f"Error processing artist {result.artist}: {str(e)}")
                return None
        
        # Create tasks for all artists with semaphore
        tasks = [process_artist(result) for result in results]
        
        # Execute all tasks with concurrency control
        search_results = await asyncio.gather(*tasks)
        
        # Filter out None results and return
        return [result for result in search_results if result is not None]
        
    except Exception as e:
        error_msg = f"Error fetching Billboard chart {chart_name}: {str(e)}"
        logger.error(error_msg)
        raise ChartException(error_msg)

async def get_billboard_songs_chart(chart_name=None):
    """
    Gets billboard songs chart
    :param chart_name: Name of the billboard chart.
    :return: Chart response for the specified billboard chart
    """
    try:
        results = billboard.ChartData(chart_name)
        
        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(10)
        
        async def process_song(result):
            try:
                async with semaphore:
                    search_result = await api.get_track_search_results(query=result.title, limit=1, artist_name=result.artist)
                    if search_result:
                        track = search_result[0]
                        if 'score' in track:
                            del track['score']
                        return track
                    return None
            except Exception as e:
                logger.error(f"Error processing song {result.title}: {str(e)}")
                return None
        
        # Create tasks for all songs with semaphore
        tasks = [process_song(result) for result in results]
        
        # Execute all tasks with concurrency control
        search_results = await asyncio.gather(*tasks)
        
        # Filter out None results and return
        return [result for result in search_results if result is not None]
        
    except Exception as e:
        error_msg = f"Error fetching Billboard chart {chart_name}: {str(e)}"
        logger.error(error_msg)
        raise ChartException(error_msg)

@cached(ttl = 60 * 60 * 24, alias='default')
async def get_lastfm_album_chart(count=10, user=None):
    """
    Gets and parses lastfm chart
    :param count: Number of results to return. Defaults to 10
    :return: Parsed chart
    """
    client = pylast.LastFMNetwork(api_key=config.get_config().LASTFM_KEY, api_secret=config.get_config().LASTFM_SECRET)
    client.enable_rate_limit()
    client.disable_caching()

    if user:
        user = client.get_user(user[0])
        lastfm_albums = user.get_top_albums(limit = count * 2)
    else:
        tag = client.get_tag('all')
        lastfm_albums = tag.get_top_albums(limit = count * 2)

    album_provider = provider.get_providers_implementing(provider.ReleaseGroupByIdMixin)[0]
    albums = []
    for lastfm_album in lastfm_albums:
        # Try to stop lastfm from erroring out
        await asyncio.sleep(1)
        
        try:
            # TODO Figure out a cleaner way to do this
            rgid = await album_provider.map_query(
                'SELECT release_group.gid '
                'FROM release '
                'JOIN release_group ON release_group.id = release.release_group '
                'WHERE release.gid = $1 '
                'LIMIT 1',
                lastfm_album.item.get_mbid()
            )

            if rgid:
                search_result = await _parse_album_search_result({'Id': rgid[0]['gid']})
                if search_result:
                    albums.append(search_result)

                    if len(albums) == count:
                        break
        except:
            pass

    if len(albums) > count:
        albums = albums[:count]

    return albums

@cached(ttl = 60 * 60 * 24, alias='default')
async def get_lastfm_artist_chart(count=10, user=None):
    """
    Gets and parses lastfm chart
    :param count: Number of results to return. Defaults to 10
    :return: Parsed chart
    """
    client = pylast.LastFMNetwork(api_key=config.get_config().LASTFM_KEY, api_secret=config.get_config().LASTFM_SECRET)
    client.enable_rate_limit()
    client.disable_caching()

    if user:
        user = client.get_user(user[0])
        lastfm_artists = user.get_top_artists(limit = count * 2)
    else:
        lastfm_artists = client.get_top_artists(limit = count * 2)

    artists = []
    search_provider = provider.get_providers_implementing(provider.ArtistNameSearchMixin)[0]
    for lastfm_artist in lastfm_artists:
        # Try to stop lastfm from erroring out
        await asyncio.sleep(1)

        artist = {'ArtistName': lastfm_artist.item.name, 'ArtistId': lastfm_artist.item.get_mbid()}

        if not all(artist.values()):
            results = await search_provider.search_artist_name(artist['ArtistName'], limit=1)
            if results:
                results = results[0]
                artist = {'ArtistName': results['ArtistName'], 'ArtistId': results['Id']}

        if all(artist.values()):
            artists.append(artist)

    if len(artists) > count:
        artists = artists[:count]

    return artists

async def _parse_album_search_result(search_result):
    album = await api.get_release_group_info_basic(search_result['Id'])
    album = album[0]
    return {
        'AlbumId': album['id'],
        'AlbumTitle': album['title'],
        'ArtistId': album['artistid'],
        'ReleaseDate': album['releasedate']
    }

charts = {
    'billboard-200-albums': lambda: get_billboard_albums_chart('billboard-200'),
    'billboard-tastemaker-albums': lambda: get_billboard_albums_chart('tastemaker-albums'),
    'billboard-radio-songs': lambda: get_billboard_songs_chart('radio-songs'),
    'billboard-streaming-songs': lambda: get_billboard_songs_chart('streaming-songs'),
    'billboard-emerging-artists': lambda: get_billboard_artists_chart('emerging-artists'),
    'billboard-independent-albums': lambda: get_billboard_albums_chart('independent-albums'),
    'apple-music-top-songs': lambda: get_apple_music_top_songs_chart(),
}