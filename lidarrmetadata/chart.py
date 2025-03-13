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

async def _parse_itunes_chart(URL, count):
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
            json = await response.json()
            results = filter(lambda r: r.get('kind', '') == 'albums', json['feed']['results'])
            search_provider = provider.get_providers_implementing(provider.ReleaseNameSearchMixin)[0]
            search_results = []
            for result in results:
                search_result = await search_provider.search_album_name(result['name'], artist_name=result['artistName'], limit=1)
                if search_result:
                    search_result = search_result[0]
                    search_results.append(await _parse_album_search_result(search_result))

                    if len(search_results) == count:
                        break
            return search_results

@cached(ttl = 60 * 60 * 24, alias='default')
async def get_apple_music_top_albums_chart(count=10):
    """
    Gets and parses itunes chart
    :param count: Number of results to return. Defaults to 10
    :return: Chart response for itunes
    """
    URL = 'https://rss.applemarketingtools.com/api/v2/us/music/most-played/{count}/albums.json'.format(
        count=4 * count)
    return await _parse_itunes_chart(URL, count)

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
        logger.error(f"Error fetching Billboard chart {chart_name}: {str(e)}")
        return []

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
                    search_result = await search_provider.search_artist_name(result.artist, limit=1)
                    if search_result:
                        return {'ArtistName': result.artist, 'ArtistId': search_result[0]['id']}
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
        logger.error(f"Error fetching Billboard chart {chart_name}: {str(e)}")
        return []

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
                        return search_result[0]
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
        logger.error(f"Error fetching Billboard chart {chart_name}: {str(e)}")
        raise e

# @cached(ttl = 60 * 60 * 24, alias='default')
# async def get_billboard_100_artists_chart(count=10):
#     """
#     Gets billboard top 100 albums
#     :param count: Number of results to return. Defaults to 10
#     :return: Chart response for artist-100
#     """
#     results = billboard.ChartData('artist-100')

#     search_provider = provider.get_providers_implementing(provider.ArtistNameSearchMixin)[0]

#     search_results = []
#     for result in results:
#         artist_search = await search_provider.search_artist_name(result.artist, limit=1)
#         if artist_search:
#             search_results.append({'ArtistName': result.artist, 'ArtistId': artist_search[0]['id']})

#         if len(search_results) == count:
#             break

#     return search_results

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
}