"""
Item definitions for the crawler.

Define here the models for your scraped items.
"""

import scrapy


class ArtistItem(scrapy.Item):
    """Item for storing artist information."""
    id = scrapy.Field()  # MusicBrainz ID
    name = scrapy.Field()
    sort_name = scrapy.Field()
    disambiguation = scrapy.Field()
    overview = scrapy.Field()
    images = scrapy.Field()
    links = scrapy.Field()
    popularity = scrapy.Field()


class AlbumItem(scrapy.Item):
    """Item for storing album information."""
    id = scrapy.Field()  # MusicBrainz Release Group ID
    title = scrapy.Field()
    artist_id = scrapy.Field()
    type = scrapy.Field()
    secondary_types = scrapy.Field()
    disambiguation = scrapy.Field()
    release_date = scrapy.Field()
    images = scrapy.Field()
    links = scrapy.Field()
    popularity = scrapy.Field()


class TrackItem(scrapy.Item):
    """Item for storing track information."""
    id = scrapy.Field()  # MusicBrainz Recording ID
    title = scrapy.Field()
    artist_id = scrapy.Field()
    album_id = scrapy.Field()
    duration = scrapy.Field()
    track_number = scrapy.Field()
    disc_number = scrapy.Field()
    release_date = scrapy.Field() 