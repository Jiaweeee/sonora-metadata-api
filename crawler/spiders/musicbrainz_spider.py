"""
Spider for crawling MusicBrainz data.
"""

import scrapy
from crawler.items import ArtistItem, AlbumItem, TrackItem


class MusicbrainzSpider(scrapy.Spider):
    """Spider for crawling MusicBrainz data."""
    
    name = 'musicbrainz'
    allowed_domains = ['musicbrainz.org']
    
    def __init__(self, *args, **kwargs):
        """Initialize the spider."""
        super(MusicbrainzSpider, self).__init__(*args, **kwargs)
        self.start_urls = ['https://musicbrainz.org/']
        
    def parse(self, response):
        """
        Parse the response and extract data.
        
        Args:
            response: The response object
            
        Yields:
            Items extracted from the response
        """
        # This is an empty implementation to be filled later
        pass 