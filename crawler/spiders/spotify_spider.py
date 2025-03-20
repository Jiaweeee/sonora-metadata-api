"""
Spider for crawling Spotify artist data using Playwright.
"""

import scrapy
import logging
from urllib.parse import quote
from scrapy.http import Request
from crawler.items import ArtistItem
from scrapy_playwright.page import PageMethod


class SpotifySpider(scrapy.Spider):
    """
    Spider for crawling Spotify artist data.
    
    This spider uses Playwright to handle dynamic content loading.
    It searches for artists by name and extracts their information.
    """
    
    name = 'spotify'
    allowed_domains = ['open.spotify.com']
    
    def __init__(self, query=None, *args, **kwargs):
        """
        Initialize the spider.
        
        Args:
            query: The search term to look for artists
        """
        super(SpotifySpider, self).__init__(*args, **kwargs)
        self.query = query
        
    def start_requests(self):
        """
        Generate the initial requests.
        
        If no query is provided, log an error.
        """
        if not self.query:
            self.logger.error("No search query provided. Use -a query='artist name' when starting the spider.")
            return
        
        # URL encode the query
        encoded_query = quote(self.query)
        url = f"https://open.spotify.com/search/{encoded_query}/artists"
        
        # Use Playwright for this request to handle JavaScript rendering
        yield Request(
            url=url,
            callback=self.parse,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_methods": [
                    # Wait until network is idle - good indicator that dynamic content has loaded
                    PageMethod("wait_for_load_state", "networkidle")
                ],
                "errback": self.errback,
            }
        )
        
    async def parse(self, response):
        """
        Parse the response and extract artist data.
        
        Args:
            response: The response object containing the page and its content
        """
        page = response.meta["playwright_page"]
        
        try:
            # Find the specific image element
            image_element = await page.query_selector('//*[@id="searchPage"]/div/div/div/div[1]/span[1]/div/div/div[3]/div[1]/div/img')
            
            if image_element:
                # Get the src attribute
                src = await image_element.get_attribute('src')
                self.logger.info(f"Found image with src: {src}")
                print("\n===== IMAGE SRC =====")
                print(src)
                print("=====================\n")
            else:
                self.logger.error("Could not find the image element")
            
            # Log basic page information
            content = await page.content()
            self.logger.info(f"Page content loaded successfully (length: {len(content)})")
            
        finally:
            # Always close the page to avoid memory leaks
            await page.close()

    async def errback(self, failure):
        """
        Handle errors that occur during the request.
        
        Args:
            failure: The failure information
        """
        page = failure.request.meta.get("playwright_page")
        if page:
            await page.close()
            
        self.logger.error(f"Error: {failure}") 