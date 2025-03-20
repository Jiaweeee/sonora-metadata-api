"""
Pipeline for processing MusicBrainz data.
"""

from itemadapter import ItemAdapter


class MusicbrainzPipeline:
    """Process MusicBrainz items and store them in the database."""
    
    def process_item(self, item, spider):
        """
        Process each scraped item.
        
        Args:
            item: The scraped item
            spider: The spider that scraped the item
            
        Returns:
            The processed item
        """
        # Process and store the item
        # This is an empty implementation to be filled later
        return item 