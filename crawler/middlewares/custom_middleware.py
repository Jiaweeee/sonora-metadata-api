"""
Custom middlewares for the crawler.
"""

from scrapy import signals


class CustomMiddleware:
    """Custom middleware for handling requests and responses."""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Initialize the middleware with the crawler settings."""
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware
        
    def process_request(self, request, spider):
        """Process a request before it is sent to the destination."""
        return None
        
    def process_response(self, request, response, spider):
        """Process a response before it is returned to the spider."""
        return response
        
    def process_exception(self, request, exception, spider):
        """Handle exceptions raised during request processing."""
        pass
        
    def spider_opened(self, spider):
        """Called when the spider is opened."""
        spider.logger.info('Spider opened: %s' % spider.name) 