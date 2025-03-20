"""
Command-line interface for running crawlers.
"""

import os
import sys
import argparse
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def main():
    """Main entry point for the crawler."""
    parser = argparse.ArgumentParser(description='Lidarr Metadata Crawler')
    
    # Add arguments
    parser.add_argument('--spider', type=str, required=True,
                        help='Name of the spider to run')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (optional)')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'jsonlines', 'csv', 'xml'],
                        help='Output format (default: json)')
    parser.add_argument('--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set up the crawler process
    settings = get_project_settings()
    
    # Override settings from command line
    settings.set('LOG_LEVEL', args.loglevel)
    
    # Configure output
    if args.output:
        feed_uri = args.output
        if not os.path.isabs(feed_uri):
            feed_uri = os.path.abspath(feed_uri)
        
        settings.set('FEEDS', {
            feed_uri: {
                'format': args.format,
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'indent': 4,
            },
        })
    
    # Start the crawler
    process = CrawlerProcess(settings)
    process.crawl(args.spider)
    process.start()
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 