"""
Scrapy settings for the crawler module.
"""

BOT_NAME = 'lidarrmetadata-crawler'

SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 16

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 1.0
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 8
CONCURRENT_REQUESTS_PER_IP = 8

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Configure item pipelines
ITEM_PIPELINES = {
    'crawler.pipelines.ArtistImageCachePipeline': 300,
}

# Enable and configure the AutoThrottle extension
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = False

# Set the User-Agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Configure Playwright settings
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Configure a download timeout
DOWNLOAD_TIMEOUT = 60

# Configure Playwright specific options
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "timeout": 60000,  # 60 seconds
}

PLAYWRIGHT_BROWSER_TYPE = "chromium"

# Configure default request meta for Playwright
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 60000  # 60 seconds

# Playwright Request Meta Keys
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"

# Configure logging
LOG_LEVEL = 'INFO'

# Configure retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429] 