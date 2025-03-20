# Lidarr Metadata Crawler

此模块包含用于爬取音乐元数据的爬虫，使用Scrapy和Playwright来处理动态加载的网站。

## 安装

首先安装所需的Python依赖：

```bash
# 使用pip
pip install -r requirements.txt

# 或者使用Poetry
poetry install
```

然后安装Playwright浏览器：

```bash
# 直接使用Playwright
playwright install --with-deps chromium

# 或者使用我们的命令行工具
python -m crawler.main --install-playwright --spider dummy
```

## 使用方法

### Spotify艺术家爬虫

此爬虫从Spotify搜索页面爬取艺术家信息。由于Spotify是一个动态加载的网站，我们使用Playwright来确保内容完全加载后再进行解析。

```bash
# 使用Poetry运行
poetry run lidarr-metadata-crawler --spider spotify --query "artist name"

# 或直接使用Python
python -m crawler.main --spider spotify --query "artist name"
```

参数说明：

- `--spider`: 指定要运行的爬虫名称
- `--query`: 搜索查询（艺术家名称）
- `--output`: 输出文件路径（可选）
- `--format`: 输出格式，可选json/jsonlines/csv/xml（默认：json）
- `--loglevel`: 日志级别（默认：INFO）
- `--install-playwright`: 安装Playwright浏览器（首次使用时需要）

## 爬虫列表

1. **spotify**: 从Spotify搜索页面爬取艺术家信息
   - 需要参数: `--query` (艺术家名称)

2. **musicbrainz**: MusicBrainz数据爬虫（待实现）

## 开发新爬虫

要创建新的爬虫，请在`crawler/spiders/`目录下创建新的Python文件，继承`scrapy.Spider`类。

对于需要处理JavaScript动态内容的爬虫，请参考`spotify_spider.py`的实现，使用Playwright。 