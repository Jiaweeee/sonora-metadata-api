# Sonora Metadata API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

A comprehensive music metadata API that provides detailed information about artists, releases, tracks, and music discovery features. Originally designed for integration with Lidarr, this API aggregates data from multiple sources including MusicBrainz, Spotify, and other music metadata providers.

## ✨ Features

### Core Metadata API
- **Artist Information**: Comprehensive artist profiles with biographies, images, and discographies
- **Release Data**: Detailed album and release information with cover art
- **Track Metadata**: Individual track information and metadata
- **Search Functionality**: Fast search across artists, releases, and tracks
- **Smart Caching**: Multi-layer caching with Redis and PostgreSQL for optimal performance

### Music Discovery
- **New Releases**: Recently released albums and tracks
- **Hot Songs**: Trending and popular tracks
- **Music Charts**: Multiple curated charts including:
  - Taste Picks
  - On Air
  - Stream Hits  
  - Indie Gems
  - Rising Stars

### Data Crawling
- **Spotify Crawler**: Automated artist image and metadata collection
- **Extensible Framework**: Built on Scrapy with Playwright for dynamic content
- **Background Processing**: Asynchronous data collection and updates

### Monitoring & Observability
- **Prometheus Metrics**: Comprehensive API performance monitoring
- **Grafana Dashboard**: Pre-configured visualization for metrics
- **Request Tracking**: Response times, request volumes, and success rates
- **Health Checks**: Service status and data freshness monitoring

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for development)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sonora-metadata-api
```

2. **Start all services**
```bash
docker-compose up -d
```

This will start:
- Redis cache server (port 6379)
- PostgreSQL database (port 5432)
- Metadata API service (port 5001)
- Prometheus monitoring (port 9090)

3. **Verify installation**
```bash
# Check service status
docker-compose ps

# View API logs
docker-compose logs -f metadata-api

# Test the API
curl http://localhost:5001/api/
```

### With Monitoring (Development)

To enable Grafana monitoring:

```bash
# Start main services
docker-compose up -d

# Start Grafana
docker-compose -f docker-compose.dev.yml up -d
```

Access the monitoring interfaces:
- **API**: http://localhost:5001/api/
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (username: `sonora`, password: `sonoramusic`)

## 📖 API Documentation

### Base URL
```
http://localhost:5001/api/
```

### Core Endpoints

#### Artist Information
```http
GET /api/artist/{mbid}
```
Returns comprehensive artist information including biography, images, and discography.

#### Release Information  
```http
GET /api/release/{mbid}
```
Get detailed release/album information with track listings and cover art.

#### Track Information
```http
GET /api/track/{mbid}
```
Retrieve individual track metadata and information.

#### Search
```http
GET /api/search?query={search_term}&type={artist|release|track}
```
Search across all entity types with fuzzy matching.

### Discovery Endpoints

#### New Releases
```http
GET /api/discover/new-releases
```
Recently released albums and tracks.

#### Hot Songs
```http
GET /api/discover/hot-songs  
```
Trending and popular tracks.

#### Charts
```http
GET /api/discover/charts
```
List all available music charts.

```http
GET /api/discover/chart/{chart_id}
```
Get specific chart content (taste-picks, on-air, stream-hits, indie-gems, rising-stars).

### Metrics
```http
GET /api/metrics
```
Prometheus-formatted metrics for monitoring.

## 🔧 Development

### Local Development Setup

1. **Install dependencies**
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

2. **Set up environment**
```bash
# Copy environment template
cp postgres.env.example postgres.env
# Edit postgres.env with your database configuration
```

3. **Run database migrations**
```bash
# Run SQL setup scripts
./admin/create-amqp-extension
./admin/setup-amqp-triggers
```

4. **Start development server**
```bash
# Using Poetry
poetry run lidarr-metadata-server

# Or using Python
python -m lidarrmetadata.server
```

### Crawler Development

The project includes a Scrapy-based crawler for collecting metadata:

```bash
# Install Playwright browsers
python -m crawler.main --install-playwright --spider dummy

# Run Spotify artist crawler
poetry run lidarr-metadata-crawler --spider spotify --query "artist name"

# Run with custom output
python -m crawler.main --spider spotify --query "artist name" --output results.json
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test files
poetry run pytest tests/api_test.py
poetry run pytest tests/provider_test.py

# Run with coverage
poetry run pytest --cov=lidarrmetadata
```

## 🏗️ Architecture

### Components

- **API Layer** (`lidarrmetadata/`): Quart-based async web API
- **Data Providers** (`lidarrmetadata/provider.py`): Abstracted data source interfaces
- **Caching** (`lidarrmetadata/cache.py`): Multi-tier caching strategy
- **Crawler** (`crawler/`): Scrapy-based metadata collection
- **Charts** (`lidarrmetadata/chart.py`): Music discovery and ranking algorithms
- **Monitoring** (`lidarrmetadata/metrics.py`): Prometheus metrics collection

### Data Flow

1. **Request** → API endpoint receives request
2. **Cache Check** → Redis/PostgreSQL cache lookup  
3. **Provider Query** → Database or external API call
4. **Data Processing** → Enrichment and formatting
5. **Cache Store** → Save processed data
6. **Response** → Return formatted JSON

### External Dependencies

- **MusicBrainz**: Primary metadata source
- **Spotify**: Artist images and additional metadata
- **Last.fm**: Charts and popularity data
- **Wikipedia/Wikidata**: Artist biographies and overviews

## 📊 Monitoring

The API includes comprehensive monitoring capabilities:

### Metrics Collected
- Request duration histograms (50th, 95th percentiles)
- Request count by endpoint and status code
- QPS (Queries Per Second) tracking
- Success rate monitoring
- Cache hit/miss ratios
- Exception tracking

### Custom Monitoring

Customize monitoring by editing:
- `prometheus.yml`: Prometheus scraping configuration
- `grafana/dashboards/`: Grafana dashboard definitions
- `lidarrmetadata/metrics.py`: Custom metric definitions

## 🔒 Production Deployment

### Environment Variables

Set these environment variables for production:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname
REDIS_URL=redis://host:port/db

# API Configuration  
DEBUG=false
CACHE_TTL_CLOUDFLARE=3600
CACHE_TTL_LOCAL=300

# External Services
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
LASTFM_API_KEY=your_lastfm_key

# Monitoring
PROMETHEUS_ENABLED=true
```

### Production Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Performance Tuning

- **Database**: Ensure proper indexing (see `lidarrmetadata/sql/CreateIndices.sql`)
- **Redis**: Configure appropriate memory limits and eviction policies
- **API**: Adjust worker count based on traffic patterns
- **Caching**: Tune TTL values based on data update frequency

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MusicBrainz** for providing comprehensive music metadata
- **Lidarr** community for the original inspiration
- **Scrapy** and **Playwright** for robust web scraping capabilities
- **Prometheus** and **Grafana** for monitoring infrastructure

## 📞 Support

- **Issues**: Report bugs and feature requests via [GitHub Issues](https://github.com/your-org/sonora-metadata-api/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/your-org/sonora-metadata-api/discussions)
- **Documentation**: Full API documentation available at [your-docs-url](https://docs.your-domain.com)

---

**Made with ❤️ for the music community**
