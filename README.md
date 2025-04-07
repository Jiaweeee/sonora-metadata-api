## 部署说明

1. 确保已安装 Docker 和 Docker Compose

2. 克隆项目并进入项目目录：
```shell
git clone <repository-url>
cd sonora-metadata-api
```

3. 启动所有服务：
```shell
docker-compose up -d
```

这将启动以下服务：
- Redis 服务（端口：6379）
- PostgreSQL 数据库（端口：5432）
- Metadata API 服务（端口：5001）

4. 检查服务状态：
```shell
docker-compose ps
```

5. 查看服务日志：
```shell
docker-compose logs -f metadata-api
```

# 性能监控

项目使用 Prometheus 和 Grafana 进行API性能监控。

## 监控指标

- **请求耗时**: 通过 Histogram 类型指标跟踪每个接口的响应时间，包括不同百分位数（50%、95%）
- **请求量**: 记录1小时内各接口的请求总数
- **QPS (每秒查询率)**: 实时监控各接口的每秒请求数
- **请求成功率**: 监控API接口的成功率（2xx状态码请求占总请求比例）

## 启动监控服务

```bash
# 启动主要服务（包括 Prometheus）
docker-compose up -d

# 启动 Grafana（使用开发配置）
docker-compose -f docker-compose.dev.yml up -d
```

## 停止监控服务

```bash
# 停止 Grafana
docker-compose -f docker-compose.dev.yml down

# 停止主要服务（包括 Prometheus）
docker-compose down
```

## 访问监控界面

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (用户名/密码: sonora/sonoramusic)

Grafana已预配置了一个用于监控API性能指标的仪表盘。

## 自定义监控

可以通过修改以下文件自定义监控:

- `prometheus.yml`: 修改Prometheus抓取配置
- `grafana/dashboards/sonora_metadata_api_dashboard.json`: 自定义Grafana仪表盘
