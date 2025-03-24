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
