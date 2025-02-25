## 开发环境配置

项目依赖可以通过以下两种方式管理：

1. 使用 [poetry](https://python-poetry.org) 管理依赖：
```shell
poetry install --with=dev
poetry shell # 或者使用 poetry run ...
```

2. 使用 conda 和 pip 管理依赖（推荐）：
```shell
conda create -n lidarr-metadata python=3.9
conda activate lidarr-metadata
pip install -r requirements.txt
```

## 运行服务
在启动服务之前，需要先运行必要的依赖服务：
```shell
docker-compose up -d
```

然后执行以下命令启动服务：
```shell
python lidarrmetadata/server.py
```