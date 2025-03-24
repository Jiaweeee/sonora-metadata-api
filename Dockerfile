FROM python:3.9-slim

ARG UID=1000
ARG COMMIT_HASH=''
ARG GIT_BRANCH=''

ENV COMMIT_HASH $COMMIT_HASH
ENV GIT_BRANCH $GIT_BRANCH

WORKDIR /metadata
COPY . /metadata

# 使用pip安装依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev && \
    pip --disable-pip-version-check --no-cache-dir install -r requirements.txt && \
    # 将当前目录安装为Python包，这样就可以导入lidarrmetadata模块
    pip install -e . && \
    apt-get purge -y --auto-remove build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --system -u $UID metadata

USER metadata

# 修改入口点为直接运行Python脚本
# 添加PYTHONPATH环境变量，确保Python能找到模块
ENV PYTHONPATH=/metadata

ENTRYPOINT ["python", "lidarrmetadata/server.py"]
