# 脚本工具

本目录包含了一些用于维护和管理 LidarrAPI.Metadata 的实用脚本。

## 缓存同步工具

### sync_cache.py

这个脚本用于将本地的 SPOTIFY_CACHE 和 ARTIST_CACHE 数据同步到远程服务器。它支持批量处理、错误重试和进度显示功能。

#### 前提条件

- Python 3.7 或更高版本
- 已安装项目依赖（在项目根目录运行 `pip install -r requirements.txt`）
- 本地和远程都有运行的 PostgreSQL 数据库
- 安装 tqdm 库: `pip install tqdm`

#### 使用方法

```bash
python sync_cache.py --remote-host <远程主机> --remote-port <远程端口> --remote-user <用户名> --remote-password <密码> --remote-db <数据库名> [选项]
```

#### 参数说明

##### 必要参数
- `--remote-host`: 远程数据库主机地址（必需）
- `--remote-port`: 远程数据库端口（默认：5432）
- `--remote-user`: 远程数据库用户名（必需）
- `--remote-password`: 远程数据库密码（必需）
- `--remote-db`: 远程数据库名称（必需）

##### 性能和执行控制
- `--batch-size`: 每批处理的记录数（默认：10000）
- `--max-retries`: 操作失败时的最大重试次数（默认：3）
- `--retry-interval`: 重试之间的间隔秒数（默认：5）
- `--ignore-errors`: 忽略错误并继续处理
- `--max-loop-count`: 最大循环次数，防止无限循环（默认：1000）

##### 缓存选择（互斥参数）
- `--all`: 同步所有缓存（默认行为）
- `--spotify`: 只同步 Spotify 缓存
- `--artist`: 只同步 Artist 缓存

##### 输出控制（互斥参数）
- `--verbose`, `-v`: 显示详细的调试信息
- `--quiet`, `-q`: 只显示警告和错误信息

#### 示例

```bash
# 基本用法 - 同步所有缓存
python sync_cache.py --remote-host 192.168.1.100 --remote-port 5432 --remote-user postgres --remote-password mypassword --remote-db lm_cache_db

# 只同步 Spotify 缓存
python sync_cache.py --remote-host 192.168.1.100 --remote-user postgres --remote-password mypassword --remote-db lm_cache_db --spotify

# 调整批处理大小和重试参数
python sync_cache.py --remote-host 192.168.1.100 --remote-user postgres --remote-password mypassword --remote-db lm_cache_db --batch-size 500 --max-retries 5 --retry-interval 10

# 静默模式，只显示警告和错误
python sync_cache.py --remote-host 192.168.1.100 --remote-user postgres --remote-password mypassword --remote-db lm_cache_db --quiet

# 忽略错误并继续处理
python sync_cache.py --remote-host 192.168.1.100 --remote-user postgres --remote-password mypassword --remote-db lm_cache_db --ignore-errors
```

#### 故障排除

如果遇到同步卡住或无进展的情况：

1. 使用 `--verbose` 选项获取详细日志
2. 使用 `--ignore-errors` 忽略错误继续处理
3. 尝试只同步特定缓存（使用 `--spotify` 或 `--artist`）
4. 调整 `--batch-size` 参数，尝试较小的值如 100 或 1000
5. 使用 `--max-loop-count` 设置较小的值，如 100，防止无限循环

#### 工作原理

1. 脚本首先连接到本地和远程数据库
2. 尝试获取源数据库中的记录总数以显示进度条
3. 然后分批获取 SPOTIFY_CACHE 和 ARTIST_CACHE 中的所有键
4. 从本地数据库获取每个键对应的值
5. 将这些键值对批量写入远程数据库
6. 显示进度和统计信息

#### 注意事项

- 同步过程可能需要较长时间，取决于数据量大小
- 如果网络连接不稳定，脚本会自动重试失败的操作
- 如需中断同步过程，可按 Ctrl+C
- 同步不会删除远程数据库中已存在但本地不存在的数据
- 默认情况下，同步会保留源数据的过期时间设置 