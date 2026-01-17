# TFMX Notes

## Docker Container Management

### 清理 Text Embeddings Inference 容器

#### 查看所有 TEI 容器

```bash
# 查看所有包含 text-embeddings 的容器
docker ps -a | grep "text-embeddings"
```

示例输出：
```
CONTAINER ID   IMAGE                                                   COMMAND                  CREATED          STATUS                      PORTS                                       NAMES
7b508dbeaf37   ghcr.io/huggingface/text-embeddings-inference:89-1.8   "text-embeddings-rou…"   5 minutes ago    Up 5 minutes (healthy)      0.0.0.0:28881->80/tcp, :::28881->80/tcp     tei--qwen--qwen3-embedding-0_6b--gpu1
989bf9afe567   ghcr.io/huggingface/text-embeddings-inference:86-1.8   "text-embeddings-rou…"   5 minutes ago    Up 5 minutes (healthy)      0.0.0.0:28880->80/tcp, :::28880->80/tcp     tei--qwen--qwen3-embedding-0_6b--gpu0
3fbef92da5e5   ghcr.io/huggingface/text-embeddings-inference:89-1.8   "text-embeddings-rou…"   7 weeks ago      Up 4 weeks                  0.0.0.0:28887->80/tcp, :::28887->80/tcp     Qwen--Qwen3-Embedding-0.6B
8d98664355ec   ghcr.io/huggingface/text-embeddings-inference:1.8      "text-embeddings-rou…"   2 months ago     Exited (0) 2 months ago                                                 BAAI--bge-large-zh-v1.5
907bb4150a76   ghcr.io/huggingface/text-embeddings-inference:1.8      "text-embeddings-rou…"   2 months ago     Exited (0) 7 weeks ago                                                  Alibaba-NLP--gte-multilingual-base
...
```

#### 停止并删除特定容器

```bash
# 停止并删除旧的运行中的容器
docker stop <container_id> && docker rm <container_id>

# 例如：
docker stop 3fbef92da5e5 && docker rm 3fbef92da5e5
```

示例输出：
```
3fbef92da5e5
3fbef92da5e5
```

#### 批量清理所有已停止的容器

```bash
# 清理所有已停止的容器（会回收磁盘空间）
docker container prune -f
```

示例输出：
```
Deleted Containers:
d9b5890117e0dd9b40267999fa49ae655113b5eee0db25f7cb02ef536d99c3f0
18b0f2534a7b1c4fda96aa8c3c718e7117cbb2b0f7d3c61190be4505741dd1a8
8d98664355ec7b063bff92be14927aa6e9a74be4ffa2b203824fae29c7369e00
907bb4150a7604bd395ad997954078f0298f5da804a62b25a36ea9c831dea281
...

Total reclaimed space: 1.252GB
```

#### 验证清理结果

```bash
# 再次查看所有 TEI 容器，确认只剩下需要的容器
docker ps -a | grep "text-embeddings"
```

示例输出（清理后）：
```
7b508dbeaf37   ghcr.io/huggingface/text-embeddings-inference:89-1.8   "text-embeddings-rou…"   5 minutes ago   Up 5 minutes (healthy)   0.0.0.0:28881->80/tcp, :::28881->80/tcp     tei--qwen--qwen3-embedding-0_6b--gpu1
989bf9afe567   ghcr.io/huggingface/text-embeddings-inference:86-1.8   "text-embeddings-rou…"   5 minutes ago   Up 5 minutes (healthy)   0.0.0.0:28880->80/tcp, :::28880->80/tcp     tei--qwen--qwen3-embedding-0_6b--gpu0
```

### 常见问题

#### 多个容器实例占用同一个 GPU

**问题**：使用 `nvidia-smi` 或 `gpustat` 查看时，发现同一个 GPU 上有多个 text-embeddings-router 实例。

**原因**：
- 修改了 `tei_compose.py` 中的项目名生成规则（如将 `.` 替换为 `_`）
- 导致新旧 compose 文件名不同，生成的容器名也不同
- 旧容器仍在运行，新容器也启动了，造成重复

**解决方案**：
1. 使用 `docker ps -a | grep "text-embeddings"` 查看所有容器
2. 停止并删除旧容器：`docker stop <old_container> && docker rm <old_container>`
3. 清理所有已停止的容器：`docker container prune -f`
4. 如果旧的 compose 文件还存在，可以使用：`docker compose -f <old_compose_file> down`

#### Docker Compose name 字段验证错误

**错误信息**：
```
validating xxx.yml: name Does not match pattern '^[a-z0-9][a-z0-9_-]*$'
```

**原因**：Docker Compose 的 `name` 字段只能包含小写字母、数字、下划线和连字符，且必须以字母或数字开头。

**解决方案**：在 `tei_compose.py` 中使用正则表达式替换不合法字符：
```python
import re
project_dash = model_name.replace("/", "--").lower()
project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
self.project_name = project_name or f"tei--{project_dash}"
```
