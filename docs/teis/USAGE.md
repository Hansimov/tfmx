# TEI 使用说明

## 统一入口

`tei` 是唯一的公开命令入口。建议先准备一组环境变量，方便在不同机器间切换：

```bash
export TEI_MACHINE_A_URL="http://$TEI_HOST_A:28800"
export TEI_MACHINE_B_URL="http://$TEI_HOST_B:28800"
export TEI_MACHINE_URL="$TEI_MACHINE_A_URL"
export TEI_BACKEND_A_URL="http://$TEI_BACKEND_HOST_A:28880"
export TEI_BACKEND_B_URL="http://$TEI_BACKEND_HOST_B:28881"
export TEI_BACKENDS="$TEI_BACKEND_A_URL,$TEI_BACKEND_B_URL"
export TEI_EPS="$TEI_MACHINE_A_URL,$TEI_MACHINE_B_URL"
```

## `tei compose`

```bash
export MODEL="Qwen/Qwen3-Embedding-0.6B"
export TEI_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"

tei compose setup -m "$MODEL"
tei compose up
tei compose up -m "$MODEL"
tei compose up -g "0,1"
tei compose up --gpu-configs "0,2"
tei compose up --gpu-configs "0:Qwen/Qwen3-Embedding-0.6B,2:Alibaba-NLP/gte-multilingual-base"
tei compose up --mount-mode manual --proxy "$TEI_PROXY_URL"
tei compose ps
tei compose health
tei compose logs -f
tei compose down
```

### 常用参数

- `-g/--gpus`：只选择某几张 GPU，但模型保持一致
- `--gpu-configs`：使用 `GPU[:MODEL],...` 语法显式指定每张 GPU 的模型；例如 `"0,2"` 表示 GPU0/GPU2 都使用默认模型
- `--mount-mode manual`：手动挂载 `/dev/nvidia*` 与驱动库，适合 NVML / runtime 异常时兜底
- `--proxy`：为模型下载设置 HTTP/HTTPS 代理
- `-p/--port`：修改后端起始端口，默认从 `28880` 开始
- `-j/--project-name`：显式指定 compose 项目名

### 行为说明

- 默认会先检测当前健康 GPU，再为这些 GPU 启动后端
- 如果你用 `--gpu-configs` 配置了多个不同模型，而又没有手动指定 `--project-name`，compose 项目名会自动使用 `tei-multi`
- `tei compose setup` 适合在首次启动前跑一次，减少首次容器启动时的补文件等待

## `tei machine`

```bash
tei machine run
tei machine run --auto-start
tei machine run --auto-start --perf-track --on-conflict replace
tei machine run --auto-start --compose-gpus "0,2"
tei machine run --auto-start --compose-gpu-configs "0,2"
tei machine run --auto-start --compose-gpu-configs "0:Qwen/Qwen3-Embedding-0.6B,2:Alibaba-NLP/gte-multilingual-base"
tei machine run -e "$TEI_BACKENDS"
tei machine discover
tei machine health
```

### 常用参数

- `-p/--port`：machine 监听端口，默认 `28800`
- `-e/--endpoints`：跳过容器自动发现，直接使用给定后端地址
- `-b/--batch-size`：每个实例允许的最大批量，默认 `300`
- `-m/--micro-batch-size`：自适应流水线调度用的小批量探测大小，默认 `100`
- `--perf-track`：打印更细的 pipeline 与 LSH 性能日志
- `--no-gpu-lsh`：关闭 GPU LSH，加速逻辑改为 CPU
- `--auto-start`：没有后端时自动调用 compose 拉起后端
- `--compose-gpus`：限制 auto-start 只用哪些 GPU
- `--compose-gpu-configs`：用 `GPU[:MODEL],...` 方式更精确地控制 auto-start 的 per-GPU 模型配置
- `--on-conflict report|replace`：控制当 `28800` 已被旧进程占用时是报错还是替换

### 行为说明

- `tei machine` 是单机多 GPU 聚合层，不是多机器入口
- 如果 auto-start 阶段只有部分后端变健康，machine 会先带着这些健康实例启动
- 如果某个后端运行期掉卡、launch failed 或健康检查明确 unhealthy，machine 会把它摘出调度
- 如果只是请求期 OOM / capacity 错误，machine 会先自动拆小 batch 再重试，而不是立刻把后端永久判死
- `--compose-gpus` 和 `--compose-gpu-configs` 只影响 auto-start 新拉起的后端；如果当前已经有别的 TEI 后端在运行，machine 仍会把它们发现进来。要做精确控制，应先手动 `tei compose up ...`，再运行 `tei machine`

## `tei client`

### 访问本机 `tei machine`

```bash
tei client health
tei client info
tei client embed "Hello, world"
tei client lsh "Hello, world"
tei client rerank -q "query" -p "doc1" "doc2"
```

### 访问指定 endpoint

```bash
tei client -e "$TEI_MACHINE_A_URL" health
tei client -e "$TEI_BACKEND_A_URL" embed "Direct backend request"
tei client --port 28800 info
tei client --port 28880 embed "Hello"
```

### 访问多个 machine

```bash
tei client -E "$TEI_EPS" health
tei client -E "$TEI_EPS" embed "Hello" "World"
tei client -E "$TEI_EPS" lsh -b 2048 "Hello, world"
tei client -E "$TEI_EPS" rerank -q "query" -p "doc1" "doc2"
tei client -E "$TEI_EPS" info -v
```

### 客户端说明

- `-E/--endpoints` 是当前统一的多 machine 入口，取代旧的 `tei_clients` / `tei_clients_stats` CLI
- `info`、`lsh`、`rerank` 依赖 `tei machine` 聚合接口；如果你直连单个 TEI 后端，最稳定的是 `/health` 与 `/embed`
- 开启 `-v/--verbose` 时，多 machine 客户端会输出更详细的进度与统计信息

## `tei benchmark`

```bash
tei benchmark health -E "$TEI_EPS"
tei benchmark run -E "$TEI_EPS"
tei benchmark run -E "$TEI_EPS" -n 100000
tei benchmark run -E "$TEI_EPS" --min-len 150 --max-len 400
tei benchmark run -E "$TEI_EPS" --bitn 1024
tei benchmark run -E "$TEI_EPS" -v -o results.json
tei benchmark tune -E "$TEI_EPS"
tei benchmark tune -E "$TEI_EPS" --min-batch 500 --max-batch 3000 --step 250
tei benchmark generate -n 1000 --show
```

### benchmark 说明

- `run`：实际压测 `/lsh` 或相关路径的吞吐与延迟
- `tune`：扫描 batch 配置，把较优结果记录到配置文件
- `health`：压测前先检查 endpoint 是否可用
- `generate`：只生成文本样本，不发请求

## Python API

虽然 CLI 已统一为 `tei`，Python API 仍保留以下类：

- `TEIClient`
- `AsyncTEIClient`
- `TEIClients`
- `TEIClientsWithStats`

单 endpoint 示例：

```python
from tfmx import TEIClient

with TEIClient(endpoint="http://127.0.0.1:28800") as client:
    print(client.health())
    print(client.embed(["Hello", "World"]))
```

多 machine 示例：

```python
from tfmx import TEIClients

with TEIClients([
    "http://host-a:28800",
    "http://host-b:28800",
]) as clients:
    print(clients.health())
    print(clients.lsh(["text1", "text2"], bitn=2048))
```

## 配置文件

多 machine 调优结果会写入：

- `src/tfmx/configs/tei_clients.config.json`

这个文件按 endpoint 组合记录推荐批量与并发配置，供后续 benchmark / clients 复用。