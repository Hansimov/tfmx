# TEI 使用说明

`tei` 是当前唯一的公开 CLI 入口。原先拆开的 `tei_compose`、`tei_machine`、`tei_client`、`tei_clients`、`tei_clients_stats`、`tei_benchmark` 已经合并为统一子命令：

- `tei compose`：启动和管理 TEI 后端容器
- `tei machine`：启动单机聚合代理，负责多 GPU 负载均衡
- `tei client`：访问单个 `tei machine`、直接访问某个后端，或通过 `-E` 访问多个 `tei machine`
- `tei benchmark`：对一个或多个 `tei machine` 做压测与调优

## 安装

```sh
pip install -e .
tei --help
```

## 端口与拓扑

- `tei compose` 默认在 `28880+` 启动后端容器，每张健康 GPU 一个后端
- `tei machine` 默认监听 `28800`
- `tei client --port 28800 ...` 默认访问本机 `tei machine`
- `tei client --port 28880 ...` 可直接访问某个 TEI 后端

典型部署结构：

```text
tei client
    -> tei machine (28800)
        -> TEI backend gpu0 (28880)
        -> TEI backend gpu1 (28881)
        -> ...
```

## 快速开始

### 方案 A：让 `tei machine` 自动拉起后端

```sh
tei machine run --auto-start --perf-track --on-conflict replace
tei client health --port 28800
tei client embed --port 28800 "Hello" "World"
```

说明：

- `--auto-start` 会在未发现运行中的 TEI 后端时自动调用 `tei compose up`
- 自动启动会优先使用所有健康 GPU
- 启动后会优先等待已发现后端变健康；如果启动过程中有个别 GPU 掉卡或后端持续不健康，`tei machine` 会跳过这些坏实例，先使用剩余健康实例对外提供服务

### 方案 B：手动控制 compose，再启动 machine

```sh
export MODEL="Qwen/Qwen3-Embedding-0.6B"
export TEI_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"

tei compose up -m "$MODEL" --proxy "$TEI_PROXY_URL"
tei machine run --perf-track
```

适用场景：

- 需要精确指定 GPU，例如 `-g "0,2"`
- 需要单独控制 compose 生命周期
- 需要先排查某个后端容器的启动问题

## `tei compose`

常用命令：

```sh
export MODEL="Qwen/Qwen3-Embedding-0.6B"
export TEI_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"

tei compose up
tei compose up -g "0,1"
tei compose up --gpu-configs "0,2"
tei compose up -m "$MODEL"
tei compose up --mount-mode manual --proxy "$TEI_PROXY_URL"
tei compose ps
tei compose logs -f
tei compose health
tei compose down
```

关键行为：

- 默认尽量识别并使用所有健康 GPU
- 启动前会做 GPU 健康检查，不健康 GPU 会被自动排除
- `--gpu-configs` 支持 `GPU[:MODEL],...`；例如 `"0,2"` 会让 GPU0/GPU2 使用默认 embedding 模型
- `--mount-mode manual` 适合 NVIDIA runtime / NVML 状态不稳定时使用
- `tei compose setup` 会预先补齐模型缓存中的必要配置文件

## `tei machine`

常用命令：

```sh
export TEI_BACKENDS="$TEI_BACKEND_A_URL,$TEI_BACKEND_B_URL"

tei machine run
tei machine run --auto-start
tei machine run --auto-start --compose-gpus "0,2"
tei machine run --auto-start --compose-gpu-configs "0,2"
tei machine run --auto-start --perf-track --on-conflict replace
tei machine run -e "$TEI_BACKENDS"
tei machine discover
tei machine health
```

关键行为：

- `tei machine` 是单机多 GPU 聚合层，不是多机器入口
- `-e/--endpoints` 允许跳过容器自动发现，直接指定后端地址
- `--auto-start` 会在没有后端时自动启动 compose，并等待健康检查通过
- `--compose-model-name`、`--compose-port`、`--compose-project-name`、`--compose-gpus`、`--compose-gpu-configs` 可覆盖 auto-start 的默认 compose 参数
- `--on-conflict report|replace` 用于控制 `28800` 端口已被旧 machine 或其他进程占用时的处理方式
- 运行期如果某个后端因为掉卡、launch failed 或显式 unhealthy 而失效，machine 会把它摘出调度；如果只是请求期 OOM/容量不足，则会先自动拆小 batch 重试，而不是立刻把后端永久判死

## `tei client`

### 访问单个 machine 或单个后端

```sh
export TEI_MACHINE_URL="$TEI_MACHINE_A_URL"

tei client health -e "$TEI_MACHINE_URL"
tei client info --port 28800
tei client embed --port 28800 "Hello" "World"
tei client embed -e "$TEI_BACKEND_A_URL" "Direct backend request"
```

### 访问多个 machine

```sh
export TEI_EPS="$TEI_MACHINE_A_URL,$TEI_MACHINE_B_URL"

tei client health -E "$TEI_EPS"
tei client embed -E "$TEI_EPS" "Hello" "World"
tei client lsh -E "$TEI_EPS" -b 2048 "Hello, world"
tei client rerank -E "$TEI_EPS" -q "query" -p "doc1" "doc2"
tei client info -E "$TEI_EPS" -v
```

说明：

- `-E/--endpoints` 是新的统一多机入口，取代旧的 `tei_clients` / `tei_clients_stats` CLI
- 带 `-E` 且开启 `-v/--verbose` 时，会自动使用带统计输出的多机客户端行为
- `info`、`lsh`、`rerank` 依赖 `tei machine` 提供的聚合接口；如果直接访问单个后端，只能稳定使用 `/embed` 与 `/health`

## `tei benchmark`

```sh
export TEI_EPS="$TEI_MACHINE_A_URL,$TEI_MACHINE_B_URL"

tei benchmark health -E "$TEI_EPS"
tei benchmark run -E "$TEI_EPS" -n 100000
tei benchmark run -E "$TEI_EPS" -v -o results.json
tei benchmark tune -E "$TEI_EPS" --min-batch 500 --max-batch 3000 --step 250
tei benchmark generate -n 1000 --show
```

说明：

- `run` 用于实际压测
- `tune` 会扫描批次大小并把最佳结果保存到配置文件
- `health` 用于压测前快速检查集群是否可用

## Python API

虽然 CLI 已统一为 `tei`，Python API 仍保留以下类：

- `TEIClient`：单 endpoint 客户端
- `AsyncTEIClient`：单 endpoint 异步客户端
- `TEIClients`：多 `tei machine` 聚合客户端
- `TEIClientsWithStats`：带进度和统计输出的多 `tei machine` 客户端

### 单 endpoint

```python
from tfmx import TEIClient

with TEIClient(endpoint="http://$TEI_MACHINE_A_HOST:28800") as client:
    health = client.health()
    embeddings = client.embed(["Hello", "World"])
```

### 多 machine

```python
from tfmx import TEIClients

endpoints = [
    "http://$TEI_MACHINE_A_HOST:28800",
    "http://$TEI_MACHINE_B_HOST:28800",
]

with TEIClients(endpoints) as clients:
    health = clients.health()
    hashes = clients.lsh(["text1", "text2"], bitn=2048)
```

### 带统计输出

```python
from tfmx import TEIClientsWithStats

endpoints = [
    "http://$TEI_MACHINE_A_HOST:28800",
    "http://$TEI_MACHINE_B_HOST:28800",
]

with TEIClientsWithStats(endpoints, verbose=True) as clients:
    clients.lsh(["text1", "text2", "text3"], bitn=2048)
```

## 配置文件

多 machine 客户端会自动读取和写入：

- `src/tfmx/configs/tei_clients.config.json`

该文件按 endpoint 集合记录机器最佳批次与并发配置。示意格式：

```json
{
  "<cluster-hash>": {
    "endpoints": [
      "$TEI_MACHINE_A_URL",
      "$TEI_MACHINE_B_URL"
    ],
    "machines": {
      "$TEI_MACHINE_A_KEY": {
        "optimal_batch_size": 2000,
        "optimal_max_concurrent": 2,
        "throughput": 990.0,
        "instances": 2,
        "updated_at": "2026-01-16T21:54:10.945352"
      },
      "$TEI_MACHINE_B_KEY": {
        "optimal_batch_size": 2000,
        "optimal_max_concurrent": 8,
        "throughput": 2600.0,
        "instances": 8,
        "updated_at": "2026-01-16T21:54:38.278253"
      }
    }
  }
}
```

## 故障排查

### 没有健康后端

```sh
nvidia-smi -L
tei compose health
tei machine discover
tei machine run --auto-start
```

- 如果 `nvidia-smi -L` 已经出现 `Unable to determine the device handle ... Unknown Error`，说明是宿主机 GPU/驱动层掉卡；`tei machine` 会跳过坏后端，但无法替代底层恢复

### 多机吞吐偏低

```sh
tei benchmark tune -E "$TEI_EPS"
tei benchmark run -E "$TEI_EPS" -v -n 10000
```

### 超大数据集内存压力过高

```python
results = clients.lsh_iter(data_generator(), total_hint=1000000)
```

- 当前 `tei machine` 遇到后端返回 OOM/容量不足时，会优先递归拆小 batch 后重试；如果仍然持续失败，再考虑降低外部批量、调小 benchmark batch，或检查具体 GPU 是否已经掉线

## 迁移提示

- `tei_clients` -> `tei client -E ...`
- `tei_clients_stats` -> `tei client -E ... -v`
- `tei_benchmark` -> `tei benchmark ...`
- `tei_machine` -> `tei machine ...`
- `tei_compose` -> `tei compose ...`
