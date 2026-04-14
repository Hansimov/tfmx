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
- 这里的“健康 GPU”只是在硬件 / runtime 层面可用，不会因为同一张卡上已经跑了 `qsr` 就自动排除；如果显存足够，TEI 与 QSR 可以共存在同一批 GPU 上
- 因此当你当前服务器有 `6` 张可见且健康的 GPU 时，默认行为就是把这 `6` 张都纳入部署；若只想用子集，再显式传 `-g` 或 `--gpu-configs`
- 如果你用 `--gpu-configs` 配置了多个不同模型，而又没有手动指定 `--project-name`，compose 项目名会自动使用 `tei-multi`
- `tei compose setup` 适合在首次启动前跑一次，减少首次容器启动时的补文件等待

## `tei machine`

```bash
tei machine run
tei machine run --auto-start
tei machine run --background --auto-start --on-conflict replace
tei machine run --auto-start --perf-track --on-conflict replace
tei machine run --background --auto-start --compose-gpus "0,1,2,3,4,5" --on-conflict replace
tei machine run --auto-start --compose-gpus "0,1"
tei machine run --auto-start --compose-gpu-configs "0,1"
tei machine run --auto-start --compose-gpu-configs "0:Qwen/Qwen3-Embedding-0.6B,1:Alibaba-NLP/gte-multilingual-base"
tei machine run -e "$TEI_BACKENDS"
tei machine discover
tei machine health
tei machine status
tei machine logs --tail 200
tei machine stop
tei machine restart --auto-start --on-conflict replace
```

### 常用参数

- `-p/--port`：machine 监听端口，默认 `28800`
- `-e/--endpoints`：跳过容器自动发现，直接使用给定后端地址
- `-b/--batch-size`：每个实例允许的最大批量，默认 `300`
- `-m/--micro-batch-size`：自适应流水线调度用的小批量探测大小，默认 `100`
- `-B/--background`：以 daemon 模式在后台运行 machine
- `--perf-track`：打印更细的 pipeline 与 LSH 性能日志
- `--no-gpu-lsh`：关闭 GPU LSH，加速逻辑改为 CPU
- `--auto-start`：没有后端时自动调用 compose 拉起后端
- `--compose-gpus`：限制 auto-start 只用哪些 GPU
- `--compose-gpu-configs`：用 `GPU[:MODEL],...` 方式更精确地控制 auto-start 的 per-GPU 模型配置
- `--on-conflict report|replace`：控制当 `28800` 已被旧进程占用时是报错还是替换
- `status|logs|stop|restart`：查看 daemon 状态、读取日志、停止或重启后台 machine

### 行为说明

- `tei machine` 是单机多 GPU 聚合层，不是多机器入口
- 若使用 `--background`，daemon 的 PID / 日志默认写到 `~/.cache/tfmx/tei_machine.pid` 与 `~/.cache/tfmx/tei_machine.log`
- 如果 auto-start 阶段只有部分后端变健康，machine 会先带着这些健康实例启动
- 如果某个后端运行期掉卡、launch failed 或健康检查明确 unhealthy，machine 会把它摘出调度
- 如果只是请求期 OOM / capacity 错误，machine 会先自动拆小 batch 再重试，并记住该实例当前已验证过的安全批量上限，后续请求会优先按这个上限预切分，而不是每次都重新撞到同样的过载错误
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

## TEI 分步脚本

如果你想按步骤单独跑 TEI 的部署、machine、健康检查和 benchmark，可直接使用仓库内分步脚本：

```bash
bash runs/teis/01_deploy_default.sh
bash runs/teis/02_start_machine.sh
bash runs/teis/03_health_check.sh
bash runs/teis/04_benchmark.sh
```

- 默认使用当前 VM 中全部可见 GPU
- 若想只跑子集，可先导出 `TEI_DEPLOY_GPUS=0,1`
- `02_start_machine.sh` 现在直接走 `tei machine run --background --auto-start --compose-gpus ... --on-conflict replace`
- 更多分步说明见 `runs/teis/README.md`

## 联合恢复脚本

如果你当前要把 TEI `28800` 和同机 QWN `27800` 一起恢复到当前全部可见 GPU，并补完最小 live 验证，可直接执行：

```bash
bash runs/recovery/restart_tei_qwn.sh
```

这个脚本会先重启 TEI 后端与 machine，再做 repeated `/lsh` 校验，然后继续恢复 QWN 并完成 models/chat/benchmark 验证。

若你想把 TEI 吞吐压测也并进同一条恢复链路里，可打开可选 benchmark：

```bash
TEI_BENCH_SAMPLES=60000 QWN_BENCH_SAMPLES=80 bash runs/recovery/restart_tei_qwn.sh
```

- `TEI_BENCH_SAMPLES>0` 时，脚本会额外执行 `tei benchmark run`
- 当 `TEI_BENCH_SAMPLES` 与 `QWN_BENCH_SAMPLES` 都大于 `0` 时，脚本会让 TEI benchmark 在后台持续跑，再在负载期间执行 QWN benchmark，用于验证两套服务共存时的稳定性
- 结果目录分别是 `runs/recovery/results/`、`runs/teis/results/` 与 `runs/qwns/results/`
- 更多覆盖参数见 `runs/recovery/README.md`

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