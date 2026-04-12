# QSR 使用说明

## 统一入口

`qsr` 是唯一的公开入口。建议先准备一组环境变量，方便在不同机器间切换：

```bash
export QSR_MACHINE_URL="http://$QSR_HOST:27900"
export QSR_MACHINE_B_URL="http://$QSR_HOST_B:27900"
export QSR_BACKEND_A_URL="http://$QSR_BACKEND_HOST_A:27980"
export QSR_BACKEND_B_URL="http://$QSR_BACKEND_HOST_B:27981"
```

## `qsr compose`

```bash
qsr compose up
qsr compose up -g 0,1
qsr compose up --gpu-layout uniform
qsr compose up --gpu-layout uniform --skip-warmup
qsr compose warmup --gpu-layout uniform -g 0,1
qsr compose up --gpu-configs "0,2"
qsr compose up --gpu-configs "0:Qwen/Qwen3-ASR-0.6B,1:Qwen/Qwen3-ASR-0.6B"
qsr compose generate -j qsr-demo --gpu-configs "0"
qsr compose ps
qsr compose logs -f
qsr compose down
```

### 常用参数

- `--gpu-configs`：显式指定每张 GPU 的模型配置，格式为 `GPU[:MODEL],...`
- `--gpu-layout uniform`：对所有选中 GPU 应用默认 Qwen3-ASR 部署
- `--mount-mode manual|nvidia-runtime`：切换 GPU 设备挂载方式
- `--proxy`：覆盖默认 build/runtime 代理
- `--hf-endpoint`：覆盖默认 Hugging Face mirror
- `--pip-index-url`：覆盖默认 PyPI mirror
- `--max-model-len`：覆盖服务端最大上下文长度，默认 `4096`
- `--max-num-seqs`：每个 vLLM 实例允许的并发序列数，默认 `8`
- `--gpu-memory-utilization`：vLLM 显存利用率上限，默认 `0.35`
- `--project-name`：自定义 compose 项目名
- `--skip-warmup`：只启动容器，不等待默认 warmup 完成；适合你只想先把后端拉起的场景
- `compose warmup --audio`：对运行中的 backend 发起一次短转写预热；未提供时自动使用内置 WAV
- `compose warmup --wait-timeout/--request-timeout`：控制 warmup 前健康等待和单 backend 请求超时

### 行为说明

- 默认会先筛掉当前不健康的 GPU，再对健康 GPU 起容器
- 若你只是想快速把所有健康 GPU 拉起，优先使用 `--gpu-layout uniform`
- 若你要精确控制 per-GPU 部署，优先使用 `--gpu-configs`
- 当前默认值已经按 `Qwen3-ASR-0.6B` 的实际需求收窄，避免 0.6B 模型在 20GB 卡上预留过大的 KV cache
- `qsr compose up` 默认会在 backend 可达后自动做一次短转写 warmup；只有在你显式加了 `--skip-warmup` 时，才需要之后单独手动执行 `qsr compose warmup`

## `qsr machine`

```bash
qsr machine discover
qsr machine health
qsr machine run
qsr machine run -b
qsr machine run --auto-start -b --on-conflict replace
qsr machine run --auto-start -b --compose-gpus "0,1" --compose-gpu-layout uniform --on-conflict replace
qsr machine run -e http://localhost:27980,http://localhost:27981
qsr machine status
qsr machine logs --tail 200
qsr machine stop
qsr machine restart
```

### 常用参数

- `-p/--port`：machine 监听端口，默认 `27900`
- `-e/--endpoints`：跳过容器自动发现，直接使用给定后端地址列表
- `-b/--background`：以 daemon 模式运行
- `--auto-start`：没有后端时自动调用 compose 拉起后端
- `--auto-start`：没有后端时自动调用 compose 拉起后端，并继承 compose 的默认 warmup 行为
- `--compose-gpus`：限制 auto-start 只用哪些 GPU
- `--compose-gpu-layout`：指定 auto-start 的 named layout，当前支持 `uniform`
- `--compose-gpu-configs`：用 `GPU[:MODEL],...` 精确控制 auto-start 的 per-GPU 模型配置
- `--on-conflict report|replace`：控制 `27900` 端口冲突时是报错还是替换

### 行为说明

- `qsr machine` 是单机多 GPU 聚合层，不是跨机器入口
- 调度算法当前是轻量级 `least_active_idle`，优先选择健康且当前活跃请求最少的实例；若活跃数相同，会在平手实例之间轮转，避免固定偏向低编号 GPU
- 若上游实例在响应开始前返回 `5xx`、超时或断连，machine 会在健康实例之间做受控 failover
- 若流式响应已经开始输出，则不会再切换实例，以避免混合多个上游流

## OpenAI 兼容接口

推荐把 base URL 指到 `http://<host>:27900`，实际请求路径走标准 OpenAI 风格：

- `POST /v1/chat/completions`
- `POST /v1/audio/transcriptions`
- `GET /v1/models`

同时也兼容无版本前缀别名：

- `POST /chat/completions`
- `POST /audio/transcriptions`
- `GET /models`

## `qsr client`

### 访问本机 `qsr machine`

```bash
qsr client health
qsr client models
qsr client info
qsr client transcribe ./sample.wav
qsr client transcribe ./sample.wav --response-format text
qsr client chat --audio ./sample.wav "请转写为简体中文"
qsr client chat --audio ./sample.wav --no-stream --json
```

### 访问指定 endpoint

```bash
qsr client health -E "$QSR_MACHINE_URL"
qsr client info -E "$QSR_MACHINE_URL"
qsr client models -E "$QSR_BACKEND_A_URL"
qsr client transcribe -E "$QSR_BACKEND_A_URL" ./sample.wav
qsr client chat -E "$QSR_BACKEND_A_URL" --audio ./sample.wav "请转写"
```

### 客户端说明

- `qsr client transcribe` 支持本地文件、URL、`data:` URI
- `qsr client chat` 支持多段文本和多段音频，会按顺序交错组成一个 OpenAI multimodal message
- 默认 chat 走流式输出；若你需要完整 JSON 响应，可加 `--no-stream --json`
- `qsr client info` 会返回 JSON，其中包含实例列表、累计请求统计、可用模型以及调度器摘要

## `qsr benchmark`

```bash
qsr benchmark health -E "$QSR_MACHINE_URL"
qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --audio ./sample.wav
qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --mode chat --audio ./sample.wav --prompt "请转写为简体中文"
qsr benchmark run -E "$QSR_MACHINE_URL" "$QSR_MACHINE_B_URL" -n 40 -o runs/qsrs/results/latest.json
```

### benchmark 说明

- `health`：压测前先检查 endpoint 健康状态
- `run --mode transcribe`：走 transcription API
- `run --mode chat`：走 chat completions API，并可选测量 TTFT
- benchmark 输出会区分 `submitted`、`successful`、`failed`、`success_rate`

## QSR 分步脚本

如果你想按步骤单独跑 QSR 的部署、machine、健康检查和 benchmark，可直接使用仓库内脚本：

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/03_health_check.sh
bash runs/qsrs/04_benchmark.sh
bash runs/qsrs/05_test_scheduling.sh
```

- 默认使用当前 VM 中全部可见 GPU
- 若想只跑子集，可先导出 `QSR_DEPLOY_GPUS=0,1`
- 更多分步说明见 `runs/qsrs/README.md`

## Python API

虽然 CLI 已统一为 `qsr`，Python API 仍保留以下类：

- `QSRClient`
- `AsyncQSRClient`
- `QSRClients`
- `QSRClientsWithStats`

单 endpoint 示例：

```python
from tfmx import QSRClient

with QSRClient(endpoint="http://127.0.0.1:27900") as client:
    print(client.health())
    print(client.models())
```

benchmark / multi-endpoint 示例：

```python
from tfmx import QSRBenchmark

with QSRBenchmark(["http://127.0.0.1:27900"]) as bench:
    print(bench.check_health())
```