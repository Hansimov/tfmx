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
qsr compose up --gpu-layout uniform --enable-sleep-mode
qsr compose up --gpu-layout uniform --profile-startup
qsr compose up --gpu-layout uniform --skip-warmup
qsr compose warmup --gpu-layout uniform -g 0,1
qsr compose wake --wait-healthy
qsr compose sleep --sleep-level 1 --sleep-mode abort
qsr compose sleep-status
python debugs/qsrs/profile_startup.py qsr-uniform--gpu0 qsr-uniform--gpu1
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
- `--enable-sleep-mode`：开启 vLLM 的 sleep/wake 端点，后续可用 `compose wake` 快速恢复，避免再次完整 cold start
- `--profile-startup`：输出 cold start 分阶段耗时，区分 compose、Docker health、HTTP readiness 和 warmup
- `--skip-warmup`：只启动容器，不等待默认 warmup 完成；适合你只想先把后端拉起的场景
- `compose warmup --audio`：对运行中的 backend 发起一次短转写预热；未提供时自动使用内置 WAV
- `compose warmup --wait-timeout/--request-timeout`：控制 warmup 前健康等待和单 backend 请求超时
- `compose sleep` / `compose wake --wait-healthy`：把已部署 backend 切到 vLLM sleep 模式，再以明显低于 cold start 的代价恢复
- `debugs/qsrs/profile_startup.py`：进一步解析 backend docker 日志，把内部冷启动拆成 model load、torch.compile、KV cache、graph capture、server start 等阶段

### 行为说明

- 默认会先筛掉当前不健康的 GPU，再对健康 GPU 起容器
- 若你只是想快速把所有健康 GPU 拉起，优先使用 `--gpu-layout uniform`
- 若你要精确控制 per-GPU 部署，优先使用 `--gpu-configs`
- 当前默认值已经按 `Qwen3-ASR-0.6B` 的实际需求收窄，避免 0.6B 模型在 20GB 卡上预留过大的 KV cache
- `qsr compose up` 默认会在 backend 可达后自动做一次短转写 warmup；只有在你显式加了 `--skip-warmup` 时，才需要之后单独手动执行 `qsr compose warmup`
- 若你追求的是重复重启场景下的恢复时间，而不是首次从零启动，请在部署时加 `--enable-sleep-mode`，随后优先使用 `qsr compose sleep` / `qsr compose wake --wait-healthy`
- 若你要定位 cold start 瓶颈，请在 `qsr compose up` 上加 `--profile-startup`

## `qsr machine`

```bash
qsr machine discover
qsr machine health
qsr machine run
qsr machine run -b
qsr machine run --auto-start -b --on-conflict replace
qsr machine run --auto-start -b --compose-gpus "0,1" --compose-gpu-layout uniform --on-conflict replace
qsr machine run --auto-start -b --compose-enable-sleep-mode --on-conflict replace
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
- `--compose-enable-sleep-mode`：auto-start 新拉起 backend 时启用 sleep-mode 端点，方便后续快速 wake/resume
- `--on-conflict report|replace`：控制 `27900` 端口冲突时是报错还是替换

### 行为说明

- `qsr machine` 是单机多 GPU 聚合层，不是跨机器入口
- 当 `--auto-start` 遇到处于 sleep 状态、且本地 sleep 状态文件仍在的 backend 时，会先请求 wake-up，再继续 machine 启动流程
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
qsr client transcribe ./meeting.mp3 --long-audio-mode auto --json
qsr client transcribe ./sample.wav --response-format text
qsr client transcribe-long ./meeting.mp3 --json
qsr client chat --audio ./sample.wav "请转写为简体中文"
qsr client chat --audio ./sample.wav --no-stream --json
```

### 访问指定 endpoint

```bash
qsr client health -E "$QSR_MACHINE_URL"
qsr client info -E "$QSR_MACHINE_URL"
qsr client models -E "$QSR_BACKEND_A_URL"
qsr client transcribe -E "$QSR_BACKEND_A_URL" ./sample.wav
qsr client transcribe -E "$QSR_MACHINE_URL" ./meeting.mp3 --long-audio-mode force --target-chunk-sec 75 --json
qsr client transcribe-long -E "$QSR_MACHINE_URL" ./meeting.mp3 --target-chunk-sec 75 --json
qsr client chat -E "$QSR_BACKEND_A_URL" --audio ./sample.wav "请转写"
```

### 客户端说明

- `qsr client transcribe` 支持本地文件、URL、`data:` URI
- `qsr client transcribe -E <machine> --long-audio-mode auto|force` 会把长音频拆分与调度下沉到 `qsr machine`；machine 侧同时暴露了 `POST /v1/audio/transcriptions/long` 与 `POST /audio/transcriptions/long` 两个显式入口
- `qsr client transcribe-long` 面向单条长音频任务：先用 `ffmpeg/ffprobe` 探测时长与静音区间，再切成带少量重叠的短片段，随后按 chunk 时长从长到短排队，并根据 `qsr machine /info` 当前可调度槽位动态补充并发
- 长音频模式不是简单等分；默认会优先在静音附近切段，并通过 `--overlap-sec` 降低边界漏字风险
- 长音频模式默认启用更激进但受控的 slot-aware 调度：每张健康 GPU 默认最多并发 `4` 个 chunk，并且会先填满空闲实例，再利用剩余可用槽位继续补货
- client-side 长音频现在按需并行提取 chunk，不再先串行切完整个文件再开始发请求
- 长音频模式现在会先自动尝试 `verbose_json` + `segment` 时间戳；如果当前 `Qwen3-ASR-0.6B` backend 仍然不支持，或者响应里没有 `segments`，会自动回退到重叠文本去重
- `qsr client transcribe-long` 依赖本机可用的 `ffmpeg` 和 `ffprobe`
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
bash runs/qsrs/06_soak_mixed.sh
bash runs/qsrs/98_sleep_backends.sh
```

- 默认使用当前 VM 中全部可见 GPU
- 若想只跑子集，可先导出 `QSR_DEPLOY_GPUS=0,1`
- staged workflow 现在默认就会启用 sleep mode 并优先走 wake-first；若你需要退回旧的 cold-start-only 行为，请导出 `QSR_ENABLE_SLEEP_MODE=0`
- 若想让 staged deploy 额外打印 cold-start phase profile，请导出 `QSR_PROFILE_STARTUP=1`
- 若想跑混合 chat/transcribe 长压，可用 `bash runs/qsrs/06_soak_mixed.sh`，并通过 `QSR_SOAK_*` 环境变量覆盖样本数、音频和并发度
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