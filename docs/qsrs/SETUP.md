# QSR 配置与启动说明

## 概览

`qsr` 是统一的 Qwen3-ASR Docker CLI，当前包含四类子命令：

- `qsr compose`：生成并管理每张 GPU 一个 vLLM 容器的部署
- `qsr machine`：启动本地 OpenAI 兼容 ASR 聚合代理，支持前台和后台模式
- `qsr client`：访问 `qsr machine` 或直接访问某个后端实例
- `qsr benchmark`：执行 transcription/chat 模式的 ASR 压测

当前默认部署目标是：

- `Qwen/Qwen3-ASR-0.6B`

默认端口布局：

- `qsr machine`：`27900`
- QSR vLLM 后端：`27980+`

## 前置条件

- 已安装 Docker 与 NVIDIA Container Toolkit
- `nvidia-smi` 可以看到目标 GPU
- Python 3.10+
- 建议本机已有可复用的 Hugging Face cache，减少首次启动等待

## 安装

```bash
pip install -e .
qsr --help
```

## 网络默认值

`qsr` 已将镜像、代理与 pip mirror 配置抽成共享模块，默认行为如下：

- Hugging Face 下载走 `https://hf-mirror.com`
- `pip` 下载走 `https://mirrors.ustc.edu.cn/pypi/simple`
- 若环境中存在 `QSR_PROXY_URL`、`TFMX_QSR_PROXY` 或系统代理变量，Docker build/runtime 会自动复用它们
- 如果模型相关文件已经在共享 Hugging Face cache 中，runtime 容器会自动切到离线模式，尽量避免启动时再次依赖外网

## 快速开始

建议先准备：

```bash
export QSR_MACHINE_URL="http://$QSR_HOST:27900"
export QSR_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"
```

### 1. 预拉基础镜像

```bash
docker pull vllm/vllm-openai:v0.19.0
```

第一次执行 `qsr compose up` 时，会在 `vllm/vllm-openai:v0.19.0` 之上构建本地镜像 `tfmx-vllm-openai:qwen3-asr-v0.19.0`。当前镜像会固定安装 `vllm[audio]==0.19.0`、`qwen-asr==0.0.6`、与 `transformers 4.57.x` 兼容的 `huggingface-hub>=0.34.0,<1.0`，以及 `qwen-asr` 运行期实际需要的辅助依赖，避免上游 `qwen-asr[vllm]` 把 `vllm` 回退到旧版本。

当前默认 runtime 也针对 `0.6B` ASR 做了收敛：默认 `max_model_len=4096`、`max_num_seqs=8`、`gpu_memory_utilization=0.35`。这样单卡 20GB RTX 3080 上的常驻显存会明显低于之前的大块 KV cache 预留；如果你确实需要更长音频上下文或更高并发，再按需覆盖这些参数。

### 2. 启动一个或多个 GPU 实例

```bash
# 单卡启动
qsr compose up --gpu-configs "0"

# 所有健康 GPU 统一部署默认 Qwen3-ASR
qsr compose up --gpu-layout uniform

# 如果你需要更快的重复启动/恢复，可在首次部署时开启 sleep mode
qsr compose up --gpu-layout uniform --enable-sleep-mode

# 如果你要分析 cold start 各阶段耗时，可显式输出 startup profile
qsr compose up --gpu-layout uniform --profile-startup

# 如需尽快返回而不等待默认预热，可显式跳过
qsr compose up --gpu-layout uniform --skip-warmup

# 只在 0,1 两张卡上部署
qsr compose up --gpu-layout uniform -g 0,1

# 显式覆盖默认代理配置
qsr compose up --gpu-layout uniform --proxy "$QSR_PROXY_URL"

# 显式覆盖 pip mirror
qsr compose up --gpu-layout uniform --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple
```

说明：

- `--gpu-layout uniform` 会对所有选中 GPU 使用同一个默认 ASR 模型
- `--gpu-configs "0,2"` 也是合法写法，表示 GPU0/GPU2 使用默认 `Qwen/Qwen3-ASR-0.6B`
- `qsr compose up` 现在默认会等待 backend 可达后做一次短转写 warmup，不再需要额外手动执行 `qsr compose warmup`
- 如果你只是想先把容器拉起、稍后再手动预热，可显式加 `--skip-warmup`
- 如果你希望以后用 wake/resume 代替重复 cold start，请在这里就加 `--enable-sleep-mode`，后续即可使用 `qsr compose sleep` / `qsr compose wake --wait-healthy`
- 如果你要看首轮 cold start 的瓶颈落在哪个阶段，加 `--profile-startup` 可以直接看到 compose、Docker health、HTTP readiness 和 warmup 的分段耗时

### 3. 启动本地聚合代理

```bash
# 前台模式
qsr machine run

# 后台模式
qsr machine run -b --on-conflict replace

# 如无后端则自动拉起 compose
qsr machine run --auto-start -b --on-conflict replace

# 只让 auto-start 使用 0,1 这两张卡
qsr machine run --auto-start -b --compose-gpus "0,1" --compose-gpu-layout uniform --on-conflict replace

# auto-start 新拉起 backend 时也可以直接启用 sleep mode
qsr machine run --auto-start -b --compose-enable-sleep-mode --on-conflict replace

qsr machine status
qsr machine logs
```

说明：

- `qsr machine` 当前监听 `0.0.0.0:27900`，不仅是 `localhost`
- `--auto-start` 会在没有检测到运行中的 QSR 后端时自动调用 `qsr compose up`
- `--auto-start` 走到 `qsr compose up` 时也会继承默认 warmup，因此冷启动后不需要再额外手动预热一遍
- 如果检测到已有后端处于 sleep 状态，`--auto-start` 会先请求 wake-up；若没有运行中的后端，再回退到普通 compose cold start
- `--on-conflict replace` 适合替换旧 machine 或已占用 `27900` 的旧进程
- 后台 daemon 的 PID 与日志位于 `~/.cache/tfmx/qsr_machine.pid` 与 `~/.cache/tfmx/qsr_machine.log`

### 4. 验证服务并发起请求

```bash
qsr client health
qsr client models
qsr client info
qsr client transcribe ./sample.wav
qsr client transcribe https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav --response-format text
qsr client transcribe ./meeting.mp3 --long-audio-mode auto --target-chunk-sec 75 --json
qsr client transcribe-long ./meeting.mp3 --target-chunk-sec 75 --json
qsr client chat --audio ./sample.wav "请转写为简体中文"
```

说明：

- `qsr client transcribe` 走的是 OpenAI 兼容 `/v1/audio/transcriptions`
- `qsr client transcribe --long-audio-mode auto|force` 会直接命中 machine 侧长音频路径；对应 HTTP 接口也可直接调用 `/v1/audio/transcriptions/long`
- `qsr client transcribe-long` 适合把单条长音频任务拆成多个短 chunk 并动态分发到多张 GPU；它依赖本机 `ffmpeg/ffprobe`，默认按静音边界切段、按需并行提取 chunk，并保留少量重叠
- 长音频调度默认每张健康 GPU 最多并发 `4` 个 chunk；如果你想保守一些，可显式传 `--per-instance-parallelism-cap 1|2|3`
- 长音频模式现在会优先自动尝试 `verbose_json` + `segment` 时间戳；如果当前 backend（例如默认的 `0.6B` 部署）返回 `400` 或者不给 `segments`，会自动回退到现有的重叠文本去重，不会中断请求
- `qsr client chat` 走的是 OpenAI 兼容 `/v1/chat/completions`，并把音频封装为 `audio_url` 内容块
- 音频输入可以是本地文件路径、HTTP/HTTPS URL，或 `data:` URI

### 5. 运行 benchmark

```bash
qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --audio ./sample.wav
qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --mode chat --audio ./sample.wav --prompt "请转写并概括主要内容"
```

benchmark 默认输出 JSON，可用 `-o` 写入文件。

## OpenAI 兼容接口

`qsr machine` 当前暴露以下接口：

- `POST /v1/chat/completions`
- `POST /chat/completions`
- `POST /v1/audio/transcriptions`
- `POST /audio/transcriptions`
- `GET /v1/models`
- `GET /models`
- `GET /health`
- `GET /info`
- `GET /metrics`
- `POST /chat`

## 分步脚本

如果你想按 repo 内的固定流程部署和验证，可直接使用：

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/03_health_check.sh
bash runs/qsrs/04_benchmark.sh
bash runs/qsrs/06_soak_mixed.sh
```

repo 内置 staged workflow 现在默认就会启用 sleep mode 并优先走 wake-first；如果你想保留旧的 cold-start-only 行为：

```bash
export QSR_ENABLE_SLEEP_MODE=0
```

如果你想让 `01_deploy_default.sh` 额外打印 cold-start phase profile：

```bash
export QSR_PROFILE_STARTUP=1
bash runs/qsrs/01_deploy_default.sh
```

如果你想继续往 backend 内部阶段深挖，而不是只看 compose 层时间：

```bash
python debugs/qsrs/profile_startup.py qsr-uniform--gpu0 qsr-uniform--gpu1
```

它会从 docker logs 中解析 model load、torch.compile、KV cache、graph capture、server start、first health 等阶段时间。

更多 staged workflow 见 `runs/qsrs/README.md`。