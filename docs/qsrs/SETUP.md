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

### 2. 启动一个或多个 GPU 实例

```bash
# 单卡启动
qsr compose up --gpu-configs "0"

# 所有健康 GPU 统一部署默认 Qwen3-ASR
qsr compose up --gpu-layout uniform

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
- 当前 compose 层没有额外的 warmup/sleep 子命令，逻辑保持尽量简单

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

qsr machine status
qsr machine logs
```

说明：

- `qsr machine` 当前监听 `0.0.0.0:27900`，不仅是 `localhost`
- `--auto-start` 会在没有检测到运行中的 QSR 后端时自动调用 `qsr compose up`
- `--on-conflict replace` 适合替换旧 machine 或已占用 `27900` 的旧进程
- 后台 daemon 的 PID 与日志位于 `~/.cache/tfmx/qsr_machine.pid` 与 `~/.cache/tfmx/qsr_machine.log`

### 4. 验证服务并发起请求

```bash
qsr client health
qsr client models
qsr client info
qsr client transcribe ./sample.wav
qsr client transcribe https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav --response-format text
qsr client chat --audio ./sample.wav "请转写为简体中文"
```

说明：

- `qsr client transcribe` 走的是 OpenAI 兼容 `/v1/audio/transcriptions`
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
```

更多 staged workflow 见 `runs/qsrs/README.md`。