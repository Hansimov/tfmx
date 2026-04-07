# QWN 配置与启动说明

## 概览

`qwn` 是统一的 Qwen 3.5 Docker CLI，当前包含四类子命令：

- `qwn compose`：生成并管理每张 GPU 一个 vLLM 容器的部署
- `qwn machine`：启动本地 OpenAI 兼容代理，支持前台和后台模式
- `qwn client`：访问代理或直接访问某个 vLLM 实例
- `qwn benchmark`：执行文本生成吞吐测试

## 前置条件

- 已安装 Docker 与 NVIDIA Container Toolkit
- `nvidia-smi` 可以看到目标 GPU
- Python 3.10+
- 本机有可复用的 Hugging Face 缓存目录 `~/.cache/huggingface`

## 默认模型

当前默认部署目标是：

- `cyankiwi/Qwen3.5-4B-AWQ-4bit`

在 compose 层，会给它统一暴露为稳定的服务模型名 `4b:4bit`，这样无论是直接访问容器还是经由 `qwn machine` 路由，模型标签都一致。

## 安装

```bash
pip install -e .
qwn --help
```

## 网络默认值

`qwn` 已经把镜像与代理配置抽成可复用模块，默认行为如下：

- Hugging Face 下载走 `https://hf-mirror.com`
- `pip` 下载走 `https://mirrors.ustc.edu.cn/pypi/simple`
- 若环境中存在 `QWN_PROXY`、`TFMX_QWN_PROXY` 或系统代理变量，则 Docker build 会自动复用它们
- 如果模型和多模态预处理文件已经在共享 Hugging Face cache 中，runtime 容器会自动切到离线模式，避免启动时再次因为容器内 DNS 或外网波动而失败

## 快速开始

建议先准备：

```bash
export QWN_MACHINE_URL="http://$QWN_HOST:27800"
export QWN_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"
```

### 1. 预拉基础镜像

```bash
docker pull vllm/vllm-openai:v0.19.0
```

第一次执行 `qwn compose up` 时，会在 `vllm/vllm-openai:v0.19.0` 之上构建本地镜像 `tfmx-vllm-openai:qwen3.5-v0.19.0`。当前只额外安装 `transformers` 主线与兼容依赖，不再像旧方案那样额外下载一整份 `vllm` wheel。

### 2. 启动一个或多个 GPU 实例

```bash
# 单卡启动
qwn compose up --gpu-configs "0:4b:4bit"

# 显式覆盖默认代理配置
qwn compose up --gpu-configs "0:4b:4bit" --proxy "$QWN_PROXY"

# 显式覆盖 pip mirror
qwn compose up --gpu-configs "0:4b:4bit" --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple

# 所有健康 GPU 统一部署默认 AWQ 模型
qwn compose up --gpu-layout uniform-awq

# 只在 0,1 两张卡上部署
qwn compose up --gpu-layout uniform-awq -g 0,1
```

### 3. 启动本地代理

```bash
# 前台模式
qwn machine run

# 后台模式（如无后端则自动拉起 compose）
qwn machine run --auto-start -b --on-conflict replace
qwn machine status
qwn machine logs
```

说明：

- `qwn machine` 当前默认监听 `0.0.0.0:27800`，不仅是 `localhost`
- `--auto-start` 会在没有检测到运行中的后端时，按默认 compose 策略拉起所有健康 GPU 对应的后端实例；如果只有部分后端在启动窗口内恢复健康，也会先带着这些健康实例启动
- `--on-conflict replace` 适合替换旧 machine 或已占用 `27800` 端口的旧进程
- 如果你已经用 `qwn compose up ...` 手动部署了精确的 GPU 布局，`qwn machine run -b` 即可；此时 `--auto-start` 不会重复启动新的后端
- 同机可用 `http://127.0.0.1:27800` 或 `http://localhost:27800` 访问；局域网内其他机器可直接用这台机器的 LAN IP，例如 `http://192.168.x.x:27800`
- 若局域网机器仍无法访问，优先检查宿主机防火墙、安全组或上层路由策略，而不是 `qwn machine` 本身的 bind 地址
- 对 OpenAI 兼容客户端，若 base URL 写成 `http://host:27800/v1`，可直接走标准 `/chat/completions` 与 `/models`；若 base URL 写成 `http://host:27800`，当前也额外兼容 `/chat/completions` 与 `/models` 这组无版本前缀别名

### 4. 验证服务并发起对话

```bash
qwn client health
qwn client models
qwn client chat "你好，请用三句话介绍你的能力。"
qwn benchmark run -E "$QWN_MACHINE_URL" -n 100
python debugs/qwn_multimodal_probe.py
```

说明：

- `qwn client chat` 与 `qwn client generate` 现在默认把 `--max-tokens` 提到 `8192`
- 这并不等于“无条件保证可以额外生成 8192 token”，因为当前部署的 `qwn compose` 默认 `--max-model-len` 也是 `8192`，实际可生成上限仍然受 `提示词 token + 输出 token <= 8192` 约束
- 在当前 20GB RTX 3080 的实测环境里，5 个健康实例空闲时每卡显存大约在 `14.5-14.9 GiB / 20 GiB`；对单个后端发起 `max_tokens=8192` 的长输出请求时，`nvidia-smi` 观测到显存占用没有高于这一已预留水位，说明当前风险点不在“客户端默认值从 512 提到 8192”本身，而在于如果你要把服务端 `--max-model-len` 再继续往上加，才更可能触发 KV cache 带来的 OOM
- 如果你确实需要在带较长 prompt 的情况下稳定生成接近 `8192` 个 token，应先评估是否要把服务端 `--max-model-len` 提高到 `12288` 或 `16384`；这在 20GB 卡上通常需要同步降低 `--max-num-seqs` 或 `--gpu-memory-utilization`
- 当前 compose 启动的 vLLM 实例会带上 `--reasoning-parser qwen3`，因此当开启 thinking 模式时，返回体会尽量把思考过程放进结构化 `reasoning` / `reasoning_content` 字段，而不是只能混在正文字符串中

### 5. 运行 benchmark

```bash
qwn benchmark run -E "$QWN_MACHINE_URL" -n 100 -o runs/qwns/results/latest.json
```

## 端口布局

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| vLLM 容器 | `27880+` | 每张 GPU 一个实例 |
| qwn machine | `27800` | OpenAI 兼容代理，默认监听 `0.0.0.0` |

## 后台服务文件

- PID 文件：`~/.cache/tfmx/qwn_machine.pid`
- 日志文件：`~/.cache/tfmx/qwn_machine.log`