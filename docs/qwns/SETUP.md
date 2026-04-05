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

## 快速开始

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

# 后台模式
qwn machine run -b
qwn machine status
qwn machine logs
```

### 4. 验证服务并发起对话

```bash
qwn client health
qwn client models
qwn client chat "你好，请用三句话介绍你的能力。"
```

### 5. 运行 benchmark

```bash
qwn benchmark run -E http://localhost:27800 -n 100 -o runs/qwns/results/latest.json
```

## 端口布局

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| vLLM 容器 | `27880+` | 每张 GPU 一个实例 |
| qwn machine | `27800` | OpenAI 兼容代理 |

## 后台服务文件

- PID 文件：`~/.cache/tfmx/qwn_machine.pid`
- 日志文件：`~/.cache/tfmx/qwn_machine.log`