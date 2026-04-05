# QWN Setup Guide

## Overview

`qwn` is a unified CLI for running quantized Qwen 3.5 text models through `vllm` in Docker. It covers:

- `qwn compose`: generate and manage one vLLM container per GPU
- `qwn machine`: run a local OpenAI-compatible proxy in foreground or background
- `qwn client`: talk to the proxy or a direct vLLM container
- `qwn benchmark`: run text-generation throughput checks

## Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU(s) visible to `nvidia-smi`
- Python 3.10+
- A Hugging Face cache under `~/.cache/huggingface`

## Default Model

Current default deployment targets the user-requested AWQ model repo:

- `cyankiwi/Qwen3.5-4B-AWQ-4bit`

The compose layer exposes it to vLLM with a stable served model name like `4b:4bit`, so routing and debugging stay consistent across direct containers and the machine proxy.

## Installation

```bash
pip install -e .
qwn --help
```

## Quick Start

`qwn` now has a reusable network config layer for mirrors and proxies. By default it uses `https://hf-mirror.com` for Hugging Face, `https://mirrors.ustc.edu.cn/pypi/simple` for `pip`, and it automatically reuses proxy settings from `QWN_PROXY`, `TFMX_QWN_PROXY`, or standard system proxy variables when they are present.

### 1. Prepare the runtime image

```bash
docker pull vllm/vllm-openai:v0.19.0
```

The first `qwn compose up` will auto-build a local runtime image named `tfmx-vllm-openai:qwen3.5-v0.19.0` on top of `vllm/vllm-openai:v0.19.0`. It only layers `transformers` main on top so the `qwen3_5` architecture is available, which avoids the previous one-time 432MB `pip install vllm` download.

### 2. Deploy on one or more GPUs

```bash
# Single GPU
qwn compose up --gpu-configs "0:4b:4bit"

# Override the default network config explicitly when needed
qwn compose up --gpu-configs "0:4b:4bit" --proxy "$QWN_PROXY"
qwn compose up --gpu-configs "0:4b:4bit" --pip-index-url https://mirrors.ustc.edu.cn/pypi/simple

# All healthy GPUs, same model on every card
qwn compose up --gpu-layout uniform-awq

# Selected GPUs, same model on both
qwn compose up --gpu-layout uniform-awq -g 0,1
```

### 3. Start the local machine proxy

```bash
# Foreground mode
qwn machine run

# Background mode with daemon management
qwn machine run -b
qwn machine status
qwn machine logs
```

### 4. Verify and chat

```bash
qwn client health
qwn client models
qwn client chat "你好，请用三句话介绍你的能力。"
```

### 5. Run a benchmark

```bash
qwn benchmark run -E http://localhost:27800 -n 100 -o runs/qwns/results/latest.json
```

## Port Layout

| Service | Port | Notes |
| --- | --- | --- |
| vLLM containers | `27880+` | one per GPU |
| qwn machine | `27800` | OpenAI-compatible proxy |

## Background Service Files

- PID file: `~/.cache/tfmx/qwn_machine.pid`
- Log file: `~/.cache/tfmx/qwn_machine.log`