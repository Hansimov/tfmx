# QVL Module Setup Guide

## Overview

The `qvls` module deploys and manages **Qwen3-VL** vision-language models via **vLLM** in Docker containers. It supports multi-GPU load balancing across multiple machines with AWQ quantization for RTX 30/40 series GPUs.

## Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU(s) with CUDA support (RTX 30xx/40xx recommended)
- Python 3.10+
- HuggingFace model cache (`~/.cache/huggingface`)

## Installation

```bash
# Install the package
pip install -e .

# Verify installation
qvl_compose --help
qvl_machine --help
qvl_client --help
```

## Supported Models

| Model | Size | Type | VRAM (FP16) | VRAM (AWQ-8bit) | VRAM (AWQ-4bit) |
|-------|------|------|------------|-----------------|-----------------|
| Qwen/Qwen3-VL-2B-Instruct | 2B | Instruct | ~5GB | ~3GB | ~2GB |
| Qwen/Qwen3-VL-2B-Thinking | 2B | Thinking | ~5GB | ~3GB | ~2GB |
| Qwen/Qwen3-VL-4B-Instruct | 4B | Instruct | ~9GB | ~5GB | ~3GB |
| Qwen/Qwen3-VL-4B-Thinking | 4B | Thinking | ~9GB | ~5GB | ~3GB |
| Qwen/Qwen3-VL-8B-Instruct | 8B | Instruct | ~17GB | ~10GB | ~6GB |
| Qwen/Qwen3-VL-8B-Thinking | 8B | Thinking | ~17GB | ~10GB | ~6GB |

### AWQ Quantized Models (from cyankiwi)

Pre-quantized AWQ models are available from the `cyankiwi` organization on HuggingFace:

- `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit`
- `cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit`
- `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit`
- `cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit`
- `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`
- `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit`
- `cyankiwi/Qwen3-VL-8B-Thinking-AWQ-8bit`

Quantization levels: **4bit** (recommended), 8bit

## Quick Start

### 1. Pull vLLM Docker Image

```bash
docker pull vllm/vllm-openai:latest
```

### 2. Download Model

```bash
# Using hf CLI (recommended)
pip install "huggingface_hub[cli]"

# Download AWQ model (recommended for RTX 30/40)
hf download cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit

# Or with Chinese mirror
HF_ENDPOINT=https://hf-mirror.com hf download cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit

# Download multiple model variants for per-GPU deployment
for repo in \
  cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit \
  cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit \
  cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit \
  cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit \
  cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit \
  cyankiwi/Qwen3-VL-8B-Thinking-AWQ-8bit; do
  hf download "$repo"
done
```

### 3. Deploy with Docker Compose

```bash
# Start on all available GPUs (default: 8B-Instruct AWQ 4bit)
qvl_compose up

# Start with specific model and quantization
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q awq

# Start on specific GPUs
qvl_compose up -g "0,1"

# Per-GPU model/quant configuration (different models on different GPUs)
# Note: model shortcuts and quant levels are case-insensitive
qvl_compose up --gpu-configs "0:8b-instruct:4bit,1:8b-instruct:8bit"

# Full 6-GPU deployment with different models
qvl_compose up --gpu-configs \
  "0:2b-instruct:4bit,1:4b-instruct:4bit,2:8b-instruct:4bit,3:4b-thinking:4bit,4:8b-instruct:8bit,5:8b-thinking:8bit"

# Check status
qvl_compose ps

# View logs
qvl_compose logs -f

# Stop
qvl_compose down
```

### 4. Start Machine Proxy

```bash
# Auto-discover vLLM containers
qvl_machine run

# Or specify endpoints manually
qvl_machine run -e "http://localhost:29880,http://localhost:29881"
```

### 5. Test with Client

```bash
# Quick health check
qvl_clients health --endpoints http://localhost:29800

# Interactive chat
qvl_clients chat --endpoints http://localhost:29800

# Generate text
qvl_clients generate --endpoints http://localhost:29800 --prompt "Describe what you see" --images photo.jpg
```

## Port Scheme

| Service | Port | Description |
|---------|------|-------------|
| vLLM instances | 29880+ | One per GPU (29880, 29881, 29882...) |
| Machine proxy | 29800 | Load-balanced proxy |

## Multi-Machine Deployment

### Machine 1 (2x GPU)
```bash
qvl_compose up -g "0,1" -m "Qwen/Qwen3-VL-8B-Instruct" -q awq
qvl_machine run
```

### Machine 2 (2x GPU, mixed models)
```bash
qvl_compose up --gpu-configs "0:4b-instruct:4bit,1:8b-instruct:8bit"
qvl_machine run
```

### Client (any machine)
```bash
qvl_clients health --endpoints http://machine1:29800 http://machine2:29800
qvl_benchmark run -E "http://machine1:29800,http://machine2:29800" -n 100
```

## Benchmark Images

Download real images from HuggingFace datasets for benchmarking:

```bash
# Install datasets library
pip install datasets

# Download benchmark images (~500 images from Visual7W)
python -m tfmx.qvls.benchimgs download -n 500

# Check downloaded images
python -m tfmx.qvls.benchimgs info

# Test image generation
python -m tfmx.qvls.benchimgs test -n 5
```

Images are stored in `data/bench_images/` and used automatically by the benchmark tools. If no local images are available, synthetic images are generated as fallback.
