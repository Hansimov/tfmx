# QVL Module Setup Guide

## Overview

The `qvls` module deploys and manages **Qwen3-VL** vision-language models via **vLLM** in Docker containers. It supports multi-GPU load balancing across multiple machines with GGUF quantization for RTX 30/40 series GPUs.

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

| Model | Size | Type | VRAM (FP16) | VRAM (Q8) | VRAM (Q4) |
|-------|------|------|------------|-----------|-----------|
| Qwen/Qwen3-VL-2B-Instruct | 2B | Instruct | ~5GB | ~3GB | ~2GB |
| Qwen/Qwen3-VL-2B-Thinking | 2B | Thinking | ~5GB | ~3GB | ~2GB |
| Qwen/Qwen3-VL-4B-Instruct | 4B | Instruct | ~9GB | ~5GB | ~3GB |
| Qwen/Qwen3-VL-4B-Thinking | 4B | Thinking | ~9GB | ~5GB | ~3GB |
| Qwen/Qwen3-VL-8B-Instruct | 8B | Instruct | ~17GB | ~10GB | ~6GB |
| Qwen/Qwen3-VL-8B-Thinking | 8B | Thinking | ~17GB | ~10GB | ~6GB |

### GGUF Quantized Models (from unsloth)

All model variants are available in GGUF format with multiple quantization levels:

- `unsloth/Qwen3-VL-2B-Instruct-GGUF`
- `unsloth/Qwen3-VL-4B-Instruct-GGUF`
- `unsloth/Qwen3-VL-8B-Instruct-GGUF`
- `unsloth/Qwen3-VL-2B-Thinking-GGUF`
- `unsloth/Qwen3-VL-4B-Thinking-GGUF`
- `unsloth/Qwen3-VL-8B-Thinking-GGUF`

Quantization levels: **Q4_K_M** (recommended), Q5_K_M, Q6_K, Q8_0

## Quick Start

### 1. Pull vLLM Docker Image

```bash
docker pull vllm/vllm-openai:latest
```

### 2. Download Model

```bash
# Using huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# Or with Chinese mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# For GGUF quantized model (default recommended)
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF

# Download specific GGUF quant file only
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF \
  Qwen3-VL-8B-Instruct-Q4_K_M.gguf

# Download multiple model variants for per-GPU deployment
for model in 2B-Instruct 4B-Instruct 8B-Instruct 4B-Thinking 8B-Thinking; do
  huggingface-cli download "unsloth/Qwen3-VL-${model}-GGUF"
  huggingface-cli download "Qwen/Qwen3-VL-${model}"  # tokenizer
done
```

### 3. Deploy with Docker Compose

```bash
# Start on all available GPUs (default: 8B-Instruct GGUF Q4_K_M)
qvl_compose up

# Start with specific model and quantization
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf

# Start on specific GPUs
qvl_compose up -g "0,1"

# Per-GPU model/quant configuration (different models on different GPUs)
qvl_compose up --gpu-configs "0:2B-Instruct:Q4_K_M,1:8B-Instruct:Q8_0"

# Full 6-GPU deployment with different models
qvl_compose up --gpu-configs \
  "0:2B-Instruct:Q4_K_M,1:4B-Instruct:Q4_K_M,2:8B-Instruct:Q4_K_M,3:4B-Thinking:Q4_K_M,4:8B-Instruct:Q8_0,5:8B-Thinking:Q4_K_M"

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
qvl_compose up -g "0,1" -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf
qvl_machine run
```

### Machine 2 (2x GPU, mixed models)
```bash
qvl_compose up --gpu-configs "0:4B-Instruct:Q4_K_M,1:8B-Instruct:Q8_0"
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
