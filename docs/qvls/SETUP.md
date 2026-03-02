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

- `unsloth/Qwen3-VL-2B-Instruct-GGUF`
- `unsloth/Qwen3-VL-4B-Instruct-GGUF`
- `unsloth/Qwen3-VL-8B-Instruct-GGUF`

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

# For GGUF quantized model
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF
```

### 3. Deploy with Docker Compose

```bash
# Start on all available GPUs (default model: Qwen3-VL-8B-Instruct)
qvl_compose up

# Start with specific model and quantization
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf

# Start on specific GPUs
qvl_compose up -g "0,1"

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

### Machine 2 (2x GPU)
```bash
qvl_compose up -g "0,1" -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf
qvl_machine run
```

### Client (any machine)
```bash
qvl_clients health --endpoints http://machine1:29800 http://machine2:29800
qvl_benchmark run -E "http://machine1:29800,http://machine2:29800" -n 100
```
