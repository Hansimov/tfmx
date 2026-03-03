# QVL Module - Hints & Tips

## Case-Insensitive Naming Convention

All model shortcuts and quant levels are **case-insensitive** internally. The convention is to use **lowercase** everywhere:

- Model shortcuts: `2b-instruct`, `4b-thinking`, `8b-instruct`, etc.
- Quant levels: `4bit`, `8bit`

Inputs in any casing are accepted (`8B-Instruct`, `8b-instruct`, `8B-INSTRUCT` all resolve the same way). Display-facing labels use original casing (e.g., `8B-Instruct:4bit`) via `get_display_shortcut()`.

## Quantization Guide

### Choosing the Right Quantization

| GPU | VRAM | Recommended Setup |
|-----|------|-------------------|
| RTX 3060 | 12GB | 8B-AWQ-4bit or 4B-FP16 |
| RTX 3070 | 8GB | 4B-AWQ-8bit or 2B-FP16 |
| RTX 3080 | 10-12GB | 8B-AWQ-4bit or 4B-FP16 |
| RTX 3090 | 24GB | 8B-FP16 or 8B-AWQ-8bit |
| RTX 4060 | 8GB | 4B-AWQ-8bit or 2B-FP16 |
| RTX 4070 | 12GB | 8B-AWQ-4bit or 4B-FP16 |
| RTX 4080 | 16GB | 8B-AWQ-8bit |
| RTX 4090 | 24GB | 8B-FP16 |

### Quantization Methods

- **AWQ** (`-q awq`): Best option for RTX 30/40 series. Uses pre-quantized AWQ models from `cyankiwi` on HuggingFace. Native vLLM support, fast inference, minimal quality loss.
- **BitsAndBytes** (`-q bitsandbytes`): Good balance. Requires bitsandbytes in container.
- **None** (default): Full FP16 precision. Requires most VRAM.

```bash
# AWQ quantization (recommended for RTX 30 series)
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q awq

# Per-GPU AWQ bit depths
qvl_compose up --gpu-configs "0:8b-instruct:4bit,1:8b-instruct:8bit"

# BitsAndBytes 4-bit
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q bitsandbytes
```

## Performance Tips

### vLLM Tuning Parameters

- **max-model-len** (default: 8096): Reduce for less VRAM usage, increase for longer context
- **max-num-seqs** (default: 5): Max concurrent sequences. Lower = less VRAM, higher = more throughput
- **limit-mm-per-prompt** (default: "image=5"): Max images per request

### Instance Count

- One vLLM instance per GPU for optimal performance
- Each instance handles internal batching via continuous batching
- Machine proxy distributes requests across instances

### Network & Proxy

If behind a firewall, use HTTP proxy for model downloads:
```bash
qvl_compose up --proxy http://127.0.0.1:11111
```

For Chinese users, HuggingFace mirror is configured by default:
```bash
# Uses https://hf-mirror.com automatically
```

## Troubleshooting

### Container Won't Start

1. Check GPU availability: `nvidia-smi`
2. Check Docker GPU runtime: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Check logs: `qvl_compose logs`

### Out of Memory (OOM)

- Use a smaller model (`-m "Qwen/Qwen3-VL-2B-Instruct"`)
- Enable quantization (`-q awq`)
- Use 4bit quant (`--gpu-configs "0:8b-instruct:4bit"`)
- Reduce `max-model-len`
- Reduce `max-num-seqs`

### Slow Startup

vLLM takes 60-120 seconds to load models. Check with:
```bash
qvl_compose logs -f  # Watch startup progress
qvl_machine health   # Check when instances become healthy
```

### Health Check Fails

- vLLM health endpoint returns 200 with empty body when healthy
- Ensure containers are fully started (wait for "Uvicorn running on" in logs)
- Check firewall: `curl http://localhost:29880/health`

### Docker Device Mounting

Two modes available:
- **nvidia-runtime** (default): Uses `nvidia` Docker runtime. Simpler.
- **manual**: Explicitly mounts GPU devices and driver libs. For systems without nvidia-runtime.

```bash
# Switch to manual mount mode if runtime mode fails
qvl_compose up --mount-mode manual
```

## Architecture Notes

### Per-GPU Model Configuration

In a multi-GPU deployment, each GPU can run a different model variant. This enables:
- Testing quality vs speed tradeoffs across models
- Running specialized models for different tasks
- Maximizing GPU utilization with appropriately-sized models

Example 6-GPU deployment:
```
GPU 0: 2b-instruct (4bit) — Fast, low quality
GPU 1: 4b-instruct (4bit) — Balanced
GPU 2: 8b-instruct (4bit) — High quality
GPU 3: 4b-thinking (4bit) — Reasoning tasks
GPU 4: 8b-instruct (8bit) — Highest quality
GPU 5: 8b-thinking (8bit) — Reasoning, high quality
```

### Model-Aware Routing

The machine proxy (`qvl_machine`) includes a router that:
1. Discovers which model each vLLM instance is running
2. Routes requests to matching instances based on `model` field
3. Falls back to any available instance if no match

Request routing formats:
- `model="8b-instruct"` → any 8B-Instruct instance
- `model="8b-instruct:8bit"` → specific quant level
- `model=""` → default/any idle instance

### Benchmark Images

Real images from HuggingFace datasets provide more realistic benchmarks than synthetic images. Download once and reuse:

```bash
python -m tfmx.qvls.benchimgs download -n 500
python -m tfmx.qvls.benchimgs info
```

Images are stored in `data/bench_images/` and automatically used by `QVLBenchImageGenerator`.

### Module Structure

```
qvls/
├── compose.py          # Docker Compose management for vLLM
├── client.py           # Sync/Async clients (OpenAI-compatible)
├── clients_core.py     # Multi-machine pipeline infrastructure
├── clients.py          # Production multi-machine client
├── clients_cli.py      # CLI infrastructure
├── clients_stats.py    # Verbose client with stats logging
├── router.py           # Model/quant-aware request routing
├── machine.py          # FastAPI load-balanced proxy (with routing)
├── scheduler.py        # Re-exports generic scheduler from teis
├── perf_tracker.py     # Re-exports generic perf tracker from teis
├── performance.py      # QVL-specific performance config
├── benchmark.py        # Throughput benchmarking
└── benchimgs.py        # Benchmark image generation (HF datasets + synthetic)
```

### Key Differences from TEI Module

| Aspect | TEI (teis) | QVL (qvls) |
|--------|-----------|------------|
| Engine | text-embeddings-inference | vLLM |
| API | `/embed`, `/lsh`, `/rerank` | `/v1/chat/completions` |
| Input | Text strings | Messages (text + images) |
| Output | Embeddings / LSH hashes | Generated text |
| Batching | Server-side batch embedding | vLLM continuous batching |
| Port base | 28880 | 29880 |
| Proxy port | 28800 | 29800 |
| Docker image | ghcr.io/huggingface/text-embeddings-inference | vllm/vllm-openai |

### Shared Components

`scheduler.py` and `perf_tracker.py` re-export from the `teis` module since they are fully generic (use TypeVar, no model-specific logic). All other files are qvls-specific implementations.
