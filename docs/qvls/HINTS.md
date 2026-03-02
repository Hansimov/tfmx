# QVL Module - Hints & Tips

## Quantization Guide

### Choosing the Right Quantization

| GPU | VRAM | Recommended Setup |
|-----|------|-------------------|
| RTX 3060 | 12GB | 8B-Q4 or 4B-FP16 |
| RTX 3070 | 8GB | 4B-Q8 or 2B-FP16 |
| RTX 3080 | 10-12GB | 8B-Q4 or 4B-FP16 |
| RTX 3090 | 24GB | 8B-FP16 or 8B-Q8 |
| RTX 4060 | 8GB | 4B-Q8 or 2B-FP16 |
| RTX 4070 | 12GB | 8B-Q4 or 4B-FP16 |
| RTX 4080 | 16GB | 8B-Q8 |
| RTX 4090 | 24GB | 8B-FP16 |

### Quantization Methods

- **GGUF** (`-q gguf`): Best compression, slight quality loss. Uses unsloth GGUF models.
- **BitsAndBytes** (`-q bitsandbytes`): Good balance. Requires bitsandbytes in container.
- **AWQ** (`-q awq`): Fast inference. Requires pre-quantized AWQ model.
- **None** (default): Full FP16 precision. Requires most VRAM.

```bash
# GGUF quantization (recommended for RTX 30 series)
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf

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
- Enable quantization (`-q gguf`)
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

### Module Structure

```
qvls/
├── compose.py          # Docker Compose management for vLLM
├── client.py           # Sync/Async clients (OpenAI-compatible)
├── clients_core.py     # Multi-machine pipeline infrastructure
├── clients.py          # Production multi-machine client
├── clients_cli.py      # CLI infrastructure
├── clients_stats.py    # Verbose client with stats logging
├── machine.py          # FastAPI load-balanced proxy
├── scheduler.py        # Re-exports generic scheduler from teis
├── perf_tracker.py     # Re-exports generic perf tracker from teis
├── performance.py      # QVL-specific performance config
├── benchmark.py        # Throughput benchmarking
└── benchimgs.py        # Synthetic image+prompt generation
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
