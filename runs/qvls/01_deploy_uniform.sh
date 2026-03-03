#!/usr/bin/env bash
# Deploy 6 GPUs with UNIFORM model config for scheduling throughput test
# All GPUs run the same model: 4b-instruct:4bit (AWQ)
set -euo pipefail

GPU_CONFIGS="0:4b-instruct:4bit,1:4b-instruct:4bit,2:4b-instruct:4bit,3:4b-instruct:4bit,4:4b-instruct:4bit,5:4b-instruct:4bit"

echo "=== Deploying 6-GPU uniform (4b-instruct:4bit AWQ) ==="
echo "GPU configs: ${GPU_CONFIGS}"
echo ""

qvl_compose up --gpu-configs "${GPU_CONFIGS}" --mount-mode manual

echo ""
echo "=== Uniform deployment started ==="
echo "Wait ~60-120s for models to load, then check:"
echo "  qvl_compose ps"
echo "  qvl_compose logs -f"
