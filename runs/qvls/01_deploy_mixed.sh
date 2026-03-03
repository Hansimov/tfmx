#!/usr/bin/env bash
# Deploy 6 vLLM instances with different model/quant combos (AWQ)
#
# GPU 0: 2b-instruct:4bit  (fast, low quality)
# GPU 1: 4b-instruct:4bit  (balanced)
# GPU 2: 8b-instruct:4bit  (high quality)
# GPU 3: 4b-thinking:4bit  (reasoning, balanced)
# GPU 4: 8b-instruct:8bit  (highest quality instruct)
# GPU 5: 8b-thinking:8bit  (reasoning, high quality)
set -euo pipefail

GPU_CONFIGS="0:2b-instruct:4bit,1:4b-instruct:4bit,2:8b-instruct:4bit,3:4b-thinking:4bit,4:8b-instruct:8bit,5:8b-thinking:8bit"

echo "=== Deploying 6-GPU mixed models (AWQ) ==="
echo "GPU configs: ${GPU_CONFIGS}"
echo ""

qvl_compose up --gpu-configs "${GPU_CONFIGS}" --mount-mode manual

echo ""
echo "=== Deployment started ==="
echo "Wait ~60-120s for models to load, then check:"
echo "  qvl_compose ps"
echo "  qvl_compose logs -f"
