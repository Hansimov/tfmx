#!/usr/bin/env bash
# Download all required AWQ models for 6-GPU deployment
# Uses `hf` CLI (huggingface_hub >= 0.25): pip install "huggingface_hub[cli]"
#
# Models needed:
#   GPU 0: 2b-instruct:4bit   (cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit)
#   GPU 1: 2b-thinking:4bit   (cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit)
#   GPU 2: 4b-instruct:4bit   (cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit)
#   GPU 3: 4b-thinking:4bit   (cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit)
#   GPU 4: 8b-instruct:4bit   (cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit)
#   GPU 5: 8b-thinking:4bit   (cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit)
#
# AWQ models are complete HF repos — vLLM auto-detects compressed-tensors quantization
set -euo pipefail

# Use HF mirror for China
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "Using HF_ENDPOINT=$HF_ENDPOINT"

AWQ_REPOS=(
    "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"
    "cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit"
    "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit"
    "cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit"
    "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
    "cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit"
)

echo "=== Downloading AWQ models ==="
for repo in "${AWQ_REPOS[@]}"; do
    echo ""
    echo "--- ${repo} ---"
    hf download "${repo}" || echo "  WARN: failed ${repo}"
done

echo ""
echo "=== Download complete ==="
echo "Cache: ~/.cache/huggingface/hub/"
echo ""
echo "AWQ model repos:"
for repo in "${AWQ_REPOS[@]}"; do
    repo_dash="${repo//\//-}"
    dir=$(ls -d ~/.cache/huggingface/hub/models--${repo_dash//\/--}/snapshots/* 2>/dev/null | head -1)
    if [[ -n "${dir:-}" ]]; then
        echo "  ${repo}: OK"
    else
        echo "  ${repo}: MISSING"
    fi
done
