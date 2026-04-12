#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_qsr_gpus)"
if [[ -z "$deploy_gpus" ]]; then
    echo "[qsr-runs] no visible GPUs detected" >&2
    exit 1
fi

echo "[qsr-runs] deploying QSR on GPUs: $deploy_gpus"

compose_args=(--gpu-layout uniform -g "$deploy_gpus")
if [[ -n "${QSR_PROXY_URL:-}" ]]; then
    compose_args+=(--proxy "$QSR_PROXY_URL")
fi
if [[ -n "${QSR_HF_ENDPOINT:-}" ]]; then
    compose_args+=(--hf-endpoint "$QSR_HF_ENDPOINT")
fi
if [[ -n "${QSR_PIP_INDEX_URL:-}" ]]; then
    compose_args+=(--pip-index-url "$QSR_PIP_INDEX_URL")
fi
if [[ -n "${QSR_PIP_TRUSTED_HOST:-}" ]]; then
    compose_args+=(--pip-trusted-host "$QSR_PIP_TRUSTED_HOST")
fi
if [[ -n "${QSR_MAX_MODEL_LEN:-}" ]]; then
    compose_args+=(--max-model-len "$QSR_MAX_MODEL_LEN")
fi
if [[ -n "${QSR_MAX_NUM_SEQS:-}" ]]; then
    compose_args+=(--max-num-seqs "$QSR_MAX_NUM_SEQS")
fi
if [[ -n "${QSR_GPU_MEMORY_UTILIZATION:-}" ]]; then
    compose_args+=(--gpu-memory-utilization "$QSR_GPU_MEMORY_UTILIZATION")
fi

cd "$QSR_REPO_ROOT"
qsr_cmd compose up "${compose_args[@]}"