#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_tei_gpus)"
if [[ -z "$deploy_gpus" ]]; then
    echo "[tei-runs] no visible GPUs detected" >&2
    exit 1
fi

echo "[tei-runs] deploying TEI on GPUs: $deploy_gpus"

compose_args=()
if [[ -n "${TEI_PROXY_URL:-}" ]]; then
    compose_args+=(--proxy "$TEI_PROXY_URL")
fi

cd "$TEI_REPO_ROOT"
tei_cmd compose up -g "$deploy_gpus" "${compose_args[@]}"