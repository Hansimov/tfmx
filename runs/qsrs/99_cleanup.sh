#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_qsr_gpus)"

cd "$QSR_REPO_ROOT"
qsr_cmd machine stop || true
qsr_cmd compose down --gpu-layout uniform -g "$deploy_gpus" || true