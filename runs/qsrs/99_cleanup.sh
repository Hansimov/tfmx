#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_qsr_gpus)"

qsr_cmd compose down -g "$deploy_gpus" || true
qsr_cmd machine stop || true