#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_qsr_gpus)"
project_name="$(resolve_qsr_project_name)"

qsr_cmd compose down --project-name "$project_name" -g "$deploy_gpus" || true
qsr_cmd machine stop || true