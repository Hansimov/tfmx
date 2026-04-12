#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_qsr_gpus)"
expected_total="$(count_csv_items "$deploy_gpus")"

if [[ "$expected_total" -eq 0 ]]; then
    echo "[qsr-runs] no GPUs selected for QSR machine startup" >&2
    exit 1
fi

cd "$QSR_REPO_ROOT"
if [[ "${QSR_ENABLE_SLEEP_MODE:-0}" == "1" ]]; then
    qsr_cmd compose wake \
        --gpu-layout uniform \
        -g "$deploy_gpus" \
        --wait-healthy || true
fi

machine_args=(
    --auto-start
    -b
    --compose-gpus "$deploy_gpus"
    --compose-gpu-layout uniform
    --on-conflict replace
)
if [[ "${QSR_ENABLE_SLEEP_MODE:-0}" == "1" ]]; then
    machine_args+=(--compose-enable-sleep-mode)
fi

qsr_cmd machine run \
    "${machine_args[@]}"

wait_machine_health "$expected_total"

qsr_cmd machine status

echo "[qsr-runs] qsr machine log: $HOME/.cache/tfmx/qsr_machine.log"
echo "[qsr-runs] qsr machine pid: $HOME/.cache/tfmx/qsr_machine.pid"