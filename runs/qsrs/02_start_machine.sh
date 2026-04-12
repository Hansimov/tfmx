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
qsr_cmd machine run \
    --auto-start \
    -b \
    --compose-gpus "$deploy_gpus" \
    --compose-gpu-layout uniform \
    --on-conflict replace

wait_machine_health "$expected_total"

if [[ "${QSR_SKIP_WARMUP:-0}" != "1" ]]; then
    warmup_audio="$(materialize_qsr_audio "${QSR_WARMUP_AUDIO:-$QSR_DEFAULT_AUDIO}")"
    echo "[qsr-runs] warming up qsr machine with: $warmup_audio"
    qsr_cmd client transcribe -E "$QSR_MACHINE_URL" "$warmup_audio" --response-format text >/dev/null
fi

qsr_cmd machine status

echo "[qsr-runs] qsr machine log: $HOME/.cache/tfmx/qsr_machine.log"
echo "[qsr-runs] qsr machine pid: $HOME/.cache/tfmx/qsr_machine.pid"