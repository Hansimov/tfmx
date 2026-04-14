#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_tei_gpus)"
expected_total="$(count_csv_items "$deploy_gpus")"

if [[ "$expected_total" -eq 0 ]]; then
    echo "[tei-runs] no GPUs selected for TEI machine startup" >&2
    exit 1
fi

cd "$TEI_REPO_ROOT"
tei_cmd machine run \
    --background \
    --auto-start \
    --perf-track \
    --compose-gpus "$deploy_gpus" \
    --on-conflict replace

wait_machine_health "$expected_total"

echo "[tei-runs] tei machine log: $HOME/.cache/tfmx/tei_machine.log"
echo "[tei-runs] tei machine pid: $HOME/.cache/tfmx/tei_machine.pid"