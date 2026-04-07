#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

deploy_gpus="$(resolve_tei_gpus)"
expected_total="$(count_csv_items "$deploy_gpus")"

if [[ "$expected_total" -eq 0 ]]; then
    echo "[tei-runs] no GPUs selected for TEI machine startup" >&2
    exit 1
fi

wait_backend_health "$deploy_gpus"
start_tei_machine_background
wait_machine_health "$expected_total"

echo "[tei-runs] tei machine log: $TEI_LOG_FILE"
echo "[tei-runs] tei machine pid: $(cat "$TEI_PID_FILE")"