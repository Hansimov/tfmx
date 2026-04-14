#!/usr/bin/env bash

TEI_RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEI_REPO_ROOT="$(cd "$TEI_RUNS_DIR/../.." && pwd)"
TEI_RESULTS_DIR="$TEI_REPO_ROOT/runs/teis/results"
TEI_MACHINE_URL="${TEI_MACHINE_URL:-http://127.0.0.1:28800}"
TEI_MACHINE_PORT="${TEI_MACHINE_PORT:-28800}"
TEI_BACKEND_BASE_PORT="${TEI_BACKEND_BASE_PORT:-28880}"

mkdir -p "$TEI_RESULTS_DIR"

tei_cmd() {
    if command -v tei >/dev/null 2>&1; then
        tei "$@"
    else
        python -m tfmx.teis.cli "$@"
    fi
}

detect_visible_gpu_csv() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
}

resolve_tei_gpus() {
    if [[ -n "${TEI_DEPLOY_GPUS:-}" ]]; then
        printf '%s\n' "$TEI_DEPLOY_GPUS"
        return
    fi
    detect_visible_gpu_csv
}

count_csv_items() {
    local csv="$1"
    local count=0
    local value
    IFS=',' read -r -a values <<< "$csv"
    for value in "${values[@]}"; do
        value="${value//[[:space:]]/}"
        [[ -z "$value" ]] && continue
        count=$((count + 1))
    done
    printf '%s\n' "$count"
}

wait_backend_health() {
    local gpu_csv="$1"
    local timeout_sec="${2:-600}"
    python3 - "$TEI_BACKEND_BASE_PORT" "$gpu_csv" "$timeout_sec" <<'PY'
import sys
import time
import urllib.request

base_port = int(sys.argv[1])
gpu_csv = sys.argv[2]
timeout_sec = float(sys.argv[3])
gpus = [int(part.strip()) for part in gpu_csv.split(',') if part.strip()]
ports = [base_port + gpu for gpu in gpus]
deadline = time.time() + timeout_sec
last_state = None

while time.time() < deadline:
    statuses = []
    all_ok = True
    for port in ports:
        ok = False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as resp:
                ok = 200 <= resp.status < 300
        except Exception:
            ok = False
        statuses.append(f'{port}:{"ok" if ok else "wait"}')
        all_ok = all_ok and ok

    state = ' '.join(statuses)
    if state != last_state:
        print(f'[tei-runs] backend health {state}', flush=True)
        last_state = state

    if all_ok:
        print('[tei-runs] backends healthy', flush=True)
        sys.exit(0)

    time.sleep(2)

print(f'[tei-runs] backend health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}

wait_machine_health() {
    local expected_total="$1"
    local timeout_sec="${2:-180}"
    python3 - "$TEI_MACHINE_URL/health" "$expected_total" "$timeout_sec" <<'PY'
import json
import sys
import time
import urllib.request

health_url = sys.argv[1]
expected_total = int(sys.argv[2])
timeout_sec = float(sys.argv[3])
deadline = time.time() + timeout_sec
last_state = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            body = json.loads(resp.read().decode())
        state = f"{body.get('healthy')}/{body.get('total')} {body.get('status')}"
        if state != last_state:
            print(f'[tei-runs] machine health {state}', flush=True)
            last_state = state
        if (
            body.get('status') == 'healthy'
            and body.get('healthy') == expected_total
            and body.get('total') == expected_total
        ):
            print('[tei-runs] machine healthy', flush=True)
            sys.exit(0)
    except Exception as exc:
        state = f'wait:{type(exc).__name__}'
        if state != last_state:
            print(f'[tei-runs] machine health {state}', flush=True)
            last_state = state
    time.sleep(2)

print(f'[tei-runs] machine health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}