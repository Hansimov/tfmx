#!/usr/bin/env bash

QSR_RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QSR_REPO_ROOT="$(cd "$QSR_RUNS_DIR/../.." && pwd)"
QSR_RESULTS_DIR="$QSR_REPO_ROOT/runs/qsrs/results"
QSR_CACHE_DIR="${QSR_CACHE_DIR:-$HOME/.cache/tfmx/qsrs}"
QSR_AUDIO_CACHE_DIR="$QSR_CACHE_DIR/audio"
QSR_MACHINE_URL="${QSR_MACHINE_URL:-http://127.0.0.1:27900}"
QSR_MACHINE_PORT="${QSR_MACHINE_PORT:-27900}"
QSR_BACKEND_BASE_PORT="${QSR_BACKEND_BASE_PORT:-27980}"
QSR_DEFAULT_AUDIO="${QSR_DEFAULT_AUDIO:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav}"
QSR_ALT_AUDIO="${QSR_ALT_AUDIO:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav}"
QSR_GPU_LAYOUT="${QSR_GPU_LAYOUT:-uniform}"
QSR_PROJECT_NAME="${QSR_PROJECT_NAME:-}"
QSR_ENABLE_SLEEP_MODE="${QSR_ENABLE_SLEEP_MODE:-1}"
QSR_PROFILE_STARTUP="${QSR_PROFILE_STARTUP:-0}"

mkdir -p "$QSR_RESULTS_DIR"
mkdir -p "$QSR_AUDIO_CACHE_DIR"

qsr_cmd() {
    if command -v qsr >/dev/null 2>&1; then
        qsr "$@"
    else
        python -m tfmx.qsrs.cli "$@"
    fi
}

materialize_qsr_audio() {
    local source="$1"
    if [[ "$source" == http://* || "$source" == https://* ]]; then
        local source_without_query="${source%%\?*}"
        local base_name="${source_without_query##*/}"
        local cache_key
        local target_path
        local tmp_path

        [[ -n "$base_name" ]] || base_name="audio.wav"
        cache_key="$(printf '%s' "$source" | sha256sum | awk '{print substr($1, 1, 16)}')"
        target_path="$QSR_AUDIO_CACHE_DIR/${cache_key}_${base_name}"
        if [[ ! -s "$target_path" ]]; then
            tmp_path="${target_path}.tmp"
            rm -f "$tmp_path"
            curl -L --fail -o "$tmp_path" "$source"
            mv "$tmp_path" "$target_path"
        fi
        printf '%s\n' "$target_path"
        return
    fi

    printf '%s\n' "$source"
}

detect_visible_gpu_csv() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
}

resolve_qsr_gpus() {
    if [[ -n "${QSR_DEPLOY_GPUS:-}" ]]; then
        printf '%s\n' "$QSR_DEPLOY_GPUS"
        return
    fi
    detect_visible_gpu_csv
}

resolve_qsr_project_name() {
    if [[ -n "$QSR_PROJECT_NAME" ]]; then
        printf '%s\n' "$QSR_PROJECT_NAME"
        return
    fi

    printf 'qsr-%s\n' "$(printf '%s' "$QSR_GPU_LAYOUT" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/-/g')"
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
    python3 - "$QSR_BACKEND_BASE_PORT" "$gpu_csv" "$timeout_sec" <<'PY'
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
        print(f'[qsr-runs] backend health {state}', flush=True)
        last_state = state

    if all_ok:
        print('[qsr-runs] backends healthy', flush=True)
        sys.exit(0)

    time.sleep(2)

print(f'[qsr-runs] backend health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}

wait_machine_health() {
    local expected_total="$1"
    local timeout_sec="${2:-240}"
    python3 - "$QSR_MACHINE_URL/health" "$expected_total" "$timeout_sec" <<'PY'
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
            print(f'[qsr-runs] machine health {state}', flush=True)
            last_state = state
        if (
            body.get('status') == 'healthy'
            and body.get('healthy') == expected_total
            and body.get('total') == expected_total
        ):
            print('[qsr-runs] machine healthy', flush=True)
            sys.exit(0)
    except Exception as exc:
        state = f'wait:{type(exc).__name__}'
        if state != last_state:
            print(f'[qsr-runs] machine health {state}', flush=True)
            last_state = state
    time.sleep(2)

print(f'[qsr-runs] machine health timed out after {timeout_sec:.0f}s', file=sys.stderr)
sys.exit(1)
PY
}

result_json_path() {
    local prefix="$1"
    printf '%s/%s_%s.json\n' "$QSR_RESULTS_DIR" "$prefix" "$(date +%Y%m%d_%H%M%S)"
}