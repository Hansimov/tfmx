#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/runs/recovery/results}"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

TEI_MACHINE_URL="${TEI_MACHINE_URL:-http://127.0.0.1:28800}"
QSR_MACHINE_URL="${QSR_MACHINE_URL:-http://127.0.0.1:27900}"

TEI_BENCH_SAMPLES="${TEI_BENCH_SAMPLES:-12000}"
QSR_BENCH_SAMPLES="${QSR_BENCH_SAMPLES:-1200}"
QSR_BENCH_MODE="${QSR_BENCH_MODE:-transcribe}"
QSR_BENCH_PROMPT="${QSR_BENCH_PROMPT:-请转写为简体中文。}"
QSR_BENCH_NO_TTFT="${QSR_BENCH_NO_TTFT:-0}"
QSR_BENCH_AUDIO="${QSR_BENCH_AUDIO:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav}"

TEI_OUTPUT="$REPO_ROOT/runs/teis/results/coexist_${TIMESTAMP}.json"
QSR_OUTPUT="$REPO_ROOT/runs/qsrs/results/coexist_${TIMESTAMP}.json"
TEI_LOG="$RESULTS_DIR/tei_coexist_${TIMESTAMP}.log"
QSR_LOG="$RESULTS_DIR/qsr_coexist_${TIMESTAMP}.log"
PRE_STATE="$RESULTS_DIR/tei_qsr_coexist_pre_${TIMESTAMP}.json"
POST_STATE="$RESULTS_DIR/tei_qsr_coexist_post_${TIMESTAMP}.json"
SUMMARY_MD="$RESULTS_DIR/tei_qsr_coexist_${TIMESTAMP}.md"

log() {
    printf '[tei-qsr-coexist] %s\n' "$*"
}

tei_cmd() {
    if command -v tei >/dev/null 2>&1; then
        tei "$@"
    else
        python -m tfmx.teis.cli "$@"
    fi
}

qsr_cmd() {
    if command -v qsr >/dev/null 2>&1; then
        qsr "$@"
    else
        python -m tfmx.qsrs.cli "$@"
    fi
}

materialize_qsr_audio() {
    local source="$1"
    local cache_dir="${QSR_AUDIO_CACHE_DIR:-$HOME/.cache/tfmx/qsrs/audio}"
    mkdir -p "$cache_dir"

    if [[ "$source" == http://* || "$source" == https://* ]]; then
        local source_without_query="${source%%\?*}"
        local base_name="${source_without_query##*/}"
        local cache_key
        local target_path
        local tmp_path

        [[ -n "$base_name" ]] || base_name="audio.wav"
        cache_key="$(printf '%s' "$source" | sha256sum | awk '{print substr($1, 1, 16)}')"
        target_path="$cache_dir/${cache_key}_${base_name}"
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

capture_state() {
    local output_path="$1"
    python3 - "$TEI_MACHINE_URL" "$QSR_MACHINE_URL" "$output_path" <<'PY'
import json
import sys
import urllib.request

tei_url = sys.argv[1]
qsr_url = sys.argv[2]
output_path = sys.argv[3]

def fetch(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())

payload = {
    "tei": {
        "health": fetch(f"{tei_url}/health"),
        "info": fetch(f"{tei_url}/info"),
    },
    "qsr": {
        "health": fetch(f"{qsr_url}/health"),
        "info": fetch(f"{qsr_url}/info"),
    },
}

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

run_post_health_checks() {
    log "running post-benchmark TEI health check"
    tei_cmd client health >/dev/null
    log "running post-benchmark QSR health check"
    qsr_cmd client health -E "$QSR_MACHINE_URL" >/dev/null
}

summarize_results() {
    python3 - "$PRE_STATE" "$POST_STATE" "$TEI_OUTPUT" "$QSR_OUTPUT" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

pre_path = Path(sys.argv[1])
post_path = Path(sys.argv[2])
tei_path = Path(sys.argv[3])
qsr_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])

pre = json.loads(pre_path.read_text())
post = json.loads(post_path.read_text())
tei = json.loads(tei_path.read_text())
qsr = json.loads(qsr_path.read_text())

def health_tuple(state, name):
    health = state[name]["health"]
    return health.get("healthy", 0), health.get("total", 0), health.get("status", "unknown")

def stats_delta(service: str, key: str) -> int:
    pre_value = pre[service]["info"]["stats"].get(key, 0)
    post_value = post[service]["info"]["stats"].get(key, 0)
    return post_value - pre_value

def requests_per_instance_delta(service: str):
    pre_map = pre[service]["info"]["stats"].get("requests_per_instance", {})
    post_map = post[service]["info"]["stats"].get("requests_per_instance", {})
    keys = sorted(set(pre_map) | set(post_map))
    return {key: post_map.get(key, 0) - pre_map.get(key, 0) for key in keys}

tei_h = health_tuple(post, "tei")
qsr_h = health_tuple(post, "qsr")
tei_errors_delta = stats_delta("tei", "total_errors")
qsr_errors_delta = stats_delta("qsr", "total_errors")
qsr_failovers_delta = stats_delta("qsr", "total_failovers")

stable = (
    tei_h[0] == tei_h[1]
    and qsr_h[0] == qsr_h[1]
    and tei_errors_delta == 0
    and qsr_errors_delta == 0
    and qsr_failovers_delta == 0
    and qsr["requests"]["failed"] == 0
)

lines = [
    "# TEI + QSR Coexist Benchmark",
    "",
    "## Stability",
    "",
    f"- Overall verdict: {'stable' if stable else 'needs review'}",
    f"- TEI post health: {tei_h[0]}/{tei_h[1]} {tei_h[2]}",
    f"- QSR post health: {qsr_h[0]}/{qsr_h[1]} {qsr_h[2]}",
    f"- TEI machine error delta: {tei_errors_delta}",
    f"- QSR machine error delta: {qsr_errors_delta}",
    f"- QSR failover delta: {qsr_failovers_delta}",
    "",
    "## Throughput",
    "",
    f"- TEI samples: {tei['config']['n_samples']}",
    f"- TEI total time: {tei['timing']['total_time_sec']} s",
    f"- TEI throughput: {tei['throughput']['samples_per_second']} samples/s",
    f"- QSR requests: {qsr['requests']['submitted']}",
    f"- QSR total time: {qsr['timing']['total_time_sec']} s",
    f"- QSR throughput: {qsr['throughput']['requests_per_second']} req/s",
    f"- QSR success rate: {qsr['requests']['success_rate']}",
    "",
    "## Routing Deltas",
    "",
    f"- TEI requests_per_instance delta: {requests_per_instance_delta('tei')}",
    f"- QSR requests_per_instance delta: {requests_per_instance_delta('qsr')}",
    "",
    "## Artifacts",
    "",
    f"- TEI benchmark: {tei_path}",
    f"- QSR benchmark: {qsr_path}",
    f"- Pre-state: {pre_path}",
    f"- Post-state: {post_path}",
]

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
PY
}

log "capturing pre-benchmark machine state"
capture_state "$PRE_STATE"

QSR_AUDIO_LOCAL="$(materialize_qsr_audio "$QSR_BENCH_AUDIO")"

TEI_PID=""
QSR_PID=""

cleanup_on_error() {
    local exit_code=$?
    if [[ "$exit_code" -eq 0 ]]; then
        return
    fi
    if [[ -n "$TEI_PID" ]] && kill -0 "$TEI_PID" 2>/dev/null; then
        kill "$TEI_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$QSR_PID" ]] && kill -0 "$QSR_PID" 2>/dev/null; then
        kill "$QSR_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup_on_error EXIT INT TERM

log "starting TEI benchmark in background: ${TEI_BENCH_SAMPLES} samples"
(
    cd "$REPO_ROOT"
    tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n "$TEI_BENCH_SAMPLES" -o "$TEI_OUTPUT"
) >"$TEI_LOG" 2>&1 &
TEI_PID=$!

qsr_cmdline=(benchmark run -E "$QSR_MACHINE_URL" -n "$QSR_BENCH_SAMPLES" --mode "$QSR_BENCH_MODE" --audio "$QSR_AUDIO_LOCAL" -o "$QSR_OUTPUT")
if [[ "$QSR_BENCH_MODE" == "chat" ]]; then
    qsr_cmdline+=(--prompt "$QSR_BENCH_PROMPT")
    if [[ "$QSR_BENCH_NO_TTFT" == "1" ]]; then
        qsr_cmdline+=(--no-ttft)
    fi
fi

log "starting QSR benchmark in background: ${QSR_BENCH_SAMPLES} ${QSR_BENCH_MODE} requests"
(
    cd "$REPO_ROOT"
    qsr_cmd "${qsr_cmdline[@]}"
) >"$QSR_LOG" 2>&1 &
QSR_PID=$!

wait "$TEI_PID"
wait "$QSR_PID"
TEI_PID=""
QSR_PID=""

log "capturing post-benchmark machine state"
capture_state "$POST_STATE"
run_post_health_checks

log "writing coexist summary"
summarize_results

log "summary: $SUMMARY_MD"
log "tei benchmark: $TEI_OUTPUT"
log "qsr benchmark: $QSR_OUTPUT"
log "tei log: $TEI_LOG"
log "qsr log: $QSR_LOG"