#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

mode="${QSR_BENCH_MODE:-transcribe}"
num_samples="${QSR_BENCH_SAMPLES:-20}"
audio="${QSR_BENCH_AUDIO:-$QSR_DEFAULT_AUDIO}"
prompt="${QSR_BENCH_PROMPT:-请转写为简体中文。}"
output_path="$(result_json_path qsr_benchmark)"

cmd=(qsr_cmd benchmark run -E "$QSR_MACHINE_URL" -n "$num_samples" --mode "$mode" --audio "$audio" -o "$output_path")

if [[ -n "${QSR_BENCH_MODEL:-}" ]]; then
    cmd+=(--model "$QSR_BENCH_MODEL")
fi

if [[ "$mode" == "chat" ]]; then
    cmd+=(--prompt "$prompt")
    if [[ "${QSR_BENCH_NO_TTFT:-0}" == "1" ]]; then
        cmd+=(--no-ttft)
    fi
else
    if [[ -n "${QSR_BENCH_LANGUAGE:-}" ]]; then
        cmd+=(--language "$QSR_BENCH_LANGUAGE")
    fi
    if [[ -n "${QSR_BENCH_RESPONSE_FORMAT:-}" ]]; then
        cmd+=(--response-format "$QSR_BENCH_RESPONSE_FORMAT")
    fi
fi

cd "$QSR_REPO_ROOT"
echo "[qsr-runs] benchmark output: $output_path"
"${cmd[@]}"