#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

num_samples="${QSR_SCHED_SAMPLES:-40}"
audio_a="${QSR_SCHED_AUDIO_A:-$QSR_DEFAULT_AUDIO}"
audio_b="${QSR_SCHED_AUDIO_B:-$QSR_ALT_AUDIO}"
prompt="${QSR_SCHED_PROMPT:-请先转写音频，再简要说明这是哪种语言。}"
output_path="$(result_json_path qsr_scheduling)"

cd "$QSR_REPO_ROOT"
echo "[qsr-runs] scheduling benchmark output: $output_path"
qsr_cmd benchmark run \
    -E "$QSR_MACHINE_URL" \
    -n "$num_samples" \
    --mode chat \
    --audio "$audio_a" \
    --audio "$audio_b" \
    --prompt "$prompt" \
    -o "$output_path"

qsr_cmd client info -E "$QSR_MACHINE_URL"