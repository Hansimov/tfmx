#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

audio_a="$(materialize_qsr_audio "${QSR_SOAK_AUDIO_A:-${QSR_BENCH_AUDIO:-$QSR_DEFAULT_AUDIO}}")"
audio_b="$(materialize_qsr_audio "${QSR_SOAK_AUDIO_B:-${QSR_SCHED_AUDIO_A:-$QSR_ALT_AUDIO}}")"
num_transcribe="${QSR_SOAK_TRANSCRIBE_SAMPLES:-1200}"
num_chat="${QSR_SOAK_CHAT_SAMPLES:-400}"
max_workers="${QSR_SOAK_MAX_WORKERS:-32}"
max_tokens="${QSR_SOAK_MAX_TOKENS:-128}"
output_path="$(result_json_path qsr_mixed_soak)"

cd "$QSR_REPO_ROOT"
echo "[qsr-runs] mixed soak output: $output_path"
python debugs/qsrs/soak_mixed.py \
    -E "$QSR_MACHINE_URL" \
    --audio "$audio_a" \
    --audio "$audio_b" \
    --num-transcribe "$num_transcribe" \
    --num-chat "$num_chat" \
    --max-workers "$max_workers" \
    --max-tokens "$max_tokens" \
    -o "$output_path"