#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
    if command -v qwn >/dev/null 2>&1; then
        qwn "$@"
    else
        python -m tfmx.qwns.cli "$@"
    fi
}

compose_args=()
if [[ -n "${QWN_PROXY:-}" ]]; then
    compose_args+=(--proxy "$QWN_PROXY")
fi
if [[ "${QWN_ENABLE_SLEEP_MODE:-1}" != "0" ]]; then
    compose_args+=(--enable-sleep-mode)
fi

qwn_cmd compose up --gpu-layout uniform-awq "${compose_args[@]}"