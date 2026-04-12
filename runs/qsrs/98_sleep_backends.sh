#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

cd "$QSR_REPO_ROOT"
qsr_cmd machine stop || true
qsr_cmd compose sleep \
    --sleep-level "${QSR_SLEEP_LEVEL:-1}" \
    --sleep-mode "${QSR_SLEEP_MODE:-abort}" || true