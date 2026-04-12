#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

smoke_audio="${QSR_SMOKE_AUDIO:-$QSR_DEFAULT_AUDIO}"

cd "$QSR_REPO_ROOT"
qsr_cmd client health -E "$QSR_MACHINE_URL"
qsr_cmd client models -E "$QSR_MACHINE_URL"
qsr_cmd client info -E "$QSR_MACHINE_URL"

if [[ "${QSR_SKIP_TRANSCRIBE_SMOKE:-0}" != "1" ]]; then
    qsr_cmd client transcribe -E "$QSR_MACHINE_URL" "$smoke_audio" --response-format text
fi