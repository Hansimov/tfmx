#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
	if command -v qwn >/dev/null 2>&1; then
		qwn "$@"
	else
		python -m tfmx.qwns.cli "$@"
	fi
}

qwn_cmd machine stop || true
qwn_cmd compose sleep \
	--sleep-level "${QWN_SLEEP_LEVEL:-1}" \
	--sleep-mode "${QWN_SLEEP_MODE:-abort}" || true