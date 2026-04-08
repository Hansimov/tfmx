#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
	if command -v qwn >/dev/null 2>&1; then
		qwn "$@"
	else
		python -m tfmx.qwns.cli "$@"
	fi
}

if [[ "${QWN_WAKE_BACKENDS:-1}" != "0" ]]; then
	qwn_cmd compose wake --wait-healthy || true
fi

qwn_cmd machine run -b --on-conflict replace
qwn_cmd machine status