#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
	if command -v qwn >/dev/null 2>&1; then
		qwn "$@"
	else
		python -m tfmx.qwns.cli "$@"
	fi
}

qwn_cmd client health
qwn_cmd client models