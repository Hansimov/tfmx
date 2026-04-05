#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
	if command -v qwn >/dev/null 2>&1; then
		qwn "$@"
	else
		python -m tfmx.qwns.cli "$@"
	fi
}

mkdir -p runs/qwns/results
output="runs/qwns/results/bench_$(date +%Y%m%d_%H%M%S).json"
qwn_cmd benchmark run -E http://localhost:27800 -n 20 --max-tokens 64 -o "$output"
echo "$output"