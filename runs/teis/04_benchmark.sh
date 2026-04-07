#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

mkdir -p "$TEI_RESULTS_DIR"
output="$TEI_RESULTS_DIR/bench_$(date +%Y%m%d_%H%M%S).json"

cd "$TEI_REPO_ROOT"
tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n 5000 -o "$output"
echo "$output"