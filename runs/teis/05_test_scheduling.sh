#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

cd "$TEI_REPO_ROOT"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== TEI pipeline throughput benchmark ==="
echo "Endpoint: $TEI_MACHINE_URL"
echo

echo "--- Warmup (2,000 samples) ---"
tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n 2000 2>&1 | tail -5
echo

echo "--- Test A: 10,000 samples ---"
tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n 10000 \
    -o "$TEI_RESULTS_DIR/bench_pipeline_10000_${timestamp}.json"
echo

echo "--- Test B: 20,000 samples ---"
tei_cmd benchmark run -E "$TEI_MACHINE_URL" -n 20000 \
    -o "$TEI_RESULTS_DIR/bench_pipeline_20000_${timestamp}.json"
echo

ls -la "$TEI_RESULTS_DIR"/bench_pipeline_*_"$timestamp".json