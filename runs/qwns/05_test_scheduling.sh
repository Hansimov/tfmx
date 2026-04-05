#!/usr/bin/env bash
set -euo pipefail

qwn_cmd() {
    if command -v qwn >/dev/null 2>&1; then
        qwn "$@"
    else
        python -m tfmx.qwns.cli "$@"
    fi
}

ENDPOINT="http://localhost:27800"
RESULTS_DIR="runs/qwns/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Scheduling throughput benchmark (uniform-awq) ==="
echo "Endpoint: ${ENDPOINT}"
echo ""

echo "--- Warmup (10 requests) ---"
qwn_cmd benchmark run -E "${ENDPOINT}" -n 10 --max-tokens 64 2>&1 | tail -5
echo ""

echo "--- Test A: 50 requests, max_tokens=64 ---"
qwn_cmd benchmark run -E "${ENDPOINT}" -n 50 --max-tokens 64 \
    -o "${RESULTS_DIR}/bench_uniform_50_t64_${TIMESTAMP}.json"
echo ""

echo "--- Test B: 100 requests, max_tokens=128 ---"
qwn_cmd benchmark run -E "${ENDPOINT}" -n 100 --max-tokens 128 \
    -o "${RESULTS_DIR}/bench_uniform_100_t128_${TIMESTAMP}.json"
echo ""

ls -la "${RESULTS_DIR}"/bench_uniform_*_${TIMESTAMP}.json