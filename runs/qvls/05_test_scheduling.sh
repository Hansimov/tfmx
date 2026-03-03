#!/usr/bin/env bash
# Test 2: Scheduling throughput test (uniform 4b-instruct:4bit on all 6 GPUs)
# Uses qvl_benchmark to measure aggregate throughput via machine proxy
set -euo pipefail

ENDPOINT="http://localhost:29800"
RESULTS_DIR="runs/qvls/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Test 2: Scheduling Throughput (6x 4b-instruct:4bit) ==="
echo "Endpoint: ${ENDPOINT}"
echo ""

# Warmup: 10 requests
echo "--- Warmup (10 requests) ---"
qvl_benchmark run -E "${ENDPOINT}" -n 10 --max-tokens 64 --text-only 2>&1 | tail -5
echo ""

# Test A: 50 requests, text-only, short
echo "--- Test A: 50 text requests, max_tokens=64 ---"
qvl_benchmark run -E "${ENDPOINT}" -n 50 --max-tokens 64 --text-only -v \
    -o "${RESULTS_DIR}/bench_uniform_50_t64_${TIMESTAMP}.json" 2>&1
echo ""

# Test B: 100 requests, text-only, medium
echo "--- Test B: 100 text requests, max_tokens=128 ---"
qvl_benchmark run -E "${ENDPOINT}" -n 100 --max-tokens 128 --text-only -v \
    -o "${RESULTS_DIR}/bench_uniform_100_t128_${TIMESTAMP}.json" 2>&1
echo ""

# Test C: 200 requests, text-only, long
echo "--- Test C: 200 text requests, max_tokens=256 ---"
qvl_benchmark run -E "${ENDPOINT}" -n 200 --max-tokens 256 --text-only -v \
    -o "${RESULTS_DIR}/bench_uniform_200_t256_${TIMESTAMP}.json" 2>&1
echo ""

echo "=== Benchmark complete ==="
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}"/bench_uniform_*_${TIMESTAMP}.json 2>/dev/null
