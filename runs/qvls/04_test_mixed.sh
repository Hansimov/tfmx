#!/usr/bin/env bash
# Test 1: Model/quant quality & speed differences
# Sends the same prompt to each GPU instance directly, compares output quality & latency
set -euo pipefail

PROMPT="Describe the main elements in this image in detail. Focus on objects, colors, and spatial relationships."
MAX_TOKENS=256
RESULTS_DIR="runs/qvls/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/test_mixed_${TIMESTAMP}.jsonl"

echo "=== Test 1: Model/Quant Differences ==="
echo "Prompt: ${PROMPT}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Results: ${RESULT_FILE}"
echo ""

# Test each instance directly (bypass machine proxy to target specific models)
PORTS=(29880 29881 29882 29883 29884 29885)
LABELS=("2b-instruct:4bit" "4b-instruct:4bit" "8b-instruct:4bit" "4b-thinking:4bit" "8b-instruct:8bit" "8b-thinking:8bit")

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    label="${LABELS[$i]}"

    echo "--- GPU ${i}: ${label} (port ${port}) ---"
    start_time=$(date +%s%N)

    response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
            \"max_tokens\": ${MAX_TOKENS},
            \"temperature\": 0.1
        }" 2>/dev/null)

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Extract text and token counts
    text=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    completion_tokens=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage']['completion_tokens'])" 2>/dev/null || echo "0")
    prompt_tokens=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage']['prompt_tokens'])" 2>/dev/null || echo "0")

    echo "  Time: ${elapsed_ms}ms | Tokens: ${completion_tokens} | Speed: $(echo "scale=1; ${completion_tokens} * 1000 / ${elapsed_ms}" | bc 2>/dev/null || echo "?") tok/s"
    echo "  Response: ${text:0:200}..."
    echo ""

    # Save to JSONL
    python3 -c "
import json, sys
record = {
    'gpu': ${i}, 'label': '${label}', 'port': ${port},
    'elapsed_ms': ${elapsed_ms},
    'completion_tokens': int('${completion_tokens}' or 0),
    'prompt_tokens': int('${prompt_tokens}' or 0),
    'response_preview': '''${text:0:500}'''[:500],
}
print(json.dumps(record))
" >> "${RESULT_FILE}" 2>/dev/null

done

echo "=== Results saved to ${RESULT_FILE} ==="
