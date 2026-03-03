#!/usr/bin/env bash
# Quick benchmark: 4 available models (2B/4B × instruct/thinking, all 4bit)
set -euo pipefail

PROMPT="Describe the main elements you would expect in a typical street scene photograph. Focus on objects, colors, and spatial relationships."
MAX_TOKENS=256
RESULTS_DIR="runs/qvls/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/bench4_${TIMESTAMP}.jsonl"

echo "=== Benchmark: 4 AWQ Models (compressed-tensors) ==="
echo "Prompt: ${PROMPT:0:80}..."
echo "Max tokens: ${MAX_TOKENS}"
echo ""

PORTS=(29880 29881 29882 29883)
LABELS=("2b-instruct:4bit" "2b-thinking:4bit" "4b-instruct:4bit" "4b-thinking:4bit")
MODELS=(
    "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"
    "cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit"
    "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit"
    "cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit"
)

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    label="${LABELS[$i]}"
    model_name="${MODELS[$i]}"

    echo "--- GPU ${i}: ${label} (port ${port}) ---"
    start_time=$(date +%s%N)

    response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model_name}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
            \"max_tokens\": ${MAX_TOKENS},
            \"temperature\": 0.1
        }" 2>/dev/null)

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    completion_tokens=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage']['completion_tokens'])" 2>/dev/null || echo "0")
    prompt_tokens=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage']['prompt_tokens'])" 2>/dev/null || echo "0")
    text=$(echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "ERROR")

    if [[ "${elapsed_ms}" -gt 0 ]]; then
        tps=$(echo "scale=1; ${completion_tokens} * 1000 / ${elapsed_ms}" | bc 2>/dev/null || echo "?")
    else
        tps="?"
    fi

    echo "  Time: ${elapsed_ms}ms | Tokens: ${completion_tokens} (prompt: ${prompt_tokens}) | Speed: ${tps} tok/s"
    echo "  Response: ${text:0:150}..."
    echo ""

    # Save result
    python3 -c "
import json
record = {
    'gpu': ${i}, 'label': '${label}', 'port': ${port},
    'elapsed_ms': ${elapsed_ms},
    'completion_tokens': int('${completion_tokens}' or 0),
    'prompt_tokens': int('${prompt_tokens}' or 0),
    'tokens_per_sec': float('${tps}') if '${tps}' != '?' else 0,
    'response_preview': '$(echo "${text}" | head -c 200 | sed "s/'/\\\\'/g")'[:200],
}
print(json.dumps(record))
" >> "${RESULT_FILE}" 2>/dev/null || true

done

echo "=== Results saved to ${RESULT_FILE} ==="
echo ""

# Summary table
echo "=== Summary ==="
echo "Model               | Time(ms) | Tokens | tok/s"
echo "--------------------|----------|--------|------"
python3 -c "
import json
with open('${RESULT_FILE}') as f:
    for line in f:
        r = json.loads(line)
        print(f\"{r['label']:20s}| {r['elapsed_ms']:8d} | {r['completion_tokens']:6d} | {r['tokens_per_sec']:5.1f}\")
" 2>/dev/null || echo "(summary failed)"
