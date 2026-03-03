#!/usr/bin/env bash
# Concurrent throughput test through machine proxy
set -euo pipefail

PROXY="http://localhost:29800"
N_REQUESTS=20
MAX_TOKENS=128
CONCURRENCY=4

echo "=== Concurrent Throughput Test via Machine Proxy ==="
echo "Requests: ${N_REQUESTS}, Concurrency: ${CONCURRENCY}, Max tokens: ${MAX_TOKENS}"
echo ""

RESULTS_DIR="runs/qvls/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/throughput_${TIMESTAMP}.jsonl"

overall_start=$(date +%s%N)

# Send concurrent requests
for i in $(seq 1 $N_REQUESTS); do
    (
        start_ns=$(date +%s%N)
        resp=$(curl -s "${PROXY}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Write a haiku about request number ${i}.\"}],
                \"max_tokens\": ${MAX_TOKENS},
                \"temperature\": 0.7
            }" 2>/dev/null)
        end_ns=$(date +%s%N)
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

        tokens=$(echo "${resp}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
        model=$(echo "${resp}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('model','?'))" 2>/dev/null || echo "?")

        echo "  req=${i} model=${model##*/} time=${elapsed_ms}ms tokens=${tokens}"
        echo "{\"req\":${i},\"model\":\"${model}\",\"elapsed_ms\":${elapsed_ms},\"tokens\":${tokens}}" >> "${RESULT_FILE}"
    ) &

    # Limit concurrency
    if (( i % CONCURRENCY == 0 )); then
        wait
    fi
done
wait

overall_end=$(date +%s%N)
overall_ms=$(( (overall_end - overall_start) / 1000000 ))

echo ""
echo "=== Total wall time: ${overall_ms}ms ==="
echo ""

# Distribution summary
python3 -c "
import json, collections
counts = collections.Counter()
total_tokens = 0
total_time = 0
n = 0
with open('${RESULT_FILE}') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        short = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
        counts[short] += 1
        total_tokens += int(r.get('tokens', 0))
        total_time += int(r.get('elapsed_ms', 0))
        n += 1

print('=== Distribution ===')
for model, count in sorted(counts.items()):
    print(f'  {model}: {count} requests')
print(f'\nTotal tokens: {total_tokens}')
print(f'Avg latency:  {total_time/n:.0f}ms' if n else '')
print(f'Wall time:    ${overall_ms}ms')
print(f'Throughput:   {total_tokens * 1000 / ${overall_ms}:.1f} tok/s (aggregate)')
"
