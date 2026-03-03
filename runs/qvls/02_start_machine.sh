#!/usr/bin/env bash
# Start the machine proxy with auto-discovery
set -euo pipefail

echo "=== Starting machine proxy ==="
echo "Port: 29800"
echo "Auto-discovering vLLM containers..."
echo ""

qvl_machine run --perf-track
