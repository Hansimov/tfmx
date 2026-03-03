#!/usr/bin/env bash
# Health check all instances and machine proxy
set -euo pipefail

echo "=== Container status ==="
qvl_compose ps
echo ""

echo "=== Instance health (direct) ==="
for port in 29880 29881 29882 29883 29884 29885; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
    echo "  localhost:${port} -> HTTP ${status}"
done
echo ""

echo "=== Machine proxy health ==="
curl -s "http://localhost:29800/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Machine proxy not reachable"
echo ""

echo "=== Machine info (models + routing) ==="
curl -s "http://localhost:29800/info" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Machine proxy not reachable"
