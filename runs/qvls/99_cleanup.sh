#!/usr/bin/env bash
# Cleanup: stop all containers
set -euo pipefail

echo "=== Stopping all QVL containers ==="
qvl_compose down
echo ""
echo "=== Cleanup complete ==="
docker ps --filter "name=qvl" --format "{{.Names}} {{.Status}}" 2>/dev/null || echo "No remaining containers"
