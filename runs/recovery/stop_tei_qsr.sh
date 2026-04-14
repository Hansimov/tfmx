#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash runs/teis/99_cleanup.sh || true
bash runs/qsrs/99_cleanup.sh || true