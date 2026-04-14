#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec bash "$REPO_ROOT/runs/recovery/restart_tei_qsr.sh" "$@"