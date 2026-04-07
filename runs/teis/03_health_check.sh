#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

cd "$TEI_REPO_ROOT"
tei_cmd client health
tei_cmd client info
tei_cmd client lsh "TEI staged workflow smoke test"