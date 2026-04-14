#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

tei_cmd machine stop

cd "$TEI_REPO_ROOT"
tei_cmd compose down