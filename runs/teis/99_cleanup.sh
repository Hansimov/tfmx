#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

stop_tei_machine_if_present

cd "$TEI_REPO_ROOT"
tei_cmd compose down