#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PATH_VALUE="${PATH_VALUE:-$PATH}"

mkdir -p "$SYSTEMD_USER_DIR"

render_unit() {
    local template_path="$1"
    local output_path="$2"
    sed \
        -e "s|@REPO_ROOT@|$REPO_ROOT|g" \
        -e "s|@PYTHON@|$PYTHON_BIN|g" \
        -e "s|@PATH@|$PATH_VALUE|g" \
        "$template_path" > "$output_path"
}

render_unit \
    "$REPO_ROOT/runs/systemd/tfmx-tei-machine.service.in" \
    "$SYSTEMD_USER_DIR/tfmx-tei-machine.service"

render_unit \
    "$REPO_ROOT/runs/systemd/tfmx-qsr-machine.service.in" \
    "$SYSTEMD_USER_DIR/tfmx-qsr-machine.service"

systemctl --user daemon-reload
systemctl --user enable tfmx-tei-machine.service tfmx-qsr-machine.service

printf '[systemd-install] installed user units in %s\n' "$SYSTEMD_USER_DIR"
printf '[systemd-install] use systemctl --user start tfmx-tei-machine.service tfmx-qsr-machine.service\n'