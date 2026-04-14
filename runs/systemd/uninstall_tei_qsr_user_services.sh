#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
UNITS=(tfmx-tei-machine.service tfmx-qsr-machine.service)

systemctl --user disable --now "${UNITS[@]}" >/dev/null 2>&1 || true
systemctl --user unset-environment TEI_SERVICE_GPUS QSR_SERVICE_GPUS >/dev/null 2>&1 || true

rm -f \
    "$SYSTEMD_USER_DIR/tfmx-tei-machine.service" \
    "$SYSTEMD_USER_DIR/tfmx-qsr-machine.service"

systemctl --user daemon-reload
systemctl --user reset-failed >/dev/null 2>&1 || true

printf '[systemd-uninstall] removed user units from %s\n' "$SYSTEMD_USER_DIR"
printf '[systemd-uninstall] boot-time auto-start for tfmx TEI/QSR user units has been cleared\n'