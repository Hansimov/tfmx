#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/runs/recovery/results}"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/restart_tei_qsr_${TIMESTAMP}.log"

log() {
    printf '[tei-qsr-recovery] %s\n' "$*"
}

detect_visible_gpu_csv() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
}

ALL_GPUS="$(detect_visible_gpu_csv)"
if [[ -z "$ALL_GPUS" ]]; then
    log "no visible GPUs detected"
    exit 1
fi

export TEI_DEPLOY_GPUS="${TEI_GPUS:-${TEI_DEPLOY_GPUS:-$ALL_GPUS}}"
export QSR_DEPLOY_GPUS="${QSR_GPUS:-${QSR_DEPLOY_GPUS:-$ALL_GPUS}}"
export QSR_ENABLE_SLEEP_MODE="${QSR_ENABLE_SLEEP_MODE:-1}"
export QSR_BENCH_SAMPLES="${QSR_BENCH_SAMPLES:-0}"

{
    log "repo root: $REPO_ROOT"
    log "tei gpus: $TEI_DEPLOY_GPUS"
    log "qsr gpus: $QSR_DEPLOY_GPUS"

    log "stopping old TEI/QSR services if present"
    bash runs/teis/99_cleanup.sh || true
    bash runs/qsrs/99_cleanup.sh || true

    log "deploying TEI backends"
    bash runs/teis/01_deploy_default.sh
    log "starting TEI machine"
    bash runs/teis/02_start_machine.sh
    log "running TEI health checks"
    bash runs/teis/03_health_check.sh

    log "deploying QSR backends"
    bash runs/qsrs/01_deploy_default.sh
    log "starting QSR machine"
    bash runs/qsrs/02_start_machine.sh
    log "running QSR health checks"
    bash runs/qsrs/03_health_check.sh

    if [[ "${TEI_RUN_BENCHMARK:-0}" == "1" ]]; then
        log "running TEI benchmark"
        bash runs/teis/04_benchmark.sh
    fi

    if [[ "${QSR_RUN_BENCHMARK:-0}" == "1" ]]; then
        log "running QSR benchmark"
        bash runs/qsrs/04_benchmark.sh
    fi

    log "completed successfully"
} | tee "$LOG_FILE"

log "recovery log: $LOG_FILE"