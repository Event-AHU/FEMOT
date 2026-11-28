#!/bin/bash
# Launch TrackSSM FEMOT training in server background.
#
# Usage:
#   bash experiments/run_train_ssm_femot_trackssm_bg.sh
#   TRAIN_GPU=6 bash experiments/run_train_ssm_femot_trackssm_bg.sh
#   bash experiments/run_train_ssm_femot_trackssm_bg.sh stop
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_trackssm_femot.pid"
NOHUP_LOG="${LOG_DIR}/nohup_train_trackssm_femot.log"

stop_training() {
    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            sleep 2
            kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi
    pkill -f "train.py --exp_name trackssm_femot" 2>/dev/null || true
    echo "Stopped."
}

if [[ "${1:-}" == "stop" ]]; then
    stop_training
    exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
        echo "Training already running (pid=${old_pid}). Use: bash $0 stop"
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

mkdir -p "${LOG_DIR}"
export TRAIN_GPU="${TRAIN_GPU:-6}"
nohup bash "${PROJECT_ROOT}/experiments/train_ssm_femot_trackssm.sh" >> "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started TrackSSM FEMOT training (GPU ${TRAIN_GPU})"
echo "  pid file: ${PID_FILE}"
echo "  nohup:    ${NOHUP_LOG}"
echo "  train:    ${LOG_DIR}/train_trackssm_femot_*.log"
