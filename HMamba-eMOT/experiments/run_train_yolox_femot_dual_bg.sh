#!/bin/bash
# Launch dual-modal YOLOX training in background (nohup).
#
# Usage:
#   bash experiments/run_train_yolox_femot_dual_bg.sh
#   TRAIN_GPU=0,6,7 NUM_GPUS=3 BATCH_SIZE=24 bash experiments/run_train_yolox_femot_dual_bg.sh
#   bash experiments/run_train_yolox_femot_dual_bg.sh stop
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_yolox_femot_dual.pid"
NOHUP_LOG="${LOG_DIR}/nohup_train_yolox_femot_dual.log"

stop_training() {
    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            sleep 2
        fi
        rm -f "${PID_FILE}"
    fi
    pkill -f "yolox_femot_dual.py" 2>/dev/null || true
    pkill -f "train.py.*yolox_femot_dual" 2>/dev/null || true
    echo "Stopped."
}

if [[ "${1:-}" == "stop" ]]; then
    stop_training
    exit 0
fi

mkdir -p "${LOG_DIR}"
export TRAIN_GPU="${TRAIN_GPU:-0,6,7}"
export NUM_GPUS="${NUM_GPUS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
nohup bash "${PROJECT_ROOT}/experiments/train_yolox_femot_dual.sh" >> "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started YOLOX dual training (GPU ${TRAIN_GPU}, ${NUM_GPUS} GPUs, batch ${BATCH_SIZE})"
echo "  pid file: ${PID_FILE}"
echo "  log:      ${LOG_DIR}/train_yolox_femot_dual_*.log"
