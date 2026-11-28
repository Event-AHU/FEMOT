#!/bin/bash
# Launch HyLA-MFG dual-modal YOLOX training in background (nohup).
#
# Single GPU (default, GPU 4 usually free):
#   bash experiments/run_train_yolox_femot_dual_hyla_mfg_bg.sh
#
# Multi-GPU example (3 cards, total batch 18 => 6 per GPU):
#   TRAIN_GPU=4,5,6 NUM_GPUS=3 BATCH_SIZE=18 bash experiments/run_train_yolox_femot_dual_hyla_mfg_bg.sh
#
# Force fresh run from dual baseline (ignore hyla_mfg latest ckpt):
#   FRESH=1 bash experiments/run_train_yolox_femot_dual_hyla_mfg_bg.sh
#
# Stop:
#   bash experiments/run_train_yolox_femot_dual_hyla_mfg_bg.sh stop
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_yolox_femot_dual_hyla_mfg.pid"
NOHUP_LOG="${LOG_DIR}/nohup_train_yolox_femot_dual_hyla_mfg.log"
EXP_NAME="yolox_femot_dual_hyla_mfg"

stop_training() {
    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            sleep 2
        fi
        rm -f "${PID_FILE}"
    fi
    pkill -f "${EXP_NAME}.py" 2>/dev/null || true
    pkill -f "train.py.*${EXP_NAME}" 2>/dev/null || true
    echo "Stopped."
}

if [[ "${1:-}" == "stop" ]]; then
    stop_training
    exit 0
fi

mkdir -p "${LOG_DIR}"
export TRAIN_GPU="${TRAIN_GPU:-4}"
export NUM_GPUS="${NUM_GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-18}"
export FRESH="${FRESH:-1}"
nohup bash "${PROJECT_ROOT}/experiments/train_yolox_femot_dual_hyla_mfg.sh" >> "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started ${EXP_NAME} (GPU ${TRAIN_GPU}, ${NUM_GPUS} GPUs, batch ${BATCH_SIZE}, FRESH=${FRESH})"
echo "  pid file: ${PID_FILE}"
echo "  nohup:    ${NOHUP_LOG}"
echo "  log:      ${LOG_DIR}/train_${EXP_NAME}_*.log"
