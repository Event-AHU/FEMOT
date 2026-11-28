#!/bin/bash
# Launch FEMOT hyper-gate training in server background (survives SSH disconnect).
#
# Usage:
#   bash experiments/run_train_ssm_femot_hyper_gate_bg.sh
#   TRAIN_GPU=0 bash experiments/run_train_ssm_femot_hyper_gate_bg.sh
#   bash experiments/run_train_ssm_femot_hyper_gate_bg.sh stop
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_mambatrack_femot_hyper_gate.pid"
NOHUP_LOG="${LOG_DIR}/nohup_train_mambatrack_femot_hyper_gate.log"

stop_training() {
    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            echo "Stopping training pid=${pid}"
            kill "${pid}"
            sleep 2
            kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi
    pkill -f "train.py --exp_name mambatrack_femot_hyper_gate" 2>/dev/null || true
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
export TRAIN_GPU="${TRAIN_GPU:-0}"
nohup bash "${PROJECT_ROOT}/experiments/train_ssm_femot_hyper_gate.sh" >> "${NOHUP_LOG}" 2>&1 &
launcher_pid=$!
echo "${launcher_pid}" > "${PID_FILE}"

echo "Training started in background (hyper-gate, GPU ${TRAIN_GPU})."
echo "  launcher pid: ${launcher_pid}"
echo "  pid file:     ${PID_FILE}"
echo "  nohup log:    ${NOHUP_LOG}"
echo "  train log:    ${LOG_DIR}/train_mambatrack_femot_hyper_gate_*.log (latest)"
echo ""
echo "Monitor:  tail -f ${NOHUP_LOG}"
echo "Stop:     bash experiments/run_train_ssm_femot_hyper_gate_bg.sh stop"
