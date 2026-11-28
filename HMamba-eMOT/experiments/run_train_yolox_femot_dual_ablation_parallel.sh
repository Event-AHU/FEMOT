#!/bin/bash
# Launch HyLA-only and MFG-only ablation training in parallel on separate GPUs.
#
# Default: GPU 4 = hyla, GPU 6 = mfg, batch 8 each.
#   bash experiments/run_train_yolox_femot_dual_ablation_parallel.sh
#
# Custom GPUs (single-GPU per job, needs ~43GB free each):
#   HYLA_GPU=4 MFG_GPU=6 BATCH_SIZE=8 bash experiments/run_train_yolox_femot_dual_ablation_parallel.sh
#
# Multi-GPU DDP per job (e.g. batch 8 on 2 GPUs => 4/GPU, ~22GB/GPU):
#   HYLA_GPU=1,2 HYLA_NUM_GPUS=2 MFG_GPU=6,7 MFG_NUM_GPUS=2 BATCH_SIZE=8 \
#     bash experiments/run_train_yolox_femot_dual_ablation_parallel.sh
#
# Scratch train (no pretrain), separate batch per variant:
#   SCRATCH=1 HYLA_GPU=0 HYLA_BATCH_SIZE=24 MFG_GPU=5 MFG_BATCH_SIZE=12 FRESH=1 \
#     bash experiments/run_train_yolox_femot_dual_ablation_parallel.sh
#
# Stop both jobs:
#   bash experiments/run_train_yolox_femot_dual_ablation_parallel.sh stop
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_yolox_femot_dual_ablation_parallel.pid"
HLYA_PID_FILE="${LOG_DIR}/train_yolox_femot_dual_hyla.pid"
MFG_PID_FILE="${LOG_DIR}/train_yolox_femot_dual_mfg.pid"
HLYA_NOHUP="${LOG_DIR}/nohup_train_yolox_femot_dual_hyla.log"
MFG_NOHUP="${LOG_DIR}/nohup_train_yolox_femot_dual_mfg.log"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/train_yolox_femot_dual_fusion_ablation.sh"

stop_one() {
  local pid_file="$1"
  local exp_name="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
    fi
    rm -f "${pid_file}"
  fi
  pkill -f "${exp_name}.py" 2>/dev/null || true
  pkill -f "train.py.*${exp_name}" 2>/dev/null || true
}

stop_all() {
  stop_one "${HLYA_PID_FILE}" "yolox_femot_dual_hyla"
  stop_one "${MFG_PID_FILE}" "yolox_femot_dual_mfg"
  rm -f "${PID_FILE}"
  echo "Stopped hyla and mfg ablation training."
}

if [[ "${1:-}" == "stop" ]]; then
  stop_all
  exit 0
fi

mkdir -p "${LOG_DIR}"
HYLA_GPU="${HYLA_GPU:-0}"
MFG_GPU="${MFG_GPU:-5}"
HYLA_NUM_GPUS="${HYLA_NUM_GPUS:-1}"
MFG_NUM_GPUS="${MFG_NUM_GPUS:-1}"
HYLA_BATCH_SIZE="${HYLA_BATCH_SIZE:-${BATCH_SIZE:-8}}"
MFG_BATCH_SIZE="${MFG_BATCH_SIZE:-${BATCH_SIZE:-8}}"
SCRATCH="${SCRATCH:-0}"
FRESH="${FRESH:-1}"

echo "Launching parallel ablation training:"
echo "  hyla -> GPU ${HYLA_GPU}, num_gpus=${HYLA_NUM_GPUS}, batch ${HYLA_BATCH_SIZE}"
echo "  mfg  -> GPU ${MFG_GPU}, num_gpus=${MFG_NUM_GPUS}, batch ${MFG_BATCH_SIZE}"
echo "  SCRATCH=${SCRATCH} FRESH=${FRESH}"

FUSION=hyla TRAIN_GPU="${HYLA_GPU}" BATCH_SIZE="${HYLA_BATCH_SIZE}" NUM_GPUS="${HYLA_NUM_GPUS}" SCRATCH="${SCRATCH}" FRESH="${FRESH}" \
  nohup bash "${TRAIN_SCRIPT}" >> "${HLYA_NOHUP}" 2>&1 &
HYLA_PID=$!
echo "${HYLA_PID}" > "${HLYA_PID_FILE}"

FUSION=mfg TRAIN_GPU="${MFG_GPU}" BATCH_SIZE="${MFG_BATCH_SIZE}" NUM_GPUS="${MFG_NUM_GPUS}" SCRATCH="${SCRATCH}" FRESH="${FRESH}" \
  nohup bash "${TRAIN_SCRIPT}" >> "${MFG_NOHUP}" 2>&1 &
MFG_PID=$!
echo "${MFG_PID}" > "${MFG_PID_FILE}"

printf '%s\n%s\n' "${HYLA_PID}" "${MFG_PID}" > "${PID_FILE}"
echo "Started."
echo "  hyla pid: ${HLYA_PID_FILE} ($(cat "${HLYA_PID_FILE}"))"
echo "  mfg  pid: ${MFG_PID_FILE} ($(cat "${MFG_PID_FILE}"))"
echo "  hyla nohup: ${HLYA_NOHUP}"
echo "  mfg  nohup: ${MFG_NOHUP}"
echo "  logs: ${LOG_DIR}/train_yolox_femot_dual_hyla_*.log"
echo "        ${LOG_DIR}/train_yolox_femot_dual_mfg_*.log"
