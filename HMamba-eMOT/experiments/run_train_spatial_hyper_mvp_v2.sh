#!/bin/bash
# Retrain Spatial-hyper MVP into a new checkpoint dir (does not overwrite old mvp weights).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"

GPU="${FEMOT_GPU:-3}"
EXP_NAME="${EXP_NAME:-mambatrack_femot_spatial_hyper_mvp_v2}"
CONFIG="${CONFIG:-cfgs/MambaTrack_femot_spatial_hyper_mvp_v2.yaml}"
INIT_CKPT="${INIT_CKPT:-saved_ckpts/mambatrack_femot/epoch25.pth}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_ROOT}/ssm_tracker"
echo "Logging to ${LOG_FILE}"
echo "GPU=${GPU} EXP=${EXP_NAME} CONFIG=${CONFIG} INIT=${INIT_CKPT}"
echo "Save dir: ${PROJECT_ROOT}/ssm_tracker/saved_ckpts/${EXP_NAME}"

nohup "${PYTHON}" -u train.py \
  --exp_name "${EXP_NAME}" \
  --config_file "${CONFIG}" \
  --init_ckpt "${INIT_CKPT}" \
  --device "${GPU}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/${EXP_NAME}.pid"
echo "Started PID=${PID}"
echo "Tail: tail -f ${LOG_FILE}"
