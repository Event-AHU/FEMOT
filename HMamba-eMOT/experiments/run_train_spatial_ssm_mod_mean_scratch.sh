#!/bin/bash
# Train SSM-mod-mean from random initialization, matching the FEMOT baseline's
# initialization regime instead of warm-starting from baseline epoch25.pth.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"

GPU="${FEMOT_GPU:-1}"
EXP_NAME="${EXP_NAME:-mambatrack_femot_spatial_ssm_mod_mean_scratch}"
CONFIG="${CONFIG:-cfgs/MambaTrack_femot_spatial_ssm_mod_mean_scratch.yaml}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ ! -s "${PROJECT_ROOT}/ssm_tracker/train.py" ]]; then
  echo "Error: ssm_tracker/train.py is empty; restore it before training." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}/ssm_tracker"

echo "Logging to ${LOG_FILE}"
echo "GPU=${GPU} EXP=${EXP_NAME} CONFIG=${CONFIG}"
echo "Initialization: random (same regime as the original FEMOT baseline)"

# Deliberately omit --init_ckpt: loading baseline epoch25.pth would leak a
# fully trained baseline into this ablation.
nohup "${PYTHON}" -u train.py \
  --exp_name "${EXP_NAME}" \
  --config_file "${CONFIG}" \
  --device "${GPU}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/${EXP_NAME}.pid"
echo "Started PID=${PID}"
echo "Tail: tail -f ${LOG_FILE}"
