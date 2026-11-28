#!/bin/bash
# Fine-tune 3 ssm_mod_mean runs from FEMOT baseline epoch25.pth.
# GPU 0: lr=5e-5 (default)
# GPU 2: lr=1e-4 (2x)
# GPU 6: lr=2.5e-5 (0.5x)
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"
INIT_CKPT="${INIT_CKPT:-saved_ckpts/mambatrack_femot/epoch25.pth}"
LOG_DIR="${PROJECT_ROOT}/logs"

if [[ ! -s "${PROJECT_ROOT}/ssm_tracker/train.py" ]]; then
  echo "Error: ssm_tracker/train.py is empty; restore it before training." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/ssm_tracker/${INIT_CKPT}" ]]; then
  echo "Error: missing init ckpt ${PROJECT_ROOT}/ssm_tracker/${INIT_CKPT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}/ssm_tracker"

JOBS=(
  "0|mambatrack_femot_spatial_ssm_mod_mean_lr5e5|cfgs/MambaTrack_femot_spatial_ssm_mod_mean_lr5e5.yaml"
  "2|mambatrack_femot_spatial_ssm_mod_mean_lr1e4|cfgs/MambaTrack_femot_spatial_ssm_mod_mean_lr1e4.yaml"
  "6|mambatrack_femot_spatial_ssm_mod_mean_lr2p5e5|cfgs/MambaTrack_femot_spatial_ssm_mod_mean_lr2p5e5.yaml"
)

PIDS=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r GPU EXP_NAME CONFIG <<< "${job}"
  LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  echo "GPU=${GPU} EXP=${EXP_NAME} CONFIG=${CONFIG} INIT=${INIT_CKPT}"
  echo "Logging to ${LOG_FILE}"
  nohup env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u train.py \
    --exp_name "${EXP_NAME}" \
    --config_file "${CONFIG}" \
    --init_ckpt "${INIT_CKPT}" \
    --device 0 \
    > "${LOG_FILE}" 2>&1 &
  PID=$!
  PIDS+=("${PID}")
  echo "${PID}" > "${LOG_DIR}/${EXP_NAME}.pid"
  echo "Started ${EXP_NAME} on GPU ${GPU} PID=${PID}"
done

echo "All started: ${PIDS[*]}"
echo "${PIDS[*]}" > "${LOG_DIR}/train_ssm_mod_mean_ft_lr_sweep.pids"
