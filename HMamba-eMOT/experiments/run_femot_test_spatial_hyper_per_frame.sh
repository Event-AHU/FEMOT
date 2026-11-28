#!/bin/bash
# Run Spatial-hyper per-frame tracking on FEMOT test + TrackEval (IoU).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"
GPU="${FEMOT_GPU:-1}"
CKPT="${SPATIAL_HYPER_CKPT:-${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot_spatial_hyper_per_frame/best.pth}"
CFG="${SPATIAL_HYPER_CFG:-${PROJECT_ROOT}/ssm_tracker/cfgs/MambaTrack_femot_spatial_hyper_per_frame.yaml}"
DET_PATH="${PROJECT_ROOT}/det_results/femot_dual/test"
PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_MambaTrack_femot_spatial_hyper_per_frame/test"
TRACKER_NAME="${TRACKER_NAME:-MambaTrack_spatial_hyper_per_frame}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/infer_eval_spatial_hyper_per_frame_$(date +%Y%m%d_%H%M%S).log"

{
  echo "GPU=${GPU}"
  echo "CKPT=${CKPT}"
  echo "CFG=${CFG}"
  echo "LOG=${LOG_FILE}"

  cd "${PROJECT_ROOT}/ssm_tracker"
  "${PYTHON}" -u track.py \
      --det_path "${DET_PATH}" \
      --motion_model_path "${CKPT}" \
      --config_file "${CFG}" \
      --data_root "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
      --device "${GPU}"

  cd "${PROJECT_ROOT}"
  "${PYTHON}" -u trackeval/run_femot_eval.py \
      --split test \
      --pred_dir "${PRED_DIR}" \
      --tracker_name "${TRACKER_NAME}"
} > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/infer_eval_spatial_hyper_per_frame.pid"
echo "Started PID=${PID}"
echo "Log: ${LOG_FILE}"
echo "Pred: ${PRED_DIR}"
