#!/bin/bash
# Baseline motion + Stage2 rematch (IoU + RGB/Event) on FEMOT test.
# Usage:
#   GPU=2 bash experiments/run_femot_test_rematch.sh
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-2}"
CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"
CFG_NAME="MambaTrack_femot_rematch"
CFG="${PROJECT_ROOT}/ssm_tracker/cfgs/${CFG_NAME}.yaml"
DET_PATH="${PROJECT_ROOT}/det_results/femot_dual/test"
DATA_ROOT="/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_${CFG_NAME}/test"
TRACKER_NAME="MambaTrack_dual_baseline_rematch"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/infer_eval_rematch_$(date +%Y%m%d_%H%M%S).log"

(
  set -eo pipefail
  echo "GPU=${GPU} CKPT=${CKPT}"
  echo "CFG=${CFG}"
  echo "PRED=${PRED_DIR}"
  echo "TRACKER=${TRACKER_NAME}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  cd "${PROJECT_ROOT}/ssm_tracker"
  "${PYTHON}" -u track.py \
    --det_path "${DET_PATH}" \
    --motion_model_path "${CKPT}" \
    --config_file "cfgs/${CFG_NAME}.yaml" \
    --data_root "${DATA_ROOT}" \
    --device 0
  cd "${PROJECT_ROOT}"
  "${PYTHON}" -u trackeval/run_femot_eval.py \
    --split test \
    --pred_dir "${PRED_DIR}" \
    --tracker_name "${TRACKER_NAME}"
  echo "Done rematch"
) > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/infer_eval_rematch.pid"
echo "Started rematch on GPU ${GPU} PID=${PID} log=${LOG_FILE}"
