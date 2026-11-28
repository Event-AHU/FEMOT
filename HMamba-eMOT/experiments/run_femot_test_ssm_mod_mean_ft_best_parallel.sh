#!/bin/bash
# Parallel infer + TrackEval for the current-best ckpt of each LR sweep.
# GPU 0: lr=5e-5 epoch6
# GPU 2: lr=1e-4 epoch4
# GPU 5: lr=2.5e-5 epoch9
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DET_PATH="${PROJECT_ROOT}/det_results/femot_dual/test"
DATA_ROOT="/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

JOBS=(
  "0|lr5e5|mambatrack_femot_spatial_ssm_mod_mean_lr5e5|MambaTrack_femot_spatial_ssm_mod_mean_lr5e5"
  "2|lr1e4|mambatrack_femot_spatial_ssm_mod_mean_lr1e4|MambaTrack_femot_spatial_ssm_mod_mean_lr1e4"
  "5|lr2p5e5|mambatrack_femot_spatial_ssm_mod_mean_lr2p5e5|MambaTrack_femot_spatial_ssm_mod_mean_lr2p5e5"
)

PIDS=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r GPU TAG CKPT_NAME CFG_NAME <<< "${job}"
  CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/${CKPT_NAME}/best.pth"
  CFG="${PROJECT_ROOT}/ssm_tracker/cfgs/${CFG_NAME}.yaml"
  if [[ ! -f "${CKPT}" ]]; then
    echo "Missing ckpt: ${CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${CFG}" ]]; then
    echo "Missing cfg: ${CFG}" >&2
    exit 1
  fi

  PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_${CFG_NAME}/test"
  TRACKER_NAME="MambaTrack_ssm_mod_mean_${TAG}_best"
  LOG_FILE="${LOG_DIR}/infer_eval_ssm_mod_mean_${TAG}_best_$(date +%Y%m%d_%H%M%S).log"

  (
    set -eo pipefail
    echo "GPU=${GPU} TAG=${TAG} CKPT=${CKPT}"
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
    echo "Done ${TAG}"
  ) > "${LOG_FILE}" 2>&1 &

  PID=$!
  PIDS+=("${PID}")
  echo "${PID}" > "${LOG_DIR}/infer_eval_ssm_mod_mean_${TAG}_best.pid"
  echo "Started ${TAG} on GPU ${GPU} PID=${PID} log=${LOG_FILE}"
done

echo "All started: ${PIDS[*]}"
echo "${PIDS[*]}" > "${LOG_DIR}/infer_eval_ssm_mod_mean_ft_best_parallel.pids"
