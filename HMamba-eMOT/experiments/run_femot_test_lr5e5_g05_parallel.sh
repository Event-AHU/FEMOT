#!/bin/bash
# Parallel infer + TrackEval for lr5e5_g05 epoch1-3.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/lyt/miniconda3/envs/mamba_trackers/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"

CKPT_DIR="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot_spatial_hyper_mvp_lr5e5_g05"
BASE_CFG="${PROJECT_ROOT}/ssm_tracker/cfgs/MambaTrack_femot_spatial_hyper_mvp_lr5e5_g05.yaml"
DET_PATH="${PROJECT_ROOT}/det_results/femot_dual/test"
DATA_ROOT="/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/ssm_tracker/cfgs"

EPOCHS=(${EPOCHS:-1 2 3})
GPUS=(${GPUS:-2 4 5})

if [[ ${#EPOCHS[@]} -ne ${#GPUS[@]} ]]; then
  echo "EPOCHS and GPUS length mismatch" >&2
  exit 1
fi

PIDS=()
for i in "${!EPOCHS[@]}"; do
  EP="${EPOCHS[$i]}"
  GPU="${GPUS[$i]}"
  CKPT="${CKPT_DIR}/epoch${EP}.pth"
  if [[ ! -f "${CKPT}" ]]; then
    echo "Missing ckpt: ${CKPT}" >&2
    exit 1
  fi

  CFG_NAME="MambaTrack_femot_spatial_hyper_mvp_lr5e5_g05_ep${EP}"
  CFG="${PROJECT_ROOT}/ssm_tracker/cfgs/${CFG_NAME}.yaml"
  cp "${BASE_CFG}" "${CFG}"

  PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_${CFG_NAME}/test"
  TRACKER_NAME="MambaTrack_spatial_hyper_lr5e5_g05_ep${EP}"
  LOG_FILE="${LOG_DIR}/infer_eval_lr5e5_g05_ep${EP}_$(date +%Y%m%d_%H%M%S).log"

  (
    set -eo pipefail
    echo "GPU=${GPU} EP=${EP} CKPT=${CKPT}"
    echo "CFG=${CFG}"
    echo "PRED=${PRED_DIR}"
    echo "TRACKER=${TRACKER_NAME}"
    cd "${PROJECT_ROOT}/ssm_tracker"
    "${PYTHON}" -u track.py \
      --det_path "${DET_PATH}" \
      --motion_model_path "${CKPT}" \
      --config_file "cfgs/${CFG_NAME}.yaml" \
      --data_root "${DATA_ROOT}" \
      --device "${GPU}"
    cd "${PROJECT_ROOT}"
    "${PYTHON}" -u trackeval/run_femot_eval.py \
      --split test \
      --pred_dir "${PRED_DIR}" \
      --tracker_name "${TRACKER_NAME}"
    echo "Done EP=${EP}"
  ) > "${LOG_FILE}" 2>&1 &

  PID=$!
  PIDS+=("${PID}")
  echo "${PID}" > "${LOG_DIR}/infer_eval_lr5e5_g05_ep${EP}.pid"
  echo "Started ep${EP} on GPU ${GPU} PID=${PID} log=${LOG_FILE}"
done

echo "All started: ${PIDS[*]}"
