#!/bin/bash
# Train dual-modal YOLOX fusion ablation (hyla / mfg / hyla_mfg) on FEMOT.
#
# Init modes (pick one):
#   SCRATCH=1          — random init, no pretrained weights
#   INIT_CKPT=path     — custom checkpoint
#   (default)          — yolox_femot_dual_init.pth.tar (FEMOT-RGB, same as baseline)
#
# Usage:
#   SCRATCH=1 FUSION=hyla TRAIN_GPU=0 BATCH_SIZE=24 FRESH=1 bash experiments/train_yolox_femot_dual_fusion_ablation.sh
#
# Override exp fields via EXTRA_OPTS or trailing positional opts (forwarded to train.py):
#   FUSION=hyla TRAIN_GPU=1 BATCH_SIZE=16 FRESH=1 \
#     EXTRA_OPTS="max_epoch 20 basic_lr_per_img 3.90625e-6 warmup_epochs 5 fusion_dropout 0.05" \
#     bash experiments/train_yolox_femot_dual_fusion_ablation.sh
#   FUSION=hyla TRAIN_GPU=1 BATCH_SIZE=16 FRESH=1 \
#     bash experiments/train_yolox_femot_dual_fusion_ablation.sh \
#     max_epoch 20 fusion_dropout 0.05
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
INIT_CKPT="${INIT_CKPT:-${WEIGHTS_DIR}/yolox_femot_dual_init.pth.tar}"
RGB_CKPT="${WEIGHTS_DIR}/yolox_femot_rgb_best.pth.tar"
FUSION="${FUSION:-hyla}"
SCRATCH="${SCRATCH:-0}"

case "${FUSION}" in
  hyla|mfg|hyla_mfg|early_hyla|hyla_dark3|hyla_shared_mlp) ;;
  *)
    echo "FUSION must be one of: hyla, mfg, hyla_mfg, early_hyla, hyla_dark3, hyla_shared_mlp (got: ${FUSION})"
    exit 1
    ;;
esac

# Exp file always follows FUSION; output/log name can be overridden for ablations.
EXP_FILE_STEM="yolox_femot_dual_${FUSION}"
EXP_FILE="${PROJECT_ROOT}/yolox_exps/custom/${EXP_FILE_STEM}.py"
EXP_NAME="${EXPERIMENT_NAME:-${EXP_FILE_STEM}${EXP_SUFFIX:-}}"
mkdir -p "${LOG_DIR}" "${WEIGHTS_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
GPU="${TRAIN_GPU:-4}"
BATCH="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${BATCH}" -lt "${NUM_GPUS}" ]]; then
  echo "BATCH_SIZE (${BATCH}) must be >= NUM_GPUS (${NUM_GPUS})"
  exit 1
fi

echo "Training ${EXP_NAME} on GPU(s) ${GPU}, num_gpus=${NUM_GPUS}, total_batch=${BATCH} (per_gpu=$((BATCH / NUM_GPUS))), SCRATCH=${SCRATCH}"

if [[ ! -f "${EXP_FILE}" ]]; then
  echo "Missing exp file: ${EXP_FILE}"
  exit 1
fi

if [[ "${SCRATCH}" == "1" ]]; then
  echo "Scratch training: random init (no pretrained checkpoint)"
elif [[ ! -f "${INIT_CKPT}" ]]; then
  if [[ ! -f "${RGB_CKPT}" ]]; then
    echo "Missing RGB pretrained ckpt: ${RGB_CKPT}"
    exit 1
  fi
  echo "Building dual init ckpt from ${RGB_CKPT} (same as baseline) ..."
  python "${PROJECT_ROOT}/tools/build_dual_yolox_init_ckpt.py" \
    --src "${RGB_CKPT}" \
    --dst "${INIT_CKPT}"
fi

OUTPUT_DIR="${PROJECT_ROOT}/third_party/ByteTrack/YOLOX_outputs/${EXP_NAME}"
LATEST_CKPT="${OUTPUT_DIR}/latest_ckpt.pth.tar"
if [[ "${FRESH:-0}" == "1" && -d "${OUTPUT_DIR}" ]]; then
  BACKUP_DIR="${OUTPUT_DIR}_bak_$(date +%Y%m%d_%H%M%S)"
  echo "FRESH=1: backing up ${OUTPUT_DIR} -> ${BACKUP_DIR}"
  mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
fi

CKPT_ARGS=()
if [[ "${FRESH:-0}" != "1" && -f "${LATEST_CKPT}" ]]; then
  echo "Resuming ${EXP_NAME} from ${LATEST_CKPT}"
  CKPT_ARGS=(--resume -c "${LATEST_CKPT}")
elif [[ "${SCRATCH}" != "1" ]]; then
  echo "Init from ${INIT_CKPT} (backbone/neck/head; fusion blocks random init)"
  CKPT_ARGS=(-c "${INIT_CKPT}")
fi

# EXTRA_OPTS: e.g. 'max_epoch 20 fusion_dropout 0.05' (no --opts prefix)
# "$@": same as trailing args after the .sh invocation
TRAIN_EXTRA=()
if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_EXTRA=(${EXTRA_OPTS})
fi
if [[ $# -gt 0 ]]; then
  TRAIN_EXTRA+=("$@")
fi
if [[ ${#TRAIN_EXTRA[@]} -gt 0 ]]; then
  echo "train.py extra args: ${TRAIN_EXTRA[*]}"
fi

cd "${PROJECT_ROOT}/third_party/ByteTrack"
python tools/train.py \
  -f "${EXP_FILE}" \
  -b "${BATCH}" \
  -d "${NUM_GPUS}" \
  --fp16 \
  ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
  --experiment-name "${EXP_NAME}" \
  "${TRAIN_EXTRA[@]}" \
  2>&1 | tee "${LOG_DIR}/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
