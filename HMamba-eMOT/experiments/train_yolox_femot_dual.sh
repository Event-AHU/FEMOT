#!/bin/bash
# Train dual-modal (RGB+event) YOLOX on FEMOT (sum fusion, no HyLA/MFG).
#
# Init modes:
#   SCRATCH=1   — random init, no pretrained weights
#   (default)   — FEMOT-RGB dual init / resume latest
#
# Example:
#   SCRATCH=1 FRESH=1 TRAIN_GPU=7 NUM_GPUS=1 BATCH_SIZE=24 \
#     bash experiments/train_yolox_femot_dual.sh
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
INIT_CKPT="${INIT_CKPT:-${WEIGHTS_DIR}/yolox_femot_dual_init.pth.tar}"
RGB_CKPT="${RGB_CKPT:-${WEIGHTS_DIR}/yolox_femot_rgb_best.pth.tar}"
SCRATCH="${SCRATCH:-0}"
EXP_NAME="${EXPERIMENT_NAME:-yolox_femot_dual}"
mkdir -p "${LOG_DIR}" "${WEIGHTS_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
GPU="${TRAIN_GPU:-0,6,7}"
BATCH="${BATCH_SIZE:-24}"
NUM_GPUS="${NUM_GPUS:-3}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${BATCH}" -lt "${NUM_GPUS}" ]]; then
    echo "BATCH_SIZE (${BATCH}) must be >= NUM_GPUS (${NUM_GPUS})"
    exit 1
fi

echo "Training ${EXP_NAME} on GPU(s) ${GPU}, num_gpus=${NUM_GPUS}, total_batch=${BATCH} (per_gpu=$((BATCH / NUM_GPUS))), SCRATCH=${SCRATCH}"

OUTPUT_DIR="${PROJECT_ROOT}/third_party/ByteTrack/YOLOX_outputs/${EXP_NAME}"
if [[ "${FRESH:-0}" == "1" && -d "${OUTPUT_DIR}" ]]; then
    BACKUP_DIR="${OUTPUT_DIR}_bak_$(date +%Y%m%d_%H%M%S)"
    echo "FRESH=1: backing up ${OUTPUT_DIR} -> ${BACKUP_DIR}"
    mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
fi

CKPT_ARGS=()
LATEST_CKPT="${OUTPUT_DIR}/latest_ckpt.pth.tar"
if [[ "${SCRATCH}" == "1" ]]; then
    echo "Scratch training: random init (no pretrained checkpoint)"
elif [[ "${FRESH:-0}" != "1" && -f "${LATEST_CKPT}" ]]; then
    echo "Resuming from ${LATEST_CKPT}"
    CKPT_ARGS=(--resume -c "${LATEST_CKPT}")
else
    if [[ ! -f "${INIT_CKPT}" ]]; then
        if [[ ! -f "${RGB_CKPT}" ]]; then
            echo "Missing RGB pretrained ckpt: ${RGB_CKPT}"
            exit 1
        fi
        echo "Building dual init ckpt from ${RGB_CKPT} ..."
        python "${PROJECT_ROOT}/tools/build_dual_yolox_init_ckpt.py" \
            --src "${RGB_CKPT}" \
            --dst "${INIT_CKPT}"
    fi
    echo "Init from ${INIT_CKPT}"
    CKPT_ARGS=(-c "${INIT_CKPT}")
fi

TRAIN_EXTRA=()
if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_EXTRA=(${EXTRA_OPTS})
fi
if [[ $# -gt 0 ]]; then
  TRAIN_EXTRA+=("$@")
fi

cd "${PROJECT_ROOT}/third_party/ByteTrack"
python tools/train.py \
    -f "${PROJECT_ROOT}/yolox_exps/custom/yolox_femot_dual.py" \
    -b "${BATCH}" \
    -d "${NUM_GPUS}" \
    --fp16 \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    --experiment-name "${EXP_NAME}" \
    "${TRAIN_EXTRA[@]}" \
    2>&1 | tee "${LOG_DIR}/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
