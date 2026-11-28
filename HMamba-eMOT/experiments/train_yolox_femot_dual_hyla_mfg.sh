#!/bin/bash
# Train dual-modal YOLOX (HyLA-MFG fusion) on FEMOT.
# Warm-start backbone/neck/head from dual sum-fusion baseline; fusion blocks train from zero-init.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
INIT_CKPT="${WEIGHTS_DIR}/yolox_femot_dual_best.pth.tar"
EXP_FILE="${PROJECT_ROOT}/yolox_exps/custom/yolox_femot_dual_hyla_mfg.py"
EXP_NAME="yolox_femot_dual_hyla_mfg"
mkdir -p "${LOG_DIR}" "${WEIGHTS_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
GPU="${TRAIN_GPU:-4}"
BATCH="${BATCH_SIZE:-18}"
NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${BATCH}" -lt "${NUM_GPUS}" ]]; then
    echo "BATCH_SIZE (${BATCH}) must be >= NUM_GPUS (${NUM_GPUS})"
    exit 1
fi

echo "Training ${EXP_NAME} on GPU(s) ${GPU}, num_gpus=${NUM_GPUS}, total_batch=${BATCH} (per_gpu=$((BATCH / NUM_GPUS)))"

if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "Missing dual baseline ckpt: ${INIT_CKPT}"
    echo "Expected trained sum-fusion weights (yolox_femot_dual_best.pth.tar)."
    exit 1
fi

LATEST_CKPT="${PROJECT_ROOT}/third_party/ByteTrack/YOLOX_outputs/${EXP_NAME}/latest_ckpt.pth.tar"
if [[ "${FRESH:-0}" != "1" && -f "${LATEST_CKPT}" ]]; then
    echo "Resuming ${EXP_NAME} from ${LATEST_CKPT}"
    CKPT_ARGS=(--resume -c "${LATEST_CKPT}")
else
    echo "Warm-start from ${INIT_CKPT} (fusion blocks keep zero/random init)"
    CKPT_ARGS=(-c "${INIT_CKPT}")
fi

cd "${PROJECT_ROOT}/third_party/ByteTrack"
python tools/train.py \
    -f "${EXP_FILE}" \
    -b "${BATCH}" \
    -d "${NUM_GPUS}" \
    --fp16 \
    "${CKPT_ARGS[@]}" \
    --experiment-name "${EXP_NAME}" \
    2>&1 | tee "${LOG_DIR}/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
