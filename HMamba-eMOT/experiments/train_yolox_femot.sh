#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEMOT_ROOT="/rydata/wangx_dataset/_dataset/FEMOT"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
PRETRAIN="${WEIGHTS_DIR}/yolox_x.pth"
EXP_FILE="${PROJECT_ROOT}/yolox_exps/custom/yolox_femot.py"
DEVICE="${CUDA_VISIBLE_DEVICES:-3}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

# 1) RGB COCO annotations
python "${PROJECT_ROOT}/tools/gen_femot_coco_anno.py" --data_root "${FEMOT_ROOT}"

# 2) COCO-pretrained YOLOX-X weights
mkdir -p "${WEIGHTS_DIR}"
if [ ! -f "${PRETRAIN}" ]; then
    wget -O "${PRETRAIN}" \
        https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1alpha/yolox_x.pth
fi

# 3) Train YOLOX on FEMOT RGB
cd /tmp/ByteTrack
CUDA_VISIBLE_DEVICES="${DEVICE}" python tools/train.py \
    -f "${EXP_FILE}" \
    -expn yolox_femot_rgb \
    -d 1 \
    -b 32 \
    --fp16 \
    -c "${PRETRAIN}" \
    -o

echo "Training done. Best ckpt:"
echo "  /tmp/ByteTrack/YOLOX_outputs/yolox_femot_rgb/best_ckpt.pth.tar"
