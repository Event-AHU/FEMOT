#!/bin/bash
# Regenerate FEMOT dual-modal detections WITH RoI appearance features for hyper-assoc tracking.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"
GPU="${FEMOT_GPU:-3}"

python "${PROJECT_ROOT}/tools/gen_det_results.py" \
    --dataset_name femot_dual \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT \
    --split "${1:-test}" \
    --dual_modal \
    --exp_file "${PROJECT_ROOT}/yolox_exps/custom/yolox_femot_dual.py" \
    --model_path "${PROJECT_ROOT}/weights/yolox_femot_dual_best.pth.tar" \
    --img_size 416 416 \
    --device "${GPU}" \
    --generate_meta_data \
    --save_feats
