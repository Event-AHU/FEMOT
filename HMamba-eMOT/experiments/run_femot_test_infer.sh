#!/bin/bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

YOLO_CKPT="${PROJECT_ROOT}/weights/yolox_femot_rgb_best.pth.tar"
SSM_CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"
FEMOT_ROOT="/rydata/wangx_dataset/_dataset/FEMOT"
GPU="${FEMOT_GPU:-7}"

echo "=== Step 1: YOLO detection on FEMOT test (GPU ${GPU}) ==="
cd "${PROJECT_ROOT}"
python tools/gen_det_results.py \
    --dataset_name femot \
    --data_root "${FEMOT_ROOT}" \
    --split test \
    --img_subdir aps \
    --exp_file yolox_exps/custom/yolox_femot.py \
    --model_path "${YOLO_CKPT}" \
    --img_size 416 416 \
    --device "${GPU}" \
    --generate_meta_data

echo "=== Step 2: MambaTrack on FEMOT test (GPU ${GPU}) ==="
cd "${PROJECT_ROOT}/ssm_tracker"
python track.py \
    --det_path ../det_results/femot/test \
    --motion_model_path saved_ckpts/mambatrack_femot/epoch25.pth \
    --config_file cfgs/MambaTrack_femot.yaml \
    --data_root "${FEMOT_ROOT}/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
    --device "${GPU}"

echo "Done."
echo "  Detections: ${PROJECT_ROOT}/det_results/femot/test/"
echo "  Tracking:   ${PROJECT_ROOT}/ssm_tracker/track_results/femot_MambaTrack_femot/test/"
