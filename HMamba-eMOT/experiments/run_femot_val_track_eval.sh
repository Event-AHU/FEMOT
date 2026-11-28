#!/bin/bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

cd "${PROJECT_ROOT}/ssm_tracker"

echo "=== Step 1: tracking on FEMOT val ==="
CUDA_VISIBLE_DEVICES=3 python track.py \
    --det_path ../det_results/femot/val \
    --motion_model_path saved_ckpts/mambatrack_femot/epoch25.pth \
    --config_file cfgs/MambaTrack_femot.yaml \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg \
    --device 0

PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_MambaTrack_femot/val"

echo "=== Step 2: evaluate metrics ==="
cd "${PROJECT_ROOT}"
python tools/eval_mot.py \
    --split val \
    --pred_dir "${PRED_DIR}"

echo "Done. Results: ${PRED_DIR}/metrics_summary.txt"
