#!/bin/bash
# Baseline MambaTrack (no ssm_mod / no hyper) on FEMOT test,
# using HyLA-MFG YOLO detections (boxes only).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

FEMOT_ROOT="/rydata/wangx_dataset/_dataset/FEMOT"
GPU="${FEMOT_GPU:-4}"
DET_PATH="${PROJECT_ROOT}/det_results/femot_dual_hyla_mfg/test"
SSM_CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"

if [[ ! -f "${SSM_CKPT}" ]]; then
    echo "Missing checkpoint: ${SSM_CKPT}"
    exit 1
fi
if [[ ! -d "${DET_PATH}" ]]; then
    echo "Missing dets: ${DET_PATH}"
    echo "Run: bash experiments/gen_femot_det_results_dual_hyla_mfg.sh test"
    exit 1
fi

echo "=== Baseline MambaTrack + HyLA-MFG dets (GPU ${GPU}) ==="
cd "${PROJECT_ROOT}/ssm_tracker"
python track.py \
    --det_path "${DET_PATH}" \
    --motion_model_path saved_ckpts/mambatrack_femot/epoch25.pth \
    --config_file saved_ckpts/mambatrack_femot/MambaTrack_femot.yaml \
    --data_root "${FEMOT_ROOT}/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
    --device "${GPU}"

echo "Done."
