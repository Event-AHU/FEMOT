#!/bin/bash
# MambaTrack (hyper-gate) on FEMOT test using dual-modal YOLOX detections.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

FEMOT_ROOT="/rydata/wangx_dataset/_dataset/FEMOT"
GPU="${FEMOT_GPU:-4}"
DET_PATH="${PROJECT_ROOT}/det_results/femot_dual/test"
SSM_CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot_hyper_gate/epoch25.pth"

if [[ ! -f "${SSM_CKPT}" ]]; then
    echo "Missing checkpoint: ${SSM_CKPT}"
    exit 1
fi

if [[ ! -d "${DET_PATH}" ]] || [[ -z "$(ls -A "${DET_PATH}"/*.txt 2>/dev/null)" ]]; then
    echo "Missing test detections under ${DET_PATH}"
    exit 1
fi

echo "=== MambaTrack hyper-gate on FEMOT test (dual YOLOX dets, GPU ${GPU}) ==="
cd "${PROJECT_ROOT}/ssm_tracker"
python track.py \
    --det_path "${DET_PATH}" \
    --motion_model_path saved_ckpts/mambatrack_femot_hyper_gate/epoch25.pth \
    --config_file cfgs/MambaTrack_femot_hyper_gate.yaml \
    --data_root "${FEMOT_ROOT}/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
    --device "${GPU}"

echo "Done."
echo "  Tracking: ${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_MambaTrack_femot_hyper_gate/test/"
