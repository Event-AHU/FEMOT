#!/bin/bash
# Generate FEMOT detections with dual-modal YOLOX + HyLA-MFG fusion.
# Boxes only (no --save_feats / --save_modal_feats) for baseline MambaTrack.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"

GPU="${FEMOT_GPU:-4}"
SPLIT="${1:-test}"
CKPT="${PROJECT_ROOT}/third_party/ByteTrack/YOLOX_outputs/yolox_femot_dual_hyla_mfg/best_ckpt.pth.tar"
# Separate dataset_name so sum-fusion dets under det_results/femot_dual/ are not overwritten.
DATASET_NAME="femot_dual_hyla_mfg"

if [[ ! -f "${CKPT}" ]]; then
    echo "Missing HyLA-MFG best ckpt: ${CKPT}"
    exit 1
fi

echo "=== Gen ${DATASET_NAME} dets (${SPLIT}), boxes only, GPU ${GPU} ==="
python "${PROJECT_ROOT}/tools/gen_det_results.py" \
    --dataset_name "${DATASET_NAME}" \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT \
    --split "${SPLIT}" \
    --dual_modal \
    --exp_file "${PROJECT_ROOT}/yolox_exps/custom/yolox_femot_dual_hyla_mfg.py" \
    --model_path "${CKPT}" \
    --img_size 416 416 \
    --device "${GPU}" \
    --generate_meta_data

echo "Done: ${PROJECT_ROOT}/det_results/${DATASET_NAME}/${SPLIT}/"
