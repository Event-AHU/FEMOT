#!/bin/bash
# Extract GT modality features, validate rule edges, and train the MVP.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"

GPU="${FEMOT_GPU:-4}"

python "${PROJECT_ROOT}/tools/gen_femot_gt_modal_features.py" \
    --split train --device "${GPU}"
python "${PROJECT_ROOT}/tools/gen_femot_gt_modal_features.py" \
    --split val --device "${GPU}"

python "${PROJECT_ROOT}/tools/analyze_femot_hyperedges.py" \
    --feature_root "${PROJECT_ROOT}/data/femot_modal_gt" \
    --output "${PROJECT_ROOT}/analysis/femot_hyperedge_feasibility.json"

cd "${PROJECT_ROOT}/ssm_tracker"
python train.py \
    --exp_name mambatrack_femot_spatial_hyper_mvp \
    --config_file cfgs/MambaTrack_femot_spatial_hyper_mvp.yaml \
    --init_ckpt saved_ckpts/mambatrack_femot/epoch25.pth \
    --device "${GPU}"
