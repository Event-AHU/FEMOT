#!/bin/bash
# Generate test modality features, run spatial-hyper MambaTrack, and evaluate.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"

GPU="${FEMOT_GPU:-4}"
CKPT="${SPATIAL_HYPER_CKPT:-${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot_spatial_hyper_mvp/epoch3.pth}"

python "${PROJECT_ROOT}/tools/gen_det_results.py" \
    --dataset_name femot_dual \
    --data_root /rydata/wangx_dataset/_dataset/FEMOT \
    --split test \
    --dual_modal \
    --exp_file "${PROJECT_ROOT}/yolox_exps/custom/yolox_femot_dual.py" \
    --model_path "${PROJECT_ROOT}/weights/yolox_femot_dual_best.pth.tar" \
    --img_size 416 416 \
    --device "${GPU}" \
    --generate_meta_data \
    --save_feats \
    --save_modal_feats

cd "${PROJECT_ROOT}/ssm_tracker"
python track.py \
    --det_path "${PROJECT_ROOT}/det_results/femot_dual/test" \
    --motion_model_path "${CKPT}" \
    --config_file cfgs/MambaTrack_femot_spatial_hyper_mvp.yaml \
    --data_root "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
    --device "${GPU}"

bash "${PROJECT_ROOT}/experiments/run_femot_trackeval.sh" test \
    "${PROJECT_ROOT}/ssm_tracker/track_results/femot_dual_MambaTrack_femot_spatial_hyper_mvp/test" \
    MambaTrack_spatial_hyper_mvp
