#!/bin/bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

cd "${PROJECT_ROOT}/ssm_tracker"

CUDA_VISIBLE_DEVICES=3 python train.py \
    --exp_name mambatrack_femot \
    --config_file cfgs/MambaTrack_femot.yaml \
    --device 0 \
    --save_path ./saved_ckpts
