#!/bin/bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_mambatrack_femot_hyper_hard05.pid"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

cd "${PROJECT_ROOT}/ssm_tracker"

LOG_FILE="${LOG_DIR}/train_mambatrack_femot_hyper_hard05_$(date +%Y%m%d_%H%M%S).log"
echo "$$" > "${PID_FILE}"

python train.py \
    --exp_name mambatrack_femot_hyper_hard05 \
    --config_file cfgs/MambaTrack_femot_hyper_hard05.yaml \
    --device 4 \
    --save_path ./saved_ckpts \
    >> "${LOG_FILE}" 2>&1

rm -f "${PID_FILE}"
