#!/bin/bash
# Train TrackSSM on FEMOT GT trajectories (no detector needed for training).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train_trackssm_femot.pid"
mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

ANNO="${PROJECT_ROOT}/ssm_tracker/traj_anno_data/femot.json"
if [[ ! -f "${ANNO}" ]]; then
    echo "Missing ${ANNO}; run: python tools/gen_traj_data.py --femot --save_name femot"
    exit 1
fi

cd "${PROJECT_ROOT}/ssm_tracker"

LOG_FILE="${LOG_DIR}/train_trackssm_femot_$(date +%Y%m%d_%H%M%S).log"
echo "$$" > "${PID_FILE}"

python train.py \
    --exp_name trackssm_femot \
    --config_file cfgs/TrackSSM_femot.yaml \
    --device "${TRAIN_GPU:-6}" \
    --save_path ./saved_ckpts \
    >> "${LOG_FILE}" 2>&1

rm -f "${PID_FILE}"
