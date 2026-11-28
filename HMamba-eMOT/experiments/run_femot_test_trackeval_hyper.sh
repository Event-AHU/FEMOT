#!/bin/bash
# TrackEval (HOTA / CLEAR / Identity) for hypergraph MambaTrack on FEMOT test.
#
# Usage:
#   bash experiments/run_femot_test_trackeval_hyper.sh
#   bash experiments/run_femot_test_trackeval_hyper.sh val
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-test}"
PRED_DIR="${PROJECT_ROOT}/ssm_tracker/track_results/femot_MambaTrack_femot_hyper/${SPLIT}"
TRACKER_NAME="MambaTrack_hyper"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

if [ ! -d "${PRED_DIR}" ]; then
    echo "Missing predictions: ${PRED_DIR}"
    echo "Run: bash experiments/run_femot_test_infer_hyper.sh"
    exit 1
fi

python "${PROJECT_ROOT}/trackeval/run_femot_eval.py" \
    --split "${SPLIT}" \
    --pred_dir "${PRED_DIR}" \
    --tracker_name "${TRACKER_NAME}"

echo "Done. Results: ${PROJECT_ROOT}/trackeval/trackers/${SPLIT}/${TRACKER_NAME}/"
