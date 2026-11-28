#!/bin/bash
# Evaluate FEMOT tracking results with TrackEval (HOTA / CLEAR / Identity).
#
# Usage:
#   bash experiments/run_femot_trackeval.sh test
#   bash experiments/run_femot_trackeval.sh val
#   bash experiments/run_femot_trackeval.sh test /path/to/pred_dir MambaTrack
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-test}"
PRED_DIR="${2:-}"
TRACKER_NAME="${3:-MambaTrack}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers

ARGS=(--split "${SPLIT}" --tracker_name "${TRACKER_NAME}")
if [[ -n "${PRED_DIR}" ]]; then
    ARGS+=(--pred_dir "${PRED_DIR}")
fi

python "${PROJECT_ROOT}/trackeval/run_femot_eval.py" "${ARGS[@]}"
