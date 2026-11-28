#!/bin/bash
# Sequential scratch training (no pretrain) on one GPU with max batch.
# HyLA and MFG each need ~70GB at batch 32, so they run one after another.
#
# Usage:
#   TRAIN_GPU=0 BATCH_SIZE=32 bash experiments/run_train_yolox_femot_dual_ablation_scratch_seq.sh
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/train_yolox_femot_dual_fusion_ablation.sh"
GPU="${TRAIN_GPU:-0}"
BATCH="${BATCH_SIZE:-32}"
FRESH="${FRESH:-1}"

mkdir -p "${LOG_DIR}"

run_one() {
  local fusion="$1"
  echo "=== [${fusion}] scratch batch ${BATCH} on GPU ${GPU} ==="
  FUSION="${fusion}" TRAIN_GPU="${GPU}" BATCH_SIZE="${BATCH}" NUM_GPUS=1 SCRATCH=1 FRESH="${FRESH}" \
    bash "${TRAIN_SCRIPT}"
}

run_one hyla
run_one mfg

echo "Done: hyla + mfg scratch training on GPU ${GPU}"
