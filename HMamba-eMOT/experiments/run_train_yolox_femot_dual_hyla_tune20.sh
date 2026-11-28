#!/bin/bash
# HyLA warm-start tuning trial: 20 epochs, then check val AP@0.50 at epoch 5/10/15/20.
#
# Default hyperparams (vs baseline yolox_femot_dual.py):
#   basic_lr_per_img  3.90625e-6  (0.0005/128, half of default 0.001/128)
#   warmup_epochs     5           (default 3)
#   fusion_dropout    0.05        (default 0.0)
#   max_epoch         20          (default 100)
#   batch             16          (~45GB; use 8 if OOM, 24-32 if GPU has ~75GB free)
#
# Init: yolox_femot_dual_init.pth.tar (same as baseline, NOT dual_best)
#
# Usage:
#   bash experiments/run_train_yolox_femot_dual_hyla_tune20.sh
#   TRAIN_GPU=1 BATCH_SIZE=16 bash experiments/run_train_yolox_femot_dual_hyla_tune20.sh
#   # override one knob:
#   BASIC_LR_PER_IMG=7.8125e-6 bash experiments/run_train_yolox_femot_dual_hyla_tune20.sh
#
# After 20 epochs, compare val AP with previous best (~85.9% @ epoch 10).
# If promising, extend: FRESH=0 resume or set max_epoch 100 in EXTRA_OPTS.
#
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/train_yolox_femot_dual_fusion_ablation.sh"

TRAIN_GPU="${TRAIN_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
FRESH="${FRESH:-1}"

# 0.001/128 = 7.8125e-6 ; 0.0005/128 = 3.90625e-6
BASIC_LR_PER_IMG="${BASIC_LR_PER_IMG:-3.90625e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.05}"
MAX_EPOCH="${MAX_EPOCH:-20}"

EXTRA_OPTS="max_epoch ${MAX_EPOCH} basic_lr_per_img ${BASIC_LR_PER_IMG} warmup_epochs ${WARMUP_EPOCHS} fusion_dropout ${FUSION_DROPOUT}"

echo "=== HyLA tune20 trial ==="
echo "  GPU=${TRAIN_GPU} batch=${BATCH_SIZE} FRESH=${FRESH}"
echo "  lr=${BASIC_LR_PER_IMG} warmup=${WARMUP_EPOCHS} dropout=${FUSION_DROPOUT} max_epoch=${MAX_EPOCH}"
echo "  EXTRA_OPTS=${EXTRA_OPTS}"
echo ""

FUSION=hyla \
TRAIN_GPU="${TRAIN_GPU}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_GPUS="${NUM_GPUS}" \
FRESH="${FRESH}" \
EXTRA_OPTS="${EXTRA_OPTS}" \
bash "${TRAIN_SCRIPT}"
