#!/bin/bash
# FEMOT test: det -> MambaTrack baseline -> TrackEval for fusion ablations.
# Usage:
#   bash experiments/run_femot_eval_dual_ablation.sh hyla
#   bash experiments/run_femot_eval_dual_ablation.sh mfg
#   bash experiments/run_femot_eval_dual_ablation.sh early_hyla
#   bash experiments/run_femot_eval_dual_ablation.sh both
#   FEMOT_GPU=2 bash experiments/run_femot_eval_dual_ablation.sh early_hyla test
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VARIANT="${1:-both}"
SPLIT="${2:-test}"
GPU="${FEMOT_GPU:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_trackers
export PYTHONPATH="${PROJECT_ROOT}/third_party/ByteTrack:${PROJECT_ROOT}/yolox_exps/custom:${PYTHONPATH:-}"

FEMOT_ROOT="/rydata/wangx_dataset/_dataset/FEMOT"
SSM_CKPT="${PROJECT_ROOT}/ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"

run_one() {
  local fusion="$1"
  local exp_name="yolox_femot_dual_${fusion}"
  local dataset_name="femot_dual_${fusion}"
  local ckpt="${CKPT_OVERRIDE:-${PROJECT_ROOT}/third_party/ByteTrack/YOLOX_outputs/${exp_name}/best_ckpt.pth.tar}"
  local det_path="${PROJECT_ROOT}/det_results/${dataset_name}/${SPLIT}"
  local track_path="${PROJECT_ROOT}/ssm_tracker/track_results/${dataset_name}_MambaTrack_femot/${SPLIT}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing ${ckpt}"
    exit 1
  fi

  echo "=== [${fusion}] gen dets (GPU ${GPU}) ==="
  cd "${PROJECT_ROOT}"
  python "${PROJECT_ROOT}/tools/gen_det_results.py" \
    --dataset_name "${dataset_name}" \
    --data_root "${FEMOT_ROOT}" \
    --split "${SPLIT}" \
    --dual_modal \
    --exp_file "${PROJECT_ROOT}/yolox_exps/custom/${exp_name}.py" \
    --model_path "${ckpt}" \
    --save_dir "${det_path}" \
    --img_size 416 416 \
    --device "${GPU}" \
    --generate_meta_data

  echo "=== [${fusion}] MambaTrack baseline ==="
  cd "${PROJECT_ROOT}/ssm_tracker"
  python track.py \
    --det_path "${det_path}" \
    --motion_model_path saved_ckpts/mambatrack_femot/epoch25.pth \
    --config_file saved_ckpts/mambatrack_femot/MambaTrack_femot.yaml \
    --data_root "${FEMOT_ROOT}/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg" \
    --device "${GPU}"

  echo "=== [${fusion}] TrackEval ==="
  cd "${PROJECT_ROOT}"
  python "${PROJECT_ROOT}/trackeval/run_femot_eval.py" \
    --split "${SPLIT}" \
    --pred_dir "${track_path}" \
    --tracker_name "${TRACKER_NAME:-MambaTrack_${fusion}}"

  echo "Done ${fusion}: ${track_path}"
}

case "${VARIANT}" in
  hyla|mfg|early_hyla|hyla_mfg|hyla_dark3|hyla_shared_mlp) run_one "${VARIANT}" ;;
  both) run_one hyla; run_one mfg ;;
  *)
    echo "Usage: $0 {hyla|mfg|early_hyla|hyla_mfg|hyla_dark3|hyla_shared_mlp|both} [test|val]"
    exit 1
    ;;
esac
