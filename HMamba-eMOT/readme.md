### 

**HMamba-eMOT: Hypergraph Modulated State Updating in Mamba for RGB-Event Multi-Object Tracking**, 
Xiao Wang, YiTao Li, Shiao Wang, Lan Chen*, Bo Jiang, Yonghong Tian, Jin Tang
arXiv (To Be Updated) 

### Abstract

Multi-Object Tracking (MOT) is essential for autonomous driving and intelligent surveillance. State-of-the-art RGB trackers struggle under challenging conditions such as blur and overexposure, while event cameras provide complementary high dynamic range, asynchronous visual cues. However, effective RGB-event fusion and relational target association remain open problems. This paper proposes HMamba-eMOT, a two-stage multimodal MOT framework. We first introduce bidirectional Hypernetwork Linear Attention (HyLA) to dynamically generate modality-aware weights for adaptive cross-modal fusion before the YOLOX detection head. We further build a frame-level hypergraph to model spatial and appearance correlations across targets. Inspired by M³amba, we leverage the hypergraph prior to residually modulate the Δ and C matrices inside bidirectional Mamba, rather than simply concatenating hypergraph context with motion inputs. A two-stage association module outputs the final tracking results. Evaluations on DSEC-MOT and FEMOT demonstrate the effectiveness of our approach.

### Models

Download the weights into `weights/`.

* FEMOT HyLA detector: [femot_hyla_detector_ep27.pth.tar](https://pan.baidu.com/s/1mZ0ihCQx-0sIvnesLEqZ5w?pwd=u29c)
* FEMOT tracker: [femot_hmamba_tracker.pth](https://pan.baidu.com/s/1HbSQHEJTIMs0c0K6d_fn9g?pwd=6i3v)
* DSEC HyLA detector: [dsec_hyla_detector_ep24.pth.tar](https://pan.baidu.com/s/1vJ_jDwABFvlkynG_CB-J0w?pwd=rvta)
* DSEC tracker: [dsec_hmamba_tracker.pth](https://pan.baidu.com/s/1Qps2cZqXo9xdeHoSjIWzuw?pwd=e6mw)

## Environment

The training environment used for the paper is Python 3.10, PyTorch 2.1.0 with CUDA 12.1, and `mamba_ssm`.

```bash
conda create -n hmamba python=3.10 -y
conda activate hmamba
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm
pip install -r third_party/ByteTrack/requirements.txt
pip install -e third_party/ByteTrack
```

`yolox` is the ByteTrack fork under `third_party/ByteTrack`. If the editable install does not put it on the path, export it before every detector command:

```bash
export PYTHONPATH="$PWD/third_party/ByteTrack:$PWD/yolox_exps/custom:$PYTHONPATH"
```

Evaluation uses the official [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository as a sibling of this repo:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git ../TrackEval
```

## Data

Edit the dataset roots before training.

- FEMOT: `configs/dataset_paths.py`, field `FEMOT_ROOT`. Each sequence has `{seq}_aps` RGB frames and `{seq}_dvs` event frames. Annotations are `train_rgb.json`, `val_rgb.json`, and `test_rgb.json`.
- DSEC: `yolox_exps/custom/yolox_dsec_dual.py`, field `DSEC_ROOT`. The detector has 7 classes.

The scripts `trackeval/run_femot_eval.py` and `trackeval/run_dsec_eval.py` also contain default roots. Pass `--pred_dir` and point those defaults at your copies of FEMOT and DSEC, or edit `FEMOT_ROOT` and `DSEC_ROOT` in those two files.

## Detector training

Build a 6-channel initialization from a single-stream YOLOX checkpoint. The stem of each modality stays 3-channel; the script copies the RGB backbone into both streams:

```bash
python tools/build_dual_yolox_init_ckpt.py \
  --src /path/to/yolox_x.pth \
  --dst weights/yolox_femot_dual_init.pth.tar
```

FEMOT HyLA detector. The experiment file defaults to a longer schedule. The paper run uses 30 epochs, 9 no-augmentation epochs, and FP16. `basic_lr_per_img` stays `0.001 / 128`.

```bash
cd third_party/ByteTrack
python tools/train.py \
  -f ../../yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py \
  -d 1 -b 24 --fp16 \
  -c ../../weights/yolox_femot_dual_init.pth.tar \
  max_epoch 30 \
  no_aug_epochs 9
```

The plain sum-fusion detector is `yolox_exps/custom/yolox_femot_dual.py`. The DSEC HyLA detector is `yolox_exps/custom/yolox_dsec_dual_hyla_shared_mlp.py`, launched the same way from `tools/train.py`. Checkpoints land in `third_party/ByteTrack/YOLOX_outputs/<exp_name>/`.

## Appearance features

The tracker reads per-box RGB and event features. For the reported FEMOT tracker these are HyLA features of ground-truth boxes, extracted with `weights/femot_hyla_detector_ep27.pth.tar`:

```bash
python tools/gen_femot_gt_modal_features.py \
  --data_root /path/to/FEMOT \
  --split train \
  --exp_file yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py \
  --model_path weights/femot_hyla_detector_ep27.pth.tar \
  --output_dir data/femot_modal_gt_hyla \
  --enhanced
```

Detection boxes plus RoI features, used at inference time:

```bash
python tools/gen_det_results.py \
  --dataset_name femot --split test \
  --data_root /path/to/FEMOT \
  --exp_file yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py \
  --model_path weights/femot_hyla_detector_ep27.pth.tar \
  --img_size 416 416 \
  --dual_modal --save_modal_feats \
  --save_dir det_results/femot_hyla
```

Set `modal_feature_root` in the tracker config to the directory that contains these `.npz` files. DSEC detection export uses `weights/dsec_hyla_detector_ep24.pth.tar` with `yolox_exps/custom/yolox_dsec_dual_hyla_shared_mlp.py`.

## Tracker training

The reported FEMOT configuration is `ssm_tracker/cfgs/ablate_retrain_hyla/retrain_hyla_drop_evt_motion_zero_b.yaml`.

That file keeps spatial and RGB hyperedges, drops the event and motion relations, and freezes the modulation of `B`. Optimization is Adam at `5e-5`. The yaml `epochs` field is the length of the run you want; the checkpoint used for the paper table is an early epoch of this recipe, not a different architecture.

```bash
cd ssm_tracker
python train.py \
  --exp_name hmamba_femot \
  --config_file cfgs/ablate_retrain_hyla/retrain_hyla_drop_evt_motion_zero_b.yaml \
  --save_path saved_ckpts/hmamba_femot \
  --device 0
```

A saved file stores `model`, `optimizer`, and `epoch`. Continue a run with `--resume --resume_ckpt saved_ckpts/hmamba_femot/epoch2.pth`; training starts at `epoch + 1`.

DSEC uses the same hypergraph recipe. Configs live in `ssm_tracker/cfgs/dsec/`. Point `traj_root` and `modal_feature_root` in the yaml at your DSEC trajectory file and feature directory before launching the same `train.py` command.

## Inference and evaluation

`ssm_tracker/cfgs/tune5376/tune5376_lost_20.yaml` is the inference schedule that matches the reported FEMOT number (`max_time_lost: 20`). The training yaml uses `25`.

```bash
cd ssm_tracker
python track.py \
  --det_path ../det_results/femot_hyla/test \
  --motion_model_path ../weights/femot_hmamba_tracker.pth \
  --config_file cfgs/tune5376/tune5376_lost_20.yaml \
  --device 0
```

DSEC tracking uses `weights/dsec_hmamba_tracker.pth` with the same entry point. The reported DSEC run sets `new_track_thresh` to `0.65`.

Predictions are written under `ssm_tracker/track_results/`. Evaluate them with TrackEval checked out at `../TrackEval`:

```bash
python trackeval/run_femot_eval.py \
  --split test \
  --pred_dir ssm_tracker/track_results/<run>/test \
  --tracker_name HMamba-eMOT

python trackeval/run_dsec_eval.py \
  --split test \
  --pred_dir ssm_tracker/track_results/<run>/test \
  --tracker_name HMamba-eMOT
```

## Layout

| Path | Role |
| --- | --- |
| `yolox_exps/custom/` | FEMOT and DSEC detector experiments |
| `third_party/ByteTrack/` | YOLOX training and the dual-modal backbone |
| `ssm_tracker/` | hypergraph prior, Mamba tracker, train and track entry points |
| `ssm_tracker/cfgs/` | tracker and inference configs |
| `tools/` | detection export and RoI feature extraction |
| `trackeval/` | wrappers around official TrackEval |
| `paper/` | manuscript sources |

## Citation

If you use this code, please cite the HMamba-eMOT paper. The manuscript source is in `paper/`.
