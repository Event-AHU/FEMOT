#!/usr/bin/env python3
"""8-GPU inference sweep for the HyLa-ep27 + ssm_mod_mean-ep2 combo.

Waves run one after another. Each wave puts one job on each GPU.
"""

import os
import subprocess
import sys
import time
from copy import deepcopy

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.environ.get("PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python")
BASE_CFG = os.path.join(
    PROJECT_ROOT, "ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod_mean_ep2.yaml"
)
CKPT = os.path.join(
    PROJECT_ROOT,
    "ssm_tracker/saved_ckpts/mambatrack_femot_spatial_ssm_mod_mean/epoch2.pth",
)
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/combo_tune")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/combo_tune")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")

# (name, {section.key: value})
WAVES = [
    [
        ("w1_s1_055", {"inference.match_thresh_stage1": 0.55}),
        ("w1_s1_080", {"inference.match_thresh_stage1": 0.80}),
        ("w1_s1_085", {"inference.match_thresh_stage1": 0.85}),
        ("w1_s2_075", {"inference.match_thresh_stage2": 0.75}),
        ("w1_s2_095", {"inference.match_thresh_stage2": 0.95}),
        ("w1_lost_15", {"inference.max_time_lost": 15}),
        ("w1_lost_40", {"inference.max_time_lost": 40}),
        ("w1_lost_60", {"inference.max_time_lost": 60}),
    ],
    [
        ("w2_new_040", {"inference.new_track_thresh": 0.40}),
        ("w2_new_050", {"inference.new_track_thresh": 0.50}),
        ("w2_new_065", {"inference.new_track_thresh": 0.65}),
        ("w2_new_075", {"inference.new_track_thresh": 0.75}),
        ("w2_filt_010", {"inference.filter_thresh": 0.10}),
        ("w2_filt_015", {"inference.filter_thresh": 0.15}),
        ("w2_filt_030", {"inference.filter_thresh": 0.30}),
        ("w2_filt_040", {"inference.filter_thresh": 0.40}),
    ],
    [
        ("w3_en_3", {"inference.enable_time_thresh": 3}),
        ("w3_en_4", {"inference.enable_time_thresh": 4}),
        ("w3_en_8", {"inference.enable_time_thresh": 8}),
        ("w3_en_10", {"inference.enable_time_thresh": 10}),
        ("w3_win_6", {"inference.max_window": 6}),
        ("w3_win_8", {"inference.max_window": 8}),
        ("w3_win_14", {"inference.max_window": 14}),
        ("w3_win_16", {"inference.max_window": 16}),
    ],
    [
        ("w4_rad_005", {"dataset.hyper_spatial_radius": 0.05}),
        ("w4_rad_008", {"dataset.hyper_spatial_radius": 0.08}),
        ("w4_rad_015", {"dataset.hyper_spatial_radius": 0.15}),
        ("w4_rad_020", {"dataset.hyper_spatial_radius": 0.20}),
        ("w4_topk_2", {"dataset.hyper_topk": 2}),
        ("w4_topk_3", {"dataset.hyper_topk": 3}),
        ("w4_topk_8", {"dataset.hyper_topk": 8}),
        ("w4_topk_12", {"dataset.hyper_topk": 12}),
    ],
    [
        ("w5_rgb_070", {"dataset.hyper_rgb_threshold": 0.70}),
        ("w5_rgb_078", {"dataset.hyper_rgb_threshold": 0.78}),
        ("w5_rgb_090", {"dataset.hyper_rgb_threshold": 0.90}),
        ("w5_rgb_095", {"dataset.hyper_rgb_threshold": 0.95}),
        ("w5_evt_970", {"dataset.hyper_evt_threshold": 0.97}),
        ("w5_evt_990", {"dataset.hyper_evt_threshold": 0.99}),
        ("w5_evt_998", {"dataset.hyper_evt_threshold": 0.998}),
        ("w5_evt_999", {"dataset.hyper_evt_threshold": 0.999}),
    ],
    [
        ("w6_re_l02", {
            "inference.match_mode": "rematch",
            "inference.assoc_modal_lambda": 0.2,
        }),
        ("w6_re_l04", {
            "inference.match_mode": "rematch",
            "inference.assoc_modal_lambda": 0.4,
        }),
        ("w6_re_l06", {
            "inference.match_mode": "rematch",
            "inference.assoc_modal_lambda": 0.6,
        }),
        ("w6_re_gate010", {
            "inference.match_mode": "rematch",
            "inference.assoc_modal_lambda": 0.4,
            "inference.assoc_iou_gate": 0.10,
        }),
        ("w6_hc_a08", {
            "inference.match_mode": "hyper_cost",
            "inference.assoc_alpha": 0.8,
            "inference.assoc_beta": 0.2,
        }),
        ("w6_hc_a06", {
            "inference.match_mode": "hyper_cost",
            "inference.assoc_alpha": 0.6,
            "inference.assoc_beta": 0.4,
        }),
        ("w6_hc_pred050", {
            "inference.match_mode": "hyper_cost",
            "inference.assoc_w_pred": 0.50,
        }),
        ("w6_hc_tri_pred", {
            "inference.match_mode": "hyper_cost",
            "inference.assoc_w_tri": 0.30,
            "inference.assoc_w_pred_stage2": 0.50,
        }),
    ],
]


def apply_overrides(cfg, overrides):
    cfg = deepcopy(cfg)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        cfg.setdefault(section, {})[key] = value
    return cfg


def run_job(gpu, name, overrides, base):
    cfg_name = f"combo_tune_{name}"
    cfg_path = os.path.join(CFG_DIR, cfg_name + ".yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(apply_overrides(base, overrides), f, sort_keys=False)

    # track.py names the folder from the det parent plus the config basename.
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"femot_dual_hyla_shared_mlp_coco_ep27_feats_{cfg_name}",
        "test",
    )
    tracker_name = f"MambaTrack_combo_tune_{name}"
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TQDM_DISABLE"] = "1"
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}/third_party/ByteTrack:{PROJECT_ROOT}/yolox_exps/custom:"
        + env.get("PYTHONPATH", "")
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    script = f"""
set -eo pipefail
echo "GPU={gpu} NAME={name}"
echo "CFG={cfg_path}"
echo "PRED={pred_dir}"
cd "{PROJECT_ROOT}/ssm_tracker"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{CKPT}" \
  --config_file "cfgs/combo_tune/{cfg_name}.yaml" \
  --data_root "{DATA_ROOT}" \
  --device 0
cd "{PROJECT_ROOT}"
flock "{LOCK}" "{PYTHON}" -u trackeval/run_femot_eval.py \
  --split test \
  --pred_dir "{pred_dir}" \
  --tracker_name "{tracker_name}"
echo "DONE {name}"
"""
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-lc", script],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return proc, log_path


ENABLE_TIME_JOBS = [
    (0, "w3_en_1", {"inference.enable_time_thresh": 1}),
    (1, "w3_en_2", {"inference.enable_time_thresh": 2}),
    (2, "w3_en_5", {"inference.enable_time_thresh": 5}),
    (4, "w3_en_7", {"inference.enable_time_thresh": 7}),
]


def main_enable_time():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, CKPT, DET):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    with open(BASE_CFG) as f:
        base = yaml.safe_load(f)
    summary = os.path.join(LOG_DIR, "summary.txt")
    print("=== enable_time 1,2,5,7 ===", flush=True)
    procs = []
    for gpu, name, overrides in ENABLE_TIME_JOBS:
        proc, log_path = run_job(gpu, name, overrides, base)
        procs.append((name, proc, log_path))
        print(f"  GPU {gpu} {name} PID {proc.pid}", flush=True)
    failed = []
    for name, proc, log_path in procs:
        code = proc.wait()
        status = "ok" if code == 0 else f"fail({code})"
        print(f"  {name} {status}", flush=True)
        if code != 0:
            failed.append(name)
        with open(summary, "a") as f:
            f.write(f"enable_time\t{name}\t{status}\t{log_path}\n")
    if failed:
        print(f"failures: {failed}", flush=True)
    print("ENABLE_TIME DONE", flush=True)


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, CKPT, DET):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    with open(BASE_CFG) as f:
        base = yaml.safe_load(f)

    summary = os.path.join(LOG_DIR, "summary.txt")
    for wave_i, wave in enumerate(WAVES, start=1):
        print(f"=== wave {wave_i}/{len(WAVES)} ===", flush=True)
        procs = []
        for gpu, (name, overrides) in enumerate(wave):
            proc, log_path = run_job(gpu, name, overrides, base)
            procs.append((name, proc, log_path))
            print(f"  GPU {gpu} {name} PID {proc.pid}", flush=True)
        failed = []
        for name, proc, log_path in procs:
            code = proc.wait()
            status = "ok" if code == 0 else f"fail({code})"
            print(f"  {name} {status}", flush=True)
            if code != 0:
                failed.append(name)
            with open(summary, "a") as f:
                f.write(f"wave{wave_i}\t{name}\t{status}\t{log_path}\n")
        if failed:
            print(f"wave {wave_i} failures: {failed}", flush=True)
    print("ALL WAVES DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "enable_time":
        main_enable_time()
    else:
        main()
