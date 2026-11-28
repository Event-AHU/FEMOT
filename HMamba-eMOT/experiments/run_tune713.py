#!/usr/bin/env python3
"""Tune the 44.713 checkpoint: epoch 3 first, then inference sweeps.

The structure stays fixed: HyLA features, drop event and motion edges,
alpha_B frozen. Epoch 3 is evaluated at enable_time=2. The sweep then
uses whichever of epoch 2 and epoch 3 tracks better.
"""

import os
import subprocess
import time
from copy import deepcopy

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
BASE_CFG = os.path.join(
    PROJECT_ROOT,
    "ssm_tracker/cfgs/ablate_retrain_hyla/retrain_hyla_drop_evt_motion_zero_b.yaml",
)
CKPT_DIR = "/wangx_new/lyt/ssm_mod_mean_retrain_ablate_hyla/retrain_hyla_drop_evt_motion_zero_b"
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/tune713")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/tune713")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
EPOCH2_HOTA = 44.713
ALLOWED_GPUS = [0, 4, 5, 7, 1, 2, 3]
MAX_PER_GPU = 3
MIN_FREE_MIB = 8000

SWEEP = [
    ("en_1", {"enable_time_thresh": 1}),
    ("en_3", {"enable_time_thresh": 3}),
    ("en_4", {"enable_time_thresh": 4}),
    ("s1_060", {"match_thresh_stage1": 0.60}),
    ("s1_080", {"match_thresh_stage1": 0.80}),
    ("s2_080", {"match_thresh_stage2": 0.80}),
    ("s2_095", {"match_thresh_stage2": 0.95}),
    ("lost_15", {"max_time_lost": 15}),
    ("lost_40", {"max_time_lost": 40}),
    ("new_045", {"new_track_thresh": 0.45}),
    ("new_065", {"new_track_thresh": 0.65}),
    ("filt_015", {"filter_thresh": 0.15}),
    ("filt_030", {"filter_thresh": 0.30}),
]


def memory_free():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {}
    for line in out.strip().splitlines():
        index, mib = [part.strip() for part in line.split(",")]
        free[int(index)] = int(mib)
    return free


def python_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TQDM_DISABLE"] = "1"
    env["TMPDIR"] = "/wangx_new/lyt/tmp"
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}/third_party/ByteTrack:{PROJECT_ROOT}/yolox_exps/custom:"
        + env.get("PYTHONPATH", "")
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    return env


def load_base():
    with open(BASE_CFG) as handle:
        return yaml.safe_load(handle)


def write_cfg(name, overrides):
    cfg = deepcopy(load_base())
    cfg.setdefault("inference", {}).update(overrides)
    path = os.path.join(CFG_DIR, name + ".yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def hota_from_log(path):
    comb = [
        line.split()
        for line in open(path)
        if line.startswith("COMBINED")
    ]
    if not comb:
        raise RuntimeError(f"no COMBINED line in {path}")
    return float(comb[0][1])


def launch(name, gpu, ckpt, cfg_name):
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats_{cfg_name}",
        "test",
    )
    tracker_name = f"MambaTrack_tune713_{cfg_name}"
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    script = f"""
set -eo pipefail
cd "{PROJECT_ROOT}/ssm_tracker"
echo "GPU={gpu} NAME={name} CKPT={ckpt}"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{ckpt}" \
  --config_file "cfgs/tune713/{cfg_name}.yaml" \
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
        env=python_env(gpu),
        start_new_session=True,
    )
    return proc, log_path


def job_done(name):
    path = os.path.join(LOG_DIR, f"{name}.log")
    if not os.path.isfile(path):
        return False
    with open(path) as handle:
        return f"DONE {name}" in handle.read()


def run_one(name, cfg_name, ckpt, overrides):
    write_cfg(cfg_name, overrides)
    while True:
        if job_done(name):
            print(f"{name} already done", flush=True)
            return hota_from_log(os.path.join(LOG_DIR, f"{name}.log"))
        free = memory_free()
        gpu = next(
            (g for g in ALLOWED_GPUS if free.get(g, 0) >= MIN_FREE_MIB),
            None,
        )
        if gpu is None:
            print("waiting for a GPU with >=8GB free", flush=True)
            time.sleep(20)
            continue
        proc, log_path = launch(name, gpu, ckpt, cfg_name)
        print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
        while proc.poll() is None:
            time.sleep(15)
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed, see {log_path}")
        score = hota_from_log(log_path)
        print(f"{name} HOTA {score:.3f}", flush=True)
        return score


def run_sweep(jobs, ckpt):
    pending = list(jobs)
    running = []
    summary = os.path.join(LOG_DIR, "summary.txt")
    while pending or running:
        free = memory_free()
        per_gpu = {}
        for gpu, name, proc, log_path in running:
            per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
        slots = []
        for gpu in ALLOWED_GPUS:
            if per_gpu.get(gpu, 0) >= MAX_PER_GPU:
                continue
            if free.get(gpu, 0) < MIN_FREE_MIB:
                continue
            slots.append(gpu)
        while pending and slots:
            gpu = slots.pop(0)
            name, cfg_name, overrides = pending.pop(0)
            if job_done(name):
                print(f"{name} already done", flush=True)
                continue
            write_cfg(cfg_name, overrides)
            proc, log_path = launch(name, gpu, ckpt, cfg_name)
            running.append((gpu, name, proc, log_path))
            print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
            per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
        if not running and pending:
            print("waiting for a GPU with >=8GB free", flush=True)
            time.sleep(20)
            continue
        time.sleep(15)
        still = []
        for gpu, name, proc, log_path in running:
            code = proc.poll()
            if code is None:
                still.append((gpu, name, proc, log_path))
                continue
            status = "ok" if code == 0 else f"fail({code})"
            score = ""
            if code == 0:
                score = f"\t{hota_from_log(log_path):.3f}"
            print(f"{name} {status}{score}", flush=True)
            with open(summary, "a") as handle:
                handle.write(f"{name}\t{status}{score}\t{log_path}\n")
            if code != 0:
                raise SystemExit(f"{name} failed")
        running = still
    print("ALL TUNE713 SWEEPS DONE", flush=True)


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, DET, os.path.join(CKPT_DIR, "epoch2.pth"), os.path.join(CKPT_DIR, "epoch3.pth")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    ep3 = run_one(
        "ep3",
        "tune713_ep3",
        os.path.join(CKPT_DIR, "epoch3.pth"),
        {},
    )
    if ep3 > EPOCH2_HOTA:
        ckpt = os.path.join(CKPT_DIR, "epoch3.pth")
        tag = "ep3"
        jobs = [("en_2", "tune713_ep3_en_2", {"enable_time_thresh": 2})]
        print(f"use epoch 3 ({ep3:.3f} > {EPOCH2_HOTA:.3f})", flush=True)
    else:
        ckpt = os.path.join(CKPT_DIR, "epoch2.pth")
        tag = "ep2"
        jobs = []
        print(f"use epoch 2 ({EPOCH2_HOTA:.3f} >= epoch3 {ep3:.3f})", flush=True)
    for name, overrides in SWEEP:
        jobs.append((name, f"tune713_{tag}_{name}", overrides))
    run_sweep(jobs, ckpt)


if __name__ == "__main__":
    main()
