#!/usr/bin/env python3
"""Local inference sweep around HOTA 45.191.

Center is epoch 2, stage-2 match 0.80, filter 0.25. Each job moves one
of filter, stage-1 match, or max_time_lost. Event and motion edges stay
dropped, and alpha_B stays zero.
"""

import os
import subprocess
import time
from copy import deepcopy

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
BASE_CFG = os.path.join(
    PROJECT_ROOT, "ssm_tracker/cfgs/tune938/tune938_filt_025.yaml"
)
CKPT = (
    "/wangx_new/lyt/ssm_mod_mean_retrain_ablate_hyla/"
    "retrain_hyla_drop_evt_motion_zero_b/epoch2.pth"
)
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/tune5191")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/tune5191")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
ALLOWED_GPUS = [0, 1, 2, 3, 4, 5, 7]
MAX_PER_GPU = 2
MIN_FREE_MIB = 8000

SWEEP = [
    ("filt_022", {"filter_thresh": 0.22}),
    ("filt_028", {"filter_thresh": 0.28}),
    ("s1_075", {"match_thresh_stage1": 0.75}),
    ("lost_20", {"max_time_lost": 20}),
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


def write_cfg(name, overrides):
    with open(BASE_CFG) as handle:
        cfg = yaml.safe_load(handle)
    cfg = deepcopy(cfg)
    cfg.setdefault("inference", {}).update(overrides)
    path = os.path.join(CFG_DIR, name + ".yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def hota_from_log(path):
    comb = [line.split() for line in open(path) if line.startswith("COMBINED")]
    if not comb:
        raise RuntimeError(f"no COMBINED line in {path}")
    return float(comb[0][1])


def launch(name, gpu, cfg_name):
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats_{cfg_name}",
        "test",
    )
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    script = f"""
set -eo pipefail
cd "{PROJECT_ROOT}/ssm_tracker"
echo "GPU={gpu} NAME={name} CKPT={CKPT}"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{CKPT}" \
  --config_file "cfgs/tune5191/{cfg_name}.yaml" \
  --data_root "{DATA_ROOT}" \
  --device 0
cd "{PROJECT_ROOT}"
flock "{LOCK}" "{PYTHON}" -u trackeval/run_femot_eval.py \
  --split test \
  --pred_dir "{pred_dir}" \
  --tracker_name "MambaTrack_{cfg_name}"
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


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, DET, CKPT):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    pending = [(name, f"tune5191_{name}", overrides) for name, overrides in SWEEP]
    running = []
    summary = os.path.join(LOG_DIR, "summary.txt")
    while pending or running:
        free = memory_free()
        per_gpu = {}
        for gpu, name, proc, log_path in running:
            per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
        slots = [
            gpu
            for gpu in ALLOWED_GPUS
            if per_gpu.get(gpu, 0) < MAX_PER_GPU and free.get(gpu, 0) >= MIN_FREE_MIB
        ]
        while pending and slots:
            gpu = slots.pop(0)
            name, cfg_name, overrides = pending.pop(0)
            if job_done(name):
                print(f"{name} already done", flush=True)
                continue
            write_cfg(cfg_name, overrides)
            proc, log_path = launch(name, gpu, cfg_name)
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
    print("ALL TUNE5191 SWEEPS DONE", flush=True)


if __name__ == "__main__":
    main()
