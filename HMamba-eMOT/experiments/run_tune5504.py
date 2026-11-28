#!/usr/bin/env python3
"""Neighbors of max_time_lost=20 on the HOTA 45.504 setting.

Center is epoch 2, stage-1 0.75, stage-2 0.80, filter 0.25, lost=20.
Jobs are lost=15 and lost=22. Event and motion stay dropped, alpha_B stays zero.
"""

import os
import subprocess
import time
from copy import deepcopy

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
BASE_CFG = os.path.join(
    PROJECT_ROOT, "ssm_tracker/cfgs/tune5376/tune5376_lost_20.yaml"
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
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/tune5504")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/tune5504")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
ALLOWED_GPUS = [0, 1, 2, 3, 4, 5, 7]
MIN_FREE_MIB = 8000

SWEEP = [
    ("lost_15", {"max_time_lost": 15}),
    ("lost_22", {"max_time_lost": 22}),
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
        cfg = deepcopy(yaml.safe_load(handle))
    cfg.setdefault("inference", {}).update(overrides)
    path = os.path.join(CFG_DIR, name + ".yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


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
  --config_file "cfgs/tune5504/{cfg_name}.yaml" \
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


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, DET, CKPT):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    pending = [(name, f"tune5504_{name}", overrides) for name, overrides in SWEEP]
    running = []
    summary = os.path.join(LOG_DIR, "summary.txt")
    while pending or running:
        free = memory_free()
        used = {gpu for gpu, _, _, _ in running}
        slots = [
            gpu
            for gpu in ALLOWED_GPUS
            if gpu not in used and free.get(gpu, 0) >= MIN_FREE_MIB
        ]
        while pending and slots:
            gpu = slots.pop(0)
            name, cfg_name, overrides = pending.pop(0)
            write_cfg(cfg_name, overrides)
            proc, log_path = launch(name, gpu, cfg_name)
            running.append((gpu, name, proc, log_path))
            print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
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
            score = f"\t{hota_from_log(log_path):.3f}" if code == 0 else ""
            print(f"{name} {status}{score}", flush=True)
            with open(summary, "a") as handle:
                handle.write(f"{name}\t{status}{score}\t{log_path}\n")
            if code != 0:
                raise SystemExit(f"{name} failed")
        running = still
    print("ALL TUNE5504 SWEEPS DONE", flush=True)


if __name__ == "__main__":
    main()
