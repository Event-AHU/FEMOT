#!/usr/bin/env python3
"""Retrain the 14 ssm_mod_mean ablations, then evaluate epoch 2.

Same recipe as the published run: init from FEMOT baseline epoch25,
Adam 5e-5, 3 epochs, mean hyperedge aggregation. Evaluation uses the
epoch-2 checkpoint, HyLa ep27 feats, and enable_time=2.
"""

import os
import subprocess
import time
from copy import deepcopy

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.environ.get("PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python")
BASE_CFG = os.path.join(
    PROJECT_ROOT, "ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod_mean_ep2.yaml"
)
INIT_CKPT = os.path.join(
    PROJECT_ROOT, "ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"
)
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/ablate_retrain")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/ablate_retrain")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
SAVE_ROOT = os.environ.get(
    "ABLATE_SAVE_ROOT", "/wangx_new/lyt/ssm_mod_mean_retrain_ablate"
)
# One full training job per card. Leave headroom for jobs already on that GPU.
MIN_FREE_MIB = 36000
ALLOWED_GPUS = list(range(8))

JOBS = [
    ("edge_drop_spatial", {"dataset": {"ablate_drop_relations": ["spatial"]}}),
    ("edge_drop_rgb", {"dataset": {"ablate_drop_relations": ["rgb"]}}),
    ("edge_drop_evt", {"dataset": {"ablate_drop_relations": ["evt"]}}),
    ("edge_drop_motion", {"dataset": {"ablate_drop_relations": ["motion"]}}),
    ("edge_only_spatial", {"dataset": {"ablate_keep_relations": ["spatial"]}}),
    ("edge_only_rgb", {"dataset": {"ablate_keep_relations": ["rgb"]}}),
    ("edge_only_evt", {"dataset": {"ablate_keep_relations": ["evt"]}}),
    ("edge_only_motion", {"dataset": {"ablate_keep_relations": ["motion"]}}),
    ("mod_zero_all", {"train": {"ablate_zero_alphas": ["dt", "b", "c"]}}),
    ("mod_zero_dt", {"train": {"ablate_zero_alphas": ["dt"]}}),
    ("mod_zero_b", {"train": {"ablate_zero_alphas": ["b"]}}),
    ("mod_zero_c", {"train": {"ablate_zero_alphas": ["c"]}}),
    ("mod_lastk_1", {"train": {"ssm_mod_last_k": 1}}),
    ("mod_lastk_3", {"train": {"ssm_mod_last_k": 3}}),
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


def job_done(name):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    if not os.path.isfile(log_path):
        return False
    with open(log_path) as handle:
        return f"DONE {name}" in handle.read()


def external_jobs():
    found = {}
    listing = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    for line in listing.splitlines():
        if "cfgs/ablate_retrain/retrain_" not in line or "train.py" not in line:
            continue
        pid_text, cmd = line.strip().split(None, 1)
        name = cmd.split("cfgs/ablate_retrain/retrain_", 1)[1].split(".yaml", 1)[0]
        gpu = 0
        try:
            with open(f"/proc/{pid_text}/environ", "rb") as handle:
                env = handle.read().split(b"\0")
            for item in env:
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    gpu = int(item.split(b"=", 1)[1].split(b",")[0])
        except OSError:
            pass
        found[name] = gpu
    return found


def run_job(gpu, name, overrides, base):
    cfg = deepcopy(base)
    cfg["inference"]["enable_time_thresh"] = 2
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    # Inference must drop the same relations the model was trained without.
    for key in ("ablate_drop_relations", "ablate_keep_relations"):
        if key in cfg.get("dataset", {}):
            cfg["inference"][key] = cfg["dataset"][key]
    cfg_name = f"retrain_{name}"
    cfg_path = os.path.join(CFG_DIR, cfg_name + ".yaml")
    with open(cfg_path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    ckpt = os.path.join(SAVE_ROOT, cfg_name, "epoch2.pth")
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"femot_dual_hyla_shared_mlp_coco_ep27_feats_{cfg_name}",
        "test",
    )
    tracker_name = f"MambaTrack_retrain_{name}"
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
echo "GPU={gpu} NAME={name} TRAIN"
cd "{PROJECT_ROOT}/ssm_tracker"
"{PYTHON}" -u train.py \
  --exp_name "{cfg_name}" \
  --config_file "cfgs/ablate_retrain/{cfg_name}.yaml" \
  --init_ckpt "{INIT_CKPT}" \
  --save_path "{SAVE_ROOT}" \
  --device 0
echo "GPU={gpu} NAME={name} EVAL {ckpt}"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{ckpt}" \
  --config_file "cfgs/ablate_retrain/{cfg_name}.yaml" \
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


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, INIT_CKPT, DET):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    with open(BASE_CFG) as handle:
        base = yaml.safe_load(handle)

    summary = os.path.join(LOG_DIR, "summary.txt")
    running = []
    while True:
        done = {name for name, _ in JOBS if job_done(name)}
        outside = external_jobs()
        owned = {name for _, name, _, _ in running}
        pending = [
            (name, overrides)
            for name, overrides in JOBS
            if name not in done and name not in outside and name not in owned
        ]
        if not pending and not running and not outside:
            break
        free = memory_free()
        used = {gpu for gpu in list(outside.values()) + [item[0] for item in running]}
        slots = [
            gpu
            for gpu in ALLOWED_GPUS
            if gpu not in used and free.get(gpu, 0) >= MIN_FREE_MIB
        ]
        while pending and slots:
            gpu = slots.pop(0)
            name, overrides = pending.pop(0)
            proc, log_path = run_job(gpu, name, overrides, base)
            running.append((gpu, name, proc, log_path))
            print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
        if not running and pending and not slots:
            print("waiting for a GPU with >=36GB free", flush=True)
            time.sleep(30)
            continue
        time.sleep(20)
        still = []
        for gpu, name, proc, log_path in running:
            code = proc.poll()
            if code is None:
                still.append((gpu, name, proc, log_path))
                continue
            status = "ok" if code == 0 else f"fail({code})"
            print(f"{name} {status}", flush=True)
            with open(summary, "a") as handle:
                handle.write(f"{name}\t{status}\t{log_path}\n")
        running = still
    print("ALL RETRAIN ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
