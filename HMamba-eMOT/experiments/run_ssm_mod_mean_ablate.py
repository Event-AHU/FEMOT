#!/usr/bin/env python3
"""Inference ablations on the best ssm_mod_mean checkpoint.

Baseline is epoch2 + HyLa ep27 feats + enable_time=2. Other inference
settings stay at the published center. Jobs run on currently free GPUs.
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
CKPT = os.path.join(
    PROJECT_ROOT,
    "ssm_tracker/saved_ckpts/mambatrack_femot_spatial_ssm_mod_mean/epoch2.pth",
)
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/ablate_ssm_mod_mean")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/ablate_ssm_mod_mean")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
ALLOWED_GPUS = list(range(7))
MAX_PER_GPU = 3
# Reserve 12GB per job so a card keeps 2 or 3 slots when others already hold ~40GB.
JOB_MIB = 12000

# (name, inference overrides). enable_time=2 is applied to every job.
JOBS = [
    ("edge_drop_spatial", {"ablate_drop_relations": ["spatial"]}),
    ("edge_drop_rgb", {"ablate_drop_relations": ["rgb"]}),
    ("edge_drop_evt", {"ablate_drop_relations": ["evt"]}),
    ("edge_drop_motion", {"ablate_drop_relations": ["motion"]}),
    ("edge_only_spatial", {"ablate_keep_relations": ["spatial"]}),
    ("edge_only_rgb", {"ablate_keep_relations": ["rgb"]}),
    ("edge_only_evt", {"ablate_keep_relations": ["evt"]}),
    ("edge_only_motion", {"ablate_keep_relations": ["motion"]}),
    ("mod_zero_all", {"ablate_zero_alphas": ["dt", "b", "c"]}),
    ("mod_zero_dt", {"ablate_zero_alphas": ["dt"]}),
    ("mod_zero_b", {"ablate_zero_alphas": ["b"]}),
    ("mod_zero_c", {"ablate_zero_alphas": ["c"]}),
    ("mod_lastk_1", {"ssm_mod_last_k": 1}),
    ("mod_lastk_3", {"ssm_mod_last_k": 3}),
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
    """Ablation trackers already started by a previous supervisor."""
    found = {}
    listing = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    for line in listing.splitlines():
        if "cfgs/ablate_ssm_mod_mean/ablate_" not in line or "track.py" not in line:
            continue
        pid_text, cmd = line.strip().split(None, 1)
        marker = "cfgs/ablate_ssm_mod_mean/ablate_"
        name = cmd.split(marker, 1)[1].split(".yaml", 1)[0]
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
    cfg["inference"].update(overrides)
    cfg_name = f"ablate_{name}"
    cfg_path = os.path.join(CFG_DIR, cfg_name + ".yaml")
    with open(cfg_path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"femot_dual_hyla_shared_mlp_coco_ep27_feats_{cfg_name}",
        "test",
    )
    tracker_name = f"MambaTrack_ablate_{name}"
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
cd "{PROJECT_ROOT}/ssm_tracker"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{CKPT}" \
  --config_file "cfgs/ablate_ssm_mod_mean/{cfg_name}.yaml" \
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
    open(LOCK, "a").close()
    for path in (BASE_CFG, CKPT, DET):
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
        load = {gpu: 0 for gpu in ALLOWED_GPUS}
        for gpu in list(outside.values()) + [item[0] for item in running]:
            if gpu in load:
                load[gpu] += 1
        slots = []
        for gpu in ALLOWED_GPUS:
            room = min(MAX_PER_GPU - load[gpu], free.get(gpu, 0) // JOB_MIB)
            slots.extend([gpu] * max(0, room))
        while pending and slots:
            gpu = slots.pop(0)
            name, overrides = pending.pop(0)
            proc, log_path = run_job(gpu, name, overrides, base)
            running.append((gpu, name, proc, log_path))
            print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
        if not running and pending and not slots:
            print("waiting for GPU memory", flush=True)
            time.sleep(30)
            continue
        time.sleep(15)
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
    print("ALL ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
