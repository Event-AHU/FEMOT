#!/usr/bin/env python3
"""Pairwise-graph control on the 44.713 thresholds.

Spatial and RGB relations are 2-node edges. The context is concatenated with
the motion embedding and projected into Mamba. Delta and C are not modulated.
"""

import os
import subprocess
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.environ.get("PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python")
INIT_CKPT = os.path.join(
    PROJECT_ROOT, "ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"
)
CFG_NAME = "graph_concat"
CFG_REL = f"cfgs/ablate_retrain_hyla_graph/{CFG_NAME}.yaml"
DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
)
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/ablate_retrain_hyla_graph")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
SAVE_ROOT = "/wangx_new/lyt/ssm_mod_mean_retrain_ablate_hyla_graph"
MIN_FREE_MIB = 8192
ALLOWED_GPUS = list(range(8))


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


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    open(LOCK, "a").close()
    cfg_path = os.path.join(PROJECT_ROOT, "ssm_tracker", CFG_REL)
    for path in (cfg_path, INIT_CKPT, DET):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    log_path = os.path.join(LOG_DIR, f"{CFG_NAME}.log")
    if os.path.isfile(log_path) and f"DONE {CFG_NAME}" in open(log_path).read():
        print(f"already done: {log_path}", flush=True)
        return

    gpu = None
    while gpu is None:
        free = memory_free()
        slots = [
            index
            for index in ALLOWED_GPUS
            if free.get(index, 0) >= MIN_FREE_MIB
        ]
        if slots:
            gpu = max(slots, key=lambda index: free[index])
            break
        print(
            "waiting for a GPU with >=8GB free: "
            + ", ".join(f"{index}={free.get(index, 0)}MiB" for index in ALLOWED_GPUS),
            flush=True,
        )
        time.sleep(60)

    dataset_name = os.path.basename(os.path.dirname(DET))
    ckpt = os.path.join(SAVE_ROOT, CFG_NAME, "epoch2.pth")
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"{dataset_name}_{CFG_NAME}",
        "test",
    )
    tracker_name = f"MambaTrack_{CFG_NAME}"
    script = f"""
set -eo pipefail
echo "GPU={gpu} NAME={CFG_NAME} TRAIN"
cd "{PROJECT_ROOT}/ssm_tracker"
"{PYTHON}" -u train.py \
  --exp_name "{CFG_NAME}" \
  --config_file "{CFG_REL}" \
  --init_ckpt "{INIT_CKPT}" \
  --save_path "{SAVE_ROOT}" \
  --device 0
echo "GPU={gpu} NAME={CFG_NAME} EVAL {ckpt}"
"{PYTHON}" -u track.py \
  --det_path "{DET}" \
  --motion_model_path "{ckpt}" \
  --config_file "{CFG_REL}" \
  --data_root "{DATA_ROOT}" \
  --device 0
cd "{PROJECT_ROOT}"
flock "{LOCK}" "{PYTHON}" -u trackeval/run_femot_eval.py \
  --split test \
  --pred_dir "{pred_dir}" \
  --tracker_name "{tracker_name}"
echo "DONE {CFG_NAME}"
"""
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-lc", script],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=python_env(gpu),
        start_new_session=True,
    )
    print(f"GPU {gpu} {CFG_NAME} PID {proc.pid} free={memory_free().get(gpu)}MiB", flush=True)
    code = proc.wait()
    status = "ok" if code == 0 else f"fail({code})"
    print(f"{CFG_NAME} {status}", flush=True)
    with open(os.path.join(LOG_DIR, "summary.txt"), "a") as handle:
        handle.write(f"{CFG_NAME}\t{status}\t{log_path}\n")
    if code != 0:
        raise SystemExit(code)
    print("GRAPH CONCAT ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
