#!/usr/bin/env python3
"""Two retrains on the 44.713 structure.

fused_rgb: rgb hyperedge uses dark3 gate-sum features; evt stays enhanced.
dt_mul: Δ uses the same multiplicative residual as C.
Both drop evt+motion, freeze alpha_B, and keep the 44.713 inference settings.
The published 44.713 checkpoint is not retrained.
"""

import os
import subprocess
import time
from copy import deepcopy

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.environ.get("PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python")
BASE_CFG = os.path.join(
    PROJECT_ROOT, "ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod_mean_ep2.yaml"
)
INIT_CKPT = os.path.join(
    PROJECT_ROOT, "ssm_tracker/saved_ckpts/mambatrack_femot/epoch25.pth"
)
HYLA_EXP = os.path.join(
    PROJECT_ROOT, "yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py"
)
HYLA_CKPT = os.path.join(
    PROJECT_ROOT,
    "third_party/ByteTrack/YOLOX_outputs/yolox_femot_dual_hyla_shared_mlp_coco/"
    "best_ckpt_ep27_ap57.5_freeze.pth.tar",
)
HYLA_FEAT = os.path.join(PROJECT_ROOT, "data/femot_modal_gt_hyla")
FUSED_FEAT = "/wangx_new/lyt/femot_modal_gt_hyla_fused_rgb"
HYLA_DET = os.path.join(
    PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
)
SRC_DET = os.path.join(
    PROJECT_ROOT, "det_results/femot_dual_hyla_shared_mlp_coco_ep27_feats/test"
)
FUSED_DET = "/wangx_new/lyt/femot_hyla_fused_rgb_feats/test"
DATA_ROOT = "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
CFG_DIR = os.path.join(PROJECT_ROOT, "ssm_tracker/cfgs/ablate_retrain_hyla_pair")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/ablate_retrain_hyla_pair")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
SAVE_ROOT = "/wangx_new/lyt/ssm_mod_mean_retrain_ablate_hyla_pair"
MIN_FREE_MIB = 36000
ALLOWED_GPUS = list(range(8))
RGB_THRESHOLD = 0.8407866
TOPK = 5
EXPECTED = {"train": 70, "val": 10, "test": 20}

JOBS = [
    (
        "dt_mul",
        "hyla",
        {
            "dataset": {"ablate_drop_relations": ["evt", "motion"]},
            "train": {"ablate_zero_alphas": ["b"], "ssm_mod_dt_combine": "mul"},
        },
    ),
    (
        "fused_rgb",
        "fused",
        {
            "dataset": {"ablate_drop_relations": ["evt", "motion"]},
            "train": {"ablate_zero_alphas": ["b"]},
        },
    ),
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


def launch(name, gpu, script):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-lc", script],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=python_env(gpu),
        start_new_session=True,
    )
    return proc, log_path


def count_npz(folder):
    if not os.path.isdir(folder):
        return 0
    return len([name for name in os.listdir(folder) if name.endswith(".npz")])


def features_ready():
    train_ok = count_npz(os.path.join(FUSED_FEAT, "train")) >= EXPECTED["train"]
    val_ok = count_npz(os.path.join(FUSED_FEAT, "val")) >= EXPECTED["val"]
    test_rgb = (
        len(
            [
                name
                for name in os.listdir(FUSED_DET)
                if name.endswith("_rgb_feat.npz")
            ]
        )
        if os.path.isdir(FUSED_DET)
        else 0
    )
    return train_ok and val_ok and test_rgb >= EXPECTED["test"]


def extract_features():
    if features_ready():
        print("fused rgb features already present", flush=True)
        return
    jobs = []
    assignments = [
        (0, "extract_train_0", "train", 4, 0),
        (1, "extract_train_1", "train", 4, 1),
        (2, "extract_train_2", "train", 4, 2),
        (3, "extract_train_3", "train", 4, 3),
        (4, "extract_val", "val", 1, 0),
    ]
    for gpu, name, split, shards, index in assignments:
        script = f"""
set -eo pipefail
cd "{PROJECT_ROOT}"
"{PYTHON}" -u tools/gen_femot_gt_modal_features.py \
  --split {split} \
  --exp_file "{HYLA_EXP}" \
  --model_path "{HYLA_CKPT}" \
  --output_dir "{FUSED_FEAT}" \
  --fused_rgb \
  --num_shards {shards} \
  --shard_index {index} \
  --device 0
echo "DONE {name}"
"""
        proc, log_path = launch(name, gpu, script)
        jobs.append((name, proc, log_path))
        print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)

    test_script = f"""
set -eo pipefail
cd "{PROJECT_ROOT}"
"{PYTHON}" -u tools/gen_det_hyla_enhanced_feats.py \
  --src_dir "{SRC_DET}" \
  --dst_dir "{FUSED_DET}" \
  --exp_file "{HYLA_EXP}" \
  --model_path "{HYLA_CKPT}" \
  --fused_rgb \
  --device 0
echo "DONE extract_test"
"""
    proc, log_path = launch("extract_test", 5, test_script)
    jobs.append(("extract_test", proc, log_path))
    print(f"GPU 5 extract_test PID {proc.pid}", flush=True)

    failed = []
    while jobs:
        time.sleep(20)
        still = []
        for name, proc, log_path in jobs:
            code = proc.poll()
            if code is None:
                still.append((name, proc, log_path))
                continue
            print(f"{name} {'ok' if code == 0 else f'fail({code})'}", flush=True)
            if code != 0:
                failed.append(name)
        jobs = still
    if failed or not features_ready():
        raise SystemExit(f"feature extraction failed: {failed}")
    print("fused rgb features ready", flush=True)


def _pair_counts(features):
    n = len(features)
    if n < 2:
        return 0, 0
    features = features.astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    similarity = features @ features.T
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(TOPK, n - 1)
    nearest = np.argpartition(work, -k, axis=1)[:, -k:]
    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[np.arange(n)[:, None], nearest] = True
    mutual = nearest_mask & nearest_mask.T
    np.fill_diagonal(mutual, False)
    mutual_pairs = int(mutual.sum() // 2)
    passed = int((mutual & (similarity >= RGB_THRESHOLD)).sum() // 2)
    return passed, mutual_pairs


def cosine_report():
    """Compare how often mutual top-5 pairs clear the published rgb threshold."""
    enhanced_root = os.path.join(HYLA_FEAT, "train")
    fused_root = os.path.join(FUSED_FEAT, "train")
    names = sorted(
        name for name in os.listdir(fused_root) if name.endswith(".npz")
    )
    totals = {"enhanced": [0, 0], "fused": [0, 0], "frames": 0}
    for name in names:
        enhanced_path = os.path.join(enhanced_root, name)
        if not os.path.isfile(enhanced_path):
            continue
        enhanced = np.load(enhanced_path)
        fused = np.load(os.path.join(fused_root, name))
        if enhanced["rgb_feats"].shape[0] != fused["rgb_feats"].shape[0]:
            raise SystemExit(f"feature length mismatch in {name}")
        enhanced_keys = list(
            zip(
                enhanced["frame_ids"].tolist(),
                enhanced["object_ids"].tolist(),
            )
        )
        fused_keys = list(
            zip(fused["frame_ids"].tolist(), fused["object_ids"].tolist())
        )
        if enhanced_keys != fused_keys:
            raise SystemExit(f"frame/object alignment mismatch in {name}")
        frames = {}
        for index, frame_id in enumerate(fused["frame_ids"].tolist()):
            frames.setdefault(int(frame_id), []).append(index)
        for indexes in frames.values():
            if len(indexes) < 2:
                continue
            totals["frames"] += 1
            for key, array in (
                ("enhanced", enhanced["rgb_feats"]),
                ("fused", fused["rgb_feats"]),
            ):
                passed, mutual = _pair_counts(array[indexes])
                totals[key][0] += passed
                totals[key][1] += mutual
    lines = [
        f"rgb pair check frames={totals['frames']} threshold={RGB_THRESHOLD} topk={TOPK}"
    ]
    for key in ("enhanced", "fused"):
        passed, mutual = totals[key]
        rate = passed / mutual if mutual else 0.0
        lines.append(
            f"{key}: passed={passed} mutual_topk_pairs={mutual} rate={rate:.4f}"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(LOG_DIR, "rgb_pair_rates.txt"), "w") as handle:
        handle.write(text + "\n")


def job_done(name):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    if not os.path.isfile(log_path):
        return False
    with open(log_path) as handle:
        return f"DONE {name}" in handle.read()


def external_jobs():
    found = {}
    listing = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    marker = "cfgs/ablate_retrain_hyla_pair/pair_"
    for line in listing.splitlines():
        if marker not in line or ("train.py" not in line and "track.py" not in line):
            continue
        pid_text, cmd = line.strip().split(None, 1)
        name = cmd.split(marker, 1)[1].split(".yaml", 1)[0]
        if name.startswith("pair_"):
            name = name[len("pair_") :]
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


def run_job(gpu, name, feature, overrides, base):
    cfg = deepcopy(base)
    cfg["inference"]["enable_time_thresh"] = 2
    if feature == "fused":
        cfg.setdefault("dataset", {})["modal_feature_root"] = FUSED_FEAT
        det = FUSED_DET
    else:
        cfg.setdefault("dataset", {})["modal_feature_root"] = "../data/femot_modal_gt_hyla"
        det = HYLA_DET
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    for key in ("ablate_drop_relations", "ablate_keep_relations"):
        if key in cfg.get("dataset", {}):
            cfg["inference"][key] = cfg["dataset"][key]
    cfg_name = f"pair_{name}"
    cfg_path = os.path.join(CFG_DIR, cfg_name + ".yaml")
    with open(cfg_path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    dataset_name = os.path.basename(os.path.dirname(det))
    ckpt = os.path.join(SAVE_ROOT, cfg_name, "epoch2.pth")
    pred_dir = os.path.join(
        PROJECT_ROOT,
        "ssm_tracker/track_results",
        f"{dataset_name}_{cfg_name}",
        "test",
    )
    tracker_name = f"MambaTrack_{cfg_name}"
    script = f"""
set -eo pipefail
echo "GPU={gpu} NAME={name} FEATURE={feature} TRAIN"
cd "{PROJECT_ROOT}/ssm_tracker"
"{PYTHON}" -u train.py \
  --exp_name "{cfg_name}" \
  --config_file "cfgs/ablate_retrain_hyla_pair/{cfg_name}.yaml" \
  --init_ckpt "{INIT_CKPT}" \
  --save_path "{SAVE_ROOT}" \
  --device 0
echo "GPU={gpu} NAME={name} EVAL {ckpt}"
"{PYTHON}" -u track.py \
  --det_path "{det}" \
  --motion_model_path "{ckpt}" \
  --config_file "cfgs/ablate_retrain_hyla_pair/{cfg_name}.yaml" \
  --data_root "{DATA_ROOT}" \
  --device 0
cd "{PROJECT_ROOT}"
flock "{LOCK}" "{PYTHON}" -u trackeval/run_femot_eval.py \
  --split test \
  --pred_dir "{pred_dir}" \
  --tracker_name "{tracker_name}"
echo "DONE {name}"
"""
    return launch(name, gpu, script)


def train_jobs(ready_names):
    with open(BASE_CFG) as handle:
        base = yaml.safe_load(handle)
    summary = os.path.join(LOG_DIR, "summary.txt")
    running = []
    failed = set()
    while True:
        done = {name for name, _, _ in JOBS if job_done(name)}
        outside = external_jobs()
        owned = {name for _, name, _, _ in running}
        pending = [
            (name, feature, overrides)
            for name, feature, overrides in JOBS
            if name in ready_names
            and name not in done
            and name not in failed
            and name not in outside
            and name not in owned
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
            name, feature, overrides = pending.pop(0)
            proc, log_path = run_job(gpu, name, feature, overrides, base)
            running.append((gpu, name, proc, log_path))
            print(f"GPU {gpu} {name} PID {proc.pid}", flush=True)
        if not running and pending and not slots:
            print("waiting for a GPU with >=36GB free", flush=True)
            time.sleep(30)
            continue
        if not running and not pending:
            break
        time.sleep(20)
        still = []
        for gpu, name, proc, log_path in running:
            code = proc.poll()
            if code is None:
                still.append((gpu, name, proc, log_path))
                continue
            status = "ok" if code == 0 else f"fail({code})"
            if code != 0:
                failed.add(name)
            print(f"{name} {status}", flush=True)
            with open(summary, "a") as handle:
                handle.write(f"{name}\t{status}\t{log_path}\n")
        running = still
    if failed:
        raise SystemExit(f"failed jobs: {sorted(failed)}")


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    open(LOCK, "a").close()
    for path in (BASE_CFG, INIT_CKPT, HYLA_CKPT, HYLA_DET, SRC_DET):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    # Δ multiply does not need new features. Keep it off the extraction cards.
    if not job_done("dt_mul"):
        with open(BASE_CFG) as handle:
            base = yaml.safe_load(handle)
        proc, _ = run_job(
            7,
            "dt_mul",
            "hyla",
            JOBS[0][2],
            base,
        )
        print(f"GPU 7 dt_mul PID {proc.pid}", flush=True)

    extract_features()
    cosine_report()
    train_jobs({"dt_mul", "fused_rgb"})
    print("ALL HYLA PAIR ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
