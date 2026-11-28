#!/usr/bin/env python3
"""HyLA-feature control plus the three combinations that beat 44.038.

full: all four hyperedges and Δ/B/C.
drop_evt_motion: drop event and motion edges.
drop_evt_zero_b: drop event edges and freeze alpha_B at 0.
drop_evt_motion_zero_b: both edge drops and alpha_B frozen.
Same recipe as the 14 retrains: epoch25 init, Adam 5e-5, 3 epochs,
epoch-2 eval, enable_time=2, HyLA-enhanced dark3 features.
"""

import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_ssm_mod_mean_retrain_ablate_hyla import (
    BASE_CFG,
    DET,
    HYLA_CKPT,
    INIT_CKPT,
    LOG_DIR,
    MIN_FREE_MIB,
    SAVE_ROOT,
    ALLOWED_GPUS,
    external_jobs,
    job_done,
    memory_free,
    run_job,
)

JOBS = [
    ("full", {}),
    (
        "drop_evt_motion",
        {"dataset": {"ablate_drop_relations": ["evt", "motion"]}},
    ),
    (
        "drop_evt_zero_b",
        {
            "dataset": {"ablate_drop_relations": ["evt"]},
            "train": {"ablate_zero_alphas": ["b"]},
        },
    ),
    (
        "drop_evt_motion_zero_b",
        {
            "dataset": {"ablate_drop_relations": ["evt", "motion"]},
            "train": {"ablate_zero_alphas": ["b"]},
        },
    ),
]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.environ.setdefault("TMPDIR", "/wangx_new/lyt/tmp")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    for path in (BASE_CFG, INIT_CKPT, DET, HYLA_CKPT):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    with open(BASE_CFG) as handle:
        base = yaml.safe_load(handle)

    summary = os.path.join(LOG_DIR, "combo_summary.txt")
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
    print("ALL HYLA COMBO ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
