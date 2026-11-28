#!/usr/bin/env python3
"""Short-train gamma/radius ablations; TrackEval ep1 only. Multi-GPU."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get(
    "PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
)
INIT_CKPT = "saved_ckpts/mambatrack_femot/epoch25.pth"
BASE_CFG = (
    PROJECT_ROOT
    / "ssm_tracker/cfgs/MambaTrack_femot_spatial_hyper_mvp_lr5e5_g05.yaml"
)
DET_PATH = PROJECT_ROOT / "det_results/femot_dual/test"
DATA_ROOT = (
    "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
)
LOG_DIR = PROJECT_ROOT / "logs"
CFG_DIR = PROJECT_ROOT / "ssm_tracker/cfgs"
CKPT_ROOT = PROJECT_ROOT / "ssm_tracker/saved_ckpts"
TRACKERS = PROJECT_ROOT / "trackeval/trackers/test"
ANALYSIS = PROJECT_ROOT / "analysis"

# One factor at a time vs best recipe (gamma=0.5, radius=0.10).
EXPS = [
    {"tag": "g03", "gamma": 0.3, "radius": 0.10},
    {"tag": "g07", "gamma": 0.7, "radius": 0.10},
    {"tag": "r008", "gamma": 0.5, "radius": 0.08},
    {"tag": "r012", "gamma": 0.5, "radius": 0.12},
]


def best_inference(cfg: dict) -> None:
    inf = cfg["inference"]
    inf["match_mode"] = "iou"
    inf["filter_thresh"] = 0.20
    inf["match_thresh_stage1"] = 0.70
    inf["match_thresh_stage2"] = 0.90
    inf["new_track_thresh"] = 0.55
    inf["max_time_lost"] = 25


def write_cfg(exp: dict) -> tuple[str, Path]:
    exp_name = f"mambatrack_femot_spatial_hyper_{exp['tag']}"
    cfg_name = f"MambaTrack_femot_spatial_hyper_{exp['tag']}"
    cfg_path = CFG_DIR / f"{cfg_name}.yaml"
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["dataset"]["hyper_spatial_radius"] = float(exp["radius"])
    cfg["train"]["lr0"] = 5e-5
    cfg["train"]["epochs"] = 1  # only need ep1
    cfg["train"]["grad_clip"] = 1.0
    cfg["train"]["spatial_hyper_gamma_init"] = float(exp["gamma"])
    best_inference(cfg)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return exp_name, cfg_path


def run_cmd(cmd, cwd, log, env):
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def train_one(gpu: str, exp: dict) -> dict:
    exp_name, cfg_path = write_cfg(exp)
    cfg_rel = f"cfgs/{cfg_path.name}"
    log_file = LOG_DIR / f"train_{exp['tag']}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}/third_party/ByteTrack:"
        f"{PROJECT_ROOT}/yolox_exps/custom:{env.get('PYTHONPATH', '')}"
    )
    with open(log_file, "w") as log:
        log.write(
            f"GPU={gpu} tag={exp['tag']} gamma={exp['gamma']} "
            f"radius={exp['radius']} epochs=1\n"
        )
        rc = run_cmd(
            [
                PYTHON, "-u", "train.py",
                "--exp_name", exp_name,
                "--config_file", cfg_rel,
                "--init_ckpt", INIT_CKPT,
                "--device", str(gpu),
            ],
            cwd=str(PROJECT_ROOT / "ssm_tracker"),
            log=log,
            env=env,
        )
    ckpt = CKPT_ROOT / exp_name / "epoch1.pth"
    return {
        "stage": "train",
        "tag": exp["tag"],
        "gamma": exp["gamma"],
        "radius": exp["radius"],
        "status": "ok" if rc == 0 and ckpt.is_file() else "fail",
        "ckpt": str(ckpt),
        "exp_name": exp_name,
        "cfg_name": cfg_path.stem,
        "log": str(log_file),
        "gpu": gpu,
    }


def eval_one(gpu: str, train_res: dict) -> dict:
    tag = train_res["tag"]
    cfg_name = train_res["cfg_name"]
    # unique cfg copy for unique pred dir
    eval_cfg_name = f"{cfg_name}_ep1_eval"
    src = CFG_DIR / f"{cfg_name}.yaml"
    dst = CFG_DIR / f"{eval_cfg_name}.yaml"
    dst.write_text(src.read_text())
    pred_dir = (
        PROJECT_ROOT
        / "ssm_tracker/track_results"
        / f"femot_dual_{eval_cfg_name}"
        / "test"
    )
    tracker_name = f"MambaTrack_spatial_hyper_{tag}_ep1"
    log_file = LOG_DIR / f"eval_{tag}_ep1_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}/third_party/ByteTrack:"
        f"{PROJECT_ROOT}/yolox_exps/custom:{env.get('PYTHONPATH', '')}"
    )
    with open(log_file, "w") as log:
        log.write(f"GPU={gpu} eval tag={tag} ckpt={train_res['ckpt']}\n")
        rc1 = run_cmd(
            [
                PYTHON, "-u", "track.py",
                "--det_path", str(DET_PATH),
                "--motion_model_path", train_res["ckpt"],
                "--config_file", f"cfgs/{eval_cfg_name}.yaml",
                "--data_root", DATA_ROOT,
                "--device", str(gpu),
            ],
            cwd=str(PROJECT_ROOT / "ssm_tracker"),
            log=log,
            env=env,
        )
        rc2 = 1
        if rc1 == 0:
            rc2 = run_cmd(
                [
                    PYTHON, "-u", "trackeval/run_femot_eval.py",
                    "--split", "test",
                    "--pred_dir", str(pred_dir),
                    "--tracker_name", tracker_name,
                ],
                cwd=str(PROJECT_ROOT),
                log=log,
                env=env,
            )
    return {
        **{k: train_res[k] for k in ("tag", "gamma", "radius", "ckpt", "exp_name")},
        "stage": "eval",
        "status": "ok" if rc1 == 0 and rc2 == 0 else "fail",
        "tracker": tracker_name,
        "log": str(log_file),
        "gpu": gpu,
    }


def read_metrics(tracker: str) -> dict:
    p = TRACKERS / tracker / "pedestrian_summary.txt"
    if not p.is_file():
        return {}
    hdr, val = p.read_text().strip().splitlines()[:2]
    return dict(zip(hdr.split(), val.split()))


def write_csv(rows: list[dict]) -> Path:
    keys = ["HOTA", "IDF1", "MOTA", "AssA", "DetA", "MOTP", "IDSW", "IDs"]
    # include baseline best g05 ep1 for reference
    ref = read_metrics("MambaTrack_assoc_sweep_s1070_nt055_mtl25")
    out_rows = []
    for r in rows:
        m = read_metrics(r["tracker"]) if r.get("tracker") else {}
        out_rows.append(
            {
                "tag": r["tag"],
                "gamma": r["gamma"],
                "radius": r["radius"],
                "tracker": r.get("tracker", ""),
                **{k: m.get(k, "") for k in keys},
            }
        )
    if ref:
        out_rows.append(
            {
                "tag": "g05_ref",
                "gamma": 0.5,
                "radius": 0.10,
                "tracker": "MambaTrack_assoc_sweep_s1070_nt055_mtl25",
                **{k: ref.get(k, "") for k in keys},
            }
        )
    out_rows.sort(
        key=lambda x: float(x["HOTA"]) if x.get("HOTA") else -1.0, reverse=True
    )
    path = ANALYSIS / "ablate_gamma_radius_ep1.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["tag", "gamma", "radius", "tracker"] + keys
        )
        w.writeheader()
        w.writerows(out_rows)
    return path


def pool_map(gpus: list[str], items: list, fn):
    q: Queue = Queue()
    for g in gpus:
        q.put(g)
    results = []

    def _wrap(item):
        gpu = q.get()
        try:
            return fn(gpu, item)
        finally:
            q.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {ex.submit(_wrap, it): it for it in items}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"[{res.get('status')}] stage={res.get('stage')} "
                f"tag={res.get('tag')} gpu={res.get('gpu')}"
            )
    return results


def main():
    gpus = [
        g.strip()
        for g in os.environ.get("ABLATION_GPUS", "2,5,0,1").split(",")
        if g.strip()
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    print(f"Train {len(EXPS)} exps on GPUs={gpus}")
    train_results = pool_map(gpus, EXPS, train_one)
    fails = [r for r in train_results if r["status"] != "ok"]
    if fails:
        print("Train failures:", fails)
        raise SystemExit(1)

    print(f"Eval ep1 on GPUs={gpus}")
    eval_results = pool_map(gpus, train_results, eval_one)
    path = write_csv(eval_results)
    print(f"Wrote {path}")
    print(path.read_text())
    fails = [r for r in eval_results if r["status"] != "ok"]
    print(f"Done. eval_ok={len(eval_results)-len(fails)} fail={len(fails)}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
