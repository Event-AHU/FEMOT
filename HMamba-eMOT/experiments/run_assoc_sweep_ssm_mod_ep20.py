#!/usr/bin/env python3
"""Association hyperparam sweep on spatial_ssm_mod epoch20 (IoU match)."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from queue import Queue

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get(
    "PYTHON", "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
)
CKPT = (
    PROJECT_ROOT
    / "ssm_tracker/saved_ckpts/mambatrack_femot_spatial_ssm_mod/epoch20.pth"
)
BASE_CFG = (
    PROJECT_ROOT / "ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod.yaml"
)
DET_PATH = PROJECT_ROOT / "det_results/femot_dual/test"
DATA_ROOT = (
    "/rydata/wangx_dataset/_dataset/FEMOT/{split}/{seq}/{seq}_aps/{frame_id:05d}.jpg"
)
LOG_DIR = PROJECT_ROOT / "logs"
CFG_DIR = PROJECT_ROOT / "ssm_tracker/cfgs"
TRACKERS = PROJECT_ROOT / "trackeval/trackers/test"
OUT_CSV = PROJECT_ROOT / "analysis/assoc_sweep_ssm_mod_ep20.csv"

STAGE1 = [0.65, 0.70, 0.75]
NEW_TRACK = [0.55, 0.60, 0.65]
MAX_LOST = [25, 30, 40]


def tag(s1: float, nt: float, mtl: int) -> str:
    return f"s1{s1:.2f}_nt{nt:.2f}_mtl{mtl}".replace(".", "")


def write_cfg(s1: float, nt: float, mtl: int) -> tuple[str, Path, str, Path]:
    name = f"MambaTrack_femot_spatial_ssm_mod_ep20_assoc_{tag(s1, nt, mtl)}"
    cfg_path = CFG_DIR / f"{name}.yaml"
    with open(BASE_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg["inference"]["match_mode"] = "iou"
    cfg["inference"]["filter_thresh"] = 0.2
    cfg["inference"]["match_thresh_stage1"] = float(s1)
    cfg["inference"]["match_thresh_stage2"] = 0.9
    cfg["inference"]["new_track_thresh"] = float(nt)
    cfg["inference"]["max_time_lost"] = int(mtl)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    pred_dir = (
        PROJECT_ROOT / "ssm_tracker/track_results" / f"femot_dual_{name}" / "test"
    )
    tracker_name = f"MambaTrack_ssm_mod_ep20_assoc_{tag(s1, nt, mtl)}"
    return name, cfg_path, tracker_name, pred_dir


def already_done(tracker_name: str) -> bool:
    return (TRACKERS / tracker_name / "pedestrian_summary.txt").is_file()


def run_one(gpu: str, s1: float, nt: float, mtl: int) -> dict:
    name, cfg_path, tracker_name, pred_dir = write_cfg(s1, nt, mtl)
    log_file = (
        LOG_DIR
        / f"assoc_ssm_mod_ep20_{tag(s1, nt, mtl)}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}/third_party/ByteTrack:"
        f"{PROJECT_ROOT}/yolox_exps/custom:"
        f"{env.get('PYTHONPATH', '')}"
    )
    # Pin physical GPU before torch import; visible device is always 0.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if already_done(tracker_name):
        return {
            "status": "skip",
            "tracker": tracker_name,
            "s1": s1,
            "nt": nt,
            "mtl": mtl,
        }

    cmd_track = [
        PYTHON, "-u", "track.py",
        "--det_path", str(DET_PATH),
        "--motion_model_path", str(CKPT),
        "--config_file", f"cfgs/{name}.yaml",
        "--data_root", DATA_ROOT,
        "--device", "0",
    ]
    cmd_eval = [
        PYTHON, "-u", "trackeval/run_femot_eval.py",
        "--split", "test",
        "--pred_dir", str(pred_dir),
        "--tracker_name", tracker_name,
    ]
    with open(log_file, "w") as log:
        log.write(f"GPU={gpu} s1={s1} nt={nt} mtl={mtl}\nCKPT={CKPT}\n")
        log.flush()
        p1 = subprocess.run(
            cmd_track,
            cwd=str(PROJECT_ROOT / "ssm_tracker"),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if p1.returncode != 0:
            return {
                "status": "fail_track",
                "tracker": tracker_name,
                "s1": s1,
                "nt": nt,
                "mtl": mtl,
                "log": str(log_file),
            }
        p2 = subprocess.run(
            cmd_eval,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if p2.returncode != 0:
            return {
                "status": "fail_eval",
                "tracker": tracker_name,
                "s1": s1,
                "nt": nt,
                "mtl": mtl,
                "log": str(log_file),
            }
    return {
        "status": "ok",
        "tracker": tracker_name,
        "s1": s1,
        "nt": nt,
        "mtl": mtl,
        "log": str(log_file),
    }


def read_metrics(tracker_name: str) -> dict:
    path = TRACKERS / tracker_name / "pedestrian_summary.txt"
    if not path.is_file():
        return {}
    hdr, val = path.read_text().strip().splitlines()[:2]
    return dict(zip(hdr.split(), val.split()))


def collect_csv(out_csv: Path) -> None:
    keys = ["HOTA", "IDF1", "MOTA", "AssA", "DetA", "MOTP", "IDSW", "IDs"]
    rows = []
    for s1, nt, mtl in product(STAGE1, NEW_TRACK, MAX_LOST):
        tracker = f"MambaTrack_ssm_mod_ep20_assoc_{tag(s1, nt, mtl)}"
        m = read_metrics(tracker)
        row = {"s1": s1, "nt": nt, "mtl": mtl, "tracker": tracker}
        for k in keys:
            row[k] = m.get(k, "")
        rows.append(row)
    # default eval of ssm_mod ep20 (s1=0.70 nt=0.55 mtl=25)
    ref = read_metrics("MambaTrack_spatial_ssm_mod_ep20")
    if ref:
        rows.append(
            {
                "s1": 0.70,
                "nt": 0.55,
                "mtl": 25,
                "tracker": "MambaTrack_spatial_ssm_mod_ep20(ref)",
                **{k: ref.get(k, "") for k in keys},
            }
        )
    rows.sort(
        key=lambda r: float(r["HOTA"]) if r.get("HOTA") else -1.0, reverse=True
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["s1", "nt", "mtl", "tracker"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")
    print("Top by HOTA:")
    for r in rows[:10]:
        if not r.get("HOTA"):
            continue
        print(
            f"  s1={r['s1']} nt={r['nt']} mtl={r['mtl']} "
            f"HOTA={r['HOTA']} IDF1={r['IDF1']} AssA={r['AssA']} "
            f"IDSW={r['IDSW']} IDs={r['IDs']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        default=os.environ.get("ASSOC_GPUS", "1,3,0"),
        help="Comma-separated GPU ids for concurrent workers",
    )
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        collect_csv(OUT_CSV)
        return

    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    combos = list(product(STAGE1, NEW_TRACK, MAX_LOST))
    print(f"Combos={len(combos)} GPUs={gpus} CKPT={CKPT}")

    gpu_q: Queue = Queue()
    for g in gpus:
        gpu_q.put(g)

    def _worker(s1: float, nt: float, mtl: int) -> dict:
        gpu = gpu_q.get()
        try:
            res = run_one(gpu, s1, nt, mtl)
            res["gpu"] = gpu
            return res
        finally:
            gpu_q.put(gpu)

    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {
            ex.submit(_worker, s1, nt, mtl): (s1, nt, mtl)
            for s1, nt, mtl in combos
        }
        for fut in as_completed(futs):
            s1, nt, mtl = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "status": "exception",
                    "s1": s1,
                    "nt": nt,
                    "mtl": mtl,
                    "error": str(e),
                }
            results.append(res)
            print(
                f"[{res.get('status')}] s1={s1} nt={nt} mtl={mtl} "
                f"gpu={res.get('gpu')} tracker={res.get('tracker')}"
            )

    collect_csv(OUT_CSV)
    fails = [r for r in results if r.get("status") not in ("ok", "skip")]
    print(f"Done. ok/skip={len(results) - len(fails)} fail={len(fails)}")
    if fails:
        for f in fails:
            print(" FAIL", f)


if __name__ == "__main__":
    main()
