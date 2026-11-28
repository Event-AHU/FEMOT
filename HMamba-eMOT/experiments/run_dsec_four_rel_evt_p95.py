#!/usr/bin/env python3
"""Three-epoch scratch run, followed by ep3 tracking and DSEC evaluation.

Retains the existing graph implementation and historical ep30 scratch recipe.
Only the relation set and event threshold differ in the training config.
There are no ep30 validation features; test metrics are exploratory, with no
checkpoint or threshold selection on test. Historical control seed is unknown.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
from run_dsec_pipeline import parse_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    name = "dsec_hyla_ep30_four_rel_evt_p95_seed" + str(args.seed)
    cfg = "cfgs/dsec/dsec_hyla_ep30_four_rel_evt_p95.yaml"
    log_dir = PROJECT / "logs" / "dsec" / name
    ckpt_dir = Path("/wangx_new/lyt/dsec_mamba") / name
    pred_dir = PROJECT / "ssm_tracker/track_results/dsec_hyla_ep30_feats_dsec_hyla_ep30_four_rel_evt_p95/test"
    if log_dir.exists() or ckpt_dir.exists() or pred_dir.exists():
        raise FileExistsError("Experiment output already exists; refusing to overwrite")
    log_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TMPDIR="/wangx_new/lyt/tmp", PYTHONHASHSEED=str(args.seed),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    manifest = {
        "name": name, "gpu": args.gpu, "seed": args.seed,
        "config": str(PROJECT / "ssm_tracker" / cfg),
        "checkpoint_dir": str(ckpt_dir), "pred_dir": str(pred_dir),
        "status": "starting", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initialization": "scratch", "modulation": "delta,C; alpha_B frozen zero",
        "relations": ["spatial", "rgb", "evt", "motion"],
        "event_threshold": 0.9859775304794312,
        "threshold_report": str(PROJECT / "analysis/dsec_ep30_evt_threshold.json"),
        "limitations": ["ep30 val features missing", "historical control seed unknown",
                         "existing hypergraph aggregation implementation retained"],
        "source_sha256": {}, "commands": [],
    }
    for rel in ["ssm_tracker/models/spatial_hypergraph.py", "ssm_tracker/train.py",
                "ssm_tracker/dataset/hypergraph_rules.py", "ssm_tracker/track_utils/tracker.py",
                "ssm_tracker/" + cfg]:
        manifest["source_sha256"][rel] = hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest()

    def save():
        (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(stage, command, cwd):
        manifest["status"] = stage
        manifest["commands"].append({"stage": stage, "argv": command, "cwd": str(cwd)})
        save()
        print(time.strftime("%H:%M:%S"), stage, flush=True)
        with (log_dir / (stage + ".log")).open("w") as handle:
            subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)

    try:
        seed_code = (
            "import random, numpy as np, torch, runpy; "
            f"random.seed({args.seed}); np.random.seed({args.seed}); torch.manual_seed({args.seed}); "
            "runpy.run_path('train.py', run_name='__main__')"
        )
        run("train", [sys.executable, "-u", "-c", seed_code,
                      "--exp_name", name, "--config_file", cfg,
                      "--save_path", str(ckpt_dir.parent), "--device", "0"],
            PROJECT / "ssm_tracker")
        run("track_ep3_test", [sys.executable, "-u", "track.py", "--det_path",
                              "/wangx_new/lyt/dsec_det_results/dsec_hyla_ep30_feats/test",
                              "--motion_model_path", str(ckpt_dir / "epoch3.pth"),
                              "--config_file", cfg, "--device", "0"],
            PROJECT / "ssm_tracker")
        if len(list(pred_dir.glob("*.txt"))) != 12:
            raise RuntimeError("Expected predictions for all 12 DSEC test sequences")
        run("eval_ep3_test", ["flock", str(PROJECT / "logs/dsec/trackeval.lock"),
                             sys.executable, "-u", str(PROJECT / "trackeval/run_dsec_eval.py"),
                             "--split", "test", "--pred_dir", str(pred_dir),
                             "--tracker_name", name], PROJECT)
        manifest["metrics"] = parse_metrics((log_dir / "eval_ep3_test.log").read_text())
        comparisons = {}
        for filename in ["dsec_hyla_ep30_scratch_ep3.json", "dsec_base_on_hyla_ep30.json"]:
            reference = json.loads((PROJECT / "logs/dsec" / filename).read_text())
            comparisons[filename] = {
                key: manifest["metrics"][key] - reference[key]
                for key in manifest["metrics"] if key in reference
            }
        manifest["deltas_vs_historical"] = comparisons
        manifest["status"] = "complete"
        manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        print(json.dumps(manifest["metrics"], indent=2), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
