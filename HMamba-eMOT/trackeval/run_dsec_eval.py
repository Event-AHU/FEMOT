#!/usr/bin/env python3
"""Run TrackEval on DSEC. Categories stay collapsed into one class."""

import argparse
import os
import sys
from multiprocessing import freeze_support

import numpy as np

if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACK_EVAL_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "TrackEval"))
DSEC_ROOT = "/wangx_new/lyt/dsec_mot"

sys.path.insert(0, TRACK_EVAL_ROOT)
import trackeval  # noqa: E402


def prepare_tracker_layout(pred_dir, split, tracker_name):
    tracker_root = os.path.join(os.path.dirname(__file__), "trackers", "dsec_" + split)
    data_dir = os.path.join(tracker_root, tracker_name, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
    skip = {"meta_data.txt", "metrics_summary.txt"}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".txt") or name in skip:
            continue
        src = os.path.abspath(os.path.join(pred_dir, name))
        dst = os.path.join(data_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    return tracker_root


def main():
    parser = argparse.ArgumentParser(description="Evaluate DSEC results with TrackEval")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--tracker_name", type=str, default="MambaTrack")
    parser.add_argument("--metrics", nargs="+", default=["HOTA", "CLEAR", "Identity"])
    args = parser.parse_args()

    gt_folder = os.path.join(DSEC_ROOT, args.split)
    seqmap = os.path.join(gt_folder, "seqmaps", f"DSEC-{args.split}.txt")
    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(args.pred_dir)
    if not os.path.isfile(seqmap):
        raise FileNotFoundError(seqmap)

    tracker_root = prepare_tracker_layout(args.pred_dir, args.split, args.tracker_name)
    print(f"GT:      {gt_folder}")
    print(f"Pred:    {args.pred_dir}")
    print(f"Seqmap:  {seqmap}")
    print(f"Tracker: {args.tracker_name}")

    dataset_config = {
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": tracker_root,
        "TRACKERS_TO_EVAL": [args.tracker_name],
        "BENCHMARK": "DSEC",
        "SPLIT_TO_EVAL": args.split,
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False,
        "PLOT_CURVES": False,
        "PRINT_CONFIG": True,
    }
    metrics_config = {"METRICS": args.metrics, "THRESHOLD": 0.5}
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["PLOT_CURVES"] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric_cls in (
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,
    ):
        if metric_cls.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric_cls(metrics_config))
    if not metrics_list:
        raise ValueError("No valid metrics selected")
    evaluator.evaluate(dataset_list, metrics_list)
    print(f"Done. Results: {tracker_root}/{args.tracker_name}/")


if __name__ == "__main__":
    freeze_support()
    main()
