#!/usr/bin/env python3
"""Estimate a DSEC RGB hyperedge threshold from positive/negative pairs.

Positive pairs are the same target in consecutive annotated frames.  Negative
pairs are different targets visible in the same frame.  Both use the exact
cosine similarity used by ``dataset.hypergraph_rules``.  The report contains
percentiles of both distributions and a small threshold sweep, so a threshold
can be selected without tuning on the test split.
"""

import argparse
import json
import os

import numpy as np


def normalize(x):
    x = x.astype(np.float32, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class Reservoir:
    def __init__(self, size, rng):
        self.size = int(size)
        self.rng = rng
        self.values = np.empty(self.size, dtype=np.float32)
        self.count = 0

    def add(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        for value in values:
            self.count += 1
            if self.count <= self.size:
                self.values[self.count - 1] = value
            else:
                slot = self.rng.integers(0, self.count)
                if slot < self.size:
                    self.values[slot] = value

    def array(self):
        return self.values[: min(self.count, self.size)].copy()


def load_split(folder, max_samples, seed, modality="rgb"):
    rng = np.random.default_rng(seed)
    positive = Reservoir(max_samples, rng)
    negative = Reservoir(max_samples, rng)
    files = 0
    frames = 0
    positive_pairs = 0
    negative_pairs = 0

    for name in sorted(os.listdir(folder)):
        if not name.endswith(".npz"):
            continue
        files += 1
        data = np.load(os.path.join(folder, name))
        frame_ids = data["frame_ids"].astype(np.int32)
        object_ids = data["object_ids"].astype(np.int32)
        rgb = normalize(data[f"{modality}_feats"])

        by_frame = {}
        for row, frame in enumerate(frame_ids):
            by_frame.setdefault(int(frame), []).append(row)
        frame_keys = sorted(by_frame)
        frames += len(frame_keys)

        # Positive: same object at t and t+1.  This matches the temporal
        # feature persistence that an appearance edge should preserve.
        lookup = {
            (int(frame_ids[row]), int(object_ids[row])): row
            for row in range(len(frame_ids))
        }
        for frame in frame_keys:
            next_frame = frame + 1
            if next_frame not in by_frame:
                continue
            ids = set(object_ids[by_frame[frame]].tolist()) & set(
                object_ids[by_frame[next_frame]].tolist()
            )
            for object_id in ids:
                left = lookup[(frame, int(object_id))]
                right = lookup[(next_frame, int(object_id))]
                positive.add(np.dot(rgb[left], rgb[right]))
                positive_pairs += 1

        # Negative: different targets in the same frame.
        for frame in frame_keys:
            rows = by_frame[frame]
            if len(rows) < 2:
                continue
            ids = object_ids[rows]
            order = np.argsort(ids, kind="stable")
            rows = np.asarray(rows, dtype=np.int64)[order]
            sim = rgb[rows] @ rgb[rows].T
            tri = sim[np.triu_indices(len(rows), k=1)]
            negative.add(tri)
            negative_pairs += len(tri)

    return {
        "files": files,
        "frames": frames,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "positive": positive.array(),
        "negative": negative.array(),
    }


def quantiles(values):
    qs = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
    return {f"p{q * 100:g}": float(np.quantile(values, q)) for q in qs}


def evaluate_thresholds(positive, negative, published_threshold=0.8407866):
    # Candidate values include positive percentiles, negative percentiles and
    # the published FEMOT threshold.  Report metrics for threshold >= sim.
    candidates = sorted(
        set(
            [published_threshold]
            + [float(np.quantile(positive, q)) for q in np.linspace(0.01, 0.99, 99)]
            + [float(np.quantile(negative, q)) for q in np.linspace(0.50, 0.999, 100)]
        )
    )
    rows = []
    p_count, n_count = len(positive), len(negative)
    for threshold in candidates:
        tp = int(np.count_nonzero(positive >= threshold))
        fp = int(np.count_nonzero(negative >= threshold))
        recall = tp / max(p_count, 1)
        fpr = fp / max(n_count, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": float(threshold),
                "positive_recall": float(recall),
                "negative_fpr": float(fpr),
                "precision": float(precision),
                "f1": float(f1),
                "youden_j": float(recall - fpr),
            }
        )
    best_f1 = max(rows, key=lambda row: (row["f1"], row["positive_recall"]))
    best_j = max(rows, key=lambda row: (row["youden_j"], row["positive_recall"]))
    return rows, best_f1, best_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_root", default="/wangx_new/lyt/dsec_modal_gt_hyla"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="analysis/dsec_rgb_threshold.json")
    parser.add_argument("--max_samples", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", choices=["rgb", "evt"], default="rgb")
    args = parser.parse_args()

    stats = load_split(
        os.path.join(args.feature_root, args.split), args.max_samples, args.seed,
        args.modality,
    )
    positive, negative = stats.pop("positive"), stats.pop("negative")
    published_threshold = 0.9955849 if args.modality == "evt" else 0.8407866
    rows, best_f1, best_j = evaluate_thresholds(positive, negative, published_threshold)
    report = {
        "feature_root": os.path.abspath(args.feature_root),
        "split": args.split,
        "modality": args.modality,
        "definition": {
            "positive": "same object ID in consecutive frames",
            "negative": "different object IDs in the same frame",
            "decision": f"{args.modality.upper()} cosine similarity >= threshold",
        },
        **stats,
        "sampled_positive": int(len(positive)),
        "sampled_negative": int(len(negative)),
        "positive_percentiles": quantiles(positive),
        "negative_percentiles": quantiles(negative),
        "published_threshold": next(
            row for row in rows if abs(row["threshold"] - published_threshold) < 1e-7
        ),
        "recommended_by_f1": best_f1,
        "recommended_by_youden_j": best_j,
        "threshold_sweep": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "positive_pairs": stats["positive_pairs"],
        "negative_pairs": stats["negative_pairs"],
        "positive_percentiles": report["positive_percentiles"],
        "negative_percentiles": report["negative_percentiles"],
        "published_threshold": report["published_threshold"],
        "recommended_by_f1": best_f1,
        "recommended_by_youden_j": best_j,
        "output": os.path.abspath(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
