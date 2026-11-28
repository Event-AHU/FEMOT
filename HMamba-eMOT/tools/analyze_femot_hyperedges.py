#!/usr/bin/env python3
"""Validate whether rule-built FEMOT hyperedges carry useful motion priors."""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", default="data/femot_modal_gt")
    parser.add_argument("--output", default="analysis/femot_hyperedge_feasibility.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min_coverage", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def normalized_motion(prev_box, box):
    prev_xywh = prev_box.copy()
    box_xywh = box.copy()
    prev_center = prev_xywh[:2] + 0.5 * prev_xywh[2:]
    center = box_xywh[:2] + 0.5 * box_xywh[2:]
    return np.asarray(
        [
            (center[0] - prev_center[0]) / max(prev_xywh[2], 1.0),
            (center[1] - prev_center[1]) / max(prev_xywh[3], 1.0),
            np.log(max(box_xywh[2], 1.0) / max(prev_xywh[2], 1.0)),
            np.log(max(box_xywh[3], 1.0) / max(prev_xywh[3], 1.0)),
        ],
        dtype=np.float32,
    )


def load_records(folder):
    records = []
    all_pair_sims = {"rgb": [], "evt": []}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".npz"):
            continue
        data = np.load(os.path.join(folder, name))
        frame_ids = data["frame_ids"]
        object_ids = data["object_ids"]
        boxes = data["boxes_tlwh"].astype(np.float32)
        rgb = l2_normalize(data["rgb_feats"].astype(np.float32))
        evt = l2_normalize(data["evt_feats"].astype(np.float32))
        image_hw = data["image_hw"].astype(np.float32)

        by_frame = {}
        for frame_id in np.unique(frame_ids):
            idx = np.flatnonzero(frame_ids == frame_id)
            by_frame[int(frame_id)] = {
                int(object_ids[k]): (boxes[k], rgb[k], evt[k]) for k in idx
            }

        for frame_id in sorted(by_frame):
            if frame_id - 1 not in by_frame or frame_id + 1 not in by_frame:
                continue
            prev, cur, nxt = by_frame[frame_id - 1], by_frame[frame_id], by_frame[frame_id + 1]
            ids = sorted(set(prev) & set(cur) & set(nxt))
            if len(ids) < 2:
                continue
            cur_boxes = np.stack([cur[i][0] for i in ids])
            cur_rgb = np.stack([cur[i][1] for i in ids])
            cur_evt = np.stack([cur[i][2] for i in ids])
            cur_motion = np.stack([normalized_motion(prev[i][0], cur[i][0]) for i in ids])
            next_motion = np.stack([normalized_motion(cur[i][0], nxt[i][0]) for i in ids])
            centers = cur_boxes[:, :2] + 0.5 * cur_boxes[:, 2:]
            centers[:, 0] /= image_hw[1]
            centers[:, 1] /= image_hw[0]

            for relation, feat in (("rgb", cur_rgb), ("evt", cur_evt)):
                sim = feat @ feat.T
                tri = sim[np.triu_indices(len(ids), k=1)]
                if len(tri):
                    all_pair_sims[relation].append(tri)
            records.append(
                {
                    "centers": centers,
                    "rgb": cur_rgb,
                    "evt": cur_evt,
                    "motion": cur_motion,
                    "next_motion": next_motion,
                }
            )
    pair_sims = {
        key: np.concatenate(value) if value else np.zeros(0)
        for key, value in all_pair_sims.items()
    }
    return records, pair_sims


def connected_groups(sim, threshold, topk):
    n = sim.shape[0]
    if n < 2:
        return []
    k = min(topk, n - 1)
    work = sim.copy()
    np.fill_diagonal(work, -np.inf)
    top = np.argpartition(work, -k, axis=1)[:, -k:]
    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[np.arange(n)[:, None], top] = True
    adjacency = (sim >= threshold) & top_mask & top_mask.T
    np.fill_diagonal(adjacency, True)
    visited = np.zeros(n, dtype=bool)
    groups = []
    for start in range(n):
        if visited[start]:
            continue
        stack, component = [start], []
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in np.flatnonzero(adjacency[node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        if len(component) >= 2:
            groups.append(np.asarray(component, dtype=np.int64))
    return groups


def spatial_groups(record, radius=0.1):
    centers = record["centers"]
    dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    groups = []
    for i in range(len(centers)):
        members = np.flatnonzero(dist[i] <= radius)
        if len(members) >= 2:
            groups.append(members)
    return groups


def motion_groups(record):
    speed = np.linalg.norm(record["motion"][:, :2], axis=1)
    if len(speed) < 3:
        return [np.arange(len(speed))] if len(speed) >= 2 else []
    boundaries = np.quantile(speed, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(speed, boundaries)
    return [np.flatnonzero(labels == k) for k in range(3) if np.sum(labels == k) >= 2]


def relation_groups(record, relation, threshold=None, topk=5):
    if relation in ("rgb", "evt"):
        feat = record[relation]
        return connected_groups(feat @ feat.T, threshold, topk)
    if relation == "spatial":
        return spatial_groups(record)
    if relation == "motion":
        return motion_groups(record)
    raise ValueError(relation)


def evaluate(records, relation, threshold, topk, rng):
    pair_dist, random_pair_dist = [], []
    loo_error, random_loo_error, self_error = [], [], []
    covered, vertices, edge_count, edge_sizes = 0, 0, 0, []

    for record in records:
        groups = relation_groups(record, relation, threshold, topk)
        n = len(record["motion"])
        vertices += n
        member_union = set()
        for group in groups:
            edge_count += 1
            edge_sizes.append(len(group))
            member_union.update(group.tolist())
            next_motion = record["next_motion"][group]
            if len(group) >= 2:
                diff = next_motion[:, None] - next_motion[None, :]
                tri = np.linalg.norm(diff, axis=2)[np.triu_indices(len(group), k=1)]
                pair_dist.extend(tri.tolist())
                random_group = rng.choice(n, size=len(group), replace=False)
                random_next = record["next_motion"][random_group]
                random_diff = random_next[:, None] - random_next[None, :]
                random_tri = np.linalg.norm(random_diff, axis=2)[
                    np.triu_indices(len(group), k=1)
                ]
                random_pair_dist.extend(random_tri.tolist())

            for index in group:
                peers = group[group != index]
                if len(peers) == 0:
                    continue
                pred = record["motion"][peers].mean(axis=0)
                target = record["next_motion"][index]
                loo_error.append(np.mean(np.abs(pred - target)))
                candidates = np.setdiff1d(np.arange(n), np.asarray([index]))
                random_peers = rng.choice(
                    candidates, size=min(len(peers), len(candidates)), replace=False
                )
                random_pred = record["motion"][random_peers].mean(axis=0)
                random_loo_error.append(np.mean(np.abs(random_pred - target)))
                self_error.append(np.mean(np.abs(record["motion"][index] - target)))
        covered += len(member_union)

    def mean(values):
        return float(np.mean(values)) if values else None

    def paired_improvement(random_values, relation_values):
        if not relation_values:
            return None
        delta = np.asarray(random_values) - np.asarray(relation_values)
        standard_error = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else 0.0
        return {
            "mean": float(delta.mean()),
            "ci95": [
                float(delta.mean() - 1.96 * standard_error),
                float(delta.mean() + 1.96 * standard_error),
            ],
            "win_rate": float(np.mean(delta > 0)),
            "samples": int(len(delta)),
        }

    return {
        "threshold": threshold,
        "frames": len(records),
        "vertices": vertices,
        "edge_count": edge_count,
        "mean_edge_size": mean(edge_sizes),
        "coverage": covered / max(vertices, 1),
        "intra_next_motion_distance": mean(pair_dist),
        "random_next_motion_distance": mean(random_pair_dist),
        "loo_mae": mean(loo_error),
        "random_loo_mae": mean(random_loo_error),
        "loo_improvement_over_random": paired_improvement(
            random_loo_error, loo_error
        ),
        "self_velocity_mae": mean(self_error),
    }


def choose_threshold(records, pair_sims, relation, topk, min_coverage, rng):
    quantiles = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
    thresholds = sorted(set(float(np.quantile(pair_sims, q)) for q in quantiles))
    candidates = [
        evaluate(records, relation, threshold, topk, rng) for threshold in thresholds
    ]
    valid = [
        item for item in candidates
        if item["coverage"] >= min_coverage and item["edge_count"] > 0
        and item["loo_mae"] is not None
    ]
    if not valid:
        valid = [item for item in candidates if item["edge_count"] > 0]
    best = min(
        valid,
        key=lambda item: (
            item["loo_mae"] / max(item["random_loo_mae"], 1e-8),
            item["intra_next_motion_distance"]
            / max(item["random_next_motion_distance"], 1e-8),
        ),
    )
    return best["threshold"], candidates


def main(args):
    rng = np.random.default_rng(args.seed)
    train_records, train_sims = load_records(os.path.join(args.feature_root, "train"))
    val_records, _ = load_records(os.path.join(args.feature_root, "val"))
    report = {"train_frames": len(train_records), "val_frames": len(val_records), "relations": {}}

    for relation in ("rgb", "evt"):
        threshold, sweep = choose_threshold(
            train_records,
            train_sims[relation],
            relation,
            args.topk,
            args.min_coverage,
            rng,
        )
        report["relations"][relation] = {
            "selected_threshold": threshold,
            "train_sweep": sweep,
            "val": evaluate(val_records, relation, threshold, args.topk, rng),
        }
        logger.info(f"{relation}: selected threshold {threshold:.4f}")

    for relation in ("spatial", "motion"):
        report["relations"][relation] = {
            "val": evaluate(val_records, relation, None, args.topk, rng)
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(report, file, indent=2)
    logger.info(f"saved report to {args.output}")


if __name__ == "__main__":
    main(parse_args())
