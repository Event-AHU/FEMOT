#!/usr/bin/env python3
"""Qualitative hypergraph lines and dark3 HyLA residual maps.

Hypergraph vertices are the final tracker's boxes on one frame. RGB edges use
the HyLA-enhanced dark3 descriptor of the detection matched to each track.
The heatmap is the channel norm of H(J, E) at dark3, the event-conditioned
residual inside the RGB stream, drawn back on the original frame.
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "ssm_tracker"))

from configs.dataset_paths import FEMOT_FRAME_FMT, FEMOT_ROOT  # noqa: E402
from dataset.hypergraph_rules import _components_from_similarity  # noqa: E402

DET_DIR = os.path.join(
    PROJECT, "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test"
)
TRACK_DIR = os.path.join(
    PROJECT,
    "ssm_tracker/track_results/"
    "femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats_tune5376_lost_20/test",
)
SPATIAL_RADIUS = 0.10
RGB_THRESHOLD = 0.841
RGB_TOPK = 5
SPATIAL_COLOR = (255, 190, 40)
RGB_COLOR = (40, 120, 255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqs",
        nargs="+",
        default=[
            "dvSave-2024_06_17_18_12_17",
            "dvSave-2024_06_26_18_22_35",
            "dvSave-2024_06_11_11_25_12",
        ],
    )
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--gap", type=int, default=40)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output_dir", default=os.path.join(PROJECT, "docs/vis_qual"))
    parser.add_argument("--hyla", action="store_true")
    parser.add_argument("--device", default="7")
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="explicit seq:frame items, for example dvSave-..._18_12_17:833",
    )
    return parser.parse_args()


def load_rows(path):
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows


def box_iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-6)


def unique_groups(groups):
    seen, out = set(), []
    for group in groups:
        key = tuple(sorted(int(index) for index in group))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def spatial_groups(centers):
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    groups = []
    for index in range(len(centers)):
        members = np.flatnonzero(distance[index] <= SPATIAL_RADIUS)
        if len(members) >= 2:
            groups.append(members)
    return unique_groups(groups)


def match_features(tracks, det_boxes, det_feats):
    features = np.zeros((len(tracks), det_feats.shape[1]), dtype=np.float32)
    matched = np.zeros(len(tracks), dtype=bool)
    for index, box in enumerate(tracks):
        if len(det_boxes) == 0:
            break
        ious = np.array([box_iou(box, det) for det in det_boxes])
        best = int(ious.argmax())
        if ious[best] >= 0.3:
            features[index] = det_feats[best]
            matched[index] = True
    return features, matched


def score_frame(tracks, features, matched):
    n = len(tracks)
    if n < 4 or n > 12 or matched.sum() < 3:
        return None
    height, width = 1.0, 1.0
    centers = np.stack(
        [
            (tracks[:, 0] + tracks[:, 2] / 2) / width,
            (tracks[:, 1] + tracks[:, 3] / 2) / height,
        ],
        axis=1,
    )
    # Caller passes centers already normalized. Here tracks are tlwh and the
    # image size is applied outside.
    return n, centers


def candidates(seq, top, gap):
    track_rows = load_rows(os.path.join(TRACK_DIR, seq + ".txt"))
    det_rows = load_rows(os.path.join(DET_DIR, seq + ".txt"))
    rgb_npz = np.load(os.path.join(DET_DIR, seq + "_rgb_feat.npz"))
    sample = cv2.imread(frame_path(seq, int(track_rows[0, 0])))
    if sample is None:
        raise FileNotFoundError(seq)
    image_h, image_w = sample.shape[:2]
    found = []
    for frame_id in sorted(np.unique(track_rows[:, 0].astype(np.int64))):
        tracks = track_rows[track_rows[:, 0].astype(np.int64) == frame_id][:, 2:6]
        dets = det_rows[det_rows[:, 0].astype(np.int64) == frame_id]
        key = f"f{frame_id}"
        if key not in rgb_npz or len(tracks) < 4 or len(tracks) > 12:
            continue
        feats = rgb_npz[key].astype(np.float32)
        if len(feats) != len(dets):
            continue
        features, matched = match_features(tracks, dets[:, 2:6], feats)
        if matched.sum() < 3:
            continue
        centers = np.stack(
            [
                (tracks[:, 0] + tracks[:, 2] / 2) / image_w,
                (tracks[:, 1] + tracks[:, 3] / 2) / image_h,
            ],
            axis=1,
        ).astype(np.float32)
        spatial = [group for group in spatial_groups(centers) if len(group) <= 6]
        rgb_index = np.flatnonzero(matched)
        rgb = _components_from_similarity(features[rgb_index], RGB_THRESHOLD, RGB_TOPK)
        rgb = [tuple(int(rgb_index[i]) for i in group) for group in rgb if len(group) <= 5]
        if not spatial or not rgb:
            continue
        areas = tracks[:, 2] * tracks[:, 3]
        if np.median(areas) < 80:
            continue
        score = 3 * len(rgb) + 2 * len(spatial) - 0.3 * max(0, len(tracks) - 7)
        found.append((score, int(frame_id), len(tracks), len(spatial), len(rgb)))
    found.sort(key=lambda item: (-item[0], item[1]))
    picked = []
    for item in found:
        if all(abs(item[1] - prev[1]) >= gap for prev in picked):
            picked.append(item)
        if len(picked) >= top:
            break
    return picked


def frame_path(seq, frame_id):
    name = FEMOT_FRAME_FMT.format(frame_id=frame_id)
    return os.path.join(FEMOT_ROOT, "test", seq, seq + "_aps", name)


def event_path(seq, frame_id):
    name = FEMOT_FRAME_FMT.format(frame_id=frame_id)
    return os.path.join(FEMOT_ROOT, "test", seq, seq + "_dvs", name)


def id_color(track_id):
    hue = int((int(track_id) * 0.61803398875 % 1.0) * 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def collapse_groups(groups, centers_px, min_dist=28):
    """Drop hyperedges whose centroids sit on top of an already drawn group."""
    ordered = sorted(groups, key=lambda group: (len(group), group))
    kept, centroids = [], []
    for group in ordered:
        if not 2 <= len(group) <= 4:
            continue
        centroid = centers_px[list(group)].mean(axis=0)
        if any(np.linalg.norm(centroid - prev) < min_dist for prev in centroids):
            continue
        kept.append(group)
        centroids.append(centroid)
    return kept


def draw_star(image, points, color, scale):
    if len(points) < 2:
        return
    centroid = np.mean(points, axis=0)
    cx, cy = int(round(centroid[0] * scale)), int(round(centroid[1] * scale))
    for point in points:
        x, y = int(round(point[0] * scale)), int(round(point[1] * scale))
        cv2.line(image, (x, y), (cx, cy), color, 2, cv2.LINE_AA)


def draw_box(image, tlwh, color, label, scale):
    x, y, w, h = [int(round(float(value) * scale)) for value in tlwh]
    cv2.rectangle(image, (x, y), (x + max(w, 1), y + max(h, 1)), color, 2, cv2.LINE_AA)
    if not label:
        return
    cv2.putText(
        image, label, (x, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def render_hyper(seq, frame_id, scale):
    track_rows = load_rows(os.path.join(TRACK_DIR, seq + ".txt"))
    det_rows = load_rows(os.path.join(DET_DIR, seq + ".txt"))
    rgb_npz = np.load(os.path.join(DET_DIR, seq + "_rgb_feat.npz"))
    frame_tracks = track_rows[track_rows[:, 0].astype(np.int64) == frame_id]
    dets = det_rows[det_rows[:, 0].astype(np.int64) == frame_id]
    tracks = frame_tracks[:, 2:6]
    ids = frame_tracks[:, 1].astype(np.int64)
    feats = rgb_npz[f"f{frame_id}"].astype(np.float32)
    rgb = cv2.imread(frame_path(seq, frame_id))
    image_h, image_w = rgb.shape[:2]
    features, matched = match_features(tracks, dets[:, 2:6], feats)
    centers_px = np.stack(
        [tracks[:, 0] + tracks[:, 2] / 2, tracks[:, 1] + tracks[:, 3] / 2], axis=1
    )
    centers = np.stack(
        [centers_px[:, 0] / image_w, centers_px[:, 1] / image_h], axis=1
    ).astype(np.float32)
    spatial = collapse_groups(spatial_groups(centers), centers_px)
    rgb_index = np.flatnonzero(matched)
    rgb_groups = []
    if len(rgb_index) >= 2:
        raw = _components_from_similarity(features[rgb_index], RGB_THRESHOLD, RGB_TOPK)
        rgb_groups = collapse_groups(
            [tuple(int(rgb_index[i]) for i in group) for group in raw],
            centers_px,
        )
    canvas = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for group in spatial:
        draw_star(canvas, centers_px[list(group)], SPATIAL_COLOR, scale)
    for group in rgb_groups:
        draw_star(canvas, centers_px[list(group)], RGB_COLOR, scale)
    for track_id, box in zip(ids, tracks):
        draw_box(canvas, box, id_color(track_id), str(int(track_id)), scale)
    return canvas, len(spatial), len(rgb_groups), int(matched.sum()), len(tracks)


def hyla_delta_map(model, rgb, event, device):
    import torch
    from dual_data import dual_preproc

    sys.path.insert(0, os.path.join(PROJECT, "third_party", "ByteTrack"))
    sys.path.insert(0, os.path.join(PROJECT, "yolox_exps", "custom"))
    image, ratio = dual_preproc(
        rgb, event, (416, 416),
        rgb_means=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225),
    )
    tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
    block = model.backbone.fusion_blocks["dark3"]
    with torch.no_grad():
        backbone_rgb, backbone_evt = model.backbone.forward_modal_features(tensor)
        aligned_rgb = block.shared_mlp(backbone_rgb["dark3"])
        aligned_evt = block.shared_mlp(backbone_evt["dark3"])
        joint = torch.cat((aligned_rgb, aligned_evt), dim=1)
        delta = block.hyla.rgb_from_evt(joint, aligned_evt)
        heat = delta.float().pow(2).mean(dim=1).sqrt()[0].cpu().numpy()
    canvas = cv2.resize(heat, (416, 416), interpolation=cv2.INTER_CUBIC)
    crop_h = int(round(rgb.shape[0] * ratio))
    crop_w = int(round(rgb.shape[1] * ratio))
    canvas = canvas[:crop_h, :crop_w]
    canvas = cv2.resize(canvas, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return canvas


def colorize(heat, rgb, scale):
    low, high = np.percentile(heat, [2, 98])
    norm = np.clip((heat - low) / max(high - low, 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blend = cv2.addWeighted(rgb, 0.45, color, 0.55, 0)
    return cv2.resize(blend, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def load_hyla_model(device_index):
    import torch
    sys.path.insert(0, os.path.join(PROJECT, "third_party", "ByteTrack"))
    sys.path.insert(0, os.path.join(PROJECT, "yolox_exps", "custom"))
    from yolox.exp import get_exp

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp = get_exp(
        os.path.join(PROJECT, "yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py"),
        "femot_dual",
    )
    model = exp.get_model().to(device).eval()
    ckpt = os.path.join(
        PROJECT,
        "third_party/ByteTrack/YOLOX_outputs/"
        "yolox_femot_dual_hyla_shared_mlp_coco/best_ckpt_ep27_ap57.5_freeze.pth.tar",
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return model, device


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plan = []
    if args.pairs:
        args.seqs = []
        for item in args.pairs:
            seq, frame_text = item.rsplit(":", 1)
            plan.append((seq, int(frame_text)))
            print(f"pair {seq} frame {frame_text}")
    for seq in args.seqs:
        if args.frames:
            picked = [(0, frame_id, 0, 0, 0) for frame_id in args.frames]
        else:
            picked = candidates(seq, args.top, args.gap)
        print(seq)
        for item in picked:
            print(
                f"  frame {item[1]} score {item[0]:.2f} "
                f"tracks {item[2]} spatial {item[3]} rgb {item[4]}"
            )
            plan.append((seq, item[1]))

    model = device = None
    if args.hyla:
        model, device = load_hyla_model(args.device)

    for seq, frame_id in plan:
        hyper, n_spatial, n_rgb, n_matched, n_tracks = render_hyper(
            seq, frame_id, args.scale
        )
        seq_dir = os.path.join(args.output_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)
        hyper_path = os.path.join(seq_dir, f"hyper_{frame_id:05d}.jpg")
        cv2.imwrite(hyper_path, hyper, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(
            f"{hyper_path} spatial {n_spatial} rgb {n_rgb} "
            f"matched {n_matched}/{n_tracks}"
        )
        if model is None:
            continue
        rgb = cv2.imread(frame_path(seq, frame_id))
        event = cv2.imread(event_path(seq, frame_id))
        heat = hyla_delta_map(model, rgb, event, device)
        heat_path = os.path.join(seq_dir, f"hyla_{frame_id:05d}.jpg")
        cv2.imwrite(heat_path, colorize(heat, rgb, args.scale), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(heat_path)


if __name__ == "__main__":
    main()
