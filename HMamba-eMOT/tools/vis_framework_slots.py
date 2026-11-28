"""Render FEMOT frames as the four slots of the framework figure.

Each requested frame produces a raw RGB image, a raw event image, YOLO boxes
with score above the tracker filter, and track boxes colored by identity.
The same track id keeps the same color across frames. Ground truth is not drawn.

Example:
  python tools/vis_framework_slots.py --frames 792 802 812
"""

import argparse
import os
import sys

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
from configs.dataset_paths import (
    FEMOT_EVT_DIR_SUFFIX,
    FEMOT_FRAME_FMT,
    FEMOT_RGB_DIR_SUFFIX,
    FEMOT_ROOT,
)

SEQ = "dvSave-2024_06_17_18_12_17"
DET_DIR = os.path.join(
    _PROJECT_ROOT,
    "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
)
TRACK_DIR = os.path.join(
    _PROJECT_ROOT,
    "ssm_tracker/track_results/"
    "femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats_tune5376_lost_20/test",
)
SCORE_THRESH = 0.25
DET_COLOR = (40, 200, 40)


def parse_args():
    parser = argparse.ArgumentParser(description="Render framework-figure image slots")
    parser.add_argument("--seq", default=SEQ)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[792, 802, 812],
        help="frame ids, in temporal order",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--score_thresh", type=float, default=SCORE_THRESH)
    parser.add_argument(
        "--output_dir",
        default="",
        help="defaults to docs/framework_real/{seq}_{frames}",
    )
    return parser.parse_args()


def id_color(track_id):
    hue = int((track_id * 0.61803398875 % 1.0) * 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


def load_rows(path):
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows


def draw_box(img, tlwh, color, label, scale):
    x, y, w, h = [int(round(float(v) * scale)) for v in tlwh]
    x2, y2 = x + max(w, 1), y + max(h, 1)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2, cv2.LINE_AA)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(label, font, text_scale, thick)
    tx = min(max(x, 0), img.shape[1] - tw - 8)
    ty = y - 4 if y - th - 6 >= 0 else min(y + th + 6, img.shape[0] - 4)
    cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 6, ty + base - 1), color, -1)
    cv2.putText(img, label, (tx + 3, ty), font, text_scale, (255, 255, 255), thick, cv2.LINE_AA)


def render_frame(seq, frame_id, det_rows, track_rows, scale, score_thresh):
    frame_name = FEMOT_FRAME_FMT.format(frame_id=frame_id)
    seq_root = os.path.join(FEMOT_ROOT, "test", seq)
    rgb = cv2.imread(os.path.join(seq_root, seq + FEMOT_RGB_DIR_SUFFIX, frame_name))
    event = cv2.imread(os.path.join(seq_root, seq + FEMOT_EVT_DIR_SUFFIX, frame_name))
    if rgb is None or event is None:
        raise FileNotFoundError(frame_name)

    dets = [row[2:6] for row in det_rows if int(row[0]) == frame_id and float(row[6]) > score_thresh]
    tracks = [(int(row[1]), row[2:6]) for row in track_rows if int(row[0]) == frame_id]
    tracks.sort(key=lambda item: item[0])

    rgb_up = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    event_up = cv2.resize(event, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    det_img = rgb_up.copy()
    track_img = rgb_up.copy()
    for box in dets:
        draw_box(det_img, box, DET_COLOR, "", scale)
    for track_id, box in tracks:
        draw_box(track_img, box, id_color(track_id), str(track_id), scale)
    return {
        "rgb": rgb_up,
        "event": event_up,
        "det": det_img,
        "tracks": track_img,
    }, [track_id for track_id, _ in tracks]


def main():
    args = parse_args()
    det_rows = load_rows(os.path.join(DET_DIR, args.seq + ".txt"))
    track_rows = load_rows(os.path.join(TRACK_DIR, args.seq + ".txt"))
    tag = "_".join(f"{frame_id:05d}" for frame_id in args.frames)
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "docs/framework_real", f"{args.seq}_{tag}"
    )
    os.makedirs(output_dir, exist_ok=True)

    strips = {key: [] for key in ("rgb", "event", "det", "tracks")}
    for frame_id in args.frames:
        images, track_ids = render_frame(
            args.seq, frame_id, det_rows, track_rows, args.scale, args.score_thresh
        )
        for key, image in images.items():
            path = os.path.join(output_dir, f"{key}_{frame_id:05d}.jpg")
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            strips[key].append(image)
            print(path)
        print(f"frame {frame_id} ids {track_ids}")

    gap = np.full((strips["rgb"][0].shape[0], 8, 3), 255, dtype=np.uint8)
    for key, images in strips.items():
        parts = []
        for image in images:
            if parts:
                parts.append(gap)
            parts.append(image)
        path = os.path.join(output_dir, f"{key}_strip.jpg")
        cv2.imwrite(path, np.concatenate(parts, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(path)


if __name__ == "__main__":
    main()
