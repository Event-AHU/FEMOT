"""
Visualize FEMOT YOLO detection results with GT overlay.

Each frame shows:
  - solid green boxes: detector predictions with score > 0.2
  - solid red boxes: detector predictions with score <= 0.2
  - dashed boxes: ground truth

No text labels, legend, or class-specific colors.

Example:
  python tools/vis_femot_det_results.py \\
      --result_dir det_results/femot_dual/test \\
      --output_dir det_results/femot_dual/vis_test
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
from configs.dataset_paths import FEMOT_ROOT, FEMOT_FRAME_FMT, FEMOT_RGB_DIR_SUFFIX

DET_COLOR = (0, 255, 0)        # green, score > thresh
DET_LOW_SCORE_COLOR = (0, 0, 255)  # red, score <= thresh
LOW_SCORE_THRESH = 0.2
GT_COLOR = (0, 128, 255)  # orange, dashed


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FEMOT detection results with GT overlay')
    parser.add_argument('--result_dir', type=str, default='det_results/femot_dual/test')
    parser.add_argument('--data_root', type=str, default=FEMOT_ROOT)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--seq', type=str, default='', help='single sequence name')
    parser.add_argument('--max_frames', type=int, default=0, help='0 means all frames')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    return parser.parse_args()


def load_det_results(path):
    rows = np.loadtxt(path, dtype=np.float32, delimiter=',')
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame = defaultdict(list)
    for row in rows:
        frame_id = int(row[0])
        tlwh = row[2:6]
        score = float(row[6])
        by_frame[frame_id].append((tlwh, score))
    return by_frame


def load_gt(path):
    rows = np.loadtxt(path, dtype=np.float32, delimiter=',')
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame = defaultdict(list)
    for row in rows:
        if int(row[6]) != 1:
            continue
        frame_id = int(row[0])
        tlwh = row[2:6]
        by_frame[frame_id].append(tlwh)
    return by_frame


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in pts:
        length = int(np.hypot(ex - sx, ey - sy))
        if length <= 0:
            continue
        for i in range(0, length, gap * 2):
            t1 = i / length
            t2 = min((i + gap) / length, 1.0)
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            p2 = (int(sx + (ex - sx) * t2), int(sy + (ey - sy) * t2))
            cv2.line(img, p1, p2, color, thickness)


def draw_box(img, tlwh, color, dashed=False, thickness=2):
    x, y, w, h = [int(v) for v in tlwh]
    x2, y2 = x + w, y + h
    if dashed:
        draw_dashed_rect(img, (x, y), (x2, y2), color, thickness=thickness)
    else:
        cv2.rectangle(img, (x, y), (x2, y2), color, thickness=thickness)


def visualize_sequence(seq, args, result_dir, output_dir):
    result_path = os.path.join(result_dir, f'{seq}.txt')
    gt_path = os.path.join(args.data_root, args.split, seq, 'gt', 'gt.txt')
    img_dir = os.path.join(args.data_root, args.split, seq, f'{seq}{FEMOT_RGB_DIR_SUFFIX}')

    if not os.path.isfile(result_path):
        raise FileNotFoundError(result_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)

    det_by_frame = load_det_results(result_path)
    gt_by_frame = load_gt(gt_path)

    frame_ids = sorted(set(det_by_frame.keys()) | set(gt_by_frame.keys()))
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    seq_out_dir = os.path.join(output_dir, seq)
    os.makedirs(seq_out_dir, exist_ok=True)

    writer = None
    if args.save_video and frame_ids:
        first_img_path = os.path.join(img_dir, FEMOT_FRAME_FMT.format(frame_id=frame_ids[0]))
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            raise FileNotFoundError(first_img_path)
        h, w = first_img.shape[:2]
        video_path = os.path.join(seq_out_dir, f'{seq}.mp4')
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            args.fps,
            (w, h),
        )

    for frame_id in tqdm(frame_ids, desc=seq, ncols=100):
        img_path = os.path.join(img_dir, FEMOT_FRAME_FMT.format(frame_id=frame_id))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        vis = img.copy()

        for tlwh in gt_by_frame.get(frame_id, []):
            draw_box(vis, tlwh, GT_COLOR, dashed=True, thickness=2)

        for tlwh, score in det_by_frame.get(frame_id, []):
            color = DET_LOW_SCORE_COLOR if score <= LOW_SCORE_THRESH else DET_COLOR
            draw_box(vis, tlwh, color, dashed=False, thickness=2)

        out_path = os.path.join(seq_out_dir, FEMOT_FRAME_FMT.format(frame_id=frame_id))
        cv2.imwrite(out_path, vis)
        if writer is not None:
            writer.write(vis)

    if writer is not None:
        writer.release()


def main():
    args = parse_args()
    result_dir = args.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_ROOT, result_dir)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(result_dir), 'vis_' + os.path.basename(result_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.seq:
        seqs = [args.seq]
    else:
        seqs = sorted([
            f[:-4] for f in os.listdir(result_dir)
            if f.endswith('.txt') and f not in {'meta_data.txt', 'metrics_summary.txt'}
        ])

    print(f'Result dir : {result_dir}')
    print(f'Output dir : {output_dir}')
    print(f'Sequences  : {len(seqs)}')

    for seq in seqs:
        visualize_sequence(seq, args, result_dir, output_dir)

    print(f'Done. Visualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
