"""
Visualize FEMOT tracking results: draw predicted track boxes and GT boxes on each frame.

Predicted boxes show track ID; both pred and GT are colored by category.
Since saved track txt does not include class, pred class is inferred by majority
vote of per-frame GT IoU matching over each track id.

Example:
  python tools/vis_femot_track_results.py \\
      --result_dir ssm_tracker/track_results/femot_dual_MambaTrack_femot_hyper_gate/test \\
      --output_dir ssm_tracker/track_results/femot_dual_MambaTrack_femot_hyper_gate/vis_test
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
from configs.dataset_paths import FEMOT_ROOT, FEMOT_FRAME_FMT, FEMOT_RGB_DIR_SUFFIX

CLASS_COLORS = {
    1: (0, 200, 0),
    3: (0, 128, 255),
}
UNKNOWN_COLOR = (128, 128, 128)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FEMOT track results with GT overlay')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='ssm_tracker/track_results/femot_dual_MambaTrack_femot_hyper_gate/test',
    )
    parser.add_argument('--data_root', type=str, default=FEMOT_ROOT)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--seq', type=str, default='', help='single sequence name, e.g. dvSave-2024_06_04_18_07_37')
    parser.add_argument('--max_frames', type=int, default=0, help='0 means all frames')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='IoU threshold for pred-class matching')
    parser.add_argument('--save_video', action='store_true', help='also save mp4 per sequence')
    parser.add_argument('--fps', type=int, default=20)
    return parser.parse_args()


def tlwh_to_tlbr(tlwh):
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_iou(a, b):
    """IoU between tlbr boxes a and b."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_track_results(path):
    rows = np.loadtxt(path, dtype=np.float32, delimiter=',')
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame = defaultdict(list)
    for row in rows:
        frame_id = int(row[0])
        track_id = int(row[1])
        tlwh = row[2:6]
        score = float(row[6])
        by_frame[frame_id].append({
            'track_id': track_id,
            'tlwh': tlwh,
            'score': score,
        })
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
        track_id = int(row[1])
        tlwh = row[2:6]
        cls_id = int(row[7])
        by_frame[frame_id].append({
            'track_id': track_id,
            'tlwh': tlwh,
            'cls_id': cls_id,
        })
    return by_frame


def infer_track_classes(pred_by_frame, gt_by_frame, iou_thresh):
    votes = defaultdict(list)
    for frame_id, preds in pred_by_frame.items():
        gts = gt_by_frame.get(frame_id, [])
        if not gts:
            continue
        gt_tlbrs = [tlwh_to_tlbr(g['tlwh']) for g in gts]
        for pred in preds:
            pred_tlbr = tlwh_to_tlbr(pred['tlwh'])
            best_iou = 0.0
            best_cls = None
            for gt, gt_tlbr in zip(gts, gt_tlbrs):
                iou = box_iou(pred_tlbr, gt_tlbr)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = gt['cls_id']
            if best_iou >= iou_thresh and best_cls is not None:
                votes[pred['track_id']].append(best_cls)

    track_cls = {}
    for track_id, cls_list in votes.items():
        track_cls[track_id] = Counter(cls_list).most_common(1)[0][0]
    return track_cls


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


def draw_box(img, tlwh, color, label='', dashed=False, thickness=2):
    x, y, w, h = [int(v) for v in tlwh]
    x2, y2 = x + w, y + h
    if dashed:
        draw_dashed_rect(img, (x, y), (x2, y2), color, thickness=thickness)
    else:
        cv2.rectangle(img, (x, y), (x2, y2), color, thickness=thickness)

    if label:
        text_y = max(y - 4, 12)
        cv2.putText(
            img,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


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

    pred_by_frame = load_track_results(result_path)
    gt_by_frame = load_gt(gt_path)
    track_cls = infer_track_classes(pred_by_frame, gt_by_frame, args.iou_thresh)

    frame_ids = sorted(pred_by_frame.keys())
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    seq_out_dir = os.path.join(output_dir, seq)
    os.makedirs(seq_out_dir, exist_ok=True)

    writer = None
    if args.save_video:
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

        for gt in gt_by_frame.get(frame_id, []):
            cls_id = gt['cls_id']
            color = CLASS_COLORS.get(cls_id, UNKNOWN_COLOR)
            draw_box(vis, gt['tlwh'], color, dashed=True, thickness=2)

        for pred in pred_by_frame.get(frame_id, []):
            cls_id = track_cls.get(pred['track_id'])
            color = CLASS_COLORS.get(cls_id, UNKNOWN_COLOR)
            label = str(pred['track_id'])
            draw_box(vis, pred['tlwh'], color, label=label, dashed=False, thickness=2)

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
