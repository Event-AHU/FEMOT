"""
Evaluate MOT tracking results (MOTA, IDF1, etc.) without motmetrics.
Uses lapjv for bipartite matching (same as MOTChallenge).
"""

import argparse
import os
import sys

import lap
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.dataset_paths import FEMOT_ROOT

VALID_GT_CLASSES = {1, 3}


def load_gt(gt_path: str):
    frames = {}
    with open(gt_path, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            if int(float(p[6])) != 1 or int(float(p[7])) not in VALID_GT_CLASSES:
                continue
            frame = int(float(p[0]))
            tid = int(float(p[1]))
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            frames.setdefault(frame, []).append((tid, box))
    return frames


def load_pred(pred_path: str):
    frames = {}
    with open(pred_path, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            tid = int(float(p[1]))
            box = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
            frames.setdefault(frame, []).append((tid, box))
    return frames


def iou_box(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def eval_sequence(gt_frames, pred_frames, iou_thresh=0.5):
    stats = dict(
        num_gt=0, num_pred=0, tp=0, fp=0, fn=0,
        idsw=0, idtp=0, idfp=0, idfn=0,
        sum_iou=0.0,
    )
    last_match = {}

    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    for frame in all_frames:
        gt_objs = gt_frames.get(frame, [])
        pred_objs = pred_frames.get(frame, [])
        stats['num_gt'] += len(gt_objs)
        stats['num_pred'] += len(pred_objs)

        n_gt, n_pred = len(gt_objs), len(pred_objs)
        if n_gt == 0:
            stats['fp'] += n_pred
            stats['idfp'] += n_pred
            continue
        if n_pred == 0:
            stats['fn'] += n_gt
            stats['idfn'] += n_gt
            continue

        iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
        for i, (_, gb) in enumerate(gt_objs):
            for j, (_, pb) in enumerate(pred_objs):
                iou_mat[i, j] = iou_box(gb, pb)

        cost = 1.0 - iou_mat
        cost[iou_mat < iou_thresh] = 1e9
        _, x, _ = lap.lapjv(cost, extend_cost=True)

        matched_gt = set()
        matched_pred = set()
        for i, j in enumerate(x):
            if j < 0 or cost[i, j] >= 1e8:
                continue
            matched_gt.add(i)
            matched_pred.add(j)
            gt_id = gt_objs[i][0]
            pred_id = pred_objs[j][0]
            stats['tp'] += 1
            stats['sum_iou'] += iou_mat[i, j]
            stats['idtp'] += 1
            if gt_id in last_match and last_match[gt_id] != pred_id:
                stats['idsw'] += 1
            last_match[gt_id] = pred_id

        fn = n_gt - len(matched_gt)
        fp = n_pred - len(matched_pred)
        stats['fn'] += fn
        stats['fp'] += fp
        stats['idfn'] += fn
        stats['idfp'] += fp

    return stats


def compute_metrics(stats):
    gt = stats['num_gt']
    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
    idtp, idfp, idfn = stats['idtp'], stats['idfp'], stats['idfn']

    mota = 1.0 - (fn + fp + stats['idsw']) / gt if gt > 0 else 0.0
    motp = stats['sum_iou'] / tp if tp > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

    return dict(
        gt=gt, tp=tp, fp=fp, fn=fn,
        idsw=stats['idsw'],
        MOTA=mota * 100,
        MOTP=motp * 100,
        IDF1=idf1 * 100,
        Precision=precision * 100,
        Recall=recall * 100,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, default=FEMOT_ROOT)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    args = parser.parse_args()

    gt_split_dir = os.path.join(args.gt_root, args.split)
    seqs = sorted([
        f[:-4] for f in os.listdir(args.pred_dir)
        if f.endswith('.txt') and f != 'meta_data.txt'
    ])

    all_stats = dict(num_gt=0, num_pred=0, tp=0, fp=0, fn=0, idsw=0, idtp=0, idfp=0, idfn=0, sum_iou=0.0)
    rows = []

    print(f'{"Sequence":<40} {"MOTA":>7} {"IDF1":>7} {"Prec":>7} {"Recall":>7} {"IDSW":>6}')
    print('-' * 80)

    for seq in seqs:
        gt_path = os.path.join(gt_split_dir, seq, 'gt', 'gt.txt')
        pred_path = os.path.join(args.pred_dir, seq + '.txt')
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f'[skip] {seq}')
            continue

        stats = eval_sequence(load_gt(gt_path), load_pred(pred_path), args.iou_thresh)
        m = compute_metrics(stats)
        rows.append((seq, m))
        for k in all_stats:
            all_stats[k] += stats[k]
        print(f'{seq:<40} {m["MOTA"]:7.2f} {m["IDF1"]:7.2f} {m["Precision"]:7.2f} {m["Recall"]:7.2f} {m["idsw"]:6d}')

    overall = compute_metrics(all_stats)
    print('-' * 80)
    print(f'{"OVERALL":<40} {overall["MOTA"]:7.2f} {overall["IDF1"]:7.2f} {overall["Precision"]:7.2f} {overall["Recall"]:7.2f} {overall["idsw"]:6d}')

    out_path = os.path.join(args.pred_dir, 'metrics_summary.txt')
    with open(out_path, 'w') as f:
        f.write(f'IoU threshold: {args.iou_thresh}\n\n')
        f.write(f'{"Sequence":<40} {"MOTA":>7} {"IDF1":>7} {"Prec":>7} {"Recall":>7} {"IDSW":>6}\n')
        for seq, m in rows:
            f.write(f'{seq:<40} {m["MOTA"]:7.2f} {m["IDF1"]:7.2f} {m["Precision"]:7.2f} {m["Recall"]:7.2f} {m["idsw"]:6d}\n')
        f.write('-' * 80 + '\n')
        f.write(f'{"OVERALL":<40} {overall["MOTA"]:7.2f} {overall["IDF1"]:7.2f} {overall["Precision"]:7.2f} {overall["Recall"]:7.2f} {overall["idsw"]:6d}\n')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
