#!/usr/bin/env python3
"""Run TrackEval on FEMOT tracking results."""

import argparse
import os
import sys
from multiprocessing import freeze_support

import numpy as np

# TrackEval uses deprecated NumPy aliases removed in NumPy 2.x.
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'bool'):
    np.bool = np.bool_

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRACK_EVAL_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'TrackEval'))
FEMOT_ROOT = '/rydata/wangx_dataset/_dataset/FEMOT'
SEQMAP_DIR = os.path.join(os.path.dirname(__file__), 'seqmaps')

sys.path.insert(0, TRACK_EVAL_ROOT)
import trackeval  # noqa: E402


def prepare_tracker_layout(pred_dir: str, split: str, tracker_name: str) -> str:
    tracker_root = os.path.join(os.path.dirname(__file__), 'trackers', split)
    data_dir = os.path.join(tracker_root, tracker_name, 'data')
    os.makedirs(data_dir, exist_ok=True)

    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)

    skip = {'meta_data.txt', 'metrics_summary.txt'}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith('.txt') or name in skip:
            continue
        src = os.path.abspath(os.path.join(pred_dir, name))
        dst = os.path.join(data_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    return tracker_root


def ensure_seqmap(split: str) -> str:
    seqmap_src = os.path.join(SEQMAP_DIR, f'FEMOT-{split}.txt')
    if not os.path.isfile(seqmap_src):
        raise FileNotFoundError(f'seqmap not found: {seqmap_src}')

    gt_seqmap_dir = os.path.join(FEMOT_ROOT, split, 'seqmaps')
    os.makedirs(gt_seqmap_dir, exist_ok=True)
    seqmap_dst = os.path.join(gt_seqmap_dir, f'FEMOT-{split}.txt')
    if os.path.islink(seqmap_dst) or os.path.isfile(seqmap_dst):
        os.remove(seqmap_dst)
    os.symlink(os.path.abspath(seqmap_src), seqmap_dst)
    return seqmap_src


def main():
    parser = argparse.ArgumentParser(description='Evaluate FEMOT results with TrackEval')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--pred_dir', type=str, default='')
    parser.add_argument('--tracker_name', type=str, default='MambaTrack')
    parser.add_argument('--metrics', nargs='+', default=['HOTA', 'CLEAR', 'Identity'])
    args = parser.parse_args()

    pred_dir = args.pred_dir or os.path.join(
        PROJECT_ROOT, 'ssm_tracker', 'track_results', 'femot_MambaTrack_femot', args.split
    )
    gt_folder = os.path.join(FEMOT_ROOT, args.split)

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f'prediction folder not found: {pred_dir}')
    if not os.path.isdir(gt_folder):
        raise FileNotFoundError(f'GT folder not found: {gt_folder}')

    seqmap = ensure_seqmap(args.split)
    tracker_root = prepare_tracker_layout(pred_dir, args.split, args.tracker_name)

    print(f'GT:      {gt_folder}')
    print(f'Pred:    {pred_dir}')
    print(f'Seqmap:  {seqmap}')
    print(f'Tracker: {args.tracker_name}')

    dataset_config = {
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': tracker_root,
        'TRACKERS_TO_EVAL': [args.tracker_name],
        'BENCHMARK': 'FEMOT',
        'SPLIT_TO_EVAL': args.split,
        'SKIP_SPLIT_FOL': True,
        'DO_PREPROC': False,
        'PLOT_CURVES': False,
        'PRINT_CONFIG': True,
    }
    metrics_config = {'METRICS': args.metrics, 'THRESHOLD': 0.5}
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['PLOT_CURVES'] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric_cls in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                       trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric_cls.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric_cls(metrics_config))

    if not metrics_list:
        raise ValueError('No valid metrics selected')

    evaluator.evaluate(dataset_list, metrics_list)
    print(f'Done. Results: {tracker_root}/{args.tracker_name}/')


if __name__ == '__main__':
    freeze_support()
    main()
