"""
Debug hyper_mask_threshold=0.2 test tracking on a single FEMOT sequence.

Set breakpoints in ssm_tracker/models/hypergraph_b.py before F5:
  - line 40: H = F.softmax(affinity, dim=-1)
  - line 42: H_mask = (H > mask_threshold)   # threshold=0.2

Uses the real MambaTracker -> predict() path with epoch25 (retrained t0.2).

Example:
  python tools/debug_hard05_test_track.py \\
      --config_file ssm_tracker/cfgs/MambaTrack_femot_hyper_hard05.yaml \\
      --motion_model_path ssm_tracker/saved_ckpts/mambatrack_femot_hyper_hard05/epoch25.pth \\
      --det_path det_results/femot/test \\
      --seq dvSave-2024_06_04_18_07_37 \\
      --start_frame 7 \\
      --max_frames 20 \\
      --device 4
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ssm_tracker'))


def _bind_gpu_before_torch(device: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before any torch import (avoids wrong cuda:0 mapping)."""
    if device == 'cpu':
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = device


def _early_parse_device() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='4')
    args, _ = parser.parse_known_args()
    _bind_gpu_before_torch(args.device)
    return args.device


_early_parse_device()

from copy import deepcopy

import numpy as np
import yaml
from loguru import logger

from track_utils.tracker import MambaTracker
from train_utils.envs import select_device


def read_meta_data(file_path):
    ret = {}
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            items = row.split(',')
            ret[items[0]] = [float(items[i]) for i in range(1, len(items))]
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='ssm_tracker/cfgs/MambaTrack_femot_hyper_hard05.yaml',
    )
    parser.add_argument(
        '--motion_model_path',
        type=str,
        default='ssm_tracker/saved_ckpts/mambatrack_femot_hyper_hard05/epoch25.pth',
    )
    parser.add_argument(
        '--det_path',
        type=str,
        default='det_results/femot/test',
    )
    parser.add_argument(
        '--seq',
        type=str,
        default='dvSave-2024_06_04_18_07_37',
        help='Sequence name without .txt suffix',
    )
    parser.add_argument('--max_frames', type=int, default=20)
    parser.add_argument(
        '--start_frame',
        type=int,
        default=1,
        help='First frame to process (use 7+ to skip init and hit motion model quickly)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='4',
        help='GPU index, or cpu. Must match value used at process start (see _early_parse_device).',
    )
    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def build_cfgs(config_file, motion_model_path):
    config_file = resolve_path(config_file)
    motion_model_path = resolve_path(motion_model_path)

    with open(config_file, 'r') as f:
        raw = yaml.safe_load(f)

    cfgs = deepcopy(raw['train'])
    cfgs.update(raw['inference'])
    cfgs.update({
        'motion_model_path': motion_model_path,
        'scale_factor': raw['dataset']['scale_factor'],
        'scale_factor_diff': raw['dataset'].get('scale_factor_diff'),
        'manner': raw['dataset']['manner'],
    })
    return cfgs


def main():
    args = parse_args()
    det_path = resolve_path(args.det_path)
    seq_file = os.path.join(det_path, args.seq + '.txt')
    meta_file = os.path.join(det_path, 'meta_data.txt')

    if not os.path.isfile(seq_file):
        raise FileNotFoundError(f'Missing detection file: {seq_file}')
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(f'Missing meta_data: {meta_file}')

    cfgs = build_cfgs(args.config_file, args.motion_model_path)
    _bind_gpu_before_torch(args.device)
    device = select_device(args.device)

    meta_data = read_meta_data(meta_file)
    seq_meta = meta_data[args.seq]
    file_content = np.loadtxt(seq_file, dtype=float, delimiter=',')
    max_frame_id = min(int(file_content[:, 0].max()), args.max_frames)
    start_frame = max(1, min(args.start_frame, max_frame_id))

    logger.info(f'Debug seq={args.seq}, frames={start_frame}..{max_frame_id}, device={device}')
    logger.info(f'Config: use_hyper_b={cfgs.get("use_hyper_b")}, '
                f'mask_threshold={cfgs.get("hyper_mask_threshold")}, '
                f'temperature={cfgs.get("hyper_temperature")}, '
                f'enable_time_thresh={cfgs.get("enable_time_thresh")}')
    logger.info('Breakpoints: hypergraph_b.py:40 (H), :42 (H_mask); mamba_hyper.py:72 (hyper_b call)')

    tracker = MambaTracker(cfgs, device=device)

    if start_frame > 1:
        logger.info(f'Warming up frames 1..{start_frame - 1} (no motion debug yet)')
        for frame_id in range(1, start_frame):
            current_det = file_content[file_content[:, 0] == float(frame_id)]
            dets = current_det[:, 2:8] if len(current_det) else np.empty((0, 6))
            tracker.update(dets, meta_data=seq_meta)

    model_calls = 0

    for frame_id in range(start_frame, max_frame_id + 1):
        current_det = file_content[file_content[:, 0] == float(frame_id)]
        dets = current_det[:, 2:8] if len(current_det) else np.empty((0, 6))

        tracker.update(dets, meta_data=seq_meta)

        active = len(tracker.tracked_tracklets)
        logger.info(f'frame {frame_id}: dets={len(dets)}, active_tracks={active}')

        # After enable_time_thresh, existing tracks call motion_predictor in predict()
        if frame_id >= cfgs['enable_time_thresh'] and active > 0:
            model_calls += active
            logger.info(
                f'  -> motion model likely invoked ~{active} times this frame '
                f'(check H at hypergraph_b.py:40 in debugger)'
            )

    logger.info(f'Done. Approx model forward calls: {model_calls}')
    logger.info(
        'In debugger at line 40, inspect: affinity, H (B,N,L), H.sum(-1)≈1, '
        'H.max(-1) for per-hyperedge peak'
    )


if __name__ == '__main__':
    main()
