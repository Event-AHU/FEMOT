"""Frame-aligned baseline windows for the UniTrack loss.

Each sample is one camera frame. Every track that ends a baseline window of
`window_size` observations on that frame is included. Histories follow each
track's own valid observations, which is the same spacing the baseline
trajectory dataset uses.
"""
import configparser
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class UniTrackClipDataset(Dataset):
    def __init__(self, cfgs):
        self.window_size = int(cfgs['window_size'])
        self.scale_factor = float(cfgs['scale_factor'])
        self.temporal_steps = int(cfgs.get('temporal_steps', 3))
        self.filter_len = int(cfgs.get('filter_len', 25))
        categories = cfgs.get('categories', None)
        self.categories = None if categories is None else set(int(c) for c in categories)
        if self.window_size < 2:
            raise ValueError('window_size must be at least 2')
        if self.temporal_steps < 3:
            raise ValueError('temporal_steps must be at least 3')

        self.sequences = []
        self.samples = []
        gt_root = cfgs['gt_root']
        for name in sorted(os.listdir(gt_root)):
            gt_path = os.path.join(gt_root, name, 'gt', 'gt.txt')
            info_path = os.path.join(gt_root, name, 'seqinfo.ini')
            if not os.path.isfile(gt_path) or not os.path.isfile(info_path):
                continue
            sequence = self._load_sequence(name, gt_path, info_path)
            if sequence is None:
                continue
            seq_index = len(self.sequences)
            self.sequences.append(sequence)
            for frame, entries in sequence['by_frame'].items():
                if entries:
                    self.samples.append((seq_index, int(frame)))
        if not self.samples:
            raise RuntimeError('no UniTrack clips under ' + gt_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq_index, frame = self.samples[index]
        sequence = self.sequences[seq_index]
        entries = sequence['by_frame'][frame]
        min_index = self.window_size - 1
        temporal_start = min_index + self.temporal_steps - 1

        conditions = []
        labels = []
        prev_boxes = []
        gt_boxes = []
        track_ids = []
        temporal_conditions = []
        temporal_prev = []
        temporal_gt = []

        for track_index, end_index in entries:
            track = sequence['tracks'][track_index]
            condition, label, prev_box, gt_box = self._window(track, end_index)
            conditions.append(condition)
            labels.append(label)
            prev_boxes.append(prev_box)
            gt_boxes.append(gt_box)
            track_ids.append(track_index + 1)
            if end_index < temporal_start:
                continue
            steps_c, steps_p, steps_g = [], [], []
            for step in range(end_index - self.temporal_steps + 1, end_index + 1):
                cond, _, prev, gt = self._window(track, step)
                steps_c.append(cond)
                steps_p.append(prev)
                steps_g.append(gt)
            temporal_conditions.append(np.stack(steps_c, axis=0))
            temporal_prev.append(np.stack(steps_p, axis=0))
            temporal_gt.append(np.stack(steps_g, axis=0))

        sample = {
            'condition': torch.from_numpy(np.stack(conditions, axis=0)),
            'label': torch.from_numpy(np.stack(labels, axis=0)),
            'prev_box': torch.from_numpy(np.stack(prev_boxes, axis=0)),
            'gt_box': torch.from_numpy(np.stack(gt_boxes, axis=0)),
            'track_id': torch.tensor(track_ids, dtype=torch.float32),
            'width': sequence['width'],
            'height': sequence['height'],
        }
        if temporal_conditions:
            sample['temporal_condition'] = torch.from_numpy(np.stack(temporal_conditions, axis=0))
            sample['temporal_prev'] = torch.from_numpy(np.stack(temporal_prev, axis=0))
            sample['temporal_gt'] = torch.from_numpy(np.stack(temporal_gt, axis=0))
        else:
            sample['temporal_condition'] = torch.zeros(0, self.temporal_steps, self.window_size - 2, 4)
            sample['temporal_prev'] = torch.zeros(0, self.temporal_steps, 4)
            sample['temporal_gt'] = torch.zeros(0, self.temporal_steps, 4)
        return sample

    def _window(self, track, end_index):
        start = end_index - self.window_size + 1
        boxes = track['boxes'][start:end_index + 1]
        diff = boxes[1:] - boxes[:-1]
        condition = diff[:-1] * self.scale_factor
        label = diff[-1] * self.scale_factor
        return condition.astype(np.float32), label.astype(np.float32), boxes[-2].astype(np.float32), boxes[-1].astype(np.float32)

    def _load_sequence(self, name, gt_path, info_path):
        parser = configparser.ConfigParser()
        parser.read(info_path)
        width = int(parser['Sequence']['imWidth'])
        height = int(parser['Sequence']['imHeight'])
        rows = np.loadtxt(gt_path, delimiter=',', dtype=np.float64)
        if rows.size == 0:
            return None
        if rows.ndim == 1:
            rows = rows[None, :]
        grouped = defaultdict(list)
        for row in rows:
            if int(row[6]) != 1:
                continue
            category = int(row[7])
            if self.categories is not None and category not in self.categories:
                continue
            grouped[int(row[1])].append(row)
        tracks = []
        by_frame = defaultdict(list)
        min_index = self.window_size - 1
        for track_id in sorted(grouped):
            track_rows = sorted(grouped[track_id], key=lambda item: int(item[0]))
            if len(track_rows) < self.filter_len:
                continue
            frames = np.array([int(item[0]) for item in track_rows], dtype=np.int32)
            boxes = []
            for item in track_rows:
                x, y, w, h = [float(v) for v in item[2:6]]
                boxes.append([
                    (x + 0.5 * w) / width,
                    (y + 0.5 * h) / height,
                    w / width,
                    h / height,
                ])
            track_index = len(tracks)
            tracks.append({
                'id': track_id,
                'frames': frames,
                'boxes': np.asarray(boxes, dtype=np.float32),
            })
            for end_index in range(min_index, len(frames)):
                by_frame[int(frames[end_index])].append((track_index, end_index))
        if not tracks:
            return None
        return {
            'name': name,
            'width': width,
            'height': height,
            'tracks': tracks,
            'by_frame': dict(by_frame),
        }
