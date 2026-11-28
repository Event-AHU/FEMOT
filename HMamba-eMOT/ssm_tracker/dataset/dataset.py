from torch.utils.data import Dataset

import os 
import os.path as osp
import numpy as np 
import json
import torch 
from collections import OrderedDict

from loguru import logger
from .hypergraph_rules import ablate_incidence, build_rule_incidence, relations_to_drop

"""
segmented trajectory

"""

class TrajDataset(Dataset):

    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.window_size = self.cfgs['window_size']
        self.skip_sample = self.cfgs['skip_sample']

        with open(cfgs['anno_path'], 'r') as f:
            self.anno_file = json.load(f, )  # get the annotation file

        self.datasets = list(self.anno_file.keys())
        self.num_datasets = len(self.datasets)

        self.num_all_objs = 0
        for dataset in self.datasets:
            self.num_all_objs += self.anno_file[dataset]['total_objs']

        # print info
        logger.info(f'Total datasets: {self.datasets}')
        logger.info(f'Total obj numbers: {self.num_all_objs}')


    def __getitem__(self, idx):
        # first, random choose a dataset
        dataset_idx = np.random.randint(0, self.num_datasets)

        dataset_chosen = self.datasets[dataset_idx]

        # second, random choose an object
        total_objs = self.anno_file[dataset_chosen]['total_objs']

        object_chosen = np.random.randint(0, total_objs)

        # third, get the trajectories
        trajectories = self.anno_file[dataset_chosen][str(object_chosen)]['bboxes']
        traj_len = self.anno_file[dataset_chosen][str(object_chosen)]['traj_len']

        assert traj_len >= self.skip_sample * self.window_size, 'check the temporal window size'

        # random sample, choose the start idx
        max_start_idx = traj_len - 1 - self.skip_sample * (self.window_size - 1)
        start_idx = np.random.randint(0, max_start_idx + 1)

        sampled_trajectories = []
        for _ in range(self.window_size):
            sampled_trajectories.append(trajectories[start_idx])
            start_idx += self.skip_sample

        sampled_trajectories = torch.tensor(sampled_trajectories)
        if self.cfgs['manner'] == 'diff':
            # calculate the difference between bboxes
            ret = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                ret[i] = sampled_trajectories[i + 1] - sampled_trajectories[i]  # normalized diff

        return ret[:-1], ret[-1]  # x and label

    def __len__(self):
        return self.num_all_objs
    

class TrajDatasetv2(Dataset):
    """
    sample-wise instead of object-wise in class `TrajDataset`
    """

    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.window_size = self.cfgs['window_size']
        self.skip_sample = self.cfgs['skip_sample']
        self.scale_factor = self.cfgs['scale_factor']
        self.scale_factor_diff = self.cfgs['scale_factor_diff'] if 'scale_factor_diff' in self.cfgs else None

        assert self.skip_sample == 1, 'not implemented yet'

        with open(cfgs['anno_path'], 'r') as f:
            self.anno_file = json.load(f, )  # get the annotation file

        self.datasets = list(self.anno_file.keys())
        self.num_datasets = len(self.datasets)

        self.num_samples = 0  # the total number of samples
        self.idx_obj_map = dict()  # key: start idx, value: obj id

        strat_idx = 0
        for dataset in self.datasets:
            for obj_id in self.anno_file[dataset].keys():
                if obj_id in ['total_objs', 'obj_id_start']: continue

                traj_len = self.anno_file[dataset][obj_id]['traj_len']
                self.idx_obj_map[strat_idx] = obj_id
                strat_idx += traj_len - self.window_size + 1
                
                self.num_samples += traj_len - self.window_size + 1 

        
        self.idx_milestone_list = list(self.idx_obj_map.keys())

    def __getitem__(self, idx):
        
        # binary search for the obj that idx corresponds to
        milestone_idx, obj = self._binary_search(idx)
        # find the dataset that obj in 
        dataset_key = None 
        for dataset in self.datasets:
            if int(obj) >= int(self.anno_file[dataset]['obj_id_start']): dataset_key = dataset
            else: break 

        bboxes = self.anno_file[dataset_key][obj]['bboxes']  # get the bboxes
        traj_start = idx - milestone_idx

        traj = torch.tensor(bboxes[traj_start: traj_start + self.window_size])

        if self.cfgs['manner'] == 'diff':
            # calculate the difference between bboxes
            condition = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                condition[i] = traj[i + 1] - traj[i]  # normalized diff

            return {'traj': traj, 'condition': condition[:-1] * self.scale_factor, 'label': condition[-1] * self.scale_factor}

        if self.cfgs['manner'] == 'bbox_and_diff':
            # calculate the difference between bboxes
            condition = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                condition[i] = traj[i + 1] - traj[i]  # normalized diff

            condition_ = torch.cat([traj[1:] * self.scale_factor, condition * self.scale_factor_diff], dim=1)  # [xc, yc, w, h, dx, dy, dw, dh]

            return {'traj': traj, 'condition': condition_[:-1], 'label': condition_[-1]}

        else:
            raise NotImplementedError

    def _binary_search(self, idx):
        # find the last i that self.idx_milestone_list[i] <= idx

        l, r = 0, len(self.idx_milestone_list) - 1

        while l < r:
            mid = (l + r + 1) >> 1
            if self.idx_milestone_list[mid] <= idx: l = mid
            else: r = mid - 1

        obj_idx = self.idx_milestone_list[l]
        return obj_idx, self.idx_obj_map[obj_idx]



    def __len__(self, ):
        return self.num_samples


class FrameTrajDataset(Dataset):
    """Frame-centric multi-object windows for the spatial-hypergraph MVP."""

    def __init__(self, cfgs, split='train'):
        super().__init__()
        self.cfgs = cfgs
        self.split = split
        self.window_size = int(cfgs['window_size'])
        self.q = self.window_size - 2
        self.scale_factor = float(cfgs['scale_factor'])
        self.max_objects = int(cfgs.get('max_objects', 32))
        self.feature_root = os.path.abspath(
            os.path.join(cfgs['modal_feature_root'], split)
        )
        self.cache_size = int(cfgs.get('sequence_cache_size', 2))
        self.cache = OrderedDict()
        self.samples = []
        dropped = relations_to_drop(cfgs)
        if dropped:
            logger.info(
                'FrameTrajDataset {} drops hyper relations {}',
                split,
                sorted(dropped),
            )

        feature_files = sorted(
            name for name in os.listdir(self.feature_root) if name.endswith('.npz')
        )
        for name in feature_files:
            path = os.path.join(self.feature_root, name)
            data = np.load(path)
            frame_ids = data['frame_ids'].astype(np.int32)
            object_ids = data['object_ids'].astype(np.int32)
            frame_to_ids = {
                int(frame): set(object_ids[frame_ids == frame].tolist())
                for frame in np.unique(frame_ids)
            }
            frames = sorted(frame_to_ids)
            frame_set = set(frames)
            for ref_frame in frames:
                required = list(range(ref_frame - self.q, ref_frame + 2))
                if not all(frame in frame_set for frame in required):
                    continue
                eligible = set.intersection(
                    *(frame_to_ids[frame] for frame in required)
                )
                if len(eligible) >= 2:
                    self.samples.append((name, ref_frame, tuple(sorted(eligible))))

        logger.info(
            f'FrameTrajDataset {split}: {len(self.samples)} frame samples '
            f'from {len(feature_files)} sequences'
        )

    def _load_sequence(self, name):
        if name in self.cache:
            data = self.cache.pop(name)
            self.cache[name] = data
            return data
        raw = np.load(os.path.join(self.feature_root, name))
        frame_ids = raw['frame_ids'].astype(np.int32)
        object_ids = raw['object_ids'].astype(np.int32)
        lookup = {
            (int(frame_ids[index]), int(object_ids[index])): index
            for index in range(len(frame_ids))
        }
        data = {
            'boxes': raw['boxes_tlwh'].astype(np.float32),
            'rgb': raw['rgb_feats'].astype(np.float32),
            'evt': raw['evt_feats'].astype(np.float32),
            'image_hw': raw['image_hw'].astype(np.float32),
            'lookup': lookup,
        }
        self.cache[name] = data
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data

    def __getitem__(self, index):
        name, ref_frame, eligible = self.samples[index]
        data = self._load_sequence(name)
        ids = np.asarray(eligible, dtype=np.int32)
        if len(ids) > self.max_objects:
            if self.split == 'train':
                ids = np.random.choice(ids, self.max_objects, replace=False)
                ids.sort()
            else:
                ids = ids[:self.max_objects]
        n = len(ids)
        h, w = data['image_hw']
        norm = np.asarray([w, h, w, h], dtype=np.float32)

        boxes = np.zeros((n, self.q + 2, 4), dtype=np.float32)
        rgb_dim = data['rgb'].shape[1]
        evt_dim = data['evt'].shape[1]
        rgb_seq = np.zeros((n, self.q + 2, rgb_dim), dtype=np.float32)
        evt_seq = np.zeros((n, self.q + 2, evt_dim), dtype=np.float32)
        for object_index, object_id in enumerate(ids):
            for time_index, frame in enumerate(
                range(ref_frame - self.q, ref_frame + 2)
            ):
                row = data['lookup'][(frame, int(object_id))]
                box = data['boxes'][row].copy()
                box[:2] += 0.5 * box[2:]
                boxes[object_index, time_index] = box / norm
                rgb_seq[object_index, time_index] = data['rgb'][row]
                evt_seq[object_index, time_index] = data['evt'][row]

        differences = boxes[:, 1:] - boxes[:, :-1]
        condition = differences[:, :self.q] * self.scale_factor
        label = differences[:, self.q] * self.scale_factor
        per_frame = bool(self.cfgs.get('spatial_hyper_per_frame', False))
        pairwise = bool(self.cfgs.get('hyper_pairwise', False))
        rule_kwargs = dict(
            max_vertices=self.max_objects,
            spatial_radius=float(self.cfgs.get('hyper_spatial_radius', 0.10)),
            rgb_threshold=float(self.cfgs.get('hyper_rgb_threshold', 0.80)),
            evt_threshold=float(self.cfgs.get('hyper_evt_threshold', 0.80)),
            topk=int(self.cfgs.get('hyper_topk', 5)),
            pairwise=pairwise,
        )
        relation_slots = (
            max(self.max_objects * (self.max_objects - 1) // 2, 1)
            if pairwise else self.max_objects
        )
        if per_frame:
            # Align H[t] with condition[:, t]: motion ends at box index t+1.
            incidence = {
                'spatial': np.zeros(
                    (self.max_objects, self.q, relation_slots), dtype=np.float32
                ),
                'rgb': np.zeros(
                    (self.max_objects, self.q, relation_slots), dtype=np.float32
                ),
                'evt': np.zeros(
                    (self.max_objects, self.q, relation_slots), dtype=np.float32
                ),
                'motion': np.zeros((self.max_objects, self.q, 3), dtype=np.float32),
            }
            for t in range(self.q):
                frame_inc = build_rule_incidence(
                    centers=boxes[:n, t + 1, :2],
                    rgb_features=rgb_seq[:n, t + 1],
                    evt_features=evt_seq[:n, t + 1],
                    motions=condition[:n, t],
                    **rule_kwargs,
                )
                for relation, matrix in frame_inc.items():
                    incidence[relation][:n, t] = matrix[:n]
        else:
            incidence = build_rule_incidence(
                centers=boxes[:n, self.q, :2],
                rgb_features=rgb_seq[:n, self.q],
                evt_features=evt_seq[:n, self.q],
                motions=condition[:n, -1],
                **rule_kwargs,
            )
        incidence = ablate_incidence(incidence, self.cfgs)

        padded_condition = np.zeros((self.max_objects, self.q, 4), dtype=np.float32)
        padded_label = np.zeros((self.max_objects, 4), dtype=np.float32)
        vertex_mask = np.zeros(self.max_objects, dtype=bool)
        padded_condition[:n] = condition
        padded_label[:n] = label
        vertex_mask[:n] = True
        return {
            'condition': torch.from_numpy(padded_condition),
            'label': torch.from_numpy(padded_label),
            'vertex_mask': torch.from_numpy(vertex_mask),
            'H_spatial': torch.from_numpy(incidence['spatial']),
            'H_rgb': torch.from_numpy(incidence['rgb']),
            'H_evt': torch.from_numpy(incidence['evt']),
            'H_motion': torch.from_numpy(incidence['motion']),
        }

    def __len__(self):
        return len(self.samples)


def frame_hypergraph_collate(batch):
    """Stack and trim padded vertices to the largest object count in a batch."""
    stacked = {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0]
    }
    max_vertices = int(stacked['vertex_mask'].sum(dim=1).max().item())
    stacked['condition'] = stacked['condition'][:, :max_vertices]
    stacked['label'] = stacked['label'][:, :max_vertices]
    stacked['vertex_mask'] = stacked['vertex_mask'][:, :max_vertices]
    for relation in ('spatial', 'rgb', 'evt'):
        key = f'H_{relation}'
        # Trim padded vertices only. Pairwise graphs store one column per
        # 2-node edge, so the edge axis can be wider than the vertex axis.
        if stacked[key].ndim == 4:
            stacked[key] = stacked[key][:, :max_vertices]
        else:
            stacked[key] = stacked[key][:, :max_vertices]
    if stacked['H_motion'].ndim == 4:
        stacked['H_motion'] = stacked['H_motion'][:, :max_vertices]
    else:
        stacked['H_motion'] = stacked['H_motion'][:, :max_vertices]
    return stacked
