"""
Generate the trajectory data (json file)

format:

{
    dataset(str): 
    {
        'total_objs': int
        'obj_id_start': int
        'object_id':
        {
            image_h: int 
            image_w: int
            traj_len: int
            bboxes: List[list]  # shape (len, 4), 4 indicates [xc, yc, w, h]
        }

    }
}
"""

import os 
import copy
import numpy as np 
import json
from typing import List, Dict, Union
import configparser
import cv2 
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.dataset_paths import FEMOT_ROOT

DATA_ROOT = '/data/wujiapeng/datasets'

MOT17_PATH = os.path.join(DATA_ROOT, 'MOT17')
VISDRONE_PATH = os.path.join(DATA_ROOT, 'VisDrone2019/VisDrone2019')
UAVDT_PATH = os.path.join(DATA_ROOT, 'UAVDT')
DANCETRACK_PATH = os.path.join(DATA_ROOT, 'DanceTrack')
FEMOT_PATH = FEMOT_ROOT

VALID_CATEGORIES = {
    'mot17': [1, ], 
    'visdrone': [1, 4, 5, 6, 9], 
    'uavdt': [1, ],
    'femot': [1, 3],
}

def gen_mot17(annotation_all: dict, split: str = 'train', obj_id_base: int = 0, 
              sequences: List[str] = None, filter: bool = True, filter_len: int = 15) -> int:

    print('parsing dataset MOT17')

    annotation_dataset = dict()

    seqs = sorted(os.listdir(os.path.join(MOT17_PATH, 'train')))

    if 'all' in sequences:
        sequences = seqs

    obj_id_start = obj_id_base
    obj_cnt, valid_obj_cnt = 0, 0
    

    for seq in seqs:

        if not seq in sequences: continue

        print(f'parsing {seq}, obj id base {obj_id_start}')

        # get h and w first
        config = configparser.ConfigParser()
        seq_info = os.path.join(MOT17_PATH, split, seq, 'seqinfo.ini')
        config.read(seq_info)
        image_w = int(config['Sequence']['imWidth'])
        image_h = int(config['Sequence']['imHeight'])

        gt_file_path = os.path.join(MOT17_PATH, split, seq, 'gt', 'gt.txt')

        seq_anno = np.loadtxt(gt_file_path, dtype=float, delimiter=',')

        obj_cnt_in_seq = np.unique(seq_anno[:, 1]).shape[0]
        valid_cnt_in_seq = 0

        for row_gt in seq_anno:

            if int(row_gt[6]) == 1 and int(row_gt[7]) in VALID_CATEGORIES['mot17']:  # valid
                obj_id = int(row_gt[1]) + obj_id_start
                bbox = [float(row_gt[i]) for i in range(2, 6)]
                # tlwh -> xywh
                bbox[0] += 0.5 * bbox[2]
                bbox[1] += 0.5 * bbox[3]
                # norm
                bbox[0] = round(bbox[0] / float(image_w), 6)
                bbox[1] = round(bbox[1] / float(image_h), 6)
                bbox[2] = round(bbox[2] / float(image_w), 6)
                bbox[3] = round(bbox[3] / float(image_h), 6)

                if not obj_id in annotation_dataset.keys():
                    # new obj
                    valid_cnt_in_seq += 1
                    annotation_dataset[obj_id] = {
                        'image_h': image_h, 
                        'image_w': image_w, 
                        'traj_len': 1, 
                        'bboxes': [bbox]
                    }
                else:
                    # update obj
                    annotation_dataset[obj_id]['traj_len'] += 1
                    annotation_dataset[obj_id]['bboxes'].append(bbox)

        print(f'in seq {seq}, total {obj_cnt_in_seq} objs, valid {valid_cnt_in_seq} objs')

        obj_cnt += obj_cnt_in_seq
        valid_obj_cnt += valid_cnt_in_seq
        obj_id_start += obj_cnt_in_seq

    # filter short trajs
    if filter:
        print(f'before filter, total {valid_obj_cnt} objects')
        all_objs = list(annotation_dataset.keys())

        # we need to update the key value to increase integer to read the data more conveniently
        annotation_dataset_filtered = dict()
        new_obj_id = 0

        for obj_id in all_objs:

            if annotation_dataset[obj_id]['traj_len'] < filter_len:
                valid_obj_cnt -= 1
            else:
                annotation_dataset_filtered[str(new_obj_id + obj_id_base)] = annotation_dataset.pop(obj_id, None)
                new_obj_id += 1

        assert new_obj_id == valid_obj_cnt, f'{new_obj_id}, {valid_obj_cnt}'

        annotation_dataset = annotation_dataset_filtered

        print(f'after filter, total {valid_obj_cnt} objs')

    annotation_dataset['total_objs'] = valid_obj_cnt
    annotation_dataset['obj_id_start'] = obj_id_base
    annotation_all['mot17'] = annotation_dataset

    return obj_id_base + valid_obj_cnt

def gen_visdrone(annotation_all: dict, split: str = 'VisDrone2019-MOT-train', obj_id_base: int = 0, 
                 sequences: List[str] = None, filter: bool = True, filter_len: int = 25) -> int:
              
    
    print('parsing dataset VisDrone')

    annotation_dataset = dict()

    seqs = sorted(os.listdir(os.path.join(VISDRONE_PATH, split, 'sequences')))

    if 'all' in sequences:
        sequences = seqs

    obj_id_start = obj_id_base
    obj_cnt, valid_obj_cnt = 0, 0

    for seq in seqs:

        if not seq in sequences: continue

        print(f'parsing {seq}, obj id base {obj_id_start}')

        # get h and w first
        seq_dir = os.path.join(VISDRONE_PATH, split, 'sequences', seq)
        frames = os.listdir(seq_dir)

        img_eg = cv2.imread(os.path.join(seq_dir, frames[0]))
        image_w, image_h = img_eg.shape[1], img_eg.shape[0]

        gt_file_path = os.path.join(VISDRONE_PATH, split, 'annotations', seq + '.txt')

        seq_anno = np.loadtxt(gt_file_path, dtype=float, delimiter=',')

        obj_cnt_in_seq = np.unique(seq_anno[:, 1]).shape[0]
        valid_cnt_in_seq = 0

        for row_gt in seq_anno:

            if int(row_gt[6]) == 1 and int(row_gt[7]) in VALID_CATEGORIES['visdrone']:  # valid
                obj_id = int(row_gt[1]) + obj_id_start
                bbox = [float(row_gt[i]) for i in range(2, 6)]
                # tlwh -> xywh
                bbox[0] += 0.5 * bbox[2]
                bbox[1] += 0.5 * bbox[3]
                # norm
                bbox[0] = round(bbox[0] / float(image_w), 6)
                bbox[1] = round(bbox[1] / float(image_h), 6)
                bbox[2] = round(bbox[2] / float(image_w), 6)
                bbox[3] = round(bbox[3] / float(image_h), 6)

                if not obj_id in annotation_dataset.keys():
                    # new obj
                    valid_cnt_in_seq += 1
                    annotation_dataset[obj_id] = {
                        'image_h': image_h, 
                        'image_w': image_w, 
                        'traj_len': 1, 
                        'bboxes': [bbox]
                    }
                else:
                    # update obj
                    annotation_dataset[obj_id]['traj_len'] += 1
                    annotation_dataset[obj_id]['bboxes'].append(bbox)

        print(f'in seq {seq}, total {obj_cnt_in_seq} objs, valid {valid_cnt_in_seq} objs')

        obj_cnt += obj_cnt_in_seq
        valid_obj_cnt += valid_cnt_in_seq
        obj_id_start += obj_cnt_in_seq

    if filter:
        print(f'before filter, total {valid_obj_cnt} objects')

        all_objs = list(annotation_dataset.keys())

        # we need to update the key value to increase integer to read the data more conveniently
        annotation_dataset_filtered = dict()
        new_obj_id = 0

        for obj_id in all_objs:

            if annotation_dataset[obj_id]['traj_len'] < filter_len:
                valid_obj_cnt -= 1
            else:
                annotation_dataset_filtered[str(new_obj_id + obj_id_base)] = annotation_dataset.pop(obj_id, None)
                new_obj_id += 1

        assert new_obj_id == valid_obj_cnt, f'{new_obj_id}, {valid_obj_cnt}'

        annotation_dataset = annotation_dataset_filtered

        print(f'after filter, total {valid_obj_cnt} objs')

    annotation_dataset['total_objs'] = valid_obj_cnt
    annotation_dataset['obj_id_start'] = obj_id_base
    annotation_all['visdrone'] = annotation_dataset

    return obj_id_base + valid_obj_cnt

def gen_dancetrack(annotation_all: dict, split: str = 'train', obj_id_base: int = 0, 
                 sequences: List[str] = None, filter: bool = True, filter_len: int = 25) -> int:

    print('parsing dataset DanceTrack')

    annotation_dataset = dict()

    seqs = sorted(os.listdir(os.path.join(DANCETRACK_PATH, 'train')))

    if 'all' in sequences:
        sequences = seqs

    obj_id_start = obj_id_base
    obj_cnt, valid_obj_cnt = 0, 0
    
    
    for seq in seqs:

        if not seq in sequences: continue

        print(f'parsing {seq}, obj id base {obj_id_start}')

        # get h and w first
        config = configparser.ConfigParser()
        seq_info = os.path.join(DANCETRACK_PATH, split, seq, 'seqinfo.ini')
        
        config.read(seq_info)
        image_w = int(config['Sequence']['imWidth'])
        image_h = int(config['Sequence']['imHeight'])

        gt_file_path = os.path.join(DANCETRACK_PATH, split, seq, 'gt', 'gt.txt')

        seq_anno = np.loadtxt(gt_file_path, dtype=float, delimiter=',')

        obj_cnt_in_seq = np.unique(seq_anno[:, 1]).shape[0]
        valid_cnt_in_seq = 0

        for row_gt in seq_anno:

            obj_id = int(row_gt[1]) + obj_id_start
            bbox = [float(row_gt[i]) for i in range(2, 6)]
            # tlwh -> xywh
            bbox[0] += 0.5 * bbox[2]
            bbox[1] += 0.5 * bbox[3]
            # norm
            bbox[0] = round(bbox[0] / float(image_w), 6)
            bbox[1] = round(bbox[1] / float(image_h), 6)
            bbox[2] = round(bbox[2] / float(image_w), 6)
            bbox[3] = round(bbox[3] / float(image_h), 6)

            if not obj_id in annotation_dataset.keys():
                # new obj
                valid_cnt_in_seq += 1
                annotation_dataset[obj_id] = {
                    'image_h': image_h, 
                    'image_w': image_w, 
                    'traj_len': 1, 
                    'bboxes': [bbox]
                }
            else:
                # update obj
                annotation_dataset[obj_id]['traj_len'] += 1
                annotation_dataset[obj_id]['bboxes'].append(bbox)

        print(f'in seq {seq}, total {obj_cnt_in_seq} objs, valid {valid_cnt_in_seq} objs')

        obj_cnt += obj_cnt_in_seq
        valid_obj_cnt += valid_cnt_in_seq
        obj_id_start += obj_cnt_in_seq

    # filter short trajs
    if filter:
        print(f'before filter, total {valid_obj_cnt} objects')
        all_objs = list(annotation_dataset.keys())

        # we need to update the key value to increase integer to read the data more conveniently
        annotation_dataset_filtered = dict()
        new_obj_id = 0

        for obj_id in all_objs:

            if annotation_dataset[obj_id]['traj_len'] < filter_len:
                valid_obj_cnt -= 1
            else:
                annotation_dataset_filtered[str(new_obj_id + obj_id_base)] = annotation_dataset.pop(obj_id, None)
                new_obj_id += 1

        assert new_obj_id == valid_obj_cnt, f'{new_obj_id}, {valid_obj_cnt}'

        annotation_dataset = annotation_dataset_filtered

        print(f'after filter, total {valid_obj_cnt} objs')

    annotation_dataset['total_objs'] = valid_obj_cnt
    annotation_dataset['obj_id_start'] = obj_id_base
    annotation_all['dancetrack'] = annotation_dataset

    return obj_id_base + valid_obj_cnt


def gen_femot(annotation_all: dict, split: str = 'train', obj_id_base: int = 0,
              sequences: List[str] = None, filter: bool = True, filter_len: int = 25) -> int:

    print('parsing dataset FEMOT')

    annotation_dataset = dict()

    split_dir = os.path.join(FEMOT_PATH, split)
    seqs = sorted([
        s for s in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, s))
    ])

    if sequences is None:
        sequences = ['all']
    if 'all' in sequences:
        sequences = seqs

    obj_id_start = obj_id_base
    obj_cnt, valid_obj_cnt = 0, 0

    for seq in seqs:

        if seq not in sequences:
            continue

        print(f'parsing {seq}, obj id base {obj_id_start}')

        config = configparser.ConfigParser()
        seq_info = os.path.join(split_dir, seq, 'seqinfo.ini')
        config.read(seq_info)
        image_w = int(config['Sequence']['imWidth'])
        image_h = int(config['Sequence']['imHeight'])

        gt_file_path = os.path.join(split_dir, seq, 'gt', 'gt.txt')
        seq_anno = np.loadtxt(gt_file_path, dtype=float, delimiter=',')

        obj_cnt_in_seq = np.unique(seq_anno[:, 1]).shape[0]
        valid_cnt_in_seq = 0

        for row_gt in seq_anno:

            if int(row_gt[6]) != 1 or int(row_gt[7]) not in VALID_CATEGORIES['femot']:
                continue

            obj_id = int(row_gt[1]) + obj_id_start
            bbox = [float(row_gt[i]) for i in range(2, 6)]
            bbox[0] += 0.5 * bbox[2]
            bbox[1] += 0.5 * bbox[3]
            bbox[0] = round(bbox[0] / float(image_w), 6)
            bbox[1] = round(bbox[1] / float(image_h), 6)
            bbox[2] = round(bbox[2] / float(image_w), 6)
            bbox[3] = round(bbox[3] / float(image_h), 6)

            if obj_id not in annotation_dataset.keys():
                valid_cnt_in_seq += 1
                annotation_dataset[obj_id] = {
                    'image_h': image_h,
                    'image_w': image_w,
                    'traj_len': 1,
                    'bboxes': [bbox]
                }
            else:
                annotation_dataset[obj_id]['traj_len'] += 1
                annotation_dataset[obj_id]['bboxes'].append(bbox)

        print(f'in seq {seq}, total {obj_cnt_in_seq} objs, valid {valid_cnt_in_seq} objs')

        obj_cnt += obj_cnt_in_seq
        valid_obj_cnt += valid_cnt_in_seq
        obj_id_start += obj_cnt_in_seq

    if filter:
        print(f'before filter, total {valid_obj_cnt} objects')
        all_objs = list(annotation_dataset.keys())

        annotation_dataset_filtered = dict()
        new_obj_id = 0

        for obj_id in all_objs:

            if annotation_dataset[obj_id]['traj_len'] < filter_len:
                valid_obj_cnt -= 1
            else:
                annotation_dataset_filtered[str(new_obj_id + obj_id_base)] = annotation_dataset.pop(obj_id, None)
                new_obj_id += 1

        assert new_obj_id == valid_obj_cnt, f'{new_obj_id}, {valid_obj_cnt}'

        annotation_dataset = annotation_dataset_filtered

        print(f'after filter, total {valid_obj_cnt} objs')

    annotation_dataset['total_objs'] = valid_obj_cnt
    annotation_dataset['obj_id_start'] = obj_id_base
    annotation_all['femot'] = annotation_dataset

    return obj_id_base + valid_obj_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gen traj annotation')

    parser.add_argument('--mot17', action='store_true')
    parser.add_argument('--visdrone', action='store_true')
    parser.add_argument('--dancetrack', action='store_true')
    parser.add_argument('--femot', action='store_true')
    parser.add_argument('--save_name', type=str, default='mot17')
    args = parser.parse_args()

    annotation_all = dict()

    obj_id_base = 0

    if args.mot17:
        obj_id_base = gen_mot17(annotation_all=annotation_all, split='train', obj_id_base=obj_id_base, sequences=['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP'])
    if args.visdrone:
        obj_id_base = gen_visdrone(annotation_all=annotation_all, sequences=['all'], obj_id_base=obj_id_base)
    if args.dancetrack:
        obj_id_base = gen_dancetrack(annotation_all=annotation_all, split='train', obj_id_base=obj_id_base, sequences=['all'])
    if args.femot:
        obj_id_base = gen_femot(annotation_all=annotation_all, split='train', obj_id_base=obj_id_base, sequences=['all'])

    # save 
    output_file = os.path.join('./ssm_tracker/traj_anno_data', f'{args.save_name}.json')
    
    with open(output_file, 'w') as f:
        json.dump(annotation_all, f)

    # python tools/gen_traj_data.py --dancetrack --save_name dancetrack
    # python tools/gen_traj_data.py --femot --save_name femot