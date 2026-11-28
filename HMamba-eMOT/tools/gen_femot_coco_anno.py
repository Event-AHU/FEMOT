"""
Generate FEMOT COCO annotations using RGB (aps) frames instead of fusion.

Output: {FEMOT_ROOT}/annotations/{split}_rgb.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.dataset_paths import FEMOT_ROOT


def convert_split(data_root: str, split: str, out_dir: str) -> None:
    data_path = os.path.join(data_root, split)
    out_path = os.path.join(out_dir, f'{split}_rgb.json')
    out = {
        'images': [],
        'annotations': [],
        'videos': [],
        'categories': [
            {'id': 1, 'name': 'pedestrian'},
            {'id': 3, 'name': 'car'},
        ],
    }

    seqs = sorted([
        s for s in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, s))
    ])

    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0

    for seq in tqdm(seqs, desc=f'{split}'):
        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'file_name': seq})

        rgb_dir = f'{seq}_aps'
        img_path = os.path.join(data_path, seq, rgb_dir)
        ann_path = os.path.join(data_path, seq, 'gt', 'gt.txt')

        images = sorted([
            f for f in os.listdir(img_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        num_images = len(images)

        for i, image_name in enumerate(images):
            frame_id = i + 1
            img = cv2.imread(os.path.join(img_path, image_name))
            if img is None:
                raise FileNotFoundError(os.path.join(img_path, image_name))
            height, width = img.shape[:2]
            out['images'].append({
                'file_name': f'{seq}/{rgb_dir}/{image_name}',
                'id': image_cnt + frame_id,
                'frame_id': frame_id,
                'prev_image_id': image_cnt + frame_id - 1 if frame_id > 1 else -1,
                'next_image_id': image_cnt + frame_id + 1 if frame_id < num_images else -1,
                'video_id': video_cnt,
                'height': height,
                'width': width,
            })

        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        if anns.ndim == 1:
            anns = anns[None, :]

        for row in anns:
            frame_id = int(row[0])
            if frame_id < 1 or frame_id > num_images:
                continue
            if int(row[6]) != 1:
                continue

            tid_curr += 1
            ann_cnt += 1
            out['annotations'].append({
                'id': ann_cnt,
                'category_id': int(row[7]),
                'image_id': image_cnt + frame_id,
                'track_id': tid_curr,
                'bbox': row[2:6].tolist(),
                'conf': float(row[6]),
                'iscrowd': 0,
                'area': float(row[4] * row[5]),
            })

        image_cnt += num_images

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)

    print(f'{split}: {len(out["images"])} images, {len(out["annotations"])} annotations -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=FEMOT_ROOT)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, 'annotations')
    for split in args.splits:
        convert_split(args.data_root, split, out_dir)


if __name__ == '__main__':
    main()
