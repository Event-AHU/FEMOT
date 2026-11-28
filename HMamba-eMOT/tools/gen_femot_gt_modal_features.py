#!/usr/bin/env python3
"""Extract modality-specific GT RoI features for frame-level hypergraphs."""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from torchvision.ops import roi_align


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "ByteTrack"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yolox_exps", "custom"))

from dual_data import dual_preproc  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/rydata/wangx_dataset/_dataset/FEMOT")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument(
        "--exp_file",
        default=os.path.join(PROJECT_ROOT, "yolox_exps/custom/yolox_femot_dual.py"),
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_ROOT, "weights/yolox_femot_dual_best.pth.tar"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "data/femot_modal_gt"),
    )
    parser.add_argument("--img_size", nargs=2, type=int, default=[416, 416])
    parser.add_argument("--device", default="0")
    parser.add_argument("--pool_size", type=int, default=7)
    parser.add_argument("--max_seqs", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="RoI-pool dark3 after shared MLP + HyLA, before gate-sum fusion",
    )
    parser.add_argument(
        "--fused_rgb",
        action="store_true",
        help="rgb RoI is dark3 gate-sum fusion; evt RoI stays pre-gate enhanced",
    )
    return parser.parse_args()


def select_device(device):
    if device != "cpu" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    use_cuda = device != "cpu" and torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def load_gt(path):
    gt = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if gt.ndim == 1:
        gt = gt[None, :]
    # MOT: frame, id, x, y, w, h, valid, class, visibility
    if gt.shape[1] >= 7:
        gt = gt[gt[:, 6] > 0]
    return gt


def image_map(folder):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(valid_ext))
    mapping = {}
    for index, name in enumerate(names, start=1):
        stem = os.path.splitext(name)[0]
        try:
            frame_id = int(stem)
        except ValueError:
            frame_id = index
        mapping[frame_id] = os.path.join(folder, name)
    return mapping


def roi_features(feature_map, tlwh, ratio, pool_size):
    if len(tlwh) == 0:
        return np.zeros((0, feature_map.shape[1]), dtype=np.float32)
    boxes = np.asarray(tlwh, dtype=np.float32).copy()
    boxes[:, 2:] += boxes[:, :2]
    boxes *= ratio
    rois = np.concatenate(
        [np.zeros((len(boxes), 1), dtype=np.float32), boxes], axis=1
    )
    rois = torch.from_numpy(rois).to(feature_map.device)
    pooled = roi_align(
        feature_map,
        rois,
        output_size=(pool_size, pool_size),
        spatial_scale=1.0 / 8.0,
        aligned=True,
    )
    features = pooled.mean(dim=(2, 3))
    features = torch.nn.functional.normalize(features, dim=1)
    return features.float().cpu().numpy()


def process_sequence(args, model, device, seq):
    seq_dir = os.path.join(args.data_root, args.split, seq)
    aps_dir = os.path.join(seq_dir, f"{seq}_aps")
    dvs_dir = os.path.join(seq_dir, f"{seq}_dvs")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if not (os.path.isdir(aps_dir) and os.path.isdir(dvs_dir) and os.path.isfile(gt_path)):
        logger.warning(f"skip incomplete sequence {seq}")
        return

    aps_images = image_map(aps_dir)
    dvs_images = image_map(dvs_dir)
    gt = load_gt(gt_path)

    frame_ids, object_ids, boxes = [], [], []
    rgb_features, evt_features = [], []
    img_hw = None

    gt_frames = sorted(np.unique(gt[:, 0].astype(np.int64)).tolist())
    for frame_id in tqdm(gt_frames, desc=seq, leave=False):
        if frame_id not in aps_images or frame_id not in dvs_images:
            logger.warning(f"{seq}: missing image for frame {frame_id}")
            continue
        rows = gt[gt[:, 0].astype(np.int64) == frame_id]
        rgb = cv2.imread(aps_images[frame_id])
        evt = cv2.imread(dvs_images[frame_id])
        if rgb is None or evt is None:
            logger.warning(f"{seq}: unreadable image for frame {frame_id}")
            continue
        img_hw = np.asarray(rgb.shape[:2], dtype=np.int32)
        image, ratio = dual_preproc(
            rgb,
            evt,
            args.img_size,
            rgb_means=(0.485, 0.456, 0.406),
            rgb_std=(0.229, 0.224, 0.225),
        )
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)
        with torch.no_grad():
            if args.fused_rgb:
                out_rgb, out_evt = model.backbone.forward_hyla_fused_rgb_features(image)
            elif args.enhanced:
                out_rgb, out_evt = model.backbone.forward_hyla_enhanced_features(image)
            else:
                out_rgb, out_evt = model.backbone.forward_modal_features(image)
            frame_rgb = roi_features(out_rgb["dark3"], rows[:, 2:6], ratio, args.pool_size)
            frame_evt = roi_features(out_evt["dark3"], rows[:, 2:6], ratio, args.pool_size)

        frame_ids.append(np.full(len(rows), frame_id, dtype=np.int32))
        object_ids.append(rows[:, 1].astype(np.int32))
        boxes.append(rows[:, 2:6].astype(np.float32))
        rgb_features.append(frame_rgb.astype(np.float16))
        evt_features.append(frame_evt.astype(np.float16))

    if not frame_ids:
        logger.warning(f"{seq}: no valid frames")
        return
    output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{seq}.npz")
    np.savez_compressed(
        output_path,
        frame_ids=np.concatenate(frame_ids),
        object_ids=np.concatenate(object_ids),
        boxes_tlwh=np.concatenate(boxes),
        rgb_feats=np.concatenate(rgb_features),
        evt_feats=np.concatenate(evt_features),
        image_hw=img_hw,
    )
    logger.info(f"saved {output_path}")


def main(args):
    if args.fused_rgb and args.enhanced:
        raise SystemExit("use only one of --fused_rgb and --enhanced")
    device = select_device(args.device)
    exp = get_exp(args.exp_file, "femot_dual")
    model = exp.get_model().to(device).eval()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    split_root = os.path.join(args.data_root, args.split)
    sequences = sorted(
        name for name in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, name))
        and os.path.isfile(os.path.join(split_root, name, "gt", "gt.txt"))
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    sequences = sequences[args.shard_index :: args.num_shards]
    if args.max_seqs > 0:
        sequences = sequences[: args.max_seqs]
    for seq in sequences:
        process_sequence(args, model, device, seq)


if __name__ == "__main__":
    main(parse_args())
