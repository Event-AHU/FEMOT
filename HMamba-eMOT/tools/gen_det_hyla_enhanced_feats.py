#!/usr/bin/env python3
"""Replace detection rgb/evt RoI features with HyLA-enhanced dark3 features.

Detection boxes stay the ones already written in the source det directory.
Appearance ``*_feat.npz`` and MOT txt files are symlinked, not recomputed.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from loguru import logger
from torchvision.ops import roi_align


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "ByteTrack"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yolox_exps", "custom"))

from dual_data import dual_preproc  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "det_results/femot_dual_hyla_shared_mlp_coco_ep27_feats/test",
        ),
    )
    parser.add_argument(
        "--dst_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "det_results/femot_dual_hyla_shared_mlp_coco_ep27_hyla_feats/test",
        ),
    )
    parser.add_argument(
        "--data_root",
        default="/rydata/wangx_dataset/_dataset/FEMOT/test",
    )
    parser.add_argument(
        "--exp_file",
        default=os.path.join(
            PROJECT_ROOT, "yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py"
        ),
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(
            PROJECT_ROOT,
            "third_party/ByteTrack/YOLOX_outputs/"
            "yolox_femot_dual_hyla_shared_mlp_coco/"
            "best_ckpt_ep27_ap57.5_freeze.pth.tar",
        ),
    )
    parser.add_argument("--img_size", nargs=2, type=int, default=[416, 416])
    parser.add_argument("--device", default="0")
    parser.add_argument("--pool_size", type=int, default=7)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
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


def link_unchanged(src_dir, dst_dir, seq):
    os.makedirs(dst_dir, exist_ok=True)
    for name in (f"{seq}.txt", f"{seq}_feat.npz"):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)


def process_sequence(args, model, device, seq):
    det_path = os.path.join(args.src_dir, f"{seq}.txt")
    det = np.loadtxt(det_path, delimiter=",", dtype=np.float32)
    if det.ndim == 1:
        det = det[None, :]
    aps_dir = os.path.join(args.data_root, seq, f"{seq}_aps")
    dvs_dir = os.path.join(args.data_root, seq, f"{seq}_dvs")
    rgb_out, evt_out = {}, {}
    frame_ids = sorted(np.unique(det[:, 0].astype(np.int64)).tolist())
    for frame_id in frame_ids:
        rows = det[det[:, 0].astype(np.int64) == frame_id]
        rgb_path = os.path.join(aps_dir, f"{frame_id:05d}.jpg")
        evt_path = os.path.join(dvs_dir, f"{frame_id:05d}.jpg")
        rgb = cv2.imread(rgb_path)
        evt = cv2.imread(evt_path)
        if rgb is None or evt is None:
            raise FileNotFoundError(f"{seq} frame {frame_id}: {rgb_path}")
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
            else:
                out_rgb, out_evt = model.backbone.forward_hyla_enhanced_features(image)
            rgb_out[f"f{frame_id}"] = roi_features(
                out_rgb["dark3"], rows[:, 2:6], ratio, args.pool_size
            ).astype(np.float16)
            evt_out[f"f{frame_id}"] = roi_features(
                out_evt["dark3"], rows[:, 2:6], ratio, args.pool_size
            ).astype(np.float16)
        if rgb_out[f"f{frame_id}"].shape[0] != len(rows):
            raise RuntimeError(f"{seq} frame {frame_id}: feature count != detections")

    os.makedirs(args.dst_dir, exist_ok=True)
    rgb_path = os.path.join(args.dst_dir, f"{seq}_rgb_feat.npz")
    evt_path = os.path.join(args.dst_dir, f"{seq}_evt_feat.npz")
    np.savez_compressed(rgb_path, **rgb_out)
    np.savez_compressed(evt_path, **evt_out)
    link_unchanged(args.src_dir, args.dst_dir, seq)
    logger.info(f"saved enhanced feats for {seq} ({len(frame_ids)} frames)")


def main(args):
    device = select_device(args.device)
    exp = get_exp(args.exp_file, "femot_dual")
    model = exp.get_model().to(device).eval()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    logger.info(f"loaded {args.model_path}")

    sequences = sorted(
        name[:-4]
        for name in os.listdir(args.src_dir)
        if name.endswith(".txt") and name != "meta_data.txt"
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    sequences = sequences[args.shard_index :: args.num_shards]
    os.makedirs(args.dst_dir, exist_ok=True)
    meta_src = os.path.join(args.src_dir, "meta_data.txt")
    meta_dst = os.path.join(args.dst_dir, "meta_data.txt")
    if os.path.isfile(meta_src) and not os.path.lexists(meta_dst):
        os.symlink(meta_src, meta_dst)
    for seq in sequences:
        process_sequence(args, model, device, seq)


if __name__ == "__main__":
    main(parse_args())
