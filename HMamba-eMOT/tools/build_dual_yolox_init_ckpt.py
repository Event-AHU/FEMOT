"""Map single-modal YOLOX checkpoint to dual-modal (sum-fusion) backbone."""

import argparse
import os

import torch


def expand_single_backbone_ckpt(single_state):
    """backbone.backbone.* -> backbone_rgb + backbone_evt; neck/head unchanged."""
    dual_state = {}
    prefix = "backbone.backbone."
    for key, value in single_state.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            dual_state[f"backbone.backbone_rgb.{suffix}"] = value
            dual_state[f"backbone.backbone_evt.{suffix}"] = value.clone()
        else:
            dual_state[key] = value
    return dual_state


def build_dual_init_ckpt(src_ckpt, dst_ckpt):
    raw = torch.load(src_ckpt, map_location="cpu")
    single = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    dual = expand_single_backbone_ckpt(single)
    os.makedirs(os.path.dirname(dst_ckpt) or ".", exist_ok=True)
    torch.save({"model": dual, "start_epoch": 0}, dst_ckpt)
    return len(single), len(dual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default="weights/yolox_femot_rgb_best.pth.tar",
        help="single-modal FEMOT RGB checkpoint",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="weights/yolox_femot_dual_init.pth.tar",
        help="dual-modal init checkpoint for training",
    )
    args = parser.parse_args()
    n_src, n_dst = build_dual_init_ckpt(args.src, args.dst)
    print(f"saved {args.dst} ({n_src} -> {n_dst} tensors)")


if __name__ == "__main__":
    main()
