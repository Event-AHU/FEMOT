#!/usr/bin/env python3
"""Standalone YOLOX detection COCO-AP evaluation (no tracking).

Examples:
  # Single checkpoint on FEMOT val
  python tools/eval_yolox_det.py \\
    --model early_hyla \\
      yolox_exps/custom/yolox_femot_dual_early_hyla.py \\
      third_party/ByteTrack/YOLOX_outputs/yolox_femot_dual_early_hyla/best_ckpt.pth.tar \\
    --device 0 --fp16

  # Compare baseline vs early_hyla on FEMOT test
  python tools/eval_yolox_det.py \\
    --model baseline \\
      yolox_exps/custom/yolox_femot_dual.py \\
      third_party/ByteTrack/YOLOX_outputs/yolox_femot_dual/best_ckpt.pth.tar \\
    --model early_hyla_ep10 \\
      yolox_exps/custom/yolox_femot_dual_early_hyla.py \\
      third_party/ByteTrack/YOLOX_outputs/yolox_femot_dual_early_hyla/best_ckpt_ep10_ap862_freeze.pth.tar \\
    --split test --device 7 --fp16 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "ByteTrack"))
sys.path.insert(0, str(_PROJECT_ROOT / "yolox_exps" / "custom"))

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX detection COCO AP eval")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "EXP_FILE", "CKPT"),
        required=True,
        help="Repeatable: display_name exp_file ckpt_path",
    )
    parser.add_argument("--device", default="0", help="CUDA device id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true", help="Fuse conv-bn before eval")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="FEMOT split for COCO AP (val_rgb.json / test_rgb.json)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override exp.test_conf (default: use exp)",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=None,
        help="Override exp.nmsthre (default: use exp)",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra exp overrides forwarded to exp.merge",
    )
    return parser.parse_args()


def load_model(exp, ckpt_path: Path, device: torch.device, fuse: bool):
    model = exp.get_model()
    logger.info("Model summary: {}".format(get_model_info(model, exp.test_size)))
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    if fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)
    return model


def eval_one(name: str, exp_file: str, ckpt: str, args) -> dict:
    exp_path = Path(exp_file)
    if not exp_path.is_absolute():
        exp_path = _PROJECT_ROOT / exp_path
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path
    if not exp_path.is_file():
        raise FileNotFoundError(exp_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    logger.info("=" * 72)
    logger.info("Evaluating [{}] on FEMOT {}", name, args.split)
    logger.info("  exp : {}", exp_path)
    logger.info("  ckpt: {}", ckpt_path)

    exp = get_exp(str(exp_path), None)
    if args.opts:
        exp.merge(args.opts)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    use_test = args.split == "test"
    model = load_model(exp, ckpt_path, device, fuse=args.fuse)
    evaluator = exp.get_evaluator(
        args.batch_size, is_distributed=False, testdev=use_test
    )
    ap50_95, ap50, summary = exp.eval(model, evaluator, is_distributed=False, half=args.fp16)

    logger.info("[{}] split={} AP50:95={:.4f}  AP50={:.4f}", name, args.split, ap50_95, ap50)
    print(summary)
    return {
        "name": name,
        "ckpt": str(ckpt_path),
        "split": args.split,
        "ap50_95": float(ap50_95),
        "ap50": float(ap50),
        "summary": summary,
    }


def print_comparison(results: list[dict], split: str = "val"):
    if len(results) < 2:
        return
    print("\n" + "=" * 72)
    print(f"Detection COCO AP comparison (FEMOT {split})")
    print("=" * 72)
    print(f"{'name':<24} {'AP50:95':>10} {'AP50':>10}")
    print("-" * 72)
    base = results[0]
    for r in results:
        d95 = r["ap50_95"] - base["ap50_95"]
        d50 = r["ap50"] - base["ap50"]
        delta = ""
        if r is not base:
            delta = f"  (Δ95={d95:+.4f}, Δ50={d50:+.4f} vs {base['name']})"
        print(f"{r['name']:<24} {r['ap50_95']:10.4f} {r['ap50']:10.4f}{delta}")
    print("=" * 72)


def main():
    args = parse_args()

    results = []
    for name, exp_file, ckpt in args.model:
        # Free GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(eval_one(name, exp_file, ckpt, args))

    print_comparison(results, split=args.split)
    return results


if __name__ == "__main__":
    main()
