#!/usr/bin/env python

import argparse
import os
import sys
import time

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BYTE_TRACK = os.path.join(ROOT, "third_party", "ByteTrack")
CUSTOM_EXPS = os.path.join(ROOT, "yolox_exps", "custom")
sys.path.insert(0, BYTE_TRACK)
sys.path.insert(0, CUSTOM_EXPS)

from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Benchmark dual-modal fusion variants")
    parser.add_argument(
        "--exp-file",
        default=os.path.join(CUSTOM_EXPS, "yolox_femot_dual_hyla_mfg.py"),
    )
    parser.add_argument(
        "--fusion-types", default="sum,mfg,hyla,hyla_mfg",
        help="Comma-separated ablation variants",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_variant(args, fusion_type, device):
    exp = get_exp(args.exp_file, None)
    exp.fusion_type = fusion_type
    exp.model = None
    model = exp.get_model().eval().to(device)
    sample = torch.randn(
        args.batch_size, 6, args.size, args.size, device=device
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    fusion_params = sum(
        parameter.numel()
        for parameter in model.backbone.fusion_blocks.parameters()
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                model(sample)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(args.iterations):
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                model(sample)
        synchronize(device)
        elapsed = time.perf_counter() - start

    images_per_second = args.batch_size * args.iterations / elapsed
    latency_ms = elapsed * 1000.0 / args.iterations
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else float("nan")
    )
    print(
        f"{fusion_type}: params={total_params / 1e6:.3f}M, "
        f"fusion_params={fusion_params / 1e6:.3f}M, "
        f"latency={latency_ms:.3f}ms/batch, "
        f"throughput={images_per_second:.3f}img/s, "
        f"peak_memory={peak_memory_mb:.1f}MiB"
    )

    del sample, model, exp
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for fusion_type in args.fusion_types.split(","):
        benchmark_variant(args, fusion_type.strip(), device)


if __name__ == "__main__":
    main()
