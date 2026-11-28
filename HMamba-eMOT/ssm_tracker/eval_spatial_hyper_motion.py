#!/usr/bin/env python3
"""Evaluate frame-centric motion prediction with or without spatial hypergraph."""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import FrameTrajDataset, frame_hypergraph_collate
from models.MambaTrack import MambaTrack
from train_utils.envs import select_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="4")
    parser.add_argument("--disable_hyper", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main(args):
    with open(args.config_file) as file:
        cfg = yaml.safe_load(file)
    model_cfg = dict(cfg["train"])
    uses_spatial = model_cfg.get("use_spatial_hyper", False) or model_cfg.get(
        "use_spatial_ssm_mod", False
    )
    if args.disable_hyper:
        model_cfg["use_spatial_hyper"] = False
        model_cfg["use_spatial_ssm_mod"] = False
        uses_spatial = False

    device = select_device(args.device)
    dataset = FrameTrajDataset(cfg["dataset"], split="val")
    loader = DataLoader(
        dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        collate_fn=frame_hypergraph_collate,
    )
    model = MambaTrack(model_cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=not args.disable_hyper)

    absolute_error = 0.0
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for data in loader:
            x = data["condition"].to(device)
            mask = data["vertex_mask"].to(device)
            hyper_inputs = None
            if uses_spatial:
                hyper_inputs = {
                    relation: data[f"H_{relation}"].to(device)
                    for relation in ("spatial", "rgb", "evt", "motion")
                }
            pred = model(
                x,
                hyper_inputs=hyper_inputs,
                vertex_mask=mask,
            ).squeeze(2)
            label = data["label"].to(device)
            error = pred[mask] - label[mask]
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            count += error.numel()

    result = {
        "checkpoint": args.checkpoint,
        "spatial_hyper": bool(model_cfg.get("use_spatial_hyper", False)),
        "spatial_ssm_mod": bool(model_cfg.get("use_spatial_ssm_mod", False)),
        "elements": count,
        "mae_scaled": absolute_error / max(count, 1),
        "rmse_scaled": (squared_error / max(count, 1)) ** 0.5,
        "mae_normalized": absolute_error / max(count, 1) / cfg["dataset"]["scale_factor"],
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main(parse_args())
