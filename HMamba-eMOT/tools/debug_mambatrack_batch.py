"""
Load one batch and run forward+backward once — useful for debugging shapes/loss.
Set breakpoints in MambaTrack.py or here, then F5 launch "single batch smoke test".
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ssm_tracker'))
from dataset.dataset import TrajDatasetv2
from models.MambaTrack import MambaTrack
from train_utils.envs import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config_path = args.config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

    with open(config_path, 'r') as f:
        cfgs = yaml.safe_load(f)

    dataset = TrajDatasetv2(cfgs['dataset'])
    loader = DataLoader(dataset, batch_size=cfgs['train']['batch_size'], shuffle=True)
    data = next(iter(loader))

    device = select_device(args.device)
    model = MambaTrack(cfgs['train']).to(device)
    model.train()

    x = data['condition'].to(device)
    label = data['label'].to(device)

    print('condition', x.shape, x.dtype)
    print('label', label.shape, label.dtype)

    loss = model(x, label)
    print('loss', loss.item())

    loss.backward()
    print('backward ok')


if __name__ == '__main__':
    main()
