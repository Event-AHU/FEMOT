"""Train PlugTrack on the frozen DSEC baseline epoch-3 motion predictor."""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset import TrajDatasetv2
from models.MambaTrack import MambaTrack
from models.plugtrack.fusion import PlugTrack
from train_utils.envs import select_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def displacement(out):
    if out.ndim == 4 and out.shape[2] == 1:
        out = out.squeeze(2)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    if out.ndim != 2 or out.shape[-1] != 4:
        raise RuntimeError('unexpected Mamba output shape %s' % (tuple(out.shape),))
    return out


def precompute(mamba, loader, device, scale):
    histories = []
    motions = []
    targets = []
    mamba.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            traj = batch['traj'].to(device).float()
            condition = batch['condition'].to(device).float()
            diffs = traj[:, 1:11] - traj[:, :10]
            if not torch.allclose(condition, diffs * scale, atol=1e-5):
                raise RuntimeError('condition is not the first 10 scaled displacements')
            history = torch.cat([traj[:, 1:11], diffs], dim=-1)
            pred = displacement(mamba(condition)) / scale
            bbox_mp = traj[:, 10] + pred
            bbox_gt = traj[:, 11]
            histories.append(history.cpu())
            motions.append(bbox_mp.cpu())
            targets.append(bbox_gt.cpu())
            if step % 50 == 0:
                logger.info('precompute step {}', step)
    return (
        torch.cat(histories, dim=0),
        torch.cat(motions, dim=0),
        torch.cat(targets, dim=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        default='cfgs/dsec/dsec_baseline_ep3_plugtrack.yaml',
    )
    parser.add_argument('--save_path', default='/wangx_new/lyt/dsec_mamba')
    parser.add_argument('--exp_name', default='dsec_baseline_ep3_plugtrack')
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.config_file, 'r') as handle:
        cfgs = yaml.safe_load(handle)
    dataset_cfgs = cfgs['dataset']
    train_cfgs = cfgs['train']
    save_dir = os.path.join(args.save_path, args.exp_name)
    if os.path.isdir(save_dir) and any(name.endswith('.pth') for name in os.listdir(save_dir)):
        raise SystemExit('refusing to overwrite ' + save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.system('cp %s %s' % (args.config_file, save_dir))

    device = select_device(args.device)
    dataset = TrajDatasetv2(dataset_cfgs)
    logger.info('train windows {}', len(dataset))
    mamba = MambaTrack(train_cfgs).to(device)
    ckpt = torch.load(train_cfgs['mamba_ckpt'], map_location=device)
    mamba.load_state_dict(ckpt['model'])
    mamba.eval()
    for param in mamba.parameters():
        param.requires_grad_(False)
    logger.info('froze Mamba from {} epoch {}', train_cfgs['mamba_ckpt'], ckpt.get('epoch'))

    source = DataLoader(
        dataset,
        batch_size=int(train_cfgs['mamba_batch_size']),
        shuffle=False,
        num_workers=0,
    )
    history, bbox_mp, bbox_gt = precompute(
        mamba, source, device, float(dataset_cfgs['scale_factor'])
    )
    del mamba
    torch.cuda.empty_cache()
    logger.info(
        'cached history {} bbox_mp {} bbox_gt {}',
        tuple(history.shape), tuple(bbox_mp.shape), tuple(bbox_gt.shape),
    )

    plugin = PlugTrack(
        process_var=float(train_cfgs['plugtrack_process_var']),
        meas_var=float(train_cfgs['plugtrack_meas_var']),
    ).to(device)
    trainable = sum(param.numel() for param in plugin.parameters() if param.requires_grad)
    logger.info('PlugTrack trainable params {}', trainable)
    optimizer = torch.optim.Adam(plugin.parameters(), lr=float(train_cfgs['lr0']))
    loader = DataLoader(
        TensorDataset(history, bbox_mp, bbox_gt),
        batch_size=int(train_cfgs['batch_size']),
        shuffle=True,
        drop_last=False,
    )
    epochs = int(train_cfgs['epochs'])
    for epoch in range(1, epochs + 1):
        plugin.train()
        total = 0.0
        blend = 0.0
        alpha = 0.0
        seen = 0
        for step, (hist, motion, target) in enumerate(loader):
            if hist.shape[0] < 2:
                continue
            hist = hist.to(device)
            motion = motion.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            _, _, _, loss_final, predictor_loss = plugin(hist, motion, target)
            loss = loss_final + predictor_loss
            loss.backward()
            optimizer.step()
            count = hist.shape[0]
            total += float(loss.item()) * count
            blend += float(loss_final.item()) * count
            alpha += float(predictor_loss.item()) * count
            seen += count
            if step % 20 == 0:
                logger.info(
                    'epoch {} step {} loss {:.6f} blend {:.6f} alpha {:.6f}',
                    epoch, step, float(loss.item()), float(loss_final.item()),
                    float(predictor_loss.item()),
                )
        logger.info(
            'epoch {} mean loss {:.6f} blend {:.6f} alpha {:.6f} samples {}',
            epoch, total / seen, blend / seen, alpha / seen, seen,
        )
        path = os.path.join(save_dir, 'epoch%d.pth' % epoch)
        torch.save(
            {
                'epoch': epoch,
                'model': plugin.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mamba_ckpt': train_cfgs['mamba_ckpt'],
            },
            path,
        )
        logger.info('saved {}', path)


if __name__ == '__main__':
    main()
