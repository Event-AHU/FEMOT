"""Train the baseline motion model with SmoothL1 plus the UniTrack loss."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.unitrack_clips import UniTrackClipDataset
from models.MambaTrack import MambaTrack
from models.unitrack_criterion import Unitrackrion
from train_utils.envs import select_device
from train_utils.lr_scheduler import NoneLRScheduler, TransformerLRScheduler


def squeeze_pred(pred):
    if pred.ndim == 3 and pred.shape[1] == 1:
        return pred.squeeze(1)
    if pred.ndim == 4 and pred.shape[2] == 1:
        return pred.squeeze(2).squeeze(1)
    if pred.ndim == 2:
        return pred
    raise RuntimeError('unexpected prediction shape %s' % (tuple(pred.shape),))


def norm_to_xyxy(boxes, width, height):
    cx = boxes[..., 0] * width
    cy = boxes[..., 1] * height
    w = boxes[..., 2] * width
    h = boxes[..., 3] * height
    out = torch.empty_like(boxes)
    out[..., 0] = cx - 0.5 * w
    out[..., 1] = cy - 0.5 * h
    out[..., 2] = cx + 0.5 * w
    out[..., 3] = cy + 0.5 * h
    return out


def criterion_for(cache, width, height, iou_threshold):
    key = (int(width), int(height), float(iou_threshold))
    if key not in cache:
        cache[key] = Unitrackrion(img_size=(int(width), int(height)), iou_threshold=float(iou_threshold))
    return cache[key]


def call_unitrack(criterion, pred_boxes, gt_boxes, track_ids):
    outputs = {
        'pred_boxes': pred_boxes,
        'pred_logits': pred_boxes.new_zeros(*pred_boxes.shape[:-1], 1),
        'track_ids': track_ids,
    }
    targets = []
    for index in range(pred_boxes.shape[0]):
        targets.append({
            'boxes': gt_boxes[index],
            'labels': gt_boxes.new_zeros(gt_boxes.shape[1]),
            'track_ids': track_ids[index],
        })
    return criterion(outputs, targets)


def mean_components(parts):
    keys = (
        'loss_unitrack',
        'loss_unitrack_tracking',
        'loss_unitrack_spatial',
        'loss_unitrack_temporal',
    )
    counts = [int(count) for _, count in parts]
    total = float(sum(counts))
    out = {}
    for key in keys:
        value = parts[0][0][key] * 0.0
        for loss, count in parts:
            value = value + loss[key] * (count / total)
        out[key] = value
    return out


def frame_loss(cache, samples, pred_norm, iou_threshold):
    offset = 0
    parts = []
    for sample in samples:
        count = int(sample['label'].shape[0])
        pred = pred_norm[offset:offset + count]
        offset += count
        width = float(sample['width'])
        height = float(sample['height'])
        pred_xyxy = norm_to_xyxy(pred, width, height).unsqueeze(0)
        gt_xyxy = norm_to_xyxy(sample['gt_box'], width, height).unsqueeze(0)
        track_ids = sample['track_id'].unsqueeze(0)
        criterion = criterion_for(cache, width, height, iou_threshold).to(pred.device)
        parts.append((
            call_unitrack(criterion, pred_xyxy, gt_xyxy, track_ids),
            count,
        ))
    return mean_components(parts)


def temporal_loss(cache, samples, pred_norm, iou_threshold):
    offset = 0
    grouped = {}
    for sample in samples:
        count = int(sample['temporal_condition'].shape[0])
        if count == 0:
            continue
        steps = sample['temporal_condition'].shape[1]
        pred = pred_norm[offset:offset + count * steps].reshape(count, steps, 4)
        offset += count * steps
        key = (int(sample['width']), int(sample['height']))
        grouped.setdefault(key, []).append((sample, pred))
    if offset != pred_norm.shape[0]:
        raise RuntimeError('temporal prediction count does not match windows')
    if not grouped:
        zero = pred_norm.new_zeros(())
        return {
            'loss_unitrack': zero,
            'loss_unitrack_tracking': zero,
            'loss_unitrack_spatial': zero,
            'loss_unitrack_temporal': zero,
        }
    parts = []
    for (width, height), items in grouped.items():
        preds, gts, ids = [], [], []
        for sample, pred in items:
            prev = sample['temporal_prev']
            pred_box = prev + pred / sample.get('scale_factor', 1.0)
            preds.append(norm_to_xyxy(pred_box, width, height))
            gts.append(norm_to_xyxy(sample['temporal_gt'], width, height))
            ids.append(torch.ones(pred.shape[0], pred.shape[1], device=pred.device))
        pred_boxes = torch.cat(preds, dim=0)
        gt_boxes = torch.cat(gts, dim=0)
        track_ids = torch.cat(ids, dim=0)
        criterion = criterion_for(cache, width, height, iou_threshold).to(pred_boxes.device)
        parts.append((
            call_unitrack(criterion, pred_boxes, gt_boxes, track_ids),
            int(pred_boxes.shape[0]),
        ))
    return mean_components(parts)


def train(args):
    with open(args.config_file, 'r') as handle:
        cfgs = yaml.safe_load(handle)
    dataset_cfgs = cfgs['dataset']
    train_cfgs = cfgs['train']
    scale_factor = float(dataset_cfgs['scale_factor'])
    weight = float(train_cfgs.get('unitrack_weight', 0.05))
    iou_threshold = float(train_cfgs.get('unitrack_iou_threshold', 0.5))

    save_dir = os.path.join(args.save_path, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    if any(name.startswith('epoch') and name.endswith('.pth') for name in os.listdir(save_dir)):
        raise SystemExit('refusing to overwrite ' + save_dir)
    os.system('cp %s %s' % (args.config_file, save_dir))

    logger.info('Loading frame-aligned clips')
    dataset = UniTrackClipDataset(dataset_cfgs)
    logger.info(
        'UniTrack clips: {} frames from {} sequences, window={}, temporal_steps={}',
        len(dataset),
        len(dataset.sequences),
        dataset.window_size,
        dataset.temporal_steps,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfgs['batch_size']),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    logger.info('Loading model')
    model = MambaTrack(train_cfgs)
    device = select_device(args.device)
    model.to(device)
    model.train()

    if train_cfgs['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=train_cfgs['lr0'],
        )
    else:
        raise RuntimeError('UniTrack baseline training uses the baseline SGD schedule')
    if train_cfgs['lr_scheduler'] == 'transformer':
        lr_scheduler = TransformerLRScheduler(
            optimizer=optimizer,
            d_model=train_cfgs['d_m'],
            warmup_steps=4000,
        )
    else:
        lr_scheduler = NoneLRScheduler(lr=train_cfgs['lr0'])

    criterion_cache = {}
    logger.info('Start training')
    log_every = 50
    for epoch in range(1, int(train_cfgs['epochs']) + 1):
        logger.info('Start epoch {}', epoch)
        means = {key: 0.0 for key in ('total', 'smooth', 'unitrack', 'tracking', 'spatial', 'temporal')}
        valid_steps = 0
        max_iter = len(dataloader)
        for step, samples in enumerate(dataloader):
            for sample in samples:
                for key in ('condition', 'label', 'prev_box', 'gt_box', 'track_id',
                            'temporal_condition', 'temporal_prev', 'temporal_gt'):
                    sample[key] = sample[key].to(device)
                sample['scale_factor'] = scale_factor

            last_cond = torch.cat([sample['condition'] for sample in samples], dim=0)
            labels = torch.cat([sample['label'] for sample in samples], dim=0)
            temporal_conds = [
                sample['temporal_condition'].reshape(-1, sample['temporal_condition'].shape[-2], 4)
                for sample in samples
                if sample['temporal_condition'].shape[0] > 0
            ]
            windows = [last_cond]
            if temporal_conds:
                windows.append(torch.cat(temporal_conds, dim=0))
            pred = squeeze_pred(model(torch.cat(windows, dim=0)))
            last_pred = pred[:last_cond.shape[0]]
            temporal_pred = pred[last_cond.shape[0]:]
            smooth = F.smooth_l1_loss(last_pred, labels)
            pred_box = []
            offset = 0
            for sample in samples:
                count = sample['prev_box'].shape[0]
                pred_box.append(sample['prev_box'] + last_pred[offset:offset + count] / scale_factor)
                offset += count
            frame_parts = frame_loss(criterion_cache, samples, torch.cat(pred_box, dim=0), iou_threshold)
            if temporal_pred.numel() == 0:
                track_parts = {key: smooth.new_zeros(()) for key in frame_parts}
            else:
                track_parts = temporal_loss(criterion_cache, samples, temporal_pred, iou_threshold)
            unitrack = frame_parts['loss_unitrack'] + track_parts['loss_unitrack']
            loss = smooth + weight * unitrack
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                logger.warning('Non-finite loss at epoch {} step {}; skip batch', epoch, step)
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_cfgs['lr_scheduler'] == 'transformer':
                lr_scheduler.step()

            values = {
                'total': loss_value,
                'smooth': float(smooth.item()),
                'unitrack': float(unitrack.item()),
                'tracking': float(frame_parts['loss_unitrack_tracking'].item()),
                'spatial': float(track_parts['loss_unitrack_spatial'].item()),
                'temporal': float(track_parts['loss_unitrack_temporal'].item()),
            }
            for key, value in values.items():
                means[key] = (means[key] * valid_steps + value) / (valid_steps + 1)
            valid_steps += 1
            if step % log_every == 0 or step == max_iter - 1:
                logger.info(
                    'epoch: {}, iter: {}, lr: {:.8f}, total: {:.6f}, smooth: {:.6f}, '
                    'unitrack: {:.6f}, tracking: {:.6f}, spatial: {:.6f}, temporal: {:.6f}',
                    epoch,
                    step,
                    optimizer.param_groups[0]['lr'],
                    values['total'],
                    values['smooth'],
                    values['unitrack'],
                    values['tracking'],
                    values['spatial'],
                    values['temporal'],
                )

        logger.info(
            'Epoch {} mean total {:.8f} smooth {:.8f} unitrack {:.8f} '
            'tracking {:.8f} spatial {:.8f} temporal {:.8f} steps {}',
            epoch,
            means['total'],
            means['smooth'],
            means['unitrack'],
            means['tracking'],
            means['spatial'],
            means['temporal'],
            valid_steps,
        )
        if epoch % int(train_cfgs.get('save_period', 1)) == 0:
            save_file = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                },
                save_file,
            )
            logger.info('Saved ckpt file at {}', save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline MambaTrack with UniTrack loss')
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--device', default='0')
    train(parser.parse_args())
