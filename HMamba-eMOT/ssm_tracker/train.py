import os 
import sys
import numpy as np 
import torch 
import yaml 
import argparse

from dataset.dataset import (
    FrameTrajDataset,
    TrajDataset,
    TrajDatasetv2,
    frame_hypergraph_collate,
)
from torch.utils.data import DataLoader
from train_utils.envs import select_device
from train_utils.lr_scheduler import TransformerLRScheduler, NoneLRScheduler

from models.MambaTrack import MambaTrack
from models.TrackSSM import TrackSSM

from loguru import logger
from tqdm import tqdm


MODEL_GALLERY = {
    'MambaTrack': MambaTrack, 
    'TrackSSM': TrackSSM
}


def _last_frame_hyper(hyper_inputs):
    return {
        relation: matrix[:, :, -1] if matrix.ndim == 4 else matrix
        for relation, matrix in hyper_inputs.items()
    }


@torch.no_grad()
def diagnose_spatial_hyper_batch(model, x, hyper_inputs, vertex_mask, label=None):
    """Stats for Spatial-hyper: h (pre-HG), u (post-HG), gates, edge density."""
    if not hasattr(model, 'spatial_hyper'):
        return None

    hyper_last = _last_frame_hyper(hyper_inputs)
    latest = x[:, :, -1]
    h = model.spatial_hyper.motion_proj(latest)
    u, gates = model.spatial_hyper(
        latest, hyper_last, vertex_mask=vertex_mask
    )

    mask = (
        vertex_mask.bool()
        if vertex_mask is not None
        else torch.ones(latest.shape[:2], dtype=torch.bool, device=latest.device)
    )
    if not mask.any():
        return {'n_valid': 0}

    h_v = h[mask]
    u_v = u[mask]
    g_v = gates[mask]
    h_norm = h_v.norm(dim=-1)
    u_norm = u_v.norm(dim=-1)
    # cosine(h, u) per vertex
    cos = torch.nn.functional.cosine_similarity(h_v, u_v, dim=-1)

    edge_stats = {}
    for relation, incidence in hyper_last.items():
        # incidence: [B, N, M]; degree over edges for valid vertices
        deg = incidence.to(dtype=torch.float32).sum(dim=-1)  # [B, N]
        deg_v = deg[mask]
        edge_stats[relation] = {
            'mean_deg': float(deg_v.mean().item()) if deg_v.numel() else 0.0,
            'max_deg': float(deg_v.max().item()) if deg_v.numel() else 0.0,
            'frac_isolated': float((deg_v <= 0).float().mean().item())
            if deg_v.numel()
            else 1.0,
        }

    gamma = float(model.spatial_hyper.gamma.detach().item())
    stats = {
        'n_valid': int(mask.sum().item()),
        'batch': int(x.shape[0]),
        'n_pad': int(x.shape[1]),
        'gamma': gamma,
        'h_norm_mean': float(h_norm.mean().item()),
        'h_norm_max': float(h_norm.max().item()),
        'u_norm_mean': float(u_norm.mean().item()),
        'u_norm_max': float(u_norm.max().item()),
        'u_over_h': float(
            (u_norm.mean() / h_norm.mean().clamp_min(1e-6)).item()
        ),
        'hu_cos_mean': float(cos.mean().item()),
        'hu_delta_l2_mean': float((u_v - h_v).norm(dim=-1).mean().item()),
        'gates_mean': {
            relation: float(g_v[:, idx].mean().item())
            for idx, relation in enumerate(model.spatial_hyper.RELATIONS)
        },
        'edges': edge_stats,
    }
    if label is not None:
        lab = label[mask]
        stats['label_abs_mean'] = float(lab.abs().mean().item())
        stats['label_abs_max'] = float(lab.abs().max().item())
        stats['label_abs_p99'] = float(
            torch.quantile(lab.abs().reshape(-1).float(), 0.99).item()
        )
    return stats


def format_spatial_diag(stats, prefix='spatial_diag'):
    if stats is None:
        return f'{prefix}: unavailable'
    if stats.get('n_valid', 0) == 0:
        return f'{prefix}: n_valid=0'
    gates = stats['gates_mean']
    gate_str = ' '.join(f'{k}={v:.3f}' for k, v in gates.items())
    edge_str = ' '.join(
        f"{k}:deg={v['mean_deg']:.2f}/iso={v['frac_isolated']:.2f}"
        for k, v in stats['edges'].items()
    )
    extra = ''
    if 'label_abs_mean' in stats:
        extra = (
            f" | label_abs mean={stats['label_abs_mean']:.4f} "
            f"max={stats['label_abs_max']:.4f} p99={stats['label_abs_p99']:.4f}"
        )
    return (
        f"{prefix}: n={stats['n_valid']}/{stats['batch']}x{stats['n_pad']} "
        f"gamma={stats['gamma']:.4f} "
        f"h_norm={stats['h_norm_mean']:.4f}(max={stats['h_norm_max']:.4f}) "
        f"u_norm={stats['u_norm_mean']:.4f}(max={stats['u_norm_max']:.4f}) "
        f"u/h={stats['u_over_h']:.4f} "
        f"cos(h,u)={stats['hu_cos_mean']:.4f} "
        f"||u-h||={stats['hu_delta_l2_mean']:.4f} "
        f"| gates[{gate_str}] | edges[{edge_str}]{extra}"
    )


_ALPHA_ATTRS = {
    'dt': 'alpha_dt',
    'b': 'alpha_b',
    'c': 'alpha_c',
}


def _freeze_ablated_alphas(model, cfgs):
    """Keep selected SSM scales at zero for the whole training run."""
    zero = cfgs.get('ablate_zero_alphas') or []
    if isinstance(zero, str):
        zero = [zero]
    zero = list(zero)
    if not zero:
        return
    unknown = [name for name in zero if name not in _ALPHA_ATTRS]
    if unknown:
        raise ValueError(f'unknown alphas to freeze: {unknown}')
    found = 0
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ != 'SpatialPriorSSMModulator':
                continue
            found += 1
            for name in zero:
                param = getattr(module, _ALPHA_ATTRS[name])
                param.zero_()
                param.requires_grad_(False)
    if found == 0:
        raise RuntimeError('ablate_zero_alphas set but the model has no SSM modulator')
    logger.info('Froze SSM alphas {} on {} modulators', zero, found)


def train(args):

    config_file = args.config_file
    with open(config_file, 'r') as f:
        cfgs = yaml.safe_load(f)
    dataset_cfgs = cfgs['dataset']
    train_cfgs = cfgs['train']

    # copy the experiment config file to the save path 
    save_dir = os.path.join(args.save_path, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system(f'cp {config_file} {save_dir}')

    # dataset 
    logger.info(f'Loading data')
    use_frame_hyper = bool(dataset_cfgs.get('frame_hypergraph', False))
    dataset = (
        FrameTrajDataset(dataset_cfgs, split='train')
        if use_frame_hyper else TrajDatasetv2(dataset_cfgs)
    )

    dataloader = DataLoader(dataset, 
                            batch_size=train_cfgs['batch_size'], 
                            shuffle=True, 
                            num_workers=int(train_cfgs.get('num_workers', 0)),
                            collate_fn=frame_hypergraph_collate if use_frame_hyper else None,
                            )
    val_dataloader = None
    val_root = os.path.join(dataset_cfgs.get('modal_feature_root', ''), 'val')
    val_ready = (
        use_frame_hyper
        and os.path.isdir(val_root)
        and any(name.endswith('.npz') for name in os.listdir(val_root))
    )
    if val_ready:
        val_dataset = FrameTrajDataset(dataset_cfgs, split='val')
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfgs['batch_size'],
            shuffle=False,
            num_workers=int(train_cfgs.get('num_workers', 0)),
            collate_fn=frame_hypergraph_collate,
        )
    elif use_frame_hyper:
        logger.info(f'No val features at {val_root}, skip validation')

    # model
    logger.info('Loading model')
    model = MODEL_GALLERY[train_cfgs['model']](train_cfgs)
    device = select_device(args.device)
    model.to(device)
    model.train()
    if args.init_ckpt:
        init_ckpt = torch.load(args.init_ckpt, map_location=device)
        incompatible = model.load_state_dict(init_ckpt['model'], strict=False)
        logger.info(
            f'Initialized from {args.init_ckpt}; '
            f'missing={len(incompatible.missing_keys)}, '
            f'unexpected={len(incompatible.unexpected_keys)}'
        )
    _freeze_ablated_alphas(model, train_cfgs)

    # Optionally freeze backbone and only train selected modules (e.g. spatial-hyper).
    # Config: train_only_modules: ['spatial_hyper', 'hyper_context_proj', 'hyper_fusion']
    train_only = train_cfgs.get('train_only_modules', None)
    if train_only:
        if isinstance(train_only, str):
            train_only = [train_only]
        prefixes = tuple(str(p) for p in train_only)
        n_train, n_freeze = 0, 0
        trainable_names = []
        for name, param in model.named_parameters():
            keep = any(name == p or name.startswith(p + '.') for p in prefixes)
            param.requires_grad = keep
            if keep:
                n_train += param.numel()
                trainable_names.append(name)
            else:
                n_freeze += param.numel()
        logger.info(
            f'train_only_modules={list(prefixes)}: '
            f'trainable_params={n_train}, frozen_params={n_freeze}, '
            f'trainable_tensors={len(trainable_names)}'
        )
        if n_train == 0:
            raise RuntimeError(
                f'No trainable params matched train_only_modules={list(prefixes)}'
            )

    # optimizer
    if train_cfgs['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params=filter(lambda x : x.requires_grad, model.parameters()), 
                                    lr=train_cfgs['lr0'], 
                                    )
    elif train_cfgs['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params=filter(lambda x : x.requires_grad, model.parameters()), 
                                     lr=train_cfgs['lr0'], 
                                     betas=[0.9, 0.98], eps=1e-8)
    else:
        optimizer = None
        
    # lr scheduler
    if train_cfgs['lr_scheduler'] == 'transformer':
        lr_scheduler = TransformerLRScheduler(optimizer=optimizer, d_model=train_cfgs['d_m'], warmup_steps=4000)
    elif train_cfgs['lr_scheduler'] == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.5, total_iters=10)
    else:
        lr_scheduler = NoneLRScheduler(lr=train_cfgs['lr0'])

    # if resume training, load the ckpt 
    if args.resume:
        load_ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(load_ckpt['model'])
        _freeze_ablated_alphas(model, train_cfgs)
        optimizer.load_state_dict(load_ckpt['optimizer'])
        start_epoch = load_ckpt['epoch'] + 1
        logger.info(f'Resume training from epoch {start_epoch}')
        # Transformer LR is step-based and not saved in old ckpts; continue
        # the schedule from the approximate number of completed steps.
        if isinstance(lr_scheduler, TransformerLRScheduler):
            lr_scheduler.current_step = (start_epoch - 1) * len(dataloader)
            logger.info(
                f'Restored TransformerLRScheduler current_step='
                f'{lr_scheduler.current_step}'
            )
    else:
        start_epoch = 1

    grad_clip = float(train_cfgs.get('grad_clip', 0.0))
    best_val = float('inf')
    best_epoch = -1
    best_path = os.path.join(save_dir, 'best.pth')
    if os.path.isfile(best_path):
        try:
            best_ckpt = torch.load(best_path, map_location='cpu')
            if 'val_loss' in best_ckpt:
                best_val = float(best_ckpt['val_loss'])
                best_epoch = int(best_ckpt.get('epoch', -1))
                logger.info(
                    f'Loaded previous best from {best_path}: '
                    f'epoch={best_epoch}, val_loss={best_val:.8f}'
                )
        except Exception as exc:
            logger.warning(f'Failed to load previous best.pth: {exc}')
    # Spatial-hyper diagnostics:
    # - log compact h/u/gates every diag_every steps (default 50)
    # - always log full diag when loss is non-finite or > diag_loss_thresh
    diag_every = int(train_cfgs.get('diag_every', 50))
    diag_loss_thresh = float(train_cfgs.get('diag_loss_thresh', 1.0))
    label_clip = float(train_cfgs.get('label_clip', 0.0))
    loss_skip_thresh = float(train_cfgs.get('loss_skip_thresh', 0.0))
    use_spatial_diag = use_frame_hyper and bool(
        train_cfgs.get('use_spatial_hyper', False)
        or train_cfgs.get('use_spatial_ssm_mod', False)
    )

    # train
    logger.info('Start training!')
    if grad_clip > 0:
        logger.info(f'Gradient clipping enabled: max_norm={grad_clip}')
    if label_clip > 0:
        logger.info(f'Label clipping enabled: |label| <= {label_clip}')
    if loss_skip_thresh > 0:
        logger.info(f'Loss skip enabled: skip batch if loss > {loss_skip_thresh}')
    if use_spatial_diag:
        logger.info(
            f'Spatial-hyper diagnostics enabled: '
            f'diag_every={diag_every}, diag_loss_thresh={diag_loss_thresh}'
        )
    
    for epoch in range(start_epoch, train_cfgs['epochs'] + 1):
        logger.info(f'Start epoch {epoch}')

        max_iter = len(dataloader)
        # When stdout is redirected to a file, tqdm redraws become new lines and
        # keep reprinting a stale description (same iter). Disable the bar then
        # and only emit logger.info every log_every steps.
        log_every = 50
        use_tqdm_bar = sys.stdout.isatty()
        process_bar = tqdm(
            enumerate(dataloader),
            total=max_iter,
            ncols=150,
            miniters=log_every,
            mininterval=1.0,
            disable=not use_tqdm_bar,
        )

        model.train()
        optimizer.zero_grad()

        mean_loss = 0.0
        window_loss = 0.0
        window_count = 0
        skipped_batches = 0
        skipped_high_loss = 0
        valid_steps = 0
        diag_alerts = 0

        if not train_cfgs['lr_scheduler'] == 'transformer':
            lr_scheduler.step()  # most lr scheduler is updated per epoch

        for step, data in process_bar:
            if use_frame_hyper:
                x = data['condition'].to(device)
                label = data['label'].to(device)
                vertex_mask = data['vertex_mask'].to(device)
                hyper_inputs = {
                    'spatial': data['H_spatial'].to(device),
                    'rgb': data['H_rgb'].to(device),
                    'evt': data['H_evt'].to(device),
                    'motion': data['H_motion'].to(device),
                }
            elif type(data) == dict:
                x = data['condition'].to(device)
                label = data['label'].to(device)
            else:
                x = data[0].to(device)
                label = data[1].to(device)

            if label_clip > 0:
                label = torch.clamp(label, min=-label_clip, max=label_clip)

            if use_frame_hyper:
                loss = model.forward(
                    x,
                    label,
                    hyper_inputs=hyper_inputs,
                    vertex_mask=vertex_mask,
                )
            else:
                loss = model.forward(x, label)

            loss_value = float(loss.item())
            abnormal = (not np.isfinite(loss_value)) or (
                loss_value > diag_loss_thresh
            )
            if use_spatial_diag and (
                abnormal or (diag_every > 0 and step % diag_every == 0)
            ):
                try:
                    stats = diagnose_spatial_hyper_batch(
                        model, x, hyper_inputs, vertex_mask, label=label
                    )
                    msg = format_spatial_diag(
                        stats,
                        prefix=(
                            f'spatial_diag ep={epoch} step={step} '
                            f'loss={loss_value}'
                        ),
                    )
                    if abnormal:
                        diag_alerts += 1
                        logger.warning(msg)
                    else:
                        logger.info(msg)
                except Exception as exc:
                    logger.warning(
                        f'spatial_diag failed at ep={epoch} step={step}: {exc}'
                    )

            if not np.isfinite(loss_value):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    f'Non-finite train loss at epoch {epoch} step {step}; skip batch'
                )
                continue

            if loss_skip_thresh > 0 and loss_value > loss_skip_thresh:
                skipped_high_loss += 1
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    f'High train loss {loss_value:.4f} > {loss_skip_thresh} '
                    f'at epoch {epoch} step {step}; skip batch'
                )
                continue

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if train_cfgs['lr_scheduler'] == 'transformer':
                lr_scheduler.step()  # transformer scheduler is updated per step

            cur_lr = optimizer.param_groups[0]['lr']

            mean_loss = (mean_loss * valid_steps + loss_value) / (valid_steps + 1)
            valid_steps += 1
            window_loss += loss_value
            window_count += 1

            # Log once every log_every steps (and on the last step).
            if step % log_every == 0 or step == max_iter - 1:
                recent = window_loss / max(window_count, 1)
                aux = getattr(model, 'last_loss_dict', {}) or {}
                aux_msg = ''
                if aux:
                    parts = [
                        f'{k}={float(v):.3e}'
                        for k, v in aux.items()
                        if k != 'bbox'
                    ]
                    if parts:
                        aux_msg = ', ' + ', '.join(parts)
                progress_msg = (
                    f'epoch: {epoch}, iter: {step}, lr: {cur_lr:.8f}, '
                    f'current: {loss_value:.6f}, mean: {mean_loss:.6f}, '
                    f'recent50: {recent:.6f}{aux_msg}'
                )
                if use_tqdm_bar:
                    process_bar.set_description(progress_msg)
                logger.info(progress_msg)
                window_loss = 0.0
                window_count = 0


        logger.info(
            f'Epoch {epoch} train SmoothL1 mean: {mean_loss:.8f} '
            f'(valid_steps={valid_steps}, skipped={skipped_batches}, '
            f'skipped_high_loss={skipped_high_loss}, diag_alerts={diag_alerts})'
        )

        if not epoch % train_cfgs['save_period']:
            # save model
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch,
            }
                
            save_file = os.path.join(save_dir, f'epoch{epoch}.pth')
            torch.save(save_dict, save_file)
            logger.info(f'Saved ckpt file at {save_file}')

        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            abs_err = 0.0
            elem_count = 0
            with torch.no_grad():
                for val_step, data in enumerate(val_dataloader):
                    hyper_inputs = {
                        'spatial': data['H_spatial'].to(device),
                        'rgb': data['H_rgb'].to(device),
                        'evt': data['H_evt'].to(device),
                        'motion': data['H_motion'].to(device),
                    }
                    x = data['condition'].to(device)
                    label = data['label'].to(device)
                    vertex_mask = data['vertex_mask'].to(device)
                    pred = model(
                        x,
                        hyper_inputs=hyper_inputs,
                        vertex_mask=vertex_mask,
                    )
                    if pred.ndim == 4 and pred.shape[2] == 1:
                        pred = pred.squeeze(2)
                    elif pred.ndim == 3 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                    valid = vertex_mask.bool()
                    err = pred[valid] - label[valid]
                    batch_loss = torch.nn.functional.smooth_l1_loss(
                        pred[valid], label[valid]
                    ).item()
                    val_loss = (val_loss * val_step + batch_loss) / (val_step + 1)
                    abs_err += err.abs().sum().item()
                    elem_count += err.numel()

            val_mae = abs_err / max(elem_count, 1)
            logger.info(
                f'Epoch {epoch} validation SmoothL1: {val_loss:.8f}, '
                f'MAE: {val_mae:.8f} (elems={elem_count})'
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                    },
                    best_path,
                )
                logger.info(
                    f'New best ckpt @ epoch {epoch}: '
                    f'val_loss={val_loss:.8f}, MAE={val_mae:.8f} -> {best_path}'
                )

    if best_epoch > 0:
        logger.info(
            f'Training done. Best val @ epoch {best_epoch}: SmoothL1={best_val:.8f}'
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba Tracker Training')

    parser.add_argument('--exp_name', type=str, default='mamba_track', help='experiment name for ckpt saving')
    parser.add_argument('--config_file', type=str, default='cfgs/MambaTrack.yaml')
    parser.add_argument('--device', type=str, default='4')
    parser.add_argument('--save_path', type=str, default='./saved_ckpts')
    parser.add_argument('--init_ckpt', type=str, default=None)

    # resume training
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    args = parser.parse_args()
    train(args)
