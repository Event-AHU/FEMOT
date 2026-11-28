"""
MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from mamba_ssm import Mamba

from models.mamba_hyper import MambaWithHyperB, MambaWithSpatialSSMMod
from models.spatial_hypergraph import SpatialHyperResidual


def build_mamba(cfgs):
    mamba_kwargs = dict(
        d_model=cfgs['d_m'],
        d_state=cfgs['d_state'],
        d_conv=4,
        expand=2,
    )
    if cfgs.get('use_hyper_b', False):
        mamba_kwargs['d_vertex'] = cfgs.get('hyper_d_vertex', 128)
        mamba_kwargs['hyper_temperature'] = cfgs.get('hyper_temperature', 1.0)
        mamba_kwargs['hyper_mask_threshold'] = cfgs.get('hyper_mask_threshold', 0.1)
        mamba_kwargs['hyper_gate_scale'] = cfgs.get('hyper_gate_scale', 1.0)
        return MambaWithHyperB(**mamba_kwargs)
    if cfgs.get('use_spatial_ssm_mod', False):
        mamba_kwargs['spatial_hyper_dim'] = cfgs.get('spatial_hyper_dim', 64)
        mamba_kwargs['ssm_mod_alpha_dt'] = cfgs.get('ssm_mod_alpha_dt', 0.1)
        mamba_kwargs['ssm_mod_alpha_b'] = cfgs.get('ssm_mod_alpha_b', 0.1)
        mamba_kwargs['ssm_mod_alpha_c'] = cfgs.get('ssm_mod_alpha_c', 0.1)
        mamba_kwargs['ssm_mod_last_k'] = cfgs.get('ssm_mod_last_k', 0)
        mamba_kwargs['ssm_mod_dt_combine'] = cfgs.get('ssm_mod_dt_combine', 'add')
        return MambaWithSpatialSSMMod(**mamba_kwargs)
    return Mamba(**mamba_kwargs)


class BiMambaBlock(nn.Module):
    '''
    Bi-Mamba Block in MTP
    '''
    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.use_spatial_ssm_mod = bool(cfgs.get('use_spatial_ssm_mod', False))

        self.forward_mamba = build_mamba(self.cfgs)
        self.backward_mamba = build_mamba(self.cfgs)
        
        self.layer_norm = nn.LayerNorm((self.cfgs['d_m']), )

        self.proj_layer = nn.Sequential(
            nn.Linear(in_features=self.cfgs['d_m'], out_features=self.cfgs['d_m'] * 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.cfgs['d_m'] * 2, out_features=self.cfgs['d_m']),
        )

    def forward(self, x, spatial_prior=None):
        # x: shape (bs, q, d_m)
        
        # reverse the temporal casual 
        x_back = torch.flip(x, dims=[-2])

        if self.use_spatial_ssm_mod and spatial_prior is not None:
            forward_mamba_out = self.forward_mamba(x, spatial_prior=spatial_prior)
            backward_mamba_out = self.backward_mamba(x_back, spatial_prior=spatial_prior)
        else:
            forward_mamba_out = self.forward_mamba(x)
            backward_mamba_out = self.backward_mamba(x_back)

        mamba_out = forward_mamba_out + backward_mamba_out

        out = mamba_out + self.proj_layer(self.layer_norm(mamba_out))

        return out

class MambaTrack(nn.Module):
    '''
    Mamba Motion Predictor in paper (MTP)
    '''
    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs

        # projection layer, one-layer mlp
        self.proj_embedding_layer = nn.Linear(in_features=4, 
                                              out_features=self.cfgs['d_m'], 
                                              )
        self.use_spatial_hyper = bool(self.cfgs.get('use_spatial_hyper', False))
        self.use_spatial_ssm_mod = bool(self.cfgs.get('use_spatial_ssm_mod', False))
        self.spatial_hyper_per_frame = bool(
            self.cfgs.get('spatial_hyper_per_frame', False)
        )
        self.spatial_hyper_dual_gate = bool(
            self.cfgs.get('spatial_hyper_dual_gate', False)
        )
        if self.use_spatial_hyper and self.use_spatial_ssm_mod:
            raise ValueError(
                'use_spatial_hyper and use_spatial_ssm_mod cannot both be true'
            )
        if self.spatial_hyper_dual_gate and not self.use_spatial_hyper:
            raise ValueError(
                'spatial_hyper_dual_gate requires use_spatial_hyper=true'
            )

        needs_spatial_hyper = self.use_spatial_hyper or self.use_spatial_ssm_mod
        if needs_spatial_hyper:
            hyper_dim = int(self.cfgs.get('spatial_hyper_dim', 64))
            gamma_init = float(self.cfgs.get('spatial_hyper_gamma_init', 0.1))
            edge_agg = str(self.cfgs.get('spatial_hyper_edge_agg', 'attn'))
            self.spatial_hyper = SpatialHyperResidual(
                input_dim=4,
                hidden_dim=hyper_dim,
                gamma_init=gamma_init,
                edge_agg=edge_agg,
            )
        if self.use_spatial_hyper:
            hyper_dim = int(self.cfgs.get('spatial_hyper_dim', 64))
            self.hyper_context_proj = nn.Linear(hyper_dim, self.cfgs['d_m'])
            self.hyper_fusion = nn.Linear(
                self.cfgs['d_m'] * 2, self.cfgs['d_m']
            )
            self._init_hyper_fusion()
            if self.spatial_hyper_dual_gate:
                # Per-object gate: α≈0 at init → prefer self-only prediction.
                gate_hidden = int(self.cfgs.get('spatial_hyper_gate_hidden', 128))
                gate_bias = float(self.cfgs.get('spatial_hyper_gate_bias', -2.0))
                self.prior_gate = nn.Sequential(
                    nn.Linear(self.cfgs['d_m'] * 2, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, 1),
                )
                nn.init.zeros_(self.prior_gate[-1].weight)
                nn.init.constant_(self.prior_gate[-1].bias, gate_bias)
        
        # L-layers Bi-Mamba Block
        self.bi_mamba_blocks = nn.ModuleList(
            [BiMambaBlock(self.cfgs) for _ in range(self.cfgs['L'])]
        )

        # prediction head, avg pooling + 2-layer mlp
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=self.cfgs['avg_pool_out_dim'])
        self.pred_head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=self.cfgs['avg_pool_out_dim']), 
                                        nn.Linear(self.cfgs['avg_pool_out_dim'][1], self.cfgs['pred_head_dims'][0]), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.cfgs['pred_head_dims'][0], self.cfgs['pred_head_dims'][1]), 
                                        )
        
        self.loss_func = BboxLoss(self.cfgs)
        self.last_loss_dict = {}
        
    def _init_hyper_fusion(self):
        """Initialize concat fusion as the exact self-motion identity."""
        with torch.no_grad():
            self.hyper_fusion.weight.zero_()
            self.hyper_fusion.bias.zero_()
            self.hyper_fusion.weight[:, :self.cfgs['d_m']].copy_(
                torch.eye(self.cfgs['d_m'])
            )
            nn.init.normal_(
                self.hyper_fusion.weight[:, self.cfgs['d_m']:],
                mean=0.0,
                std=0.001,
            )

    def _run_bi_mamba(self, x, spatial_prior=None):
        for block in self.bi_mamba_blocks:
            x = block(x, spatial_prior=spatial_prior)
        return x

    def _collect_mod_mags(self):
        """Gather mean-abs modulation magnitudes from all SSM-mod blocks."""
        mags = []
        for block in self.bi_mamba_blocks:
            for mamba in (block.forward_mamba, block.backward_mamba):
                mod = getattr(mamba, 'spatial_ssm_mod', None)
                if mod is None:
                    continue
                mag = getattr(mod, 'last_mod_mag', None)
                if mag is not None:
                    mags.append(mag)
        return mags

    def _modulation_aux_loss(self):
        """L_mod: push mean |Δ/B/C residual| toward a target band."""
        lam = float(self.cfgs.get('aux_mod_lambda', 0.0))
        if lam <= 0 or not self.use_spatial_ssm_mod:
            return None
        mags = self._collect_mod_mags()
        if not mags:
            return None
        mag = torch.stack(mags).mean()
        target = float(self.cfgs.get('aux_mod_target', 0.01))
        over_w = float(self.cfgs.get('aux_mod_over_weight', 1.0))
        # Under-use: encourage non-trivial modulation; over-use: L1 on magnitude.
        loss = F.relu(target - mag).pow(2) + over_w * mag
        return lam * loss

    def _neighbor_aux_loss(self, pred, hyper_inputs, vertex_mask=None):
        """L_nbr: predictions of hypergraph-linked objects should be close."""
        lam = float(self.cfgs.get('aux_nbr_lambda', 0.0))
        if lam <= 0 or hyper_inputs is None:
            return None
        if pred.ndim == 4 and pred.shape[2] == 1:
            pred = pred.squeeze(2)
        elif pred.ndim == 3 and pred.shape[1] == 1:
            # [B,1,4] single-object — no neighbors
            return None
        if pred.ndim != 3:
            return None

        relations = self.cfgs.get(
            'aux_nbr_relations', ['spatial', 'rgb', 'evt', 'motion']
        )
        affinity = None
        for relation in relations:
            if relation not in hyper_inputs:
                continue
            incidence = hyper_inputs[relation]
            if incidence.ndim == 4:
                incidence = incidence[:, :, -1]
            # Shared hyperedges → pairwise affinity: H H^T
            pair = torch.bmm(incidence, incidence.transpose(1, 2))
            affinity = pair if affinity is None else affinity + pair
        if affinity is None:
            return None

        batch_size, num_objects, _ = pred.shape
        eye = torch.eye(
            num_objects, device=pred.device, dtype=affinity.dtype
        )[None]
        affinity = affinity * (1.0 - eye)

        if vertex_mask is not None:
            valid = vertex_mask.bool().float()
            affinity = affinity * valid[:, :, None] * valid[:, None, :]

        weight_sum = affinity.sum()
        if weight_sum.item() <= 0:
            return None

        # Pairwise Smooth-L1 between linked objects, weighted by affinity.
        diff = pred[:, :, None, :] - pred[:, None, :, :]
        pair_loss = F.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction='none'
        ).mean(dim=-1)
        loss = (affinity * pair_loss).sum() / weight_sum.clamp_min(1e-6)
        return lam * loss

    def _prior_consistency_loss(self, context, vertex_mask=None):
        """L_prior: same-track spatial prior u should vary smoothly over time.

        Expects per-frame context [B, N, q, d] from spatial_hyper_per_frame.
        The default cosine metric is scale-invariant, so the loss cannot be
        driven to zero by shrinking ``u`` (which the plain L1 form allowed).
        """
        lam = float(self.cfgs.get('aux_prior_lambda', 0.0))
        if lam <= 0 or context is None or context.ndim != 4:
            return None
        if context.shape[2] < 2:
            return None

        # Optionally only the last K consecutive pairs (K=0 → all pairs).
        last_k = int(self.cfgs.get('aux_prior_last_k', 0))
        u = context
        if last_k > 0 and last_k < u.shape[2]:
            u = u[:, :, -(last_k + 1) :]

        metric = str(self.cfgs.get('aux_prior_metric', 'cos')).lower()
        if metric == 'cos':
            cos = F.cosine_similarity(u[:, :, 1:], u[:, :, :-1], dim=-1)
            pair_loss = 1.0 - cos  # [B, N, T-1]
        else:
            diff = u[:, :, 1:] - u[:, :, :-1]
            pair_loss = F.smooth_l1_loss(
                diff, torch.zeros_like(diff), reduction='none'
            ).mean(dim=-1)  # [B, N, T-1]

        if vertex_mask is not None:
            valid = vertex_mask.bool().float()[:, :, None]
            pair_loss = pair_loss * valid
            denom = valid.sum() * pair_loss.shape[-1]
            if denom.item() <= 0:
                return None
            loss = pair_loss.sum() / denom.clamp_min(1e-6)
        else:
            loss = pair_loss.mean()
        return lam * loss

    def _spatial_context(self, x, hyper_inputs, vertex_mask=None):
        """Return context [B,N,d] or [B,N,q,d] depending on per-frame mode."""
        sample = next(iter(hyper_inputs.values()))
        per_frame = self.spatial_hyper_per_frame and sample.ndim == 4
        if per_frame:
            # x: [B,N,q,4], H: [B,N,q,M] -> batch over time
            batch_size, num_objects, seqlen, _ = x.shape
            x_flat = x.permute(0, 2, 1, 3).reshape(
                batch_size * seqlen, num_objects, 4
            )
            hyper_flat = {
                relation: matrix.permute(0, 2, 1, 3).reshape(
                    batch_size * seqlen, num_objects, matrix.shape[-1]
                )
                for relation, matrix in hyper_inputs.items()
            }
            mask_flat = None
            if vertex_mask is not None:
                mask_flat = (
                    vertex_mask[:, None, :]
                    .expand(-1, seqlen, -1)
                    .reshape(batch_size * seqlen, num_objects)
                )
            context, gates = self.spatial_hyper(
                x_flat, hyper_flat, vertex_mask=mask_flat
            )
            context = context.reshape(
                batch_size, seqlen, num_objects, context.shape[-1]
            ).permute(0, 2, 1, 3)
            return context, gates

        # Legacy MVP: last-frame graph, optional squeeze of trailing time dim.
        hyper_last = {
            relation: matrix[:, :, -1] if matrix.ndim == 4 else matrix
            for relation, matrix in hyper_inputs.items()
        }
        return self.spatial_hyper(
            x[:, :, -1], hyper_last, vertex_mask=vertex_mask
        )

    def _fuse_prior_embedding(self, self_embedding, context):
        if context.ndim == 3:
            context_emb = self.hyper_context_proj(context)
            context_emb = context_emb[:, :, None, :].expand(
                -1, -1, self_embedding.shape[2], -1
            )
        else:
            context_emb = self.hyper_context_proj(context)
        fused = self.hyper_fusion(
            torch.cat([self_embedding, context_emb], dim=-1)
        )
        return fused, context_emb

    def forward(self, x, label=None, hyper_inputs=None, vertex_mask=None):
        multi_object = x.ndim == 4
        batch_size = x.shape[0]
        num_objects = x.shape[1] if multi_object else 1

        self_embedding = self.proj_embedding_layer(x)
        spatial_prior = None
        context_for_prior = None

        if multi_object and (self.use_spatial_hyper or self.use_spatial_ssm_mod):
            if hyper_inputs is None:
                raise ValueError(
                    'hyper_inputs are required when spatial hypergraph is enabled'
                )
            context, _ = self._spatial_context(
                x, hyper_inputs, vertex_mask=vertex_mask
            )
            if self.use_spatial_hyper:
                fused_embedding, context_emb = self._fuse_prior_embedding(
                    self_embedding, context
                )
                if self.spatial_hyper_dual_gate:
                    # Dual branch: self-only vs prior-fused, then gated mix.
                    self_flat = self_embedding.reshape(
                        batch_size * num_objects,
                        self_embedding.shape[2],
                        self_embedding.shape[3],
                    )
                    prior_flat = fused_embedding.reshape(
                        batch_size * num_objects,
                        fused_embedding.shape[2],
                        fused_embedding.shape[3],
                    )
                    y_self = self._run_bi_mamba(self_flat)
                    y_prior = self._run_bi_mamba(prior_flat)
                    out_self = self.pred_head(y_self)
                    out_prior = self.pred_head(y_prior)
                    out_self = out_self.reshape(
                        batch_size, num_objects, *out_self.shape[1:]
                    )
                    out_prior = out_prior.reshape(
                        batch_size, num_objects, *out_prior.shape[1:]
                    )

                    # Gate from last-step self + prior context embeddings.
                    gate_self = self_embedding[:, :, -1]
                    gate_ctx = context_emb[:, :, -1]
                    alpha = torch.sigmoid(
                        self.prior_gate(torch.cat([gate_self, gate_ctx], dim=-1))
                    )
                    # Match pred shape [B,N,1,4] or [B,N,4]
                    while alpha.ndim < out_self.ndim:
                        alpha = alpha.unsqueeze(-1)
                    out = (1.0 - alpha) * out_self + alpha * out_prior

                    if label is not None:
                        return self.loss_func(out, label, mask=vertex_mask)
                    return out

                x = fused_embedding
            else:
                x = self_embedding
                # Keep full [B,N,q,d] for L_prior; SSM mod still uses last step.
                if context.ndim == 4:
                    context_for_prior = context
                    context = context[:, :, -1]
                spatial_prior = context.reshape(
                    batch_size * num_objects, context.shape[-1]
                )
        else:
            x = self_embedding

        if multi_object:
            x = x.reshape(batch_size * num_objects, x.shape[2], x.shape[3])

        y = self._run_bi_mamba(x, spatial_prior=spatial_prior)

        out = self.pred_head(y)
        if multi_object:
            out = out.reshape(batch_size, num_objects, *out.shape[1:])

        if label is not None:
            loss = self.loss_func(out, label, mask=vertex_mask)
            loss_dict = {'bbox': loss.detach()}
            if self.use_spatial_ssm_mod:
                mod_loss = self._modulation_aux_loss()
                if mod_loss is not None:
                    loss = loss + mod_loss
                    loss_dict['mod'] = mod_loss.detach()
                nbr_loss = self._neighbor_aux_loss(
                    out, hyper_inputs, vertex_mask=vertex_mask
                )
                if nbr_loss is not None:
                    loss = loss + nbr_loss
                    loss_dict['nbr'] = nbr_loss.detach()
                prior_loss = self._prior_consistency_loss(
                    context_for_prior, vertex_mask=vertex_mask
                )
                if prior_loss is not None:
                    loss = loss + prior_loss
                    loss_dict['prior'] = prior_loss.detach()
            self.last_loss_dict = loss_dict
            return loss
        else:
            return out
    
class BboxLoss(nn.Module):
    '''Smooth L1 loss for training'''
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.loss_func = nn.SmoothL1Loss()

        self.const_term = 1

    def forward(self, pred, label, mask=None):
        '''
        pred: shape (bs, 1, 4)  the predicted offset
        label: shape (bs, 4)  the gt offset
        '''

        if pred.ndim == 3:
            pred = pred.squeeze(dim=1)
        elif pred.ndim == 4 and pred.shape[2] == 1:
            pred = pred.squeeze(dim=2)

        if mask is None:
            loss = self.loss_func.forward(pred, label)
        else:
            valid = mask.bool()
            loss = self.loss_func.forward(pred[valid], label[valid])
        return loss
