from einops import rearrange

import torch
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from .hypergraph_b import HypergraphBModulator
from .spatial_ssm_mod import SpatialPriorSSMModulator


def _mamba_pre_scan(module, hidden_states, inference_params=None):
    """Shared path up to (and including) x_proj split of dt / B / C."""
    batch, seqlen, dim = hidden_states.shape

    conv_state, ssm_state = None, None
    if inference_params is not None:
        conv_state, ssm_state = module._get_states_from_cache(inference_params, batch)
        if inference_params.seqlen_offset > 0:
            out, _, _ = module.step(hidden_states, conv_state, ssm_state)
            return {"early_out": out}

    xz = rearrange(
        module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if module.in_proj.bias is not None:
        xz = xz + rearrange(module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    A = -torch.exp(module.A_log.float())

    x, z = xz.chunk(2, dim=1)
    if conv_state is not None:
        conv_state.copy_(F.pad(x, (module.d_conv - x.shape[-1], 0)))
    if causal_conv1d_fn is None:
        x = module.act(module.conv1d(x)[..., :seqlen])
    else:
        assert module.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(module.conv1d.weight, "d 1 w -> d w"),
            bias=module.conv1d.bias,
            activation=module.activation,
        )

    x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt, B, C = torch.split(
        x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1
    )
    dt = module.dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    return {
        "x": x,
        "z": z,
        "dt": dt,
        "B": B,
        "C": C,
        "A": A,
        "ssm_state": ssm_state,
        "seqlen": seqlen,
    }


def _mamba_scan_and_proj(module, pack):
    assert module.activation in ["silu", "swish"]
    y = selective_scan_fn(
        pack["x"],
        pack["dt"],
        pack["A"],
        pack["B"],
        pack["C"],
        module.D.float(),
        z=pack["z"],
        delta_bias=module.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=pack["ssm_state"] is not None,
    )
    if pack["ssm_state"] is not None:
        y, last_state = y
        pack["ssm_state"].copy_(last_state)
    y = rearrange(y, "b d l -> b l d")
    return module.out_proj(y)


class MambaWithHyperB(Mamba):
    """Mamba block with hypergraph modulation on B before selective scan."""

    def __init__(
        self,
        *args,
        d_vertex=128,
        hyper_temperature=1.0,
        hyper_mask_threshold=0.1,
        hyper_gate_scale=1.0,
        **kwargs,
    ):
        kwargs["use_fast_path"] = False
        super().__init__(*args, **kwargs)
        self.hyper_b = HypergraphBModulator(
            d_state=self.d_state,
            d_vertex=d_vertex,
            d_inner=self.d_inner,
            temperature=hyper_temperature,
            mask_threshold=hyper_mask_threshold,
            gate_scale=hyper_gate_scale,
        )

    def forward(self, hidden_states, inference_params=None):
        pack = _mamba_pre_scan(self, hidden_states, inference_params)
        if "early_out" in pack:
            return pack["early_out"]
        pack["B"], _ = self.hyper_b(pack["B"], pack["x"])
        return _mamba_scan_and_proj(self, pack)


class MambaWithSpatialSSMMod(Mamba):
    """Mamba block with spatial-hypergraph residual modulation of Δ / B / C."""

    def __init__(
        self,
        *args,
        spatial_hyper_dim=64,
        ssm_mod_alpha_dt=0.1,
        ssm_mod_alpha_b=0.1,
        ssm_mod_alpha_c=0.1,
        ssm_mod_last_k=0,
        ssm_mod_dt_combine="add",
        **kwargs,
    ):
        kwargs["use_fast_path"] = False
        super().__init__(*args, **kwargs)
        self.spatial_ssm_mod = SpatialPriorSSMModulator(
            d_ctx=spatial_hyper_dim,
            d_inner=self.d_inner,
            d_state=self.d_state,
            alpha_dt=ssm_mod_alpha_dt,
            alpha_b=ssm_mod_alpha_b,
            alpha_c=ssm_mod_alpha_c,
            last_k=ssm_mod_last_k,
            dt_combine=ssm_mod_dt_combine,
        )

    def forward(self, hidden_states, inference_params=None, spatial_prior=None):
        pack = _mamba_pre_scan(self, hidden_states, inference_params)
        if "early_out" in pack:
            return pack["early_out"]
        if spatial_prior is not None:
            pack["dt"], pack["B"], pack["C"] = self.spatial_ssm_mod(
                pack["dt"],
                pack["B"],
                pack["C"],
                pack["x"],
                spatial_prior,
            )
        return _mamba_scan_and_proj(self, pack)
