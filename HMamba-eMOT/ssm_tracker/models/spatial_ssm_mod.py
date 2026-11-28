"""Residual modulation of Mamba selective parameters with a spatial hypergraph prior."""

import torch
import torch.nn as nn


class SpatialPriorSSMModulator(nn.Module):
    """Residual-modulate Δ / B / C with a per-object spatial prior u.

    Δ' = Δ + α_Δ · MLP_Δ([x_t; u])                         if dt_combine == "add"
    Δ' = Δ ⊙ (1 + α_Δ · tanh(MLP_Δ([x_t; u])))             if dt_combine == "mul"
    B' = B ⊙ (1 + α_B · tanh(MLP_B([x_t; u])))
    C' = C ⊙ (1 + α_C · tanh(MLP_C([x_t; u])))
    """

    def __init__(
        self,
        d_ctx,
        d_inner,
        d_state,
        alpha_dt=0.1,
        alpha_b=0.1,
        alpha_c=0.1,
        last_k=0,
        dt_combine="add",
    ):
        super().__init__()
        if dt_combine not in ("add", "mul"):
            raise ValueError(f"dt_combine must be 'add' or 'mul', got {dt_combine!r}")
        self.dt_combine = dt_combine
        self.last_k = int(last_k)
        self.alpha_dt = nn.Parameter(torch.tensor(float(alpha_dt)))
        self.alpha_b = nn.Parameter(torch.tensor(float(alpha_b)))
        self.alpha_c = nn.Parameter(torch.tensor(float(alpha_c)))
        # Filled during forward for auxiliary L_mod (mean abs residual).
        self.last_mod_mag = None

        in_dim = d_inner + d_ctx
        self.mlp_dt = nn.Sequential(
            nn.Linear(in_dim, d_inner),
            nn.SiLU(),
            nn.Linear(d_inner, d_inner),
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(in_dim, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(in_dim, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
        )
        self._zero_last(self.mlp_dt)
        self._zero_last(self.mlp_b)
        self._zero_last(self.mlp_c)

    @staticmethod
    def _zero_last(module):
        last = module[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, dt, B, C, x, spatial_prior):
        """
        Args:
            dt: (batch, d_inner, L)
            B:  (batch, d_state, L)
            C:  (batch, d_state, L)
            x:  (batch, d_inner, L) conv features
            spatial_prior: (batch, d_ctx)
        """
        batch, _, seqlen = dt.shape
        if spatial_prior is None:
            self.last_mod_mag = None
            return dt, B, C

        x_t = x.transpose(1, 2)
        prior = spatial_prior[:, None, :].expand(-1, seqlen, -1)
        feat = torch.cat([x_t, prior], dim=-1)

        dt_delta = self.mlp_dt(feat).transpose(1, 2)
        if self.dt_combine == "mul":
            dt_delta = torch.tanh(dt_delta)
        b_gate = torch.tanh(self.mlp_b(feat)).transpose(1, 2)
        c_gate = torch.tanh(self.mlp_c(feat)).transpose(1, 2)

        if self.last_k > 0 and self.last_k < seqlen:
            mask = dt.new_zeros(1, 1, seqlen)
            mask[..., -self.last_k :] = 1.0
            dt_delta = dt_delta * mask
            b_gate = b_gate * mask
            c_gate = c_gate * mask

        dt_res = self.alpha_dt * dt_delta
        b_res = self.alpha_b * b_gate
        c_res = self.alpha_c * c_gate
        # Mean abs residual magnitude (kept in graph for L_mod).
        self.last_mod_mag = (
            dt_res.abs().mean() + b_res.abs().mean() + c_res.abs().mean()
        ) / 3.0

        if self.dt_combine == "mul":
            dt_out = dt * (1.0 + dt_res)
        else:
            dt_out = dt + dt_res
        b_out = B * (1.0 + b_res)
        c_out = C * (1.0 + c_res)
        return dt_out, b_out, c_out
