"""Trainable message passing over fixed, typed frame-level hyperedges."""

import torch
import torch.nn as nn


class TypedHypergraphConv(nn.Module):
    """Leave-one-out vertex-edge-vertex aggregation for one relation type.

    ``agg``:
      - ``attn``: learnable soft weights over vertices on each hyperedge
      - ``mean``: uniform leave-one-out average (no edge attention)
    """

    def __init__(self, dim, agg='attn'):
        super().__init__()
        self.agg = str(agg).lower()
        if self.agg not in ('attn', 'mean'):
            raise ValueError(f'Unsupported hyperedge agg={agg!r}; use attn|mean')
        if self.agg == 'attn':
            self.edge_attention = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.SiLU(),
                nn.Linear(dim, 1),
            )
        self.peer_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.vertex_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    def forward(self, vertex_features, incidence):
        # vertex_features: [B, N, D], incidence: [B, N, M]
        incidence = incidence.to(vertex_features.dtype)
        edge_degree = incidence.sum(dim=1)  # [B, M]
        edge_mean = torch.einsum("bnm,bnd->bmd", incidence, vertex_features)
        edge_mean = edge_mean / edge_degree.clamp_min(1.0)[:, :, None]

        if self.agg == 'attn':
            n_vertices, n_edges = incidence.shape[1:3]
            vertex_expanded = vertex_features[:, :, None, :].expand(
                -1, -1, n_edges, -1
            )
            edge_expanded = edge_mean[:, None, :, :].expand(
                -1, n_vertices, -1, -1
            )
            attention_logits = self.edge_attention(
                torch.cat([vertex_expanded, edge_expanded], dim=-1)
            ).squeeze(-1)
            attention_logits = attention_logits.masked_fill(incidence <= 0, -1e4)
            weights = torch.softmax(attention_logits, dim=1) * incidence
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            edge_sum = torch.einsum("bnm,bnd->bmd", weights, vertex_features)

            # Each target receives the mean of other vertices in its incident edges.
            peer_sum = edge_sum[:, None, :, :] - (
                weights[:, :, :, None] * vertex_features[:, :, None, :]
            )
            peer_weight = 1.0 - weights
            valid_peer = (edge_degree[:, None, :] - incidence) > 0
            peer_mean = peer_sum / peer_weight.clamp_min(1e-8)[:, :, :, None]
        else:
            # Uniform leave-one-out: (deg·edge_mean - x_i) / (deg - 1).
            valid_peer = (edge_degree[:, None, :] - incidence) > 0
            peer_sum = (
                edge_mean[:, None, :, :] * edge_degree[:, None, :, None]
                - incidence[:, :, :, None] * vertex_features[:, :, None, :]
            )
            peer_den = (edge_degree[:, None, :] - incidence).clamp_min(1.0)
            peer_mean = peer_sum / peer_den[:, :, :, None]

        peer_message = self.peer_mlp(peer_mean)
        peer_message = peer_message * valid_peer[:, :, :, None]

        incident = incidence * valid_peer.to(incidence.dtype)
        vertex_message = peer_message.sum(dim=2)
        vertex_message = vertex_message / incident.sum(dim=2).clamp_min(1.0)[
            :, :, None
        ]
        return self.vertex_mlp(vertex_message)


class SpatialHyperResidual(nn.Module):
    """Multi-relation hypergraph context from the latest observed motion."""

    RELATIONS = ("spatial", "rgb", "evt", "motion")

    def __init__(
        self, input_dim=4, hidden_dim=64, gamma_init=0.1, edge_agg='attn'
    ):
        super().__init__()
        self.edge_agg = str(edge_agg).lower()
        self.motion_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.convs = nn.ModuleDict(
            {
                relation: TypedHypergraphConv(hidden_dim, agg=self.edge_agg)
                for relation in self.RELATIONS
            }
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(hidden_dim * (len(self.RELATIONS) + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self.RELATIONS)),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, latest_motion, incidences, vertex_mask=None):
        motion_features = self.motion_proj(latest_motion)
        contexts = [
            self.convs[relation](motion_features, incidences[relation])
            for relation in self.RELATIONS
        ]
        gate_input = torch.cat([motion_features] + contexts, dim=-1)
        gates = torch.softmax(self.relation_gate(gate_input), dim=-1)
        stacked = torch.stack(contexts, dim=2)
        context = (stacked * gates[:, :, :, None]).sum(dim=2)
        context = self.gamma * self.out_proj(context)
        if vertex_mask is not None:
            context = context * vertex_mask[:, :, None].to(context.dtype)
        return context, gates
