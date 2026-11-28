import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphBModulator(nn.Module):
    """
    Hypergraph-style refinement of selective-scan matrix B.

    Vertices: L time steps, features from conv output x.
    Hyperedges: N=d_state learnable prototype vectors.
    Affinity H (B, N, L) aggregates vertex features per hyperedge, then
    B' = B * (1 + gate_scale * sigmoid(gate)).
    """

    def __init__(
        self,
        d_state=16,
        d_vertex=128,
        d_inner=1024,
        temperature=1.0,
        mask_threshold=0.1,
        gate_scale=1.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.temperature = temperature
        self.mask_threshold = mask_threshold
        self.gate_scale = gate_scale

        self.vertex_proj = nn.Linear(d_inner, d_vertex)# 顶点投影：把卷积输出x（d_inner通道）映射到统一顶点特征维度d_vertex
        self.hyperedge_proto = nn.Parameter(torch.randn(d_state, d_vertex) * 0.02)# 可学习超边原型：d_state条超边，每条是d_vertex维可训练向量

    def forward(self, B, x):
        """
        Args:
            B: (B, d_state, L) input-dependent SSM B matrix
            x: (B, d_inner, L) conv output before selective scan
        Returns:
            B_refined: (B, d_state, L)
            H: (B, d_state, L) hyperedge-vertex affinity (softmax over hyperedges)
        """
        v = self.vertex_proj(x.transpose(1, 2))
        batch_size = B.shape[0]
        e = self.hyperedge_proto.unsqueeze(0).expand(batch_size, -1, -1)

        v_n = F.normalize(v, dim=-1)
        e_n = F.normalize(e, dim=-1)
        affinity = torch.einsum("bnd,bld->bnl", e_n, v_n) / self.temperature
        # H = F.softmax(affinity, dim=-1)
        H = F.softmax(affinity, dim=1)

        # 加权聚合顶点特征：用H把所有顶点信息汇聚到每条超边
        msg = torch.einsum("bnl,bld->bnd", H, v) # [B, d_state, d_vertex]
        # 超边聚合特征 和 原始顶点特征做内积，得到门控分数
        gate = torch.einsum("bnd,bld->bnl", msg, v) # [B, d_state, L]
        gate = torch.sigmoid(gate) # 压缩到(0,1)区间
        B_refined = B * (1.0 + self.gate_scale * gate)

        # H_mask = (H > self.mask_threshold).to(B.dtype)
        # B_refined = H_mask * B
        # return B_refined, H_mask
        return B_refined, H
