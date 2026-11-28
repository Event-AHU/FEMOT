#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_pair(rgb, evt):
    if rgb.shape != evt.shape:
        raise ValueError(
            "RGB and event features must have identical shapes, got "
            f"{tuple(rgb.shape)} and {tuple(evt.shape)}"
        )
    if rgb.ndim != 4:
        raise ValueError(f"Expected BCHW feature maps, got {tuple(rgb.shape)}")


def _validate_cross_attention_inputs(query_feature, context_feature):
    if query_feature.ndim != 4 or context_feature.ndim != 4:
        raise ValueError(
            "Expected BCHW feature maps, got "
            f"{tuple(query_feature.shape)} and {tuple(context_feature.shape)}"
        )
    if query_feature.shape[0] != context_feature.shape[0]:
        raise ValueError(
            "Query and context batch sizes must match, got "
            f"{query_feature.shape[0]} and {context_feature.shape[0]}"
        )
    if query_feature.shape[2:] != context_feature.shape[2:]:
        raise ValueError(
            "Query and context spatial shapes must match, got "
            f"{tuple(query_feature.shape)} and {tuple(context_feature.shape)}"
        )


def _window_partition(x, window_size):
    """Convert BCHW features to aligned window tokens, padding when needed."""
    batch, channels, height, width = x.shape
    if window_size <= 0:
        window_size = max(height, width)
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(
        batch,
        channels,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, channels)
    metadata = (batch, height, width, padded_h, padded_w, window_size)
    return windows, metadata


def _window_reverse(windows, metadata):
    batch, height, width, padded_h, padded_w, window_size = metadata
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(batch, channels, padded_h, padded_w)
    return x[:, :, :height, :width]


class HyLACrossAttention2D(nn.Module):
    """Windowed 2D port of HyLA's nonlinear hypernetwork attention."""

    def __init__(
        self,
        channels,
        attention_dim=64,
        num_heads=4,
        window_size=7,
        attention_norm="rms_head",
        dropout=0.0,
        eps=1e-6,
        query_channels=None,
    ):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(
                f"attention_dim ({attention_dim}) must divide num_heads ({num_heads})"
            )
        if attention_norm not in ("rms_head", "softmax"):
            raise ValueError(f"Unsupported HyLA normalization: {attention_norm}")
        self.channels = channels
        self.query_channels = query_channels if query_channels is not None else channels
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.window_size = window_size
        self.attention_norm = attention_norm
        self.eps = eps

        self.query_norm = nn.GroupNorm(1, self.query_channels)
        self.context_norm = nn.GroupNorm(1, channels)
        self.query = nn.Conv2d(self.query_channels, attention_dim, 1)
        self.key = nn.Conv2d(channels, attention_dim, 1)
        self.value = nn.Conv2d(channels, attention_dim, 1)
        self.out = nn.Conv2d(attention_dim, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def _normalize_attention(self, logits):
        if self.attention_norm == "softmax":
            return torch.softmax(logits, dim=-1)
        # HyLA normalizes each (query, key) latent code across attention heads.
        compute = logits.float()
        rms = compute.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return (compute / rms).to(dtype=logits.dtype)

    def forward(self, query_feature, context_feature):
        _validate_cross_attention_inputs(query_feature, context_feature)
        query = self.query(self.query_norm(query_feature))
        key = self.key(self.context_norm(context_feature))
        value = self.value(self.context_norm(context_feature))

        query, metadata = _window_partition(query, self.window_size)
        key, key_metadata = _window_partition(key, self.window_size)
        value, value_metadata = _window_partition(value, self.window_size)
        if metadata != key_metadata or metadata != value_metadata:
            raise RuntimeError("Query, key and value window layouts do not match")

        batch_windows, num_tokens, _ = query.shape
        query = query.view(
            batch_windows, num_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key.view(
            batch_windows, num_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = value.view(
            batch_windows, num_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        logits = torch.einsum("bhqd,bhkd->bhqk", query / math.sqrt(self.head_dim), key)
        attention = self.dropout(self._normalize_attention(logits))

        # HyLA target network: attention acts twice with a ReLU in between.
        hidden = torch.einsum("bhqk,bhkd->bqkd", attention, value)
        hidden = F.relu(hidden, inplace=False)
        output = torch.einsum("bhqk,bqkd->bqhd", attention, hidden)
        output = output.reshape(batch_windows, num_tokens, -1)
        output = _window_reverse(output, metadata)
        return self.out(output)


class BidirectionalHyLAFusion(nn.Module):
    """Bidirectional cross-HyLA with joint-modal queries."""

    def __init__(
        self,
        channels,
        attention_dim=64,
        num_heads=4,
        window_size=7,
        attention_norm="rms_head",
        dropout=0.0,
    ):
        super().__init__()
        kwargs = dict(
            channels=channels,
            attention_dim=attention_dim,
            num_heads=num_heads,
            window_size=window_size,
            attention_norm=attention_norm,
            dropout=dropout,
            query_channels=2 * channels,
        )
        self.rgb_from_evt = HyLACrossAttention2D(**kwargs)
        self.evt_from_rgb = HyLACrossAttention2D(**kwargs)
        self.rgb_scale = nn.Parameter(torch.zeros(1))
        self.evt_scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, evt):
        _validate_pair(rgb, evt)
        joint_query = torch.cat((rgb, evt), dim=1)
        rgb_delta = self.rgb_from_evt(joint_query, evt)
        evt_delta = self.evt_from_rgb(joint_query, rgb)
        return rgb + self.rgb_scale * rgb_delta, evt + self.evt_scale * evt_delta


class SharedModalMLP(nn.Module):
    """Parameter-shared residual MLP that maps either modality into one space."""

    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        hidden = max(int(channels * expansion), channels)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Zero-init residual so the block starts as identity.
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        return x + self.dropout(self.fc2(self.act(self.norm(self.fc1(x)))))


class SharedMLPHyLAFusion(nn.Module):
    """Align with shared MLP, enhance with HyLA, then gate-sum like full-scale hyla.

    Pipeline (per FPN scale):
      rgb, evt --shared MLP--> aligned space
           --bidirectional HyLA--> enhanced modalities
           --rgb'+evt'+gate correction--> fused feature for the neck
    """

    def __init__(
        self,
        channels,
        attention_dim=64,
        num_heads=4,
        window_size=7,
        attention_norm="rms_head",
        dropout=0.0,
        mlp_expansion=2,
    ):
        super().__init__()
        self.shared_mlp = SharedModalMLP(
            channels, expansion=mlp_expansion, dropout=dropout
        )
        self.hyla = BidirectionalHyLAFusion(
            channels=channels,
            attention_dim=attention_dim,
            num_heads=num_heads,
            window_size=window_size,
            attention_norm=attention_norm,
            dropout=dropout,
        )
        # Same synthesis as _HyLAOnlyFusion (full-scale hyla).
        self.modality_gate = nn.Conv2d(2 * channels, channels, 1)
        self.output_projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def enhance_modalities(self, rgb, evt):
        """Shared MLP + bidirectional HyLA, before the gate-sum fusion."""
        _validate_pair(rgb, evt)
        rgb = self.shared_mlp(rgb)
        evt = self.shared_mlp(evt)
        return self.hyla(rgb, evt)

    def fuse_enhanced(self, rgb, evt):
        """Gate-sum of modalities that have already passed enhance_modalities."""
        _validate_pair(rgb, evt)
        joint = torch.cat((rgb, evt), dim=1)
        gate = torch.sigmoid(self.modality_gate(joint))
        correction = self.output_projection(
            torch.cat((gate * rgb, (1.0 - gate) * evt), dim=1)
        )
        return rgb + evt + correction

    def forward(self, rgb, evt):
        rgb, evt = self.enhance_modalities(rgb, evt)
        return self.fuse_enhanced(rgb, evt)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        average = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        average = x.mean(dim=1, keepdim=True)
        maximum = x.max(dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


def _dynamic_depthwise_conv2d(x, kernels, padding=1):
    """Apply one depthwise kernel per batch item and channel."""
    batch, channels, height, width = x.shape
    expected = (batch, channels, 3, 3)
    if kernels.shape != expected:
        raise ValueError(
            f"Expected dynamic kernels with shape {expected}, got {tuple(kernels.shape)}"
        )
    grouped_input = x.reshape(1, batch * channels, height, width)
    grouped_kernels = kernels.reshape(batch * channels, 1, 3, 3)
    output = F.conv2d(
        grouped_input, grouped_kernels, padding=padding, groups=batch * channels
    )
    return output.view(batch, channels, output.shape[-2], output.shape[-1])


class MFGFusionBlock(nn.Module):
    """Batch-safe MFG dynamic cross-filtering with a sum-compatible output."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        joint_channels = 2 * channels
        self.channel_attention = ChannelAttention(joint_channels, reduction)
        self.spatial_attention = SpatialAttention()

        self.rgb_kernel_key = nn.Conv2d(joint_channels, channels, 1)
        self.rgb_kernel_query = nn.Conv2d(joint_channels, 9, 1)
        self.evt_kernel_key = nn.Conv2d(joint_channels, channels, 1)
        self.evt_kernel_query = nn.Conv2d(joint_channels, 9, 1)

        self.rgb_scale = nn.Parameter(torch.zeros(1))
        self.evt_scale = nn.Parameter(torch.zeros(1))
        self.modality_gate = nn.Conv2d(joint_channels, channels, 1)
        self.output_projection = nn.Conv2d(joint_channels, channels, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _make_kernels(key, query):
        batch, channels, height, width = key.shape
        key = key.flatten(2)
        query = query.flatten(2)
        # Spatial mean prevents kernel magnitude from depending on feature-map size.
        kernels = torch.einsum("bcn,bkn->bck", key, query) / (height * width)
        return kernels.view(batch, channels, 3, 3)

    def forward_modalities(self, rgb, evt):
        _validate_pair(rgb, evt)
        joint = torch.cat((rgb, evt), dim=1)
        joint = joint * self.channel_attention(joint)
        joint = joint * self.spatial_attention(joint)

        rgb_kernels = self._make_kernels(
            self.rgb_kernel_key(joint), self.rgb_kernel_query(joint)
        )
        evt_kernels = self._make_kernels(
            self.evt_kernel_key(joint), self.evt_kernel_query(joint)
        )
        rgb_delta = torch.tanh(_dynamic_depthwise_conv2d(evt, rgb_kernels))
        evt_delta = torch.tanh(_dynamic_depthwise_conv2d(rgb, evt_kernels))
        rgb = rgb + self.rgb_scale * rgb_delta
        evt = evt + self.evt_scale * evt_delta
        return rgb, evt

    def fuse_modalities(self, rgb, evt):
        joint = torch.cat((rgb, evt), dim=1)
        gate = torch.sigmoid(self.modality_gate(joint))
        gated = torch.cat((gate * rgb, (1.0 - gate) * evt), dim=1)
        # The zero-initialized correction makes the initial output exactly sum fusion.
        return rgb + evt + self.output_projection(gated)

    def forward(self, rgb, evt):
        rgb, evt = self.forward_modalities(rgb, evt)
        return self.fuse_modalities(rgb, evt)


class HyLAMFGFusionBlock(nn.Module):
    """Serial bidirectional HyLA interaction followed by MFG local fusion."""

    def __init__(
        self,
        channels,
        attention_dim=64,
        num_heads=4,
        window_size=7,
        attention_norm="rms_head",
        dropout=0.0,
        mfg_reduction=16,
    ):
        super().__init__()
        self.hyla = BidirectionalHyLAFusion(
            channels=channels,
            attention_dim=attention_dim,
            num_heads=num_heads,
            window_size=window_size,
            attention_norm=attention_norm,
            dropout=dropout,
        )
        self.mfg = MFGFusionBlock(channels, reduction=mfg_reduction)

    def forward(self, rgb, evt):
        rgb, evt = self.hyla(rgb, evt)
        return self.mfg(rgb, evt)


def build_modal_fusion(
    fusion_type,
    channels,
    attention_dim=64,
    num_heads=4,
    window_size=7,
    attention_norm="rms_head",
    dropout=0.0,
    mfg_reduction=16,
    mlp_expansion=2,
):
    if fusion_type == "sum":
        return None
    if fusion_type == "mfg":
        return MFGFusionBlock(channels, reduction=mfg_reduction)
    if fusion_type == "hyla":
        return _HyLAOnlyFusion(
            channels,
            attention_dim,
            num_heads,
            window_size,
            attention_norm,
            dropout,
        )
    if fusion_type == "hyla_shared_mlp":
        return SharedMLPHyLAFusion(
            channels=channels,
            attention_dim=attention_dim,
            num_heads=num_heads,
            window_size=window_size,
            attention_norm=attention_norm,
            dropout=dropout,
            mlp_expansion=mlp_expansion,
        )
    if fusion_type == "hyla_mfg":
        return HyLAMFGFusionBlock(
            channels,
            attention_dim,
            num_heads,
            window_size,
            attention_norm,
            dropout,
            mfg_reduction,
        )
    raise ValueError(
        f"Unknown fusion_type '{fusion_type}'. "
        "Expected one of: sum, mfg, hyla, hyla_shared_mlp, hyla_mfg"
    )


class _HyLAOnlyFusion(nn.Module):
    def __init__(
        self,
        channels,
        attention_dim,
        num_heads,
        window_size,
        attention_norm,
        dropout,
    ):
        super().__init__()
        self.hyla = BidirectionalHyLAFusion(
            channels,
            attention_dim,
            num_heads,
            window_size,
            attention_norm,
            dropout,
        )
        self.modality_gate = nn.Conv2d(2 * channels, channels, 1)
        self.output_projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, rgb, evt):
        rgb, evt = self.hyla(rgb, evt)
        joint = torch.cat((rgb, evt), dim=1)
        gate = torch.sigmoid(self.modality_gate(joint))
        correction = self.output_projection(
            torch.cat((gate * rgb, (1.0 - gate) * evt), dim=1)
        )
        return rgb + evt + correction
