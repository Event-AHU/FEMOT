#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .modal_fusion import BidirectionalHyLAFusion, build_modal_fusion
from .network_blocks import BaseConv, CSPLayer, DWConv


class DualModalYOLOPAFPN(nn.Module):
    """
    Dual CSPDarknet backbones with configurable dark3/4/5 fusion and a shared PA-FPN.
    Input: (B, 6, H, W) with RGB in [:3] and event in [3:].
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=(256, 512, 1024),
        depthwise=False,
        act="silu",
        fusion_type="sum",
        fusion_scales=("dark3", "dark4", "dark5"),
        hyla_dims=(64, 96, 128),
        hyla_heads=(4, 4, 4),
        hyla_windows=(7, 7, 0),
        hyla_norm="rms_head",
        fusion_dropout=0.0,
        mfg_reduction=16,
        mlp_expansion=2,
        early_hyla=False,
        early_hyla_dim=64,
        early_hyla_heads=4,
        early_hyla_window=7,
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.fusion_type = fusion_type
        self.fusion_scales = tuple(fusion_scales)
        self.capture_modal_features = False
        self.last_modal_features = None
        self.backbone_rgb = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.backbone_evt = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.early_hyla = None
        if early_hyla:
            self.early_hyla = BidirectionalHyLAFusion(
                channels=int(128 * width),
                attention_dim=early_hyla_dim,
                num_heads=early_hyla_heads,
                window_size=early_hyla_window,
                attention_norm=hyla_norm,
                dropout=fusion_dropout,
            )

        feature_names = ("dark3", "dark4", "dark5")
        settings = (hyla_dims, hyla_heads, hyla_windows)
        if any(len(values) != len(feature_names) for values in settings):
            raise ValueError("HyLA dims, heads and windows must each contain three values")
        unknown_scales = set(self.fusion_scales) - set(feature_names)
        if unknown_scales:
            raise ValueError(f"Unknown fusion scales: {sorted(unknown_scales)}")
        self.fusion_blocks = nn.ModuleDict()
        for index, feature_name in enumerate(feature_names):
            if feature_name not in self.fusion_scales or fusion_type == "sum":
                continue 
            channels = int(in_channels[index] * width)
            self.fusion_blocks[feature_name] = build_modal_fusion(
                fusion_type=fusion_type,
                channels=channels,
                attention_dim=hyla_dims[index],
                num_heads=hyla_heads[index],
                window_size=hyla_windows[index],
                attention_norm=hyla_norm,
                dropout=fusion_dropout,
                mfg_reduction=mfg_reduction,
                mlp_expansion=mlp_expansion,
            )

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    @staticmethod
    def _split_modalities(x):
        if isinstance(x, (tuple, list)):
            rgb, evt = x
            return rgb, evt
        if x.shape[1] == 6:
            return x[:, :3], x[:, 3:]
        # YOLOX get_model_info() probes with 3-channel dummy input
        if x.shape[1] == 3:
            return x, x
        raise ValueError(f"Expected 6-channel input or (rgb, evt), got shape {tuple(x.shape)}")

    def _forward_backbones(self, rgb, evt):
        if self.early_hyla is None:
            return self.backbone_rgb(rgb), self.backbone_evt(evt)

        rgb = self.backbone_rgb.stem(rgb)
        evt = self.backbone_evt.stem(evt)
        rgb = self.backbone_rgb.dark2(rgb)
        evt = self.backbone_evt.dark2(evt)
        rgb, evt = self.early_hyla(rgb, evt)

        rgb3 = self.backbone_rgb.dark3(rgb)
        evt3 = self.backbone_evt.dark3(evt)
        rgb4 = self.backbone_rgb.dark4(rgb3)
        evt4 = self.backbone_evt.dark4(evt3)
        rgb5 = self.backbone_rgb.dark5(rgb4)
        evt5 = self.backbone_evt.dark5(evt4)
        return (
            {"dark3": rgb3, "dark4": rgb4, "dark5": rgb5},
            {"dark3": evt3, "dark4": evt4, "dark5": evt5},
        )

    def _fuse_backbone(self, x):
        rgb, evt = self._split_modalities(x)
        out_rgb, out_evt = self._forward_backbones(rgb, evt)
        if self.capture_modal_features:
            self.last_modal_features = (out_rgb, out_evt)
        fused = {}
        for feature_name in ("dark3", "dark4", "dark5"):
            if feature_name in self.fusion_blocks:
                fused[feature_name] = self.fusion_blocks[feature_name](
                    out_rgb[feature_name], out_evt[feature_name]
                )
            else:
                fused[feature_name] = out_rgb[feature_name] + out_evt[feature_name]
        x2, x1, x0 = fused["dark3"], fused["dark4"], fused["dark5"]
        return x2, x1, x0

    def forward_modal_features(self, x):
        """Return modality-specific backbone features before fusion."""
        rgb, evt = self._split_modalities(x)
        return self._forward_backbones(rgb, evt)

    def forward_hyla_enhanced_features(self, x):
        """Return rgb/evt after shared MLP + HyLA and before gate-sum fusion."""
        rgb, evt = self._split_modalities(x)
        out_rgb, out_evt = self._forward_backbones(rgb, evt)
        enhanced_rgb, enhanced_evt = {}, {}
        for feature_name in ("dark3", "dark4", "dark5"):
            block = (
                self.fusion_blocks[feature_name]
                if feature_name in self.fusion_blocks
                else None
            )
            if block is None or not hasattr(block, "enhance_modalities"):
                raise RuntimeError(
                    f"{feature_name} has no HyLA enhance_modalities; "
                    "this extractor requires hyla_shared_mlp fusion"
                )
            enhanced_rgb[feature_name], enhanced_evt[feature_name] = (
                block.enhance_modalities(out_rgb[feature_name], out_evt[feature_name])
            )
        return enhanced_rgb, enhanced_evt

    def forward_hyla_fused_rgb_features(self, x):
        """dark3 gate-sum fusion as rgb, enhanced evt still before the gate.

        The rgb map is the neck input at dark3: enhanced rgb + enhanced evt
        plus the gated correction. Event features stay the pre-gate pair so
        an event hyperedge can keep its own modality.
        """
        rgb, evt = self._split_modalities(x)
        out_rgb, out_evt = self._forward_backbones(rgb, evt)
        block = (
            self.fusion_blocks["dark3"]
            if "dark3" in self.fusion_blocks
            else None
        )
        if block is None or not hasattr(block, "fuse_enhanced"):
            raise RuntimeError(
                "dark3 has no HyLA fuse_enhanced; "
                "this extractor requires hyla_shared_mlp fusion"
            )
        enhanced_rgb, enhanced_evt = block.enhance_modalities(
            out_rgb["dark3"], out_evt["dark3"]
        )
        fused = block.fuse_enhanced(enhanced_rgb, enhanced_evt)
        return {"dark3": fused}, {"dark3": enhanced_evt}

    def forward(self, x):
        x2, x1, x0 = self._fuse_backbone(x)

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0
