# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .mlp import MLP
from .ffn import FFN
from .backbone import BackboneWithPE
from .deformable_transformer import DeformableTransformer
from .query_updater import build as build_query_updater
from .utils import get_clones, pos_to_pos_embed

from .backbone import build as build_backbone_with_pe
from .deformable_transformer import build as build_deformable_transformer

from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid

from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
from torch.fft import fftn, ifftn


class FFTFusion(nn.Module):
    def __init__(self, resnet_dim=256, complex=True):
        super().__init__()
        self.resnet_dim = resnet_dim
        self.cat_dim = resnet_dim * 2

        self.rgb_ln = nn.LayerNorm(self.resnet_dim)
        self.event_ln = nn.LayerNorm(self.resnet_dim)
        self.rgb_event_11_conv = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_depthwise = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=3, stride=1, padding=1, groups=self.cat_dim)
        
        # 振幅和相位处理模块（代替原来的实部虚部融合）
        self.amp_processor = nn.Sequential(
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        )
        
        self.phase_processor = nn.Sequential(
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        )
        
        self.re_depthwise_11_conv = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_amp_11_conv = nn.Conv2d(self.cat_dim, self.resnet_dim, kernel_size=1)
        self.re_phase_11_conv = nn.Conv2d(self.cat_dim, self.resnet_dim, kernel_size=1)
        self.re_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)

        # rgb branch - 振幅和相位处理
        self.r_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.r_depthwise = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=3, stride=1, padding=1, groups=self.resnet_dim)
        self.r_depthwise_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.r_amp_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.r_phase_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.r_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.r_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.r_preprocessBack = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        
        # event branch - 振幅和相位处理
        self.e_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.e_depthwise = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=3, stride=1, padding=1, groups=self.resnet_dim)
        self.e_depthwise_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.e_amp_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.e_phase_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.e_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.e_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.e_preprocessBack = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        
    def _make_multiply_and_softmax(self, vis, event):
        # 用于振幅和相位特征融合的函数
        # Normalize features to avoid numerical instability
        vis = F.normalize(vis, dim=1)
        event = F.normalize(event, dim=1)

        # Flatten and multiply
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = event.view(event.size(0), event.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)

        # Apply softmax
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))

        # Residual connection
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, rgb, event):
        B, C, H, W = rgb.shape
        
        # Layer normalization
        ori_rgb = rgb.clone()
        ori_event = event.clone()
        rgb = self.rgb_ln(rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        event = self.event_ln(event.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # RGB-EVENT fusion path
        re = torch.cat([rgb, event], dim=1).contiguous()
        re = self.rgb_event_11_conv(re)
        re = self.re_depthwise(re)
        re = self.re_depthwise_11_conv(re).contiguous()
        
        # FFT Transform using rfft2
        rgb_event_freq = torch.fft.rfft2(re, norm='backward')
        
        # Extract amplitude and phase
        re_amp = torch.abs(rgb_event_freq)
        re_phase = torch.angle(rgb_event_freq)
        
        # Process amplitude and phase
        re_amp = self.amp_processor(re_amp)
        re_phase = self.phase_processor(re_phase)
        
        # Adaptive fusion of amplitude and phase
        re_amp = self._make_multiply_and_softmax(re_amp, re_amp)  # Self attention for regularization
        re_phase = self._make_multiply_and_softmax(re_phase, re_phase)  # Keep phase as is
        re_amp = self.re_amp_11_conv(re_amp)
        re_phase = self.re_phase_11_conv(re_phase)
        
        # Reconstruct complex tensor
        filter_real = re_amp * torch.cos(re_phase)
        filter_imag = re_amp * torch.sin(re_phase)
        f = torch.cat((filter_real, filter_imag), dim=1)
        f = F.relu(self.re_conv1_fft(f))
        f = self.re_conv2_fft(f).float()
        filter_real, filter_imag = torch.chunk(f, 2, dim=1)
        filter_freq = torch.complex(filter_real, filter_imag)
        
        # Inverse FFT
        enhanced_filter = torch.fft.irfft2(filter_freq, s=(H, W), norm='backward')
        
        # Convert filter to complex for multiplication with rgb/event frequency
        filter_amp = torch.abs(filter_freq)
        filter_phase = torch.angle(filter_freq)

        # RGB branch
        rgb = self.r_11conv(rgb)
        rgb = self.r_depthwise(rgb)
        rgb = self.r_depthwise_11conv(rgb).contiguous()
        
        # FFT for RGB
        rgb_freq = torch.fft.rfft2(rgb, norm='backward')
        rgb_amp = torch.abs(rgb_freq)
        rgb_phase = torch.angle(rgb_freq)
        
        # Amplitude and phase processing for RGB
        rgb_amp_processed = self.r_amp_processor(rgb_amp)
        rgb_phase_processed = self.r_phase_processor(rgb_phase)
        
        # Fusion with filter in frequency domain
        # Amplitude multiplication, phase addition (complex multiplication)
        fused_rgb_amp = self._make_multiply_and_softmax(rgb_amp_processed, filter_amp)
        fused_rgb_phase = self._make_multiply_and_softmax(rgb_phase_processed, filter_phase)
        
        # Reconstruct complex tensor
        rgb_real = fused_rgb_amp * torch.cos(fused_rgb_phase)
        rgb_imag = fused_rgb_amp * torch.sin(fused_rgb_phase)
        f = torch.cat((rgb_real, rgb_imag), dim=1)
        f = F.relu(self.r_conv1_fft(f))
        f = self.r_conv2_fft(f).float()
        rgb_real, rgb_imag = torch.chunk(f, 2, dim=1)
        rgb_freq_fused = torch.complex(rgb_real, rgb_imag)
        
        # Inverse FFT
        rgb_out = torch.fft.irfft2(rgb_freq_fused, s=(H, W), norm='backward')
        rgb_out = self.r_preprocessBack(rgb_out)
        rgb_out = rgb_out + ori_rgb
        
        # EVENT branch (similar to RGB branch)
        event_spatial = self.e_11conv(event)
        event_spatial = self.e_depthwise(event_spatial)
        event_spatial = self.e_depthwise_11conv(event_spatial).contiguous()
        
        # FFT for EVENT
        event_freq = torch.fft.rfft2(event_spatial, norm='backward')
        event_amp = torch.abs(event_freq)
        event_phase = torch.angle(event_freq)
        
        # Amplitude and phase processing for EVENT
        event_amp_processed = self.e_amp_processor(event_amp)
        event_phase_processed = self.e_phase_processor(event_phase)
        
        # Fusion with filter in frequency domain
        fused_event_amp = self._make_multiply_and_softmax(event_amp_processed, filter_amp)
        fused_event_phase = self._make_multiply_and_softmax(event_phase_processed, filter_phase)
        
        
        # Reconstruct complex tensor
        event_real = fused_event_amp * torch.cos(fused_event_phase)
        event_imag = fused_event_amp * torch.sin(fused_event_phase)
        f = torch.cat((event_real, event_imag), dim=1)
        f = F.relu(self.e_conv1_fft(f))
        f = self.e_conv2_fft(f).float()
        event_real, event_imag = torch.chunk(f, 2, dim=1)
        event_freq_fused = torch.complex(event_real, event_imag)
        
        # Inverse FFT
        event_out = torch.fft.irfft2(event_freq_fused, s=(H, W), norm='backward')
        event_out = self.e_preprocessBack(event_out)
        event_out = event_out + ori_event
        
        fused = rgb_out + event_out + enhanced_filter
        return fused
    
    def forward_srcs(self, rgb_srcs, event_srcs):
        final_scrs = []
        for rgb, event in zip(rgb_srcs, event_srcs):
            fused = self.forward(rgb, event)
            final_scrs.append(fused)
        return final_scrs    

class MeMOTR(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False):
        super(MeMOTR, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize

        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 4))  # (N_det, 4)
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))   # (N_det, 2C)
        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[i], out_channels=self.hidden_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[-1], out_channels=self.hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        self.feature_projs = nn.ModuleList(feature_proj_list)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.transformer.get_n_dec_layers())])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())])

        self.enhancer = FFTFusion()

    def forward(self, frame: NestedTensor, event_frame: NestedTensor, tracks: list[TrackInstances]):
        # frame: torch.Size([1, 3, 992, 1344])
        # event_frame: torch.Size([1, 3, 992, 1344])
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)
   
        if self.use_checkpoint and self.checkpoint_level != 3:
            features, pos = checkpoint(self.backbone, frame, use_reentrant=False)  
            event_features, event_pos = checkpoint(self.backbone, event_frame, use_reentrant=False)
        else:
            features, pos = self.backbone(frame)  # torch.Size([1, 512, 124, 168]) / torch.Size([1, 1024, 62, 84]) / torch.Size([1, 2048, 31, 42])
            event_features, event_pos = self.backbone(event_frame)
    
        # features = features + event_features
        event_srcs, event_masks = [], []

        for layer, feat in enumerate(event_features):
            event_src, event_mask = feat.decompose()
            event_srcs.append(self.feature_projs[layer](event_src))
            event_masks.append(event_mask)  

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)  
        # torch.Size([1, 256, 124, 168]) / torch.Size([1, 256, 62, 84]) / torch.Size([1, 256, 31, 42])
 
        if self.n_feature_levels > len(srcs):  # self.n_feature_levels = 4
            srcs_len = len(srcs)
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    src = self.feature_projs[layer](features[-1].tensors)
                    event_src = self.feature_projs[layer](event_features[-1].tensors)
                else:
                    src = self.feature_projs[layer](srcs[-1])
                    event_src = self.feature_projs[layer](event_srcs[-1])
                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[0].to(torch.bool)
                pos.append(self.backbone.position_embedding(NestedTensor(src, mask)).to(src.device))
                # srcs.append(src + event_src)   # tomx
                srcs.append(src)
                event_srcs.append(event_src)
                masks.append(mask)

                # event stream branch
                # event_mask = event_features.masks
                # event_mask = F.interpolate(event_mask[None, ...].float(), size=event_src.shape[-2:])[0].to(torch.bool)
                # event_pos.append(self.backbone.position_embedding(NestedTensor(event_src, event_mask)).to(event_src.device))
                # event_srcs.append(event_src)
                # event_masks.append(event_mask)
        
        # torch.Size([1, 256, 124, 168]) / torch.Size([1, 256, 62, 84]) / torch.Size([1, 256, 31, 42]) / torch.Size([1, 256, 16, 21])
        
        srcs = self.enhancer.forward_srcs(srcs, event_srcs)
        # torch.Size([1, 256, 124, 168]) / torch.Size([1, 256, 62, 84]) / torch.Size([1, 256, 31, 42]) / torch.Size([1, 256, 16, 21])
        
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]

        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)      # torch.Size([1, 300, 4]) [B, Nd + Nq, 4]
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)                # torch.Size([1, 300, 256]) [B, Nd + Nq, C]
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)                  # torch.Size([1, 300])

        # DETR:
        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            query_embed=query_embed,
            ref_pts=reference_points,
            query_mask=query_mask
        )
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        
        output_classes, output_bboxes = [], []
        assert outputs.ndim == 4, f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, " \
                                  f"but get n_dim={outputs.ndim}"
        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            if reference.shape[-1] == 4:
                bbox_tmp += reference
            else:
                assert reference.shape[-1] == 2, f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)

            if self.visualize:
                torch.save(reference[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_ref_pts_layer_{level}.tensor")
                torch.save(reference[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_ref_pts_layer_{level}.tensor")
                torch.save(output_class[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_logits_layer_{level}.tensor")
                torch.save(output_class[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_logits_layer_{level}.tensor")
                torch.save(output_bbox[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_boxes_layer_{level}.tensor")
                torch.save(output_bbox[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_boxes_layer_{level}.tensor")

        output_classes = torch.stack(output_classes, dim=0)
        output_bboxes = torch.stack(output_bboxes, dim=0)
        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]) if self.use_dab       # (B, Nd+Nq, 4)
            else inverse_sigmoid(inter_references[-2, :, :, :]),                                 # (B, Nd+Nq, 2)
            "query_mask": query_mask,                   # (B, Nd+Nq)
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference)
        }
        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(output_classes=output_classes,
                                                   output_bboxes=output_bboxes,
                                                   query_mask=query_mask,
                                                   queries=inter_queries)
        res["outputs"] = outputs[-1]     # (B, Nd+Nq, C)
        return res

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

    def get_det_reference_points(self) -> torch.Tensor:
        """
        Returns: (Nd, 2)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 4))
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            references[i, :len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: list[TrackInstances]):
        det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)                      # (B, Nd, 2)
        mean, std = 0., 1.
        for b in range(len(tracks)):
            track_query_nums = tracks[b].ref_pts.shape[0]
            if len(tracks[b]) > 0 & self.n_det_queries > 2*track_query_nums:
                noise = torch.randn_like(tracks[b].ref_pts) * std + mean
                det_references[:, :track_query_nums] = tracks[b].ref_pts + noise
                noise = torch.randn_like(tracks[b].ref_pts) * std + mean
                det_references[:, track_query_nums:2*track_query_nums] = tracks[b].ref_pts + noise
        if det_references.shape[-1] == 2:
            det_references = torch.cat(
                (det_references, torch.zeros_like(det_references, device=det_references.device)),
                dim=-1
            )
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)     # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)

    def get_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq, 2C)
        """
        if self.use_dab:
            det_query_embed = self.det_query_embed
            det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)                    # (B, Nd, 2C)
        
        # Track Query Sent to Det Query
        mean, std = 0., 1.
        for b in range(len(tracks)):
            track_query_nums = tracks[b].long_memory.shape[0]
            if len(tracks[b]) > 0 & self.n_det_queries > 2*track_query_nums:
                noise = torch.randn_like(tracks[b].long_memory) * std + mean
                det_query_embed[:, :track_query_nums] = tracks[b].long_memory + noise
                noise = torch.randn_like(tracks[b].output_embed) * std + mean
                det_query_embed[:, track_query_nums:2*track_query_nums] = tracks[b].output_embed + noise             
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)       # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq)
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(self.det_query_embed.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances],
                                 new_tracks: List[TrackInstances],
                                 unmatched_dets: List[TrackInstances] | None,
                                 no_augment: bool = False):
        """
        Query updating.
        """
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
        "FEMOT": 2,  # TODO: added by tomx 
        "DSEC_distorted": 8,
    }
    assert config["DATASET"] in dataset_num_classes, f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = dataset_num_classes[config["DATASET"]]

    backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config)
    query_updater = build_query_updater(config=config)
 
    return MeMOTR(
        backbone=backbone_with_pe,
        transformer=deformable_transformer,
        query_updater=query_updater,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        aux_loss=True,
        with_box_refine=True,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"]
    )
