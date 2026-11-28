import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBlendingWeightGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        self.alpha_min = 0.3
        self.alpha_max = 0.7
        self.alpha_step = 0.1

        self.alpha_candidates = self._generate_alpha_candidates()

    def _generate_alpha_candidates(self):
        alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        all_combinations = list(itertools.product(alpha_values, repeat=4))
        return torch.tensor(all_combinations, dtype=torch.float32)

    def forward(self, f_context, bbox_kf, bbox_mp, bbox_gt=None, training=True):
        context_emb = f_context

        if training and bbox_gt is not None:
            optimal_alpha = self._exhaustive_search_vectorized(
                context_emb, bbox_kf, bbox_mp, bbox_gt
            )

            predicted_alpha = self._scale_alpha_to_range(self.alpha_predictor(context_emb))
            predictor_loss = F.mse_loss(predicted_alpha, optimal_alpha.detach())

            bbox_final = predicted_alpha * bbox_kf + (1 - predicted_alpha) * bbox_mp

            loss_l1 = F.smooth_l1_loss(bbox_final, bbox_gt, reduction='mean')
            loss_giou = 1 - self.compute_giou_vectorized(bbox_final, bbox_gt).mean()
            loss_final = loss_l1 + loss_giou

            return bbox_final, predicted_alpha, optimal_alpha, loss_final, predictor_loss

        predicted_alpha = self._scale_alpha_to_range(self.alpha_predictor(context_emb))
        bbox_final = predicted_alpha * bbox_kf + (1 - predicted_alpha) * bbox_mp
        return bbox_final, None, None, None, None

    def _scale_alpha_to_range(self, alpha_sigmoid):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_sigmoid

    def _exhaustive_search_vectorized(self, context_emb, bbox_kf, bbox_mp, bbox_gt):
        batch_size = bbox_kf.size(0)
        device = bbox_kf.device

        best_alpha_candidates = self.alpha_candidates.to(device)
        num_candidates = best_alpha_candidates.size(0)

        noise = torch.randn(batch_size, num_candidates, 4, device=device) * 0.1

        alpha_candidates = best_alpha_candidates.unsqueeze(0) + noise
        alpha_candidates = torch.clamp(alpha_candidates, 0.0, 1.0)

        kf_expanded = bbox_kf.unsqueeze(1).expand(-1, num_candidates, -1)
        mp_expanded = bbox_mp.unsqueeze(1).expand(-1, num_candidates, -1)
        gt_expanded = bbox_gt.unsqueeze(1).expand(-1, num_candidates, -1)

        bbox_candidates = alpha_candidates * kf_expanded + (1 - alpha_candidates) * mp_expanded

        l1_losses = F.smooth_l1_loss(
            bbox_candidates, gt_expanded, reduction='none'
        ).mean(dim=2)

        giou_values = self.compute_giou_vectorized(bbox_candidates, gt_expanded)
        giou_losses = 1 - giou_values

        total_losses = l1_losses + giou_losses

        best_indices = torch.argmin(total_losses, dim=1)
        optimal_alphas = alpha_candidates[torch.arange(batch_size), best_indices]

        return optimal_alphas

    def compute_giou_vectorized(self, pred, target):
        # Baseline boxes are center xywh. The released blender treats x, y as
        # the top-left corner; convert to corners before the same GIoU formula.
        pred_x1 = pred[..., 0] - 0.5 * pred[..., 2]
        pred_y1 = pred[..., 1] - 0.5 * pred[..., 3]
        pred_x2 = pred[..., 0] + 0.5 * pred[..., 2]
        pred_y2 = pred[..., 1] + 0.5 * pred[..., 3]

        target_x1 = target[..., 0] - 0.5 * target[..., 2]
        target_y1 = target[..., 1] - 0.5 * target[..., 3]
        target_x2 = target[..., 0] + 0.5 * target[..., 2]
        target_y2 = target[..., 1] + 0.5 * target[..., 3]

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + 1e-10

        iou = inter_area / union_area

        enc_x1 = torch.min(pred_x1, target_x1)
        enc_y1 = torch.min(pred_y1, target_y1)
        enc_x2 = torch.max(pred_x2, target_x2)
        enc_y2 = torch.max(pred_y2, target_y2)
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-10

        giou = iou - (enc_area - union_area) / enc_area
        return giou
