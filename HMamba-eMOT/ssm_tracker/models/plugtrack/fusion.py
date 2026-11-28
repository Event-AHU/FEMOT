import torch.nn as nn

from models.plugtrack.blender import AdaptiveBlendingWeightGenerator
from models.plugtrack.contextual_motion_encoder import ContextualMotionEncoder
from models.plugtrack.kalman_filter import KalmanFilter


class PlugTrack(nn.Module):
    """Blend a frozen displacement predictor with a constant-velocity Kalman box."""

    def __init__(self, process_var=1e-5, meas_var=5e-4):
        super().__init__()
        self.encoder = ContextualMotionEncoder()
        self.blender = AdaptiveBlendingWeightGenerator()
        self.kf = KalmanFilter(device='cpu', process_var=process_var, meas_var=meas_var)

    def _kalman(self, history):
        self.kf.device = history.device
        return self.kf.predict(history)

    def forward(self, history, bbox_mp, bbox_gt=None, kf_history=None):
        bbox_kf, uncertainty = self._kalman(history if kf_history is None else kf_history)
        context = self.encoder(history, bbox_kf, bbox_mp, uncertainty)
        return self.blender(
            context,
            bbox_kf,
            bbox_mp,
            bbox_gt,
            training=bbox_gt is not None,
        )
