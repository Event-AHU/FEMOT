import torch
import torch.nn as nn


class ContextualMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracklet_encoder = nn.LSTM(
            input_size=8,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.discrepancy_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        motion_dim = 128
        discrepancy_dim = 32
        uncertainty_dim = 32
        total_dim = motion_dim + discrepancy_dim + uncertainty_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

    def forward(self, tracklet, bbox_kf, bbox_mp, kf_uncertainty):
        _, (hidden, _) = self.tracklet_encoder(tracklet)
        f_motion = hidden[-1]

        prediction_diff = bbox_kf - bbox_mp
        f_diff = self.discrepancy_encoder(prediction_diff)

        f_uncertainty = self.uncertainty_encoder(kf_uncertainty)

        combined_features = torch.cat([f_motion, f_diff, f_uncertainty], dim=1)
        return self.feature_fusion(combined_features)
