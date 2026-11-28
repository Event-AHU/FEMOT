import torch


class KalmanFilter:
    def __init__(self, device: torch.device, process_var: float = 1.0, meas_var: float = 1.0):
        self.device = device
        self.kf_process_var = process_var
        self.kf_meas_var = meas_var

    def predict(self, tracklets: torch.Tensor):
        batch, T, _ = tracklets.shape

        z = tracklets[:, :, :4]

        x_state = torch.cat([tracklets[:, 0, :4], tracklets[:, 0, 4:]], dim=-1)

        P = torch.eye(8, device=self.device).unsqueeze(0).repeat(batch, 1, 1) * 10.0

        F_single = torch.eye(8, device=self.device)
        F_single[:4, 4:] = torch.eye(4, device=self.device)
        F = F_single.unsqueeze(0).repeat(batch, 1, 1)

        H_single = torch.zeros(4, 8, device=self.device)
        H_single[:4, :4] = torch.eye(4, device=self.device)
        H = H_single.unsqueeze(0).repeat(batch, 1, 1)

        Q = torch.eye(8, device=self.device).unsqueeze(0).repeat(batch, 1, 1) * self.kf_process_var
        R = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch, 1, 1) * self.kf_meas_var

        I8 = torch.eye(8, device=self.device).unsqueeze(0).repeat(batch, 1, 1)

        normalized_innovations = []

        for t in range(T):
            x_state = (F @ x_state.unsqueeze(-1)).squeeze(-1)
            P = F @ P @ F.transpose(-1, -2) + Q

            zt = z[:, t].unsqueeze(-1)
            y = zt - (H @ x_state.unsqueeze(-1))
            S = H @ P @ H.transpose(-1, -2) + R

            S_diag = torch.diagonal(S, dim1=-2, dim2=-1)
            normalized_innovation = (y.squeeze(-1) ** 2) / (S_diag + 1e-10)
            normalized_innovations.append(normalized_innovation)

            K = P @ H.transpose(-1, -2) @ torch.inverse(S)

            x_state = (x_state.unsqueeze(-1) + K @ y).squeeze(-1)

            P = (I8 - K @ H) @ P

        if len(normalized_innovations) >= 3:
            recent_nis = torch.stack(normalized_innovations[-3:], dim=1)
            nis_mean = recent_nis.mean(dim=1)
            nis_var = recent_nis.var(dim=1)
            uncertainty = nis_mean + torch.sqrt(nis_var + 1e-6)
        elif len(normalized_innovations) > 0:
            uncertainty = normalized_innovations[-1]
        else:
            uncertainty = torch.ones(batch, 4, device=self.device)

        x_pred = (F @ x_state.unsqueeze(-1)).squeeze(-1)
        bbox_pred = x_pred[:, :4]

        return bbox_pred, uncertainty
