import unittest

import torch

from models.MambaTrack import MambaTrack
from models.spatial_ssm_mod import SpatialPriorSSMModulator


class SpatialSSMModTest(unittest.TestCase):
    def test_modulator_identity_when_mlps_zero(self):
        mod = SpatialPriorSSMModulator(
            d_ctx=8, d_inner=16, d_state=4, alpha_dt=0.1, alpha_b=0.1, alpha_c=0.1
        )
        batch, seqlen = 2, 5
        dt = torch.randn(batch, 16, seqlen)
        B = torch.randn(batch, 4, seqlen)
        C = torch.randn(batch, 4, seqlen)
        x = torch.randn(batch, 16, seqlen)
        prior = torch.randn(batch, 8)
        dt_out, b_out, c_out = mod(dt, B, C, x, prior)
        self.assertTrue(torch.allclose(dt_out, dt, atol=1e-6))
        self.assertTrue(torch.allclose(b_out, B, atol=1e-6))
        self.assertTrue(torch.allclose(c_out, C, atol=1e-6))

    def test_dt_mul_identity_and_scale(self):
        mod = SpatialPriorSSMModulator(
            d_ctx=4, d_inner=8, d_state=2, alpha_dt=1.0, dt_combine="mul"
        )
        dt = torch.ones(1, 8, 3)
        B = torch.ones(1, 2, 3)
        C = torch.ones(1, 2, 3)
        x = torch.randn(1, 8, 3)
        prior = torch.randn(1, 4)
        dt_out, b_out, c_out = mod(dt, B, C, x, prior)
        self.assertTrue(torch.allclose(dt_out, dt, atol=1e-6))
        self.assertTrue(torch.allclose(b_out, B, atol=1e-6))
        self.assertTrue(torch.allclose(c_out, C, atol=1e-6))
        with torch.no_grad():
            mod.mlp_dt[-1].weight.zero_()
            mod.mlp_dt[-1].bias.fill_(1.0)
        dt_out, _, _ = mod(dt, B, C, x, prior)
        scale = 1.0 + torch.tanh(torch.tensor(1.0))
        self.assertTrue(torch.allclose(dt_out, dt * scale, atol=1e-5))

    def test_last_k_mask(self):
        mod = SpatialPriorSSMModulator(
            d_ctx=4, d_inner=8, d_state=2, alpha_dt=1.0, last_k=2
        )
        with torch.no_grad():
            mod.mlp_dt[-1].weight.fill_(0.01)
            mod.mlp_dt[-1].bias.fill_(0.5)
        dt = torch.zeros(1, 8, 6)
        B = torch.ones(1, 2, 6)
        C = torch.ones(1, 2, 6)
        x = torch.randn(1, 8, 6)
        prior = torch.randn(1, 4)
        dt_out, _, _ = mod(dt, B, C, x, prior)
        self.assertTrue(torch.allclose(dt_out[..., :-2], torch.zeros_like(dt_out[..., :-2])))
        self.assertFalse(torch.allclose(dt_out[..., -2:], torch.zeros_like(dt_out[..., -2:])))

    def test_mambatrack_forward_and_backward(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for mamba_ssm selective scan")
        cfgs = {
            "d_m": 32,
            "d_state": 4,
            "L": 1,
            "avg_pool_out_dim": [1, 8],
            "pred_head_dims": [8, 4],
            "use_hyper_b": False,
            "use_spatial_hyper": False,
            "use_spatial_ssm_mod": True,
            "spatial_hyper_dim": 8,
            "ssm_mod_alpha_dt": 0.1,
            "ssm_mod_alpha_b": 0.1,
            "ssm_mod_alpha_c": 0.1,
            "ssm_mod_last_k": 0,
        }
        device = torch.device("cuda")
        model = MambaTrack(cfgs).to(device)
        x = torch.randn(2, 3, 5, 4, device=device)
        label = torch.randn(2, 3, 4, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        incidence = torch.ones(2, 3, 2, device=device)
        hyper_inputs = {
            name: incidence for name in ("spatial", "rgb", "evt", "motion")
        }
        loss = model(x, label, hyper_inputs=hyper_inputs, vertex_mask=mask)
        loss.backward()
        self.assertTrue(loss.ndim == 0)

    def test_mutex_flags(self):
        cfgs = {
            "d_m": 16,
            "d_state": 4,
            "L": 1,
            "avg_pool_out_dim": [1, 8],
            "pred_head_dims": [8, 4],
            "use_spatial_hyper": True,
            "use_spatial_ssm_mod": True,
            "spatial_hyper_dim": 8,
        }
        with self.assertRaises(ValueError):
            MambaTrack(cfgs)


if __name__ == "__main__":
    unittest.main()
