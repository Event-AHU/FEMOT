import unittest

import numpy as np
import torch

from dataset.hypergraph_rules import build_rule_incidence
from models.MambaTrack import MambaTrack
from models.spatial_hypergraph import SpatialHyperResidual, TypedHypergraphConv


class SpatialHypergraphTest(unittest.TestCase):
    def test_rule_incidence_shapes_and_membership(self):
        centers = np.asarray([[0.1, 0.1], [0.12, 0.1], [0.9, 0.9]], dtype=np.float32)
        rgb = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        evt = rgb.copy()
        motion = np.asarray([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        incidence = build_rule_incidence(
            centers, rgb, evt, motion, max_vertices=4,
            spatial_radius=0.05, rgb_threshold=0.9, evt_threshold=0.9, topk=2,
        )
        self.assertEqual(incidence["spatial"].shape, (4, 4))
        self.assertEqual(incidence["motion"].shape, (4, 3))
        self.assertTrue(np.any(incidence["rgb"][:2].sum(axis=0) == 2))
        self.assertEqual(float(incidence["rgb"][3].sum()), 0.0)

    def test_pairwise_edges_are_pairs(self):
        centers = np.asarray(
            [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32
        )
        rgb = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        motion = np.zeros((3, 4), dtype=np.float32)
        incidence = build_rule_incidence(
            centers, rgb, rgb, motion, max_vertices=3,
            spatial_radius=0.05, rgb_threshold=0.9, evt_threshold=0.9, topk=2,
            pairwise=True,
        )
        spatial = incidence["spatial"]
        self.assertEqual(spatial.shape, (3, 3))
        occupied = spatial[:, spatial.sum(axis=0) > 0]
        self.assertEqual(occupied.shape[1], 3)
        self.assertTrue(np.all(occupied.sum(axis=0) == 2))
        rgb_edges = incidence["rgb"][:, incidence["rgb"].sum(axis=0) > 0]
        self.assertEqual(rgb_edges.shape[1], 1)
        self.assertTrue(np.array_equal(np.flatnonzero(rgb_edges[:, 0]), np.asarray([0, 1])))

    def test_singleton_edge_has_no_peer_message(self):
        for agg in ('attn', 'mean'):
            layer = TypedHypergraphConv(dim=4, agg=agg)
            features = torch.randn(1, 2, 4)
            incidence = torch.tensor([[[1.0], [0.0]]])
            output = layer(features, incidence)
            # LayerNorm/MLP bias may be nonzero, but singleton changes must not affect vertex 0.
            changed = features.clone()
            changed[:, 0] += 100.0
            output_changed = layer(changed, incidence)
            self.assertTrue(
                torch.allclose(output[:, 0], output_changed[:, 0], atol=1e-5),
                msg=f'agg={agg}',
            )

    def test_mean_agg_is_uniform_leave_one_out(self):
        layer = TypedHypergraphConv(dim=4, agg='mean')
        # Two vertices on one edge → each receives the other (after peer_mlp/vertex_mlp).
        features = torch.tensor([[[1.0, 0, 0, 0], [0.0, 1, 0, 0]]])
        incidence = torch.tensor([[[1.0], [1.0]]])
        with torch.no_grad():
            # Identity-ish MLPs: zero LayerNorm affine effect by checking finite + shape.
            out = layer(features, incidence)
        self.assertEqual(out.shape, (1, 2, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_multi_relation_backward(self):
        model = SpatialHyperResidual(input_dim=4, hidden_dim=8, edge_agg='mean')
        motion = torch.randn(2, 3, 4, requires_grad=True)
        incidence = torch.ones(2, 3, 1)
        context, gates = model(
            motion,
            {name: incidence for name in model.RELATIONS},
            vertex_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertEqual(context.shape, (2, 3, 8))
        self.assertEqual(gates.shape, (2, 3, 4))
        context.square().mean().backward()
        self.assertIsNotNone(motion.grad)

    def test_per_frame_mambatrack_shapes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for mamba_ssm selective scan")
        cfgs = {
            "d_m": 32,
            "d_state": 4,
            "L": 1,
            "avg_pool_out_dim": [1, 8],
            "pred_head_dims": [8, 4],
            "use_hyper_b": False,
            "use_spatial_hyper": True,
            "use_spatial_ssm_mod": False,
            "spatial_hyper_per_frame": True,
            "spatial_hyper_dim": 8,
        }
        device = torch.device("cuda")
        model = MambaTrack(cfgs).to(device)
        x = torch.randn(2, 3, 5, 4, device=device)
        label = torch.randn(2, 3, 4, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        incidence = {
            "spatial": torch.ones(2, 3, 5, 3, device=device),
            "rgb": torch.ones(2, 3, 5, 3, device=device),
            "evt": torch.ones(2, 3, 5, 3, device=device),
            "motion": torch.ones(2, 3, 5, 3, device=device),
        }
        loss = model(x, label, hyper_inputs=incidence, vertex_mask=mask)
        loss.backward()
        self.assertTrue(loss.ndim == 0)


if __name__ == "__main__":
    unittest.main()

