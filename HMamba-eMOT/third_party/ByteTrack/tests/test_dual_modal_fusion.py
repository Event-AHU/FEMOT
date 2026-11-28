import unittest

import torch

from yolox.models import (
    DualModalYOLOPAFPN,
    HyLACrossAttention2D,
    HyLAMFGFusionBlock,
    MFGFusionBlock,
)


class ModalFusionTest(unittest.TestCase):
    def test_hyla_handles_padding_and_backward(self):
        module = HyLACrossAttention2D(
            channels=16,
            attention_dim=16,
            num_heads=4,
            window_size=4,
            query_channels=32,
        )
        rgb = torch.randn(2, 16, 7, 9, requires_grad=True)
        evt = torch.randn(2, 16, 7, 9, requires_grad=True)
        joint_query = torch.cat((rgb, evt), dim=1)
        output = module(joint_query, evt)
        self.assertEqual(output.shape, rgb.shape)
        output.mean().backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(evt.grad)

    def test_mfg_is_batch_safe_and_sum_initialized(self):
        module = MFGFusionBlock(channels=16)
        rgb = torch.randn(2, 16, 8, 8, requires_grad=True)
        evt = torch.randn(2, 16, 8, 8, requires_grad=True)
        output = module(rgb, evt)
        self.assertEqual(output.shape, rgb.shape)
        self.assertTrue(torch.allclose(output, rgb + evt))
        output.square().mean().backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(evt.grad)

    def test_serial_hyla_mfg_is_sum_initialized(self):
        module = HyLAMFGFusionBlock(
            channels=16,
            attention_dim=16,
            num_heads=4,
            window_size=4,
        )
        rgb = torch.randn(2, 16, 8, 8)
        evt = torch.randn(2, 16, 8, 8)
        output = module(rgb, evt)
        self.assertEqual(output.shape, rgb.shape)
        self.assertTrue(torch.allclose(output, rgb + evt))

    def test_dual_pafpn_keeps_probe_and_export_interfaces(self):
        model = DualModalYOLOPAFPN(
            depth=0.33,
            width=0.25,
            fusion_type="hyla_mfg",
            hyla_dims=(16, 16, 16),
            hyla_heads=(4, 4, 4),
            hyla_windows=(4, 4, 0),
        ).eval()
        model.capture_modal_features = True
        dual_input = torch.randn(2, 6, 64, 64)
        with torch.no_grad():
            outputs = model(dual_input)
        self.assertEqual([x.shape[1] for x in outputs], [64, 128, 256])
        self.assertEqual([x.shape[-1] for x in outputs], [8, 4, 2])
        self.assertIsNotNone(model.last_modal_features)

        with torch.no_grad():
            modal_outputs = model.forward_modal_features(dual_input)
            probe_outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(len(modal_outputs), 2)
        self.assertEqual([x.shape[1] for x in probe_outputs], [64, 128, 256])

    def test_early_hyla_exchanges_after_dark2_and_uses_sum_outputs(self):
        model = DualModalYOLOPAFPN(
            depth=0.33,
            width=0.25,
            fusion_type="sum",
            early_hyla=True,
            early_hyla_dim=16,
            early_hyla_heads=4,
            early_hyla_window=4,
        ).eval()
        self.assertIsNotNone(model.early_hyla)
        self.assertEqual(len(model.fusion_blocks), 0)

        dual_input = torch.randn(2, 6, 64, 64)
        with torch.no_grad():
            modal_outputs = model.forward_modal_features(dual_input)
            outputs = model(dual_input)
        self.assertEqual(
            [tuple(x.shape) for x in modal_outputs[0].values()],
            [(2, 64, 8, 8), (2, 128, 4, 4), (2, 256, 2, 2)],
        )
        self.assertEqual([x.shape[1] for x in outputs], [64, 128, 256])


if __name__ == "__main__":
    unittest.main()
