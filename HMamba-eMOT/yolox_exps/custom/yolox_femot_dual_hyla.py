# encoding: utf-8

import torch.nn as nn

from yolox_femot_dual import Exp as DualModalExp


class Exp(DualModalExp):
  """FEMOT dual-modal YOLOX with bidirectional HyLA fusion only."""

  def __init__(self):
    super(Exp, self).__init__()
    self.exp_name = "yolox_femot_dual_hyla"
    self.fusion_type = "hyla"
    self.fusion_scales = ("dark3", "dark4", "dark5")
    self.hyla_dims = (64, 96, 128)
    self.hyla_heads = (4, 4, 4)
    self.hyla_windows = (7, 7, 0)
    self.hyla_norm = "rms_head"
    self.fusion_dropout = 0.0
    self.mfg_reduction = 16

  def get_model(self):
    from yolox.models import DualModalYOLOPAFPN, YOLOX, YOLOXHead

    def init_yolo(module):
      for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
          layer.eps = 1e-3
          layer.momentum = 0.03

    if getattr(self, "model", None) is None:
      in_channels = [256, 512, 1024]
      backbone = DualModalYOLOPAFPN(
        self.depth,
        self.width,
        in_channels=in_channels,
        fusion_type=self.fusion_type,
        fusion_scales=self.fusion_scales,
        hyla_dims=self.hyla_dims,
        hyla_heads=self.hyla_heads,
        hyla_windows=self.hyla_windows,
        hyla_norm=self.hyla_norm,
        fusion_dropout=self.fusion_dropout,
        mfg_reduction=self.mfg_reduction,
      )
      head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
      self.model = YOLOX(backbone, head)

    self.model.apply(init_yolo)
    self.model.head.initialize_biases(1e-2)
    return self.model
