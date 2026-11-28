# encoding: utf-8
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.dataset_paths import (
  FEMOT_ROOT,
  FEMOT_TEST_ANN,
  FEMOT_TRAIN_ANN,
  FEMOT_VAL_ANN,
)


class Exp(MyExp):
  """FEMOT dual-modal YOLOX: RGB (_aps) + event (_dvs), sum-fusion at dark3/4/5."""

  def __init__(self):
    super(Exp, self).__init__()
    self.num_classes = 2
    self.depth = 1.33
    self.width = 1.25
    self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    self.data_dir = FEMOT_ROOT
    self.train_ann = FEMOT_TRAIN_ANN
    self.val_ann = FEMOT_VAL_ANN
    self.input_size = (416, 416)
    self.test_size = (416, 416)
    self.random_size = (10, 20)
    self.max_epoch = 100
    self.print_interval = 100
    self.eval_interval = 5
    self.test_conf = 0.01
    self.nmsthre = 0.5
    self.no_aug_epochs = 20
    self.basic_lr_per_img = 0.001 / 128.0
    self.warmup_epochs = 3
    self.weight_decay = 1e-4
    self.ema = True
    self.data_num_workers = 8
    self.input_channels = 6

  def get_model(self):
    from yolox.models import DualModalYOLOPAFPN, YOLOX, YOLOXHead

    def init_yolo(M):
      for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eps = 1e-3
          m.momentum = 0.03

    if getattr(self, "model", None) is None:
      in_channels = [256, 512, 1024]
      backbone = DualModalYOLOPAFPN(
        self.depth, self.width, in_channels=in_channels
      )
      head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
      self.model = YOLOX(backbone, head)

    self.model.apply(init_yolo)
    self.model.head.initialize_biases(1e-2)
    return self.model

  def get_data_loader(self, batch_size, is_distributed, no_aug=False):
    from yolox.data import (
      DataLoader,
      InfiniteSampler,
      YoloBatchSampler,
    )
    from dual_data import DualMosaicDetection, DualTrainTransform
    from mot_dataset import FEMOTDualDataset

    dataset = FEMOTDualDataset(
      data_dir=self.data_dir,
      json_file=self.train_ann,
      name='train',
      img_size=self.input_size,
      preproc=DualTrainTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_labels=500,
      ),
    )

    dataset = DualMosaicDetection(
      dataset,
      mosaic=not no_aug,
      img_size=self.input_size,
      preproc=DualTrainTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_labels=1000,
      ),
      degrees=self.degrees,
      translate=self.translate,
      scale=self.scale,
      shear=self.shear,
      perspective=self.perspective,
      enable_mixup=self.enable_mixup,
    )
    self.dataset = dataset

    if is_distributed:
      batch_size = batch_size // dist.get_world_size()

    sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
    batch_sampler = YoloBatchSampler(
      sampler=sampler,
      batch_size=batch_size,
      drop_last=False,
      input_dimension=self.input_size,
      mosaic=not no_aug,
    )

    dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    return DataLoader(self.dataset, **dataloader_kwargs)

  def get_eval_loader(self, batch_size, is_distributed, testdev=False):
    from dual_data import DualValTransform
    from mot_dataset import FEMOTDualDataset

    split_name = "test" if testdev else "val"
    json_file = FEMOT_TEST_ANN if testdev else self.val_ann
    valdataset = FEMOTDualDataset(
      data_dir=self.data_dir,
      json_file=json_file,
      img_size=self.test_size,
      name=split_name,
      preproc=DualValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
      ),
    )

    if is_distributed:
      batch_size = batch_size // dist.get_world_size()
      sampler = torch.utils.data.distributed.DistributedSampler(
        valdataset, shuffle=False
      )
    else:
      sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
      "num_workers": self.data_num_workers,
      "pin_memory": True,
      "sampler": sampler,
      "batch_size": batch_size,
    }
    return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

  def get_evaluator(self, batch_size, is_distributed, testdev=False):
    from yolox.evaluators import COCOEvaluator

    val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
    return COCOEvaluator(
      dataloader=val_loader,
      img_size=self.test_size,
      confthre=self.test_conf,
      nmsthre=self.nmsthre,
      num_classes=self.num_classes,
      testdev=testdev,
    )
