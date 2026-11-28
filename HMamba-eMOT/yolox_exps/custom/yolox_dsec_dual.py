# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox_femot_dual import Exp as FemotDualExp


DSEC_ROOT = "/wangx_new/lyt/dsec_mot"


class Exp(FemotDualExp):
    """DSEC dual-modal YOLOX. Sum fusion, seven classes, COCO-pretrained backbone."""

    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 7
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = DSEC_ROOT
        self.train_ann = "train_rgb.json"
        self.val_ann = "val_rgb.json"
        self.test_ann = "test_rgb.json"
        self.max_epoch = 30
        self.no_aug_epochs = 9
        self.eval_interval = 5
        self.data_num_workers = 4
        self.output_dir = "/wangx_new/lyt/dsec_yolox_outputs"

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from dual_data import DualValTransform
        from mot_dataset import FEMOTDualDataset

        split_name = "test" if testdev else "val"
        json_file = self.test_ann if testdev else self.val_ann
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
