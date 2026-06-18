# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset


class SeqDataset(Dataset): # TODO: test dataloader modify
    def __init__(self, seq_dir: str):
        # a hack implementation for BDD100K and others:
        if "BDD100K" in seq_dir:
            image_paths = sorted(os.listdir(os.path.join(seq_dir)))
            image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        elif "FEMOT" in seq_dir:  
            seq_name = os.path.basename(os.path.normpath(seq_dir))
            image_paths = sorted(os.listdir(os.path.join(seq_dir, f"{seq_name}_aps")))
            image_paths = [os.path.join(seq_dir,f"{seq_name}_aps", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            event_paths = sorted(os.listdir(os.path.join(seq_dir, f"{seq_name}_dvs")))
            event_paths = [os.path.join(seq_dir, f"{seq_name}_dvs", _) for _ in event_paths if ("jpg" in _) or ("png" in _)]
        elif "DSEC" in seq_dir:
            seq_name = os.path.basename(os.path.normpath(seq_dir))
            imgages_subDir = "images/left/distorted"
            events_subDir = "events/left_dvs"
            image_paths = sorted(os.listdir(os.path.join(seq_dir, imgages_subDir)))
            image_paths = [os.path.join(seq_dir,imgages_subDir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            event_paths = sorted(os.listdir(os.path.join(seq_dir, events_subDir)))
            event_paths = [os.path.join(seq_dir, events_subDir, _) for _ in event_paths if ("jpg" in _) or ("png" in _)]
        elif "VTMOT" in seq_dir:  
            seq_name = os.path.basename(os.path.normpath(seq_dir))
            image_paths = sorted(os.listdir(os.path.join(seq_dir, "visible")))
            image_paths = [os.path.join(seq_dir,"visible", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            event_paths = sorted(os.listdir(os.path.join(seq_dir, "infrared")))
            event_paths = [os.path.join(seq_dir, "infrared", _) for _ in event_paths if ("jpg" in _) or ("png" in _)]
        else:
            image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
            image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.image_paths = image_paths
        self.event_paths = event_paths
        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        return

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image):
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        info = self.image_paths[item]
        event_image = self.load(self.event_paths[item])
        event_info = self.event_paths[item]
        return self.process_image(image=image), info, self.process_image(image=event_image), event_info

    def __len__(self):
        return len(self.image_paths)
