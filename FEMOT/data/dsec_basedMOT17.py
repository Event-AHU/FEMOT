# @Author       : tomx
# @Date         : 2025/2/11
import math
import os
import torch
import random

from collections import defaultdict
from random import randint
from PIL import Image

import data.transforms as T
from .mot import MOTDataset
import numpy as np



class DSEC_distorted(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        super(DSEC_distorted, self).__init__(config=config, split=split, transform=transform)

        self.config = config
        self.transform = transform
        self.use_motsynth = config["USE_MOTSYNTH"]
        self.use_crowdhuman = config["USE_CROWDHUMAN"]
        self.motsynth_rate = config["MOTSYNTH_RATE"]
        if self.use_motsynth:
            multi_random_state = random.getstate()
            random.seed(config["SEED"])
            self.unified_random_state = random.getstate()
            random.setstate(multi_random_state)
        else:
            self.unified_random_state = None

        assert split == "train", f"Split {split} is NOT supported."
        self.mot17_seqs_dir = os.path.join(config["DATA_ROOT"], config["DATASET"], split)
        self.mot17_gts_dir = os.path.join(config["DATA_ROOT"], config["DATASET"], split)


        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.sample_mot17_join: int = config["SAMPLE_MOT17_JOIN"]
        self.sample_stage = None
        self.sample_begin_frame_paths = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.mot17_gts = defaultdict(lambda: defaultdict(list))
        self.crowdhuman_gts = defaultdict(list)
        self.motsynth_gts = defaultdict(lambda: defaultdict(list))
        
        self.mot17_seq_names = [seq for seq in os.listdir(self.mot17_seqs_dir)]

        # CLASS_MAPPING = {0: 0, 2: 1}
        
        for vid in self.mot17_seq_names:
            mot17_gts_dir = os.path.join(self.mot17_gts_dir, vid, "object_detections", "left")
            mot17_gt_paths = [os.path.join(mot17_gts_dir, filename) for filename in os.listdir(mot17_gts_dir)]
            count = -1
            for mot17_gt_path in mot17_gt_paths:
                preKey = -1
                for item in np.load(mot17_gt_path):
                    # fid, i, x, y, w, h, score, c, _ = line.strip("\n").split(",")
                    # x, y, w, h, score = map(float, (item[1], item[2], item[3], item[4], item[-2]))
                    # fid, i, c = map(int, (item[0], item[-1], item[-3]))
                    curKey = item[0]
                    if curKey != preKey:
                        count += 1
                        preKey = curKey
                    self.mot17_gts[vid][count].append([item[-1], item[1], item[2], item[3], item[4], item[-3]])
                # self.mot17_gts[vid][key] 转化为从0开始的idx


        self.set_epoch(epoch=0)     # init datasets

        return

    def __len__(self):
        assert self.sample_begin_frame_paths is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frame_paths)

    def __getitem__(self, item):
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        begin_frame_path = self.sample_begin_frame_paths[item]

        frame_paths = self.sample_frame_paths(begin_frame_path=begin_frame_path)
        imgs, infos = self.get_multi_frames(frame_paths=frame_paths)
        
        imgs, infos = self.transform["DSEC_distorted"](imgs, infos)

        random.seed(seed)
        np.random.seed(seed)
        begin_event_frame_path = self.sample_begin_event_frame_paths[item]
        event_frame_paths = self.sample_event_frame_paths(begin_frame_path=begin_event_frame_path)
        event_imgs, event_infos = self.get_multi_frames(frame_paths=event_frame_paths)
        event_imgs, event_infos = self.transform["DSEC_distorted"](event_imgs, event_infos)

        # if infos[0]["dataset"] == "FEMOT":
        #     imgs, infos = self.transform["FEMOT"](imgs, infos)
        # else:
        #     imgs, infos = self.transform["CrowdHuman"](imgs, infos)

        return {
            "imgs": imgs,
            "event_imgs": event_imgs,
            "infos": infos
        }

    def set_epoch(self, epoch: int):
        # Copy from dancetrack.py
        self.sample_begin_frame_paths = list()
        self.sample_begin_event_frame_paths = list()

        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        # End of Copy
        if epoch >= self.sample_mot17_join:
            for vid in self.mot17_gts.keys():
                t_min = min(self.mot17_gts[vid].keys())
                t_max = max(self.mot17_gts[vid].keys())
                self.sample_vid_tmax[vid] = t_max
                for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                    self.sample_begin_frame_paths.append(
                        # os.path.join(self.mot17_seqs_dir, vid, "img1", str(t).zfill(6) + ".png")
                        os.path.join(self.mot17_seqs_dir, vid, "images", "left", "distorted", str(t).zfill(6) + ".png")
                    )
                    self.sample_begin_event_frame_paths.append(
                        os.path.join(self.mot17_seqs_dir, vid, "events", "left_dvs", str(t).zfill(6) + ".png")
                    )

        return

    def sample_frame_paths(self, begin_frame_path: str) -> list[str]:
        if "CrowdHuman" in begin_frame_path:
            return [begin_frame_path] * self.sample_length
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample Length is less than 2."
            vid = begin_frame_path.split("/")[-5]  # 这个数据集的images和events不对称，所以要额外处理

            begin_t = int(begin_frame_path.split("/")[-1].split(".")[0])
            remain_frames = self.sample_vid_tmax[vid] - begin_t
            max_interval = math.floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idx = [begin_t + interval * i for i in range(self.sample_length)]
            frame_paths = [os.path.join(self.mot17_seqs_dir, vid, "images", "left", "distorted", str(t).zfill(6) + ".png") for t in frame_idx]

            return frame_paths
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")
        
    def sample_event_frame_paths(self, begin_frame_path: str) -> list[str]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample Length is less than 2."
            vid = begin_frame_path.split("/")[-4]  # 这个数据集的images和events不对称，所以要额外处理
            begin_t = int(begin_frame_path.split("/")[-1].split(".")[0])
            remain_frames = self.sample_vid_tmax[vid] - begin_t
            max_interval = math.floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idx = [begin_t + interval * i for i in range(self.sample_length)]
            # if "MOTSynth" in begin_frame_path:
            #     frame_paths = [os.path.join(self.motsynth_seqs_dir, vid, "rgb", str(t).zfill(4) + ".png") for t in frame_idx]
            # else:
            # frame_paths = [os.path.join(self.mot17_seqs_dir, vid, "events", "left_dvs", str(t).zfill(6) + ".png") for t in frame_idx]
            frame_paths = [os.path.join(self.mot17_seqs_dir, vid, "events/left_dvs", str(t).zfill(6) + ".png") for t in frame_idx]

            return frame_paths
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")


    def get_single_frame(self, frame_path: str):
        frame_idx = int(frame_path.split("/")[-1].split(".")[0])
        if "images" in frame_path:
            vid = frame_path.split("/")[-5]
        elif "events" in frame_path:
            vid = frame_path.split("/")[-4]
        else:
            raise NotImplementedError
        gt = self.mot17_gts[vid][frame_idx]

        img = Image.open(frame_path)

        crowdhuman_ids_offset = 100000

        info = {}
        
        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["dataset"] = "MOT17" if ("MOT17" in frame_path or "MOTSynth" in frame_path) else "CrowdHuman"

        for i, x, y, w, h, c in gt:
            info["boxes"].append(list(map(float, (x, y, w, h))))
            info["areas"].append(w * h)
            info["ids"].append(i if "MOT17" in frame_path else i + crowdhuman_ids_offset)
            info["labels"].append(c)
        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)
        # xywh to xyxy
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0, ), dtype=torch.long)
            info["labels"] = torch.zeros((0, ), dtype=torch.long)

        return img, info

    def get_multi_frames(self, frame_paths: list[str]):
        return zip(*[self.get_single_frame(frame_path=path) for path in frame_paths])


def transforms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
    # 更合理的尺度范围，接近原始分辨率的1-2倍 TODO: myModification
    # scales = [256, 288, 320, 352, 384, 416, 448, 480, 512]

    return {
       "DSEC_distorted": T.MultiCompose([  # 同MOT17
            T.MultiRandomHorizontalFlip(),
            T.MultiRandomSelect(
                T.MultiRandomResize(scales, max_size=1536),
                # T.MultiRandomResize(scales, max_size=720),
                T.MultiCompose([
                    # 更合理的裁剪尺寸
                    # T.MultiRandomResize([240, 320, 400]),
                    # T.MultiRandomCrop(min_size=224, max_size=400, overflow_bbox=overflow_bbox),
                    T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                    T.MultiRandomCrop(
                        min_size=384 if coco_size else 800,
                        max_size=600 if coco_size else 1200,
                        overflow_bbox=overflow_bbox),
                    T.MultiRandomResize(scales, max_size=1536)
                    # T.MultiRandomResize(scales, max_size=720)
                ])
            ),
            T.MultiHSV(),
            T.MultiCompose([
                T.MultiToTensor(),
                T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from COCO/MOTR
            ]),
            T.MultiReverseClip(reverse=reverse_clip)
        ]),
}


def build(config: dict, split: str):
    if split == "train":
        return DSEC_distorted(
            config=config,
            split=split,
            transform=transforms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    else:
        raise NotImplementedError(f"MOT Dataset 'build' function do not support split {split}.")
