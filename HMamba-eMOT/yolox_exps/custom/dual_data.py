# encoding: utf-8
"""Dual-modal (RGB + event) transforms and mosaic for YOLOX."""

import random

import cv2
import numpy as np

from yolox.data.data_augment import (
    _distort,
    _mirror,
    box_candidates,
    preproc,
    random_perspective,
)
from yolox.data.datasets.datasets_wrapper import Dataset
from yolox.data.datasets.mosaicdetection import get_mosaic_coordinate
from yolox.utils import adjust_box_anns, xyxy2cxcywh


def to_dual_stack(rgb, evt):
    return np.concatenate([rgb, evt], axis=2)


def split_dual_stack(img6):
    return img6[:, :, :3], img6[:, :, 3:6]


def dual_preproc(rgb, evt, input_size, rgb_means, rgb_std, evt_means=None, evt_std=None):
    evt_means = rgb_means if evt_means is None else evt_means
    evt_std = rgb_std if evt_std is None else evt_std
    rgb_out, r = preproc(rgb, input_size, rgb_means, rgb_std)
    evt_out, _ = preproc(evt, input_size, evt_means, evt_std)
    return np.concatenate([rgb_out, evt_out], axis=0), r


class DualTrainTransform:
    def __init__(self, rgb_means=None, std=None, max_labels=100):
        self.rgb_means = rgb_means
        self.std = std
        self.max_labels = max_labels

    def __call__(self, rgb, evt, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, _ = dual_preproc(rgb, evt, input_dim, self.rgb_means, self.std)
            return image, targets

        rgb_o, evt_o = rgb.copy(), evt.copy()
        targets_o = targets.copy()
        boxes_o = xyxy2cxcywh(targets_o[:, :4])
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]

        rgb_t = _distort(rgb.copy())
        evt_t = evt.copy()
        rgb_t, boxes = _mirror(rgb_t, boxes)
        evt_t, _ = _mirror(evt_t, boxes)

        image_t, r_ = dual_preproc(rgb_t, evt_t, input_dim, self.rgb_means, self.std)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = dual_preproc(rgb_o, evt_o, input_dim, self.rgb_means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)
        targets_t = np.hstack((labels_t, boxes_t, ids_t))
        padded_labels = np.zeros((self.max_labels, 6), dtype=np.float32)
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        return np.ascontiguousarray(image_t, dtype=np.float32), np.ascontiguousarray(
            padded_labels, dtype=np.float32
        )


class DualValTransform:
    def __init__(self, rgb_means=None, std=None):
        self.rgb_means = rgb_means
        self.std = std

    def __call__(self, rgb, evt, res, input_size):
        img, _ = dual_preproc(rgb, evt, input_size, self.rgb_means, self.std)
        return img, np.zeros((1, 5))


class DualMosaicDetection(Dataset):
    """Mosaic + mixup for RGB/event pairs (6-channel stack internally)."""

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=10.0,
        translate=0.1,
        scale=(0.5, 1.5),
        mscale=(0.5, 1.5),
        shear=2.0,
        perspective=0.0,
        enable_mixup=True,
    ):
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                rgb, evt, _labels, _, _ = self._dataset.pull_item(index)
                img = to_dual_stack(rgb, evt)
                h0, w0 = img.shape[:2]
                scale = min(1.0 * input_h / h0, 1.0 * input_w / w0)
                img = cv2.resize(
                    img,
                    (int(w0 * scale), int(h0 * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
                h, w, c = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )

            if self.enable_mixup and len(mosaic_labels) != 0:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)

            rgb, evt = split_dual_stack(mosaic_img)
            mix_img, padded_labels = self.preproc(rgb, evt, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[2], mix_img.shape[1])
            return mix_img, padded_labels, img_info, np.array([idx])

        self._dataset._input_dim = self.input_dim
        rgb, evt, label, img_info, id_ = self._dataset.pull_item(idx)
        mix_img, label = self.preproc(rgb, evt, label, self.input_dim)
        return mix_img, label, img_info, id_

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        rgb, evt, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        img = to_dual_stack(rgb, evt)

        cp_img = np.ones((input_dim[0], input_dim[1], 6)) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 6)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if flip:
            cp_bboxes_origin_np[:, 0::2] = origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            id_labels = cp_labels[keep_list, 5:6].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels, id_labels))
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = 0.5 * origin_img.astype(np.float32) + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img, origin_labels
