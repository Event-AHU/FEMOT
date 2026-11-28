#!/usr/bin/env python3
"""Turn DSEC tracks.npy into the FEMOT layout used by YOLOX and MambaTrack.

Each sequence keeps every RGB and event frame. tracks.npy rows are matched to
images/timestamps.txt, then written as 1-based MOT frames. Boxes are clipped
to the image. Empty frames stay in the image list and have no gt rows.
"""

import json
import os
from collections import defaultdict

import cv2
import numpy as np

SRC_ROOT = "/rydata/dataset/DSEC_distorted"
OUT_ROOT = "/wangx_new/lyt/dsec_mot"
TRAJ_PATH = "/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/traj_anno_data/dsec.json"
CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
)
# Held out of detector training and of the motion-regression trajectories.
VAL_SEQS = (
    "interlaken_00_g",
    "zurich_city_01_e",
    "zurich_city_04_c",
    "zurich_city_05_b",
    "zurich_city_08_a",
    "zurich_city_09_d",
    "zurich_city_11_a",
)
FILTER_LEN = 25


def list_sequences(split):
    root = os.path.join(SRC_ROOT, split)
    seqs = []
    for name in sorted(os.listdir(root)):
        base = os.path.join(root, name)
        tracks = os.path.join(base, "object_detections", "left", "tracks.npy")
        if os.path.isdir(base) and os.path.isfile(tracks):
            seqs.append(name)
    return seqs


def split_of(official_split, seq):
    if official_split == "test":
        return "test"
    if seq in VAL_SEQS:
        return "val"
    return "train"


def link_frames(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(src_dir) if name.lower().endswith(".png")
    )
    stems = [int(os.path.splitext(name)[0]) for name in names]
    if stems != list(range(len(stems))):
        raise RuntimeError(f"frame stems are not 0..N-1 in {src_dir}")
    for index, name in enumerate(names):
        dst = os.path.join(dst_dir, f"{index + 1:05d}.jpg")
        src = os.path.abspath(os.path.join(src_dir, name))
        if os.path.islink(dst) or os.path.exists(dst):
            current = os.path.realpath(dst)
            if current != src:
                raise RuntimeError(f"existing link {dst} -> {current}, expected {src}")
            continue
        os.symlink(src, dst)
    return len(names)


def clip_box(x, y, w, h, width, height):
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2 - x1, y2 - y1


def convert_sequence(official_split, seq, coco):
    src = os.path.join(SRC_ROOT, official_split, seq)
    out_split = split_of(official_split, seq)
    dst = os.path.join(OUT_ROOT, out_split, seq)
    n_rgb = link_frames(
        os.path.join(src, "images", "left", "distorted"),
        os.path.join(dst, f"{seq}_aps"),
    )
    n_evt = link_frames(
        os.path.join(src, "events", "left_dvs"),
        os.path.join(dst, f"{seq}_dvs"),
    )
    if n_rgb != n_evt:
        raise RuntimeError(f"{seq}: rgb frames {n_rgb} != event frames {n_evt}")

    timestamps = np.loadtxt(
        os.path.join(src, "images", "timestamps.txt"), dtype=np.int64
    )
    if timestamps.ndim == 0:
        timestamps = timestamps.reshape(1)
    if len(timestamps) != n_rgb:
        raise RuntimeError(f"{seq}: {len(timestamps)} timestamps != {n_rgb} frames")

    sample = cv2.imread(os.path.join(dst, f"{seq}_aps", "00001.jpg"))
    if sample is None:
        raise RuntimeError(f"{seq}: failed to read linked frame")
    height, width = sample.shape[:2]
    evt_sample = cv2.imread(os.path.join(dst, f"{seq}_dvs", "00001.jpg"))
    if evt_sample is None or evt_sample.shape[:2] != (height, width):
        raise RuntimeError(f"{seq}: event frame does not match rgb")

    time_to_frame = {int(stamp): index + 1 for index, stamp in enumerate(timestamps)}
    tracks = np.load(os.path.join(src, "object_detections", "left", "tracks.npy"))
    chosen = {}
    missed_time = 0
    dropped_box = 0
    replaced = 0
    for row in tracks:
        frame = time_to_frame.get(int(row["t"]))
        if frame is None:
            missed_time += 1
            continue
        box = clip_box(row["x"], row["y"], row["w"], row["h"], width, height)
        if box is None:
            dropped_box += 1
            continue
        class_id = int(row["class_id"])
        if class_id < 0 or class_id >= len(CLASSES):
            raise RuntimeError(f"{seq}: class_id {class_id} is outside 0..6")
        key = (frame, int(row["track_id"]))
        record = (float(row["class_confidence"]), box, class_id)
        if key in chosen:
            replaced += 1
            if record[0] <= chosen[key][0]:
                continue
        chosen[key] = record

    os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
    gt_path = os.path.join(dst, "gt", "gt.txt")
    rows = []
    for (frame, track_id), (_, box, class_id) in sorted(chosen.items()):
        x, y, w, h = box
        category_id = class_id + 1
        rows.append(
            f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{category_id},1"
        )
    with open(gt_path, "w") as handle:
        if rows:
            handle.write("\n".join(rows))
            handle.write("\n")

    with open(os.path.join(dst, "seqinfo.ini"), "w") as handle:
        handle.write(
            "\n".join(
                [
                    "[Sequence]",
                    f"name={seq}",
                    f"imDir_aps={seq}_aps",
                    f"imDir_dvs={seq}_dvs",
                    f"seqLength={n_rgb}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    "imExt=.jpg",
                    "",
                ]
            )
        )

    video_id = len(coco["videos"]) + 1
    coco["videos"].append({"id": video_id, "file_name": seq})
    image_base = len(coco["images"])
    for frame in range(1, n_rgb + 1):
        coco["images"].append(
            {
                "file_name": f"{seq}/{seq}_aps/{frame:05d}.jpg",
                "id": image_base + frame,
                "frame_id": frame,
                "prev_image_id": image_base + frame - 1 if frame > 1 else -1,
                "next_image_id": image_base + frame + 1 if frame < n_rgb else -1,
                "video_id": video_id,
                "height": height,
                "width": width,
            }
        )
    for (frame, _track_id), (_, box, class_id) in sorted(chosen.items()):
        x, y, w, h = box
        coco["annotations"].append(
            {
                "id": len(coco["annotations"]) + 1,
                "category_id": class_id + 1,
                "image_id": image_base + frame,
                "track_id": int(_track_id),
                "bbox": [x, y, w, h],
                "conf": 1.0,
                "iscrowd": 0,
                "area": float(w * h),
            }
        )

    print(
        f"{out_split}/{seq}: frames={n_rgb} boxes={len(rows)} "
        f"dropped={dropped_box} duplicate_rows={replaced} unmatched_t={missed_time}",
        flush=True,
    )
    return {
        "split": out_split,
        "seq": seq,
        "frames": n_rgb,
        "boxes": len(rows),
        "width": width,
        "height": height,
        "gt": gt_path,
    }


def empty_coco():
    return {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [
            {"id": index + 1, "name": name} for index, name in enumerate(CLASSES)
        ],
    }


def write_traj(infos):
    annotation = {}
    new_id = 0
    kept = 0
    dropped = 0
    for info in infos:
        if info["split"] != "train":
            continue
        if os.path.getsize(info["gt"]) == 0:
            continue
        gt = np.loadtxt(info["gt"], delimiter=",", dtype=np.float64)
        if gt.ndim == 1:
            gt = gt[None, :]
        grouped = defaultdict(list)
        for row in gt:
            if int(row[6]) != 1:
                continue
            grouped[int(row[1])].append(row)
        for track_id in sorted(grouped):
            rows = sorted(grouped[track_id], key=lambda item: int(item[0]))
            if len(rows) < FILTER_LEN:
                dropped += 1
                continue
            bboxes = []
            for row in rows:
                x, y, w, h = row[2:6]
                bboxes.append(
                    [
                        round((x + 0.5 * w) / info["width"], 6),
                        round((y + 0.5 * h) / info["height"], 6),
                        round(w / info["width"], 6),
                        round(h / info["height"], 6),
                    ]
                )
            annotation[str(new_id)] = {
                "image_h": info["height"],
                "image_w": info["width"],
                "traj_len": len(bboxes),
                "bboxes": bboxes,
            }
            new_id += 1
            kept += 1
    annotation["total_objs"] = kept
    annotation["obj_id_start"] = 0
    os.makedirs(os.path.dirname(TRAJ_PATH), exist_ok=True)
    with open(TRAJ_PATH, "w") as handle:
        json.dump({"dsec": annotation}, handle)
    print(f"traj tracks kept={kept} shorter_than_{FILTER_LEN}={dropped} -> {TRAJ_PATH}")


def write_seqmap(split, seqs):
    folder = os.path.join(OUT_ROOT, split, "seqmaps")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"DSEC-{split}.txt")
    with open(path, "w") as handle:
        handle.write("name\n")
        for seq in seqs:
            handle.write(seq + "\n")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    infos = []
    coco = {split: empty_coco() for split in ("train", "val", "test")}
    for official in ("train", "test"):
        for seq in list_sequences(official):
            out_split = split_of(official, seq)
            infos.append(convert_sequence(official, seq, coco[out_split]))

    ann_dir = os.path.join(OUT_ROOT, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for split, payload in coco.items():
        path = os.path.join(ann_dir, f"{split}_rgb.json")
        with open(path, "w") as handle:
            json.dump(payload, handle)
        print(
            f"{split} coco: images={len(payload['images'])} "
            f"boxes={len(payload['annotations'])} -> {path}"
        )

    by_split = {split: [] for split in ("train", "val", "test")}
    for info in infos:
        by_split[info["split"]].append(info["seq"])
    for split, seqs in by_split.items():
        write_seqmap(split, seqs)
    manifest = {
        "src": SRC_ROOT,
        "out": OUT_ROOT,
        "val_seqs": list(VAL_SEQS),
        "classes": list(CLASSES),
        "splits": by_split,
        "frames": {
            split: sum(info["frames"] for info in infos if info["split"] == split)
            for split in by_split
        },
        "boxes": {
            split: sum(info["boxes"] for info in infos if info["split"] == split)
            for split in by_split
        },
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    write_traj(infos)
    print("prepared", OUT_ROOT)


if __name__ == "__main__":
    main()
