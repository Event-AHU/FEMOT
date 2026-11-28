#!/usr/bin/env python3
"""Train and evaluate the DSEC baseline and the 44.713 structure.

Detectors start from the COCO YOLOX-X backbone. The baseline tracker is the
original MambaTrack recipe. The 44.713 run then adapts that DSEC tracker for
three epochs on HyLA-enhanced features, with event and motion edges removed
and alpha_B frozen. Inference thresholds stay at the FEMOT settings.
"""

import json
import os
import shutil
import subprocess
import time

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/home/lyt/miniconda3/envs/mamba_trackers/bin/python"
DATA = "/wangx_new/lyt/dsec_mot"
DET_ROOT = "/wangx_new/lyt/dsec_det_results"
FEAT_ROOT = "/wangx_new/lyt/dsec_modal_gt_hyla"
CKPT_ROOT = "/wangx_new/lyt/dsec_mamba"
YOLO_OUT = "/wangx_new/lyt/dsec_yolox_outputs"
INIT_CKPT = os.path.join(YOLO_OUT, "yolox_x_dual_init.pth.tar")
COCO_SRC = os.path.join(PROJECT, "weights/yolox_x.pth")
LOG_DIR = os.path.join(PROJECT, "logs/dsec")
LOCK = os.path.join(LOG_DIR, "trackeval.lock")
TRACK_ROOT = os.path.join(PROJECT, "ssm_tracker/track_results")
HYLA_EXP = os.path.join(
    PROJECT, "yolox_exps/custom/yolox_dsec_dual_hyla_shared_mlp.py"
)
SUM_EXP = os.path.join(PROJECT, "yolox_exps/custom/yolox_dsec_dual.py")
BASE_CFG = "cfgs/MambaTrack_dsec.yaml"
HYLA_CFG = "cfgs/dsec/dsec_hyla_drop_evt_motion_zero_b.yaml"
MIN_TRACK_MIB = 8192
MIN_DET_MIB = 45000
MIN_SUM_MIB = 62000
MIN_HYLA_MIB = 72000


def log(message):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "orchestrator.log"), "a") as handle:
        handle.write(line + "\n")


def memory_free():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {}
    for line in out.strip().splitlines():
        index, mib = [part.strip() for part in line.split(",")]
        free[int(index)] = int(mib)
    return free


def wait_gpu(min_free, avoid):
    while True:
        free = memory_free()
        slots = [
            index
            for index, mib in free.items()
            if mib >= min_free and index not in avoid
        ]
        if slots:
            return max(slots, key=lambda index: free[index])
        log(
            "waiting for GPU free>="
            + str(min_free)
            + " "
            + ", ".join(f"{index}={mib}" for index, mib in sorted(free.items()))
        )
        time.sleep(60)


def env_for(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TQDM_DISABLE"] = "1"
    env["TMPDIR"] = "/wangx_new/lyt/tmp"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = (
        f"{PROJECT}/third_party/ByteTrack:{PROJECT}/yolox_exps/custom:"
        + env.get("PYTHONPATH", "")
    )
    return env


def launch(name, cmd, gpu, cwd, log_name):
    path = os.path.join(LOG_DIR, log_name)
    handle = open(path, "a")
    handle.write(f"\n===== START {name} gpu={gpu} =====\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env_for(gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log(f"started {name} pid={proc.pid} gpu={gpu} log={path}")
    return proc, handle


def wait_proc(name, proc, handle):
    code = proc.wait()
    handle.close()
    log(f"finished {name} pid={proc.pid} code={code}")
    if code != 0:
        raise RuntimeError(f"{name} failed with code {code}")


def run(name, cmd, gpu, cwd, log_name):
    proc, handle = launch(name, cmd, gpu, cwd, log_name)
    wait_proc(name, proc, handle)


def prepare():
    manifest = os.path.join(DATA, "manifest.json")
    if os.path.isfile(manifest) and os.path.isfile(
        os.path.join(PROJECT, "ssm_tracker/traj_anno_data/dsec.json")
    ):
        log(f"data already prepared: {manifest}")
        return
    run(
        "prepare",
        [PYTHON, "-u", os.path.join(PROJECT, "tools/prepare_dsec_mot.py")],
        gpu=None if False else "cpu",
        cwd=PROJECT,
        log_name="prepare.log",
    )


def build_init():
    if os.path.isfile(INIT_CKPT):
        log(f"dual init exists: {INIT_CKPT}")
        return
    os.makedirs(YOLO_OUT, exist_ok=True)
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from tools.build_dual_yolox_init_ckpt import build_dual_init_ckpt; "
        "print(build_dual_init_ckpt(%r, %r))"
    ) % (PROJECT, COCO_SRC, INIT_CKPT)
    run(
        "dual-init",
        [PYTHON, "-c", code],
        gpu="cpu",
        cwd=PROJECT,
        log_name="dual_init.log",
    )


def detector_ckpt(exp_name):
    return os.path.join(YOLO_OUT, exp_name, "latest_ckpt.pth.tar")


def detector_epoch(exp_name):
    path = detector_ckpt(exp_name)
    if not os.path.isfile(path):
        return 0
    out = subprocess.check_output(
        [
            PYTHON,
            "-c",
            "import torch,sys; ck=torch.load(sys.argv[1], map_location='cpu'); "
            "print(int(ck.get('start_epoch', 0)))",
            path,
        ],
        text=True,
    )
    return int(out.strip())


def detector_cmd(exp_file, exp_name, batch, resume_path=None):
    cmd = [
        PYTHON,
        "-u",
        "tools/train.py",
        "-f",
        exp_file,
        "-b",
        str(batch),
        "-d",
        "1",
        "--fp16",
    ]
    if resume_path:
        cmd.extend(["--resume", "-c", resume_path])
    else:
        cmd.extend(["-c", INIT_CKPT])
    cmd.extend(
        [
            "--experiment-name",
            exp_name,
            "max_epoch",
            "30",
            "no_aug_epochs",
            "9",
        ]
    )
    return cmd


def detector_weights(exp_name):
    best = os.path.join(YOLO_OUT, exp_name, "best_ckpt.pth.tar")
    latest = detector_ckpt(exp_name)
    if os.path.isfile(best):
        return best
    if os.path.isfile(latest) and detector_epoch(exp_name) >= 30:
        return latest
    raise RuntimeError(f"no finished detector checkpoint for {exp_name}")


def tracker_cmd(exp_name, cfg, init_ckpt, epochs_note):
    cmd = [
        PYTHON,
        "-u",
        "train.py",
        "--exp_name",
        exp_name,
        "--config_file",
        cfg,
        "--save_path",
        CKPT_ROOT,
        "--device",
        "0",
    ]
    if init_ckpt:
        cmd.extend(["--init_ckpt", init_ckpt])
    return cmd


def det_cmd(exp_file, ckpt, split, save_dir, save_feats):
    cmd = [
        PYTHON,
        "-u",
        os.path.join(PROJECT, "tools/gen_det_results.py"),
        "--dataset_name",
        "dsec",
        "--split",
        split,
        "--data_root",
        DATA,
        "--exp_file",
        exp_file,
        "--model_path",
        ckpt,
        "--img_size",
        "416",
        "416",
        "--save_dir",
        save_dir,
        "--device",
        "0",
        "--dual_modal",
        "--generate_meta_data",
    ]
    if save_feats:
        cmd.append("--save_feats")
    return cmd


def copy_cfg(src_rel, name):
    src = os.path.join(PROJECT, "ssm_tracker", src_rel)
    dst_dir = os.path.join(PROJECT, "ssm_tracker/cfgs/dsec/eval")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name + ".yaml")
    shutil.copyfile(src, dst)
    return "cfgs/dsec/eval/" + name + ".yaml"


def pred_dir(det_dir, cfg_rel, split):
    dataset = os.path.basename(os.path.dirname(det_dir))
    cfg_name = os.path.splitext(os.path.basename(cfg_rel))[0]
    return os.path.join(TRACK_ROOT, dataset + "_" + cfg_name, split)


def parse_metrics(text):
    section = None
    found = {}
    for line in text.splitlines():
        if line.startswith("HOTA:"):
            section = "hota"
        elif line.startswith("CLEAR:"):
            section = "clear"
        elif line.startswith("Identity:"):
            section = "identity"
        elif line.startswith("COMBINED") and section and section not in found:
            found[section] = [float(part) for part in line.split()[1:]]
    hota, clear, identity = found["hota"], found["clear"], found["identity"]
    return {
        "HOTA": hota[0],
        "DetA": hota[1],
        "AssA": hota[2],
        "MOTA": clear[0],
        "MOTP": clear[1],
        "IDSW": int(clear[12]),
        "IDF1": identity[0],
    }


def track_and_eval(name, ckpt, src_cfg, det_dir, split, gpu):
    cfg = copy_cfg(src_cfg, name)
    out = pred_dir(det_dir, cfg, split)
    if os.path.isdir(out):
        shutil.rmtree(out)
    run(
        "track-" + name + "-" + split,
        [
            PYTHON,
            "-u",
            "track.py",
            "--det_path",
            det_dir,
            "--motion_model_path",
            ckpt,
            "--config_file",
            cfg,
            "--device",
            "0",
        ],
        gpu,
        os.path.join(PROJECT, "ssm_tracker"),
        name + "_" + split + ".log",
    )
    eval_log = os.path.join(LOG_DIR, name + "_" + split + "_eval.log")
    with open(eval_log, "w") as handle:
        proc = subprocess.run(
            [
                "flock",
                LOCK,
                PYTHON,
                "-u",
                os.path.join(PROJECT, "trackeval/run_dsec_eval.py"),
                "--split",
                split,
                "--pred_dir",
                out,
                "--tracker_name",
                name,
            ],
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = open(eval_log).read()
    if proc.returncode != 0:
        raise RuntimeError(f"eval {name} {split} failed\n" + text[-2000:])
    metrics = parse_metrics(text)
    log(f"metrics {name} {split} {metrics}")
    return metrics


def best_epoch(rows):
    return max(rows, key=lambda row: (row["HOTA"], row["epoch"]))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("/wangx_new/lyt/tmp", exist_ok=True)
    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(DET_ROOT, exist_ok=True)
    open(LOCK, "a").close()
    log("DSEC pipeline start")
    # prepare and init do not need a GPU; env_for still sets a device string.
    manifest = os.path.join(DATA, "manifest.json")
    if not (
        os.path.isfile(manifest)
        and os.path.isfile(os.path.join(PROJECT, "ssm_tracker/traj_anno_data/dsec.json"))
    ):
        subprocess.check_call(
            [PYTHON, "-u", os.path.join(PROJECT, "tools/prepare_dsec_mot.py")],
            cwd=PROJECT,
        )
        log("data prepared")
    else:
        log("data already prepared")
    if not os.path.isfile(INIT_CKPT):
        os.makedirs(YOLO_OUT, exist_ok=True)
        subprocess.check_call(
            [
                PYTHON,
                "-c",
                (
                    "import sys; sys.path.insert(0, %r); "
                    "from tools.build_dual_yolox_init_ckpt import build_dual_init_ckpt; "
                    "print(build_dual_init_ckpt(%r, %r))"
                )
                % (PROJECT, COCO_SRC, INIT_CKPT),
            ],
            cwd=PROJECT,
        )
        log(f"wrote {INIT_CKPT}")

    hyla_gpu = wait_gpu(MIN_DET_MIB, set())
    sum_gpu = wait_gpu(MIN_DET_MIB, {hyla_gpu})
    track_gpu = wait_gpu(MIN_TRACK_MIB, {hyla_gpu, sum_gpu})
    hyla_proc, hyla_handle = launch(
        "det-hyla",
        detector_cmd(HYLA_EXP, "yolox_dsec_dual_hyla_shared_mlp", 24),
        hyla_gpu,
        os.path.join(PROJECT, "third_party/ByteTrack"),
        "det_hyla.log",
    )
    sum_proc, sum_handle = launch(
        "det-sum",
        detector_cmd(SUM_EXP, "yolox_dsec_dual", 24),
        sum_gpu,
        os.path.join(PROJECT, "third_party/ByteTrack"),
        "det_sum.log",
    )
    base_proc, base_handle = launch(
        "track-baseline-train",
        tracker_cmd("mambatrack_dsec_baseline", BASE_CFG, None, 30),
        track_gpu,
        os.path.join(PROJECT, "ssm_tracker"),
        "track_baseline_train.log",
    )

    wait_proc("det-sum", sum_proc, sum_handle)
    wait_proc("track-baseline-train", base_proc, base_handle)
    hyla_code = hyla_proc.poll()
    if hyla_code not in (None, 0):
        hyla_handle.close()
        raise RuntimeError(f"det-hyla failed with code {hyla_code}")
    sum_ckpt = os.path.join(YOLO_OUT, "yolox_dsec_dual/best_ckpt.pth.tar")
    if not os.path.isfile(sum_ckpt):
        raise RuntimeError(f"missing {sum_ckpt}")
    gpu = wait_gpu(MIN_DET_MIB, set())
    for split in ("val", "test"):
        run(
            f"det-sum-{split}",
            det_cmd(SUM_EXP, sum_ckpt, split, os.path.join(DET_ROOT, "dsec_sum", split), False),
            gpu,
            PROJECT,
            f"det_sum_{split}.log",
        )

    base_rows = []
    gpu = wait_gpu(MIN_TRACK_MIB, set())
    for epoch in range(1, 31):
        ckpt = os.path.join(CKPT_ROOT, "mambatrack_dsec_baseline", f"epoch{epoch}.pth")
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"missing {ckpt}")
        metrics = track_and_eval(
            f"dsec_baseline_ep{epoch}",
            ckpt,
            BASE_CFG,
            os.path.join(DET_ROOT, "dsec_sum", "val"),
            "val",
            gpu,
        )
        metrics["epoch"] = epoch
        base_rows.append(metrics)
        with open(os.path.join(LOG_DIR, "baseline_val.json"), "w") as handle:
            json.dump(base_rows, handle, indent=2)
    chosen = best_epoch(base_rows)
    log(f"baseline val best epoch {chosen['epoch']} HOTA {chosen['HOTA']}")
    base_ckpt = os.path.join(
        CKPT_ROOT, "mambatrack_dsec_baseline", f"epoch{chosen['epoch']}.pth"
    )
    test_metrics = track_and_eval(
        f"dsec_baseline_ep{chosen['epoch']}",
        base_ckpt,
        BASE_CFG,
        os.path.join(DET_ROOT, "dsec_sum", "test"),
        "test",
        gpu,
    )
    test_metrics["epoch"] = chosen["epoch"]
    with open(os.path.join(LOG_DIR, "baseline_test.json"), "w") as handle:
        json.dump(test_metrics, handle, indent=2)

    wait_proc("det-hyla", hyla_proc, hyla_handle)
    hyla_ckpt = os.path.join(
        YOLO_OUT, "yolox_dsec_dual_hyla_shared_mlp/best_ckpt.pth.tar"
    )
    if not os.path.isfile(hyla_ckpt):
        raise RuntimeError(f"missing {hyla_ckpt}")
    gpu = wait_gpu(MIN_DET_MIB, set())
    for split in ("train", "val"):
        run(
            f"feat-{split}",
            [
                PYTHON,
                "-u",
                os.path.join(PROJECT, "tools/gen_femot_gt_modal_features.py"),
                "--data_root",
                DATA,
                "--split",
                split,
                "--exp_file",
                HYLA_EXP,
                "--model_path",
                hyla_ckpt,
                "--output_dir",
                FEAT_ROOT,
                "--img_size",
                "416",
                "416",
                "--device",
                "0",
                "--enhanced",
            ],
            gpu,
            PROJECT,
            f"feat_{split}.log",
        )
    gpu = wait_gpu(MIN_TRACK_MIB, set())
    run(
        "track-hyla-train",
        tracker_cmd("dsec_hyla_drop_evt_motion_zero_b", HYLA_CFG, base_ckpt, 3),
        gpu,
        os.path.join(PROJECT, "ssm_tracker"),
        "track_hyla_train.log",
    )
    gpu = wait_gpu(MIN_DET_MIB, set())
    for split in ("val", "test"):
        src = os.path.join(DET_ROOT, "dsec_hyla", split)
        run(
            f"det-hyla-{split}",
            det_cmd(HYLA_EXP, hyla_ckpt, split, src, True),
            gpu,
            PROJECT,
            f"det_hyla_{split}.log",
        )
        run(
            f"hyla-feats-{split}",
            [
                PYTHON,
                "-u",
                os.path.join(PROJECT, "tools/gen_det_hyla_enhanced_feats.py"),
                "--src_dir",
                src,
                "--dst_dir",
                os.path.join(DET_ROOT, "dsec_hyla_feats", split),
                "--data_root",
                os.path.join(DATA, split),
                "--exp_file",
                HYLA_EXP,
                "--model_path",
                hyla_ckpt,
                "--img_size",
                "416",
                "416",
                "--device",
                "0",
            ],
            gpu,
            PROJECT,
            f"hyla_feats_{split}.log",
        )
    hyla_rows = []
    gpu = wait_gpu(MIN_TRACK_MIB, set())
    for epoch in (1, 2, 3):
        ckpt = os.path.join(
            CKPT_ROOT, "dsec_hyla_drop_evt_motion_zero_b", f"epoch{epoch}.pth"
        )
        metrics = track_and_eval(
            f"dsec_hyla_ep{epoch}",
            ckpt,
            HYLA_CFG,
            os.path.join(DET_ROOT, "dsec_hyla_feats", "val"),
            "val",
            gpu,
        )
        metrics["epoch"] = epoch
        hyla_rows.append(metrics)
        with open(os.path.join(LOG_DIR, "hyla_val.json"), "w") as handle:
            json.dump(hyla_rows, handle, indent=2)
    hyla_chosen = best_epoch(hyla_rows)
    log(f"hyla val best epoch {hyla_chosen['epoch']} HOTA {hyla_chosen['HOTA']}")
    hyla_test = track_and_eval(
        f"dsec_hyla_ep{hyla_chosen['epoch']}",
        os.path.join(
            CKPT_ROOT,
            "dsec_hyla_drop_evt_motion_zero_b",
            f"epoch{hyla_chosen['epoch']}.pth",
        ),
        HYLA_CFG,
        os.path.join(DET_ROOT, "dsec_hyla_feats", "test"),
        "test",
        gpu,
    )
    hyla_test["epoch"] = hyla_chosen["epoch"]
    with open(os.path.join(LOG_DIR, "hyla_test.json"), "w") as handle:
        json.dump(hyla_test, handle, indent=2)
    log("DSEC pipeline done")


if __name__ == "__main__":
    main()
