#!/usr/bin/env python3
"""Train RGB/spatial hypergraph tracker from baseline epoch30 and evaluate ep3/ep10."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
from run_dsec_pipeline import parse_metrics

CFG = "cfgs/dsec/dsec_hyla_ep30_rgb_spatial_init_baseline30_ep10.yaml"
DET = "/wangx_new/lyt/dsec_det_results/dsec_hyla_ep30_feats/test"
INIT = "/wangx_new/lyt/dsec_mamba/mambatrack_dsec_baseline/epoch30.pth"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--attempt", default="retry1")
    a = ap.parse_args(); name = f"dsec_hyla_ep30_rgb_spatial_init_baseline30_ep10_seed{a.seed}_{a.attempt}"
    logdir = PROJECT / "logs/dsec" / name; ckptdir = Path("/wangx_new/lyt/dsec_mamba") / name
    pred_root = PROJECT / "ssm_tracker/track_results" / f"dsec_hyla_ep30_feats_{name}"
    if logdir.exists() or ckptdir.exists() or pred_root.exists(): raise FileExistsError("output already exists")
    logdir.mkdir(parents=True)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR="/wangx_new/lyt/tmp", PYTHONHASHSEED=str(a.seed), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    man = {"name":name,"gpu":a.gpu,"seed":a.seed,"status":"starting","config":str(PROJECT/"ssm_tracker"/CFG),"init_ckpt":INIT,"det_path":DET,"checkpoint_dir":str(ckptdir),"pred_root":str(pred_root),"relations":["spatial","rgb"],"modulation":"delta,C; alpha_B frozen zero","detector":"HyLA shared-MLP epoch30","source_sha256":{},"commands":[]}
    for rel in ["ssm_tracker/models/spatial_hypergraph.py","ssm_tracker/train.py","ssm_tracker/dataset/hypergraph_rules.py","ssm_tracker/track_utils/tracker.py","ssm_tracker/"+CFG]: man["source_sha256"][rel]=hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()
    def save(): (logdir/"manifest.json").write_text(json.dumps(man,indent=2)+"\n")
    def run(stage, cmd, cwd):
        man["status"]=stage; man["commands"].append({"stage":stage,"argv":cmd,"cwd":str(cwd)}); save()
        with (logdir/(stage+".log")).open("w") as f: subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        seedcode=f"import random,numpy as np,torch,runpy; random.seed({a.seed}); np.random.seed({a.seed}); torch.manual_seed({a.seed}); runpy.run_path('train.py',run_name='__main__')"
        run("train",[sys.executable,"-u","-c",seedcode,"--exp_name",name,"--config_file",CFG,"--save_path",str(ckptdir.parent),"--device","0","--init_ckpt",INIT],PROJECT/"ssm_tracker")
        metrics={}
        for ep in (3,10):
            pred = pred_root / "test"
            run(f"track_ep{ep}_test",[sys.executable,"-u","track.py","--det_path",DET,"--motion_model_path",str(ckptdir/f"epoch{ep}.pth"),"--config_file",CFG,"--device","0"],PROJECT/"ssm_tracker")
            if len(list(pred.glob("*.txt"))) != 12: raise RuntimeError(f"expected 12 predictions for ep{ep}")
            stage=f"eval_ep{ep}_test"
            run(stage,["flock",str(PROJECT/"logs/dsec/trackeval.lock"),sys.executable,"-u",str(PROJECT/"trackeval/run_dsec_eval.py"),"--split","test","--pred_dir",str(pred),"--tracker_name",f"{name}_ep{ep}"],PROJECT)
            metrics[str(ep)] = parse_metrics((logdir/(stage+".log")).read_text())
        man["metrics"]=metrics; man["status"]="complete"; man["finished"]=time.strftime("%Y-%m-%d %H:%M:%S"); save()
    except Exception as e:
        man.update(status="failed",error=repr(e)); save(); raise

if __name__ == "__main__": main()
