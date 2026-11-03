"""Run job_main.predict on the local CPU (no Colab compute units).

Same code path as a remote "predict" job (conf 0.001, max_det 300, student export,
32-image batches); on CPU Ultralytics runs FP32 instead of the FP16 used on the L4.
Outputs go to runs/local/<name>/ws/out/<name>/preds_<split>[_ir].json so remote
snapshots are never touched.

Run one model at a time under the memory cap:
  systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=0 \
    python src/predict_local.py <name> <weights.pt> <split> [--data-root ir] [--imgsz 640]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("DV_DATA", str(ROOT / "data/dronevehicle"))

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("weights")
ap.add_argument("split", choices=["val", "test"])
ap.add_argument("--data-root", default="rgb", choices=["rgb", "ir"])
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()

ws = ROOT / "runs/local" / a.name / "ws"
(ws / "inputs").mkdir(parents=True, exist_ok=True)
w = ws / "inputs" / "model.pt"
shutil.copy2(ROOT / a.weights, w)
os.chdir(ws)  # job_main.WS is the current directory
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

torch.set_num_threads(a.threads)
import job_main  # noqa: E402

job = {"name": a.name, "kind": "predict",
       "predict": {"weights": "inputs/model.pt", "split": a.split, "data_root": a.data_root, "imgsz": a.imgsz}}
(ws / "job.json").write_text(json.dumps(job, indent=1))
job_main.log("local predict", json.dumps(job["predict"]), "source", a.weights)
job_main.predict(job)
shutil.rmtree(ws / "inputs", ignore_errors=True)
