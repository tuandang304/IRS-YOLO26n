"""Score one prediction file (COCO AP50:95 and AP50, %) against RGB and/or IR labels: all / night / day.

  python src/score_local.py <preds.json> <val|test> [--labels rgb ir] [--out result.json]

Same evaluator (src/evaluate.py) and illumination manifests as the paper tables. Run one file at a
time under the 7G cap (a test evaluator holds 8,980 images).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

D = HERE.parent / "data/dronevehicle/proc"
M = HERE.parent / "splits"

ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("split", choices=["val", "test"])
ap.add_argument("--labels", nargs="+", default=["rgb", "ir"])
ap.add_argument("--out")
a = ap.parse_args()

preds = json.load(open(a.preds))
res = {"preds": a.preds, "split": a.split}
for lab in a.labels:
    m = ev.Matched(D / f"{lab}/{a.split}/boxes.json", preds)
    ws = {"all": np.ones(len(m.ids))}
    for b in ("night", "day"):
        ws[b] = ev.subset_weights(m.ids, (M / f"{a.split}_{b}.txt").read_text().split())
    res[lab] = {k: {"ap": round(m.ap(w) * 100, 2), "ap50": round(m.ap(w, iou=0.5) * 100, 2)} for k, w in ws.items()}
    print(lab, json.dumps(res[lab]), flush=True)
    del m
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
