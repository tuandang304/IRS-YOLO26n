"""Analyses requested by the round-0 paper review, from saved test predictions (CPU, no training).

1. thresholds: night recall of IR-only and RGB-annotated vehicles at score >= 0.1 / 0.25 / 0.5, and the
   IR-only recall when each model's threshold is set so that it reaches the same RGB-annotated recall
   (85 %, 90 %): rules out that the IR-only gain is a calibration shift.
2. iou: night AP50 and AP75 against both annotations (registration sensitivity of the IR-label gain).
Pairs: IRS-YOLO26n vs YOLO26n and IR-supervised YOLO11n vs YOLO11n (all 704).
Output: runs/analysis/review_analyses.json. Memory: one prediction file at a time.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402
from analysis_visibility import band_of, split_boxes  # noqa: E402
from dv_pairs import iou_matrix  # noqa: E402

ROOT = HERE.parent
R = ROOT / "runs/remote"
D = ROOT / "data/dronevehicle/proc"
M = ROOT / "splits"
PREDS = {"B1_704": R / "test_b1_704/ws/out/test_b1_704/preds_test.json",
         "P_704": R / "test_p_704/ws/out/test_p_704/preds_test.json",
         "YOLO11n_704": R / "test_yolo11n_704/ws/out/test_yolo11n_704/preds_test.json",
         "P11_704": R / "test_p11_704/ws/out/test_p11_704/preds_test.json"}


def recall_curves(preds, boxes, keys):
    """Per night vehicle, the best score of a detection with IoU >= 0.5 (0 if none), for IR-only and RGB boxes."""
    by = {}
    for p in preds:
        by.setdefault(str(p["image_id"]), []).append(p)
    best = {"iro": [], "rgb": []}
    for k in keys:
        d = by.get(k, [])
        db = np.array([q["bbox"] for q in d], float).reshape(-1, 4)
        ds = np.array([q["score"] for q in d], float)
        for name, gt in (("iro", boxes[k]["ir_only"]), ("rgb", boxes[k]["rgb"])):
            g = np.array([x[1:] for x in gt], float).reshape(-1, 4)
            if not len(g):
                continue
            if not len(db):
                best[name] += [0.0] * len(g)
                continue
            hit = iou_matrix(g, db) >= 0.5
            best[name] += list(np.where(hit, ds[None, :], 0.0).max(1))
    return {k: np.sort(np.array(v)) for k, v in best.items()}


def recall_at(scores, t):
    return float((scores >= t).mean() * 100)


def main():
    bands, boxes = band_of("test"), split_boxes("test")
    night = [k for k in boxes if bands.get(k) == "night"]
    out = {"thresholds": {}, "iou": {}}
    for name, path in PREDS.items():
        preds = json.load(open(path))
        c = recall_curves(preds, boxes, night)
        r = {f"t{t}": {"iro": round(recall_at(c["iro"], t), 1), "rgb": round(recall_at(c["rgb"], t), 1)}
             for t in (0.1, 0.25, 0.5)}
        for target in (85, 90):  # threshold that gives this recall of RGB-annotated vehicles
            t = float(np.quantile(c["rgb"], 1 - target / 100))
            r[f"rgb{target}"] = {"thr": round(t, 3), "iro": round(recall_at(c["iro"], t), 1)}
        out["thresholds"][name] = r
        for lab in ("rgb", "ir"):
            m = ev.Matched(D / lab / "test/boxes.json", preds)
            w = ev.subset_weights(m.ids, (M / "test_night.txt").read_text().split())
            out["iou"].setdefault(name, {})[lab] = {"ap": round(m.ap(w) * 100, 2),
                                                    "ap50": round(m.ap(w, 0.5) * 100, 2),
                                                    "ap75": round(m.ap(w, 0.75) * 100, 2)}
            del m
        del preds, c
        gc.collect()
        print(name, out["thresholds"][name], out["iou"][name], flush=True)
    (ROOT / "runs/analysis/review_analyses.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
