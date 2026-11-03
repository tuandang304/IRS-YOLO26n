"""Label-visibility gap analysis (local, no GPU).

1. gap:   per split and illumination band, how many IR-annotated vehicles have no RGB box
          (Hungarian IoU >= 0.3 matching from dv_pairs), and how large they are.
2. hidden: for each model's val predictions, recall of IR-only vehicles (absent from the RGB
          labels used for scoring) vs recall of RGB-labeled vehicles, and the share of confident
          detections that RGB scoring counts as false positives but that hit an IR-only vehicle.

Usage: python src/analysis_visibility.py gap
       python src/analysis_visibility.py hidden name=path/to/preds_val.json [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv_pairs import iou_matrix, match  # noqa: E402

D = HERE.parent / "data/dronevehicle/proc"
M = HERE.parent / "splits"
BANDS = ("night", "twilight", "day")


def band_of(split):
    out = {}
    for b in BANDS:
        for i in (M / f"{split}_{b}.txt").read_text().split():
            out[i] = b
    return out


def split_boxes(split):
    rgb = json.load(open(D / "rgb" / split / "boxes.json"))["images"]
    ir = json.load(open(D / "ir" / split / "boxes.json"))["images"]
    res = {}
    for k in sorted(rgb):
        pairs, only = match(rgb[k], ir[k])
        res[k] = {"rgb": rgb[k], "ir": ir[k], "ir_only": [ir[k][j] for j in only],
                  "ir_matched": [ir[k][j] for _, j, _ in pairs]}
    return res


def gap():
    rows = []
    for split in ("train", "val", "test"):
        bands, boxes = band_of(split), split_boxes(split)
        for b in BANDS + ("all",):
            ks = [k for k in boxes if b == "all" or bands.get(k) == b]
            n_rgb = sum(len(boxes[k]["rgb"]) for k in ks)
            n_ir = sum(len(boxes[k]["ir"]) for k in ks)
            only = [x for k in ks for x in boxes[k]["ir_only"]]
            matched = [x for k in ks for x in boxes[k]["ir_matched"]]
            area = lambda xs: float(np.median([(x[3] - x[1]) * (x[4] - x[2]) for x in xs])) if xs else 0.0
            rows.append({"split": split, "band": b, "images": len(ks), "rgb_boxes": n_rgb, "ir_boxes": n_ir,
                         "ir_only": len(only), "ir_only_pct_of_ir": round(100 * len(only) / max(n_ir, 1), 1),
                         "median_area_ir_only": round(area(only)), "median_area_matched": round(area(matched))})
    print("| split | band | images | RGB boxes | IR boxes | IR-only | IR-only % of IR | median area IR-only | median area matched |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['split']} | {r['band']} | {r['images']} | {r['rgb_boxes']} | {r['ir_boxes']} | {r['ir_only']} | "
              f"{r['ir_only_pct_of_ir']} | {r['median_area_ir_only']} | {r['median_area_matched']} |")
    return rows


def hidden(named, thr=(0.25, 0.5), split="val"):
    bands, boxes = band_of(split), split_boxes(split)
    out = {}
    for name, path in named:
        import gzip
        preds = json.load(gzip.open(path) if str(path).endswith(".gz") else open(path))
        by = {}
        for p in preds:
            by.setdefault(str(p["image_id"]), []).append(p)
        res = {}
        for b in ("night", "all"):
            ks = [k for k in boxes if b == "all" or bands.get(k) == b]
            for t in thr:
                hit_only = hit_rgb = n_only = n_rgb = fp_rgb = fp_on_only = n_det = 0
                for k in ks:
                    d = np.array([q["bbox"] for q in by.get(k, []) if q["score"] >= t]).reshape(-1, 4)
                    rgb = np.array([x[1:] for x in boxes[k]["rgb"]], float).reshape(-1, 4)
                    only = np.array([x[1:] for x in boxes[k]["ir_only"]], float).reshape(-1, 4)
                    n_det += len(d)
                    n_only += len(only)
                    n_rgb += len(rgb)
                    if len(d):
                        if len(only):
                            hit_only += int((iou_matrix(only, d).max(1) >= 0.5).sum())
                        if len(rgb):
                            hit_rgb += int((iou_matrix(rgb, d).max(1) >= 0.5).sum())
                        unmatched = (iou_matrix(d, rgb).max(1) < 0.5) if len(rgb) else np.ones(len(d), bool)
                        fp_rgb += int(unmatched.sum())
                        if len(only) and unmatched.any():
                            fp_on_only += int((iou_matrix(d[unmatched], only).max(1) >= 0.5).sum())
                res[f"{b}@{t}"] = {"recall_ir_only": round(100 * hit_only / max(n_only, 1), 1),
                                   "recall_rgb_labeled": round(100 * hit_rgb / max(n_rgb, 1), 1),
                                   "fp_under_rgb": fp_rgb, "fp_that_are_ir_only_vehicles": fp_on_only,
                                   "pct_fp_real": round(100 * fp_on_only / max(fp_rgb, 1), 1), "detections": n_det}
        out[name] = res
    for key in [f"{b}@{t}" for b in ("night", "all") for t in thr]:
        print(f"\n{key}: | model | recall IR-only | recall RGB-labeled | FP under RGB labels | of which IR-only vehicles | % |")
        print("|---|---:|---:|---:|---:|---:|")
        for name in out:
            r = out[name][key]
            print(f"| {name} | {r['recall_ir_only']} | {r['recall_rgb_labeled']} | {r['fp_under_rgb']} | "
                  f"{r['fp_that_are_ir_only_vehicles']} | {r['pct_fp_real']} |")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "gap":
        rows = gap()
        (HERE.parent / "runs/analysis").mkdir(parents=True, exist_ok=True)
        (HERE.parent / "runs/analysis/label_gap.json").write_text(json.dumps(rows, indent=1))
    else:
        split = sys.argv[2] if sys.argv[2] in ("val", "test") else "val"
        named = [a.split("=", 1) for a in sys.argv[2:] if "=" in a]
        res = hidden(named, split=split)
        (HERE.parent / "runs/analysis").mkdir(parents=True, exist_ok=True)
        suffix = "" if split == "val" else "_test"
        (HERE.parent / f"runs/analysis/hidden_recall{suffix}.json").write_text(json.dumps(res, indent=1))
