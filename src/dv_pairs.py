"""RGB-IR box matching: registration audit and IR-only boxes (Stage 0).

For each image, RGB and IR horizontal boxes are matched one-to-one by IoU
(Hungarian assignment on -IoU). A pair with IoU >= MATCH_IOU is a match; its
centre offset feeds the registration audit (gate G0). IR boxes without a match
are written as normalized xywh to <out>/ironly/<split>/<id>.txt, which the
paired trainer uses to mask TGD.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_IOU = 0.3
W, H = 640, 512


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter)


def match(rgb, ir):
    """Return (pairs [(i_rgb, j_ir, iou)], unmatched IR indices)."""
    m = iou_matrix([r[1:] for r in rgb], [r[1:] for r in ir])
    pairs, used = [], set()
    if m.size:
        rows, cols = linear_sum_assignment(-m)
        for i, j in zip(rows, cols):
            if m[i, j] >= MATCH_IOU:
                pairs.append((i, j, m[i, j]))
                used.add(j)
    return pairs, [j for j in range(len(ir)) if j not in used]


def run(proc, split):
    proc = Path(proc)
    rgb = json.load(open(proc / "rgb" / split / "boxes.json"))["images"]
    ir = json.load(open(proc / "ir" / split / "boxes.json"))["images"]
    out = proc / "ironly" / split
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    outc = proc / "ironly_cls" / split  # same boxes, with the IR class in front
    shutil.rmtree(outc, ignore_errors=True)
    outc.mkdir(parents=True, exist_ok=True)
    per_image = {}
    for k in sorted(rgb):
        pairs, only = match(rgb[k], ir[k])
        offs = []
        for i, j, _ in pairs:
            a, b = rgb[k][i], ir[k][j]
            offs.append(((b[1] + b[3] - a[1] - a[3]) / 2, (b[2] + b[4] - a[2] - a[4]) / 2))
        with open(out / f"{k}.txt", "w") as f, open(outc / f"{k}.txt", "w") as fc:
            for j in only:
                c, x1, y1, x2, y2 = ir[k][j]
                xywh = (f"{(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                f.write(xywh + "\n")
                fc.write(f"{int(c)} {xywh}\n")
        per_image[k] = {"n_rgb": len(rgb[k]), "n_ir": len(ir[k]), "matched": len(pairs),
                        "ir_only": len(only), "offsets": [[round(x, 2), round(y, 2)] for x, y in offs]}
    with open(proc / "stats" / f"pairs_{split}.json", "w") as f:
        json.dump(per_image, f)
    return per_image


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args(argv)
    for s in a.splits:
        r = run(a.proc, s)
        d = np.array([np.hypot(*o) for v in r.values() for o in v["offsets"]])
        print(s, "images", len(r), "matched", len(d), "ir_only", sum(v["ir_only"] for v in r.values()),
              "median_offset_px", round(float(np.median(d)), 2) if len(d) else None, flush=True)


if __name__ == "__main__":
    main()
