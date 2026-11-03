"""Illumination thresholds, manifests and LLSA calibration (data only, no model).

Thresholds (rule frozen 2026-10-05 before any training): train+val mean RGB
luminance, 5-unit bins, 3-bin moving average; valley = minimum bin between 60
and 140; tau_night / tau_day = lower / upper edge of the contiguous run of bins
around the valley whose smoothed count is <= 1.5 x the valley count.

LLSA calibration (rule frozen 2026-10-05): darken a seeded sample of 400 train
day images with LLSA for each (k_min, k_max) on a grid; keep pairs whose darkened
luminance p1 <= the train-night p5 and p99 >= the train-night p95; choose the
narrowest log-range, ties broken by the larger k_min. Val and test are not used.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def lum_table(proc, split):
    rows = list(csv.DictReader(open(Path(proc) / "stats" / f"{split}.csv")))
    return {r["id"]: (float(r["lum_rgb"]), int(r["n_rgb"]), int(r["n_ir"])) for r in rows}


def thresholds(lums):
    h, edges = np.histogram(lums, bins=np.arange(0, 260, 5))
    s = np.convolve(h, np.ones(3) / 3, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    cand = np.flatnonzero((centres > 60) & (centres < 140))
    v = cand[np.argmin(s[cand])]
    lo = hi = v
    while lo - 1 >= 0 and s[lo - 1] <= 1.5 * s[v]:
        lo -= 1
    while hi + 1 < len(s) and s[hi + 1] <= 1.5 * s[v]:
        hi += 1
    return float(edges[lo]), float(edges[hi + 1]), {"hist": h.tolist(), "smoothed": s.round(1).tolist(),
                                                     "valley_bin": float(edges[v])}


def bins(t, tau_night, tau_day):
    return {k: ("night" if v[0] < tau_night else "day" if v[0] >= tau_day else "twilight")
            for k, v in t.items()}


def calibrate(proc, train_t, tau_night, tau_day, n=400, seed=0):
    import torch
    from PIL import Image

    import lowlight as ll

    rng = np.random.default_rng(seed)
    day = sorted(k for k, v in train_t.items() if v[0] >= tau_day)
    night = np.array([v[0] for v in train_t.values() if v[0] < tau_night])
    target = np.percentile(night, [5, 95])
    pick = rng.choice(day, min(n, len(day)), replace=False)
    imgs = torch.stack([torch.from_numpy(np.asarray(
        Image.open(Path(proc) / "rgb" / "train" / "images" / f"{k}.jpg").convert("RGB").resize((320, 256)),
        dtype=np.float32) / 255).permute(2, 0, 1) for k in pick])
    grid_min = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    grid_max = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5]
    results = []
    for kmin in grid_min:
        for kmax in grid_max:
            torch.manual_seed(0)
            out = torch.cat([ll.llsa(imgs[i:i + 50], {"k_min": kmin, "k_max": kmax, "jpeg_p": 0.0})
                             for i in range(0, len(imgs), 50)])
            y = ll.luminance(out).numpy()
            p1, p99 = np.percentile(y, [1, 99])
            ok = p1 <= target[0] and p99 >= target[1]
            results.append({"k_min": kmin, "k_max": kmax, "p1": round(float(p1), 2),
                            "p99": round(float(p99), 2), "covers": bool(ok),
                            "log_range": math.log(kmax / kmin)})
    ok = [r for r in results if r["covers"]]
    best = min(ok, key=lambda r: (r["log_range"], -r["k_min"])) if ok else None
    rule = "frozen"
    if best is None:  # Amendment A1: the read-noise floor bounds p1; match the upper tail, minimise p1
        upper = [r for r in results if r["p99"] >= target[1]]
        best = min(upper, key=lambda r: (r["p1"], r["log_range"])) if upper else None
        rule = "amendment A1 fallback"
    return {"night_p5_p95": target.round(2).tolist(), "n_day_sample": len(pick), "grid": results,
            "chosen": best, "rule_used": rule}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-day", type=float, help="one-time audit adjustment of tau_day")
    ap.add_argument("--audit-note", default="")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = {s: lum_table(a.proc, s) for s in ("train", "val", "test")}
    tv = np.array([v[0] for s in ("train", "val") for v in t[s].values()])
    tau_night, tau_day, info = thresholds(tv)
    man = {"rule": __doc__.split("\n\n")[1], "tau_night_rule": tau_night, "tau_day_rule": tau_day, **info}
    if a.tau_day is not None:
        tau_day = a.tau_day
        man["audit_adjustment"] = {"tau_day": tau_day, "note": a.audit_note}
    man.update(tau_night=tau_night, tau_day=tau_day)
    summary = {}
    for s in t:
        b = bins(t[s], tau_night, tau_day)
        lists = {c: sorted(k for k, v in b.items() if v == c) for c in ("night", "twilight", "day")}
        for c, ids in lists.items():
            txt = "".join(i + "\n" for i in ids)
            (out / f"{s}_{c}.txt").write_text(txt)
            summary[f"{s}_{c}"] = {"images": len(ids), "rgb_boxes": sum(t[s][i][1] for i in ids),
                                   "ir_boxes": sum(t[s][i][2] for i in ids),
                                   "sha256": hashlib.sha256(txt.encode()).hexdigest()}
    man["subsets"] = summary
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("tau_night", tau_night, "tau_day", tau_day)
    for k, v in summary.items():
        print(k, v["images"], v["rgb_boxes"])
    cal = calibrate(a.proc, t["train"], tau_night, tau_day)
    (out / "llsa_calibration.json").write_text(json.dumps(cal, indent=1))
    print("LLSA chosen", cal["chosen"], "night p5/p95", cal["night_p5_p95"])


if __name__ == "__main__":
    main()
