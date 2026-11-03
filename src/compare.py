"""Paired bootstrap of AP50:95 differences between two models on the same split.

Usage: python src/compare.py A=path/preds_a.json B=path/preds_b.json [--split val] [--labels rgb] [--n 1000]
Prints delta (A - B) and 95 % CI per illumination band, resampling images and, for night,
also blocks of 100 consecutive image ids (sensitivity to correlated frames).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels", default="rgb")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    (na, pa), (nb, pb) = args.a.split("=", 1), args.b.split("=", 1)
    gt = D / args.labels / args.split / "boxes.json"
    ma, mb = ev.Matched(gt, json.load(open(pa))), ev.Matched(gt, json.load(open(pb)))
    assert list(ma.ids) == list(mb.ids)
    res = {}
    for band in ("all", "night", "twilight", "day"):
        w = (np.ones(len(ma.ids)) if band == "all"
             else ev.subset_weights(ma.ids, (M / f"{args.split}_{band}.txt").read_text().split()))
        r = ev.bootstrap_delta(ma, mb, w, n=args.n)
        r["a"], r["b"] = ma.ap(w), mb.ap(w)
        if band == "night":
            r["block100"] = ev.bootstrap_delta(ma, mb, w, n=args.n, block=100)["ci95"]
        res[band] = r
        extra = f" block100 CI [{r['block100'][0] * 100:+.2f}, {r['block100'][1] * 100:+.2f}]" if band == "night" else ""
        print(f"{args.split}/{args.labels} {band}: {na} {r['a'] * 100:.2f} vs {nb} {r['b'] * 100:.2f}  "
              f"delta {r['delta'] * 100:+.2f}  CI95 [{r['ci95'][0] * 100:+.2f}, {r['ci95'][1] * 100:+.2f}]{extra}",
              flush=True)
    out = HERE.parent / "runs/analysis" / f"compare_{args.split}_{args.labels}_{na}_vs_{nb}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
