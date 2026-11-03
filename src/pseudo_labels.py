"""Fusion-teacher pseudo-labels (round 2, S3 option 1).

predict: run the RGB+IR fusion teacher (4-channel R,G,B,IR) on train images and save its
         NMS detections (conf >= 0.05) as preds json.
build:   keep teacher boxes with score >= tau that match neither an RGB ground-truth box nor an
         IR-only box (IoU >= 0.5, class-agnostic), and write them as {id: [[cls, cx, cy, w, h]]}
         normalized, for LowLightDistill's pseudo soft labels. Prints statistics per illumination.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
D = HERE.parent / "data/dronevehicle/proc"
M = HERE.parent / "splits"


def predict(weights, ids, dst, conf=0.05, batch=16):
    from ultralytics.utils.nms import non_max_suppression

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"]).float().eval()
    out, t = {}, time.time()
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        xs = []
        for i in chunk:
            rgb = cv2.imread(str(D / f"rgb/train/images/{i}.jpg"))[:, :, ::-1]
            ir = cv2.imread(str(D / f"ir/train/images/{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            xs.append(torch.from_numpy(np.dstack([rgb, ir]).copy()).permute(2, 0, 1))
        x = torch.stack(xs).float() / 255
        with torch.no_grad():
            y = m(x)
        dets = non_max_suppression(y[0] if isinstance(y, (tuple, list)) else y, conf, 0.7, max_det=300)
        for i, d in zip(chunk, dets):
            out[i] = [[int(c), *[round(v, 1) for v in b], round(sc, 4)] for *b, sc, c in d.tolist()]
        if s % (batch * 20) == 0:
            print(f"{s + len(chunk)}/{len(ids)} {time.time() - t:.0f}s", flush=True)
    Path(dst).write_text(json.dumps(out))
    print("saved", dst, flush=True)


def iou(a, b):
    """a: [n,4], b: [m,4] xyxy -> [n,m]."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def build(preds, dst, tau):
    g = json.load(open(D / "rgb/train/boxes.json"))
    W, H = g["width"], g["height"]
    p = json.load(open(preds))
    bands = {b: set((M / f"train_{b}.txt").read_text().split()) for b in ("night", "twilight", "day")}
    res, n_new, n_img, per = {}, 0, 0, {b: [0, 0] for b in bands}
    for i, dets in p.items():
        gt = np.array([b[1:] for b in g["images"].get(i, [])], dtype=float).reshape(-1, 4)
        f = D / f"ironly/train/{i}.txt"
        io = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 4))
        if len(io):
            io = np.stack([(io[:, 0] - io[:, 2] / 2) * W, (io[:, 1] - io[:, 3] / 2) * H,
                           (io[:, 0] + io[:, 2] / 2) * W, (io[:, 1] + io[:, 3] / 2) * H], 1)
        d = np.array([x for x in dets if x[5] >= tau], dtype=float).reshape(-1, 6)
        ref = np.concatenate([gt, io], 0)
        keep = d[(iou(d[:, 1:5], ref).max(1) if len(ref) else np.zeros(len(d))) < 0.5] if len(d) else d
        band = next((b for b in bands if i in bands[b]), None)
        if band:
            per[band][0] += 1
            per[band][1] += len(keep)
        if len(keep):
            x1, y1, x2, y2 = keep[:, 1].clip(0, W), keep[:, 2].clip(0, H), keep[:, 3].clip(0, W), keep[:, 4].clip(0, H)
            res[i] = [[int(c), round((a + b) / 2 / W, 6), round((e + f_) / 2 / H, 6), round((b - a) / W, 6),
                       round((f_ - e) / H, 6)] for c, a, e, b, f_ in zip(keep[:, 0], x1, y1, x2, y2)]
            n_new += len(keep)
            n_img += 1
    Path(dst).write_text(json.dumps(res))
    n_gt = sum(len(g["images"].get(i, [])) for i in p)
    print(f"tau {tau}: {n_new} pseudo boxes on {n_img}/{len(p)} images (GT {n_gt});",
          " ".join(f"{b} {v[1] / max(v[0], 1):.2f}/img ({v[0]} imgs)" for b, v in per.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "build"])
    ap.add_argument("--weights")
    ap.add_argument("--ids")
    ap.add_argument("--preds")
    ap.add_argument("--out")
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "predict":
        torch.set_num_threads(16)
        predict(a.weights, json.load(open(a.ids)), a.out)
    else:
        build(a.preds, a.out, a.tau)
    sys.exit(0)
