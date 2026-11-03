"""COCO-style horizontal-box evaluation with paired bootstrap (Stage 5).

Per-image matching comes from pycocotools (maxDets=300). Accumulation is
re-implemented with per-image weights so that a bootstrap resample, or an
illumination subset, is a weight vector over images: an image drawn k times
contributes its detections and ground truth k times. With all weights equal to
one this reproduces COCOeval.accumulate (checked by `selfcheck`).

Predictions: JSON list of {"image_id": "<id>", "category_id": 0-4,
"bbox": [x1, y1, x2, y2], "score": s} in 640x512 pixel coordinates.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOUS = np.linspace(0.5, 0.95, 10)
REC = np.linspace(0, 1, 101)
MAXDET = 300


def _coco_gt(boxes_json):
    g = json.load(open(boxes_json))
    ids = sorted(g["images"])
    images, anns = [], []
    for n, k in enumerate(ids):
        images.append({"id": n, "file_name": k, "width": g["width"], "height": g["height"]})
        for c, x1, y1, x2, y2 in g["images"][k]:
            anns.append({"id": len(anns) + 1, "image_id": n, "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": 0})
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(g["classes"])]
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, ids, g["classes"]


class Matched:
    """Per-image matching results of one model, ready for weighted accumulation."""

    def __init__(self, boxes_json, preds, area="all", keep_ev=False):
        coco, self.ids, self.classes = _coco_gt(boxes_json)
        index = {k: n for n, k in enumerate(self.ids)}
        dets = [{"image_id": index[p["image_id"]], "category_id": int(p["category_id"]) + 1,
                 "bbox": [p["bbox"][0], p["bbox"][1], p["bbox"][2] - p["bbox"][0],
                          p["bbox"][3] - p["bbox"][1]], "score": float(p["score"])}
                for p in preds if p["image_id"] in index]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = coco.loadRes(dets) if dets else COCO()
            ev = COCOeval(coco, dt, "bbox")
            ev.params.maxDets = [1, 10, MAXDET]
            # only the requested area range: identical matches, a quarter of the memory
            a = ev.params.areaRngLbl.index(area)
            ev.params.areaRng, ev.params.areaRngLbl = [ev.params.areaRng[a]], [area]
            ev.evaluate()
        del dets
        a = 0
        n_img, n_area = len(ev.params.imgIds), len(ev.params.areaRng)
        self.per_cat = []
        for ci, _ in enumerate(ev.params.catIds):
            scores, matched, ignored, img_of, npig = [], [], [], [], np.zeros(n_img)
            for ii in range(n_img):
                e = ev.evalImgs[ci * n_area * n_img + a * n_img + ii]
                if e is None:
                    continue
                d = np.asarray(e["dtScores"])[:MAXDET]
                scores.append(d)
                matched.append(np.asarray(e["dtMatches"])[:, :MAXDET] > 0)
                ignored.append(np.asarray(e["dtIgnore"])[:, :MAXDET].astype(bool))
                img_of.append(np.full(len(d), ii))
                npig[ii] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            if scores:
                s = np.concatenate(scores)
                order = np.argsort(-s, kind="mergesort")
                self.per_cat.append({
                    "img": np.concatenate(img_of)[order],
                    "tp": np.concatenate(matched, 1)[:, order],
                    "ig": np.concatenate(ignored, 1)[:, order],
                    "npig": npig})
            else:
                self.per_cat.append({"img": np.zeros(0, int), "tp": np.zeros((10, 0), bool),
                                     "ig": np.zeros((10, 0), bool), "npig": npig})
        # COCOeval holds every detection several times over; keep it only for selfcheck
        self.ev = ev if keep_ev else None

    def ap(self, weights, iou=None):
        """Mean over classes (with ground truth) of AP; weights: per-image float array."""
        aps = self.class_ap(weights, iou)
        aps = aps[~np.isnan(aps)]
        return float(aps.mean()) if len(aps) else float("nan")

    def class_ap(self, weights, iou=None):
        t_sel = slice(None) if iou is None else [int(np.argmin(abs(IOUS - iou)))]
        out = []
        for c in self.per_cat:
            npig = float((c["npig"] * weights).sum())
            if npig == 0:
                out.append(np.nan)
                continue
            w = weights[c["img"]]
            vals = []
            for t in np.arange(10)[t_sel]:
                keep = ~c["ig"][t] & (w > 0)
                tpw = np.where(c["tp"][t] & keep, w, 0.0)
                fpw = np.where(~c["tp"][t] & keep, w, 0.0)
                m = keep
                tp, fp = np.cumsum(tpw[m]), np.cumsum(fpw[m])
                if len(tp) == 0:
                    vals.append(0.0)
                    continue
                rc = tp / npig
                pr = tp / (tp + fp + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                inds = np.searchsorted(rc, REC, side="left")
                q = np.zeros(len(REC))
                ok = inds < len(pr)
                q[ok] = pr[inds[ok]]
                vals.append(q.mean())
            out.append(float(np.mean(vals)))
        return np.array(out)


def selfcheck(m: Matched):
    """Compare the unweighted accumulation with pycocotools' own."""
    with contextlib.redirect_stdout(io.StringIO()):
        m.ev.accumulate()
    p = m.ev.eval["precision"][:, :, :, 0, -1]  # area all, maxDets 300
    return m.ap(np.ones(len(m.ids))), float(p[p > -1].mean())


def subset_weights(ids, subset_ids):
    s = set(subset_ids)
    return np.array([1.0 if k in s else 0.0 for k in ids])


def bootstrap_delta(ma: Matched, mb: Matched, base_w, n=1000, seed=0, block=None, iou=None):
    """Paired bootstrap of AP(a) - AP(b) over the images where base_w > 0.

    block: resample contiguous blocks of `block` image ids instead of images."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(base_w > 0)
    groups = [idx[i:i + block] for i in range(0, len(idx), block)] if block else [[i] for i in idx]
    point = ma.ap(base_w, iou) - mb.ap(base_w, iou)
    deltas = []
    for _ in range(n):
        w = np.zeros_like(base_w)
        for g in rng.integers(0, len(groups), len(groups)):
            w[groups[g]] += 1
        deltas.append(ma.ap(w, iou) - mb.ap(w, iou))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci95": [float(lo), float(hi)], "n_boot": n,
            "unit": f"block{block}" if block else "image"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    m = Matched(a.gt, json.load(open(a.preds)), keep_ev=a.selfcheck)
    print("AP50:95", m.ap(np.ones(len(m.ids))))
    if a.selfcheck:
        print("ours vs pycocotools", selfcheck(m))


if __name__ == "__main__":
    main()
