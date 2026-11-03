"""Deterministic DroneVehicle preprocessing (Stage 0).

Reads the split archives of the Hugging Face mirror McCheng/DroneVehicle
(pinned revision below) and writes, for each modality and split:

    <out>/<modality>/<split>/images/<id>.jpg   640x512, border removed
    <out>/<modality>/<split>/labels/<id>.txt   YOLO horizontal boxes
    <out>/<modality>/<split>/boxes.json        pixel boxes for the COCO evaluator

plus <out>/stats/<split>.csv (mean RGB luminance per image) and
<out>/stats/prep_report.json (counts, dropped boxes, class-name variants).

Archive layout: <split>/<split>img = RGB, <split>/<split>imgr = IR,
<split>/<split>label = RGB labels, <split>/<split>labelr = IR labels.
The same code runs locally and on Colab VMs, so both see identical data.
"""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HF_REPO = "McCheng/DroneVehicle"
HF_REVISION = "947bd13f002857eb79a3e5b4feb2c6aca453ade7"
BORDER = 100
W, H = 640, 512
JPEG_QUALITY = 95
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
NAME_MAP = {
    "car": "car", "truck": "truck", "truvk": "truck", "bus": "bus", "van": "van",
    "freight_car": "freight_car", "freight car": "freight_car",
    "feright_car": "freight_car", "feright car": "freight_car", "feright": "freight_car",
}
SKIP_NAMES = {"*"}  # one object per modality in train image 13099 has no class
MIN_INSIDE = 0.5  # keep a box only if at least half of its area is inside the image


def parse_objects(xml_bytes):
    """Return kept boxes [(cls_idx, x1, y1, x2, y2)] in cropped pixels and a drop log."""
    root = ET.fromstring(xml_bytes)
    kept, log = [], {"outside": 0, "no_geometry": 0, "degenerate": 0, "unlabeled": 0, "names": {}}
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        log["names"][raw] = log["names"].get(raw, 0) + 1
        if raw in SKIP_NAMES:
            log["unlabeled"] += 1
            continue
        if raw not in NAME_MAP:
            raise ValueError(f"unknown class name {raw!r}")
        cls = CLASSES.index(NAME_MAP[raw])
        poly, bnd = obj.find("polygon"), obj.find("bndbox")
        if poly is not None:
            xs = [float(poly.findtext(f"x{i}")) for i in range(1, 5)]
            ys = [float(poly.findtext(f"y{i}")) for i in range(1, 5)]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        elif bnd is not None:
            x1, y1 = float(bnd.findtext("xmin")), float(bnd.findtext("ymin"))
            x2, y2 = float(bnd.findtext("xmax")), float(bnd.findtext("ymax"))
        else:
            log["no_geometry"] += 1
            continue
        x1, x2, y1, y2 = x1 - BORDER, x2 - BORDER, y1 - BORDER, y2 - BORDER
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            log["degenerate"] += 1
            continue
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if cx2 - cx1 < 1 or cy2 - cy1 < 1 or (cx2 - cx1) * (cy2 - cy1) < MIN_INSIDE * area:
            log["outside"] += 1
            continue
        kept.append((cls, cx1, cy1, cx2, cy2))
    return kept, log


def _read(z, name, log):
    """Read a zip member; on a CRC mismatch, read it unchecked and record the name."""
    try:
        return z.read(name)
    except zipfile.BadZipFile:
        with z.open(name) as f:
            f._expected_crc = None  # read the stored bytes despite the CRC mismatch
            data = f.read()
        log.append(name)
        return data


def _crop(data):
    img = Image.open(io.BytesIO(data))
    img.load()  # raise on truncated or corrupt JPEG data
    img = img.convert("RGB")
    if img.size != (W + 2 * BORDER, H + 2 * BORDER):
        raise ValueError(f"unexpected image size {img.size}")
    return img.crop((BORDER, BORDER, BORDER + W, BORDER + H))


def _work(args):
    zip_path, split, ids, out = args
    out = Path(out)
    z = zipfile.ZipFile(zip_path)
    res = []
    for i in ids:
        row = {"id": i, "crc": []}
        try:
            imgs = {mod: _crop(_read(z, f"{split}/{split}{d}/{i}.jpg", row["crc"]))
                    for mod, d in (("rgb", "img"), ("ir", "imgr"))}
        except Exception as e:  # undecodable image: drop the pair, record it
            res.append({"id": i, "dropped": f"{type(e).__name__}: {e}", "crc": row["crc"]})
            continue
        if row["crc"]:  # decodable but unverifiable content: drop the pair too
            res.append({"id": i, "dropped": "CRC mismatch", "crc": row["crc"]})
            continue
        for mod, img_dir, lab_dir in (("rgb", "img", "label"), ("ir", "imgr", "labelr")):
            img = imgs[mod]
            d = out / mod / split
            img.save(d / "images" / f"{i}.jpg", quality=JPEG_QUALITY, subsampling=0)
            if mod == "rgb":
                a = np.asarray(img, dtype=np.float32)
                row["lum"] = float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())
            boxes, log = parse_objects(_read(z, f"{split}/{split}{lab_dir}/{i}.xml", row["crc"]))
            with open(d / "labels" / f"{i}.txt", "w") as f:
                for c, x1, y1, x2, y2 in boxes:
                    f.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            row[mod] = {"boxes": [[c, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                                  for c, x1, y1, x2, y2 in boxes], "log": log}
        res.append(row)
    return res


def prepare(zip_path, split, out, workers=os.cpu_count()):
    out = Path(out)
    z = zipfile.ZipFile(zip_path)
    rgb = sorted(Path(n).stem for n in z.namelist()
                 if n.startswith(f"{split}/{split}img/") and n.endswith(".jpg"))
    ir = sorted(Path(n).stem for n in z.namelist()
                if n.startswith(f"{split}/{split}imgr/") and n.endswith(".jpg"))
    if rgb != ir:
        raise ValueError(f"{split}: RGB and IR image ids differ")
    import shutil

    for mod in ("rgb", "ir"):
        shutil.rmtree(out / mod / split, ignore_errors=True)  # no stale files from earlier attempts
        for sub in ("images", "labels"):
            (out / mod / split / sub).mkdir(parents=True, exist_ok=True)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    chunks = [rgb[k:k + 200] for k in range(0, len(rgb), 200)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_work, [(str(zip_path), split, c, str(out)) for c in chunks]):
            rows.extend(part)
    rows.sort(key=lambda r: r["id"])
    dropped = {r["id"]: r["dropped"] for r in rows if "dropped" in r}
    crc = sorted(n for r in rows for n in r["crc"])
    rows = [r for r in rows if "dropped" not in r]
    for mod in ("rgb", "ir"):
        with open(out / mod / split / "boxes.json", "w") as f:
            json.dump({"classes": CLASSES, "width": W, "height": H,
                       "images": {r["id"]: r[mod]["boxes"] for r in rows}}, f)
    with open(out / "stats" / f"{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lum_rgb", "n_rgb", "n_ir"])
        for r in rows:
            w.writerow([r["id"], f"{r['lum']:.3f}", len(r["rgb"]["boxes"]), len(r["ir"]["boxes"])])
    report = {"split": split, "images": len(rows), "dropped": dropped, "crc_mismatch": crc}
    for mod in ("rgb", "ir"):
        agg = {"outside": 0, "no_geometry": 0, "degenerate": 0, "unlabeled": 0, "names": {}, "kept": 0,
               "max_per_image": 0}
        for r in rows:
            lg = r[mod]["log"]
            for k in ("outside", "no_geometry", "degenerate", "unlabeled"):
                agg[k] += lg[k]
            for n, c in lg["names"].items():
                agg["names"][n] = agg["names"].get(n, 0) + c
            agg["kept"] += len(r[mod]["boxes"])
            agg["max_per_image"] = max(agg["max_per_image"], len(r[mod]["boxes"]))
        agg["images_over_300"] = sum(len(r[mod]["boxes"]) > 300 for r in rows)
        report[mod] = agg
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with <split>.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    reports = {}
    for s in a.splits:
        reports[s] = prepare(Path(a.raw) / f"{s}.zip", s, a.out, a.workers)
        print(json.dumps(reports[s]), flush=True)
    path = Path(a.out) / "stats" / "prep_report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(reports)
    path.write_text(json.dumps(old, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
