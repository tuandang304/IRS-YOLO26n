"""Freeze the results package artifacts/results-v1 and generate every table/number of the paper.

Every number in the manuscript comes from the files this script writes:
  artifacts/results-v1/preds/<model>_<split>.json   copies of the scored predictions
  artifacts/results-v1/metrics.json                  AP50:95 per model/split/labels/band
  artifacts/results-v1/bootstrap.json                paired bootstrap deltas and 95 % CIs
  artifacts/results-v1/hidden_recall.json            IR-only vs RGB-labeled recall (val)
  artifacts/results-v1/MANIFEST.sha256               hashes of everything above
  paper/generated/*.tex                              LaTeX tables and \\newcommand numbers

Usage: python src/make_results.py [--splits val test] [--n-boot 1000]
Models whose predictions are missing are reported as "--" and listed in the log.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

ROOT = HERE.parent
D = ROOT / "data/dronevehicle/proc"
M = ROOT / "splits"
R = ROOT / "runs/remote"
OUT = ROOT / "artifacts/results-v1"
GEN = ROOT / "paper/generated"
BANDS = ("all", "night", "twilight", "day")

# name: (display, run dir, pred file stem pattern, input size, GFLOPs, params M, family)
MODELS = {
    "b2": ("Zero-DCE++ + YOLO26n", "b2_704", "preds_{s}.json", 704, None, None, "base"),
    # D-FINE-N@704 (dfine_n_704) was trained and scored (val and test, kept in metrics.json/bootstrap.json) but
    # removed from the paper by the user on 2026-10-07: 1.45x the inference GFLOPs of the compared nano models.
    "b1_704": ("YOLO26n", "b1_704", "preds_{s}.json", 704, None, None, "base"),
    "yolo11n_704": ("YOLO11n", "yolo11n_704", "preds_{s}.json", 704, None, None, "base"),
    "p_704": ("Ours (IRS-YOLO26n)", "p_704", "preds_{s}.json", 704, None, None, "ours"),
    "p11_704": ("Ours on YOLO11n", "p11_704", "preds_{s}.json", 704, None, None, "ours"),
    # round 3 (2026-10-07): third detector and other uses of IR on YOLO26n
    "v10_704": ("YOLOv10n", "v10_704", "preds_{s}.json", 704, None, None, "base"),
    "p10_704": ("Ours on YOLOv10n", "p10_704", "preds_{s}.json", 704, None, None, "ours"),
    "m1_union_704": ("YOLO26n + union IR labels", "m1_union_704", "preds_{s}.json", 704, None, None, "base"),
    "m2_soft_704": ("YOLO26n + IR soft labels", "m2_soft_704", "preds_{s}.json", 704, None, None, "base"),
    "m3_plainkd_704": ("YOLO26n + IR feature KD", "m3_plainkd_704", "preds_{s}.json", 704, None, None, "base"),
}
# test predictions come from dedicated predict jobs named test_<model>
TEST_RUN = {"b1": "test_b1", "yolo11n": "test_yolo11n", "b2": "test_b2_704", "dfine": "dfine_n_704",
            "p_640": "test_p_640", "b1_704": "test_b1_704", "yolo11n_704": "test_yolo11n_704",
            "p_704": "test_p_704", "p11_704": "test_p11_704", **{k: f"test_{k}" for k in (
                "v10_704", "p10_704", "m1_union_704", "m2_soft_704", "m3_plainkd_704")}}
# proxy screening (20 % train subset, 15 epochs, val): run, description, input size
PROXIES = [
    ("s1_b1_proxy", "Baseline", 640),
    ("s3x_tgd_only1", "TGD (IR teacher, feature)", 640),
    ("s1_soft0.3_tgd1", "IR soft labels + TGD", 640),
    ("s2_llsa", "Low-light sensor augmentation", 640),
    ("s2_dnc1", "Low-light augmentation + day-to-night consistency", 640),
    ("s2_fuse_feat", "IR soft labels ($\\omega$=0.1) + fusion feature KD", 640),
    ("s2_fuse_feat_head", "IR soft labels ($\\omega$=0.1) + fusion feature/head KD", 640),
    ("s2_soft0.3_fuse_feat_head", "IR soft labels + fusion feature/head KD", 640),
    ("s3_pl0.3_t0.3", "IR soft labels + TGD + fusion pseudo-labels", 640),
    ("s4_img704", "Baseline", 704),
    ("s4c_img704_s1", "IR soft labels + TGD", 704),
    ("s5_night2x_704", "Night oversampling", 704),
    ("s5_rgbkd_704", "RGB teacher, feature + head KD", 704),
    ("s7_o2okd_704", "RGB teacher, one-to-one head KD", 704),
    ("s4_img768_pred", "Baseline", 768),
]
PAIRS = [("p_704", "b1_704"), ("p_704", "yolo11n_704"), ("p_704", "b2"), ("p11_704", "yolo11n_704"),
         ("p10_704", "v10_704"), ("p_704", "m1_union_704"), ("p_704", "m2_soft_704"), ("p_704", "m3_plainkd_704"),
         ("m1_union_704", "b1_704"), ("m2_soft_704", "b1_704"), ("m3_plainkd_704", "b1_704")]


def pred_path(key, split):
    run = MODELS[key][1] if split == "val" else TEST_RUN[key]
    stem = MODELS[key][2].format(s=split)
    if key == "b2":
        stem = f"preds_{split}_rgb_zdce.json"
    if key == "dfine" and split == "test":
        stem = "preds_test.json.gz"
    p = R / run / "ws" / "out" / run / stem
    return p if p.exists() else None


def band_w(m, split, band):
    if band == "all":
        return np.ones(len(m.ids))
    return ev.subset_weights(m.ids, (M / f"{split}_{band}.txt").read_text().split())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-test", type=int, default=300)  # test has 6x more images
    ap.add_argument("--tables-only", action="store_true",
                    help="rewrite the LaTeX tables and macros from the saved metrics.json/bootstrap.json")
    ap.add_argument("--only", nargs="+", help="score only these model keys (and their pairs); skip the proxies")
    args = ap.parse_args()
    if args.tables_only:
        write_tex(json.loads((OUT / "metrics.json").read_text()), json.loads((OUT / "bootstrap.json").read_text()),
                  args.splits)
        return
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    # Memory: one model (or one pair) is held at a time; the local machine has 14 GiB RAM and a
    # test-set evaluator costs about 1-2 GiB per model and label set.
    import gc

    def load(key, split):
        p = pred_path(key, split)
        if p is None:
            return None
        dst = OUT / "preds" / (f"{key}_{split}.json" + (".gz" if p.suffix == ".gz" else ""))
        if not dst.exists() or dst.stat().st_mtime < p.stat().st_mtime:
            shutil.copy2(p, dst)
        return json.load(gzip.open(dst) if p.suffix == ".gz" else open(dst))

    old = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    metrics, missing = {}, []
    for split in args.splits:
        for key in (args.only or MODELS):
            preds = load(key, split)
            if preds is None:
                missing.append(f"{key}/{split}")
                continue
            for lab in ("rgb", "ir"):
                m = ev.Matched(D / lab / split / "boxes.json", preds)
                metrics.setdefault(key, {}).setdefault(split, {})[lab] = {
                    b: round(m.ap(band_w(m, split, b)) * 100, 2) for b in BANDS}
                del m
            del preds
            gc.collect()
            print("scored", key, split, metrics[key][split]["rgb"], flush=True)
            (OUT / "metrics.json").write_text(json.dumps({**old, **metrics}, indent=1))

    boot_path = OUT / "bootstrap.json"
    boot = json.loads(boot_path.read_text()) if boot_path.exists() else {}
    for split in args.splits:
        for a, b in PAIRS:
            if args.only and a not in args.only and b not in args.only:
                continue
            for lab in ("rgb", "ir"):
                keys = [f"{a}-{b}/{split}/{lab}/{band}" for band in ("all", "night")]
                if all(k in boot for k in keys):
                    continue
                pa, pb = load(a, split), load(b, split)
                if pa is None or pb is None:
                    continue
                ma = ev.Matched(D / lab / split / "boxes.json", pa)
                mb = ev.Matched(D / lab / split / "boxes.json", pb)
                del pa, pb
                for band, k in zip(("all", "night"), keys):
                    w = band_w(ma, split, band)
                    nb = args.n_boot if split == "val" else args.n_boot_test
                    r = ev.bootstrap_delta(ma, mb, w, n=nb)
                    rb = ev.bootstrap_delta(ma, mb, w, n=nb, block=100)
                    boot[k] = {"delta": round(r["delta"] * 100, 2), "ci95": [round(v * 100, 2) for v in r["ci95"]],
                               "ci95_block100": [round(v * 100, 2) for v in rb["ci95"]], "n_boot": nb}
                    print("boot", k, boot[k], flush=True)
                    boot_path.write_text(json.dumps(boot, indent=1))
                del ma, mb
                gc.collect()
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=1))
    if args.only:
        gflops()
        write_tex({**old, **metrics}, boot, ["val", "test"])
        print("missing:", missing)
        return

    prox = []
    for run, desc, res in PROXIES:
        p = R / run / "ws" / "out" / run / "preds_val.json"
        if not p.exists():
            missing.append(f"proxy {run}")
            continue
        preds = json.load(open(p))
        mr = ev.Matched(D / "rgb" / "val" / "boxes.json", preds)
        mi = ev.Matched(D / "ir" / "val" / "boxes.json", preds)
        prox.append({"run": run, "desc": desc, "imgsz": res,
                     "rgb_all": round(mr.ap(band_w(mr, "val", "all")) * 100, 2),
                     "rgb_night": round(mr.ap(band_w(mr, "val", "night")) * 100, 2),
                     "rgb_day": round(mr.ap(band_w(mr, "val", "day")) * 100, 2),
                     "ir_night": round(mi.ap(band_w(mi, "val", "night")) * 100, 2)})
    (OUT / "proxy_screening.json").write_text(json.dumps(prox, indent=1))
    (GEN / "table_proxy.tex").write_text("".join(
        f"{x['desc']} & {x['imgsz']} & {x['rgb_all']:.2f} & {x['rgb_night']:.2f} & {x['rgb_day']:.2f} & "
        f"{x['ir_night']:.2f} \\\\\n" for x in prox))

    for f in ("label_gap.json", "hidden_recall.json", "hidden_recall_test.json"):
        src = ROOT / "runs/analysis" / f
        if src.exists():
            shutil.copy2(src, OUT / f)
    for f in ("latency.json",):
        src = R / "latency_l4" / "ws" / "out" / "latency_l4" / f
        if src.exists():
            shutil.copy2(src, OUT / f)
    gflops()
    write_tex(metrics, boot, args.splits)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (OUT / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print("missing:", missing)


def gflops():
    """GFLOPs of each deployed model for one 640x512 frame at its input size (rect letterbox).
    D-FINE-N: the value reported by its authors for 640x640 input (not measured here)."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    out = {}
    for key, w in (("b1", "runs/remote/b1/ws/out/b1/weights/best.pt"),  # trained 5-class heads
                   ("yolo11n", "runs/remote/yolo11n/ws/out/yolo11n/weights/best.pt")):
        m = YOLO(str(ROOT / w)).model
        for res in (640, 704):
            h = int(-(-res * 512 / 640 // 32) * 32)  # rect letterbox height for a 640x512 frame
            out[key if res == 640 else f"{key}_{res}"] = round(get_flops(m, [h, res]), 2)
    out["b2"] = out["b1_704"]  # enhancer cost not included; see paper
    out["p_640"], out["p_704"] = out["b1"], out["b1_704"]
    out["p11_704"] = out["yolo11n_704"]  # same deployed graph as YOLO11n
    from ultralytics.nn.tasks import DetectionModel
    v10 = DetectionModel("yolov10n.yaml", nc=5, verbose=False)  # deployed YOLOv10n graph, 5 classes
    out["v10_704"] = out["p10_704"] = round(get_flops(v10, [576, 704]), 2)
    for k in ("m1_union_704", "m2_soft_704", "m3_plainkd_704"):  # training-only changes to YOLO26n
        out[k] = out["b1_704"]
    out["dfine"] = round(7.0 * (704 / 640) ** 2, 2)  # authors' 640x640 figure scaled to 704x704
    out["_note"] = "dfine: 7.0 GFLOPs reported by the D-FINE authors for 640x640, scaled by (704/640)^2; b2 excludes the Zero-DCE++ enhancer"
    (OUT / "gflops.json").write_text(json.dumps(out, indent=1))
    print("gflops", out, flush=True)


def write_tex(metrics, boot, splits):
    def row(key, split, lab):
        v = metrics.get(key, {}).get(split, {}).get(lab)
        return [None] * 4 if v is None else [v[b] for b in BANDS]

    gf = json.loads((OUT / "gflops.json").read_text()) if (OUT / "gflops.json").exists() else {}
    size = {"dfine": "704$\\times$704"}
    for split in splits:
        for lab in ("rgb", "ir"):
            lines = []
            best = [max((row(k, split, lab)[i] or -1) for k in MODELS) for i in range(4)]
            for key, (disp, *_rest) in MODELS.items():
                vals = row(key, split, lab)
                cells = [("\\textbf{%s}" % fmt(v) if v is not None and v == best[i] else fmt(v))
                         for i, v in enumerate(vals)]
                g = gf.get(key)
                lines.append(f"{disp} & {size.get(key, MODELS[key][3])} & {fmt(g) if g else '--'} & "
                             + " & ".join(cells) + " \\\\")
            (GEN / f"table_main_{split}_{lab}.tex").write_text("\n".join(lines) + "\n")
    # \\bd{<a>-<b>/<split>/<labels>/<band>} -> delta; \\bci{...} -> image CI; \\bcib{...} -> block-100 CI
    nums = ["\\makeatletter",
            "\\newcommand{\\bd}[1]{\\@nameuse{bd@#1}}",
            "\\newcommand{\\bci}[1]{\\@nameuse{bci@#1}}",
            "\\newcommand{\\bcib}[1]{\\@nameuse{bcib@#1}}",
            "\\newcommand{\\ap}[1]{\\@nameuse{ap@#1}}"]
    for k, v in boot.items():
        nums.append(f"\\@namedef{{bd@{k}}}{{{v['delta']:+.2f}}}")
        nums.append(f"\\@namedef{{bci@{k}}}{{[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]}}")
        nums.append(f"\\@namedef{{bcib@{k}}}{{[{v['ci95_block100'][0]:+.2f}, {v['ci95_block100'][1]:+.2f}]}}")
    for key, by in metrics.items():  # \\ap{<model>/<split>/<labels>/<band>}
        for split, bl in by.items():
            for lab, bands in bl.items():
                for band, val in bands.items():
                    nums.append(f"\\@namedef{{ap@{key}/{split}/{lab}/{band}}}{{{val:.2f}}}")
    nums.append("\\makeatother")
    (GEN / "numbers.tex").write_text("\n".join(nums) + "\n")


if __name__ == "__main__":
    main()
