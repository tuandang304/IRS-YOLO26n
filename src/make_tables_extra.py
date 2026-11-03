"""Light-weight tables for the paper from files already in artifacts/results-v1 (no test re-scoring).

table_hidden.tex   recall of IR-only / RGB-labeled night vehicles (val and test, score >= 0.25)
table_boot.tex     paired bootstrap of ours vs each baseline (test; val in brackets)
table_gap.tex      IR-annotated vehicles without an RGB annotation, per split and band
res640.json        the 640-pixel check reported in the limitations (val only, RGB labels)
table_p11.tex      the same training recipe on YOLO11n (P-YOLO11n vs YOLO11n, from src/compare.py outputs)
Memory: at most two validation prediction files are loaded at once.
"""
import gc
import numpy as np
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "artifacts/results-v1"
GEN = ROOT / "paper/generated"
R = ROOT / "runs/remote"
D = ROOT / "data/dronevehicle/proc"
M = ROOT / "splits"
NAMES = {"B2": "Zero-DCE++ + YOLO26n", "B1_704": "YOLO26n", "M1_704": "YOLO26n + union labels",
         "M2_704": "YOLO26n + IR soft labels", "M3_704": "YOLO26n + IR feature KD", "P_704": "YOLO26n + P (ours)",
         "YOLO11n_704": "YOLO11n", "P11_704": "YOLO11n + P", "V10_704": "YOLOv10n", "P10_704": "YOLOv10n + P"}
# Table 3: ours minus every other way of using the data with the same YOLO26n network
BASE = {"b1_704": "YOLO26n (no IR)", "b2": "Zero-DCE++ enhancement", "m1_union_704": "Union of RGB and IR labels",
        "m2_soft_704": "IR soft labels", "m3_plainkd_704": "IR feature KD"}
# Table 1: each detector without / with P (pairs scored by make_results.py)
PLUGIN = [("YOLO26n", "yolo26n.yaml", "b1_704", "p_704"), ("YOLO11n", "yolo11n.yaml", "yolo11n_704", "p11_704"),
          ("YOLOv10n", "yolov10n.yaml", "v10_704", "p10_704")]
# Table 2: ways of using the IR data (or low-light enhancement) with YOLO26n
METHODS = [("b1_704", "None (baseline)", "--", "--"),
           ("b2", "Zero-DCE++ enhancement~\\cite{li2021zerodcepp}", "--", "enhancer"),
           ("m1_union_704", "Union of RGB and IR labels ($\\omega=1$)", "labels", "--"),
           ("m2_soft_704", "IR soft labels ($\\omega=0.3$), as in UA-CMDet~\\cite{sun2022dronevehicle}", "labels", "--"),
           ("m3_plainkd_704", "Feature KD from the IR teacher (uniform L2)", "features", "--"),
           ("p_704", "\\textbf{P (ours)}: IR soft labels + TGD", "labels + features", "--")]
COLS = [("rgb", "all"), ("rgb", "night"), ("ir", "all"), ("ir", "night")]


def _cell(v, best=False):
    return "--" if v is None else ("\\textbf{%.2f}" % v if best else "%.2f" % v)


def plugin():
    """table_plugin.tex: test AP of each detector without and with P, delta with paired-bootstrap marker."""
    from ultralytics.nn.tasks import DetectionModel

    m = json.load(open(OUT / "metrics.json"))
    b = json.load(open(OUT / "bootstrap.json"))
    gf = json.load(open(OUT / "gflops.json"))
    get = lambda k, lab, band: m.get(k, {}).get("test", {}).get(lab, {}).get(band)
    lines = []
    for n, (name, yaml, base, ours) in enumerate(PLUGIN):
        params = sum(p.numel() for p in DetectionModel(yaml, nc=5, verbose=False).parameters()) / 1e6
        g = gf.get(base)
        cost = f"{params:.2f} & {g:.2f}" if g else f"{params:.2f} & --"
        lines.append(f"{name} & {cost} & " + " & ".join(_cell(get(base, *c)) for c in COLS) + " \\\\")
        lines.append(f"\\quad + P & {cost} & " + " & ".join(_cell(get(ours, *c)) for c in COLS) + " \\\\")
        cells = []
        for lab, band in COLS:
            r = b.get(f"{ours}-{base}/test/{lab}/{band}")
            if r is None:
                cells.append("--")
                continue
            sig = r["ci95"][0] > 0 or r["ci95"][1] < 0
            cells.append(f"\\textit{{{r['delta']:+.2f}}}" + ("$^\\dagger$" if sig else ""))
        lines.append("\\quad $\\Delta$ & & & " + " & ".join(cells) + " \\\\")
        if n < len(PLUGIN) - 1:
            lines.append("\\midrule")
    (GEN / "table_plugin.tex").write_text("\n".join(lines) + "\n")


def methods():
    """table_methods.tex: test AP of YOLO26n with each way of using the IR data; bold = best per column."""
    m = json.load(open(OUT / "metrics.json"))
    get = lambda k, lab, band: m.get(k, {}).get("test", {}).get(lab, {}).get(band)
    best = {c: max((get(k, *c) or -1) for k, *_ in METHODS) for c in COLS}
    lines = []
    for key, desc, ir_use, extra in METHODS:
        cells = [_cell(get(key, *c), get(key, *c) is not None and get(key, *c) == best[c]) for c in COLS]
        lines.append(f"{desc} & {ir_use} & {extra} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_methods.tex").write_text("\n".join(lines) + "\n")


def hidden():
    rows = []
    for split, f in (("val", "hidden_recall.json"), ("test", "hidden_recall_test.json")):
        src = ROOT / "runs/analysis" / f
        if src.exists():
            shutil.copy2(src, OUT / f)
            rows.append((split, json.load(open(src))))
    p11 = ROOT / "runs/analysis/hidden_recall_p11.json"
    if p11.exists():  # IR-supervised YOLO11n, same analysis (src/analysis_visibility.hidden)
        extra = json.load(open(p11))
        for split, h in rows:
            h.setdefault("P11_704", extra[split]["P11_704"])
    r5 = ROOT / "runs/analysis/hidden_recall_r5.json"
    if r5.exists():  # round-5 models (YOLOv10n with/without P, YOLO26n alternatives), same analysis
        extra = json.load(open(r5))
        shutil.copy2(r5, OUT / r5.name)
        for split, h in rows:
            for key, v in extra.get(split, {}).items():
                h.setdefault(key, v)
    lines = []
    for key in NAMES:
        cells = []
        for split, h in rows:
            r = h.get(key, {}).get("night@0.25")
            cells += [f"{r['recall_ir_only']:.1f}", f"{r['recall_rgb_labeled']:.1f}", f"{r['pct_fp_real']:.1f}"] if r \
                else ["--"] * 3
        if any(c != "--" for c in cells):
            lines.append(f"{NAMES[key]} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_hidden.tex").write_text("\n".join(lines) + "\n")
    # \\hr{<key>/<split>/<field>} with field in iro (IR-only recall), rgb (RGB-labeled recall), fp (% of FP real)
    mac = ["\\makeatletter", "\\providecommand{\\hr}[1]{\\@nameuse{hr@#1}}"]
    for split, h in rows:
        for key, v in h.items():
            r = v.get("night@0.25")
            if r:
                for f, k in (("iro", "recall_ir_only"), ("rgb", "recall_rgb_labeled"), ("fp", "pct_fp_real")):
                    mac.append(f"\\@namedef{{hr@{key}/{split}/{f}}}{{{r[k]:.1f}}}")
    mac.append("\\makeatother")
    (GEN / "hidden_macros.tex").write_text("\n".join(mac) + "\n")


def boot():
    b = json.load(open(OUT / "bootstrap.json"))
    lines = []
    for key, name in BASE.items():
        cells = []
        for lab in ("rgb", "ir"):
            for band in ("night", "all"):
                t, v = b.get(f"p_704-{key}/test/{lab}/{band}"), b.get(f"p_704-{key}/val/{lab}/{band}")
                if t is None:
                    cells.append("--")
                    continue
                sig = t["ci95"][0] > 0 or t["ci95"][1] < 0
                s = f"{t['delta']:+.2f}" + ("$^\\dagger$" if sig else "")
                s += f" [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}]"
                cells.append(s + (f" ({v['delta']:+.2f})" if v else ""))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_boot.tex").write_text("\n".join(lines) + "\n")


def gap():
    rows = json.load(open(OUT / "label_gap.json"))
    lines = []
    for split in ("train", "val", "test"):
        cells = []
        for band in ("night", "twilight", "day"):
            r = next(x for x in rows if x["split"] == split and x["band"] == band)
            cells.append(f"{r['ir_only']:,} / {r['ir_boxes']:,} ({r['ir_only_pct_of_ir']:.1f}\\%)")
        lines.append(f"{split.capitalize()} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_gap.tex").write_text("\n".join(lines) + "\n")
    mac = ["\\makeatletter", "\\providecommand{\\gap}[1]{\\@nameuse{gap@#1}}"]
    for r in rows:
        mac.append(f"\\@namedef{{gap@{r['split']}/{r['band']}}}{{{r['ir_only_pct_of_ir']:.1f}}}")
        mac.append(f"\\@namedef{{gap@{r['split']}/{r['band']}/aonly}}{{{r['median_area_ir_only']:,}}}")
        mac.append(f"\\@namedef{{gap@{r['split']}/{r['band']}/amatch}}{{{r['median_area_matched']:,}}}")
    mac.append("\\makeatother")
    (GEN / "gap_macros.tex").write_text("\n".join(mac) + "\n")


def res640():
    out = {}
    for key, run in (("b1_640", "b1"), ("p_640", "p_640")):
        preds = json.load(open(R / run / "ws/out" / run / "preds_val.json"))
        m = ev.Matched(D / "rgb/val/boxes.json", preds)
        w = ev.subset_weights(m.ids, (M / "val_night.txt").read_text().split())
        out[key] = {"night": round(m.ap(w) * 100, 2), "all": round(m.ap(np.ones(len(m.ids))) * 100, 2)}
        del preds, m
        gc.collect()
    out["delta_night"] = round(out["p_640"]["night"] - out["b1_640"]["night"], 2)
    out["delta_all"] = round(out["p_640"]["all"] - out["b1_640"]["all"], 2)
    (OUT / "res640.json").write_text(json.dumps(out, indent=1))
    (GEN / "res640.tex").write_text(
        f"\\newcommand{{\\resNightDelta}}{{{out['delta_night']:+.2f}}}\n"
        f"\\newcommand{{\\resAllDelta}}{{{out['delta_all']:+.2f}}}\n")


def latency():
    src = R / "latency_l4/ws/out/latency_l4/latency.json"
    shutil.copy2(src, OUT / "latency.json")
    lat = json.load(open(src))
    gf = json.load(open(OUT / "gflops.json"))
    mac = ["\\makeatletter", "\\providecommand{\\lat}[1]{\\@nameuse{lat@#1}}",
           "\\providecommand{\\gf}[1]{\\@nameuse{gf@#1}}"]
    for k, v in lat.items():
        mac.append(f"\\@namedef{{lat@{k}}}{{{v['inference_ms']:.2f}}}")
    for k, v in gf.items():
        if not k.startswith("_"):
            mac.append(f"\\@namedef{{gf@{k}}}{{{v:.2f}}}")
    mac.append("\\makeatother")
    (GEN / "cost_macros.tex").write_text("\n".join(mac) + "\n")


def p11():
    """Parse runs/analysis/compare_p11_<split>_<lab>.txt (src/compare.py output) into a table and macros."""
    pat = re.compile(r"^(\w+)/(\w+) (\w+): P11 ([\d.]+) vs Y11 ([\d.]+)  delta ([+-][\d.]+)  "
                     r"CI95 \[([+-][\d.]+), ([+-][\d.]+)\]")
    res = {}
    for split in ("val", "test"):
        for lab in ("rgb", "ir"):
            f = ROOT / f"runs/analysis/compare_p11_{split}_{lab}.txt"
            for line in (f.read_text().splitlines() if f.exists() else []):
                m = pat.match(line)
                if m:
                    res[f"{m[1]}/{m[2]}/{m[3]}"] = {"p11": float(m[4]), "yolo11n": float(m[5]), "delta": float(m[6]),
                                                    "ci95": [float(m[7]), float(m[8])]}
    (OUT / "p11_compare.json").write_text(json.dumps(res, indent=1))
    lines, mac = [], ["\\makeatletter", "\\providecommand{\\pe}[1]{\\@nameuse{pe@#1}}"]
    for split in ("val", "test"):
        for lab in ("rgb", "ir"):
            cells = []
            for band in ("night", "twilight", "day", "all"):
                r = res.get(f"{split}/{lab}/{band}")
                if r is None:
                    cells.append("--")
                    continue
                sig = r["ci95"][0] > 0 or r["ci95"][1] < 0
                cells.append(f"{r['yolo11n']:.2f} / {r['p11']:.2f} ({r['delta']:+.2f}" + ("$^\\dagger$" if sig else "")
                             + ")")
                for k, v in (("y", r["yolo11n"]), ("p", r["p11"]), ("d", r["delta"]), ("lo", r["ci95"][0]),
                             ("hi", r["ci95"][1])):
                    mac.append(f"\\@namedef{{pe@{split}/{lab}/{band}/{k}}}{{{v:+.2f}}}" if k in ("d", "lo", "hi")
                               else f"\\@namedef{{pe@{split}/{lab}/{band}/{k}}}{{{v:.2f}}}")
            lines.append(f"{split.capitalize()} & {lab.upper()} & " + " & ".join(cells) + " \\\\")
    hp = ROOT / "runs/analysis/hidden_recall_p11.json"
    if hp.exists():  # IR-only recall of P-YOLO11n and YOLO11n (src/analysis_visibility.hidden, night, score >= 0.25)
        shutil.copy2(hp, OUT / hp.name)
        for split, h in json.load(open(hp)).items():
            for key, v in h.items():
                r = v["night@0.25"]
                for f, k in (("iro", "recall_ir_only"), ("rgb", "recall_rgb_labeled"), ("fp", "pct_fp_real"),
                             ("nfp", "fp_under_rgb"), ("nfpiro", "fp_that_are_ir_only_vehicles")):
                    mac.append(f"\\@namedef{{pe@hr/{split}/{key}/{f}}}{{{r[k]:,}}}" if f.startswith("n")
                               else f"\\@namedef{{pe@hr/{split}/{key}/{f}}}{{{r[k]:.1f}}}")
    mac.append("\\makeatother")
    (GEN / "table_p11.tex").write_text("\n".join(lines) + "\n")
    (GEN / "p11_macros.tex").write_text("\n".join(mac) + "\n")


def review_analyses():
    """\\ra{<model>/<field>} from src/analysis_review.py: recall at thresholds / matched RGB recall, AP50/AP75."""
    src = ROOT / "runs/analysis/review_analyses.json"
    if not src.exists():
        return
    shutil.copy2(src, OUT / src.name)
    r = json.load(open(src))
    r5 = ROOT / "runs/analysis/review_thresholds_r5.json"  # round-5 models, same threshold check (runs/thresholds_r5.py)
    if r5.exists():
        shutil.copy2(r5, OUT / r5.name)
        r["thresholds"].update(json.load(open(r5)))
    mac = ["\\makeatletter", "\\providecommand{\\ra}[1]{\\@nameuse{ra@#1}}"]
    hid = {}  # night false positives under RGB labels that match no IR-only vehicle either (score >= 0.25)
    for f in ("hidden_recall_test.json", "hidden_recall_r5.json"):
        if (ROOT / "runs/analysis" / f).exists():
            h = json.load(open(ROOT / "runs/analysis" / f))
            hid.update(h.get("test", h))
    for key, h in hid.items():
        n = h["night@0.25"]
        mac.append(f"\\@namedef{{ra@{key}/fpother}}{{{n['fp_under_rgb'] - n['fp_that_are_ir_only_vehicles']:,}}}")
    for key, v in r["thresholds"].items():
        for t, x in v.items():
            for f, val in x.items():
                mac.append(f"\\@namedef{{ra@{key}/{t}/{f}}}{{{val}}}")
    for key, v in r["iou"].items():
        for lab, x in v.items():
            for f, val in x.items():
                mac.append(f"\\@namedef{{ra@{key}/{lab}/{f}}}{{{val:.2f}}}")
    base = {"P_704": "B1_704", "P11_704": "YOLO11n_704"}
    for key, b in base.items():
        for lab in ("rgb", "ir"):
            for f in ("ap50", "ap75"):
                mac.append(f"\\@namedef{{ra@{key}/{lab}/{f}/d}}{{{r['iou'][key][lab][f] - r['iou'][b][lab][f]:+.2f}}}")
    mac.append("\\makeatother")
    (GEN / "review_macros.tex").write_text("\n".join(mac) + "\n")


def fix_minus():
    """Typeset negative decimals in generated LaTeX with a true minus sign (idempotent)."""
    for f in GEN.glob("*.tex"):
        t = f.read_text()
        u = re.sub(r"(?<![\w$\\.-])-(?=\d+\.\d)", "$-$", t)
        if u != t:
            f.write_text(u)


if __name__ == "__main__":
    latency()
    hidden()
    boot()
    gap()
    p11()
    review_analyses()
    plugin()
    methods()
    if not (OUT / "res640.json").exists():
        res640()
    fix_minus()
    print("ok")
