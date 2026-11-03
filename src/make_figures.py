"""Paper figures (paper/figures/*.pdf and 600-dpi *.png).

fig_dataset      RGB/IR pairs for night, twilight and day with RGB boxes and IR-only boxes
fig_method       training/inference diagram of the IR-supervised YOLO26n
fig_label_gap    share of IR-annotated vehicles missing from the RGB annotation, per band and split
fig_qualitative  night detections of YOLO26n@704 and ours@704 against RGB and IR-only vehicles
fig_curves       validation mAP50:95 per epoch (Ultralytics), each detector without and with the IR supervision
fig_curves_var   the same for the YOLO26n variants that use the paired data differently (union, soft, plain KD)
fig_tradeoff     night AP (RGB labels) vs GFLOPs
fig_labels       test night AP against the RGB annotation vs against the IR annotation, all 704-pixel models
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analysis_visibility import band_of, split_boxes  # noqa: E402

ROOT = HERE.parent
D = ROOT / "data/dronevehicle/proc"
R = ROOT / "runs/remote"
OUT = ROOT / "paper/figures"
PKG = ROOT / "artifacts/results-v1"
# Every figure is drawn at its printed width (CMC: text at least 8 pt), so LaTeX does not scale it down.
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42})
W_FULL = 6.4    # \textwidth of the CMC template is 6.5 in (469.8 pt); a little less, so LaTeX never shrinks it
W_HALF = 3.05   # panel of a two-column figure (0.48\textwidth)
FS = 8
C_RGB, C_IRO, C_DET = "#2ca02c", "#d62728", "#1f77b4"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def img(mod, split, k):
    im = cv2.imread(str(D / mod / split / "images" / f"{k}.jpg"))
    return im[:, :, ::-1] if mod == "rgb" else im


def draw(ax, boxes, color, ls="-", lw=0.8):
    for b in boxes:
        x1, y1, x2, y2 = b[-4:]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls))


def fig_dataset(ids=(("night", "00007"), ("twilight", "00031"), ("day", "00017"))):
    """RGB images in the top row, the paired IR images below; one column per illumination band."""
    boxes = split_boxes("val")
    fig, axes = plt.subplots(2, len(ids), figsize=(W_FULL, 2 * (W_FULL / len(ids) / 1.25 + 0.25)))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.03, hspace=0.12)
    for c, (band, k) in enumerate(ids):
        for r, mod in enumerate(("rgb", "ir")):
            ax = axes[r, c]
            ax.imshow(img(mod, "val", k), interpolation="none")
            if mod == "rgb":
                draw(ax, boxes[k]["rgb"], C_RGB)
            else:
                draw(ax, boxes[k]["ir_matched"], C_RGB)
                draw(ax, boxes[k]["ir_only"], C_IRO, lw=1.0)
            ax.set_xticks([]), ax.set_yticks([])
            n = len(boxes[k]["rgb"]) if mod == "rgb" else len(boxes[k]["ir"])
            extra = f", {len(boxes[k]['ir_only'])} IR-only" if mod == "ir" else ""
            ax.set_title(f"({'abcdef'[3 * r + c]}) {band.capitalize()} {mod.upper()}: {n} boxes{extra}", fontsize=FS, loc="left")
    save(fig, "fig_dataset")


def box(ax, xy, w, h, text, fc, ec="#333333", fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, color="#333333", ls="-", text=None, toff=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9, ls=ls))
    if text:
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, ha="center", va="bottom",
                fontsize=FS, color=color)


def fig_method():
    gf = json.load(open(PKG / "gflops.json"))
    fig, ax = plt.subplots(figsize=(W_FULL, W_FULL * 9.4 / 14))
    fig.subplots_adjust(0, 0, 1, 1)  # the axes fill the printed width, so 8 pt stays 8 pt
    ax.set_xlim(0, 14), ax.set_ylim(0, 9.4), ax.axis("off")
    fs = FS
    ax.add_patch(Rectangle((0.05, 3.1), 13.9, 6.25, fill=False, ec="#999999", ls="--", lw=0.7))
    ax.text(0.2, 9.0, "Training only", fontsize=8, fontweight="bold", color="#555555")
    box(ax, (0.3, 7.2), 1.7, 1.2, "RGB image\n(704 px)", "#e8f1fb", fs=fs)
    box(ax, (2.5, 7.2), 1.8, 1.2, "YOLO26n\nbackbone", "#d6e6f7", fs=fs, bold=True)
    box(ax, (4.8, 7.2), 1.8, 1.2, "Neck\nP3 / P4 / P5", "#d6e6f7", fs=fs, bold=True)
    box(ax, (7.1, 7.95), 2.4, 0.6, "One-to-one head", "#d6e6f7", fs=fs)
    box(ax, (7.1, 7.05), 2.4, 0.6, "One-to-many head", "#d6e6f7", fs=fs)
    box(ax, (10.2, 6.95), 3.5, 1.65, "Detection loss\nRGB boxes: weight 1\nIR-only: weight $\\omega$ = 0.3",
        "#fff3cd", fs=fs)
    box(ax, (4.8, 5.3), 1.8, 0.9, "1x1 adapters", "#eeeeee", fs=fs)
    box(ax, (7.0, 4.95), 3.05, 1.6, "TGD loss ($\\lambda$ = 1)\nnormalized L2,\nGT-weighted; IR-only cells\nexcluded from background", "#fff3cd", fs=fs)
    box(ax, (0.3, 3.5), 1.7, 1.2, "Paired IR\nimage", "#fbe9e7", fs=fs)
    box(ax, (2.5, 3.5), 4.1, 1.2, "Frozen IR teacher\n(YOLO26n trained on IR images)\nneck features P3 / P4 / P5", "#f8d7d2", fs=fs)
    box(ax, (10.2, 3.4), 3.5, 1.75, "RGB-IR box matching\n(RGB vs IR annotation)\nHungarian, IoU $\\geq$ 0.3\nIR-only: no RGB match",
        "#eeeeee", fs=fs)
    arrow(ax, (2.0, 7.8), (2.5, 7.8))
    arrow(ax, (4.3, 7.8), (4.8, 7.8))
    arrow(ax, (6.6, 8.0), (7.1, 8.25))
    arrow(ax, (6.6, 7.6), (7.1, 7.35))
    arrow(ax, (9.5, 8.25), (10.2, 8.0))
    arrow(ax, (9.5, 7.35), (10.2, 7.55))
    arrow(ax, (5.7, 7.2), (5.7, 6.2))
    arrow(ax, (6.6, 5.75), (7.0, 5.75))
    arrow(ax, (2.0, 4.1), (2.5, 4.1))
    arrow(ax, (6.6, 4.3), (7.4, 4.95))
    ax.text(7.0, 3.92, "teacher features", fontsize=fs, color="#333333")
    arrow(ax, (12.6, 5.15), (12.6, 6.95), color="#b23b2e")
    ax.text(12.45, 5.95, "soft targets", fontsize=fs, color="#b23b2e", ha="right")
    arrow(ax, (10.2, 4.25), (9.6, 4.95), color="#b23b2e", ls="--")
    ax.text(9.55, 4.42, "IR-only mask", fontsize=fs, color="#b23b2e", ha="right")
    ax.add_patch(Rectangle((0.05, 0.1), 13.9, 2.75, fill=False, ec="#999999", ls="--", lw=0.7))
    ax.text(0.2, 2.45, "Deployment: unchanged YOLO26n, RGB input only", fontsize=8, fontweight="bold", color="#555555")
    box(ax, (0.3, 0.55), 1.7, 1.2, "RGB image", "#e8f1fb", fs=fs)
    box(ax, (2.5, 0.55), 7.0, 1.2, "YOLO26n, one-to-one head (NMS-free)\nteacher, adapters, IR input removed\n"
        f"{gf['p_704']:.2f} GFLOPs per 640x512 frame (704 px input)", "#d6e6f7", fs=fs)
    box(ax, (10.2, 0.55), 3.5, 1.2, "Vehicle boxes\n(5 classes)", "#e2f0d9", fs=fs)
    arrow(ax, (2.0, 1.15), (2.5, 1.15))
    arrow(ax, (9.5, 1.15), (10.2, 1.15))
    save(fig, "fig_method")


def fig_label_gap():
    rows = json.load(open(PKG / "label_gap.json"))
    bands, splits = ("night", "twilight", "day"), ("train", "val", "test")
    fig, ax = plt.subplots(figsize=(W_HALF, 2.2))
    w = 0.25
    for i, sp in enumerate(splits):  # values are listed in Table 4
        vals = [next(r["ir_only_pct_of_ir"] for r in rows if r["split"] == sp and r["band"] == b) for b in bands]
        ax.bar([j + (i - 1) * w for j in range(3)], vals, w, label=sp, color=["#4c72b0", "#dd8452", "#55a868"][i])
    ax.set_xticks(range(3), [b.capitalize() for b in bands])
    ax.set_ylabel("IR-annotated vehicles\nwithout RGB box (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_label_gap")


def fig_qualitative(ids=("00007", "00001"), thr=0.25):
    boxes = split_boxes("val")
    preds = {}
    for name, run in (("YOLO26n", "b1_704"), ("YOLO26n+P (ours)", "p_704")):
        by = {}
        for q in json.load(open(R / run / "ws/out" / run / "preds_val.json")):
            if q["score"] >= thr:
                by.setdefault(str(q["image_id"]), []).append(q["bbox"])
        preds[name] = by
    fig, axes = plt.subplots(len(ids), 3, figsize=(W_FULL, (W_FULL / 3 / 1.25 + 0.42) * len(ids)))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0.03, hspace=0.3)
    for r, k in enumerate(ids):
        ax = axes[r, 0]
        ax.imshow(img("ir", "val", k), interpolation="none")
        draw(ax, boxes[k]["ir_matched"], C_RGB)
        draw(ax, boxes[k]["ir_only"], C_IRO, lw=1.0)
        ax.set_title(f"({'abcdefghi'[3 * r]}) IR image: RGB-annotated (green),\nIR-only (red) vehicles", fontsize=FS, loc="left")
        for c, (name, by) in enumerate(preds.items(), start=1):
            ax = axes[r, c]
            ax.imshow(img("rgb", "val", k), interpolation="none")
            draw(ax, boxes[k]["ir_only"], C_IRO, ls="--", lw=0.7)
            draw(ax, by.get(k, []), C_DET, lw=0.9)
            ax.set_title(f"({'abcdefghi'[3 * r + c]}) {name}:\ndetections (blue)", fontsize=FS, loc="left")
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    save(fig, "fig_qualitative")


def curve(run):
    rows = [{k.strip(): v for k, v in x.items()} for x in csv.DictReader(open(R / run / "ws/out" / run / "results.csv"))]
    return [int(x["epoch"]) for x in rows], [float(x["metrics/mAP50-95(B)"]) * 100 for x in rows]


def _curves(runs, name=None, ylim=(40, 57.5), ax=None):
    """runs: (run, label, color, linestyle); validation mAP50:95 per epoch, zoomed on the second half."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    for run, lab, col, ls in runs:
        if (R / run / "ws/out" / run / "results.csv").exists():
            e, m = curve(run)
            ax.plot(e, m, label=lab, color=col, lw=1.0, ls=ls)
    ax.set_xlabel("Epoch"), ax.set_ylabel("Validation mAP50:95 (%)")
    ax.set_ylim(*ylim), ax.set_xlim(0, 61)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        save(fig, name)
    return ax


CURVES_MAIN = (("b1_704", "YOLO26n", "#4c72b0", "-"), ("yolo11n_704", "YOLO11n", "#55a868", "-"),
               ("v10_704", "YOLOv10n", "#dd8452", "-"), ("p_704", "YOLO26n+P", "#4c72b0", "--"),
               ("p11_704", "YOLO11n+P", "#55a868", "--"), ("p10_704", "YOLOv10n+P", "#dd8452", "--"))
CURVES_VAR = (("b1_704", "YOLO26n", "#4c72b0", "-"), ("m1_union_704", "+ union", "#64b5cd", "-"),
              ("m2_soft_704", "+ soft labels", "#937860", "-"), ("m3_plainkd_704", "+ feature KD", "#55a868", "-"),
              ("p_704", "+ P (ours)", "#c44e52", "--"))
TRADEOFF_PTS = (("b1_704", "YOLO26n", "o", "#4c72b0"), ("yolo11n_704", "YOLO11n", "s", "#55a868"),
                ("p_704", "YOLO26n+P", "*", "#c44e52"), ("v10_704", "YOLOv10n", "D", "#dd8452"),
                ("p10_704", "YOLOv10n+P", "X", "#c44e52"), ("p11_704", "YOLO11n+P", "P", "#c44e52"))


def tradeoff_data():
    """Three-seed means (with SDs) for the six deployed detectors; p11_704 shares YOLO11n's graph."""
    sm = seed_means()
    gf = json.load(open(PKG / "gflops.json"))
    pts, sds = [], {}
    for key, lab, mk, col in TRADEOFF_PTS:
        gf_key = "yolo11n_704" if key == "p11_704" else key
        pts.append((lab, gf[gf_key], sm[key]["rgb"][0], mk, col))
        sds[lab] = sm[key]["rgb"][1]
    return pts, sds


def labels_map():
    """{config: nested mean AP map} for fig_labels: three-seed means; single-run configs keep seed 0."""
    m0 = json.load(open(PKG / "metrics.json"))
    sm = seed_means()
    m = {}
    for k in set(m0) | set(sm):
        if k in sm:
            m[k] = {"test": {lab: {"night": sm[k][lab][0]} for lab in ("rgb", "ir")}}
        elif "test" in m0.get(k, {}):
            m[k] = m0[k]
    return m


def fig_results_combined():
    fig, axes = plt.subplots(2, 2, figsize=(W_FULL, 5.4), gridspec_kw={"wspace": 0.34, "hspace": 0.45})
    _curves(CURVES_MAIN, ax=axes[0, 0])
    _curves(CURVES_VAR, ylim=(50, 57.5), ax=axes[0, 1])
    pts, sds = tradeoff_data()
    # three-seed means only; their SDs are in Table 1 (whiskers made the panel unreadable at print size)
    fig_tradeoff(pts, arrows=(("YOLO26n", "YOLO26n+P"), ("YOLO11n", "YOLO11n+P"), ("YOLOv10n", "YOLOv10n+P")),
                 ax=axes[1, 0])
    fig_labels(labels_map(), ax=axes[1, 1])
    for ax, tag in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        ax.set_title(tag, loc="left", fontsize=8, fontweight="bold", pad=2)
    save(fig, "fig_results_combined")


def fig_labels(m, ax=None):
    """Test night AP50:95 against the RGB annotation (x) and the IR annotation (y) for every 704-pixel model.
    m: {config: {"test": {"rgb": {"night": mean}, "ir": {"night": mean}}}}; open markers = single-seed rows."""
    pts = (("b1_704", "YOLO26n", "o", "#4c72b0"), ("b2", "Zero-DCE++", "^", "#8172b3"),
           ("m1_union_704", "Union labels", "v", "#64b5cd"), ("m2_soft_704", "Soft labels", "<", "#937860"),
           ("m3_plainkd_704", "Feature KD", ">", "#55a868"), ("p_704", "YOLO26n+P", "*", "#c44e52"),
           ("yolo11n_704", "YOLO11n", "s", "#55a868"), ("p11_704", "YOLO11n+P", "P", "#c44e52"),
           ("v10_704", "YOLOv10n", "D", "#dd8452"), ("p10_704", "YOLOv10n+P", "X", "#c44e52"))
    SINGLE = {"b2", "m1_union_704"}  # trained once (seed 0); the rest show three-seed means
    off = {"Zero-DCE++": (6, -8, "left"), "YOLO26n": (5, -3, "left"), "Union labels": (5, -3, "left"),
           "Soft labels": (-5, -3, "right"), "YOLOv10n": (-5, -3, "right"), "YOLO26n+P": (-6, -1, "right"),
           "YOLO11n": (5, -3, "left"), "Feature KD": (3, -11, "left"), "YOLO11n+P": (5, -4, "left"),
           "YOLOv10n+P": (5, -3, "left")}
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    at = {}
    for k, lab, mk, col in pts:
        if k in m and "test" in m[k]:
            x, y = m[k]["test"]["rgb"]["night"], m[k]["test"]["ir"]["night"]
            at[lab] = (x, y)
            fc = "none" if k in SINGLE else col
            ax.scatter(x, y, marker=mk, facecolors=fc, edgecolors=col,
                       s=55 if mk == "*" else 24, zorder=3, linewidths=1.0)
            dx, dy, ha = off[lab]
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=FS)
    for a, b in (("YOLO26n", "YOLO26n+P"), ("YOLO11n", "YOLO11n+P"), ("YOLOv10n", "YOLOv10n+P")):
        if a in at and b in at:
            ax.annotate("", at[b], at[a], arrowprops=dict(arrowstyle="->", color="#c44e52", lw=0.7, ls="--",
                                                          shrinkA=4, shrinkB=5))
    ax.set_xlabel("Night AP50:95, RGB labels (%)"), ax.set_ylabel("Night AP50:95, IR labels (%)")
    ax.set_xlim(43.7, 46.6), ax.set_ylim(33.0, 37.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        save(fig, "fig_labels")
    return ax


def seed_means():
    """Per-config 3-seed mean and SD of test night AP (RGB and IR labels), from rev1_stats.json."""
    s = json.load(open(ROOT / "runs/analysis/rev1/rev1_stats.json"))["configs"]
    out = {}
    for k, v in s.items():
        cfg, split, lab, band = k.split("/")
        if split == "test" and band == "night":
            out.setdefault(cfg, {})[lab] = (v["mean"], v["sd"])
    return out


def fig_tradeoff(points, arrows=(), sds=None, ax=None):
    """points: list of (label, gflops, night_ap, marker, color); arrows: (label_from, label_to) pairs.
    sds: optional {label: SD over the three seeds}, drawn as vertical whiskers."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    at = {p[0]: (p[1], p[2]) for p in points}
    for a, b in arrows:
        if a in at and b in at:
            ax.annotate("", at[b], at[a], arrowprops=dict(arrowstyle="->", color="#c44e52", lw=0.8, ls="--",
                                                          shrinkA=4, shrinkB=5))
    for lab, g, apn, mk, col in points:
        if sds and lab in sds and sds[lab] is not None:
            ax.errorbar(g, apn, yerr=sds[lab], fmt="none", ecolor="#333333", elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=2)
        ax.scatter(g, apn, marker=mk, color=col, s=28, zorder=3)
        dx, dy, ha = {"YOLO26n": (7, 0, "left"), "YOLO26n+P": (7, -8, "left"), "YOLO11n": (7, -8, "left"),
                      "YOLO11n+P": (0, 10, "center"), "YOLOv10n": (-8, 0, "right"),
                      "YOLOv10n+P": (-8, 0, "right")}.get(lab, (5, -3, "left"))
        ax.annotate(lab, (g, apn), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=FS)
    ax.set_xlabel("GFLOPs per 640x512 frame"), ax.set_ylabel("Night AP50:95,\nRGB labels (%)")
    ax.set_xlim(5.6, 9.0)  # room for the point labels inside the axes, so all panels scale alike
    ax.set_ylim(43.9, 45.75)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5)), ax.yaxis.set_major_formatter("{x:.1f}")
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        save(fig, "fig_tradeoff")
    return ax


def fig_hidden(key="night@0.25"):
    """Test recall of IR-only night vehicles (score >= 0.25, IoU >= 0.5): YOLO26n and its variants (Table 2),
    then each detector without and with P. Validation values and Zero-DCE++ are in Table 5."""
    h = json.load(open(ROOT / "runs/analysis/hidden_recall_test.json"))
    h["P11_704"] = json.load(open(ROOT / "runs/analysis/hidden_recall_p11.json"))["test"]["P11_704"]
    h.update(json.load(open(ROOT / "runs/analysis/hidden_recall_r5.json"))["test"])
    bars = (("B1_704", "YOLO26n", "#4c72b0"), ("M1_704", "+ union", "#8c8c8c"), ("M2_704", "+ soft labels", "#8c8c8c"),
            ("M3_704", "+ feature KD", "#8c8c8c"), ("P_704", "+ P (ours)", "#c44e52"),
            ("YOLO11n_704", "YOLO11n", "#4c72b0"), ("P11_704", "YOLO11n+P", "#c44e52"),
            ("V10_704", "YOLOv10n", "#4c72b0"), ("P10_704", "YOLOv10n+P", "#c44e52"))
    xs = [0, 1, 2, 3, 4, 5.4, 6.4, 7.8, 8.8]  # gaps separate the YOLO26n group from the other detectors
    fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    for x, (k, lab, col) in zip(xs, bars):
        v = h[k][key]["recall_ir_only"]
        ax.bar(x, v, 0.75, color=col)
        ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=FS)
    ax.set_xticks(xs, [b[1] for b in bars], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Test recall of night\nIR-only vehicles (%)")
    ax.set_ylim(0, 58), ax.set_xlim(-0.6, 9.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_hidden")


def fig_blindspot_combined():
    """Submitted Fig. 4 layout: (a) annotation gap per split/band; (b) test recall of night IR-only vehicles.
    (b) uses three-seed means (runs/analysis/rev1/rev1_stats.json); the union variant was trained once (seed 0,
    hatched bar)."""
    rows = json.load(open(PKG / "label_gap.json"))
    rec = json.load(open(ROOT / "runs/analysis/rev1/rev1_stats.json"))["recall"]
    union = json.load(open(ROOT / "runs/analysis/hidden_recall_r5.json"))["test"]["M1_704"]["night@0.25"]["recall_ir_only"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"wspace": 0.38})
    bands, splits = ("night", "twilight", "day"), ("train", "val", "test")
    w = 0.25
    for i, sp in enumerate(splits):
        vals = [next(r["ir_only_pct_of_ir"] for r in rows if r["split"] == sp and r["band"] == bd) for bd in bands]
        a.bar([j + (i - 1) * w for j in range(3)], vals, w, label=sp, color=["#4c72b0", "#dd8452", "#55a868"][i])
    a.set_xticks(range(3), [bd.capitalize() for bd in bands])
    a.set_ylabel("IR-annotated vehicles\nwithout RGB box (%)")
    a.legend(frameon=False)
    bars = (("b1_704", "YOLO26n", "#4c72b0"), ("union", "+ union", "#8c8c8c"), ("m2_soft_704", "+ soft labels", "#8c8c8c"),
            ("m3_plainkd_704", "+ feature KD", "#8c8c8c"), ("p_704", "+ P (ours)", "#c44e52"),
            ("yolo11n_704", "YOLO11n", "#4c72b0"), ("p11_704", "YOLO11n+P", "#c44e52"),
            ("v10_704", "YOLOv10n", "#4c72b0"), ("p10_704", "YOLOv10n+P", "#c44e52"))
    xs = [0, 1, 2, 3, 4, 5.4, 6.4, 7.8, 8.8]  # gaps separate the YOLO26n group from the other detectors
    for x, (k, lab, col) in zip(xs, bars):
        v = union if k == "union" else sum(rec[k]) / len(rec[k])
        b.bar(x, v, 0.75, color=col, hatch="///" if k == "union" else None, edgecolor="white", linewidth=0)
        b.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=FS)
    b.set_xticks(xs, [bb[1] for bb in bars], rotation=45, ha="right", rotation_mode="anchor")
    b.set_ylabel("Test recall of night\nIR-only vehicles (%)")
    b.set_ylim(0, 58), b.set_xlim(-0.6, 9.4)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(tag, loc="left", fontsize=8, fontweight="bold", pad=2)
    save(fig, "fig_blindspot_combined")


if __name__ == "__main__":
    fig_dataset()
    fig_method()
    fig_label_gap()
    fig_qualitative()
    fig_results_combined()
    fig_blindspot_combined()
