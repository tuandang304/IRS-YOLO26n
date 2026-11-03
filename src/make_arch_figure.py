"""Fig. 1 of the revision: layer-level architecture of IRS-YOLO26n (YOLO26n trained with P).

Layout follows paper/revision_r1/template_fig/hrn_arch.svg: Input / Backbone / Neck / Head panels,
one box per YOLO26n layer with its output shape, plus a band with the training-only IR components.
Shapes are the measured outputs of ultralytics 8.4.173 yolo26n.yaml (nc=5) for a 704x704 input.
Drawn at printed width (6.4 in) with all text at 8 pt or larger.

Writes paper/figures/fig_method_arch.{pdf,png,svg}.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from make_figures import OUT, R, W_FULL, img, split_boxes

INK = "#1d2329"
TRAIN = "#8a2f9c"
STYLE = {  # fill, edge (template palette)
    "Conv": ("#ddd3f5", "#8a74c9"), "C3k2": ("#bcdcf7", "#4a90d0"), "SPPF": ("#fbc48f", "#e08a34"),
    "C2PSA": ("#b4c3ce", "#3f4f5a"), "Upsample": ("#fbc9d0", "#d9596c"), "Concat": ("#e7e9eb", "#7f8a92"),
    "Detect": ("#bcefc4", "#46a65a"), "train": ("#efd0f4", TRAIN), "label": ("#ffe1d6", "#c8553d"),
}
PANELS = [  # x0, x1, title, fill, edge, title colour
    (0.02, 1.04, "Input", "#fafafa", "#9aa3a9", "#3f4f5a"),
    (1.10, 2.28, "Backbone", "#f2f8ea", "#7fae5b", "#2d6a1f"),
    (2.34, 4.86, "Neck", "#fdf0ef", "#df7f78", "#a83a32"),
    (4.92, 6.38, "Head", "#eef6fc", "#6fb2e0", "#1f6391"),
]
XB, XL, XR, XH = 1.69, 3.03, 4.21, 5.77      # column centres: backbone, neck (two), head
BW, BH = 1.00, 0.30                            # layer box size (in)
HW = 1.06                                      # head box width


def y(r):
    return 0.62 + 0.40 * r


def box(ax, xc, yc, title, sub=None, kind="Conv", w=BW, h=BH, fs=8):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=0.9))
    if sub is None:
        ax.text(xc, yc, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
    else:
        ax.text(xc, yc - 0.062, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
        ax.text(xc, yc + 0.072, sub, ha="center", va="center", fontsize=fs, color=INK)


def path(ax, pts, color=INK, ls="-", head=True, lw=0.8):
    """Polyline; the last segment carries the arrow head."""
    if len(pts) > 2:
        ax.add_line(Line2D([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw, ls=ls,
                           solid_capstyle="butt"))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=6.5,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def dot(ax, x, yy, color=INK):
    ax.plot([x], [yy], "o", ms=2.4, color=color, zorder=5)


def main(image_id="00007", thr=0.25):
    H = 6.86
    fig = plt.figure(figsize=(W_FULL, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FULL), ax.set_ylim(H, 0), ax.axis("off")

    # panels
    top, bottom = 0.32, 5.30
    for x0, x1, title, fc, ec, tc in PANELS:
        ax.add_patch(FancyBboxPatch((x0, top), x1 - x0, bottom - top, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=0.8, ls=(0, (4, 3))))
        ax.text((x0 + x1) / 2, 0.19, title, ha="center", va="center", fontsize=10, fontweight="bold", color=tc)
    band0, band1 = 5.40, 6.84
    ax.add_patch(FancyBboxPatch((0.02, band0), 6.36, band1 - band0, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#fbf3fc", ec=TRAIN, lw=0.8, ls=(0, (4, 3))))
    ax.text(0.10, band0 + 0.12, "Training only (removed at deployment)", ha="left", va="center", fontsize=9,
            fontweight="bold", color=TRAIN)

    # input image
    rgb = img("rgb", "val", image_id)
    iw = 0.92
    ih = iw * rgb.shape[0] / rgb.shape[1]
    ix0, iy0 = 0.07, y(0) - 0.14
    ax.imshow(rgb, extent=(ix0, ix0 + iw, iy0 + ih, iy0), interpolation="bilinear", zorder=2)
    ax.add_patch(Rectangle((ix0, iy0), iw, ih, fill=False, ec=INK, lw=0.5, zorder=3))
    ax.text(ix0 + iw / 2, iy0 + ih + 0.11, "RGB image", ha="center", va="center", fontsize=8, color=INK)
    ax.text(ix0 + iw / 2, iy0 + ih + 0.245, "704×704×3", ha="center", va="center", fontsize=8, color=INK)
    path(ax, [(ix0 + iw, y(0)), (XB - BW / 2, y(0))])

    # backbone
    bb = [("L0 · Conv", "352×352×16", "Conv"), ("L1 · Conv", "176×176×32", "Conv"),
          ("L2 · C3k2", "176×176×64", "C3k2"), ("L3 · Conv", "88×88×64", "Conv"),
          ("L4 · C3k2", "88×88×128", "C3k2"), ("L5 · Conv", "44×44×128", "Conv"),
          ("L6 · C3k2", "44×44×128", "C3k2"), ("L7 · Conv", "22×22×256", "Conv"),
          ("L8 · C3k2", "22×22×256", "C3k2"), ("L9 · SPPF", "22×22×256", "SPPF"),
          ("L10 · C2PSA", "22×22×256", "C2PSA")]
    for r, (t, s, k) in enumerate(bb):
        box(ax, XB, y(r), t, s, k)
        if r:
            path(ax, [(XB, y(r - 1) + BH / 2), (XB, y(r) - BH / 2)])

    # neck: top-down path (left column, upwards) and bottom-up path (right column, downwards)
    left = {11: (8, "L11 · Upsample", "44×44×256", "Upsample"), 12: (6, "L12 · Concat", "44×44×384", "Concat"),
            13: (5, "L13 · C3k2", "44×44×128", "C3k2"), 14: (4, "L14 · Upsample", "88×88×128", "Upsample"),
            15: (3, "L15 · Concat", "88×88×256", "Concat"), 16: (2, "L16 · C3k2", "88×88×64", "C3k2")}
    right = {17: (3, "L17 · Conv", "44×44×64", "Conv"), 18: (5, "L18 · Concat", "44×44×192", "Concat"),
             19: (6, "L19 · C3k2", "44×44×128", "C3k2"), 20: (8, "L20 · Conv", "22×22×128", "Conv"),
             21: (10, "L21 · Concat", "22×22×384", "Concat"), 22: (11, "L22 · C3k2", "22×22×256", "C3k2")}
    for col, xc in ((left, XL), (right, XR)):
        for r, t, s, k in col.values():
            box(ax, xc, y(r), t, s, k)
    for a, b in ((11, 12), (12, 13), (13, 14), (14, 15), (15, 16)):
        path(ax, [(XL, y(left[a][0]) - BH / 2), (XL, y(left[b][0]) + BH / 2)])
    for a, b in ((17, 18), (18, 19), (19, 20), (20, 21), (21, 22)):
        path(ax, [(XR, y(right[a][0]) + BH / 2), (XR, y(right[b][0]) - BH / 2)])
    bx1, lx0, lx1, rx0, rx1 = XB + BW / 2, XL - BW / 2, XL + BW / 2, XR - BW / 2, XR + BW / 2
    # lateral inputs
    path(ax, [(bx1, y(6)), (lx0, y(6))])                                   # L6 -> L12
    path(ax, [(bx1, y(4)), (2.42, y(4)), (2.42, y(3)), (lx0, y(3))])         # L4 -> L15
    path(ax, [(bx1, y(10)), (rx0, y(10))])                                 # L10 -> L21
    path(ax, [(2.46, y(10)), (2.46, y(8)), (lx0, y(8))])                    # L10 -> L11
    dot(ax, 2.46, y(10))
    path(ax, [(lx1, y(5)), (rx0, y(5))])                                   # L13 -> L18
    # outputs P3 / P4 / P5 to the head
    hx0 = XH - HW / 2
    path(ax, [(lx1, y(2)), (hx0, y(2))])                                   # L16 -> P3
    path(ax, [(XR, y(2)), (XR, y(3) - BH / 2)])                            # L16 -> L17
    dot(ax, XR, y(2))
    path(ax, [(rx1, y(6)), (hx0, y(6))])                                   # L19 -> P4
    path(ax, [(rx1, y(11)), (hx0, y(11))])                                 # L22 -> P5

    # head
    for r, t, s in ((2, "Detect P3", "stride 8"), (6, "Detect P4", "stride 16"), (11, "Detect P5", "stride 32")):
        box(ax, XH, y(r), t, s, "Detect", w=HW)
    hy0, hy1 = y(2) + BH / 2 + 0.12, y(6) - BH / 2 - 0.12
    ax.add_patch(FancyBboxPatch((hx0, hy0), HW, hy1 - hy0, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec="#6fb2e0", lw=0.7))
    for i, (t, bold) in enumerate((("One-to-one head", True), ("inference, NMS-free", False),
                                   ("One-to-many head", True), ("training only", False))):
        ax.text(XH, hy0 + 0.15 + 0.135 * i + (0.06 if i >= 2 else 0), t, ha="center", va="center", fontsize=8,
                fontweight="bold" if bold else "normal", color=INK)
    det = [q["bbox"] for q in json.load(open(R / "p_704/ws/out/p_704/preds_val.json"))
           if str(q["image_id"]) == image_id and q["score"] >= thr]
    oh = HW * rgb.shape[0] / rgb.shape[1]
    oy0 = y(6) + BH / 2 + 0.16
    ax.imshow(rgb, extent=(hx0, hx0 + HW, oy0 + oh, oy0), interpolation="bilinear", zorder=2)
    sx, sy = HW / rgb.shape[1], oh / rgb.shape[0]
    for x1, y1, x2, y2 in (b[-4:] for b in det):
        ax.add_patch(Rectangle((hx0 + x1 * sx, oy0 + y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy, fill=False,
                               ec="#ffd400", lw=0.45, zorder=3))
    ax.add_patch(Rectangle((hx0, oy0), HW, oh, fill=False, ec=INK, lw=0.5, zorder=4))
    ax.text(XH, oy0 + oh + 0.12, "Output detections", ha="center", va="center", fontsize=8, color=INK)
    ax.text(XH, oy0 + oh + 0.255, f"(score ≥ {thr})", ha="center", va="center", fontsize=8, color=INK)

    # module legend (input panel)
    ly = 1.88
    ax.text(0.53, ly, "Modules", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    for i, (k, d) in enumerate((("Conv", "3×3, stride 2"), ("C3k2", "CSP bottleneck"), ("SPPF", "pyramid pooling"),
                                ("C2PSA", "CSP + attention"), ("Upsample", "×2, nearest"), ("Concat", "channel concat"),
                                ("Detect", "detection head"))):
        yy = ly + 0.24 + 0.36 * i
        fc, ec = STYLE[k]
        ax.add_patch(FancyBboxPatch((0.09, yy - 0.05), 0.10, 0.10, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(0.25, yy, k, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(0.09, yy + 0.15, d, ha="left", va="center", fontsize=8, color=INK)

    # training-only band
    r1, r2 = band0 + 0.42, band0 + 0.98
    ir = img("ir", "val", image_id)[:, :, ::-1]
    gt = split_boxes("val")[image_id]
    tw = 0.86
    th = tw * ir.shape[0] / ir.shape[1]
    tx0, ty0 = 0.10, band0 + 0.27
    ax.imshow(ir, extent=(tx0, tx0 + tw, ty0 + th, ty0), interpolation="bilinear", zorder=2)
    sx, sy = tw / ir.shape[1], th / ir.shape[0]
    for key, col in (("ir_matched", "#2ca02c"), ("ir_only", "#d62728")):
        for b in gt[key]:
            x1, y1, x2, y2 = b[-4:]
            ax.add_patch(Rectangle((tx0 + x1 * sx, ty0 + y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy, fill=False,
                                   ec=col, lw=0.4, zorder=3))
    ax.add_patch(Rectangle((tx0, ty0), tw, th, fill=False, ec=INK, lw=0.5, zorder=4))
    ax.text(tx0 + tw / 2, ty0 + th + 0.11, "Paired IR image", ha="center", va="center", fontsize=8, color=INK)
    ax.text(tx0 + tw / 2, ty0 + th + 0.245, "red: IR-only boxes", ha="center", va="center", fontsize=8,
            color="#d62728")

    box(ax, XB, r1, "Frozen IR teacher", "YOLO26n on IR", "train")
    box(ax, XL, r1, "Teacher features", "L16, L19, L22", "train")
    box(ax, XR, r1, "TGD loss (Eq. 2)", "GT-weighted L2", "train")
    ax_ = 5.45
    box(ax, ax_, r1, "1×1 adapters", r"$A_l(S_l)$", "train", w=0.90)
    box(ax, XB, r2, "Box matching", "Hungarian, IoU ≥ 0.3", "label")
    box(ax, XL, r2, "IR-only boxes", "soft targets ω = 0.3", "label")
    box(ax, 5.70, r2, "Detection loss (Eq. 1)", "RGB boxes + soft targets", "label", w=1.28)
    path(ax, [(tx0 + tw, r1), (XB - BW / 2, r1)], TRAIN)
    path(ax, [(tx0 + tw, r2), (XB - BW / 2, r2)], "#c8553d")
    path(ax, [(XB + BW / 2, r1), (XL - BW / 2, r1)], TRAIN)
    path(ax, [(XL + BW / 2, r1), (XR - BW / 2, r1)], TRAIN)
    path(ax, [(ax_ - 0.45, r1), (XR + BW / 2, r1)], TRAIN)
    path(ax, [(XB + BW / 2, r2), (XL - BW / 2, r2)], "#c8553d")
    path(ax, [(XL + BW / 2, r2), (5.70 - 0.64, r2)], "#c8553d")
    path(ax, [(XR, r2), (XR, r1 + BH / 2)], "#c8553d")                    # IR-only mask -> TGD
    dot(ax, XR, r2, "#c8553d")
    ax.text(XR + 0.05, (r1 + r2) / 2 + 0.02, "IR-only mask", ha="left", va="center", fontsize=8, color="#c8553d")
    # student taps S3, S4, S5 on the neck outputs, down to the adapters
    for xt, r in ((5.02, 2), (5.10, 6), (5.18, 11)):
        dot(ax, xt, y(r), TRAIN)
        path(ax, [(xt, y(r)), (xt, r1 - BH / 2)], TRAIN, ls=(0, (2.5, 1.5)), lw=0.7)
    ax.text(5.24, band0 + 0.13, r"student $S_l$", ha="left", va="center", fontsize=8, color=TRAIN)
    # head outputs to the detection loss
    path(ax, [(6.12, y(11) + BH / 2), (6.12, r2 - BH / 2)], "#c8553d", ls=(0, (2.5, 1.5)), lw=0.7)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig_method_arch.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
