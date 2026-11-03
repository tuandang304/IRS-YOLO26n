"""LaTeX tables and macros for the CMC round-1 revision (R2.2 seeds, R2.4 640 px, R2.5 IR teacher).

  python src/make_rev1_tables.py

Inputs: artifacts/results-v1/metrics.json (seed 0), runs/analysis/rev1/{test,val}_<run>.json (seeds 1-2),
runs/analysis/rev1/A*.json (640 px, teacher), runs/analysis/rev1/boot/*.npz (shared-resample bootstrap,
src/rev1_boot.py), IR-only recall JSONs. Analysis rules: paper/revision_r1/REVIEW_R1.md section 4.
Outputs: paper/generated/rev1_macros.tex, rev1_table_seedcfg.tex, rev1_table_seeds.tex, rev1_table_res.tex,
rev1_table_teacher.tex, and runs/analysis/rev1/rev1_stats.json (all numbers used).
"""
import json, math, statistics, zlib
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "runs/analysis/rev1"
BOOT = AN / "boot"
GEN = ROOT / "paper/generated"
M0 = json.loads((ROOT / "artifacts/results-v1/metrics.json").read_text())
SEEDS = (0, 1, 2)
CFG = {"b1_704": "YOLO26n", "m2_soft_704": "YOLO26n + IR soft labels", "m3_plainkd_704": "YOLO26n + IR feature KD",
       "p_704": "YOLO26n + P (\\ours{})", "yolo11n_704": "YOLO11n", "p11_704": "YOLO11n + P",
       "v10_704": "YOLOv10n", "p10_704": "YOLOv10n + P"}
CONTRASTS = [("p_704", "b1_704", "YOLO26n + P $-$ YOLO26n"), ("m2_soft_704", "b1_704", "IR soft labels $-$ YOLO26n"),
             ("m3_plainkd_704", "b1_704", "IR feature KD $-$ YOLO26n"), ("p_704", "m2_soft_704", "P $-$ IR soft labels"),
             ("p_704", "m3_plainkd_704", "P $-$ IR feature KD"), ("p11_704", "yolo11n_704", "YOLO11n + P $-$ YOLO11n"),
             ("p10_704", "v10_704", "YOLOv10n + P $-$ YOLOv10n")]
FAMILY = {("p_704", "b1_704", "rgb"), ("p_704", "b1_704", "ir"), ("m2_soft_704", "b1_704", "ir"),
          ("p_704", "m2_soft_704", "rgb"), ("p_704", "m2_soft_704", "ir"), ("m3_plainkd_704", "b1_704", "rgb"),
          ("p_704", "m3_plainkd_704", "rgb"), ("p_704", "m3_plainkd_704", "ir")}  # test, night (frozen 2026-11-02)


def f2(x, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "$-$")


def ap(cfg, s, sp, lab, band):
    if s == 0:
        return M0[cfg][sp][lab][band]
    f = AN / f"{sp}_{cfg}_seed{s}.json"
    return json.loads(f.read_text())[lab][band]["ap"] if f.exists() else None


def boot(name, sp, lab, band):
    f = BOOT / f"{name}__{sp}__{lab}__{band}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return float(z["point"]), z["reps"]


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def seed_image_ci(a, b, lab, band, n_seed_draws=1):
    """Crossed bootstrap: replicate r uses image resample r (shared by all models) and a seed draw with
    replacement (matched seeds, so a and b share the drawn seed indices)."""
    va = [boot(f"{a}_seed{s}", "test", lab, band) for s in SEEDS]
    vb = [boot(f"{b}_seed{s}", "test", lab, band) for s in SEEDS]
    if any(x is None for x in va + vb):
        return None
    d = np.stack([va[s][1] - vb[s][1] for s in SEEDS])  # 3 x N
    rng = np.random.default_rng(zlib.crc32(f"seeds/{a}/{b}/{lab}/{band}".encode()))
    pick = rng.integers(0, 3, size=(d.shape[1], 3))
    reps = d[pick, np.arange(d.shape[1])[:, None]].mean(axis=1)
    return ci(reps)


out = {"configs": {}, "contrasts": [], "res640": {}, "teacher": {}}
# ---- per-config mean ± SD ---------------------------------------------------------------------------
for c in CFG:
    for sp in ("test", "val"):
        for lab in ("rgb", "ir"):
            for band in ("all", "night"):
                v = [ap(c, s, sp, lab, band) for s in SEEDS]
                x = [y for y in v if y is not None]
                out["configs"][f"{c}/{sp}/{lab}/{band}"] = {"seeds": v, "mean": statistics.mean(x),
                                                            "sd": statistics.stdev(x) if len(x) > 1 else None}
# ---- seed-paired contrasts --------------------------------------------------------------------------
for a, b, _ in CONTRASTS:
    for sp in ("test", "val"):
        for lab in ("rgb", "ir"):
            for band in ("night", "all"):
                d = [ap(a, s, sp, lab, band) - ap(b, s, sp, lab, band) for s in SEEDS]
                m, sd = statistics.mean(d), statistics.stdev(d)
                se = sd / math.sqrt(3)
                tc = stats.t.ppf(0.975, 2)
                p = float(2 * stats.t.sf(abs(m / se), 2)) if se > 0 else 0.0
                allpos = all(z > 0 for z in d)
                lo = m - tc * se
                rule = 1 if allpos and lo > 0 else (2 if allpos else 3)
                r = {"a": a, "b": b, "split": sp, "labels": lab, "band": band, "per_seed": d, "mean": m, "sd": sd,
                     "t_ci": [lo, m + tc * se], "p": p, "rule": rule,
                     "confirmatory": sp == "test" and band == "night" and (a, b, lab) in FAMILY}
                if sp == "test":
                    r["si_ci"] = seed_image_ci(a, b, lab, band)
                out["contrasts"].append(r)
fam = sorted([r for r in out["contrasts"] if r["confirmatory"]], key=lambda r: r["p"])
run = 0.0
for i, r in enumerate(fam):
    run = max(run, min(1.0, (len(fam) - i) * r["p"]))
    r["p_holm"] = run
# ---- 640 vs 704 (seed 0) ----------------------------------------------------------------------------
for lab in ("rgb", "ir"):
    for band in ("all", "night"):
        B = {k: boot(k, "test", lab, band) for k in ("b1_640", "p_640", "b1_704_seed0", "p_704_seed0")}
        if any(v is None for v in B.values()):
            continue
        d640 = B["p_640"][1] - B["b1_640"][1]
        d704 = B["p_704_seed0"][1] - B["b1_704_seed0"][1]
        out["res640"][f"{lab}/{band}"] = {
            "b1_640": B["b1_640"][0], "p_640": B["p_640"][0], "b1_704": B["b1_704_seed0"][0], "p_704": B["p_704_seed0"][0],
            "d640": B["p_640"][0] - B["b1_640"][0], "d640_ci": ci(d640),
            "d704": B["p_704_seed0"][0] - B["b1_704_seed0"][0], "d704_ci": ci(d704),
            "inter": (B["p_704_seed0"][0] - B["b1_704_seed0"][0]) - (B["p_640"][0] - B["b1_640"][0]),
            "inter_ci": ci(d704 - d640),
            "b1_res": B["b1_704_seed0"][0] - B["b1_640"][0], "b1_res_ci": ci(B["b1_704_seed0"][1] - B["b1_640"][1]),
            "p_res": B["p_704_seed0"][0] - B["p_640"][0], "p_res_ci": ci(B["p_704_seed0"][1] - B["p_640"][1])}
# ---- IR teacher --------------------------------------------------------------------------------------
for res in ("640", "704"):
    for sp in ("val", "test"):
        for band in ("all", "night"):
            t = boot(f"tir{res}_{sp}", sp, "ir", band)
            if t:
                out["teacher"][f"{res}/{sp}/{band}"] = {"ap": t[0], "ci": ci(t[1])}
        rgb = json.loads((AN / {"640": {"val": "A2_tir640_val", "test": "A3_tir640_test"},
                                "704": {"val": "A4_tir704_val", "test": "A4_tir704_test"}}[res][sp]
                          ).with_suffix(".json").read_text())["rgb"]
        out["teacher"][f"{res}/{sp}/rgblabels_night"] = rgb["night"]["ap"]
for sp in ("val", "test"):
    for band in ("all", "night"):
        a, b = boot(f"tir704_{sp}", sp, "ir", band), boot(f"tir640_{sp}", sp, "ir", band)
        if a and b:
            out["teacher"][f"d704/{sp}/{band}"] = {"d": a[0] - b[0], "ci": ci(a[1] - b[1])}
# ---- IR-only recall over seeds ----------------------------------------------------------------------
hr0 = {}
for f in ("hidden_recall_test.json", "hidden_recall_r5.json", "hidden_recall_p11.json"):
    p = ROOT / "artifacts/results-v1" / f
    if p.exists():
        d = json.loads(p.read_text())
        hr0.update(d.get("test", d))
hrs = json.loads((AN / "hidden_recall_seeds.json").read_text()) if (AN / "hidden_recall_seeds.json").exists() else {}
KEY0 = {"b1_704": "B1_704", "p_704": "P_704", "m2_soft_704": "M2_704", "m3_plainkd_704": "M3_704",
        "yolo11n_704": "YOLO11n_704", "p11_704": "P11_704", "v10_704": "V10_704", "p10_704": "P10_704"}
out["recall"] = {}
for c, k in KEY0.items():
    v = [(hr0.get(k) if s == 0 else hrs.get(f"{c}_seed{s}")) for s in SEEDS]
    v = [x["night@0.25"]["recall_ir_only"] if x else None for x in v]
    out["recall"][c] = v
(AN / "rev1_stats.json").write_text(json.dumps(out, indent=1))

# ---- LaTeX --------------------------------------------------------------------------------------------
mac = [r"\makeatletter", r"\newcommand{\rv}[1]{\@nameuse{rv@#1}}"]
def put(k, v):
    mac.append(r"\@namedef{rv@" + k + "}{" + v + "}")
for k, v in out["configs"].items():
    put(f"cfg/{k}/mean", f2(v["mean"], False)); put(f"cfg/{k}/sd", f2(v["sd"], False))
for r in out["contrasts"]:
    k = f"{r['a']}-{r['b']}/{r['split']}/{r['labels']}/{r['band']}"
    put(k + "/mean", f2(r["mean"])); put(k + "/sd", f2(r["sd"], False))
    put(k + "/seeds", ", ".join(f2(x) for x in r["per_seed"]))
    put(k + "/tci", f"[{f2(r['t_ci'][0])}, {f2(r['t_ci'][1])}]")
    if r.get("si_ci"):
        put(k + "/sici", f"[{f2(r['si_ci'][0])}, {f2(r['si_ci'][1])}]")
    if "p_holm" in r:
        put(k + "/holm", f"{r['p_holm']:.3f}")
    put(k + "/p", f"{r['p']:.3f}")
for k, v in out["res640"].items():
    for q in ("d640", "d704", "inter", "b1_res", "p_res"):
        put(f"res/{k}/{q}", f2(v[q])); put(f"res/{k}/{q}ci", f"[{f2(v[q + '_ci'][0])}, {f2(v[q + '_ci'][1])}]")
    for q in ("b1_640", "p_640", "b1_704", "p_704"):
        put(f"res/{k}/{q}", f2(v[q], False))
for k, v in out["teacher"].items():
    if isinstance(v, dict) and "ap" in v:
        put(f"tir/{k}", f2(v["ap"], False)); put(f"tir/{k}/ci", f"[{f2(v['ci'][0], False)}, {f2(v['ci'][1], False)}]")
    elif isinstance(v, dict):
        put(f"tir/{k}", f2(v["d"])); put(f"tir/{k}/ci", f"[{f2(v['ci'][0])}, {f2(v['ci'][1])}]")
    else:
        put(f"tir/{k}", f2(v, False))
for c, v in out["recall"].items():
    xs = [x for x in v if x is not None]
    put(f"hr/{c}/range", f"{min(xs):.1f}--{max(xs):.1f}" if xs else "--")
    put(f"hr/{c}/mean", f"{statistics.mean(xs):.1f}" if xs else "--")
mac.append(r"\makeatother")
(GEN / "rev1_macros.tex").write_text("\n".join(mac) + "\n")

rows = []
MAIN = [("b1_704", "YOLO26n", "2.51", "5.84"), ("p_704", "YOLO26n + P (\\ours{})", "2.51", "5.84"),
        ("yolo11n_704", "YOLO11n", "2.59", "6.44"), ("p11_704", "YOLO11n + P", "2.59", "6.44"),
        ("v10_704", "YOLOv10n", "2.71", "8.38"), ("p10_704", "YOLOv10n + P", "2.71", "8.38")]
for i, (c, name, params, gf) in enumerate(MAIN):
    if i in (2, 4):
        rows.append(r"\midrule")
    cells = []
    for lab in ("rgb", "ir"):
        for band in ("all", "night"):
            v = out["configs"][f"{c}/test/{lab}/{band}"]
            cells.append(f"{v['mean']:.2f}~$\\pm$~{v['sd']:.2f}")
    rows.append(f"{name} & {params} & {gf} & " + " & ".join(cells) + r" \\")
(GEN / "rev1_table_main.tex").write_text("\n".join(rows) + "\n")

BANDS = [("rgb", "all"), ("rgb", "night"), ("ir", "all"), ("ir", "night")]
def msd(c, lab, band):
    v = out["configs"][f"{c}/test/{lab}/{band}"]
    return f"{v['mean']:.2f}~$\\pm$~{v['sd']:.2f}"  # unbreakable, so cells never wrap mid-value
def s0(c, lab, band):
    return f"{M0[c]['test'][lab][band]:.2f}"
# Multi-seed rows report mean +/- SD; rows marked with a dagger in the method cell are single runs (seed 0).
METHODS = [("None (baseline)", "--", "--", [msd("b1_704", l, b) for l, b in BANDS]),
           ("Zero-DCE++ enhancement~\\cite{li2021zerodcepp}$^{\\ddagger}$", "--", "enhancer",
            [s0("b2", l, b) for l, b in BANDS]),
           ("Union of RGB and IR labels ($\\omega=1$)$^{\\ddagger}$", "labels", "--",
            [s0("m1_union_704", l, b) for l, b in BANDS]),
           ("IR soft labels ($\\omega=0.3$), as in UA-CMDet~\\cite{sun2022dronevehicle}", "labels", "--",
            [msd("m2_soft_704", l, b) for l, b in BANDS]),
           ("Feature KD from the IR teacher (uniform L2)", "features", "--",
            [msd("m3_plainkd_704", l, b) for l, b in BANDS]),
           ("\\textbf{P (ours)}: IR soft labels + TGD", "labels + features", "--",
            [msd("p_704", l, b) for l, b in BANDS])]
# Bold the best value per column across all rows (single-seed rows compete on the same footing;
# the table caption says so).
def cell_num(cell):
    return float(cell.replace("~", " ").split(" ")[0].replace("$", ""))
colmax = [max(cell_num(cells[j]) for _, _, _, cells in METHODS) for j in range(4)]
def bold_if_max(cell, j):
    return f"\\textbf{{{cell}}}" if abs(cell_num(cell) - colmax[j]) < 1e-9 else cell
rows = []
for name, i_tr, at_inf, cells in METHODS:
    cells = [bold_if_max(c, j) for j, c in enumerate(cells)]
    rows.append(f"{name} & {i_tr} & {at_inf} & " + " & ".join(cells) + r" \\")
(GEN / "rev1_table_methods.tex").write_text("\n".join(rows) + "\n")

# Seed-0 paired-bootstrap rows for the two single-seed alternatives, carried over unchanged from
# the audited table_boot.tex (computed by src/make_tables_extra.py from the paired bootstrap).
def two(d, ci, val):
    """Stack 'difference (validation)' over '[test CI]' inside one cell so the four columns fit the text width."""
    return "\\begin{tabular}[t]{@{}c@{}}" + d + " (" + val + ")\\\\{}" + ci + "\\end{tabular}"
boot2 = [
    "Zero-DCE++ enhancement & " + " & ".join([
        two("+1.33$^{\\dagger}$", "[+0.81, +1.77]", "+0.56"), two("+1.12$^{\\dagger}$", "[+0.74, +1.43]", "+0.63"),
        two("+2.63$^{\\dagger}$", "[+2.21, +3.03]", "+2.03"), two("+2.04$^{\\dagger}$", "[+1.72, +2.36]", "+1.25")]) + " \\\\",
    "Union of RGB and IR labels & " + " & ".join([
        two("+0.59$^{\\dagger}$", "[+0.11, +1.03]", "+0.68"), two("+0.67$^{\\dagger}$", "[+0.29, +1.01]", "+1.12"),
        two("$-$1.01$^{\\dagger}$", "[$-$1.36, $-$0.58]", "$-$0.83"), two("$-$0.21", "[$-$0.48, +0.12]", "+0.00")]) + " \\\\",
]
(GEN / "rev1_table_boot2.tex").write_text("\n".join(boot2) + "\n")

rows = []
for j, (a, b, name) in enumerate(CONTRASTS):
    if j in (3, 5):
        rows.append(r"\midrule")
    for lab in ("rgb", "ir"):
        r = next(x for x in out["contrasts"] if x["a"] == a and x["b"] == b and x["split"] == "test"
                 and x["labels"] == lab and x["band"] == "night")
        si = f"[{f2(r['si_ci'][0])}, {f2(r['si_ci'][1])}]" if r.get("si_ci") else "--"
        holm = f"{r['p_holm']:.3f}" if "p_holm" in r else "--"
        mark = {1: "", 2: "$^{\\ast}$", 3: "$^{\\circ}$"}[r["rule"]]
        rows.append(f"{name if lab == 'rgb' else ''} & {lab.upper()} & {', '.join(f2(x) for x in r['per_seed'])} & "
                    f"{f2(r['mean'])}{mark} $\\pm$ {r['sd']:.2f} & [{f2(r['t_ci'][0])}, {f2(r['t_ci'][1])}] & {si} & {holm} \\\\")
(GEN / "rev1_table_seeds.tex").write_text("\n".join(rows) + "\n")

rows = []
R = out["res640"]
def _res(key):
    return R.get(key)
if R:
    def line(label, key, fmt=lambda v: f2(v, False)):
        return label + " & " + " & ".join(
            fmt(_res(f"{l}/{b}")[key]) if _res(f"{l}/{b}") else "--" for l in ("rgb", "ir") for b in ("all", "night")) + r" \\"
    def dline(label, key):
        return label + " & " + " & ".join(
            (f"{f2(v[key])} [{f2(v[key + '_ci'][0])}, {f2(v[key + '_ci'][1])}]" if (v := _res(f"{l}/{b}")) else "--")
            for l in ("rgb", "ir") for b in ("all", "night")) + r" \\"
    rows += [line("YOLO26n, 640", "b1_640"), line("YOLO26n + P, 640", "p_640"), dline("$\\Delta_{640}$", "d640"),
             r"\midrule", line("YOLO26n, 704", "b1_704"), line("YOLO26n + P, 704", "p_704"), dline("$\\Delta_{704}$", "d704"),
             r"\midrule", dline("$\\Delta_{704}-\\Delta_{640}$", "inter")]
else:
    rows = [r"\multicolumn{5}{l}{(bootstrap pending)} \\"]
(GEN / "rev1_table_res.tex").write_text("\n".join(rows) + "\n")

rows = []
T = out["teacher"]
for res in ("640", "704"):
    cells = []
    for sp in ("val", "test"):
        for band in ("all", "night"):
            v = T.get(f"{res}/{sp}/{band}")
            cells.append(f"{v['ap']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" if v else "--")
    rows.append(f"{res} & " + " & ".join(cells) + r" \\")
(GEN / "rev1_table_teacher.tex").write_text("\n".join(rows) + "\n")
print("wrote", [p.name for p in GEN.glob("rev1_*.tex")])
for f in ("rev1_table_main", "rev1_table_methods", "rev1_table_boot2"):
    print("---", f); print((GEN / (f + ".tex")).read_text())
for r in out["contrasts"]:
    if r["split"] == "test" and r["band"] == "night":
        print(r["a"], r["b"], r["labels"], [round(x, 2) for x in r["per_seed"]], round(r["mean"], 2),
              [round(x, 2) for x in r["t_ci"]], r.get("si_ci") and [round(x, 2) for x in r["si_ci"]],
              round(r.get("p_holm", float("nan")), 3), r["rule"])
