"""Shared-resample bootstrap AP vectors for the CMC revision (R2.2 seed x image, R2.4 640/704, R2.5 teacher).

  python src/rev1_boot.py NAME [...]      (names from MODELS; one model in memory at a time)

For every (split, labels, band) the image-resampling weights come from one fixed RNG seed, so the b-th
replicate uses the same resampled images for every model; contrasts, interactions and seed x image
intervals are then formed from the saved vectors (src/rev1_boot_stats.py). Writes
runs/analysis/rev1/boot/<name>__<split>__<labels>__<band>.npz (point AP and N replicate APs, in %).
"""
import json, sys, zlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402
D = ROOT / "data/dronevehicle/proc"
MAN = ROOT / "splits"
OUT = ROOT / "runs/analysis/rev1/boot"
N = 1000
R = ROOT / "runs/remote"
def t(n): return R / f"test_{n}/ws/out/test_{n}/preds_test.json"
MODELS = {"b1_640": (t("b1"), "test", ("rgb", "ir")),
          "p_640": (ROOT / "runs/local/local_p_640_test/ws/out/local_p_640_test/preds_test.json", "test", ("rgb", "ir")),
          "tir640_val": (ROOT / "runs/local/local_tir640_val/ws/out/local_tir640_val/preds_val_ir.json", "val", ("ir",)),
          "tir640_test": (ROOT / "runs/local/local_tir640_test/ws/out/local_tir640_test/preds_test_ir.json", "test", ("ir",)),
          "tir704_val": (ROOT / "runs/local/local_tir704_val/ws/out/local_tir704_val/preds_val_ir.json", "val", ("ir",)),
          "tir704_test": (ROOT / "runs/local/local_tir704_test/ws/out/local_tir704_test/preds_test_ir.json", "test", ("ir",))}
for c in ("b1_704", "p_704", "m2_soft_704", "m3_plainkd_704", "yolo11n_704", "p11_704", "v10_704", "p10_704"):
    MODELS[f"{c}_seed0"] = (t(c), "test", ("rgb", "ir"))
    for s in (1, 2):
        MODELS[f"{c}_seed{s}"] = (t(f"{c}_seed{s}"), "test", ("rgb", "ir"))
def weights_seq(ids, split, band, labels):
    base = np.ones(len(ids)) if band == "all" else ev.subset_weights(ids, (MAN / f"{split}_{band}.txt").read_text().split())
    idx = np.flatnonzero(base > 0)
    rng = np.random.default_rng(zlib.crc32(f"{split}/{labels}/{band}".encode()))
    for _ in range(N):
        w = np.zeros(len(ids))
        np.add.at(w, idx[rng.integers(0, len(idx), len(idx))], 1)
        yield w
    return base
for name in sys.argv[1:]:
    path, split, labs = MODELS[name]
    for lab in labs:
        if all((OUT / f"{name}__{split}__{lab}__{b}.npz").exists() for b in ("all", "night")):
            continue
        m = ev.Matched(D / f"{lab}/{split}/boxes.json", json.load(open(path)))
        for band in ("all", "night"):
            f = OUT / f"{name}__{split}__{lab}__{band}.npz"
            if f.exists():
                continue
            base = np.ones(len(m.ids)) if band == "all" else ev.subset_weights(m.ids, (MAN / f"{split}_{band}.txt").read_text().split())
            reps = np.array([m.ap(w) for w in weights_seq(m.ids, split, band, lab)]) * 100
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez(f, point=m.ap(base) * 100, reps=reps, ids_hash=zlib.crc32("".join(map(str, m.ids)).encode()))
            print(name, split, lab, band, round(float(m.ap(base) * 100), 2), flush=True)
        del m
