"""Score proxy runs on val against RGB labels (primary) and IR labels (secondary)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

D = HERE.parent / "data/dronevehicle/proc"
M = HERE.parent / "splits"
for run in sys.argv[1:]:
    preds = json.load(open(HERE.parent / f"runs/remote/{run}/ws/out/{run}/preds_val.json"))
    out = {}
    for lab in ("rgb", "ir"):
        m = ev.Matched(D / f"{lab}/val/boxes.json", preds)
        out[lab] = [round(m.ap(np.ones(len(m.ids))) * 100, 2)] + [
            round(m.ap(ev.subset_weights(m.ids, (M / f"val_{b}.txt").read_text().split())) * 100, 2)
            for b in ("night", "day")]
    print(run, "RGB all/night/day", out["rgb"], "| IR", out["ir"], flush=True)
