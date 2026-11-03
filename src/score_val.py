"""Score proxy runs on val subsets (development only; never reported as findings).

Usage: score_val.py RUN [RUN ...]   (RUN = runs/remote/<name>)
Prints AP50:95 on val all / night / twilight / day, plus AP50, from preds_val.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

ROOT = HERE.parent
GT = ROOT / "data/dronevehicle/proc/rgb/val/boxes.json"
MAN = HERE.parent / "splits"


def score(run):
    run = Path(run)
    name = run.name
    preds = run / "ws" / "out" / name / f"preds_val.json"
    m = ev.Matched(GT, json.load(open(preds)))
    out = {"run": name, "all": m.ap(np.ones(len(m.ids)))}
    for b in ("night", "twilight", "day"):
        w = ev.subset_weights(m.ids, (MAN / f"val_{b}.txt").read_text().split())
        out[b] = m.ap(w)
        out[f"{b}_ap50"] = m.ap(w, 0.5)
    return out, m


if __name__ == "__main__":
    rows = []
    for r in sys.argv[1:]:
        o, _ = score(r)
        rows.append(o)
        print(json.dumps({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in o.items()}))
