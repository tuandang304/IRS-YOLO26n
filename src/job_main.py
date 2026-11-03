"""Job driver for one training, prediction or latency job (cwd = job workspace; see scripts/run_queue.py).

job.json fields:
  name         run name; outputs go to out/<name>/ (kept across chunks)
  kind         "train" | "predict" | "enhance" | "latency" | "probe"
  splits       dataset splits that must be prepared on this VM
  train        Ultralytics overrides for kind=train (data key: "rgb", "ir", "rgb_proxy"...)
  config       lowlight.CONFIG values (stop_after_epoch is the chunk end)
  predict      {"weights": path, "split": "val"|"test", "data_root": "rgb"|"rgb_zdce"}
  proxy_ids    optional list of train image ids for proxy runs

Data are prepared once per VM under /content/data/dv from the pinned Hugging
Face mirror, with the same dv_prep/dv_pairs code used locally.
"""

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path.cwd()
SRC = WS / "src"
sys.path.insert(0, str(SRC))
DATA = Path(os.environ.get("DV_DATA", "/content/data/dv"))
ULTRA = "8.4.173"
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_env():
    try:
        import ultralytics  # noqa: F401
        ok = ultralytics.__version__ == ULTRA
    except ImportError:
        ok = False
    if not ok:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRA}",
                        "pycocotools", "albumentations"], check=True)
    import torch
    log("torch", torch.__version__, "cuda", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
        "cpus", os.cpu_count(), "ram_gb", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))


def _hash_labels(split):
    h = hashlib.sha256()
    for mod in ("rgb", "ir"):
        d = DATA / "proc" / mod / split / "labels"
        for f in sorted(d.iterdir()):
            h.update(f.name.encode() + f.read_bytes())
    for f in sorted((DATA / "proc" / "ironly" / split).iterdir()):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def ensure_data(splits):
    import dv_pairs
    import dv_prep
    from huggingface_hub import hf_hub_download

    proc = DATA / "proc"
    out = {}
    for s in splits:
        ready = proc / f".ready_{s}"
        if not ready.exists():
            t = time.time()
            z = hf_hub_download(dv_prep.HF_REPO, f"{s}.zip", repo_type="dataset",
                                revision=dv_prep.HF_REVISION, local_dir=str(DATA / "raw"))
            log("downloaded", s, round(time.time() - t), "s")
            rep = dv_prep.prepare(z, s, proc, workers=os.cpu_count())
            dv_pairs.run(proc, s)
            (proc / "stats" / f"prep_{s}.json").write_text(json.dumps(rep))
            os.remove(z)
            ready.write_text(_hash_labels(s))
            log("prepared", s, round(time.time() - t), "s")
        if not (proc / "ironly_cls" / s).exists():  # added after the first preparation
            dv_pairs.run(proc, s)
        out[s] = ready.read_text()
    for mod in ("rgb", "ir"):
        y = DATA / f"{mod}.yaml"
        y.write_text(f"path: {proc}\ntrain: {mod}/train/images\nval: {mod}/val/images\n"
                     f"test: {mod}/test/images\nnames: {json.dumps(NAMES)}\n")
    return out


def ensure_zdce(splits):
    """Zero-DCE++-enhanced RGB copy of each split (labels are links to the originals)."""
    import zdce

    for s in splits:
        ready = DATA / "zdce" / f".ready_{s}"
        if ready.exists():
            continue
        t = time.time()
        n = zdce.enhance_dir(DATA / "proc" / "rgb" / s / "images", DATA / "zdce" / "rgb" / s / "images",
                             WS / "inputs" / "zdcepp_Epoch99.pth")
        lab = DATA / "zdce" / "rgb" / s / "labels"
        if not lab.exists():
            lab.symlink_to(DATA / "proc" / "rgb" / s / "labels")
        ready.write_text(str(n))
        log("zero-dce++", s, n, "images", round(time.time() - t), "s")


def _data_yaml(key, job):
    """rgb | ir | rgb_proxy | ir_proxy | rgb_zdce."""
    proc = DATA / "proc"
    if key in ("rgb", "ir"):
        return str(DATA / f"{key}.yaml")
    if key.endswith("_proxy") and not key.startswith("rgb4"):
        mod = key[:-6]
        ids = job.get("proxy_ids")
        if ids is None:  # seeded random subset of the sorted train ids
            import numpy as np

            allids = sorted(p.stem for p in (proc / "rgb" / "train" / "images").glob("*.jpg"))
            rng = np.random.default_rng(job.get("proxy_seed", 0))
            n = round(len(allids) * job["proxy_fraction"])
            ids = sorted(rng.choice(allids, n, replace=False).tolist())
            (WS / "out" / job["name"]).mkdir(parents=True, exist_ok=True)
            (WS / "out" / job["name"] / "proxy_ids.json").write_text(json.dumps(ids))
        rep = job.get("night_repeat")  # {"ids_file": "inputs/...txt", "k": 2}: oversample night
        if rep:
            night = set((WS / rep["ids_file"]).read_text().split())
            ids = ids + [i for i in ids if i in night] * (rep["k"] - 1)
        lst = proc / f"{mod}_train_{job['name']}.txt"
        lst.write_text("".join(f"{proc}/{mod}/train/images/{i}.jpg\n" for i in ids))
        y = DATA / f"{key}_{job['name']}.yaml"
        y.write_text(f"path: {proc}\ntrain: {lst}\nval: {mod}/val/images\nnames: {json.dumps(NAMES)}\n")
        return str(y)
    if key in ("rgb4", "rgb4_proxy"):  # 4-channel R,G,B,IR fusion data (IR found by path)
        train = "rgb/train/images"
        if key == "rgb4_proxy":
            ids = job.get("proxy_ids")
            if ids is None:
                import numpy as np

                allids = sorted(p.stem for p in (proc / "rgb" / "train" / "images").glob("*.jpg"))
                rng = np.random.default_rng(job.get("proxy_seed", 0))
                ids = sorted(rng.choice(allids, round(len(allids) * job["proxy_fraction"]),
                                        replace=False).tolist())
            lst = proc / f"rgb4_train_{job['name']}.txt"
            lst.write_text("".join(f"{proc}/rgb/train/images/{i}.jpg\n" for i in ids))
            train = str(lst)
        y = DATA / f"{key}_{job['name']}.yaml"
        y.write_text(f"path: {proc}\ntrain: {train}\nval: rgb/val/images\nchannels: 4\n"
                     f"names: {json.dumps(NAMES)}\n")
        return str(y)
    if key == "rgb_zdce":
        y = DATA / "rgb_zdce.yaml"
        y.write_text(f"path: {DATA / 'zdce'}\ntrain: rgb/train/images\nval: rgb/val/images\n"
                     f"test: rgb/test/images\nnames: {json.dumps(NAMES)}\n")
        return str(y)
    raise ValueError(key)


def train(job):
    import lowlight as ll

    ll.CONFIG.update(job.get("config", {}))
    if ll.CONFIG.get("ir_only_dir") == "auto":
        ll.CONFIG["ir_only_dir"] = str(DATA / "proc" / "ironly" / "train")
    if ll.CONFIG.get("pseudo_file"):
        ll.CONFIG["pseudo_file"] = str(WS / ll.CONFIG["pseudo_file"])
    if ll.CONFIG.get("ir_soft_dir") == "auto":
        ll.CONFIG["ir_soft_dir"] = str(DATA / "proc" / "ironly_cls" / "train")
    over = dict(job["train"])
    over["data"] = _data_yaml(over["data"], job)
    out = WS / "out"
    last = out / job["name"] / "weights" / "last.pt"
    if last.exists():
        keep = {k: over[k] for k in ("data", "workers", "cache", "batch", "distill_model") if k in over}
        # save_dir must be overridden: the checkpoint stores the previous job's workspace path
        over = dict(resume=str(last), save_dir=str(out / job["name"]), **keep)
        log("resuming", last)
    else:
        over.update(project=str(out), name=job["name"], exist_ok=True)
    logf = out / job["name"] / "train.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logf, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        tr = ll.LLTrainer(overrides=over)
        tr.train()
    secs = time.time() - t0
    if Path(tr.save_dir).resolve() != (out / job["name"]).resolve():
        raise RuntimeError(f"trainer wrote to {tr.save_dir}, not {out / job['name']}")
    done = not tr._chunk_stopped
    log("trained", job["name"], "epochs_done", tr.epoch + 1, "secs", round(secs), "finished", done)
    res = out / job["name"] / "results.csv"
    if res.exists():
        print(res.read_text().splitlines()[-1][:300], flush=True)
    return done


def predict(job):
    from ultralytics import YOLO

    import lowlight as ll

    p = job["predict"]
    w = WS / p["weights"]
    student = w.with_name(w.stem + "_student.pt")
    ll.export_student(w, student)
    roots = {"rgb": DATA / "proc" / "rgb", "ir": DATA / "proc" / "ir", "rgb_zdce": DATA / "zdce" / "rgb"}
    root = roots[p.get("data_root", "rgb")] / p["split"] / "images"
    files = sorted(root.glob("*.jpg"))
    m = YOLO(str(student))
    preds = []
    t = time.time()
    # a list source is one batch in Ultralytics (batch= is ignored), so feed 32 images at a time
    results = (r for i in range(0, len(files), 32)
               for r in m.predict(source=[str(f) for f in files[i:i + 32]], imgsz=p.get("imgsz", 640),
                                  conf=0.001, max_det=300, half=True, stream=True, verbose=False))
    for r in results:
        b = r.boxes
        for xyxy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
            preds.append({"image_id": Path(r.path).stem, "category_id": int(c),
                          "bbox": [round(v, 2) for v in xyxy], "score": round(s, 5)})
    tag = "" if p.get("data_root", "rgb") == "rgb" else "_" + p["data_root"]
    dst = WS / "out" / job["name"] / f"preds_{p['split']}{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(preds))
    log("predicted", len(files), "images", len(preds), "boxes", round(time.time() - t), "s ->", dst.name)


def latency(job):
    """Batch-1 fp16 latency on the job's GPU: median Ultralytics inference/total ms over 300 val
    images after 50 warmup images, for each {"name", "weights", "imgsz"} in job["latency"]."""
    import numpy as np
    from ultralytics import YOLO

    import lowlight as ll

    files = sorted((DATA / "proc" / "rgb" / "val" / "images").glob("*.jpg"))[:350]
    res = {}
    for e in job["latency"]:
        w = WS / e["weights"]
        student = w.with_name(w.stem + "_student.pt")
        ll.export_student(w, student)
        m = YOLO(str(student))
        inf, tot = [], []
        for i, f in enumerate(files):
            r = m.predict(source=str(f), imgsz=e["imgsz"], conf=0.001, max_det=300, half=True, verbose=False)[0]
            if i >= 50:
                inf.append(r.speed["inference"])
                tot.append(sum(r.speed.values()))
        res[e["name"]] = {"imgsz": e["imgsz"], "inference_ms": round(float(np.median(inf)), 3),
                          "total_ms": round(float(np.median(tot)), 3)}
        log("latency", e["name"], json.dumps(res[e["name"]]))
    dst = WS / "out" / job["name"] / "latency.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))


def main(path="job.json"):
    job = json.loads((WS / path).read_text())
    log("job", job["name"], job["kind"])
    ensure_env()
    hashes = ensure_data(job.get("splits", ["train", "val"]))
    log("label hashes", json.dumps(hashes))
    if job.get("zdce_splits"):
        ensure_zdce(job["zdce_splits"])
    if job["kind"] == "train":
        if train(job) and job.get("predict"):
            import gc

            import torch

            gc.collect()  # the finished trainer still holds GPU memory (OOM at imgsz 768)
            torch.cuda.empty_cache()
            predict(job)
    elif job["kind"] == "dfine":
        import dfine_job

        dfine_job.run(job, WS, DATA)
    elif job["kind"] == "latency":
        latency(job)
    elif job["kind"] == "dfine_predict":
        import dfine_job

        dfine_job.predict_job(job, WS, DATA)
    elif job["kind"] == "dfine_finish":
        import dfine_job

        dfine_job.finish(job, WS, DATA)
    elif job["kind"] == "retrieve":
        src = Path(job["retrieve_from"])
        if "*" in str(src):  # e.g. a chunk whose controller was stopped: newest matching run dir
            import glob

            hits = [Path(h) for h in glob.glob(str(src)) if (Path(h) / "weights" / "last.pt").exists()]
            src = max(hits, key=lambda h: (h / "weights" / "last.pt").stat().st_mtime)
        dst = WS / "out" / job["name"]
        for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "train.log"):
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, dst / f)
        log("retrieved", src, "->", dst)
        if job.get("predict"):
            predict(job)
    elif job["kind"] == "predict":
        predict(job)
    elif job["kind"] == "probe":
        subprocess.run("nvidia-smi; df -h /content; free -g; nproc", shell=True)
    shutil.rmtree(WS / "inputs", ignore_errors=True)
    log("done")
