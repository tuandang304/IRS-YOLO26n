"""D-FINE-N baseline (non-YOLO) on DroneVehicle RGB, run inside a harness job.

Official code: https://github.com/Peterande/D-FINE (Apache-2.0), pinned commit below,
fine-tuned from the COCO checkpoint dfine_n_coco.pth. Our single-GPU adaptation (not an
official recipe): 60 epochs (augmentation and multi-scale stop at epoch 48), total batch 32,
AMP + EMA as in the repo, learning rates scaled from the batch-128 recipe by sqrt(32/128)
(lr 4e-4, backbone 2e-4). Images are stretch-resized to 640x640 for training and evaluation,
as in the repo; predictions are rescaled to the 640x512 image and scored with the study's
COCO evaluator (maxDets 300) like every other model.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/Peterande/D-FINE.git"
COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
CLASSES = ["car", "truck", "bus", "van", "freight_car"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_repo(root):
    root = Path(root)
    code = root / "D-FINE"
    if not (code / ".ready").exists():
        subprocess.run(["git", "clone", "-q", REPO, str(code)], check=True)
        subprocess.run(["git", "-C", str(code), "checkout", "-q", COMMIT], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-coco-eval>=1.6.6", "calflops",
                        "loguru"], check=True)
        subprocess.run(["wget", "-q", "-O", str(root / "dfine_n_coco.pth"), WEIGHTS], check=True)
        (code / ".ready").write_text(COMMIT)
    return code


def make_coco(proc, split, out):
    """COCO json for one split from the study's boxes.json (category ids 0..4)."""
    g = json.load(open(Path(proc) / "rgb" / split / "boxes.json"))
    images, anns = [], []
    for n, k in enumerate(sorted(g["images"])):
        images.append({"id": n, "file_name": f"{k}.jpg", "width": g["width"], "height": g["height"]})
        for c, x1, y1, x2, y2 in g["images"][k]:
            anns.append({"id": len(anns) + 1, "image_id": n, "category_id": int(c),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    cats = [{"id": i, "name": c} for i, c in enumerate(CLASSES)]
    Path(out).write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return out


def write_config(code, proc, work, epochs=60, stop=48, batch=32, workers=10, size=640):
    import yaml

    proc, work = Path(proc), Path(work)
    cfg = work / "dfine_n_dronevehicle.yml"
    base = code / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml"
    c = {
        "__include__": [str(base)],
        "output_dir": str(work / "output"),
        "num_classes": 5,
        "remap_mscoco_category": False,
        "checkpoint_freq": 1000,
        "epochs": epochs,
        "optimizer": {"type": "AdamW", "params": [
            {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.0002},
            {"params": "^(?=.*backbone)(?=.*norm|bn).*$", "lr": 0.0002, "weight_decay": 0.0},
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0}],
            "lr": 0.0004, "betas": [0.9, 0.999], "weight_decay": 0.0001},
        "train_dataloader": {
            "total_batch_size": batch, "num_workers": workers,
            "dataset": {"img_folder": str(proc / "rgb" / "train" / "images"), "ann_file": str(work / "train.json"),
                        "transforms": {"policy": {"epoch": stop}}},
            "collate_fn": {"stop_epoch": stop, "ema_restart_decay": 0.9999, "base_size_repeat": None}},
        "val_dataloader": {
            "total_batch_size": 64, "num_workers": workers,
            "dataset": {"img_folder": str(proc / "rgb" / "val" / "images"), "ann_file": str(work / "val.json")}},
    }
    if size != 640:  # same pipeline as configs/dfine/include/dataloader.yml at another square size
        c["eval_spatial_size"] = [size, size]
        c["train_dataloader"]["dataset"]["transforms"]["ops"] = [
            {"type": "RandomPhotometricDistort", "p": 0.5}, {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8}, {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"}, {"type": "Resize", "size": [size, size]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True}]
        c["train_dataloader"]["collate_fn"]["base_size"] = size
        c["val_dataloader"]["dataset"]["transforms"] = {"ops": [
            {"type": "Resize", "size": [size, size]}, {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]}
    cfg.write_text(yaml.safe_dump(c, sort_keys=False))
    return cfg


def train(code, cfg, weights, work, epochs=None):
    cmd = [sys.executable, "train.py", "-c", str(cfg), "--use-amp", "--seed", "0", "-t", str(weights)]
    if epochs is not None:
        cmd += ["-u", f"epochs={epochs}"]
    log("train", " ".join(cmd[1:]))
    with open(Path(work) / "train.log", "a") as f:
        subprocess.run(cmd, cwd=code, stdout=f, stderr=subprocess.STDOUT, check=True)


def predict(code, cfg, ckpt, img_dir, dst, device="cuda", batch=32, size=640):
    """Write study-format predictions: stretch-resize to size x size, rescale to the original size."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    sys.path.insert(0, str(code))
    from src.core import YAMLConfig

    c = YAMLConfig(str(cfg))
    c.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(ckpt, map_location="cpu")
    state = state["ema"]["module"] if "ema" in state else state["model"]
    c.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = c.model.deploy()
            self.post = c.postprocessor.deploy()

        def forward(self, x, sizes):
            return self.post(self.model(x), sizes)

    m = Model().to(device).eval()
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    files = sorted(Path(img_dir).glob("*.jpg"))
    preds, t = [], time.time()
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            ims = [Image.open(f).convert("RGB") for f in chunk]
            x = torch.stack([tf(im) for im in ims]).to(device)
            sizes = torch.tensor([[im.width, im.height] for im in ims], device=device)
            labels, boxes, scores = m(x, sizes)
            for f, lab, box, sc, im in zip(chunk, labels, boxes, scores, ims):
                box = box.clone()
                box[:, 0::2] = box[:, 0::2].clamp(0, im.width)  # clip like Ultralytics
                box[:, 1::2] = box[:, 1::2].clamp(0, im.height)
                for l, b, s in zip(lab.tolist(), box.tolist(), sc.tolist()):
                    preds.append({"image_id": f.stem, "category_id": int(l),
                                  "bbox": [round(v, 2) for v in b], "score": round(s, 5)})
    Path(dst).write_text(json.dumps(preds))
    log("dfine predicted", len(files), "images", len(preds), "boxes", round(time.time() - t), "s")
    return dst


def run(job, ws, data):
    """job['dfine'] = {'epochs': 60, 'stop': 48, 'probe_epochs': None|int}."""
    spec = job["dfine"]
    root = Path(os.environ.get("DFINE_ROOT", "/content/dfine"))
    root.mkdir(parents=True, exist_ok=True)
    code = ensure_repo(root)
    work = Path(ws) / "out" / job["name"]
    work.mkdir(parents=True, exist_ok=True)
    proc = Path(data) / "proc"
    make_coco(proc, "train", work / "train.json")
    make_coco(proc, "val", work / "val.json")
    size = spec.get("size", 640)
    cfg = write_config(code, proc, work, spec.get("epochs", 60), spec.get("stop", 48), size=size)
    t = time.time()
    train(code, cfg, root / "dfine_n_coco.pth", work, spec.get("probe_epochs"))
    log("dfine trained", round(time.time() - t), "s")
    out = work / "output"
    ckpt = out / "best_stg2.pth" if (out / "best_stg2.pth").exists() else out / "best_stg1.pth"
    if not ckpt.exists():
        ckpt = out / "last.pth"
    log("using checkpoint", ckpt.name)
    predict(code, cfg, ckpt, proc / "rgb" / "val" / "images", work / "preds_val.json", size=size)
    if spec.get("predict_test"):  # gzip: a full test split exceeds the 100 MB transfer limit
        import gzip

        tmp = work / "preds_test.json"
        predict(code, cfg, ckpt, proc / "rgb" / "test" / "images", tmp, size=size)
        with open(tmp, "rb") as f, gzip.open(work / "preds_test.json.gz", "wb") as g:
            g.write(f.read())
        tmp.unlink()
    keep = out / "best.pth"
    import torch

    sd = torch.load(ckpt, map_location="cpu")
    torch.save({"ema": sd["ema"]} if "ema" in sd else {"model": sd["model"]}, keep)
    for p in out.glob("*.pth"):
        if p != keep:
            p.unlink()
    for name in ("train.json", "val.json"):
        (work / name).unlink(missing_ok=True)
    import shutil

    for d in ("eval", "summary", "train_samples", "val_samples"):
        shutil.rmtree(out / d, ignore_errors=True)


def finish(job, ws, data):
    """Recover a D-FINE run whose workspace was too large to return (e.g. stopped by the job
    timeout). job['dfine_src'] = the remote work dir of that run. Reports its state; if training
    is incomplete and job['dfine'].get('resume'), resumes from last.pth for the remaining epochs;
    then predicts val and copies best.pth, preds_val.json and log.txt into this job's out dir."""
    import shutil

    import torch

    spec = job.get("dfine", {})
    root = Path(os.environ.get("DFINE_ROOT", "/content/dfine"))
    code = ensure_repo(root)
    src = Path(job["dfine_src"])
    out = src / "output"
    dst = Path(ws) / "out" / job["name"]
    dst.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                         capture_output=True, text=True).stdout
    log("gpu processes", gpu.replace("\n", " | "))
    ps = subprocess.run(["pgrep", "-af", "train.py"], capture_output=True, text=True).stdout
    log("train.py processes", ps.replace("\n", " | "))
    for p in sorted(src.rglob("*")):
        if p.is_file() and "samples" not in p.parts[-2]:
            log("file", p.relative_to(src), round(p.stat().st_size / 2**20, 1), "MiB")
    logs = (out / "log.txt").read_text().splitlines() if (out / "log.txt").exists() else []
    done = len(logs)
    log("epochs logged", done)
    if logs:
        log("last epoch stats", logs[-1][:400])
    proc = Path(data) / "proc"
    cfg = src / "dfine_n_dronevehicle.yml"
    if done < spec.get("epochs", 60) and spec.get("resume") and (out / "last.pth").exists():
        make_coco(proc, "train", src / "train.json")
        make_coco(proc, "val", src / "val.json")
        cmd = [sys.executable, "train.py", "-c", str(cfg), "--use-amp", "--seed", "0", "-r", str(out / "last.pth")]
        log("resume", " ".join(cmd[1:]))
        t = time.time()
        with open(src / "train.log", "a") as f:
            subprocess.run(cmd, cwd=code, stdout=f, stderr=subprocess.STDOUT, check=True)
        log("resumed training", round(time.time() - t), "s")
        logs = (out / "log.txt").read_text().splitlines()
    ckpt = next((out / n for n in ("best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth")
                 if (out / n).exists()), None)
    log("using checkpoint", ckpt.name if ckpt else None, "epochs logged", len(logs))
    for split in job.get("dfine_splits", ["val"]):
        predict(code, cfg, ckpt, proc / "rgb" / split / "images", dst / f"preds_{split}.json")
    sd = torch.load(ckpt, map_location="cpu")
    torch.save({"ema": sd["ema"]} if "ema" in sd else {"model": sd["model"]}, dst / "best.pth")
    shutil.copy2(out / "log.txt", dst / "log.txt") if (out / "log.txt").exists() else None
    (dst / "status.json").write_text(json.dumps({"checkpoint": ckpt.name, "epochs_logged": len(logs)}))


def predict_job(job, ws, data):
    """Predict a split with a saved D-FINE checkpoint (job['dfine_ckpt'] in the workspace) and
    write gzip-compressed predictions (a full test split exceeds the 100 MB transfer limit)."""
    import gzip

    root = Path(os.environ.get("DFINE_ROOT", "/content/dfine"))
    root.mkdir(parents=True, exist_ok=True)
    code = ensure_repo(root)
    work = Path(ws) / "out" / job["name"]
    work.mkdir(parents=True, exist_ok=True)
    proc = Path(data) / "proc"
    cfg = write_config(code, proc, work)
    for split in job.get("dfine_splits", ["test"]):
        tmp = work / f"preds_{split}.json"
        predict(code, cfg, Path(ws) / job["dfine_ckpt"], proc / "rgb" / split / "images", tmp)
        with open(tmp, "rb") as f, gzip.open(work / f"preds_{split}.json.gz", "wb") as g:
            g.write(f.read())
        tmp.unlink()
    (work / "dfine_n_dronevehicle.yml").unlink(missing_ok=True)
