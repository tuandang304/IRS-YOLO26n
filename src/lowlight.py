"""Training helpers for the uav-lowlight-yolo26 study (Ultralytics 8.4.173).

One trainer class serves every run so that B1, T_IR, B2 and P share a code path:

- ``LLTrainer``: Ultralytics DetectionTrainer plus
  * chunked training: stop cleanly after ``CONFIG["stop_after_epoch"]`` epochs
    while keeping the full schedule, so the next job resumes from last.pt;
  * optional LLSA (low-light sensor-model augmentation) applied on the GPU to
    day composites after the geometric augmentation;
  * optional paired RGB-IR loading (4-channel images) for thermal-guided
    distillation, with IR-only boxes carried as an ignore class.
- ``LowLightDistill``: Ultralytics DistillationModel with two frozen teachers:
  B1 (RGB, day-to-night consistency on LLSA samples, DNC) and T_IR (thermal,
  TGD). Only the student is deployed; see ``export_student``.

Configure through the module-level CONFIG before building the trainer; the
same CONFIG must be set again before resuming.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.augment import Albumentations, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine import trainer as _trainer_mod
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import distill_model as _dm
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import unwrap_model

IGNORE_CLS = 5  # IR-only boxes; removed from the detection loss, used to mask TGD
SOFT_OFFSET = 10  # IR-only boxes kept as soft labels travel as class + SOFT_OFFSET
PSEUDO_OFFSET = 20  # fusion-teacher pseudo boxes travel as class + PSEUDO_OFFSET

CONFIG = {
    "paired": False,          # train on 4-channel RGB+IR images (needed for TGD)
    "ir_only_dir": None,      # <dir>/<id>.txt, normalized xywh IR-only boxes
    "ir_soft": None,          # weight in (0, 1]: keep IR-only boxes as soft labels (needs ir_soft_dir)
    "ir_soft_dir": None,      # <dir>/<id>.txt, "cls x y w h" IR-only boxes
    "pseudo": None,           # weight in (0, 1]: fusion-teacher pseudo boxes as soft labels
    "pseudo_file": None,      # json {id: [[cls, cx, cy, w, h], ...]} normalized pseudo boxes
    "llsa": None,             # dict of LLSA parameters, or None
    "dnc_lambda": 0.0,
    "tgd_lambda": 0.0,
    "tgd_levels": [0, 1, 2],  # neck levels (strides 8/16/32) used by TGD
    "tgd_plain": False,       # baseline cross-modal feature KD: uniform weights, no GT weighting or IR-only mask
    "ir_teacher": None,       # path to the frozen IR (or RGB+IR fusion) teacher checkpoint
    "teacher_fused": False,   # the IR teacher takes 4-channel R,G,B,IR input (fusion teacher)
    "teacher_rgb": False,     # the "IR teacher" slot holds an RGB teacher fed the student's clean RGB
    "head_kd": 0.0,           # weight of one-to-many head distillation (class scores + boxes)
    "head_kd_o2o": 0.0,       # weight of the same distillation on the one-to-one (inference) branch
    "fusion_input": False,    # train a 4-channel R,G,B,IR model (the fusion teacher itself)
    "ramp_epochs": 3,
    "stop_after_epoch": None, # chunk end: stop once this many epochs are done
}

LLSA_DEFAULT = {
    "p": 0.5, "tau_gate": None, "k_min": 0.02, "k_max": 0.25, "wb": (0.8, 1.2),
    "a": (1e-3, 1e-2), "sigma_r": (2e-3, 2e-2), "jpeg_p": 0.5, "jpeg_q": (60, 95),
}


# ----------------------------------------------------------------------------- LLSA

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def linear_to_srgb(y):
    return torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def _log_uniform(n, lo, hi, device, shape=(1, 1, 1)):
    u = torch.rand(n, *shape, device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _jpeg(img, quality):
    """Re-encode one float RGB image in [0, 1] (3xHxW) as JPEG on the CPU."""
    from torchvision.io import decode_jpeg, encode_jpeg

    u8 = (img * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    out = decode_jpeg(encode_jpeg(u8, quality=int(quality)))
    return out.to(img.device, torch.float32) / 255


@torch.no_grad()
def llsa(x, params=None):
    """Darken RGB images x (n x 3 x H x W, float in [0, 1]) with a sensor-noise model."""
    p = {**LLSA_DEFAULT, **(params or {})}
    n, dev = x.shape[0], x.device
    lin = srgb_to_linear(x.float())
    wb = torch.empty(n, 3, 1, 1, device=dev).uniform_(*p["wb"])
    k = _log_uniform(n, p["k_min"], p["k_max"], dev)
    a = _log_uniform(n, *p["a"], dev)
    sr = _log_uniform(n, *p["sigma_r"], dev)
    y = lin * wb * k
    y = y + torch.randn_like(y) * torch.sqrt(a * y.clamp(min=0) + sr ** 2)
    s = linear_to_srgb(y.clamp(0, 1))
    s = torch.round(s * 255) / 255
    for j in range(n):
        if torch.rand(1).item() < p["jpeg_p"]:
            q = torch.randint(p["jpeg_q"][0], p["jpeg_q"][1] + 1, (1,)).item()
            s[j] = _jpeg(s[j], q)
    return s


def luminance(x):
    """Mean luminance on the 0-255 scale for RGB images in [0, 1] (n x 3 x H x W)."""
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean((1, 2)) * 255


# --------------------------------------------------------------------- paired dataset

class _RGBOnly:
    """Apply a pixel-level transform to the first three channels of a 4-channel image."""

    def __init__(self, t):
        self.t = t

    def __call__(self, labels):
        img = labels["img"]
        if img.ndim != 3 or img.shape[2] != 4:
            return self.t(labels)
        labels["img"] = np.ascontiguousarray(img[..., :3])
        labels = self.t(labels)
        img[..., :3] = labels["img"]
        labels["img"] = img
        return labels


class IRPairDataset(YOLODataset):
    """YOLO dataset that loads BGR + IR as one 4-channel image so geometric
    augmentation is identical for both, and appends IR-only boxes as IGNORE_CLS."""

    def load_image(self, i, rect_mode=True, resize_short=False):
        im = self.ims[i]
        if im is not None:
            return im, self.im_hw0[i], self.im_hw[i]
        f = self.im_files[i]
        rgb = cv2.imread(f, cv2.IMREAD_COLOR)
        ir = cv2.imread(f.replace("/rgb/", "/ir/"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None or rgb.shape[:2] != ir.shape[:2]:
            raise FileNotFoundError(f"missing or mismatched RGB/IR pair for {f}")
        im = np.dstack([rgb, ir])
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.augment and self.cache != "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]

    def update_labels_info(self, label):
        if CONFIG.get("pseudo"):
            if not hasattr(self, "_pseudo"):
                self._pseudo = json.loads(Path(CONFIG["pseudo_file"]).read_text())
            a = np.array(self._pseudo.get(Path(label["im_file"]).stem, []), dtype=np.float32).reshape(-1, 5)
            if len(a):
                label["bboxes"] = np.concatenate([label["bboxes"], a[:, 1:5]], 0)
                label["cls"] = np.concatenate(
                    [label["cls"], (a[:, :1] + PSEUDO_OFFSET).astype(label["cls"].dtype)], 0)
        if CONFIG.get("ir_soft"):
            p = Path(CONFIG["ir_soft_dir"]) / f"{Path(label['im_file']).stem}.txt"
            if p.exists() and p.stat().st_size:
                a = np.loadtxt(p, ndmin=2, dtype=np.float32)
                label["bboxes"] = np.concatenate([label["bboxes"], a[:, 1:5]], 0)
                label["cls"] = np.concatenate(
                    [label["cls"], (a[:, :1] + SOFT_OFFSET).astype(label["cls"].dtype)], 0)
            return super().update_labels_info(label)
        d = CONFIG.get("ir_only_dir")
        if d:
            p = Path(d) / f"{Path(label['im_file']).stem}.txt"
            if p.exists() and p.stat().st_size:
                ign = np.loadtxt(p, ndmin=2, dtype=np.float32)
                label["bboxes"] = np.concatenate([label["bboxes"], ign], 0)
                label["cls"] = np.concatenate(
                    [label["cls"], np.full((len(ign), 1), IGNORE_CLS, dtype=label["cls"].dtype)], 0)
        return super().update_labels_info(label)

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        t.transforms = [_RGBOnly(x) if isinstance(x, (Albumentations, RandomHSV)) else x
                        for x in t.transforms]
        return t


# ------------------------------------------------------------------ distillation model

def _box_masks(bboxes, idx, bs, h, w, dilate=1.0):
    """bs x h x w mask of feature cells whose centre lies in a box dilated by `dilate` cells.

    bboxes: N x 4 normalized xywh relative to the square training image."""
    out = torch.zeros(bs, h, w, device=bboxes.device)
    if len(bboxes) == 0:
        return out
    gy = (torch.arange(h, device=bboxes.device) + 0.5) / h
    gx = (torch.arange(w, device=bboxes.device) + 0.5) / w
    x1 = bboxes[:, 0] - bboxes[:, 2] / 2 - dilate / w
    x2 = bboxes[:, 0] + bboxes[:, 2] / 2 + dilate / w
    y1 = bboxes[:, 1] - bboxes[:, 3] / 2 - dilate / h
    y2 = bboxes[:, 1] + bboxes[:, 3] / 2 + dilate / h
    inx = (gx[None] >= x1[:, None]) & (gx[None] <= x2[:, None])  # N x w
    iny = (gy[None] >= y1[:, None]) & (gy[None] <= y2[:, None])  # N x h
    inside = (iny[:, :, None] & inx[:, None, :]).float()        # N x h x w
    return out.index_reduce_(0, idx.long(), inside, "amax")


class LowLightDistill(_dm.DistillationModel):
    """Student YOLO26n with two frozen teachers: B1 for DNC and T_IR for TGD."""

    def __init__(self, teacher_model, student_model):
        super().__init__(teacher_model, student_model)
        dims = [p[0].in_channels for p in self.projector]
        device = next(student_model.parameters()).device
        self.ramp = 1.0
        self._ir_feats, self._ir_hooks = {}, []
        self.ir_teacher_model = None
        tdims = dims
        path = CONFIG.get("ir_teacher")
        if path and (CONFIG.get("tgd_lambda", 0) > 0 or CONFIG.get("head_kd", 0) > 0
                     or CONFIG.get("head_kd_o2o", 0) > 0):
            self.ir_teacher_model = load_checkpoint(path)[0].float().to(device)
            self.ir_teacher_model.eval()
            for v in self.ir_teacher_model.parameters():
                v.requires_grad = False
            self._register_ir_hooks()
            ch = 4 if CONFIG.get("teacher_fused") else 3
            with torch.no_grad():
                self.ir_teacher_model(torch.zeros(1, ch, 64, 64, device=device))
            tdims = [self.decouple_outputs(self._ir_feats[i]).shape[1] for i in range(len(dims))]
            self._ir_feats.clear()
        adapters = []
        for c, ct in zip(dims, tdims):  # training-only 1x1 adapters toward the IR teacher
            conv = nn.Conv2d(c, ct, 1, bias=True)
            nn.init.zeros_(conv.bias)
            if c == ct:  # identity init when widths match
                with torch.no_grad():
                    conv.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            adapters.append(conv)
        self.projector = nn.ModuleList(adapters).to(device)

    def _register_ir_hooks(self):
        for h in self._ir_hooks:
            h.remove()
        self._ir_hooks = []
        if self.ir_teacher_model is None:
            return
        for i, t in enumerate(self.get_distill_layers(self.ir_teacher_model)):  # neck levels + head
            self._clear_feature_hooks(self.ir_teacher_model.model[t])
            self._ir_hooks.append(
                self.ir_teacher_model.model[t].register_forward_hook(_dm.FeatureHook(self._ir_feats, i)))

    def __getstate__(self):
        # Copies (EMA, checkpoints) never carry the IR teacher; it is rebuilt from CONFIG.
        self._ir_feats.clear()
        state = super().__getstate__()
        state["_ir_hooks"] = []
        state["ir_teacher_model"] = None
        return state

    def __setstate__(self, state):
        state.setdefault("ir_teacher_model", None)
        state.setdefault("ramp", 1.0)
        super().__setstate__(state)
        self._ir_feats, self._ir_hooks = {}, []
        self._register_ir_hooks()

    def train(self, mode=True):
        super().train(mode)
        if self.ir_teacher_model is not None:
            self.ir_teacher_model.eval()
        return self

    def loss(self, batch, preds=None):
        dev = batch["img"].device
        zero = torch.zeros(1, device=dev)
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular, items = self.student_model.loss(batch, preds)
            items["dnc"], items["tgd"] = zero.detach(), zero.detach()
            return torch.cat([regular, zero, zero]), items

        bs = batch["img"].shape[0]
        lam_d = CONFIG["dnc_lambda"] * self.ramp
        lam_t = CONFIG["tgd_lambda"] * self.ramp if self.ir_teacher_model is not None else 0.0
        dark = batch.get("dark")
        use_dnc = lam_d > 0 and dark is not None and bool(dark.any())
        lam_h = CONFIG.get("head_kd", 0.0) * self.ramp if self.ir_teacher_model is not None else 0.0
        lam_o = CONFIG.get("head_kd_o2o", 0.0) * self.ramp if self.ir_teacher_model is not None else 0.0
        use_tgd = (lam_t > 0 or lam_h > 0 or lam_o > 0) and ("img_ir" in batch or CONFIG.get("teacher_rgb"))
        self._teacher_feats.clear()
        self._student_feats.clear()
        self._ir_feats.clear()
        with torch.no_grad():
            if use_dnc:
                self.teacher_model(batch["img_clean"])
            if use_tgd:
                if CONFIG.get("teacher_rgb"):
                    tin = batch.get("img_clean", batch["img"])
                elif CONFIG.get("teacher_fused"):  # R, G, B (clean if LLSA darkened the student) + IR
                    tin = torch.cat([batch.get("img_clean", batch["img"]), batch["img_ir"][:, :1]], 1)
                else:
                    tin = batch["img_ir"]
                self.ir_teacher_model(tin)
        preds = self.student_model(batch["img"])
        regular, items = self.student_model.loss(batch, preds)

        n = len(self.feats_idx) - 1
        dnc, tgd = zero.clone(), zero.clone()
        for i in range(n):
            s = self.decouple_outputs(self._student_feats[i])
            h, w = s.shape[-2:]
            gt = _box_masks(batch["bboxes"], batch["batch_idx"], bs, h, w)
            weight = (0.1 + 0.9 * gt)[:, None]
            if use_dnc:
                t = self.decouple_outputs(self._teacher_feats[i])
                e = (F.normalize(s.float(), dim=1) - F.normalize(t.float(), dim=1)).pow(2).sum(1, keepdim=True)
                wd = weight * dark.float().view(-1, 1, 1, 1)
                dnc = dnc + (e * wd).sum() / (wd.sum() + 1e-9)
            if lam_t > 0 and use_tgd and i in CONFIG["tgd_levels"]:
                t = self.decouple_outputs(self._ir_feats[i])
                ign = _box_masks(batch.get("ign_bboxes", batch["bboxes"][:0]),
                                 batch.get("ign_idx", batch["batch_idx"][:0]), bs, h, w, dilate=0.0)
                wt = weight * (1 - ign * (1 - gt))[:, None]
                if CONFIG.get("tgd_plain"):
                    wt = torch.ones_like(wt)
                e = (F.normalize(self.projector[i](s).float(), dim=1)
                     - F.normalize(t.float(), dim=1)).pow(2).sum(1, keepdim=True)
                tgd = tgd + (e * wt).sum() / (wt.sum() + 1e-9)
        dnc = dnc / n
        tgd = tgd / max(len(CONFIG["tgd_levels"]), 1)
        hkd = zero.clone()
        if lam_h > 0 and use_tgd:
            hkd = self._head_kd(preds, self._ir_feats[n])
            # follow YOLO26's one-to-many weight decay (0.8 -> 0.1) so KD does not dominate late
            o2m = getattr(self.student_model.criterion, "o2m", 0.8) if self.student_model.criterion else 0.8
            hkd = hkd * (o2m / 0.8)
        hko = zero.clone()
        if lam_o > 0 and use_tgd:  # the one-to-one branch is the one used at inference
            hko = self._head_kd(preds, self._ir_feats[n], branch="one2one")
        items["dnc"], items["tgd"] = dnc.detach(), (tgd + hkd + hko).detach()
        return torch.cat([regular, lam_d * dnc * bs, (lam_t * tgd + lam_h * hkd + lam_o * hko) * bs]), items

    def _head_kd(self, spreds, tpreds, branch="one2many"):
        """One-to-many head distillation: BCE to the teacher's class probabilities and L1 to its
        box distances, weighted per anchor by the teacher's max class probability."""
        s = self.decouple_outputs(spreds, branch=branch)
        t = self.decouple_outputs(tpreds, branch=branch)
        if not (isinstance(s, dict) and isinstance(t, dict)):
            return torch.zeros(1, device=next(self.parameters()).device)
        ss, ts = s["scores"].float(), t["scores"].float().detach()
        sb, tb = s["boxes"].float(), t["boxes"].float().detach()
        if ss.shape != ts.shape:  # different strides or input size: skip rather than misalign
            return torch.zeros(1, device=ss.device)
        tp = ts.sigmoid()
        w = tp.max(1, keepdim=True).values  # (b, 1, A)
        cls = (F.binary_cross_entropy_with_logits(ss, tp, reduction="none") * w).sum() / (w.sum() * ss.shape[1] + 1e-9)
        size = ((tb[:, 0] + tb[:, 2] + tb[:, 1] + tb[:, 3]) / 2).clamp(min=1).unsqueeze(1)  # box scale
        box = ((sb - tb).abs() / size * w).sum() / (w.sum() * 4 + 1e-9)
        return (cls + box).view(1)


# ------------------------------------------------------------------ soft-label detection loss

from ultralytics.utils import loss as _loss_mod  # noqa: E402

_orig_assigned = _loss_mod.v8DetectionLoss.get_assigned_targets_and_loss


def _weighted_assigned(self, preds, batch):
    """v8DetectionLoss with per-box weights: target scores of anchors assigned to a box are
    multiplied by its weight (soft IR-only labels). Identical to the original without box_w."""
    w = batch.get("box_w")
    if w is None or bool((w == 1).all()):
        return _orig_assigned(self, preds, batch)
    from ultralytics.utils.tal import make_anchors

    loss = torch.zeros(3, device=self.device)
    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"],
                         w.view(-1, 1).to(batch["bboxes"].dtype)), 1)
    targets = self.preprocess(targets.to(self.device), batch_size,
                              scale_tensor=imgsz[[1, 0, 1, 0]])  # weight column 5 is not scaled
    gt_labels, gt_bboxes, gt_w = targets.split((1, 4, 1), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
    _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
        pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
    aw = gt_w.squeeze(-1).gather(1, target_gt_idx)  # (b, anchors) weight of the assigned box
    target_scores = target_scores * torch.where(fg_mask, aw, torch.ones_like(aw)).unsqueeze(-1)
    target_scores_sum = target_scores.sum().clamp_(min=1)
    bce_loss = self.bce(pred_scores, target_scores.to(dtype))
    if self.class_weights is not None:
        bce_loss *= self.class_weights
    loss[1] = bce_loss.sum() / target_scores_sum
    loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                      target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
    loss[0] *= self.hyp.box
    loss[1] *= self.hyp.cls
    loss[2] *= self.hyp.dfl
    return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
            dict(zip(self.loss_names, loss.detach())))


_loss_mod.v8DetectionLoss.get_assigned_targets_and_loss = _weighted_assigned

# The trainer constructs and rebuilds (on resume) DistillationModel by this module global.
_trainer_mod.DistillationModel = LowLightDistill


# ------------------------------------------------------------------------------ trainer

class LLTrainer(DetectionTrainer):
    """DetectionTrainer with chunked stopping, LLSA and paired RGB-IR loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk_stopped = False
        self.add_callback("on_train_epoch_start", _set_ramp)
        self.add_callback("on_fit_epoch_end", _chunk_stop)

    def build_dataset(self, img_path, mode="train", batch=None):
        fusion = CONFIG.get("fusion_input")
        if not fusion and (mode != "train" or not CONFIG["paired"]):
            return super().build_dataset(img_path, mode, batch)
        if fusion and mode != "train":  # 4-channel validation, standard letterbox
            from ultralytics.data import build as _build

            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            _build.YOLODataset = IRPairDataset
            try:
                return _build.build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode,
                                                 rect=True, stride=gs)
            finally:
                _build.YOLODataset = YOLODataset
        from ultralytics.data import build as _build

        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        _build.YOLODataset = IRPairDataset  # same constructor arguments, paired loader
        try:
            return _build.build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode,
                                             rect=False, stride=gs)
        finally:
            _build.YOLODataset = YOLODataset

    def preprocess_batch(self, batch):
        cls = batch["cls"].view(-1)
        soft = cls >= SOFT_OFFSET
        pseudo = cls >= PSEUDO_OFFSET
        batch["box_w"] = torch.where(pseudo, float(CONFIG.get("pseudo") or 1.0),
                                     torch.where(soft, float(CONFIG.get("ir_soft") or 1.0), 1.0)).to(torch.float32)
        if soft.any():
            off = torch.where(pseudo, PSEUDO_OFFSET, SOFT_OFFSET).to(batch["cls"].dtype).view_as(batch["cls"])
            batch["cls"] = torch.where(soft.view_as(batch["cls"]), batch["cls"] - off, batch["cls"])
            cls = batch["cls"].view(-1)
        ign = cls == IGNORE_CLS
        if ign.any():
            batch["ign_bboxes"], batch["ign_idx"] = batch["bboxes"][ign], batch["batch_idx"][ign]
            keep = ~ign
            for k in ("cls", "bboxes", "batch_idx", "box_w"):
                batch[k] = batch[k][keep]
        img = batch["img"]
        if img.shape[1] == 4 and CONFIG.get("fusion_input"):  # fusion model input: R, G, B, IR
            batch["img"] = img[:, [2, 1, 0, 3]].contiguous()
        elif img.shape[1] == 4:  # B, G, R, IR (Format does not reorder 4-channel images)
            batch["img"] = img[:, [2, 1, 0]].contiguous()
            batch["img_ir"] = img[:, 3:4].expand(-1, 3, -1, -1).contiguous()
        batch = super().preprocess_batch(batch)
        if "img_ir" in batch:
            batch["img_ir"] = batch["img_ir"].float() / 255
        if "ign_bboxes" not in batch:
            batch["ign_bboxes"] = batch["bboxes"][:0]
            batch["ign_idx"] = batch["batch_idx"][:0]
        params = CONFIG.get("llsa")
        x = batch["img"]
        batch["dark"] = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if params:
            p = {**LLSA_DEFAULT, **params}
            sel = (luminance(x) >= p["tau_gate"]) & (torch.rand(x.shape[0], device=x.device) < p["p"])
            if sel.any():
                dark = x.clone()
                dark[sel] = llsa(x[sel], p).to(x.dtype)
                batch["img_clean"], batch["img"], batch["dark"] = x, dark, sel
        return batch

    def final_eval(self):
        if self._chunk_stopped:
            return  # keep the optimizer in last.pt for the next chunk
        if CONFIG.get("fusion_input"):  # the stock final validator builds 3-channel data
            from ultralytics.utils.torch_utils import strip_optimizer

            for f in (self.last, self.best):
                if f.exists():
                    strip_optimizer(f)
            return
        super().final_eval()

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if weights is not None and self.data.get("channels", 3) == 4:
            w0 = weights.model[0].conv.weight
            c0 = model.model[0].conv.weight
            if w0.shape[1] == 3 and c0.shape[1] == 4:  # RGB stem from COCO, IR channel = RGB mean
                with torch.no_grad():
                    c0[:, :3].copy_(w0.to(c0.dtype))
                    c0[:, 3:].copy_(w0.mean(1, keepdim=True).to(c0.dtype))
        return model

    def get_validator(self):
        v = super().get_validator()
        if CONFIG.get("fusion_input"):
            orig = v.preprocess

            def pre(batch):
                batch = orig(batch)
                if batch["img"].shape[1] == 4:
                    batch["img"] = batch["img"][:, [2, 1, 0, 3]].contiguous()
                return batch

            v.preprocess = pre
        return v


def _set_ramp(trainer):
    m = unwrap_model(trainer.model)
    if isinstance(m, LowLightDistill):
        m.ramp = min(1.0, (trainer.epoch + 1) / max(CONFIG["ramp_epochs"], 1))


def _chunk_stop(trainer):
    stop = CONFIG.get("stop_after_epoch")
    if stop and trainer.epoch + 1 >= stop and trainer.epoch + 1 < trainer.epochs:
        trainer.stop = True
        trainer._chunk_stopped = True


# ------------------------------------------------------------------------------ export

def export_student(src, dst):
    """Write a plain YOLO checkpoint holding only the deployed student network."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if isinstance(m, _dm.DistillationModel):
            ckpt[key] = m.student_model
    ckpt["optimizer"] = None
    torch.save(ckpt, dst)
    return dst
