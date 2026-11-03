"""Zero-DCE++ enhancement for benchmark B2 (enhance-then-detect).

Network and weights: Li, Guo and Loy, Zero-DCE++ (TPAMI 2021), official code
https://github.com/Li-Chongyi/Zero-DCE_extension (commit 09f202b), snapshot
Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth (sha256 ca8855b9...). The model code
below is copied from model.py of that repository, which is released for academic
research under CC BY-NC 4.0. Used without fine-tuning, at scale_factor=1 (curve
maps estimated at full 640x512 resolution; the official demo downscales by 12 for
speed and crops images to multiples of 12).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class enhance_net_nopool(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        f = 32
        self.e_conv1 = CSDN_Tem(3, f)
        self.e_conv2 = CSDN_Tem(f, f)
        self.e_conv3 = CSDN_Tem(f, f)
        self.e_conv4 = CSDN_Tem(f, f)
        self.e_conv5 = CSDN_Tem(f * 2, f)
        self.e_conv6 = CSDN_Tem(f * 2, f)
        self.e_conv7 = CSDN_Tem(f * 2, 3)

    @staticmethod
    def enhance(x, x_r):
        for _ in range(8):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x

    def forward(self, x):
        x_down = x if self.scale_factor == 1 else F.interpolate(x, scale_factor=1 / self.scale_factor,
                                                                mode="bilinear")
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        return self.enhance(x, x_r), x_r


def load(weights, device):
    net = enhance_net_nopool(1).to(device).eval()
    net.load_state_dict(torch.load(weights, map_location=device))
    return net


@torch.no_grad()
def enhance_dir(src, dst, weights, device="cuda", batch=16, quality=95):
    """Enhance every JPEG in src into dst (same names, JPEG quality 95, 4:4:4)."""
    net = load(weights, device)
    files = sorted(Path(src).glob("*.jpg"))
    Path(dst).mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(os.cpu_count())

    def read(f):
        return np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255

    def save(arr, f):
        Image.fromarray(arr).save(Path(dst) / f.name, quality=quality, subsampling=0)

    pending = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        x = torch.from_numpy(np.stack(list(pool.map(read, chunk)))).permute(0, 3, 1, 2).to(device)
        y, _ = net(x)
        out = (y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        pending += [pool.submit(save, o, f) for o, f in zip(out, chunk)]
    for p in pending:
        p.result()
    return len(files)
