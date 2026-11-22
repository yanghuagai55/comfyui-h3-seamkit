# -*- coding: utf-8 -*-
"""Stage 0–1 用的指标（无 ComfyUI 依赖）

三个量，都对着 Oracle 比：

* `bg_hf_sharpness` —— 背景区高频能量比：run / oracle。任务书要求 ≥ 0.90×
* `seam_grad_ratio` —— 前景边界一圈的梯度能量 / 前景内部梯度能量。越接近 1 越没有接缝带
* `fg_psnr`         —— 前景区 PSNR（保留细节的底线）

视频用 ffmpeg 解成帧（Colab 自带，不用装 opencv）。
"""

from __future__ import annotations

import subprocess

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["video_frames", "hf_energy", "seam_grad_ratio", "psnr", "evaluate"]


def video_frames(path, size=None, limit=None):
    """→ [T,H,W,3] uint8。size=(W,H)。"""
    if size is None:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True).stdout.strip()
        w, h = [int(v) for v in probe.split(",")[:2]]
        size = (w, h)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=%d:%d" % size,
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if not raw:
        raise RuntimeError("ffmpeg 解码失败: %s" % path)
    w, h = size
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return fr[:limit] if limit else fr


def _gray_t(x):
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 3, 1, 2) / 255.0
    return t.mean(1, keepdim=True)


def _lap(x):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1).abs()


def _grad(x):
    """同尺寸的 |dx|+|dy|（replicate 补边，输出严格 H×W）。"""
    gx = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dx = (gx[:, :, 1:-1, 2:] - gx[:, :, 1:-1, :-2]).abs()
    dy = (gx[:, :, 2:, 1:-1] - gx[:, :, :-2, 1:-1]).abs()
    return dx + dy


def _mask_at(mask, T, H, W):
    m = torch.as_tensor(mask)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = m.float()
    if m.shape[0] != T:
        idx = torch.linspace(0, m.shape[0] - 1, T).long()
        m = m[idx]
    if tuple(m.shape[-2:]) != (H, W):
        m = F.interpolate(m.unsqueeze(1), (H, W), mode="nearest").squeeze(1)
    return m > 0.5


def _mean_in(x, w):
    s = w.sum()
    return float((x * w).sum() / s) if float(s) > 1 else 0.0


def hf_energy(frames, mask=None):
    g = _gray_t(frames)
    e = _lap(g)
    if mask is None:
        return float(e.mean())
    m = _mask_at(mask, g.shape[0], g.shape[2], g.shape[3]).unsqueeze(1).float()
    return _mean_in(e, m)


def seam_grad_ratio(frames, mask, ring=2):
    g = _gray_t(frames)
    grad = _grad(g)
    m = _mask_at(mask, g.shape[0], g.shape[2], g.shape[3]).unsqueeze(1).float()
    grown = F.max_pool2d(m, 2 * ring + 1, 1, ring)
    shrunk = 1.0 - F.max_pool2d(1.0 - m, 2 * ring + 1, 1, ring)
    band = (grown - m).clamp(0, 1)
    core = (m - shrunk).clamp(0, 1)
    return _mean_in(grad, band) / max(_mean_in(grad, core), 1e-6)


def psnr(a, b, mask=None):
    x, y = _gray_t(a), _gray_t(b)
    n = min(x.shape[0], y.shape[0])
    x, y = x[:n], y[:n]
    d = (x - y) ** 2
    if mask is None:
        mse = float(d.mean())
    else:
        m = _mask_at(mask, x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1).float()
        mse = _mean_in(d, m)
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def evaluate(run_path, oracle_path, mask, size=(320, 180)):
    a = video_frames(oracle_path, size=size)
    b = video_frames(run_path, size=size)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = _mask_at(mask, n, size[1], size[0]).numpy()
    inv = ~m
    return {
        "fg_psnr": round(psnr(b, a, m), 2),
        "bg_hf_sharpness": round(hf_energy(b, inv) / max(hf_energy(a, inv), 1e-9), 3),
        "seam_grad_ratio": round(seam_grad_ratio(b, m), 3),
        "frames": n,
    }
