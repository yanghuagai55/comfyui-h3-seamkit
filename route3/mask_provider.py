# -*- coding: utf-8 -*-
"""把一份像素帧级掩码，适配到任意 latent token 网格。

为什么需要：一采和二采的网格不一样（1280×736 → patch 23×40；二采 1.5× → 35×60），
而掩码是从 Oracle 视频（像素帧）上算出来的。所以掩码必须能"按当前形状现算"，
不能存成某一档的固定尺寸。

时间轴：H3 的 5 个 latent token = 17 像素帧（FRAME_PER_TOKEN=(1,4,4,4,4)）。
一个 token 覆盖多帧时取 **max**（宁可多留前景，也不要漏掉运动）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .splitstream_attn import FRAME_PER_TOKEN

__all__ = ["token_spans", "MaskProvider", "save_mask", "load_mask"]


def token_spans(latent_t: int, n_frames: int | None = None):
    """每个 latent token 覆盖的像素帧区间 [start, end)。"""
    spans, cur = [], 0
    k = 0
    for _ in range(latent_t):
        span = FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)]
        spans.append((cur, cur + span))
        cur += span
        k += 1
    if n_frames is not None and spans:
        spans = [(min(a, n_frames - 1), min(b, n_frames)) for a, b in spans]
    return spans


class MaskProvider:
    """mask: [F, H, W] bool / 0-1，像素空间（Oracle 分辨率）。"""

    def __init__(self, mask):
        m = torch.as_tensor(mask)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        self.mask = m.float()
        self.frames = int(self.mask.shape[0])

    def for_grid(self, latent_t: int, lat_h: int, lat_w: int, start_token: int = 0):
        """→ [latent_t, gh*gw] bool（直接可喂 route3_plan 的 fg_mask）。

        start_token > 0 用于多窗二采（如 10s/15s 分 2 窗以上）：精确切取本窗对应的
        [start_token .. start_token + latent_t) 帧区间，绝不把第 1 窗的掩码错套到第 2 窗。
        """
        gh, gw = lat_h // 2, lat_w // 2
        m_in = self.mask.unsqueeze(1)
        # 保守空间降采样（与时间轴 seg.amax 一致）：任何指尖/发梢落在格内均保留为前景，绝不啃出边缘缺口！
        if m_in.shape[-2] >= gh and m_in.shape[-1] >= gw:
            m = F.adaptive_max_pool2d(m_in, (gh, gw)).squeeze(1) > 0.5
        else:
            m = F.interpolate(m_in, size=(gh, gw), mode="nearest").squeeze(1) > 0.5
        st = max(0, int(start_token))
        total_tok = st + int(latent_t)
        # 若掩码本身已在 latent 时间轴上（如 AutoMask 未接 video_vae 的 latent_motion 模式），
        # 直接按 token 轴 1:1 切片，避免按像素帧 token_spans 截断后半段。
        if self.frames <= total_tok + 2 and (self.frames == int(latent_t) or st > 0 or self.frames < total_tok * 2):
            idx = torch.arange(st, total_tok, dtype=torch.long).clamp(0, max(0, self.frames - 1))
            return m[idx].reshape(int(latent_t), -1)
        spans = token_spans(total_tok, self.frames)[st:total_tok]
        rows = []
        for a, b in spans:
            seg = m[a:max(b, a + 1)]
            rows.append(seg.amax(0) if seg.numel() else torch.zeros(gh, gw, dtype=torch.bool))
        return torch.stack(rows).reshape(int(latent_t), -1)

    def coverage(self, latent_t, lat_h, lat_w):
        g = self.for_grid(latent_t, lat_h, lat_w)
        return round(float(g.float().mean()), 4)


def save_mask(path, mask, meta=None):
    m = torch.as_tensor(mask).cpu().numpy().astype(bool)
    np.savez_compressed(path, mask=m, meta=np.array([str(meta or {})]))
    return path


def load_mask(path):
    d = np.load(path, allow_pickle=True)
    return MaskProvider(d["mask"])
