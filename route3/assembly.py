# -*- coding: utf-8 -*-
"""Route 3 · 中间帧背景还原（法 A / 法 B）

任务书 §4.2.3 要求 Stage 1 做 A/B：

  法 A（线性插值）  z_bg(j) = z_p1_up(j) + lerp(δ(5k), δ(5k+5), j/5)
  法 B（光流/位移对齐双向 warp，主推）
                    δ(5k) 顺流到 j、δ(5k+5) 逆流到 j，再按 j/5 混合

两者都保留 —— 任务书把法 A 当对照组，不是要删掉它。静机位档（T_bg=1）则直接
把 z_bg_out(0) 铺满全片背景区。

这里的「流」是 latent 网格上的**分块位移场**（默认 6×8 块，搜索半径 4~6 个 latent 像素），
用归一化互相关暴力匹配。够用、无依赖、能在 4060 上几秒跑完一条 5s 片。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["estimate_tile_flow", "flow_to_grid", "warp", "lerp_residual",
           "flow_residual", "reconstruct_static", "reconstruct_moving"]


def _gray(x):
    return x.mean(dim=0) if x.dim() == 3 else x


@torch.no_grad()
def estimate_tile_flow(a, b, tiles=(6, 8), search=4):
    """估计把 a 对齐到 b 的分块位移。

    a, b : [C,H,W] 或 [H,W]
    返回  flow [Th,Tw,2]（末维 = (dy, dx)，单位 latent 像素，b 中的位移量）

    搜索前把 b 四周 replicate-pad `search` 格 —— 否则贴边的块搜不到候选，
    会返回垃圾位移，补偿后的残差全堆在画面四边（实测会把 25% 的分辨率误判成前景）。
    """
    A, B = _gray(a).float(), _gray(b).float()
    H, W = A.shape[-2:]
    th, tw = tiles
    bh, bw = H // th, W // tw
    s = int(search)
    Bp = F.pad(B[None, None], (s, s, s, s), mode="replicate")[0, 0]
    flow = torch.zeros(th, tw, 2, device=A.device, dtype=A.dtype)

    for i in range(th):
        for j in range(tw):
            pa = A[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            pa = pa - pa.mean()
            na = pa.norm() + 1e-6
            best, bdy, bdx = -2.0, 0, 0
            for dy in range(-s, s + 1):
                for dx in range(-s, s + 1):
                    y0, x0 = i * bh + dy + s, j * bw + dx + s
                    pb = Bp[y0:y0 + bh, x0:x0 + bw]
                    pb = pb - pb.mean()
                    sc = (pa * pb).sum() / (na * (pb.norm() + 1e-6))
                    if sc > best:
                        best, bdy, bdx = sc, dy, dx
            flow[i, j, 0] = bdy
            flow[i, j, 1] = bdx
    return flow


def flow_to_grid(flow, H, W):
    """分块位移 → grid_sample 用的采样网格（[1,H,W,2]，已归一化到 [-1,1]）。

    warp 的语义：输出在位置 p 处取输入 p + flow(p) 的值（后向采样）。
    """
    th, tw = flow.shape[:2]
    ys = torch.arange(H, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(W, device=flow.device, dtype=flow.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ti = (gy * th / H).long().clamp(0, th - 1)
    tj = (gx * tw / W).long().clamp(0, tw - 1)
    fy = flow[ti, tj, 0]
    fx = flow[ti, tj, 1]
    sy = gy + fy
    sx = gx + fx
    nx = (sx + 0.5) / W * 2 - 1
    ny = (sy + 0.5) / H * 2 - 1
    return torch.stack([nx, ny], dim=-1).unsqueeze(0)


def warp(x, flow, mode="bilinear"):
    """x: [C,H,W] → 按 flow 重采样。位移是「a 在 b 里的位置」，所以采样用负号。"""
    C, H, W = x.shape
    g = flow_to_grid(-flow, H, W)
    return F.grid_sample(x.unsqueeze(0), g, mode=mode, align_corners=False,
                         padding_mode="border").squeeze(0)


def lerp_residual(da, db, frac):
    """法 A：固定坐标线性插值。"""
    return (1.0 - frac) * da + frac * db


def flow_residual(da, db, frac, flow_ab):
    """法 B：双向 warp 后再混合。

    flow_ab 是「anchor k 的块 → anchor k+1 的块」的位移；中间帧 j=k+frac 时
    把 da 顺流 frac 份、db 逆流 (1-frac) 份，两个残差都落到 j 的位置上。
    """
    wa = warp(da, flow_ab * frac)
    wb = warp(db, flow_ab * (frac - 1.0))
    return (1.0 - frac) * wa + frac * wb


def reconstruct_moving(z_p1_up, delta_keys, anchor_idx, method="flow",
                       tiles=(6, 8), search=4):
    """按关键帧残差还原全部帧的背景。

    z_p1_up    : [T,C,H,W] 一采上采样后的全帧底图（自带 24fps 运镜）
    delta_keys : {anchor_t: δ [C,H,W]}  二采在锚点帧真正画出来的高清细节增量
    anchor_idx : 锚点帧下标（升序）
    method     : "flow"（法 B）| "lerp"（法 A）| "none"（只留关键帧，其余原样）
    返回 [T,C,H,W]
    """
    T, C, H, W = z_p1_up.shape
    out = z_p1_up.clone()
    anchors = list(anchor_idx)
    for t0 in anchors:
        out[t0] = out[t0] + delta_keys[t0]
    for k in range(len(anchors) - 1):
        t0, t1 = anchors[k], anchors[k + 1]
        d0, d1 = delta_keys[t0], delta_keys[t1]
        if method == "none" or t1 - t0 <= 1:
            continue
        fl = estimate_tile_flow(z_p1_up[t0], z_p1_up[t1], tiles=tiles, search=search)
        for j in range(t0 + 1, t1):
            frac = (j - t0) / float(t1 - t0)
            d = flow_residual(d0, d1, frac, fl) if method == "flow" \
                else lerp_residual(d0, d1, frac)
            out[j] = out[j] + d
    return out


def reconstruct_static(z_p1_up, bg_hd, fg_mask=None):
    """静机位档（T_bg=1）：整片背景区直接铺 z_bg_out(0)（或外部 4K 底图）。

    bg_hd   : [C,H,W] 已重绘好的高清背景
    fg_mask : [T,H,W] bool 或 None（None 表示整帧都是背景，只在纯静机位用）
    """
    T, C, H, W = z_p1_up.shape
    out = z_p1_up.clone()
    for t in range(T):
        if fg_mask is None:
            out[t] = bg_hd
        else:
            m = fg_mask[t].unsqueeze(0)
            out[t] = torch.where(m, out[t], bg_hd)
    return out
