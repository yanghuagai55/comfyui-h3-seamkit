# -*- coding: utf-8 -*-
"""Route 3 · 活跃区掩码

任务书 v2 §4.2.1 的定义：

    Active(t) = M_fg(t) ∪ HighMotionBG(t) ∪ Ring(r)

* `M_fg`          —— 人物/主体（抠像模型给，或色键）
* `HighMotionBG`  —— **局部**高频运动的背景（窗帘、水流、近景穿行物）
* `Ring(r)`       —— 主体外扩 r 个 patch（阴影、接触边缘、反光）

关键在「局部」：全局运镜（推拉摇移）**不算** HighMotionBG，它由压缩的 BG 关键帧承载。
所以这里先估计逐帧的全局位移场、把它补偿掉，再看残差——否则一推镜整帧都会判成前景，
压缩率直接归零（这是纯帧差方案在 Case-2 上的致命伤）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from route3.assembly import estimate_tile_flow, warp

__all__ = ["dilate", "motion_energy", "micro_motion_guard", "active_mask", "mask_stats"]


def dilate(mask, r: int = 1):
    """[..., H, W] bool/float → 外扩 r 格（方形结构元）。"""
    if r <= 0:
        return mask
    m = mask.float().unsqueeze(0)
    m = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)
    return m.squeeze(0) > 0.5


@torch.no_grad()
def motion_energy(frames, tiles=(6, 8), search=4, blur=1):
    """逐帧「局部」运动能量（已扣掉全局位移）。

    frames : [T,C,H,W] 像素空间（0~1），建议先降到 ~256 宽的预览分辨率再算
    返回    [T,H,W]，t=0 记 0
    """
    T, C, H, W = frames.shape
    e = torch.zeros(T, H, W, device=frames.device, dtype=frames.dtype)
    prev = frames[0].mean(0, keepdim=True)
    for t in range(1, T):
        cur = frames[t].mean(0, keepdim=True)
        fl = estimate_tile_flow(prev[0], cur[0], tiles=tiles, search=search)
        aligned = warp(prev, fl)
        e[t] = (cur - aligned).abs().mean(0)
        prev = cur
    if blur > 0:
        k = 2 * blur + 1
        e = F.avg_pool2d(e.unsqueeze(1), k, stride=1, padding=blur).squeeze(1)
    return e


def micro_motion_guard(e, low_rel=0.04, persist=0.6, min_frames=3):
    """静止人物的保底：低阈值 + 时间持续性。

    盲区：人物站着不动，只有口型/眨眼/呼吸。这类微动的能量远小于画面里真正在动的
    东西（窗帘、车流），用 `rel*max` 这种单一阈值会把静立的人判进 BG → 微表情发糊。

    为什么用「持续性」而不是单纯降阈值：压缩噪声/传感器噪声是随机闪烁，同一个像素
    不会连续多帧都超阈；而呼吸、眨眼是持续发生的。所以「低阈值 + 出现帧数占比」能把
    两者分开，且不需要任何额外模型。

    e: [T,H,W] 运动能量；返回 [T,H,W] bool（逐帧相同，保证静立人物整段都被保护）
    """
    mx = float(e.max())
    T = e.shape[0]
    if mx <= 0:
        return torch.zeros_like(e, dtype=torch.bool)
    hits = (e > low_rel * mx).float()
    need = max(int(min_frames), int(round(persist * T)))
    pers = hits.sum(0) >= need
    return pers.unsqueeze(0).expand(T, *pers.shape).contiguous()


def active_mask(frames, fg_mask=None, ring=1, rel=0.15, floor=1e-3, thr=None,
                tiles=(6, 8), search=4, patch_grid=None, micro=True,
                micro_rel=0.04, persist=0.6):
    """算出 `Active` 掩码，并按需要降采样到 patch 网格。

    frames   : [T,C,H,W]，像素空间
    fg_mask  : [T,H,W] bool 或 None
    patch_grid: (gh, gw) —— 直接给目标网格；不给就用 out_size=(H,W) 自己减半
    返回      [T,gh,gw] bool
    """
    T, C, H, W = frames.shape
    e = motion_energy(frames, tiles=tiles, search=search)
    if thr is None:
        # 相对阈值：按本片自身的运动幅度定标（绝对阈值在不同片子上差别太大）
        thr = max(float(rel) * float(e.max()), float(floor))
    hm = e > thr
    act = hm if not micro else (hm | micro_motion_guard(e, low_rel=micro_rel, persist=persist))
    if fg_mask is not None:
        act = act | fg_mask.to(torch.bool)
    if ring > 0:
        act = torch.stack([dilate(act[t], ring) for t in range(T)])

    if patch_grid is None:
        gh, gw = max(1, H // 2), max(1, W // 2)
    else:
        gh, gw = patch_grid
    # 面积平均后过半即算活跃（保守：宁可多留前景）
    pooled = F.adaptive_avg_pool2d(act.float(), (gh, gw))
    return pooled > 0.5, {"thr": round(float(thr), 5), "energy": e,
                          "micro": bool(micro)}


def mask_stats(mask):
    """mask: [T,gh,gw] bool → 每帧活跃比例 + 平均。"""
    per = mask.float().flatten(1).mean(1)
    return {"per_frame": [round(float(v), 4) for v in per],
            "mean": round(float(per.mean()), 4)}
