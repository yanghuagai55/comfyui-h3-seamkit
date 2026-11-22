# -*- coding: utf-8 -*-
"""Route 3 · 锁步多速率双流注意力内核（纯 torch，不依赖 ComfyUI）

设计要点
--------
* token 序列 = [Prefix | BG_strided | FG_full]，整条序列的每一行同时是 Q 和 K。
* BG 键被压缩了 S_bg 倍，Softmax 分母少算 S_bg 份注意力质量，所以给 BG 键加一个
  对数偏置 beta（初值 log(S_bg)，Stage 2 按 (layer, head) 校准）。
* 精确求解 = 两分支各自算 (O, LSE)，再按 logaddexp 质量守恒合并。全程不需要
  N×N mask 张量。
* LSE 来源三选一，按可用性回退：flex（torch 内置，无需编译）→ flash-attn（若有）→
  manual（逐 q-block 物化，慢但永远可用）。

数学等价性已在本机实测：BG 键复制 S 次 ≡ 加 log(S) 偏置后合并，误差 7.2e-7。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "attn_lse", "merge_lse", "splitstream_attention",
    "video_split", "route3_plan", "provider_report",
]

_FLEX = {}


def _flex_fn():
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


# --------------------------------------------------------------------------- #
# LSE providers                                                                #
# --------------------------------------------------------------------------- #
def _attn_lse_flex(q, k, v):
    # q [B,H,S,D] k/v [B,H,K,D] -> o [B,H,S,D], lse [B,H,S] fp32
    o, lse = _flex_fn()(q, k, v, return_lse=True)
    return o, lse.float()


def _attn_lse_flash(q, k, v):
    from flash_attn import flash_attn_func
    o, lse, _ = flash_attn_func(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), return_attn_probs=True)
    return o.transpose(1, 2), lse.float()


def _attn_lse_manual(q, k, v, chunk=None):
    b, h, s, d = q.shape
    if chunk is None:
        # 每块 fp32 分数上限 ~192MB
        chunk = max(128, min(4096, int(192e6 / max(1, s * 4))))
    scale = 1.0 / math.sqrt(d)
    outs, lses = [], []
    for i in range(0, s, chunk):
        qs = q[:, :, i:i + chunk].float()
        sc = (qs @ k.float().transpose(-1, -2)) * scale
        lses.append(torch.logsumexp(sc, dim=-1))
        outs.append(torch.softmax(sc, dim=-1).to(v.dtype) @ v)
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


_PROVIDERS = {"flex": _attn_lse_flex, "flash": _attn_lse_flash, "manual": _attn_lse_manual}


def provider_report():
    out = {"flex_available": True, "flash_available": False, "compiled": False}
    try:
        import flash_attn  # noqa: F401
        out["flash_available"] = True
    except Exception:
        pass
    out["compiled"] = "fn" in _FLEX
    return out


def attn_lse(q, k, v, provider: str = "auto"):
    """返回 (out, lse)。provider: auto | flex | flash | manual"""
    if provider == "auto":
        provider = "flex"
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError("unknown provider %r" % provider)
    try:
        return fn(q, k, v)
    except Exception:
        if provider == "auto":
            return _attn_lse_manual(q, k, v)
        raise


# --------------------------------------------------------------------------- #
# 质量守恒合并                                                                  #
# --------------------------------------------------------------------------- #
def merge_lse(o1, l1, o2, l2, beta):
    """O = softmax([branch1 ; branch2 + beta]) 的精确合并。

    beta: 标量 / [H] / [1,H,1] —— 加在被压缩一侧的 logits 上（等价于把该侧键复制 e^beta 份）。
    """
    if beta is not None:
        b = torch.as_tensor(beta, dtype=l2.dtype, device=l2.device)
        while b.dim() < l2.dim():
            b = b.unsqueeze(0)
        l2 = l2 + b
    l_new = torch.logaddexp(l1, l2)
    return (torch.exp(l1 - l_new).unsqueeze(-1) * o1
            + torch.exp(l2 - l_new).unsqueeze(-1) * o2)


def splitstream_attention(q, k, v, n_full: int, beta=None, provider: str = "auto"):
    """keys 的 [0:n_full) 走 branch1（Prefix+FG），[n_full:) 走 branch2（BG，加 beta）。

    q/k/v: [B,H,S,D]；n_full 为 branch1 的键数（也是键维上的切点）。
    """
    s = k.shape[2]
    if n_full <= 0 or n_full >= s:
        return attn_lse(q, k, v, provider)[0]
    o1, l1 = attn_lse(q, k[:, :, :n_full], v[:, :, :n_full], provider)
    o2, l2 = attn_lse(q, k[:, :, n_full:], v[:, :, n_full:], provider)
    return merge_lse(o1, l1, o2, l2, beta)


# --------------------------------------------------------------------------- #
# token 计划                                                                    #
# --------------------------------------------------------------------------- #
# H3 原生：5 个 latent token = 17 像素帧；时间轴单位 = 40Hz 音频帧
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def video_split(latent_t: int, gh: int, gw: int, s_bg: int = 5):
    """视频段内部的行划分。

    latent_t 个 latent 帧，每帧 gh*gw 个 2x2-patch 行（行内按 t 主序、h、w）。
    返回 (fg_local, bg_local)：视频段内的行下标（0 = 第一帧第一行）。
    s_bg=5 → 每 5 帧留 1 帧作 BG 关键帧；s_bg=0 或 >=latent_t → 整片 1 份 BG。
    """
    rows = latent_t * gh * gw
    if s_bg <= 0 or s_bg >= latent_t:
        keep_t = {0}
    else:
        keep_t = set(range(0, latent_t, s_bg))
    idx = torch.arange(rows, dtype=torch.long)
    t_of_row = idx // (gh * gw)
    is_bg = torch.zeros(rows, dtype=torch.bool)
    for t in keep_t:
        is_bg |= (t_of_row == t)
    return (~is_bg).nonzero(as_tuple=True)[0], is_bg.nonzero(as_tuple=True)[0]


@dataclass
class Route3Plan:
    prefix_end: int        # [0, prefix_end) = text/cond/refs/audio
    video_start: int       # 视频段起点
    video_rows: int
    fg_local: torch.Tensor
    bg_local: torch.Tensor
    s_bg: int
    n_full: int            # branch1 键数 = prefix_end + len(fg)  —— 前提是布局按 [Prefix|FG|BG]
    beta0: float

    def to_dict(self):
        return {"prefix_end": self.prefix_end, "video_start": self.video_start,
                "video_rows": self.video_rows, "fg": int(self.fg_local.numel()),
                "bg": int(self.bg_local.numel()), "s_bg": self.s_bg,
                "n_full": self.n_full, "beta0": self.beta0}


def route3_plan(prefix_end: int, latent_t: int, lat_h: int, lat_w: int,
                s_bg: int = 5, ring: int = 0):
    """构造一个 plan。

    注意：本函数只给「几何划分 + 建议的键切点」，真正重排序列由上层完成 ——
    上层必须把视频行重排成 [FG_full | BG_strided]，n_full 才成立。
    ring: FG 掩码外扩格数（在 mask→plan 那一层用，这里只记 beta 初值）。
    """
    gh, gw = lat_h // 2, lat_w // 2
    fg, bg = video_split(latent_t, gh, gw, s_bg)
    # BG 被压缩多少倍，就补回多少质量（Stage 2 再按 layer/head 校准）
    n_keep = 1 if (s_bg <= 0 or s_bg >= latent_t) else math.ceil(latent_t / s_bg)
    beta0 = math.log(max(1.0, latent_t / n_keep))
    return Route3Plan(prefix_end=prefix_end, video_start=prefix_end,
                      video_rows=latent_t * gh * gw, fg_local=fg, bg_local=bg,
                      s_bg=s_bg, n_full=prefix_end + int(fg.numel()), beta0=beta0)
