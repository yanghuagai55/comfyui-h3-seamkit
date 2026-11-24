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
    "bg_anchor_indices", "video_split", "route3_plan", "provider_report",
    "PHASES", "phase_of_rows", "beta_for_sequence", "warmup",
    "reset_provider", "MAX_PHASE_DT",
]

_FLEX = {}
_STATE = {"provider": None, "last_error": None}


# 小显存卡上 autotune 会挑到超 smem 的 config（4060 报 Required 131072 / limit 101376）。
# 第二档强制一个小而稳的 config：实测 4060 与 A100 都能过。
_KO_SAFE = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}


def _flex_fn():
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def _attn_lse_flex(q, k, v):
    """flex + compile。降级链：默认 config → 保守 config → （上层）manual。

    实测（torch 2.14 / 4060）：默认 config 偶尔会挑到 num_warps/stages 过大的方案，
    直接抛 "No valid triton configs ... Shared memory"; 换 _KO_SAFE 即过。
    """
    try:
        o, lse = _flex_fn()(q, k, v, return_lse=True)
        return o, lse.float()
    except Exception as e:
        _STATE["last_error"] = "flex default: " + repr(e)[:160]
        _FLEX.pop("fn", None)
    o, lse = _flex_fn()(q, k, v, return_lse=True, kernel_options=_KO_SAFE)
    return o, lse.float()


def warmup(shapes, dtype=torch.bfloat16, device="cuda"):
    """预编译 flex kernel。Stage 0 前把 (S1,S2) 全跑一遍，避免第一帧卡 6 秒。"""
    import time
    t0 = time.time()
    for s in shapes:
        q = torch.zeros(1, 8, int(s), 128, dtype=dtype, device=device)
        attn_lse(q, q, q, "auto")
    torch.cuda.synchronize()
    return round(time.time() - t0, 1)


def _attn_lse_cudnn(q, k, v):
    """cuDNN FlashAttention，零编译，支持 Q != K，lse 是 fp32 自然对数。

    **实测（本机 torch 2.14.0+cu130 / 4060，Q=256 K=128 H=4 D=128 bf16）**：
    误差 9.5e-7；返回 lse 形状 [B,H,Q,1]（要 squeeze 掉最后一维）；out 可反传（q/k 都有梯度）。
    非连续输入（从 [1,S,H,D] 转置而来）也接受。
    """
    out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q, k, v, None, True, 0.0, False)[:2]
    lse = lse.float().squeeze(-1)
    return out, lse[..., :q.shape[-2]]


def _attn_lse_memeff(q, k, v):
    """MemEff（xFormers 系），零编译，支持 Q != K，lse 已是 [B,H,Q] fp32。"""
    out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, False)[:2]
    # ⚠ 实测：MemEff 会把 lse padding 到 8 的倍数（S=95 → 96，S=17 → 32），
    # 而 out 保持真实长度。不切掉就会和 β 的形状对不上，或者在 S 恰好对齐时静默错位。
    return out, lse.float()[..., :q.shape[-2]]


def _attn_lse_flash(q, k, v):
    from flash_attn import flash_attn_func
    o, lse, _ = flash_attn_func(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), return_attn_probs=True)
    return o.transpose(1, 2), lse.float()[..., :q.shape[-2]]


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


_PROVIDERS = {"cudnn": _attn_lse_cudnn, "memeff": _attn_lse_memeff,
              "flex": _attn_lse_flex, "flash": _attn_lse_flash,
              "manual": _attn_lse_manual}

# 推理默认顺序：零编译的 ATen 融合算子优先，flex 只作兜底
# （cudnn 71.9ms vs flex 68.2ms @8192x8192x56 头，速度基本持平，但省掉 6s 首编与
#   autotune 撞 smem 的风险；需要 lse 可反传时才必须用 flex）
_ORDER = ("cudnn", "memeff", "flash", "flex", "manual")


def provider_report():
    import torch
    out = {"available": [], "compiled": "fn" in _FLEX}
    for n in ("cudnn", "memeff"):
        out["available"].append(n) if hasattr(torch.ops.aten, "_scaled_dot_product_%s_attention"
                                              % ("cudnn" if n == "cudnn" else "efficient")) else None
    try:
        import flash_attn  # noqa: F401
        out["available"].append("flash")
    except Exception:
        pass
    out["available"] += ["flex", "manual"]
    out["flash_available"] = "flash" in out["available"]
    return out


def attn_lse(q, k, v, provider: str = "auto"):
    """返回 (out, lse)。provider: auto | flex | flash | manual

    auto 的顺序：上次成功的 → flash（若有）→ flex(+compile) → manual。
    第一次成功后记住，后续 100 次调用（50 层 × 2 分支）不再重复试探。
    """
    if provider != "auto":
        fn = _PROVIDERS.get(provider)
        if fn is None:
            raise ValueError("unknown provider %r" % provider)
        return fn(q, k, v)

    cached = _STATE["provider"]
    if cached:
        try:
            return _PROVIDERS[cached](q, k, v)
        except Exception as e:
            _STATE["last_error"] = "cached %s: %s" % (cached, repr(e)[:160])
            _STATE["provider"] = None

    order = [p for p in _ORDER if p in provider_report()["available"]]
    for name in order:
        try:
            out = _PROVIDERS[name](q, k, v)
            _STATE["provider"] = name
            if name != "flex":
                print("[SeamKit] Route3: LSE provider -> %s" % name)
            return out
        except Exception as e:
            _STATE["last_error"] = "%s: %s" % (name, repr(e)[:160])
    raise RuntimeError("no LSE provider worked; last=%s" % _STATE["last_error"])


def reset_provider():
    _STATE["provider"] = None
    _STATE["last_error"] = None
    _FLEX.clear()


# --------------------------------------------------------------------------- #
# 质量守恒合并                                                                  #
# --------------------------------------------------------------------------- #
def merge_lse(o1, l1, o2, l2, beta):
    """O = softmax([branch1 ; branch2 + beta]) 的精确合并。

    beta: 标量 / [H] / [1,H,1] / **[1,H,S]（逐 query 相位）** —— 加在被压缩一侧的 logits 上，
          等价于把该侧键复制 e^beta 份。返回时已把 beta 折进 l2，因此调用方只需保证
          「beta 加在 L2 上」这一件事：
              L = logaddexp(L1, L2 + beta)
              O = exp(L1 - L) * O1 + exp(L2 + beta - L) * O2
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
PHASES = len(FRAME_PER_TOKEN)          # 5 个组内相位
# 组内 5 个 token 的起始像素帧 = (0, 1, 5, 9, 13)
_PHASE_START_FRAME = tuple(sum(FRAME_PER_TOKEN[:i]) for i in range(PHASES))
# 最远相位到锚点的距离：13 帧 = 13/24*40 = 21.667 个时间单位
MAX_PHASE_DT = _PHASE_START_FRAME[-1] / 24.0 * 40.0


def phase_of_rows(latent_t: int, gh: int, gw: int):
    """每个视频行所属的组内相位 p = t % 5（0..4）。"""
    idx = torch.arange(latent_t * gh * gw, dtype=torch.long)
    return (idx // (gh * gw)) % PHASES


def beta_for_sequence(beta_lhp, layer: int, phase_full: torch.Tensor):
    """把 [L,H,P] 或 [H,P] 的门控按 query 相位展开成 [1,H,S]，可直接喂 merge_lse。"""
    b = beta_lhp[layer] if beta_lhp.dim() == 3 else beta_lhp
    return b[:, phase_full].unsqueeze(0)





def bg_anchor_indices(latent_t: int, s_bg: int = 3, phase_aware: bool = True,
                      pin_tokens=None) -> list[int]:
    """返回背景锚点的 latent token 下标（升序，唯一真相源：引擎与重装共用）。

    为什么必须做 I/P 相位感知对齐（2026-11-23 绿幕实片 exp_4v10a_00003 实证）：
      H3 3D Video VAE 按每 5 token = 17 帧分组，FRAME_PER_TOKEN = (1, 4, 4, 4, 4)：
      * p = 0 (t % 5 == 0) 是 1 帧独立 I-token（重置因果状态，通道均值/方差与 P-token 迥异）；
      * p ∈ {1,2,3,4}      是 4 帧因果 P-token。
      实测丢弃 p=0 并用邻近 P-token 插值（P→I）会在 f17/34/68/85/119 整圈人物边缘爆出亮绿白光晕，
      而把 I-token 混进邻近 P-token（I→P）会在 f50/52/101/103 产生暗边缘污影。

    对齐策略（latent_t >= 5 且 phase_aware=True）：
      * s_bg <= 0 或 >= latent_t ：[0]（静机位单帧档）
      * s_bg == 1                ：全部帧（0% 压缩等价自检）
      * s_bg >= 2                ：100% 保留全部 p=0 I-token（放入 branch1 零压缩、β_I=0，彻底保住原生 20%:80% I/P 质量守恒，
                                    杜绝 I-key 过权导致 4 帧锯齿明度频闪与 17k 边缘光晕）；
                                    P-pool (p∈{1,2,3,4}) 内部跨相位均匀抽样（如 s_bg=3 走 p_all[::3] 覆盖全部 4 个 P 相位，
                                    最大 P 间距 3~4、含首尾 P 锚零外推，β_P = ln(N_P_dense/N_P_kept) ≈ 0.969）。
      * pin_tokens（可选）        ：硬切边界及其跨镜过渡 token（如 {c-1, c, c+1}）强制钉为全速率锚点，
                                    避免跨镜头 token 被删后在因果 3D VAE 组内引发 17 帧跨镜叠影！
    """
    T = int(latent_t)
    if T <= 0:
        return []
    pins = {int(t) for t in (pin_tokens or ()) if 0 <= int(t) < T}
    if s_bg <= 0 or s_bg >= T:
        return sorted({0} | pins)
    if s_bg == 1:
        return list(range(T))
    if (not phase_aware) or T < PHASES:
        return sorted(set(range(0, T, int(s_bg))) | pins)

    i_all = list(range(0, T, PHASES))
    p_all = [t for t in range(T) if t % PHASES != 0]
    if not p_all:
        return sorted(set(i_all) | pins)

    if s_bg == 2:
        p_keep = p_all[::2]
    elif s_bg == 3:
        # gcd(3, 4) == 1：跨全部 4 个 P 相位 (1,4,3,2) 均匀轮转，T=37 共 11 个 P 锚 (β_P=0.969)，最大间距 3~4！
        p_keep = p_all[::3]
    else:
        # s_bg >= 4：按目标数量在 P 池均匀取点（避免 s_bg=4 与 4 相位整除只命中 p=1）
        n_p = max(2, int(round(len(p_all) / float(s_bg))) + (1 if s_bg == 4 else 0))
        n_p = min(len(p_all), n_p)
        if n_p >= len(p_all):
            p_keep = list(p_all)
        else:
            p_keep = sorted(set(
                p_all[int(round(k * (len(p_all) - 1) / float(max(1, n_p - 1))))]
                for k in range(n_p)
            ))
    if p_keep[-1] != p_all[-1]:
        p_keep = list(p_keep) + [p_all[-1]]
    return sorted(set(i_all) | set(p_keep) | pins)


def video_split(latent_t: int, gh: int, gw: int, s_bg: int = 3, fg_mask=None,
                phase_aware: bool = True, pin_tokens=None):
    """视频段内部的行划分：**必须给前景掩码，否则不会有任何压缩**。

    参数
    ----
    fg_mask : [latent_t, gh*gw] 或 [latent_t*gh*gw] 的 bool。True = 前景，全速率保留。
              **None 等价于「整帧都是前景」→ 一行都删不掉**（这是有意的诚实默认值，
              不做「假装压缩了」的事）。
    s_bg    : 背景锚点压缩档位（3 = 默认推荐：全留 p=0 I-token + 跨 4 个 P 相位均匀抽样）。
    pin_tokens : 强制全速率保留的额外 token 下标（如硬切边界 {c-1, c, c+1}）。

    返回 (fg_local, bg_local)：视频段内的行下标（0 = 第一帧第一行）。
    """
    rows = latent_t * gh * gw
    idx = torch.arange(rows, dtype=torch.long)
    t_of_row = idx // (gh * gw)

    if fg_mask is None:
        is_fg = torch.ones(rows, dtype=torch.bool)
    else:
        m = torch.as_tensor(fg_mask).reshape(-1)
        if m.numel() != rows:
            raise ValueError("fg_mask 大小 %d != %d" % (m.numel(), rows))
        is_fg = m.to(torch.bool).cpu()   # 行号/掩码一律留 CPU，省显存也省同步

    anchors = bg_anchor_indices(latent_t, s_bg, phase_aware=phase_aware, pin_tokens=pin_tokens)
    anc_mask_t = torch.zeros(latent_t, dtype=torch.bool)
    if anchors:
        anc_mask_t[anchors] = True
    is_anchor = anc_mask_t[t_of_row]
    keep = is_fg | is_anchor
    fg = (keep & is_fg).nonzero(as_tuple=True)[0]
    bg = (keep & ~is_fg).nonzero(as_tuple=True)[0]
    return fg, bg


@dataclass
class Route3Plan:
    prefix_end: int        # [0, prefix_end) = text/cond/refs/audio
    video_start: int       # 视频段起点
    video_rows: int
    fg_local: torch.Tensor
    bg_local: torch.Tensor
    s_bg: int
    n_full: int            # branch1 键数 = prefix_end + len(fg) + 未压缩 I-BG/钉住硬切 P-BG
    beta0: float
    phase: Optional[torch.Tensor] = None   # 每个视频行的组内相位 0..4
    anchors: Optional[list[int]] = None    # 全部完整保留的锚点 token 下标（含硬切钉住点）

    def to_dict(self):
        return {"prefix_end": self.prefix_end, "video_start": self.video_start,
                "video_rows": self.video_rows, "fg": int(self.fg_local.numel()),
                "bg": int(self.bg_local.numel()), "s_bg": self.s_bg,
                "n_full": self.n_full, "beta0": round(self.beta0, 4),
                "phases": PHASES, "max_phase_dt": round(MAX_PHASE_DT, 3),
                "n_anchors": len(self.anchors) if self.anchors is not None else 0}

    def full_phase(self):
        """整条序列的相位（prefix 段记 0）。顺序与 [Prefix|FG|BG] 一致，供 beta_for_sequence 用。"""
        s = self.prefix_end + int(self.fg_local.numel()) + int(self.bg_local.numel())
        out = torch.zeros(s, dtype=torch.long)
        kept = torch.cat([self.phase[self.fg_local], self.phase[self.bg_local]])
        out[self.prefix_end:self.prefix_end + kept.numel()] = kept
        return out


def route3_plan(prefix_end: int, latent_t: int, lat_h: int, lat_w: int,
                s_bg: int = 3, ring: int = 0, fg_mask=None, phase_aware: bool = True,
                pin_tokens=None):
    """构造一个 plan（严格分池质量守恒 + 硬切边界 token 全速率钉住）。

    关键修正（2026-11-23）：
      * 未压缩的 p=0 I-token 与硬切边界额外钉住的 P-token（extra_p_pins）归入 branch1 (β=0)；
      * 仅被抽帧压缩的常规 P-token 留在 branch2，且 β_P = ln((N_P_dense - N_P_pin) / N_P_b2)，
        精确保持原生 I : P = 20% : 80% 与总 P 质量守恒，零过权！
    """
    gh, gw = lat_h // 2, lat_w // 2
    p_per_frame = gh * gw
    fg, bg = video_split(latent_t, gh, gw, s_bg, fg_mask=fg_mask,
                         phase_aware=phase_aware, pin_tokens=pin_tokens)
    base_anchors = bg_anchor_indices(latent_t, s_bg, phase_aware=phase_aware, pin_tokens=None)
    anchors = bg_anchor_indices(latent_t, s_bg, phase_aware=phase_aware, pin_tokens=pin_tokens)
    i_all = list(range(0, latent_t, PHASES))
    p_all = [t for t in range(latent_t) if t % PHASES != 0]
    all_i_kept = (phase_aware and latent_t >= PHASES and s_bg > 1 and s_bg < latent_t
                  and all(i in anchors for i in i_all) and len(p_all) > 0)

    if s_bg == 1 or fg_mask is None:
        beta0 = 0.0
        n_full = prefix_end + int(fg.numel())
    elif all_i_kept:
        p_kept = [t for t in anchors if t % PHASES != 0]
        p_kept_set = set(p_kept)
        # 局部 Voronoi 质量守恒：只有紧邻至少一个「被丢弃 P-token」的 P-锚点才真正代表多帧压缩区间，
        # 放入 branch2 乘 β_P；而在局部全稠密区（左右相邻 P-token 均被保留，如高速运镜钉住段 Shot 1、
        # 硬切钉住点以及紧贴 p_all[-2]=34 的尾帧 36）的 P-token 真实权重就是 1.0，必须归入 branch1 (β=0)，
        # 彻底消灭同镜头内相邻 token 间 1.8x vs 1.0x 权重锯齿振荡以及跨镜头总质量偏斜！
        b2_p_set = set()
        for idx, t in enumerate(p_all):
            if t not in p_kept_set:
                continue
            left_drop = (idx > 0) and (p_all[idx - 1] not in p_kept_set)
            right_drop = (idx + 1 < len(p_all)) and (p_all[idx + 1] not in p_kept_set)
            if left_drop or right_drop:
                b2_p_set.add(t)
        b1_p_set = p_kept_set - b2_p_set
        bg_t = bg // p_per_frame
        is_full_bg = (bg_t % PHASES) == 0
        if b1_p_set:
            b1_mask_t = torch.zeros(latent_t, dtype=torch.bool)
            for pt in b1_p_set:
                b1_mask_t[pt] = True
            is_full_bg = is_full_bg | b1_mask_t[bg_t]
        bg_full = bg[is_full_bg]
        bg_comp = bg[~is_full_bg]
        bg = torch.cat([bg_full, bg_comp])
        n_p_b2 = max(1, len(b2_p_set))
        n_p_rem = max(n_p_b2, len(p_all) - len(b1_p_set))
        beta0 = math.log(max(1.0, float(n_p_rem) / float(n_p_b2))) if bg_comp.numel() > 0 else 0.0
        n_full = prefix_end + int(fg.numel()) + int(bg_full.numel())
    else:
        n_keep = max(1, len(anchors))
        beta0 = math.log(max(1.0, float(latent_t) / float(n_keep)))
        n_full = prefix_end + int(fg.numel())
    return Route3Plan(prefix_end=prefix_end, video_start=prefix_end,
                      video_rows=latent_t * p_per_frame, fg_local=fg, bg_local=bg,
                      s_bg=s_bg, n_full=n_full, beta0=beta0,
                      phase=phase_of_rows(latent_t, gh, gw), anchors=anchors)
