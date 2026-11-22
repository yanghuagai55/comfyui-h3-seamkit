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
                print("[route3] LSE provider -> %s" % name)
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





def video_split(latent_t: int, gh: int, gw: int, s_bg: int = 5, fg_mask=None):
    """视频段内部的行划分：**必须给前景掩码，否则不会有任何压缩**。

    参数
    ----
    fg_mask : [latent_t, gh*gw] 或 [latent_t*gh*gw] 的 bool。True = 前景，全速率保留。
              **None 等价于「整帧都是前景」→ 一行都删不掉**（这是有意的诚实默认值，
              不做「假装压缩了」的事）。
    s_bg    : 每隔几个 latent token 留一个 BG 锚点帧（H3 里 5 token = 17 像素帧）。

    规则
    ----
    前景行：全部帧保留；
    背景行：只在锚点帧（t % s_bg == 0）保留。

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

    is_anchor = (t_of_row % max(1, s_bg) == 0)
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
    n_full: int            # branch1 键数 = prefix_end + len(fg)  —— 前提是布局按 [Prefix|FG|BG]
    beta0: float
    phase: Optional[torch.Tensor] = None   # 每个视频行的组内相位 0..4

    def to_dict(self):
        return {"prefix_end": self.prefix_end, "video_start": self.video_start,
                "video_rows": self.video_rows, "fg": int(self.fg_local.numel()),
                "bg": int(self.bg_local.numel()), "s_bg": self.s_bg,
                "n_full": self.n_full, "beta0": round(self.beta0, 4),
                "phases": PHASES, "max_phase_dt": round(MAX_PHASE_DT, 3)}

    def full_phase(self):
        """整条序列的相位（prefix 段记 0）。顺序与 [Prefix|FG|BG] 一致，供 beta_for_sequence 用。"""
        s = self.n_full + int(self.bg_local.numel())
        out = torch.zeros(s, dtype=torch.long)
        kept = torch.cat([self.phase[self.fg_local], self.phase[self.bg_local]])
        out[self.prefix_end:self.prefix_end + kept.numel()] = kept
        return out


def route3_plan(prefix_end: int, latent_t: int, lat_h: int, lat_w: int,
                s_bg: int = 5, ring: int = 0, fg_mask=None):
    """构造一个 plan。

    注意：本函数只给「几何划分 + 建议的键切点」，真正重排序列由上层完成 ——
    上层必须把视频行重排成 [FG_full | BG_strided]，n_full 才成立。
    ring: FG 掩码外扩格数（在 mask→plan 那一层用，这里只记 beta 初值）。
    """
    gh, gw = lat_h // 2, lat_w // 2
    fg, bg = video_split(latent_t, gh, gw, s_bg, fg_mask=fg_mask)
    # BG 被压缩多少倍，就补回多少质量（Stage 2 再按 layer/head/相位 校准）
    n_keep = 1 if (s_bg <= 0 or s_bg >= latent_t) else math.ceil(latent_t / s_bg)
    beta0 = math.log(max(1.0, latent_t / n_keep)) if (s_bg > 1 and fg_mask is not None) else 0.0
    return Route3Plan(prefix_end=prefix_end, video_start=prefix_end,
                      video_rows=latent_t * gh * gw, fg_local=fg, bg_local=bg,
                      s_bg=s_bg, n_full=prefix_end + int(fg.numel()), beta0=beta0,
                      phase=phase_of_rows(latent_t, gh, gw))
