# -*- coding: utf-8 -*-
"""Route 3 · Stage 0 探针：H3 的注意力跨 13 帧还认不认得自己的背景？

要回答的问题
------------
组内 5 个相位的起始像素帧是 (0,1,5,9,13)，最远相位到锚点是 **13 帧 = 21.667 个时间单位**
（时间轴单位 = 40Hz 音频帧）。S_bg=5 就是把中间帧的背景换成 13 帧前的那份。

如果某一层的注意力几乎只在 Δt≈0 上有效，那么给它喂 13 帧前的背景键，等于喂了错的相位——
β 补得回「质量」，补不回「相位」。

做法
----
1. 正常跑一遍稠密 forward；
2. 每层在 RoPE 之后截获 q/k，抽样 query 行（每个相位 n_query 行）；
3. 对全部视频键算 fp32 logits → softmax → 按 Δt 分桶统计质量；
4. 输出每 (layer, head) 的桶分布 + 「时间头分数」T = mass(Δt≥8.33) / (mass(Δt≈0)+ε)。

判定（写进报告，由人拍板；脚本只给数）
--------------------------------------
T 中位数 ≥ 0.3 → 跨帧锚定有效，继续 S_bg=5
0.1–0.3        → 降到 S_bg=3
< 0.1 且 Δt=0 质量 > 0.7 → 止损，转路线 A
"""

from __future__ import annotations

import json
import types

import torch

__all__ = ["probe_attention", "summarize", "BUCKETS"]

# 桶边界（40Hz 时间单位）：0 / 1 帧 / 5 帧 / 9 帧 / 13 帧
BUCKETS = (0.0, 1.667, 8.333, 15.0, 21.667)


def _bucketize(dt):
    out = torch.zeros_like(dt, dtype=torch.long)
    for i, b in enumerate(BUCKETS[1:], start=1):
        out = out + (dt >= b - 1e-6).long()
    return out.clamp(max=len(BUCKETS) - 1)


def probe_attention(model, x, timestep, context, transformer_options=None,
                    minimax_payload=None, s_bg=5, n_query=8, layers=None,
                    max_video_keys=None):
    """返回 {"layers": [...], "buckets": BUCKETS, "s_bg": s_bg}"""
    from comfy.ldm.minimax.model import PackedLayout
    import comfy.ldm.common_dit
    from route3.route3_block import rms_rope_split_half_ref

    transformer_options = dict(transformer_options or {})
    payload = dict(minimax_payload or {})
    payload.pop("layout", None)

    video_x, audio_x = x[0], x[1]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    layout = PackedLayout(context.shape[1], latent_t, lat_h, lat_w, audio_x.shape[-1],
                          keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    t_of_token = layout.position_ids[:, 0].to(torch.float64)      # 时间坐标（40Hz 单位）
    v_t = t_of_token[va:vb]
    gh, gw = lat_h // 2, lat_w // 2
    t_index = torch.arange(vb - va, dtype=torch.long) // (gh * gw)
    phase = t_index % 5

    # 每个相位抽 n_query 行
    sel = []
    for p in range(5):
        idx = (phase == p).nonzero(as_tuple=True)[0]
        if idx.numel():
            step = max(1, idx.numel() // n_query)
            sel.append(idx[::step][:n_query])
    sel = torch.cat(sel)
    dt = (v_t[sel][:, None] - v_t[None, :]).abs()                 # [nq, S_v]
    bk = _bucketize(dt)

    query_dtype = context.dtype
    rec = {}

    def _wrap(attn, li):
        orig = attn.forward

        def fwd(self, xx, rope_freqs=None, transformer_options={}, **kw):
            out = orig(xx, rope_freqs=rope_freqs, transformer_options=transformer_options, **kw)
            if li < 0:
                return out
            try:
                with torch.no_grad():
                    s = xx.shape[0]
                    q, k, _ = self.qkv_proj(xx.detach()).split(self.heads * self.head_dim, dim=-1)
                    q = q.view(1, s, self.heads, self.head_dim)
                    k = k.view(1, s, self.heads, self.head_dim)
                    if rope_freqs is not None:
                        import comfy.model_management
                        qw = comfy.model_management.cast_to(self.q_norm.weight, device=xx.device)
                        kw = comfy.model_management.cast_to(self.k_norm.weight, device=xx.device)
                        q, k = rms_rope_split_half_ref(q, k, rope_freqs, qw, kw,
                                                       epsilon=self.q_norm.eps,
                                                       rot_dim=rope_freqs.shape[-3] * 2)
                    q = q[0][sel.to(xx.device)]                    # [nq,H,D]
                    k = k[0][va:vb]                                # [S_v,H,D]
                    if max_video_keys and k.shape[0] > max_video_keys:
                        kk = k[:max_video_keys]
                    else:
                        kk = k
                    logits = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) / (self.head_dim ** 0.5)
                    mass = torch.softmax(logits, dim=-1)
                    b = bk.to(mass.device)[:, :kk.shape[0]]
                    acc = torch.zeros(self.heads, len(BUCKETS), device=mass.device)
                    for i in range(len(BUCKETS)):
                        m = (b == i).unsqueeze(0).expand_as(mass)
                        acc[:, i] = (mass * m).sum(-1).mean(-1)
                    rec[li] = acc.cpu()
            except Exception as e:      # 探针失败绝不能拖垮 forward
                rec[li] = ("error", repr(e)[:120])
            return out

        attn.forward = types.MethodType(fwd, attn)

    want = set(layers) if layers is not None else set(range(len(model.blocks)))
    for i, blk in enumerate(model.blocks):
        _wrap(blk.attn, i if i in want else -1)
    try:
        with torch.no_grad():
            model([video_x, audio_x], timestep, context, transformer_options,
                  minimax_payload=payload)
    finally:
        pass

    return {"buckets": list(BUCKETS), "s_bg": s_bg, "n_query": int(sel.numel()),
            "layers": [{"layer": i,
                        "heads": (rec[i].tolist() if isinstance(rec[i], torch.Tensor) else None),
                        "error": (rec[i][1] if isinstance(rec[i], tuple) else None)}
                       for i in sorted(rec)]}


def summarize(probe, temporal_from=2):
    """每层中位『时间头分数』；temporal_from=2 对应 Δt≥8.333。"""
    out = []
    for L in probe["layers"]:
        h = L.get("heads")
        if not h:
            continue
        t = torch.tensor(h)                       # [H, B]
        near = t[:, 0].clamp(min=1e-6)
        far = t[:, temporal_from:].sum(-1)
        score = far / near
        out.append({"layer": L["layer"],
                    "median_temporal": round(float(score.median()), 4),
                    "max_temporal": round(float(score.max()), 4),
                    "near_mass": round(float(t[:, 0].mean()), 4)})
    med = sorted(x["median_temporal"] for x in out)
    return {"per_layer": out,
            "overall_median": med[len(med) // 2] if med else 0.0,
            "verdict": _verdict(med[len(med) // 2] if med else 0.0)}


def _verdict(m):
    if m >= 0.3:
        return "continue: S_bg=5 可用"
    if m >= 0.1:
        return "downgrade: 改 S_bg=3"
    return "abort: 转路线 A"


def dump(probe, summary, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"probe": probe, "summary": summary}, f, ensure_ascii=False, indent=2)
    return path
