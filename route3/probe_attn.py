# -*- coding: utf-8 -*-
"""Route 3 · Stage 0 探针（升级版）：直接输出替换误差 + 双口径 Δt 衰减分布

核心指标（无拍脑袋阈值，直接量“删掉中间帧 BG 后，注入残差流的真实扰动”）
----------------------------------------------------------------------------
对每层抽样的视频 query（覆盖 5 个相位 p=0..4，含最恶劣 j=3/j=4）：
  * `rel_err`       = ||O_approx - O_true||_2 / ||O_true||_2（整体残差流相对 L2 误差）
  * `bg_cos`        = cos_sim(O_bg_approx, O_bg_true)（背景分支输出方向保真度；方向错调 β 救不回）
  * `bg_mass_ratio` = mass_bg_approx / mass_bg_true（背景 Softmax 质量比；只偏质量调 β[L,H,5] 能修）
  * 双口径 Δt 分桶（辅助诊断）：
      - `nearest`：到最近锚点距离（像素帧 0 / 1 / 4 / 5 / 8，最大 Δt=13.33）
      - `prev`   ：到前一个锚点距离（像素帧 0 / 1 / 5 / 9 / 13，最大 Δt=21.67）

判定门禁
--------
* 全层中位 `rel_err <= 0.08`                        → continue: S_bg=5 免训练稳过
* `0.08 < rel_err <= 0.18` 且 `bg_cos >= 0.90`      → calibrate: 进 Stage 2 校准相位 β[L,H,5]
* `rel_err > 0.20` 或 `bg_cos < 0.85`               → downgrade_or_abort: 降 S_bg=3 或转路线 A
"""

from __future__ import annotations

import json
import math
import types

import torch
import torch.nn.functional as F

from .route3_block import rms_rope_split_half_ref

__all__ = ["probe_attention", "summarize", "dump", "BUCKETS_PREV", "BUCKETS_NEAREST", "BUCKETS"]

# 40Hz 时间单位桶边界：
# prev（到前锚点）：0 / 1f / 5f / 9f / 13f
BUCKETS_PREV = (0.0, 1.667, 8.333, 15.0, 21.667)
# nearest（到最近前后锚点）：0 / 1f / 4f / 5f / 8f
BUCKETS_NEAREST = (0.0, 1.667, 6.667, 8.333, 13.333)
BUCKETS = BUCKETS_PREV


def _bucketize(dt, buckets):
    out = torch.zeros_like(dt, dtype=torch.long)
    for i, b in enumerate(buckets[1:], start=1):
        out = out + (dt >= b - 1e-6).long()
    return out.clamp(max=len(buckets) - 1)


def probe_attention(model, x, timestep, context, transformer_options=None,
                    minimax_payload=None, s_bg=5, n_query=8, layers=None,
                    max_video_keys=None, head_chunk=8, fg_mask=None):
    """跑一次稠密 forward，逐层量真实 Route 3 双流替换误差 + 双口径 Δt 分布。

    关键口径（本机真机冒烟修正）：
    Route 3 只评估 `FG_full ∪ BG_anchors` 中的 query（没有任何被保留的 query 会丢失自己所在的同帧
    FG 邻域！）。探针按 `[Prefix | FG_full]`（Branch 1）+ `[BG_anchors] + beta0`（Branch 2）
    精确模拟 Route 3 双流输出并与稠密全量输出对比；未提供 `fg_mask` 时用画面中心 25% 作标准前景代理。

    保证在 finally 中完整恢复所有 `blk.attn.forward`，绝不泄漏 hook。
    """
    from comfy.ldm.minimax.model import PackedLayout
    import comfy.ldm.common_dit
    import comfy.model_management

    transformer_options = dict(transformer_options or {})
    payload = dict(minimax_payload or {})
    payload.pop("layout", None)

    video_x, audio_x = x[0], x[1]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    layout = PackedLayout(context.shape[1], latent_t, lat_h, lat_w, audio_x.shape[-1],
                          keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    s_v = vb - va
    t_of_token = layout.position_ids[:, 0].to(torch.float64)
    v_t = t_of_token[va:vb]
    gh, gw = lat_h // 2, lat_w // 2
    p_per_frame = gh * gw
    t_index = torch.arange(s_v, dtype=torch.long) // p_per_frame
    phase = t_index % 5

    # 前景掩码（未给时用中心 25% 盒作代理，避免把被保留 FG query 自己的同帧删掉造成假警报）
    if fg_mask is None:
        fm_2d = torch.zeros(gh, gw, dtype=torch.bool)
        h0, h1 = gh // 4, max(gh // 4 + 1, (3 * gh) // 4)
        w0, w1 = gw // 4, max(gw // 4 + 1, (3 * gw) // 4)
        fm_2d[h0:h1, w0:w1] = True
        fg_bool = fm_2d.reshape(1, p_per_frame).expand(latent_t, p_per_frame).reshape(-1)
    else:
        fg_bool = torch.as_tensor(fg_mask, dtype=torch.bool).reshape(-1).cpu()
        if fg_bool.numel() != s_v:
            fg_bool = fg_bool[:s_v]

    # 锚点帧掩码：t % s_bg == 0（s_bg<=0 则只留第 0 帧）
    if s_bg <= 0 or s_bg >= latent_t:
        anchor_mask = (t_index == 0)
        n_anchors = 1
    else:
        anchor_mask = (t_index % s_bg == 0)
        n_anchors = math.ceil(latent_t / s_bg)
    fg_local = fg_bool.nonzero(as_tuple=True)[0]
    bg_all_bool = ~fg_bool
    bg_all_local = bg_all_bool.nonzero(as_tuple=True)[0]
    bg_anc_local = (bg_all_bool & anchor_mask).nonzero(as_tuple=True)[0]
    anchor_local = anchor_mask.nonzero(as_tuple=True)[0]
    kept_query_mask = fg_bool | (bg_all_bool & anchor_mask)
    beta0 = math.log(max(1.0, latent_t / max(1, n_anchors)))

    # 每个相位从 Route 3 真实保留的 query（FG_full ∪ BG_anchors）里抽 n_query 行
    sel = []
    for p in range(5):
        idx = ((phase == p) & kept_query_mask).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            idx = (phase == p).nonzero(as_tuple=True)[0]
        if idx.numel():
            step = max(1, idx.numel() // n_query)
            sel.append(idx[::step][:n_query])
    sel = torch.cat(sel)
    sel_global = va + sel

    # 两种 Δt 分桶（视频 query vs 全部视频 key）
    dt_abs = (v_t[sel][:, None] - v_t[None, :]).abs()              # [nq, S_v]
    bk_prev = _bucketize(dt_abs, BUCKETS_PREV)
    # 到最近锚点（前或后）的距离：对每个 query 算其与全部锚点时间差的最小值
    anchor_t_vals = v_t[anchor_local].unique()
    dt_to_nearest_anchor = (v_t[sel][:, None] - anchor_t_vals[None, :]).abs().amin(dim=-1)  # [nq]
    bk_nearest = _bucketize(dt_abs, BUCKETS_NEAREST)

    rec = {}
    originals = []

    def _wrap(attn, li):
        orig = attn.forward
        originals.append((attn, orig))

        def fwd(self, xx, rope_freqs=None, transformer_options={}, **kwargs):
            # 兼容 KJNodes MiniMaxLowVRAMAttention：先拿 tensor 引用，防止 orig pop 清空 list
            x_tensor = xx[0] if isinstance(xx, list) else xx
            x_snap = x_tensor.detach() if li >= 0 else None
            out = orig(xx, rope_freqs=rope_freqs, transformer_options=transformer_options, **kwargs)
            if li < 0 or x_snap is None:
                return out
            try:
                with torch.no_grad():
                    s = x_snap.shape[0]
                    dev = x_snap.device
                    q, k, v = self.qkv_proj(x_snap).split(self.heads * self.head_dim, dim=-1)
                    v = v.view(s, self.heads, self.head_dim)
                    if rope_freqs is not None:
                        q = q.view(1, s, self.heads, self.head_dim)
                        k = k.view(1, s, self.heads, self.head_dim)
                        qw = comfy.model_management.cast_to(self.q_norm.weight, device=dev)
                        kw = comfy.model_management.cast_to(self.k_norm.weight, device=dev)
                        rot = rope_freqs.shape[-3] * 2
                        try:
                            from comfy import quant_ops as _qops
                            _qops.ck.rms_rope_split_half_(
                                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                        except Exception:
                            q, k = rms_rope_split_half_ref(q, k, rope_freqs, qw, kw,
                                                           epsilon=self.q_norm.eps, rot_dim=rot)
                        q, k = q[0], k[0]
                    else:
                        q = self.q_norm(q.view(s, self.heads, self.head_dim))
                        k = self.k_norm(k.view(s, self.heads, self.head_dim))

                    # ⚠ 必须用 sel_global = va + sel，确保抽样的是视频 query 而非 text/refs！
                    q_sel = q[sel_global.to(dev)]                  # [nq, H, D]
                    k_pref, v_pref = k[:va], v[:va]                # [va, H, D]
                    k_vid, v_vid = k[va:vb], v[va:vb]              # [S_v, H, D]
                    if max_video_keys and k_vid.shape[0] > max_video_keys:
                        k_vid = k_vid[:max_video_keys]
                        v_vid = v_vid[:max_video_keys]
                        fg_idx = fg_local[fg_local < max_video_keys].to(dev)
                        bg_all_idx = bg_all_local[bg_all_local < max_video_keys].to(dev)
                        bg_anc_idx = bg_anc_local[bg_anc_local < max_video_keys].to(dev)
                    else:
                        fg_idx = fg_local.to(dev)
                        bg_all_idx = bg_all_local.to(dev)
                        bg_anc_idx = bg_anc_local.to(dev)

                    bp = bk_prev.to(dev)[:, :k_vid.shape[0]]
                    bn = bk_nearest.to(dev)[:, :k_vid.shape[0]]
                    scale = self.head_dim ** -0.5

                    acc_prev = torch.zeros(self.heads, len(BUCKETS_PREV), device=dev)
                    acc_near = torch.zeros(self.heads, len(BUCKETS_NEAREST), device=dev)
                    rel_errs = torch.zeros(self.heads, device=dev)
                    bg_coss = torch.zeros(self.heads, device=dev)
                    bg_ratios = torch.zeros(self.heads, device=dev)

                    # 分头计算，8GB 显卡上峰值仅 ~40MB
                    hc = max(1, int(head_chunk))
                    for h0 in range(0, self.heads, hc):
                        h1 = min(self.heads, h0 + hc)
                        qh = q_sel[:, h0:h1].float()               # [nq, hc, D]
                        kp = k_pref[:, h0:h1].float()
                        vp = v_pref[:, h0:h1].float()
                        kv = k_vid[:, h0:h1].float()
                        vv = v_vid[:, h0:h1].float()

                        # 1) 全量真实输出 O_true = Attn(q, [prefix | FG_full | BG_all])
                        lp = torch.einsum("qhd,khd->hqk", qh, kp) * scale if va > 0 else None
                        lv = torch.einsum("qhd,khd->hqk", qh, kv) * scale
                        l_vid_true = torch.logsumexp(lv, dim=-1)   # [hc, nq]
                        w_vid_only = torch.softmax(lv, dim=-1)
                        o_vid_true = torch.einsum("hqk,khd->hqd", w_vid_only, vv)

                        if lp is not None:
                            l_pref = torch.logsumexp(lp, dim=-1)
                            o_pref = torch.einsum("hqk,khd->hqd", torch.softmax(lp, dim=-1), vp)
                            l_all_true = torch.logaddexp(l_pref, l_vid_true)
                            w_p = torch.exp(l_pref - l_all_true)
                            w_v = torch.exp(l_vid_true - l_all_true)
                            o_true = w_p.unsqueeze(-1) * o_pref + w_v.unsqueeze(-1) * o_vid_true
                        else:
                            l_all_true = l_vid_true
                            o_true = o_vid_true

                        # 真实 BG 子分支（用于精确量 bg_cos 与 bg_mass_ratio）
                        if bg_all_idx.numel() > 0:
                            lb_true = lv[:, :, bg_all_idx]
                            l_bg_true = torch.logsumexp(lb_true, dim=-1)
                            o_bg_true = torch.einsum("hqk,khd->hqd", torch.softmax(lb_true, dim=-1), vv[bg_all_idx])
                            mass_bg_true = torch.exp(l_bg_true - l_all_true)
                        else:
                            o_bg_true = o_vid_true
                            mass_bg_true = torch.ones_like(l_vid_true)

                        # 2) Route 3 真实双流替换输出：
                        #    Branch 1 = [Prefix | FG_full]（全速率保留），Branch 2 = [BG_anchors] + beta0
                        lf = lv[:, :, fg_idx] if fg_idx.numel() > 0 else None
                        if lp is not None and lf is not None:
                            l_fg = torch.logsumexp(lf, dim=-1)
                            o_fg = torch.einsum("hqk,khd->hqd", torch.softmax(lf, dim=-1), vv[fg_idx])
                            l_b1 = torch.logaddexp(l_pref, l_fg)
                            o_b1 = (torch.exp(l_pref - l_b1).unsqueeze(-1) * o_pref
                                    + torch.exp(l_fg - l_b1).unsqueeze(-1) * o_fg)
                        elif lf is not None:
                            l_b1 = torch.logsumexp(lf, dim=-1)
                            o_b1 = torch.einsum("hqk,khd->hqd", torch.softmax(lf, dim=-1), vv[fg_idx])
                        elif lp is not None:
                            l_b1, o_b1 = l_pref, o_pref
                        else:
                            l_b1, o_b1 = None, None

                        la = lv[:, :, bg_anc_idx]
                        l_anc = torch.logsumexp(la, dim=-1)
                        o_anc = torch.einsum("hqk,khd->hqd", torch.softmax(la, dim=-1), vv[bg_anc_idx])
                        l_anc_eff = l_anc + beta0

                        if l_b1 is not None:
                            l_all_app = torch.logaddexp(l_b1, l_anc_eff)
                            w_1 = torch.exp(l_b1 - l_all_app)
                            w_2 = torch.exp(l_anc_eff - l_all_app)
                            o_approx = w_1.unsqueeze(-1) * o_b1 + w_2.unsqueeze(-1) * o_anc
                            mass_bg_app = w_2
                        else:
                            o_approx = o_anc
                            mass_bg_app = torch.ones_like(l_anc_eff)

                        # 汇总核心物理指标
                        diff_norm = (o_approx - o_true).flatten(1).norm(dim=-1)
                        true_norm = o_true.flatten(1).norm(dim=-1).clamp(min=1e-6)
                        rel_errs[h0:h1] = diff_norm / true_norm
                        bg_coss[h0:h1] = F.cosine_similarity(o_anc, o_bg_true, dim=-1).mean(dim=-1)
                        bg_ratios[h0:h1] = (mass_bg_app / mass_bg_true.clamp(min=1e-6)).mean(dim=-1)

                        # 3) 双口径 Δt 桶分布（诊断用）
                        for bi in range(len(BUCKETS_PREV)):
                            m = (bp == bi).unsqueeze(0).expand_as(w_vid_only)
                            acc_prev[h0:h1, bi] = (w_vid_only * m).sum(-1).mean(-1)
                        for bi in range(len(BUCKETS_NEAREST)):
                            m = (bn == bi).unsqueeze(0).expand_as(w_vid_only)
                            acc_near[h0:h1, bi] = (w_vid_only * m).sum(-1).mean(-1)

                    rec[li] = {
                        "heads_prev": acc_prev.cpu(),
                        "heads_nearest": acc_near.cpu(),
                        "rel_err": rel_errs.cpu(),
                        "bg_cos": bg_coss.cpu(),
                        "bg_mass_ratio": bg_ratios.cpu(),
                    }
            except Exception as e:
                rec[li] = ("error", repr(e)[:180])
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
        # 必须恢复所有原始 attn.forward，绝不泄漏 hook
        for attn, orig in originals:
            attn.forward = orig

    layer_rows = []
    for i in sorted(rec):
        val = rec[i]
        if isinstance(val, tuple):
            layer_rows.append({"layer": i, "heads": None, "error": val[1]})
        else:
            layer_rows.append({
                "layer": i,
                "heads": val["heads_prev"].tolist(),
                "heads_nearest": val["heads_nearest"].tolist(),
                "rel_err": [round(float(x), 5) for x in val["rel_err"]],
                "bg_cos": [round(float(x), 5) for x in val["bg_cos"]],
                "bg_mass_ratio": [round(float(x), 4) for x in val["bg_mass_ratio"]],
            })

    return {
        "buckets": list(BUCKETS_PREV),
        "buckets_nearest": list(BUCKETS_NEAREST),
        "s_bg": s_bg,
        "beta0": round(beta0, 4),
        "n_query": int(sel.numel()),
        "max_nearest_anchor_dt": round(float(dt_to_nearest_anchor.max()), 3),
        "layers": layer_rows,
    }


def summarize(probe, temporal_from=2):
    """汇总 50 层的直接输出替换误差（主判据）+ 双口径 Δt 统计（诊断）。"""
    per_layer = []
    for L in probe["layers"]:
        h = L.get("heads")
        if not h:
            continue
        t_prev = torch.tensor(h)
        near = t_prev[:, 0].clamp(min=1e-6)
        far = t_prev[:, temporal_from:].sum(-1)
        score = far / near
        row = {
            "layer": L["layer"],
            "median_temporal": round(float(score.median()), 4),
            "near_mass": round(float(t_prev[:, 0].mean()), 4),
        }
        if L.get("rel_err") is not None:
            re_t = torch.tensor(L["rel_err"])
            cos_t = torch.tensor(L["bg_cos"])
            rat_t = torch.tensor(L["bg_mass_ratio"])
            row.update({
                "rel_err_median": round(float(re_t.median()), 4),
                "rel_err_p90": round(float(re_t.quantile(0.9)), 4),
                "bg_cos_median": round(float(cos_t.median()), 4),
                "bg_mass_ratio_median": round(float(rat_t.median()), 4),
            })
        per_layer.append(row)

    def _med(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[len(s) // 2]

    t_med = _med([x["median_temporal"] for x in per_layer])
    re_vals = [x["rel_err_median"] for x in per_layer if "rel_err_median" in x]
    cos_vals = [x["bg_cos_median"] for x in per_layer if "bg_cos_median" in x]
    rat_vals = [x["bg_mass_ratio_median"] for x in per_layer if "bg_mass_ratio_median" in x]

    re_med = _med(re_vals)
    cos_med = _med(cos_vals)
    rat_med = _med(rat_vals)

    # 分段 U 型层画像（早期 L0-L4 / 中段 / 末段最后 5 层），用于定位是否只需保留末层稠密
    n_tot = max((x["layer"] for x in per_layer), default=-1) + 1
    early = [x for x in per_layer if x["layer"] < min(5, n_tot)]
    late = [x for x in per_layer if x["layer"] >= max(0, n_tot - 5)]
    mid = [x for x in per_layer if 5 <= x["layer"] < max(5, n_tot - 5)]
    layer_bands = {
        "early_rel_err": _med([x["rel_err_median"] for x in early if "rel_err_median" in x]),
        "mid_rel_err": _med([x["rel_err_median"] for x in mid if "rel_err_median" in x]),
        "late_rel_err": _med([x["rel_err_median"] for x in late if "rel_err_median" in x]),
        "early_bg_cos": _med([x["bg_cos_median"] for x in early if "bg_cos_median" in x]),
        "mid_bg_cos": _med([x["bg_cos_median"] for x in mid if "bg_cos_median" in x]),
        "late_bg_cos": _med([x["bg_cos_median"] for x in late if "bg_cos_median" in x]),
    }

    return {
        "per_layer": per_layer,
        "layer_bands": layer_bands,
        "overall_median": t_med,
        "rel_err_median": re_med,
        "bg_cos_median": cos_med,
        "bg_mass_ratio_median": rat_med,
        "verdict": _verdict(re_med, cos_med, t_med, has_direct=bool(re_vals)),
    }


def _verdict(rel_err, bg_cos, t_med, has_direct=True):
    """诊断排序器结论（仅作归因/排优参考，**不作为盲目硬止损开关**；
    真正止损由 Stage 1a 1.5 分钟金丝雀成片实测 bg_hf_sharpness / fg_psnr 决定）。
    """
    if has_direct:
        if bg_cos < 0.60:
            return "catastrophic_warn: BG 方向严重错位，先跑金丝雀核验再决定是否转路线 A (rel_err=%.3f, bg_cos=%.3f)" % (rel_err, bg_cos)
        if rel_err <= 0.12 and bg_cos >= 0.92:
            return "strong: 低扰动候选，直进金丝雀与 Stage 1 (rel_err=%.3f, bg_cos=%.3f)" % (rel_err, bg_cos)
        if bg_cos >= 0.85:
            return "moderate: 方向可用、可进金丝雀实测；若末层噪点则保留末5层稠密或校准β (rel_err=%.3f, bg_cos=%.3f)" % (rel_err, bg_cos)
        return "caution: 跨帧方向偏弱，金丝雀不达标即降 S_bg=3 或转路线 A (rel_err=%.3f, bg_cos=%.3f)" % (rel_err, bg_cos)
    if t_med >= 0.3:
        return "strong: 长程时间注意力强 (T=%.3f)" % t_med
    return "caution: 时间头偏局部 (T=%.3f)" % t_med


def dump(probe, summary, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"probe": probe, "summary": summary}, f, ensure_ascii=False, indent=2)
    import os
    os.replace(tmp, path)
    return path
