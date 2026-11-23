# -*- coding: utf-8 -*-
"""Route 3 · 压缩序列的 forward

思路（不 fork 整个 `_forward`，只换「喂给 50 层的行集」）：

1. 先用模型自己的 `_embed_and_pack` 在**完整布局**上算嵌入 —— 文本/条件/参考/音频的
   位置编码、时间步全部沿用原生实现，零重复实现风险。
2. 按 `route3_plan` 挑出 `keep = [Prefix | FG_full | BG_strided]` 这些行。
3. 用挑出来的行跑 50 层（rope 表用 `position_ids[keep]`；mod_segments 前缀原样复用，
   视频段并成一条）。
4. 只对保留下来的视频行过 `final_layer`，再 scatter 回完整网格 → `unpatchify_video`。
   中间帧由上层做残差传播（本文件不负责）。

性质：
* `s_bg=1`（全部帧都是 BG 关键帧、FG 为空、β=0）时，`keep` 就是恒等置换，
  本函数必须与 `model.forward` **逐位一致** —— 这是自检的黄金用例。
* `beta=None` 表示不做质量补偿，等价于「直接拼完跑普通 attention」的朴素压缩。
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import torch

import comfy.ldm.common_dit
from comfy.ldm.minimax.model import (FRAME_RESCALE, PackedLayout, SEG_TAG,
                                     _tag_runs, pack_audio, patchify_video, pdd_grid,
                                     unpatchify_video, unpack_audio)
from .assembly import reconstruct_moving
from .splitstream_attn import beta_for_sequence, route3_plan, splitstream_attention
from .route3_block import rms_rope_split_half_ref

__all__ = ["splitstream_forward", "LayoutShim", "beta_attention", "fill_dropped"]


class LayoutShim:
    """`_run_blocks` 只读 `position_ids`（其余字段在 blocks_replace 里才用）。"""

    def __init__(self, position_ids, segments):
        self.position_ids = position_ids
        self.segments = segments
        self.seq_len = position_ids.shape[0]
        self.signature = None


def _seg_of(layout, kind):
    return next(s for s in layout.segments if s[2] == kind)


def _fill_dropped_linear(v, src_keep, t_coords=None, return_dist=False):
    """在单一时间子序列 v:[K,P,...], src_keep:[K,P] 内按真实时间坐标 t_coords 线性插值。

    注意首尾外推：
      * prev < 0（左侧无保留源，如 P 池首帧 t=1 被删、t=2 保留）：pv 取右侧 nxt，绝不读 0 号空行！
      * nxt >= K（右侧无保留源）：nx 取左侧 pv。
    """
    K, P = src_keep.shape
    shape = v.shape[2:]
    tt = torch.arange(K, device=v.device)[:, None].expand(K, P)
    prev = torch.where(src_keep, tt, torch.full_like(tt, -1)).cummax(0).values
    nxt = torch.where(src_keep, tt, torch.full_like(tt, K)).flip(0).cummin(0).values.flip(0)

    pv = torch.where(prev < 0, nxt.clamp(max=K - 1), prev.clamp(min=0))
    nx = torch.where(nxt >= K, pv, nxt.clamp(max=K - 1))
    if t_coords is not None:
        tc = t_coords.to(device=v.device, dtype=v.dtype)
        t_cur = tc[:, None].expand(K, P)
        t_pv = tc[pv]
        t_nx = tc[nx]
        denom = (t_nx - t_pv).clamp(min=1.0)
        frac = ((t_cur - t_pv) / denom).clamp(0.0, 1.0)
    else:
        t_cur = tt.to(v.dtype)
        t_pv = pv.to(v.dtype)
        t_nx = nx.to(v.dtype)
        denom = (nx - pv).clamp(min=1).to(v.dtype)
        frac = ((tt - pv).to(v.dtype) / denom).clamp(0.0, 1.0)
    frac = torch.where((prev < 0) | (nxt >= K), torch.zeros_like(frac), frac)

    def gather_t(i):
        return v.gather(0, i.unsqueeze(-1).expand(-1, -1, *shape))

    filled = (gather_t(pv) * (1 - frac).view(K, P, *([1] * len(shape)))
              + gather_t(nx) * frac.view(K, P, *([1] * len(shape))))
    out = torch.where(src_keep.view(K, P, *([1] * len(shape))), v, filled)
    if not return_dist:
        return out
    inf = torch.full_like(t_cur, 1e9)
    d_l = torch.where(prev >= 0, (t_cur - t_pv).abs(), inf)
    d_r = torch.where(nxt < K, (t_nx - t_cur).abs(), inf)
    return out, torch.minimum(d_l, d_r)


def _detect_anchor_cuts(v, keep_2d, cut_ratio=3.5, cut_floor=0.25):
    """在没有外部 route3_cuts 时，从全保留锚点帧上自动检测镜头硬切起始下标。"""
    T = int(v.shape[0])
    if T < 4:
        return []
    full_t = [t for t in range(T) if bool(keep_2d[t].all().item())]
    if len(full_t) < 3:
        return []
    vs = v[full_t].float()
    diffs = (vs[1:] - vs[:-1]).abs().mean(dim=tuple(range(1, vs.ndim)))
    med = float(diffs.median().item()) if diffs.numel() else 0.0
    thr = max(float(cut_ratio) * max(med, 1e-6), float(cut_floor))
    cuts = []
    for k in range(len(full_t) - 1):
        if float(diffs[k].item()) > thr:
            # 切点落在 full_t[k] 与 full_t[k+1] 之间；保守取中间后半段起点 full_t[k+1]
            cuts.append(int(full_t[k + 1]))
    return cuts


def fill_dropped(v_full, keep_mask_2d, phase_split=True, source_mask_2d=None,
                 cuts=None, max_src_gap=8):
    """把被删掉的 BG 帧的预测，用同镜头内、同池相邻的**纯 BG 保留帧**线性插值补上。

    关键修复（2026-11-23 实片 00003/00004/00007 实证）：
      1. phase_split=True：严格分离 1 帧 I-token (t%5==0) 与 4 帧 P-token (t%5!=0) 两池插值；
      2. prev < 0 首帧外推修复：t=1 被删时取右侧 t=2，绝不读全 0 空行（消灭 f001..f004 紫边环）；
      3. source_mask_2d（纯 BG 锚点）+ max_src_gap 跨度封顶：被删 BG 格只从近邻纯 BG 锚点插值，
         不混入运动手臂 x0_hat；若该格深处人物内部（附近无纯 BG 源，如偶发掩码空洞）则回落近邻 keep；
      4. cuts 硬切自动断链：绝不跨镜头硬切插值（彻底消灭 00007 镜头 1 亮窗跨镜传送到镜头 2 头发的白色方块！）。
    """
    T, P = keep_mask_2d.shape
    shape = v_full.shape[1:]
    v_all = v_full.view(T, P, *shape)
    keep_all = keep_mask_2d.to(v_all.device)
    src_all = keep_all if source_mask_2d is None else source_mask_2d.to(v_all.device)

    cut_list = sorted(set(int(c) for c in cuts if 0 < int(c) < T)) if cuts is not None else _detect_anchor_cuts(v_all, keep_all)
    bounds = [0] + cut_list + [T]

    def _interp_with_src(v_in, s_mask, k_mask, t_coords=None):
        """优先用近邻 s_mask（纯 BG 源，距离 <= max_src_gap）插值；否则回落同池 k_mask。"""
        has_s = s_mask.any(dim=0)
        f_k = _fill_dropped_linear(v_in, k_mask, t_coords=t_coords)
        if s_mask is k_mask or not bool(has_s.any()):
            return f_k
        f_s, d_s = _fill_dropped_linear(v_in, s_mask, t_coords=t_coords, return_dist=True)
        use_s = has_s.unsqueeze(0).expand_as(s_mask)
        if max_src_gap is not None and float(max_src_gap) > 0:
            use_s = use_s & (d_s <= float(max_src_gap))
        return torch.where(use_s.view(v_in.shape[0], P, *([1] * len(shape))), f_s, f_k)

    def _fill_segment(v, keep, src, seg_lo):
        Ts = v.shape[0]
        t_glob = torch.arange(seg_lo, seg_lo + Ts, device=v.device)
        if (not phase_split) or Ts < 2:
            filled = _interp_with_src(v, src, keep, t_coords=t_glob)
            return torch.where(keep.view(Ts, P, *([1] * len(shape))), v, filled)

        fallback = _interp_with_src(v, src, keep, t_coords=t_glob)
        out = fallback.clone()
        i_rel = torch.tensor([r for r in range(Ts) if (seg_lo + r) % 5 == 0], dtype=torch.long, device=v.device)
        p_rel = torch.tensor([r for r in range(Ts) if (seg_lo + r) % 5 != 0], dtype=torch.long, device=v.device)

        for sub_rel in (i_rel, p_rel):
            if sub_rel.numel() == 0:
                continue
            k_sub = keep.index_select(0, sub_rel)
            s_sub = src.index_select(0, sub_rel)
            has_s = s_sub.any(dim=0)
            has_k = k_sub.any(dim=0)
            if not bool(has_k.any()):
                continue
            sub_t = t_glob.index_select(0, sub_rel)
            f_sub = _interp_with_src(v.index_select(0, sub_rel), s_sub, k_sub, t_coords=sub_t)
            fb_sub = fallback.index_select(0, sub_rel)
            use_pool = has_s | has_k
            merged = torch.where(use_pool.view(1, P, *([1] * len(shape))), f_sub, fb_sub)
            out.index_copy_(0, sub_rel, merged)
        return torch.where(keep.view(Ts, P, *([1] * len(shape))), v, out)

    out_all = v_all.clone()
    for si in range(len(bounds) - 1):
        lo, hi = bounds[si], bounds[si + 1]
        if hi <= lo:
            continue
        out_all[lo:hi] = _fill_segment(v_all[lo:hi], keep_all[lo:hi], src_all[lo:hi], lo)
    return out_all.reshape(T * P, *shape)


def splitstream_forward(model, x, timestep, context, transformer_options=None,
                        minimax_payload=None, s_bg=3, beta=None, fg_mask=None,
                        keep_audio=True, return_plan=False, full_velocity=False,
                        use_beta=True, beta_scale=1.0, beta_max=0.0):
    """返回 `[-video_out, -audio_out]`，与 `model.forward` 同签名同语义。"""
    transformer_options = dict(transformer_options or {})
    payload = dict(minimax_payload or {})
    if "layout" in payload and payload["layout"] is not None:
        payload.pop("layout")            # 我们自己建，避免签名不匹配

    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]

    layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                          keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    va, vb, _ = _seg_of(layout, "video")
    aa, ab, _ = _seg_of(layout, "audio")
    n_video = vb - va

    r3_pins = transformer_options.get("route3_pin_tokens")
    plan = route3_plan(prefix_end=va, latent_t=latent_t, lat_h=lat_h, lat_w=lat_w,
                       s_bg=s_bg, fg_mask=fg_mask, pin_tokens=r3_pins)

    # keep = [0..va) + FG + BG，行号是完整布局的下标
    fg_global = va + plan.fg_local
    bg_global = va + plan.bg_local
    keep = torch.cat([torch.arange(va, dtype=torch.long),
                      fg_global, bg_global]).to(device)
    keep_video = torch.cat([plan.fg_local, plan.bg_local]).to(device)   # 视频段内下标
    s_kept = int(keep.numel())

    # 时间步 / t_emb：与原生完全一致
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    from comfy.ldm.minimax.model import time_shift_sigma, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments})
    t_row = {t: i for i, t in enumerate(unique_t)}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            mod_segments.extend(_tag_runs(a, b, row_base, text_tags))
        elif kind == "video":
            mod_segments.append((va, s_kept, row_base + SEG_TAG["video"]))
        else:
            mod_segments.append((a, b, row_base + SEG_TAG[kind]))

    h_full = model._embed_and_pack(video_x, audio_x, context, layout, payload, transformer_options)
    h = h_full[keep]
    del h_full
    pos = layout.position_ids[keep.cpu()].to(device)
    shim = LayoutShim(pos, [(0, s_kept, "kept")])
    t_emb = model._time_embedding(unique_t, device, dtype)

    n_full = int(plan.n_full)                         # branch1 键数（含 Prefix + FG + 未压缩 I-BG）

    # 当 beta 未显式传入且开启质量补偿、确实发生了压缩（plan.beta0 > 0）时，
    # 自动启用理论初值 beta0 = log(N_P_dense / N_P_kept)，仅加在 branch2（压缩 P-BG）上！
    eff_beta = beta
    if eff_beta is None and use_beta and plan.beta0 > 0.0 and n_full < s_kept:
        eff_beta = float(plan.beta0)
    if eff_beta is not None:
        if float(beta_scale) != 1.0:
            eff_beta = eff_beta * float(beta_scale)
        if float(beta_max) > 0.0:
            if torch.is_tensor(eff_beta):
                eff_beta = eff_beta.clamp(max=float(beta_max))
            else:
                eff_beta = min(float(eff_beta), float(beta_max))

    phase_full = plan.full_phase()
    with (beta_attention(model, n_full, eff_beta, phase_full) if eff_beta is not None else _nullctx()):
        transformer_options["route3"] = {
            "n_full": n_full, "plan": plan, "phase_full": phase_full,
        }
        h = model._run_blocks(h, t_emb, mod_segments, shim, transformer_options)

    # final layer：只喂保留下来的视频行
    v_kept, a_out = model.final_layer(
        h, t_emb, (va, s_kept, t_row[seg_t["video"]]), (aa, ab, t_row[seg_t["audio"]]),
        sigma_v, transformer_options.get("sample_sigmas"),
        (pdd_grid(model.final_layer.pdd_steps, shift_v),
         pdd_grid(model.final_layer.pdd_steps, shift_a)))

    # 暂停 comfy_aimdo MallocGraph 录制包裹整个后处理段：避免光流估计与张量重排撑爆 C 递归析构栈
    try:
        import comfy.model_prefetch as _mp
        _mg_post = _mp.pause_malloc_graph()
    except Exception:
        _mg_post = _nullctx()

    with _mg_post:
        v_full = torch.zeros(n_video, *v_kept.shape[1:], dtype=v_kept.dtype, device=v_kept.device)
        v_full[keep_video] = v_kept
        if full_velocity:
            gh, gw = lat_h // 2, lat_w // 2
            km = torch.zeros(latent_t * gh * gw, dtype=torch.bool, device=v_kept.device)
            km[keep_video] = True
            if not bool(km.all()):
                # 纯 BG 锚点掩码：被删的 BG 格只能从纯 BG 锚点插值，绝不混入运动前景手臂/发丝的 x0_hat！
                bg_src = torch.zeros(latent_t * gh * gw, dtype=torch.bool, device=v_kept.device)
                if plan.bg_local.numel() > 0:
                    bg_src[plan.bg_local.to(v_kept.device)] = True
                else:
                    bg_src = km
                sig = float(sigma_v)
                r3_cuts = transformer_options.get("route3_cuts")
                r3_p1_up = transformer_options.get("route3_p1_up")
                r3_flow_cache = transformer_options.get("route3_flow_cache")
                if sig > 1e-5:
                    x_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
                    x0_full = torch.zeros_like(x_rows)
                    x0_full[keep_video] = x_rows[keep_video] + sig * v_kept.to(torch.float32)

                    use_p1_guided = (
                        torch.is_tensor(r3_p1_up)
                        and r3_p1_up.ndim == 5
                        and r3_p1_up.shape[2] == orig_t
                        and r3_p1_up.shape[3] == orig_h
                        and r3_p1_up.shape[4] == orig_w
                        and bool(plan.anchors)
                    )
                    if use_p1_guided:
                        # Pass-1 引导的流场残差轨迹（根治推镜重影与沙发角/毛巾体积跳变）：
                        # 被删 BG 帧保持 z_p1_up[j] 精确运镜底图 + 短基线双向光流配准与高频偶极能量守恒后的残差 δ = x0_hat - z_p1_up
                        p1_pad = comfy.ldm.common_dit.pad_to_patch_size(
                            r3_p1_up.to(device=v_kept.device, dtype=torch.float32), model.patch_size
                        )
                        p1_rows = patchify_video(p1_pad, model.patch_size)
                        delta_rows = torch.zeros_like(x_rows)
                        delta_rows[keep_video] = x0_full[keep_video] - p1_rows[keep_video]
                        delta_5d = unpatchify_video(
                            delta_rows, latent_t, gh, gw, model.latents_dim, model.patch_size
                        )
                        p1_tchw = p1_pad[0].permute(1, 0, 2, 3).contiguous()
                        delta_tchw = delta_5d[0].permute(1, 0, 2, 3).contiguous()
                        delta_keys = {int(t): delta_tchw[int(t)] for t in plan.anchors}
                        fg_lat = (
                            (km & (~bg_src)).view(latent_t, gh, gw)
                            .repeat_interleave(2, dim=1)
                            .repeat_interleave(2, dim=2)
                        )
                        x0_rec_tchw = reconstruct_moving(
                            p1_tchw, delta_keys, plan.anchors,
                            method="flow", cut_protect=True, phase_split=True,
                            fg_mask=fg_lat, cuts=r3_cuts, flow_cache=r3_flow_cache,
                        )
                        x0_interp = patchify_video(
                            x0_rec_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
                            model.patch_size,
                        )
                        del p1_pad, p1_rows, delta_rows, delta_5d, p1_tchw, delta_tchw, delta_keys, fg_lat, x0_rec_tchw
                    else:
                        x0_interp = fill_dropped(
                            x0_full, km.view(latent_t, gh * gw),
                            source_mask_2d=bg_src.view(latent_t, gh * gw),
                            cuts=r3_cuts,
                        )
                    drop = ~km
                    v_full[drop] = ((x0_interp[drop] - x_rows[drop]) / sig).to(v_kept.dtype)
                else:
                    v_full = fill_dropped(
                        v_full, km.view(latent_t, gh * gw),
                        source_mask_2d=bg_src.view(latent_t, gh * gw),
                        cuts=r3_cuts,
                    )
        video_out = unpatchify_video(v_full, latent_t, lat_h // 2, lat_w // 2,
                                     model.latents_dim, model.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a_out)

        out = [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
    if return_plan:
        return out, plan, keep_video
    return out


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@contextmanager
def beta_attention(model, n_full, beta, phase_full=None):
    """给每层 attention 注入「BG 键加 beta + LSE 合并」。"""
    originals = []
    for i, blk in enumerate(model.blocks):
        attn = blk.attn
        originals.append((attn, attn.forward))
        b = beta[i] if (torch.is_tensor(beta) and beta.dim() == 3) else beta

        def fwd(self, x, rope_freqs=None, transformer_options={}, _b=b):
            return _attn_with_beta(self, x, rope_freqs, transformer_options, n_full, _b,
                                   phase_full)

        attn.forward = types.MethodType(fwd, attn)
    try:
        yield
    finally:
        for attn, orig in originals:
            attn.forward = orig


def _attn_with_beta(self, x, rope_freqs, transformer_options, n_full, beta, phase_full=None):
    import comfy.model_management
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    # 兼容 KJNodes MiniMaxLowVRAMAttention（以单元素 list 移交 normed h 以便及时释放）
    if isinstance(x, list):
        x = x.pop()
    s = x.shape[0]
    dtype = x.dtype
    device = x.device
    qkv = self.qkv_proj(x)
    del x
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        if (not torch.is_grad_enabled()) and (not getattr(comfy.model_management, "in_training", False)):
            try:
                from comfy import quant_ops as _qops
                _qops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            except Exception:
                q, k = rms_rope_split_half_ref(q, k, rope_freqs, qw, kw,
                                               epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q, k = rms_rope_split_half_ref(q, k, rope_freqs, qw, kw,
                                           epsilon=self.q_norm.eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    q = q.transpose(0, 1).unsqueeze(0)   # [1,H,S,D]
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    if beta is None or n_full <= 0 or n_full >= s:
        out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k),
                                  AttentionTensorContainer(v), self.heads, mask=None,
                                  skip_reshape=True, transformer_options=transformer_options)
        del q, k, v, qkv
        return self.out_proj(out.squeeze(0) if out.dim() == 3 else out)

    b = beta.to(device) if torch.is_tensor(beta) else beta
    if phase_full is not None and torch.is_tensor(b) and b.dim() == 2:
        b = beta_for_sequence(b, 0, phase_full.to(device))

    # 支持 MiniMaxLowVRAMAttention 的 minimax_head_chunks 分头切片，压住 8GB 峰值显存
    n_chunks = (min(int(transformer_options.get("minimax_head_chunks", 1)), self.heads)
                if isinstance(transformer_options, dict) else 1)
    if n_chunks <= 1:
        out = splitstream_attention(q, k, v, n_full, beta=b, provider="auto")
        out = out.transpose(1, 2).reshape(s, self.heads * self.head_dim)
    else:
        out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
        hs = 0
        for i in range(n_chunks):
            size = self.heads // n_chunks + (1 if i < self.heads % n_chunks else 0)
            he = hs + size
            b_slice = (b[:, hs:he] if (torch.is_tensor(b) and b.dim() >= 2 and b.shape[1] == self.heads)
                       else (b[hs:he] if (torch.is_tensor(b) and b.dim() == 1 and b.shape[0] == self.heads) else b))
            o = splitstream_attention(q[:, hs:he], k[:, hs:he], v[:, hs:he],
                                      n_full, beta=b_slice, provider="auto")
            out[:, hs * self.head_dim:he * self.head_dim] = o.transpose(1, 2).reshape(s, size * self.head_dim)
            hs = he
    del q, k, v, qkv
    return self.out_proj(out)
