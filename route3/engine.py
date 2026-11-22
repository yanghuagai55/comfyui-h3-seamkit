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


def fill_dropped(v_full, keep_mask_2d):
    """把被删掉的帧的速度，用同位置相邻两个保留帧线性插值补上。

    为什么必须补：ComfyUI 的采样器每一步都对**整个 latent 体**做积分，被删掉的行
    如果没有速度，就会原地不动 → 冻结成噪声。所以引擎不能只返回紧凑序列的输出，
    必须散回全尺寸并把空缺填好，才能当 `forward` 的 drop-in。

    v_full      : [T*P, ...] 已按 t*P+p 排好，保留行已填
    keep_mask_2d: [T, P] bool
    """
    T, P = keep_mask_2d.shape
    shape = v_full.shape[1:]
    v = v_full.view(T, P, *shape)
    keep = keep_mask_2d.to(v.device)
    tt = torch.arange(T, device=v.device)[:, None].expand(T, P)
    prev = torch.where(keep, tt, torch.full_like(tt, -1)).cummax(0).values
    nxt = torch.where(keep, tt, torch.full_like(tt, T)).flip(0).cummin(0).values.flip(0)

    pv = prev.clamp(min=0)
    # 没有后继保留帧时，退化成「沿用前一个保留帧」，不能去读被删掉的行
    nx = torch.where(nxt >= T, pv, nxt.clamp(max=T - 1))
    denom = (nx - pv).clamp(min=1).to(v.dtype)
    frac = ((tt - pv).to(v.dtype) / denom).clamp(0.0, 1.0)
    frac = torch.where((prev < 0) | (nxt >= T), torch.zeros_like(frac), frac)

    def gather_t(i):
        return v.gather(0, i.unsqueeze(-1).expand(-1, -1, *shape))

    filled = gather_t(pv) * (1 - frac).view(T, P, *([1] * len(shape))) +         gather_t(nx) * frac.view(T, P, *([1] * len(shape)))
    out = torch.where(keep.view(T, P, *([1] * len(shape))), v, filled)
    return out.reshape(T * P, *shape)


def splitstream_forward(model, x, timestep, context, transformer_options=None,
                        minimax_payload=None, s_bg=5, beta=None, fg_mask=None,
                        keep_audio=True, return_plan=False, full_velocity=False,
                        use_beta=True):
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

    plan = route3_plan(prefix_end=va, latent_t=latent_t, lat_h=lat_h, lat_w=lat_w,
                       s_bg=s_bg, fg_mask=fg_mask)

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

    n_full = int(va + plan.fg_local.numel())          # branch1 键数（切点）

    # 当 beta 未显式传入且开启质量补偿、确实发生了压缩（plan.beta0 > 0）时，
    # 自动启用理论初值 beta0 = log(T / n_keep)，防止静默退化成无补偿压缩。
    eff_beta = beta
    if eff_beta is None and use_beta and plan.beta0 > 0.0 and plan.bg_local.numel() > 0:
        eff_beta = float(plan.beta0)

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

    v_full = torch.zeros(n_video, *v_kept.shape[1:], dtype=v_kept.dtype, device=v_kept.device)
    v_full[keep_video] = v_kept
    if full_velocity:
        gh, gw = lat_h // 2, lat_w // 2
        km = torch.zeros(latent_t * gh * gw, dtype=torch.bool, device=v_kept.device)
        km[keep_video] = True
        if not bool(km.all()):
            # Flow Matching 关键修正（本机真机冒烟实证）：
            # 不可直接插值速度 v = ε - x0 —— 各帧输入噪声 ε(t) 独立，插值 v 会残余 √2·σ 的纯噪声！
            # 正确做法：先由保留行算干净去噪预测 x0_hat = x_t + σ·v_kept（因外部 CONST 返回 -v_kept），
            # 对干净 x0_hat 跨帧线性插值，再反算被删帧的速度 v_drop = (x0_interp - x_t) / σ，
            # 使 Euler 步恰好消去被删帧自身的独立噪声 ε(j)！
            sig = float(sigma_v)
            if sig > 1e-5:
                x_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
                x0_full = torch.zeros_like(x_rows)
                x0_full[keep_video] = x_rows[keep_video] + sig * v_kept.to(torch.float32)
                x0_interp = fill_dropped(x0_full, km.view(latent_t, gh * gw))
                drop = ~km
                v_full[drop] = ((x0_interp[drop] - x_rows[drop]) / sig).to(v_kept.dtype)
            else:
                v_full = fill_dropped(v_full, km.view(latent_t, gh * gw))
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
