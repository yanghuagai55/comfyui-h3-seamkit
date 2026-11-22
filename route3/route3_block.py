# -*- coding: utf-8 -*-
"""Route 3 · 可训练的 DiT 块 / LoRA 适配

为什么需要这个文件
------------------
`comfy/ldm/minimax/model.py` 的 `DiTBlock.forward` 全部用原地算子：
    h[a:b].mul_(1+scale).add_(shift)          # _mod_scale_shift
    x[a:b].addcmul_(other, gate)              # _mod_gate
而 `norm1(x)` 已经把 x 存进了反向图。于是第一次 backward 必抛：
    RuntimeError: one of the variables needed for gradient computation has been
    modified by an inplace operation: [torch.cuda.FloatTensor [207, 256]] ...
（本机 tiny 模型已复现，见 docs/ROUTE3_READINESS.md §2-E）

另外 `rms_rope_split_half_` 在非训练模式下是原地算子，会抛
    "in-place RoPE operations are inference-only and do not support autograd"
所以训练期间必须 `comfy.model_management.in_training = True`（内置 TrainLoraNode 会置）。

本文件只做两件事：
1. 提供 out-of-place 版 block forward（推理时自动回落到原实现，不牺牲速度）；
2. 提供一个最小的 LoRA 包，可套在 int8/量化 Linear 外面（`y = W x + scale·B A x`）。
"""

from __future__ import annotations

import math
import types

import torch
from torch import nn

__all__ = ["mod_scale_shift_out", "mod_gate_out", "train_safe_block_forward",
           "install", "LoRALinear", "attach_lora", "lora_state_dict",
           "rms_rope_split_half_ref", "attn_forward_train", "patch_attention"]


# --------------------------------------------------------------------------- #
# 可反传的融合 RoPE                                                            #
# --------------------------------------------------------------------------- #
def rms_rope_split_half_ref(q, k, freqs_cis, q_scale, k_scale=None,
                            epsilon=1e-6, rot_dim=0):
    """`comfy_kitchen.rms_rope_split_half` 的纯 torch 等价实现（可反传）。

    为什么必须自己写：comfy_kitchen 的 custom op 没有注册 autograd，backward 会抛
        RuntimeError: Trying to backward through comfy_kitchen.rms_rope_split_half.default
        but no autograd formula was registered.
    （本机已复现，见 docs/ROUTE3_READINESS.md）

    语义严格对齐 eager 参考实现 `_rms_rope1(split_half=True, rot_dim=rot_dim)`：
        x = rms_norm(x, weight=scale, eps)
        out0 = x0*cos - x1*sin ; out1 = x0*sin + x1*cos   （配对步距 half = rot_dim/2）
        后 (D - rot_dim) 维原样透传
    freqs_cis: [..., rot_dim/2, 2, 2]，其中 [...,0,0]=cos, [...,0,1]=-sin,
               [...,1,0]=sin, [...,1,1]=cos
    """
    if k_scale is None:
        k_scale = q_scale

    def one(x, scale):
        xn = nn.functional.rms_norm(x, (x.shape[-1],), weight=scale, eps=epsilon)
        half = rot_dim // 2
        if not rot_dim or rot_dim == x.shape[-1]:
            xr, tail = xn, None
            half = x.shape[-1] // 2
        else:
            xr, tail = xn[..., :rot_dim], xn[..., rot_dim:]
        cos = freqs_cis[..., 0, 0]
        sin = freqs_cis[..., 1, 0]
        x0, x1 = xr[..., :half], xr[..., half:]
        out = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        if tail is not None:
            out = torch.cat([out, tail], dim=-1)
        return out

    return one(q, q_scale), one(k, k_scale)


def _row(vecs, row, dtype):
    return vecs[row].to(dtype)


def mod_scale_shift_out(h, shift, scale, segments):
    """h*(1+scale)+shift，逐段。不原地改写 h。"""
    parts = []
    for a, b, row in segments:
        if b <= a:
            continue
        parts.append(h[a:b] * (1.0 + _row(scale, row, h.dtype)) + _row(shift, row, h.dtype))
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def mod_gate_out(x, gate, other, segments):
    """x + other*gate，逐段。不原地改写 x。"""
    parts = []
    for a, b, row in segments:
        if b <= a:
            continue
        parts.append(x[a:b] + other[a:b] * _row(gate, row, x.dtype))
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def train_safe_block_forward(self, x, t_emb, mod_segments, rope_freqs,
                             transformer_options={}, attention=None):
    attention = self.attn if attention is None else attention
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = mod_scale_shift_out(self.norm1(x), shift_msa, scale_msa, mod_segments)
    attn_out = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    x = mod_gate_out(x, gate_msa, attn_out, mod_segments)
    h = mod_scale_shift_out(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return mod_gate_out(x, gate_mlp, self.mlp(h), mod_segments)


def install(model, force: bool = False):
    """给每个 DiTBlock 装 out-of-place forward。

    force=False 时只在梯度开启的调用里生效（推理保持原实现的速度）。
    """
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise ValueError("not a MiniMax H3 model (no .blocks)")
    for blk in blocks:
        if getattr(blk, "_route3_train_safe", False):
            continue
        orig = blk.forward

        def fwd(self, x, t_emb, mod_segments, rope_freqs, transformer_options={},
                attention=None, _orig=orig):
            if force or torch.is_grad_enabled():
                return train_safe_block_forward(self, x, t_emb, mod_segments,
                                               rope_freqs, transformer_options, attention)
            return _orig(x, t_emb, mod_segments, rope_freqs,
                         transformer_options=transformer_options, attention=attention)

        blk.forward = types.MethodType(fwd, blk)
        blk._route3_train_safe = True
    return model


def attn_forward_train(self, x, rope_freqs=None, transformer_options={}):
    """Attention.forward 的可反传版本（唯一区别：RoPE 走纯 torch 实现）。"""
    import comfy.model_management
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        q, k = rms_rope_split_half_ref(q, k, rope_freqs, qw, kw,
                                       epsilon=self.q_norm.eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(q, k, v, self.heads, preferred_attention=self.comfy_attention,
                              mask=None, skip_reshape=True,
                              transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))


def patch_attention(model):
    """把每个 Attention.forward 换成可反传版（推理时保持原实现）。"""
    n = 0
    for blk in getattr(model, "blocks", []):
        attn = blk.attn
        if getattr(attn, "_route3_train_attn", False):
            continue
        orig = attn.forward

        def fwd(self, x, rope_freqs=None, transformer_options={}, _orig=orig):
            if torch.is_grad_enabled():
                return attn_forward_train(self, x, rope_freqs, transformer_options)
            return _orig(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

        attn.forward = types.MethodType(fwd, attn)
        attn._route3_train_attn = True
        n += 1
    return n


class LoRALinear(nn.Module):
    """套在任意 Linear（含量化）外面的低秩增量：y = base(x) + scale * B(A(x))。"""

    def __init__(self, base: nn.Module, rank: int = 16, alpha: float | None = None,
                 dtype=None, device=None):
        super().__init__()
        self.base = base
        out_f, in_f = base.weight.shape[0], base.weight.shape[1]
        self.rank = rank
        self.scale = (alpha or rank) / rank
        dt = dtype or base.weight.dtype
        dev = device or base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=dt, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def attach_lora(model, rank: int = 16, targets=("qkv_proj", "out_proj")):
    """只挂 attention 的 qkv/out（任务书 Stage 3 口径）。"""
    wrapped = []
    for i, blk in enumerate(model.blocks):
        for name in targets:
            lin = getattr(blk.attn, name, None)
            if lin is None or isinstance(lin, LoRALinear):
                continue
            setattr(blk.attn, name, LoRALinear(lin, rank=rank))
            wrapped.append("blocks.%d.attn.%s" % (i, name))
    return wrapped


def lora_state_dict(model):
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            out[name + ".lora_A"] = mod.lora_A.detach().cpu()
            out[name + ".lora_B"] = mod.lora_B.detach().cpu()
    return out
