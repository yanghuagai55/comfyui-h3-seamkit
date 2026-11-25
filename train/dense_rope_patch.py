# SPDX-License-Identifier: GPL-3.0-or-later
"""纯 PyTorch 稠密注意力 + Split-Band RoPE 补丁（教师生成专用）。

为什么需要它
------------
`MiniMaxH3ChunkedSageAttentionPatch` 依赖新版 `sageattention`（`get_cuda_arch_versions`）。
Colab A100 上 `pip install sageattention` 只给到 1.0.6，缺这个 API，补丁会在 execute
里直接抛错。教师生成只需要**精确稠密注意力**，不需要 Sage 的 INT8 加速，所以这里用
`F.scaled_dot_product_attention` 复刻同一套 Split-Band RoPE（1.32）与 attn_temp（1.08），
数值上比 Sage 更接近原生。

不改变模型权重；只替换每个 block 的 `attn.forward`，并在 `_run_blocks` 外层临时包
`rope_freqs` 做分频段放大。
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from comfy_api.latest import io

CATEGORY = "MiniMax H3 Distill"


def _build_split_band_rope_scale(rope_hf_scale: float, n_freq: int, device: torch.device) -> torch.Tensor:
    """与 chunked_sage 完全一致：只放大空间高频 k=0..5，k=6/7 平滑过渡。"""
    g = torch.ones((n_freq,), dtype=torch.float32, device=device)
    if rope_hf_scale == 1.0 or n_freq <= 0:
        return g
    s = float(rope_hf_scale)
    hi_end = min(6, n_freq)
    g[:hi_end] = s
    if n_freq > 6:
        g[6] = 1.0 + (s - 1.0) * 0.75
    if n_freq > 7:
        g[7] = 1.0 + (s - 1.0) * 0.25
    return g


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # F.rms_norm 内部按 FP32 累积均值，BF16 输入不会因手工 mean 掉精度
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _apply_split_half_rope(q: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """q: [S, H, D]；table: [1, S, 1, half, 2, 2]。只旋转前 2*half 维。

    旋转本身强制 FP32：BF16 下 c*q0 - s*q1 会累积 1e-2 级相位误差，
    20+ 步教师会肉眼可见地漂。转回原 dtype 后再进 attention。
    """
    half = table.shape[-3]
    rot = half * 2
    if rot <= 0 or rot > q.shape[-1]:
        return q
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    q0 = q_rot[..., :half].float()
    q1 = q_rot[..., half:].float()
    # table: [1, S, 1, half, 2, 2] -> c/s: [S, 1, half]
    c = table[..., 0, 0][0].float()
    s = table[..., 1, 0][0].float()
    o0 = c * q0 - s * q1
    o1 = s * q0 + c * q1
    return torch.cat([o0, o1, q_pass.float()], dim=-1).to(q.dtype)


def _apply_split_band_rope(model, rope_hf_scale: float, va: int, vb: int):
    """临时把 Split-Band 包装装到 model 上，返回 restore()。

    与原 `_run_blocks` 内联版的唯一区别：套娃 bug 修在这里 ——
    原 `rope_freqs` 通常是类方法（不在实例 `__dict__`），restore 时必须
    `del model.__dict__["rope_freqs"]`，否则下一步会把上一次的包装当 base
    再包一层，20 步 = 1.32^20 ≈ 787×。
    """
    if rope_hf_scale == 1.0 or not hasattr(model, "rope_freqs"):
        return None
    had_instance = "rope_freqs" in getattr(model, "__dict__", {})
    prev = model.__dict__.get("rope_freqs") if had_instance else None
    base_fn = model.rope_freqs

    def split_band_rope_freqs(position_ids, dev):
        angles = base_fn(position_ids, dev)
        if angles.ndim == 2 and angles.shape[0] >= vb and angles.shape[1] % 6 == 0:
            nf = angles.shape[1] // 6
            half = nf * 3
            g = _build_split_band_rope_scale(rope_hf_scale, nf, angles.device)
            angles = angles.clone()
            angles[va:vb, nf:2 * nf].mul_(g)
            angles[va:vb, 2 * nf:3 * nf].mul_(g)
            angles[va:vb, half + nf:half + 2 * nf].mul_(g)
            angles[va:vb, half + 2 * nf:half + 3 * nf].mul_(g)
        return angles

    model.rope_freqs = split_band_rope_freqs

    def restore():
        if had_instance:
            model.rope_freqs = prev
        elif "rope_freqs" in getattr(model, "__dict__", {}):
            del model.__dict__["rope_freqs"]

    return restore


def _make_dense_attn_forward(head_dim: int):
    scale_base = head_dim ** -0.5

    def forward(self, x, rope_freqs=None, transformer_options=None):
        transformer_options = transformer_options or {}
        s = x.shape[0]
        heads = self.heads
        inner = heads * head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)
        q = q.view(s, heads, head_dim)
        k = k.view(s, heads, head_dim)
        v = v.view(s, heads, head_dim)

        qw = self.q_norm.weight.to(device=q.device, dtype=q.dtype)
        kw = self.k_norm.weight.to(device=k.device, dtype=k.dtype)
        q = _rms_norm(q, qw, float(self.q_norm.eps))
        k = _rms_norm(k, kw, float(self.k_norm.eps))
        if rope_freqs is not None:
            q = _apply_split_half_rope(q, rope_freqs)
            k = _apply_split_half_rope(k, rope_freqs)

        # [S,H,D] -> [1,H,S,D]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        temp = float(transformer_options.get("_dense_rope_attn_temp", 1.0))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             scale=scale_base * temp)
        out = out.squeeze(0).transpose(0, 1).reshape(s, inner)
        return self.out_proj(out)

    return forward


class MiniMaxH3DenseRopePatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DenseRopePatch",
            display_name="MiniMax H3 Dense SDPA + Split-Band RoPE",
            description=(
                "纯 PyTorch 稠密注意力（SDPA）+ Split-Band RoPE 高频放大 + attn_temp 温度补偿。\n"
                "教师生成用：不依赖 sageattention/comfy_kitchen，数值比 Sage 更接近原生。\n"
                "不改变权重；只替换 attn.forward 与 _run_blocks 的 rope_freqs 包装。"
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("rope_hf_scale", default=1.32, min=0.5, max=2.0, step=0.02),
                io.Float.Input("attn_temp_scale", default=1.08, min=0.8, max=1.3, step=0.01),
                io.Boolean.Input("chunk_io_layers", default=True,
                                 tooltip="首尾 I/O 层按 token 分块，省显存；不影响数值。"),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, rope_hf_scale, attn_temp_scale, chunk_io_layers):
        rope_hf_scale = float(rope_hf_scale)
        attn_temp_scale = float(attn_temp_scale)
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn"):
            raise RuntimeError("MiniMaxH3DenseRopePatch: unexpected model layout (no blocks[*].attn)")

        head_dim = int(blocks[0].attn.head_dim)
        attn_fwd = _make_dense_attn_forward(head_dim)
        for idx, block in enumerate(blocks):
            m.add_object_patch(
                f"diffusion_model.blocks.{idx}.attn.forward",
                types.MethodType(attn_fwd, block.attn),
            )

        orig_run_blocks = type(dm)._run_blocks

        def packed_run_blocks(self, h, t_emb, mod_segments, layout, transformer_options):
            prev_temp = transformer_options.get("_dense_rope_attn_temp", None)
            transformer_options["_dense_rope_attn_temp"] = attn_temp_scale
            restore = None
            try:
                if rope_hf_scale != 1.0 and hasattr(self, "rope_freqs"):
                    segs = getattr(layout, "segments", None) or []
                    span = next(((int(a), int(b)) for a, b, kind in segs if kind == "video"), None)
                    if span is not None and span[1] > span[0]:
                        restore = _apply_split_band_rope(self, rope_hf_scale, *span)
                return orig_run_blocks(self, h, t_emb, mod_segments, layout, transformer_options)
            finally:
                # restore() 内部已按 chunked_sage.py:780-784 处理「原 rope_freqs 是类方法」
                # 的情形：必须 del 实例属性，否则下一步会再包一层（1.32^n 套娃）。
                if restore is not None:
                    restore()
                if prev_temp is None:
                    transformer_options.pop("_dense_rope_attn_temp", None)
                else:
                    transformer_options["_dense_rope_attn_temp"] = prev_temp

        m.add_object_patch("diffusion_model._run_blocks", types.MethodType(packed_run_blocks, dm))
        print(
            f"[Distill] dense SDPA patch: rope_hf={rope_hf_scale:.2f} attn_temp={attn_temp_scale:.2f} "
            f"blocks={len(blocks)} head_dim={head_dim}",
            flush=True,
        )
        return io.NodeOutput(m)


NODES = [MiniMaxH3DenseRopePatch]
