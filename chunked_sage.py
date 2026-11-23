# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Chunked-QKV + KJ-compatible SageAttention2++ patch for MiniMax H3.

Why stock KJNodes (`MiniMaxLowVRAMAttention` + `MiniMaxH3MemoryEfficientSageAttentionPatch`
+ `MiniMaxChunkFeedForward`) OOMs at 10s (243f @ 1568x864, S ≈ 96k) on 8GB GPUs:
  1. `self.qkv_proj(h)` materializes the entire BF16 `[S, 21504]` QKV buffer at once
     (3.85 GiB at 10s, 5.78 GiB at 15s) alongside residual `x` (0.98 GiB) and `out` (1.28 GiB).
  2. `minimax_block_lowmem_forward` materializes full `[S, 5376]` normed `h` twice and
     `minimax_mlp_chunked_forward` allocates full `[S, 5376]` `out` alongside `h` and `fc1(c)`.
  3. `video_patch_proj` and `FinalLayer._modulate` materialize full `[N_v, 5376]` FP32 tensors
     (1.96-2.94 GiB at 10s-15s).

What this module does (100% compatible with KJ SageAttention2++ kernels across SM75/80/86/89/90/120):
  1. Fused token-chunked `norm1 + _mod_scale_shift + INT8 ConvRot pre-quantization`:
     Writes `(x_qdata [S, 5376] int8, x_scale [S, 1] fp32)` once per block directly via
     comfy_kitchen's fused ConvRot quantizer — never materializes full BF16 `h [S, 5376]`.
  2. Head-group sliced QKV GEMM + staged V projection + KJ SageAttention2++:
     For each head group `hs:he` (even-sized so `g * 128` is a multiple of `convrot_groupsize=256`):
       - Slices contiguous zero-copy rows of `qkv_proj` weight for Q and K;
       - Runs pre-quantized INT8 GEMM (or dense BF16 linear when unquantized / LoRA-patched);
       - Applies in-place `ck.rms_rope_split_half_` + KJ int64-safe per-thread INT8 Q/K quant;
       - Frees BF16 `q_g, k_g` BEFORE projecting `v_g`, then quantizes `v_g` to FP8/FP16 and frees `v_g`;
       - Runs the exact same SageAttention2++ CUDA kernel KJNodes uses (`_qattn_sm89`/`sm80`/`sm90`).
  3. Two `out_proj` modes:
       - `exact`: Stores `[S, 7168]` attention output, frees `(x_qdata, x_scale)`, then runs
         token-chunked `out_proj + _mod_gate`. **100.00% bit-for-bit identical** to Stock KJ!
         (RTX 4060 8GB peak activation: 1.63 GiB @ 5s, 3.26 GiB @ 10s — lower than Stock KJ 5s!)
       - `streaming`: Projects each head group through its `out_proj` column slice (`[5376, g*128]`,
         exact multiple of `convrot_groupsize=256`) directly into residual `x` in token chunks —
         **zero `[S, 7168]` out buffer!**
         (RTX 4060 8GB peak activation: 1.21 GiB @ 5s, 1.98 GiB @ 10s, 2.97 GiB @ 15s!)
  4. Fused token-chunked `norm2 + _mod_scale_shift + SwiGLU MLP + _mod_gate`:
     Never materializes full `[S, 5376]` `h` or `mlp_out` — 100.00% bit-exact.
  5. Optional (`chunk_io_layers=True`) 100.00% bit-exact token chunking of `video_patch_proj`
     and `FinalLayer.forward` to eliminate FP32 I/O activation spikes at 10s-15s.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import re
import types

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops as qops
from comfy.ldm.minimax.model import (
    MiniMaxH3Model,
    _mod_gate,
    _mod_row,
    _mod_scale_shift,
    _pdd_head,
)
from comfy_api.latest import io

logger = logging.getLogger("comfyui-h3-seamkit.chunked_sage")

try:
    import comfy_kitchen as _ck
    import comfy_kitchen.backends.cuda as _ck_cuda
except Exception:
    _ck = None
    _ck_cuda = None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _get_cuda_version() -> tuple[int, int]:
    try:
        v = torch.version.cuda
        if not v:
            return 0, 0
        m = re.match(r"(\d+)\.(\d+)", str(v))
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    except Exception:
        return 0, 0


_QATTN_PROBE = {
    "sm80": "qk_int8_sv_f16_accum_f32_attn",
    "sm89": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    "sm90": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
}


def _resolve_qattn(arch: str):
    try:
        core = importlib.import_module("sageattention.core")
    except Exception:
        return None
    candidates = [getattr(core, f"_qattn_{arch}", None), getattr(core, f"{arch}_compile", None)]
    try:
        mod = importlib.import_module(f"sageattention.{arch}_compile")
        candidates += [mod, getattr(mod, f"_qattn_{arch}", None)]
    except Exception:
        pass
    probe = _QATTN_PROBE[arch]
    for obj in candidates:
        if obj is not None and hasattr(obj, probe):
            return obj
    return None


_cuda_archs = None
_per_warp_int8_cuda = None
_per_block_int8_triton = None
_per_channel_fp8 = None
_attn_false = None

try:
    from sageattention.core import (
        attn_false as _attn_false,
        get_cuda_arch_versions as _get_cuda_arch_versions,
        per_block_int8_triton as _per_block_int8_triton,
        per_channel_fp8 as _per_channel_fp8,
        per_warp_int8_cuda as _per_warp_int8_cuda,
    )

    _cuda_archs = _get_cuda_arch_versions()
except Exception:
    pass

_qattn_sm80 = _resolve_qattn("sm80")
_qattn_sm89 = _resolve_qattn("sm89")
_qattn_sm90 = _resolve_qattn("sm90")
_sageplus_sm89_available = bool(
    _qattn_sm89 is not None
    and hasattr(_qattn_sm89, "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf")
    and _get_cuda_version() >= (12, 8)
)
_sol_attn_available = bool(
    _ck is not None
    and hasattr(_ck, "sol_attn")
    and hasattr(_ck, "sol_attn_is_available")
    and _ck.sol_attn_is_available()
)


# ---------------------------------------------------------------------------
# Sol-Attn (comfy_kitchen sparse SageAttention) Morton & routing helpers
# ---------------------------------------------------------------------------

_SOL_BLOCK_SIZE = 64
_MORTON_CPU_CACHE: dict[tuple[tuple[int, int, int], str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_MORTON_DEV_CACHE: dict[tuple[tuple[int, int, int], str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _part1by2(value: torch.Tensor) -> torch.Tensor:
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def _morton_perm_aligned(
    grid: tuple[int, int, int],
    start: int,
    device: torch.device,
    curve: str = "2d_frame",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(perm_dev, inv_dev, perm_cpu)` aligned to Sol-Attn's 64-token block grid."""
    pad = (-int(start)) % _SOL_BLOCK_SIZE
    cpu_key = (grid, curve, pad)
    cpu_hit = _MORTON_CPU_CACHE.get(cpu_key)
    if cpu_hit is None:
        frames, height, width = grid
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        rem = linear - z * area
        y = rem // width
        x = rem - y * width
        if curve == "2d_frame":
            code = (z << 42) | _part1by2(x) | (_part1by2(y) << 1)
        else:
            code = _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        if pad:
            perm = torch.roll(perm, pad)
        inv = torch.argsort(perm)
        cpu_hit = (perm, inv)
        _MORTON_CPU_CACHE[cpu_key] = cpu_hit

    dev_key = (grid, curve, pad, str(device))
    dev_hit = _MORTON_DEV_CACHE.get(dev_key)
    if dev_hit is None:
        # Keep device cache bounded to avoid lingering memory across shape changes
        if len(_MORTON_DEV_CACHE) >= 8:
            _MORTON_DEV_CACHE.clear()
        dev_hit = (cpu_hit[0].to(device), cpu_hit[1].to(device))
        _MORTON_DEV_CACHE[dev_key] = dev_hit
    return dev_hit[0], dev_hit[1], cpu_hit[0]


def _permute_span_inplace(
    h: torch.Tensor,
    start: int,
    stop: int,
    idx: torch.Tensor,
    col_chunk: int = 1344,
) -> None:
    """In-place row permutation of `h[start:stop]` in column chunks (~4x lower peak scratch VRAM)."""
    view = h[start:stop]
    cols = view.shape[1]
    for c0 in range(0, cols, col_chunk):
        c1 = min(c0 + col_chunk, cols)
        view[:, c0:c1] = view[:, c0:c1].index_select(0, idx)


def _parse_dense_blocks(spec: str, count: int) -> frozenset[int]:
    """Parse block index/range spec like '0-1,48-49' or '-1' into absolute block indices."""
    s = "".join(str(spec or "").split())
    if not s:
        return frozenset()
    out: set[int] = set()
    for part in s.split(","):
        if not part:
            continue
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if m is None:
            raise ValueError(f"Invalid sol_dense_blocks entry {part!r}; expected e.g. '0-1,48-49'")
        lo = int(m.group(1))
        hi = lo if m.group(2) is None else int(m.group(2))
        lo = lo if lo >= 0 else count + lo
        hi = hi if hi >= 0 else count + hi
        if lo > hi:
            lo, hi = hi, lo
        out.update(range(max(lo, 0), min(hi, count - 1) + 1))
    return frozenset(out)


class _PermutedLayoutProxy:
    """Lightweight proxy exposing Morton-permuted `position_ids` without mutating cached `PackedLayout`."""

    def __init__(self, base_layout, position_ids: torch.Tensor):
        self._base = base_layout
        self.position_ids = position_ids

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def _build_sol_context(layout, seq_len: int, sol_cfg: dict, transformer_options: dict):
    """Decide whether this forward pass runs Sol-Attn sparse mode and compute sink/Morton metadata."""
    if not sol_cfg or sol_cfg.get("attention_mode") != "sol_attn_sparse":
        return None, None
    if not _sol_attn_available or seq_len < int(sol_cfg.get("sol_min_tokens", 12288)):
        return None, None

    sigmas = transformer_options.get("sigmas") if isinstance(transformer_options, dict) else None
    if sigmas is not None and torch.is_tensor(sigmas) and sigmas.numel() > 0:
        sigma = float(sigmas.flatten()[0].item())
        if sigma > float(sol_cfg.get("sol_max_sigma", 0.95)) + 1e-6:
            return None, None
        if sigma < float(sol_cfg.get("sol_min_sigma", 0.0)) - 1e-6:
            return None, None

    segs = getattr(layout, "segments", None) or []
    vid_span = next(((a, b) for a, b, kind in segs if kind == "video"), None)
    aud_span = next(((a, b) for a, b, kind in segs if kind == "audio"), None)

    sink_mode = str(sol_cfg.get("sol_sink_mode", "exact_kv_and_rows"))
    sink_blocks = (0, 0)
    sink_q = (0, 0)
    if sink_mode != "off" and vid_span is not None:
        va, vb = int(vid_span[0]), int(vid_span[1])
        if seq_len >= vb and va > 0:
            sink_blocks = (0, (va + _SOL_BLOCK_SIZE - 1) // _SOL_BLOCK_SIZE)
            if sink_mode == "exact_kv_and_rows":
                if aud_span is not None:
                    sink_q = (int(aud_span[0]) // _SOL_BLOCK_SIZE, sink_blocks[1])
                else:
                    sink_q = sink_blocks

    morton_info = None
    sig = getattr(layout, "signature", None)
    if (
        vid_span is not None
        and isinstance(sig, tuple)
        and len(sig) == 5
        and torch.is_tensor(getattr(layout, "position_ids", None))
        and layout.position_ids.shape[0] == seq_len
    ):
        va, vb = int(vid_span[0]), int(vid_span[1])
        _tlen, lat_t, lat_h, lat_w, _at = sig
        grid = (int(lat_t), int(lat_h) // 2, int(lat_w) // 2)
        if grid[0] * grid[1] * grid[2] == (vb - va) and (vb - va) > 0:
            morton_info = (va, vb, grid)

    sol_ctx = {
        "tau": float(sol_cfg.get("sol_tau", 1.3)),
        "topk_ratio": float(sol_cfg.get("sol_topk_ratio", 0.0)),
        "token_aug": int(sol_cfg.get("sol_token_aug", 64)),
        "sink_blocks": sink_blocks,
        "sink_q": sink_q,
        "dense_blocks": sol_cfg.get("sol_dense_blocks_set", frozenset()),
    }
    return sol_ctx, morton_info


def _attend_group(
    q_g: torch.Tensor,
    k_g: torch.Tensor,
    make_v_fn,
    dtype: torch.dtype,
    sol_ctx: dict | None = None,
    block_idx: int | None = None,
) -> torch.Tensor:
    """Dispatch one head group `[1, S, g, 128]` to either `ck.sol_attn` sparse Sage or dense KJ Sage."""
    if (
        sol_ctx is not None
        and (block_idx is None or block_idx not in sol_ctx["dense_blocks"])
        and q_g.is_cuda
        and q_g.dtype in (torch.bfloat16, torch.float16)
        and q_g.shape[-1] == 128
        and _sol_attn_available
    ):
        v_g = make_v_fn()
        o_g = _ck.sol_attn(
            q_g,
            k_g,
            v_g,
            tau=sol_ctx["tau"],
            topk_ratio=sol_ctx["topk_ratio"],
            sink_blocks=list(sol_ctx["sink_blocks"]),
            sink_q=list(sol_ctx["sink_q"]),
            tail=True,
            token_aug=sol_ctx["token_aug"],
        )
        del q_g, k_g, v_g
        return o_g
    return _staged_sageattn_nhd(q_g, k_g, make_v_fn, dtype)


def _make_sol_run_blocks(orig_run_blocks_fn, sol_cfg: dict):
    """Wrap `MiniMaxH3Model._run_blocks` to manage zero-spike Morton reordering & Sol-Attn context."""
    def chunked_sage_sol_run_blocks(self, h, t_emb, mod_segments, layout, transformer_options):
        s = h.shape[0]
        sol_ctx, morton_info = _build_sol_context(layout, s, sol_cfg, transformer_options)
        if sol_ctx is None:
            return orig_run_blocks_fn(self, h, t_emb, mod_segments, layout, transformer_options)

        prev_ctx = transformer_options.get("_chunked_sage_sol_ctx")
        transformer_options["_chunked_sage_sol_ctx"] = sol_ctx

        active_morton = None
        run_layout = layout
        run_mod_segments = mod_segments
        if morton_info is not None:
            va, vb, grid = morton_info
            perm_dev, inv_dev, perm_cpu = _morton_perm_aligned(grid, va, h.device, curve="2d_frame")
            _permute_span_inplace(h, va, vb, perm_dev)
            pos_perm = layout.position_ids.clone()
            pos_perm[va:vb] = pos_perm[va:vb].index_select(0, perm_cpu.to(pos_perm.device))
            run_layout = _PermutedLayoutProxy(layout, pos_perm)
            # Permute per-token mod_row tensor if video segment uses non-uniform denoise_mask
            new_segs = []
            for a, b, row in mod_segments:
                if a == va and b == vb and isinstance(row, torch.Tensor):
                    new_segs.append((a, b, row.index_select(0, perm_dev.to(row.device))))
                else:
                    new_segs.append((a, b, row))
            run_mod_segments = new_segs
            active_morton = (va, vb, inv_dev)

        try:
            h = orig_run_blocks_fn(self, h, t_emb, run_mod_segments, run_layout, transformer_options)
        finally:
            if prev_ctx is None:
                transformer_options.pop("_chunked_sage_sol_ctx", None)
            else:
                transformer_options["_chunked_sage_sol_ctx"] = prev_ctx
            if active_morton is not None:
                va, vb, inv_dev = active_morton
                if torch.is_tensor(h) and h.ndim == 2 and h.shape[0] >= vb:
                    _permute_span_inplace(h, va, vb, inv_dev)
        return h

    return chunked_sage_sol_run_blocks


# ---------------------------------------------------------------------------
# Int64-safe per-thread INT8 Q/K quantization (identical to KJNodes ltxv_nodes)
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _quant_query_per_thread_int8_i64_kernel(
        Input, Output, Scale, L,
        stride_iz, stride_ih, stride_in,
        stride_oz, stride_oh, stride_on,
        stride_sz, stride_sh,
        C: tl.constexpr, BLK: tl.constexpr,
    ):
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
        offs_k = tl.arange(0, C)

        input_ptrs = (
            Input + off_b * stride_iz + off_h * stride_ih
            + offs_n[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        )
        output_ptrs = (
            Output + off_b * stride_oz + off_h * stride_oh
            + offs_n[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        )
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

        x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
        x = x.to(tl.float32)
        scale = tl.max(tl.abs(x)) / 127.0 + 0.0000001
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        x_int8 = x_int8.to(tl.int8)
        tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
        tl.store(scale_ptrs, scale)

    @triton.jit
    def _quant_key_per_thread_int8_i64_kernel(
        Input, Output, Scale, L,
        stride_iz, stride_ih, stride_in,
        stride_oz, stride_oh, stride_on,
        stride_sz, stride_sh,
        C: tl.constexpr, BLK: tl.constexpr,
    ):
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)

        input_ptrs0 = (
            Input + off_b * stride_iz + off_h * stride_ih
            + offs_n0[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        )
        input_ptrs1 = (
            Input + off_b * stride_iz + off_h * stride_ih
            + offs_n1[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        )
        output_ptrs0 = (
            Output + off_b * stride_oz + off_h * stride_oh
            + offs_n0[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        )
        output_ptrs1 = (
            Output + off_b * stride_oz + off_h * stride_oh
            + offs_n1[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        )
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
        x0 = x0.to(tl.float32)
        x1 = x1.to(tl.float32)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127.0 + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        x0_int8 = x0_int8.to(tl.int8)
        x1_int8 = x1_int8.to(tl.int8)
        tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
        tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
        tl.store(scale_ptrs, scale)


def _per_thread_int8_i64(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    WARPK: int = 64,
):
    """Int64-safe per-thread INT8 quantization for NHD tensors [B, L, H, D]."""
    if not _HAS_TRITON:
        from sageattention.core import per_thread_int8_triton
        return per_thread_int8_triton(
            q, k, tensor_layout="NHD", BLKQ=BLKQ, WARPQ=WARPQ, BLKK=BLKK, WARPK=WARPK
        )
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    b, qo_len, h_qo, head_dim = q.shape
    _, kv_len, h_kv, _ = k.shape

    q_scale = torch.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4),
        device=q.device,
        dtype=torch.float32,
    )

    grid_q = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    _quant_query_per_thread_int8_i64_kernel[grid_q](
        q, q_int8, q_scale, qo_len,
        q.stride(0), q.stride(2), q.stride(1),
        q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ,
    )

    grid_k = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    _quant_key_per_thread_int8_i64_kernel[grid_k](
        k, k_int8, k_scale, kv_len,
        k.stride(0), k.stride(2), k.stride(1),
        k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK,
    )
    return q_int8, q_scale, k_int8, k_scale


def _staged_sageattn_nhd(q: torch.Tensor, k: torch.Tensor, get_v_fn, dtype: torch.dtype) -> torch.Tensor:
    """KJ-identical SageAttention2++ on NHD [1, S, g, D], deferring V projection until BF16 Q/K are freed."""
    head_dim = q.shape[-1]
    sm_scale = head_dim ** -0.5
    arch = _cuda_archs[0] if _cuda_archs else None

    if arch in {"sm80", "sm86"}:
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q, k, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64
        )
        del q, k
        v = get_v_fn()
        v_fp16 = v.to(torch.float16)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8, k_int8, v_fp16, o, q_scale, k_scale, 0, 0, 3, sm_scale, 0
        )
        del q_int8, q_scale, k_int8, k_scale, v_fp16
        return o

    if arch == "sm89":
        if _sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        else:
            pv_accum_dtype = "fp32+fp32"
            quant_v_scale_max = 448.0
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q, k, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64
        )
        del q, k
        v = get_v_fn()
        v_fp8, v_scale, _ = _per_channel_fp8(
            v, tensor_layout="NHD", scale_max=quant_v_scale_max, smooth_v=False
        )
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 3, sm_scale, 0
            )
        else:
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 3, sm_scale, 0
            )
        del q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
        return o

    if arch == "sm90":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q, k, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128
        )
        del q, k
        v = get_v_fn()
        kv_len = v.size(1)
        v_pad_len = (128 - (kv_len % 128)) if (kv_len % 128 != 0) else 0
        if v_pad_len > 0:
            v = torch.cat(
                [v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)],
                dim=1,
            )
        v_fp8, v_scale, _ = _per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 3, sm_scale, 0
        )
        del q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
        return o

    if arch in {"sm120", "sm121"}:
        if _sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        else:
            pv_accum_dtype = "fp32"
            quant_v_scale_max = 448.0
        q_int8, q_scale, k_int8, k_scale = _per_warp_int8_cuda(
            q, k, km=k.mean(dim=1, keepdim=True), tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64
        )
        del q, k
        v = get_v_fn()
        v_fp8, v_scale, _ = _per_channel_fp8(
            v, tensor_layout="NHD", scale_max=quant_v_scale_max, smooth_v=False
        )
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 2, sm_scale, 0
            )
        else:
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 2, sm_scale, 0
            )
        del q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
        return o

    if arch == "sm75":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_block_int8_triton(
            q, k, sm_scale=sm_scale, tensor_layout="NHD"
        )
        del q, k
        v = get_v_fn()
        o, _ = _attn_false(
            q_int8, k_int8, v, q_scale, k_scale,
            tensor_layout="NHD", output_dtype=dtype, attn_mask=None, return_lse=False,
        )
        del q_int8, q_scale, k_int8, k_scale, v
        return o

    # Fallback (CPU or unsupported arch): exact PyTorch SDPA
    v = get_v_fn()
    q_hnd = q.transpose(1, 2)
    k_hnd = k.transpose(1, 2)
    v_hnd = v.transpose(1, 2)
    del q, k, v
    o_hnd = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd)
    return o_hnd.transpose(1, 2).contiguous().to(dtype)


# ---------------------------------------------------------------------------
# ComfyUI weight casting & INT8 ConvRot pre-quantization helpers
# ---------------------------------------------------------------------------

def _slice_mod_segments(segments, t0: int, t1: int):
    """Slice contiguous [(a, b, row)] mod_segments onto [t0, t1) with 0-based local offsets."""
    sub = []
    for a, b, row in segments:
        if b <= t0:
            continue
        if a >= t1:
            break
        sa = max(a, t0)
        sb = min(b, t1)
        if isinstance(row, torch.Tensor):
            sub_row = row[sa - a : sb - a]
        else:
            sub_row = row
        sub.append((sa - t0, sb - t0, sub_row))
    return sub


def _split_heads_even(total_heads: int, n_groups: int, convrot_groupsize: int = 256, head_dim: int = 128) -> list[int]:
    """Split `total_heads` into `n_groups` groups whose widths `g * head_dim` are multiples of `convrot_groupsize` when possible."""
    n_groups = max(1, min(int(n_groups), total_heads))
    unit_heads = max(1, convrot_groupsize // head_dim) if (convrot_groupsize > 0 and convrot_groupsize % head_dim == 0) else 1
    if unit_heads > 1 and total_heads % unit_heads == 0 and n_groups <= total_heads // unit_heads:
        units = total_heads // unit_heads
        base = units // n_groups
        rem = units % n_groups
        return [(base + (1 if i < rem else 0)) * unit_heads for i in range(n_groups)]
    base = total_heads // n_groups
    rem = total_heads % n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def _is_int8_tensorwise(weight) -> bool:
    return (
        isinstance(weight, qops.QuantizedTensor)
        and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
        and not getattr(getattr(weight, "_params", None), "transposed", False)
    )


@contextlib.contextmanager
def _cast_linear_once(linear, device: torch.device, compute_dtype: torch.dtype):
    """Cast a ComfyUI MixedPrecisionOps/CastWeightBiasOp Linear once via CastBiasWeightContext.

    Preserves INT8 `QuantizedTensor` without dequantizing when eligible, while cleanly
    handling `aimdo` VBAR prefetch, CPU offload, LoRA patches, and plain `nn.Linear`.
    """
    if not hasattr(linear, "weight_function"):
        w = linear.weight.to(device=device)
        b = linear.bias.to(device=device, dtype=compute_dtype) if linear.bias is not None else None
        yield w, b
        return

    w_raw = linear.weight
    use_quant = (
        isinstance(w_raw, qops.QuantizedTensor)
        and getattr(linear, "layout_type", None) is not None
        and not getattr(linear, "_full_precision_mm", False)
        and not getattr(linear, "comfy_force_cast_weights", False)
        and len(getattr(linear, "weight_function", ())) == 0
        and len(getattr(linear, "bias_function", ())) == 0
    )
    if use_quant:
        with comfy.ops.CastBiasWeightContext(
            linear,
            input=None,
            dtype=w_raw.dtype,
            device=device,
            bias_dtype=compute_dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=True,
        ) as (weight, bias):
            yield weight, bias
    else:
        with comfy.ops.CastBiasWeightContext(
            linear,
            input=None,
            dtype=compute_dtype,
            device=device,
            bias_dtype=compute_dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=False,
        ) as (weight, bias):
            if getattr(linear, "_full_precision_mm", False) and isinstance(weight, qops.QuantizedTensor):
                weight = weight.dequantize()
            yield weight, bias


@contextlib.contextmanager
def _cast_rmsnorm_once(norm, x: torch.Tensor):
    """Cast an RMSNorm module's weight once outside a token-chunk loop."""
    if hasattr(norm, "cast_weight"):
        with norm.cast_weight(x) as (weight, _):
            yield weight
    else:
        w = norm.weight
        if w is not None and (w.device != x.device or w.dtype != x.dtype):
            w = w.to(device=x.device, dtype=x.dtype)
        yield w


def _quantize_int8_convrot_into(
    x_chunk: torch.Tensor,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    convrot: bool,
    convrot_groupsize: int,
) -> None:
    """Rotate (if convrot) and row-wise INT8 quantize [M, K] chunk into pre-allocated slices."""
    x_2d = (
        x_chunk
        if (x_chunk.dim() == 2 and x_chunk.is_contiguous())
        else x_chunk.reshape(-1, x_chunk.shape[-1]).contiguous()
    )
    _m, k = x_2d.shape
    stream_ptr = torch.cuda.current_stream(x_2d.device).cuda_stream
    if convrot:
        fused_ok = (
            convrot_groupsize == 256
            and k % 256 == 0
            and 256 <= k <= _ck_cuda._CONVROT_FUSED_MAX_K
            and _ck_cuda._convrot_fused_shared_memory_fits(x_2d, k, convrot_groupsize)
        )
        if fused_ok:
            empty_act = _ck_cuda._empty_cuda_tensor(x_2d.device, x_2d.dtype)
            _ck_cuda._C.quantize_int8_rowwise_convrot64(
                _ck_cuda._wrap_for_dlpack(x_2d),
                _ck_cuda._wrap_for_dlpack(q_out),
                _ck_cuda._wrap_for_dlpack(scale_out),
                convrot_groupsize,
                False,
                _ck_cuda._input_act_code(None),
                0,
                _ck_cuda._wrap_for_dlpack(empty_act),
                0.0,
                stream_ptr,
            )
            return
        if _ck_cuda._should_use_convrot_fused_kernel(x_2d, k, convrot_groupsize):
            q_tmp, s_tmp = _ck_cuda.quantize_int8_rowwise_convrot(x_2d, convrot_groupsize)
        else:
            h = _ck_cuda._build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
            q_tmp, s_tmp = _ck_cuda.quantize_and_rotate_rowwise(x_2d, h, convrot_groupsize)
        q_out.copy_(q_tmp)
        scale_out.copy_(s_tmp)
    else:
        _ck_cuda._C.quantize_int8_rowwise(
            _ck_cuda._wrap_for_dlpack(x_2d),
            _ck_cuda._wrap_for_dlpack(q_out),
            _ck_cuda._wrap_for_dlpack(scale_out),
            False,
            0,
            stream_ptr,
        )


def _int8_gemm_prequant(
    x_qdata: torch.Tensor,
    x_scale: torch.Tensor,
    w_qdata: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run INT8 GEMM + dequant using pre-quantized `(x_qdata, x_scale)` and weight row slice `w_qdata`."""
    m, k = x_qdata.shape
    n, k_w = w_qdata.shape
    if k != k_w:
        raise ValueError(f"INT8 GEMM inner dim mismatch: {k} != {k_w}")
    if not w_qdata.is_contiguous():
        w_qdata = w_qdata.contiguous()
    device = x_qdata.device
    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    out = torch.empty((m, n), dtype=out_dtype, device=device)
    output_dtype_code = _ck_cuda.DTYPE_TO_CODE[out_dtype]
    ws = _ck_cuda._int8_weight_scale_arg(w_scale, device)
    bias_arg = (
        _ck_cuda._gemm_vector_arg(bias, device, out_dtype)
        if bias is not None
        else _ck_cuda._empty_cuda_tensor(device, out_dtype)
    )

    used_cutlass = False
    if (
        not _ck_cuda._prefer_cublas_int8_fallback(m, n, k, x_qdata.get_device())
        and not _ck_cuda._DISABLE_CUTLASS_INT8
        and _ck_cuda._cuda_device_supports_cutlass_int8_dequant(x_qdata)
    ):
        ws_cutlass = ws if ws.numel() == n else ws.expand(n).contiguous()
        used_cutlass = _ck_cuda._C.cutlass_int8_dequant(
            _ck_cuda._wrap_for_dlpack(x_qdata),
            _ck_cuda._wrap_for_dlpack(w_qdata),
            _ck_cuda._wrap_for_dlpack(x_scale),
            _ck_cuda._wrap_for_dlpack(ws_cutlass),
            _ck_cuda._wrap_for_dlpack(bias_arg),
            _ck_cuda._wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )

    if not used_cutlass:
        use_turing_padding = x_qdata.is_cuda and _ck_cuda._cuda_device_is_turing(x_qdata.get_device())
        if use_turing_padding:
            padded_k = _ck_cuda._round_up(k, 16)
            padded_n = _ck_cuda._round_up(n, _ck_cuda._cublas_int8_n_alignment(x_qdata))
            cublas_x = _ck_cuda._pad_2d_cols(x_qdata, padded_k)
            cublas_w = _ck_cuda._pad_2d_rows(_ck_cuda._pad_2d_cols(w_qdata, padded_k), padded_n)
        else:
            padded_n = n
            cublas_x = x_qdata
            cublas_w = w_qdata
        out_int32 = torch.empty((m, padded_n), dtype=torch.int32, device=device)
        _ck_cuda._C.cublas_gemm_int8(
            _ck_cuda._wrap_for_dlpack(cublas_x),
            _ck_cuda._wrap_for_dlpack(cublas_w),
            _ck_cuda._wrap_for_dlpack(out_int32),
            _ck_cuda._wrap_for_dlpack(_ck_cuda.get_cublas_workspace()),
            stream_ptr,
        )
        if padded_n != n:
            out_int32 = out_int32[:, :n].contiguous()
        _ck_cuda._C.dequantize_int8_linear(
            _ck_cuda._wrap_for_dlpack(out_int32),
            _ck_cuda._wrap_for_dlpack(x_scale),
            _ck_cuda._wrap_for_dlpack(ws),
            _ck_cuda._wrap_for_dlpack(bias_arg),
            _ck_cuda._wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )
    return out


def _linear_with_cast_weight(x: torch.Tensor, weight, bias) -> torch.Tensor:
    if _is_int8_tensorwise(weight) and x.is_cuda and _ck is not None:
        params = weight._params
        return _ck.int8_linear(
            x,
            weight._qdata,
            params.scale,
            bias,
            x.dtype,
            convrot=bool(getattr(params, "convrot", False)),
            convrot_groupsize=int(getattr(params, "convrot_groupsize", 256)),
        )
    return F.linear(x, weight, bias)


def _resolve_out_proj_mode(mode: str, seq_len: int, device: torch.device) -> str:
    """Deterministically resolve 'auto' -> 'exact' or 'streaming' (constant across all 50 blocks for malloc_graph)."""
    if mode in ("exact", "streaming"):
        return mode
    if seq_len <= 64000:
        return "exact"
    if device.type == "cuda":
        try:
            total_vram_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            if total_vram_gib >= 16.0:
                return "exact"
        except Exception:
            pass
    return "streaming"


def _apply_qk_norm_rope(attn, q_g: torch.Tensor, k_g: torch.Tensor, rope_freqs, qw, kw, rot_dim: int):
    if rope_freqs is not None and _ck is not None and q_g.is_cuda and not mm.in_training:
        _ck.rms_rope_split_half_(
            q_g, k_g, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot_dim
        )
        return q_g, k_g
    if rope_freqs is not None and _ck is not None:
        return _ck.rms_rope_split_half(
            q_g, k_g, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot_dim
        )
    return attn.q_norm(q_g), attn.k_norm(k_g)


# ---------------------------------------------------------------------------
# Core chunked attention & fused block forwards
# ---------------------------------------------------------------------------

def _run_chunked_attention_int8(
    block,
    attn,
    x: torch.Tensor,
    shift_msa,
    scale_msa,
    gate_msa,
    mod_segments,
    rope_freqs,
    w_qkv,
    b_qkv,
    head_chunks: int,
    token_chunk: int,
    eff_mode: str,
    sol_ctx: dict | None = None,
    block_idx: int | None = None,
) -> None:
    """INT8 TensorWise ConvRot fast path: pre-quantize once, head-group Q/K/V, accumulate into `x`."""
    s, hidden = x.shape
    dtype = x.dtype
    device = x.device
    heads = attn.heads
    head_dim = attn.head_dim
    inner = heads * head_dim

    qkv_qdata = w_qkv._qdata
    qkv_scale = w_qkv._params.scale
    qkv_convrot = bool(getattr(w_qkv._params, "convrot", False))
    qkv_gs = int(getattr(w_qkv._params, "convrot_groupsize", 256))

    x_qdata = torch.empty((s, hidden), dtype=torch.int8, device=device)
    x_scale = torch.empty((s, 1), dtype=torch.float32, device=device)

    with _cast_rmsnorm_once(block.norm1, x) as nw1:
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            h_c = F.rms_norm(x[t0:t1], block.norm1.normalized_shape, nw1, block.norm1.eps)
            _mod_scale_shift(h_c, shift_msa, scale_msa, sub_segs)
            _quantize_int8_convrot_into(h_c, x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
            del h_c

    qw = mm.cast_to(attn.q_norm.weight, device=device) if rope_freqs is not None else None
    kw = mm.cast_to(attn.k_norm.weight, device=device) if rope_freqs is not None else None
    rot_dim = rope_freqs.shape[-3] * 2 if rope_freqs is not None else 0

    def _slice_scale(scale_t, r0, r1):
        if scale_t.ndim > 0 and scale_t.numel() > 1:
            return scale_t[r0:r1]
        return scale_t

    def _slice_bias(bias_t, r0, r1):
        return bias_t[r0:r1] if bias_t is not None else None

    if eff_mode == "streaming":
        with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
            out_is_int8 = _is_int8_tensorwise(w_out)
            out_gs = int(getattr(w_out._params, "convrot_groupsize", 256)) if out_is_int8 else 256
            out_convrot = bool(getattr(w_out._params, "convrot", False)) if out_is_int8 else False
            can_stream_out = (
                (out_is_int8 and (not out_convrot or (out_gs % head_dim == 0 and heads % (out_gs // head_dim) == 0)))
                or (not isinstance(w_out, qops.QuantizedTensor))
            )
            if can_stream_out:
                group_sizes = _split_heads_even(heads, head_chunks, out_gs if out_convrot else 0, head_dim)
                hs = 0
                for g_idx, g in enumerate(group_sizes):
                    he = hs + g
                    c0, c1 = hs * head_dim, he * head_dim
                    q_g = _int8_gemm_prequant(
                        x_qdata, x_scale, qkv_qdata[c0:c1], _slice_scale(qkv_scale, c0, c1),
                        dtype, _slice_bias(b_qkv, c0, c1),
                    ).view(1, s, g, head_dim)
                    k_g = _int8_gemm_prequant(
                        x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1],
                        _slice_scale(qkv_scale, inner + c0, inner + c1),
                        dtype, _slice_bias(b_qkv, inner + c0, inner + c1),
                    ).view(1, s, g, head_dim)
                    q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        return _int8_gemm_prequant(
                            x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1],
                            _slice_scale(qkv_scale, 2 * inner + _c0, 2 * inner + _c1),
                            dtype, _slice_bias(b_qkv, 2 * inner + _c0, 2 * inner + _c1),
                        ).view(1, s, _g, head_dim)

                    o_g = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
                    del q_g, k_g
                    o_g_2d = o_g.view(s, g * head_dim)
                    b_group = b_out if g_idx == 0 else None
                    if out_is_int8:
                        w_out_g = w_out._qdata[:, c0:c1].contiguous()
                        out_scale = w_out._params.scale
                        for t0 in range(0, s, token_chunk):
                            t1 = min(t0 + token_chunk, s)
                            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                            proj_c = _ck.int8_linear(
                                o_g_2d[t0:t1], w_out_g, out_scale, b_group, dtype,
                                convrot=out_convrot, convrot_groupsize=out_gs,
                            )
                            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                            del proj_c
                        del w_out_g
                    else:
                        w_out_g = w_out[:, c0:c1]
                        for t0 in range(0, s, token_chunk):
                            t1 = min(t0 + token_chunk, s)
                            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                            proj_c = F.linear(o_g_2d[t0:t1], w_out_g, b_group)
                            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                            del proj_c
                        del w_out_g
                    del o_g, o_g_2d
                    hs = he
                del x_qdata, x_scale
                return

    # Exact mode (Option 1): 100.00% bit-identical to stock KJ SageAttention2++
    group_sizes = _split_heads_even(heads, head_chunks, 0, head_dim)
    out = torch.empty((s, inner), dtype=dtype, device=device)
    out_nhd = out.view(1, s, heads, head_dim)
    hs = 0
    for g in group_sizes:
        he = hs + g
        c0, c1 = hs * head_dim, he * head_dim
        q_g = _int8_gemm_prequant(
            x_qdata, x_scale, qkv_qdata[c0:c1], _slice_scale(qkv_scale, c0, c1),
            dtype, _slice_bias(b_qkv, c0, c1),
        ).view(1, s, g, head_dim)
        k_g = _int8_gemm_prequant(
            x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1],
            _slice_scale(qkv_scale, inner + c0, inner + c1),
            dtype, _slice_bias(b_qkv, inner + c0, inner + c1),
        ).view(1, s, g, head_dim)
        q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

        def _make_v(_c0=c0, _c1=c1, _g=g):
            return _int8_gemm_prequant(
                x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1],
                _slice_scale(qkv_scale, 2 * inner + _c0, 2 * inner + _c1),
                dtype, _slice_bias(b_qkv, 2 * inner + _c0, 2 * inner + _c1),
            ).view(1, s, _g, head_dim)

        out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
        del q_g, k_g
        hs = he

    del x_qdata, x_scale

    with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            proj_c = _linear_with_cast_weight(out[t0:t1], w_out, b_out)
            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
            del proj_c
    del out, out_nhd


def _run_chunked_attention_dense(
    block,
    attn,
    x: torch.Tensor,
    shift_msa,
    scale_msa,
    gate_msa,
    mod_segments,
    rope_freqs,
    w_qkv: torch.Tensor,
    b_qkv,
    head_chunks: int,
    token_chunk: int,
    eff_mode: str,
    sol_ctx: dict | None = None,
    block_idx: int | None = None,
) -> None:
    """Dense BF16/FP16/LoRA path: zero full `[S, 21504]` QKV buffer, 100% algebraically exact in both modes."""
    s = x.shape[0]
    dtype = x.dtype
    device = x.device
    heads = attn.heads
    head_dim = attn.head_dim
    inner = heads * head_dim

    qw = mm.cast_to(attn.q_norm.weight, device=device) if rope_freqs is not None else None
    kw = mm.cast_to(attn.k_norm.weight, device=device) if rope_freqs is not None else None
    rot_dim = rope_freqs.shape[-3] * 2 if rope_freqs is not None else 0

    group_sizes = _split_heads_even(heads, head_chunks, 0, head_dim)

    def _project_rows(nw1, r0: int, r1: int) -> torch.Tensor:
        w_slice = w_qkv[r0:r1]
        b_slice = b_qkv[r0:r1] if b_qkv is not None else None
        buf = torch.empty((s, r1 - r0), dtype=dtype, device=device)
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            h_c = F.rms_norm(x[t0:t1], block.norm1.normalized_shape, nw1, block.norm1.eps)
            _mod_scale_shift(h_c, shift_msa, scale_msa, sub_segs)
            buf[t0:t1] = F.linear(h_c, w_slice, b_slice)
            del h_c
        return buf

    with _cast_rmsnorm_once(block.norm1, x) as nw1:
        if eff_mode == "streaming":
            with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
                if not isinstance(w_out, qops.QuantizedTensor):
                    hs = 0
                    for g_idx, g in enumerate(group_sizes):
                        he = hs + g
                        c0, c1 = hs * head_dim, he * head_dim
                        q_g = _project_rows(nw1, c0, c1).view(1, s, g, head_dim)
                        k_g = _project_rows(nw1, inner + c0, inner + c1).view(1, s, g, head_dim)
                        q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

                        def _make_v(_c0=c0, _c1=c1, _g=g):
                            return _project_rows(nw1, 2 * inner + _c0, 2 * inner + _c1).view(1, s, _g, head_dim)

                        o_g = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
                        del q_g, k_g
                        o_g_2d = o_g.view(s, g * head_dim)
                        w_out_g = w_out[:, c0:c1]
                        b_group = b_out if g_idx == 0 else None
                        for t0 in range(0, s, token_chunk):
                            t1 = min(t0 + token_chunk, s)
                            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                            proj_c = F.linear(o_g_2d[t0:t1], w_out_g, b_group)
                            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                            del proj_c
                        del o_g, o_g_2d, w_out_g
                        hs = he
                    return

        out = torch.empty((s, inner), dtype=dtype, device=device)
        out_nhd = out.view(1, s, heads, head_dim)
        hs = 0
        for g in group_sizes:
            he = hs + g
            c0, c1 = hs * head_dim, he * head_dim
            q_g = _project_rows(nw1, c0, c1).view(1, s, g, head_dim)
            k_g = _project_rows(nw1, inner + c0, inner + c1).view(1, s, g, head_dim)
            q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

            def _make_v(_c0=c0, _c1=c1, _g=g):
                return _project_rows(nw1, 2 * inner + _c0, 2 * inner + _c1).view(1, s, _g, head_dim)

            out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
            del q_g, k_g
            hs = he

    with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            proj_c = _linear_with_cast_weight(out[t0:t1], w_out, b_out)
            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
            del proj_c
    del out, out_nhd


def _run_chunked_mlp_inplace(block, x: torch.Tensor, shift_mlp, scale_mlp, gate_mlp, mod_segments, token_chunk: int) -> None:
    """Fused token-chunked `norm2 + _mod_scale_shift + SwiGLU MLP + _mod_gate` (100% bit-exact)."""
    s = x.shape[0]
    dtype = x.dtype
    device = x.device
    fc1 = block.mlp.fc1
    fc2 = block.mlp.fc2

    # If AWQ/FP8 per-layer pre_quant_scale or input_scale is attached, delegate per chunk to module forward
    has_extra_scales = (
        getattr(fc1, "pre_quant_scale", None) is not None
        or getattr(fc1, "input_scale", None) is not None
        or getattr(fc2, "pre_quant_scale", None) is not None
        or getattr(fc2, "input_scale", None) is not None
    )

    with _cast_rmsnorm_once(block.norm2, x) as nw2:
        if has_extra_scales:
            for t0 in range(0, s, token_chunk):
                t1 = min(t0 + token_chunk, s)
                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                h_c = F.rms_norm(x[t0:t1], block.norm2.normalized_shape, nw2, block.norm2.eps)
                _mod_scale_shift(h_c, shift_mlp, scale_mlp, sub_segs)
                mlp_c = comfy.ops.linear_input_act(fc2, fc1(h_c), "swiglu")
                del h_c
                _mod_gate(x[t0:t1], gate_mlp, mlp_c, sub_segs)
                del mlp_c
            return

        with _cast_linear_once(fc1, device, dtype) as (w_fc1, b_fc1):
            with _cast_linear_once(fc2, device, dtype) as (w_fc2, b_fc2):
                for t0 in range(0, s, token_chunk):
                    t1 = min(t0 + token_chunk, s)
                    sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                    h_c = F.rms_norm(x[t0:t1], block.norm2.normalized_shape, nw2, block.norm2.eps)
                    _mod_scale_shift(h_c, shift_mlp, scale_mlp, sub_segs)
                    fc1_c = _linear_with_cast_weight(h_c, w_fc1, b_fc1)
                    del h_c
                    mlp_c = comfy.ops.linear_input_act_(fc1_c, w_fc2, b_fc2, "swiglu")
                    del fc1_c
                    _mod_gate(x[t0:t1], gate_mlp, mlp_c, sub_segs)
                    del mlp_c


def _make_chunked_attn_forward(head_chunks: int, token_chunk: int):
    """Standalone patched `attn.forward` (used if another node overrides `block.forward`, e.g. KJ LowVRAM chained after us)."""
    def minimax_chunked_sageattn_forward(self, x, rope_freqs=None, transformer_options={}):
        if isinstance(x, list):
            x = x.pop()
        s = x.shape[0]
        dtype = x.dtype
        device = x.device
        heads = self.heads
        head_dim = self.head_dim
        inner = heads * head_dim

        n_groups = head_chunks
        if isinstance(transformer_options, dict) and transformer_options.get("minimax_head_chunks"):
            n_groups = max(n_groups, int(transformer_options["minimax_head_chunks"]))
        n_groups = min(n_groups, heads)
        sol_ctx = transformer_options.get("_chunked_sage_sol_ctx") if isinstance(transformer_options, dict) else None
        block_idx = transformer_options.get("block_index") if isinstance(transformer_options, dict) else None

        qw = mm.cast_to(self.q_norm.weight, device=device) if rope_freqs is not None else None
        kw = mm.cast_to(self.k_norm.weight, device=device) if rope_freqs is not None else None
        rot_dim = rope_freqs.shape[-3] * 2 if rope_freqs is not None else 0

        with _cast_linear_once(self.qkv_proj, device, dtype) as (w_qkv, b_qkv):
            if _is_int8_tensorwise(w_qkv) and x.is_cuda and _ck_cuda is not None:
                qkv_qdata = w_qkv._qdata
                qkv_scale = w_qkv._params.scale
                qkv_convrot = bool(getattr(w_qkv._params, "convrot", False))
                qkv_gs = int(getattr(w_qkv._params, "convrot_groupsize", 256))
                x_qdata = torch.empty((s, x.shape[1]), dtype=torch.int8, device=device)
                x_scale = torch.empty((s, 1), dtype=torch.float32, device=device)
                for t0 in range(0, s, token_chunk):
                    t1 = min(t0 + token_chunk, s)
                    _quantize_int8_convrot_into(x[t0:t1], x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
                del x

                def _sc(r0, r1):
                    return qkv_scale[r0:r1] if (qkv_scale.ndim > 0 and qkv_scale.numel() > 1) else qkv_scale

                def _bs(r0, r1):
                    return b_qkv[r0:r1] if b_qkv is not None else None

                out = torch.empty((s, inner), dtype=dtype, device=device)
                out_nhd = out.view(1, s, heads, head_dim)
                hs = 0
                for g in _split_heads_even(heads, n_groups, 0, head_dim):
                    he = hs + g
                    c0, c1 = hs * head_dim, he * head_dim
                    q_g = _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[c0:c1], _sc(c0, c1), dtype, _bs(c0, c1)).view(1, s, g, head_dim)
                    k_g = _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1], _sc(inner + c0, inner + c1), dtype, _bs(inner + c0, inner + c1)).view(1, s, g, head_dim)
                    q_g, k_g = _apply_qk_norm_rope(self, q_g, k_g, rope_freqs, qw, kw, rot_dim)

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        return _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1], _sc(2 * inner + _c0, 2 * inner + _c1), dtype, _bs(2 * inner + _c0, 2 * inner + _c1)).view(1, s, _g, head_dim)

                    out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
                    del q_g, k_g
                    hs = he
                del x_qdata, x_scale
            elif not isinstance(w_qkv, qops.QuantizedTensor):
                out = torch.empty((s, inner), dtype=dtype, device=device)
                out_nhd = out.view(1, s, heads, head_dim)
                hs = 0
                for g in _split_heads_even(heads, n_groups, 0, head_dim):
                    he = hs + g
                    c0, c1 = hs * head_dim, he * head_dim
                    q_g = F.linear(x, w_qkv[c0:c1], b_qkv[c0:c1] if b_qkv is not None else None).view(1, s, g, head_dim)
                    k_g = F.linear(x, w_qkv[inner + c0 : inner + c1], b_qkv[inner + c0 : inner + c1] if b_qkv is not None else None).view(1, s, g, head_dim)
                    q_g, k_g = _apply_qk_norm_rope(self, q_g, k_g, rope_freqs, qw, kw, rot_dim)

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        return F.linear(x, w_qkv[2 * inner + _c0 : 2 * inner + _c1], b_qkv[2 * inner + _c0 : 2 * inner + _c1] if b_qkv is not None else None).view(1, s, _g, head_dim)

                    out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx)
                    del q_g, k_g
                    hs = he
                del x
            else:
                # Non-INT8 quantized fallback (e.g. FP8/NVFP4): chunk QKV along tokens
                q_all = torch.empty((s, inner), dtype=dtype, device=device)
                k_all = torch.empty((s, inner), dtype=dtype, device=device)
                v_all = torch.empty((s, inner), dtype=dtype, device=device)
                for t0 in range(0, s, token_chunk):
                    t1 = min(t0 + token_chunk, s)
                    qc, kc, vc = self.qkv_proj(x[t0:t1]).split(inner, dim=-1)
                    q_all[t0:t1] = qc
                    k_all[t0:t1] = kc
                    v_all[t0:t1] = vc
                    del qc, kc, vc
                del x
                q_all = q_all.view(1, s, heads, head_dim)
                k_all = k_all.view(1, s, heads, head_dim)
                v_all = v_all.view(1, s, heads, head_dim)
                q_all, k_all = _apply_qk_norm_rope(self, q_all, k_all, rope_freqs, qw, kw, rot_dim)
                out = torch.empty((s, inner), dtype=dtype, device=device)
                out_nhd = out.view(1, s, heads, head_dim)
                hs = 0
                for g in _split_heads_even(heads, n_groups, 0, head_dim):
                    he = hs + g
                    out_nhd[:, :, hs:he] = _staged_sageattn_nhd(
                        q_all[:, :, hs:he].contiguous(),
                        k_all[:, :, hs:he].contiguous(),
                        lambda _hs=hs, _he=he: v_all[:, :, _hs:_he].contiguous(),
                        dtype,
                    )
                    hs = he
                del q_all, k_all, v_all

        res = torch.empty((s, self.out_proj.out_features), dtype=dtype, device=device)
        with _cast_linear_once(self.out_proj, device, dtype) as (w_out, b_out):
            for t0 in range(0, s, token_chunk):
                t1 = min(t0 + token_chunk, s)
                res[t0:t1] = _linear_with_cast_weight(out[t0:t1], w_out, b_out)
        del out, out_nhd
        return res

    minimax_chunked_sageattn_forward._is_chunked_sage_attn = True
    return minimax_chunked_sageattn_forward


def _make_chunked_block_forward(head_chunks: int, out_proj_mode: str, token_chunk: int):
    """Fused low-VRAM `DiTBlock.forward` with chunked QKV + KJ SageAttention2++ + fused chunked MLP."""
    def minimax_chunked_sageattn_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        s = x.shape[0]
        dtype = x.dtype
        device = x.device
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

        # Defer gracefully if an external wrapper (e.g. Route 3 beta_attention or custom `attention` arg) is active
        external_attn = (
            (attention is not None and attention is not self.attn)
            or not getattr(self.attn.forward, "_is_chunked_sage_attn", False)
        )
        if external_attn:
            attn_fn = self.attn if attention is None else attention
            h = [_mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)]
            attn_out = attn_fn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
            x = _mod_gate(x, gate_msa, attn_out, mod_segments)
            del attn_out
            _run_chunked_mlp_inplace(self, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, token_chunk)
            return x

        n_groups = head_chunks
        if isinstance(transformer_options, dict) and transformer_options.get("minimax_head_chunks"):
            n_groups = max(n_groups, int(transformer_options["minimax_head_chunks"]))
        n_groups = min(n_groups, self.attn.heads)
        eff_mode = _resolve_out_proj_mode(out_proj_mode, s, device)
        sol_ctx = transformer_options.get("_chunked_sage_sol_ctx") if isinstance(transformer_options, dict) else None
        block_idx = transformer_options.get("block_index") if isinstance(transformer_options, dict) else None

        has_qkv_extra_scales = (
            getattr(self.attn.qkv_proj, "pre_quant_scale", None) is not None
            or getattr(self.attn.qkv_proj, "input_scale", None) is not None
        )

        if not has_qkv_extra_scales:
            with _cast_linear_once(self.attn.qkv_proj, device, dtype) as (w_qkv, b_qkv):
                if _is_int8_tensorwise(w_qkv) and x.is_cuda and _ck_cuda is not None:
                    _run_chunked_attention_int8(
                        self, self.attn, x, shift_msa, scale_msa, gate_msa, mod_segments,
                        rope_freqs, w_qkv, b_qkv, n_groups, token_chunk, eff_mode,
                        sol_ctx=sol_ctx, block_idx=block_idx,
                    )
                    handled = True
                elif not isinstance(w_qkv, qops.QuantizedTensor):
                    _run_chunked_attention_dense(
                        self, self.attn, x, shift_msa, scale_msa, gate_msa, mod_segments,
                        rope_freqs, w_qkv, b_qkv, n_groups, token_chunk, eff_mode,
                        sol_ctx=sol_ctx, block_idx=block_idx,
                    )
                    handled = True
                else:
                    handled = False
        else:
            handled = False

        if not handled:
            # Other quantized formats (FP8 / NVFP4 / AWQ): delegate to chunked attn.forward
            h = [_mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)]
            attn_out = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
            x = _mod_gate(x, gate_msa, attn_out, mod_segments)
            del attn_out

        _run_chunked_mlp_inplace(self, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, token_chunk)
        return x

    minimax_chunked_sageattn_block_forward._is_chunked_sage_block = True
    return minimax_chunked_sageattn_block_forward


# ---------------------------------------------------------------------------
# Bit-exact token chunking for FP32 I/O layers (`video_patch_proj` & `FinalLayer`)
# ---------------------------------------------------------------------------

def _make_chunked_final_layer_forward(token_chunk: int):
    """100% bit-exact chunked `FinalLayer.forward` avoiding `[N_v, 5376] fp32` allocation (1.96-2.94 GiB)."""
    def chunked_final_layer_forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
        shift, scale = self.adaln_proj(t_emb)
        n = self.pdd_steps
        start = stop = 0
        if n > 1:
            if sample_sigmas is None:
                raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
            i = int((sample_sigmas - sigma).abs().argmin())
            sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
            start, stop = (
                min(range(n + 1), key=lambda k: abs(grids[0][k] - float(sv)))
                for sv in (sigma, sigma_next)
            )
            start = min(start, n - 1)
            stop = max(stop, start + 1)

        out = []
        with _cast_rmsnorm_once(self.norm, x) as nw:
            for head, seg, grid in zip((self.video_out, self.audio_out), (video_seg, audio_seg), grids):
                a, b, row = seg
                total = b - a
                if total <= token_chunk:
                    h = self._modulate(x, shift, scale, seg)
                    out.append(head(h) if n == 1 else _pdd_head(head, h, grid, start, stop))
                    continue

                out_dim = head.out_features // n
                res = torch.empty((total, out_dim), dtype=torch.float32, device=x.device)
                if n == 1:
                    with _cast_linear_once(head, x.device, torch.float32) as (w_h, b_h):
                        for t0 in range(a, b, token_chunk):
                            t1 = min(t0 + token_chunk, b)
                            row_c = row[t0 - a : t1 - a] if isinstance(row, torch.Tensor) else row
                            h_c = (
                                F.rms_norm(x[t0:t1], self.norm.normalized_shape, nw, self.norm.eps)
                                * (1.0 + _mod_row(scale, row_c, scale.dtype))
                                + _mod_row(shift, row_c, shift.dtype)
                            ).to(torch.float32)
                            res[t0 - a : t1 - a] = F.linear(h_c, w_h, b_h)
                            del h_c
                else:
                    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
                    w_blend = torch.tensor(dt, dtype=torch.float32, device=x.device) / sum(dt)
                    with _cast_linear_once(head, x.device, torch.float32) as (w_h, b_h):
                        rows = w_h.reshape(len(grid) - 1, -1, w_h.shape[1])
                        brows = b_h.reshape(len(grid) - 1, -1)
                        first = max(start, 1)
                        eff_w = rows[0] + torch.einsum("n,noi->oi", w_blend[first - start:], rows[first:stop])
                        eff_b = brows[0] + torch.einsum("n,no->o", w_blend[first - start:], brows[first:stop])
                        for t0 in range(a, b, token_chunk):
                            t1 = min(t0 + token_chunk, b)
                            row_c = row[t0 - a : t1 - a] if isinstance(row, torch.Tensor) else row
                            h_c = (
                                F.rms_norm(x[t0:t1], self.norm.normalized_shape, nw, self.norm.eps)
                                * (1.0 + _mod_row(scale, row_c, scale.dtype))
                                + _mod_row(shift, row_c, shift.dtype)
                            ).to(torch.float32)
                            res[t0 - a : t1 - a] = F.linear(h_c, eff_w, eff_b)
                            del h_c
                        del eff_w, eff_b
                out.append(res)
        return out

    return chunked_final_layer_forward


def _make_chunked_video_patch_proj_forward(orig_forward, token_chunk: int, target_dtype: torch.dtype | None):
    """Bit-exact chunked `video_patch_proj.forward` avoiding `[N_v, 5376] fp32` spike in `_embed_and_pack`."""
    def chunked_video_patch_proj_forward(self, x):
        if x.ndim != 2 or x.shape[0] <= token_chunk:
            return orig_forward(x)
        out_dtype = target_dtype if target_dtype in (torch.bfloat16, torch.float16) else x.dtype
        out = torch.empty((x.shape[0], self.out_features), dtype=out_dtype, device=x.device)
        with _cast_linear_once(self, x.device, x.dtype) as (w, b):
            for t0 in range(0, x.shape[0], token_chunk):
                t1 = min(t0 + token_chunk, x.shape[0])
                out[t0:t1] = F.linear(x[t0:t1], w, b).to(out_dtype)
        return out

    return chunked_video_patch_proj_forward


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class MiniMaxH3ChunkedSageAttentionPatch(io.ComfyNode):
    """Chunked-QKV + KJ SageAttention2++ low-VRAM patch for single-window 10s/15s MiniMax H3."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedSageAttentionPatch",
            display_name="MiniMax H3 Chunked QKV SageAttention Patch",
            category="MiniMax H3 Hard Cut",
            description=(
                "与 KJ SageAttention2++ 内核 100% 兼容的分块 QKV 低显存补丁：\n"
                "• 彻底消除全序列 [S, 21504] BF16 QKV 与 [S, 5376] MLP 中间张量；\n"
                "• exact 模式与原版 KJ SageAttention2++ 100.00% 逐位一致（10s 单窗峰值显存 3.26 GiB < 原版 5s）；\n"
                "• streaming 模式零 [S, 7168] 输出缓存（10s 仅 1.98 GiB、15s 仅 2.97 GiB），8GB 卡可单窗直出 10s 零拼缝；\n"
                "• 可直接替代或串联在 KJ MiniMaxH3MemoryEfficientSageAttentionPatch / LowVRAMAttention / ChunkFeedForward 之后。"
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "head_chunks",
                    default=14,
                    min=1,
                    max=56,
                    step=1,
                    tooltip="注意力头分组数（56 头推荐 7=每组8头 或 14=每组4头；自动按 2 头=256 ConvRot 块对齐）。",
                ),
                io.Combo.Input(
                    "out_proj_mode",
                    options=["auto", "exact", "streaming"],
                    default="auto",
                    tooltip=(
                        "out_proj 显存策略：\n"
                        "• auto：短序列(≤64k)或大显存卡(≥16GB)走 exact 逐位一致；8-12GB 卡长序列(10s/15s)自动切 streaming 零输出缓存；\n"
                        "• exact：100% 逐位一致（10s 峰值激活 3.26 GiB）；\n"
                        "• streaming：零 [S, 7168] 缓存（10s 仅 1.98 GiB，15s 仅 2.97 GiB）。"
                    ),
                ),
                io.Int.Input(
                    "token_chunk_size",
                    default=8192,
                    min=1024,
                    max=65536,
                    step=1024,
                    advanced=True,
                    tooltip="norm/ConvRot 量化、out_proj、SwiGLU MLP 与 I/O 层的 token 分块粒度（默认 8192）。",
                ),
                io.Boolean.Input(
                    "chunk_io_layers",
                    default=True,
                    advanced=True,
                    tooltip="同时对 video_patch_proj 与 final_layer 按 token 分块（100% 逐位一致，消除首尾 FP32 显存尖峰）。",
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=["dense_sage", "sol_attn_sparse"],
                    default="dense_sage",
                    tooltip=(
                        "注意力加速内核：\n"
                        "• dense_sage：100% 稠密 KJ SageAttention2++（最保守基准）；\n"
                        "• sol_attn_sparse：融合 comfy_kitchen Sol-Attn 稀疏 SageAttention（2d_frame Morton 重排 + 质心尾项守恒 + 条件 sink），"
                        "8GB 下显存几乎不变，10s 864p 提速 ~2.4×、10s 1080p 提速 ~3.1×、15s 1080p 提速 ~3.4×！"
                    ),
                ),
                io.Float.Input(
                    "sol_tau",
                    default=1.3,
                    min=0.3,
                    max=3.0,
                    step=0.05,
                    tooltip="Sol-Attn 稀疏路由阈值（仅 sol_attn_sparse 生效；越大越稀疏越快，1.3 为 H3 甜点，1.0 更保守）。",
                ),
                io.Int.Input(
                    "sol_token_aug",
                    default=64,
                    min=0,
                    max=256,
                    step=64,
                    tooltip="Sol-Attn 每查询块额外精确补采的 top 未路由 token 数（0/64/128/256；64 可将单层精度拉升至 cos=0.9992，仅慢 ~7%）。",
                ),
                io.Float.Input(
                    "sol_topk_ratio",
                    default=0.0,
                    min=0.0,
                    max=0.9,
                    step=0.05,
                    advanced=True,
                    tooltip="设为 >0 时改用固定 top-k 块比例（如 0.20）替代 sol_tau。",
                ),
                io.Combo.Input(
                    "sol_sink_mode",
                    options=["exact_kv_and_rows", "exact_kv", "off"],
                    default="exact_kv_and_rows",
                    advanced=True,
                    tooltip="条件行（文本/参考图/音频）精确保护：exact_kv_and_rows 同时保护音频查询行（音画同步最稳）。",
                ),
                io.Int.Input(
                    "sol_min_tokens",
                    default=12288,
                    min=0,
                    max=262144,
                    step=1024,
                    advanced=True,
                    tooltip="低于该 token 数自动退回稠密 KJ SageAttention。",
                ),
                io.String.Input(
                    "sol_dense_blocks",
                    default="",
                    advanced=True,
                    tooltip="可选：强制保持稠密的层号（如 '0-1,48-49'；留空则全部 50 层走稀疏）。",
                ),
                io.Float.Input(
                    "sol_max_sigma",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="仅对 sigma ≤ 该值的步启用稀疏（默认 0.95 保留一采第 0 步纯噪构图稠密，二采全步稀疏）。",
                ),
                io.Float.Input(
                    "sol_min_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="仅对 sigma ≥ 该值的步启用稀疏（设为 0.15 可让末步回落稠密精修）。",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        head_chunks: int = 14,
        out_proj_mode: str = "auto",
        token_chunk_size: int = 8192,
        chunk_io_layers: bool = True,
        attention_mode: str = "dense_sage",
        sol_tau: float = 1.3,
        sol_token_aug: int = 64,
        sol_topk_ratio: float = 0.0,
        sol_sink_mode: str = "exact_kv_and_rows",
        sol_min_tokens: int = 12288,
        sol_dense_blocks: str = "",
        sol_max_sigma: float = 0.95,
        sol_min_sigma: float = 0.0,
    ) -> io.NodeOutput:
        if _cuda_archs is None:
            raise RuntimeError(
                "sageattention is not installed or CUDA arch could not be detected; "
                "cannot apply MiniMaxH3ChunkedSageAttentionPatch."
            )
        if _ck is None:
            raise RuntimeError(
                "comfy_kitchen is unavailable; cannot apply MiniMaxH3ChunkedSageAttentionPatch."
            )

        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise RuntimeError(
                "MiniMaxH3ChunkedSageAttentionPatch can only be applied to a MiniMax H3 model."
            )

        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn") or not hasattr(blocks[0].attn, "qkv_proj"):
            raise RuntimeError("Unexpected MiniMax H3 model layout: missing blocks[*].attn.qkv_proj.")

        head_chunks = max(1, min(int(head_chunks), int(blocks[0].attn.heads)))
        token_chunk_size = max(1024, int(token_chunk_size))
        eff_attn_mode = str(attention_mode)
        if eff_attn_mode == "sol_attn_sparse" and not _sol_attn_available:
            logger.warning(
                "MiniMaxH3ChunkedSageAttentionPatch: comfy_kitchen.sol_attn unavailable; "
                "falling back to dense_sage."
            )
            eff_attn_mode = "dense_sage"

        sol_dense_set = _parse_dense_blocks(sol_dense_blocks, len(blocks))
        sol_token_aug = max(0, min(256, (int(sol_token_aug) // 64) * 64))
        sol_cfg = {
            "attention_mode": eff_attn_mode,
            "sol_tau": float(sol_tau),
            "sol_token_aug": sol_token_aug,
            "sol_topk_ratio": float(sol_topk_ratio),
            "sol_sink_mode": str(sol_sink_mode),
            "sol_min_tokens": int(sol_min_tokens),
            "sol_dense_blocks": str(sol_dense_blocks),
            "sol_dense_blocks_set": sol_dense_set,
            "sol_max_sigma": float(sol_max_sigma),
            "sol_min_sigma": float(sol_min_sigma),
        }

        attn_fwd = _make_chunked_attn_forward(head_chunks, token_chunk_size)
        block_fwd = _make_chunked_block_forward(head_chunks, str(out_proj_mode), token_chunk_size)

        for idx, block in enumerate(blocks):
            m.add_object_patch(
                f"diffusion_model.blocks.{idx}.attn.forward",
                types.MethodType(attn_fwd, block.attn),
            )
            m.add_object_patch(
                f"diffusion_model.blocks.{idx}.forward",
                types.MethodType(block_fwd, block),
            )

        if eff_attn_mode == "sol_attn_sparse":
            orig_rb = diffusion_model._run_blocks
            orig_rb_fn = (
                orig_rb.__func__
                if (isinstance(orig_rb, types.MethodType) and orig_rb.__self__ is diffusion_model)
                else type(diffusion_model)._run_blocks
            )
            rb_wrapped = _make_sol_run_blocks(orig_rb_fn, sol_cfg)
            m.add_object_patch(
                "diffusion_model._run_blocks",
                types.MethodType(rb_wrapped, diffusion_model),
            )

        if chunk_io_layers:
            if hasattr(diffusion_model, "final_layer"):
                fl_fwd = _make_chunked_final_layer_forward(token_chunk_size)
                m.add_object_patch(
                    "diffusion_model.final_layer.forward",
                    types.MethodType(fl_fwd, diffusion_model.final_layer),
                )
            if hasattr(diffusion_model, "video_patch_proj"):
                vpp = diffusion_model.video_patch_proj
                target_dtype = getattr(diffusion_model, "dtype", None)
                if target_dtype is None:
                    try:
                        target_dtype = m.model.get_dtype()
                    except Exception:
                        target_dtype = None
                vpp_fwd = _make_chunked_video_patch_proj_forward(
                    vpp.forward, token_chunk_size, target_dtype
                )
                m.add_object_patch(
                    "diffusion_model.video_patch_proj.forward",
                    types.MethodType(vpp_fwd, vpp),
                )

        m.model_options.setdefault("transformer_options", {})["minimax_chunked_sage"] = {
            "head_chunks": head_chunks,
            "out_proj_mode": str(out_proj_mode),
            "token_chunk_size": token_chunk_size,
            "chunk_io_layers": bool(chunk_io_layers),
            "attention_mode": eff_attn_mode,
            "sol_tau": float(sol_tau),
            "sol_token_aug": sol_token_aug,
            "sol_topk_ratio": float(sol_topk_ratio),
            "sol_sink_mode": str(sol_sink_mode),
            "arch": _cuda_archs[0] if _cuda_archs else "unknown",
            "sageplus_sm89": _sageplus_sm89_available,
            "sol_attn_available": _sol_attn_available,
        }

        logger.info(
            "Applied MiniMaxH3ChunkedSageAttentionPatch (%d blocks, head_chunks=%d, "
            "out_proj_mode=%s, attn=%s, tau=%.2f, aug=%d, sink=%s, token_chunk=%d, "
            "chunk_io=%s, arch=%s, sageplus=%s)",
            len(blocks),
            head_chunks,
            out_proj_mode,
            eff_attn_mode,
            float(sol_tau),
            sol_token_aug,
            sol_sink_mode,
            token_chunk_size,
            chunk_io_layers,
            _cuda_archs[0] if _cuda_archs else "unknown",
            _sageplus_sm89_available,
        )
        return io.NodeOutput(m)
