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

from collections import OrderedDict
import contextlib
import importlib
import logging
import re
import types
import uuid

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

try:
    from .lora_compat_bridge import qr_core_collapse_loras
    from .rot_lora_ops import (
        get_rot_lora_active_backend,
        pre_rotate_lora_down,
        rot_int8_lora_down,
    )
except ImportError:
    from lora_compat_bridge import qr_core_collapse_loras
    from rot_lora_ops import (
        get_rot_lora_active_backend,
        pre_rotate_lora_down,
        rot_int8_lora_down,
    )

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
_MORTON_SHIFT_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (4, 4), (2, 6), (6, 2))
_MORTON_CPU_CACHE: dict[tuple[tuple[int, int, int], str, int, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]] = {}
_MORTON_DEV_CACHE: dict[tuple[tuple[int, int, int], str, int, tuple[int, int], str], tuple[torch.Tensor, torch.Tensor]] = {}
_MORTON_SHIFT_DEV_CACHE: dict[
    tuple[tuple[int, int, int], str, int, str],
    tuple[tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]],
] = {}


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
    offset: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(perm_dev, inv_dev, perm_cpu)` aligned to Sol-Attn's 64-token block grid."""
    pad = (-int(start)) % _SOL_BLOCK_SIZE
    off = (int(offset[0]), int(offset[1]))
    cpu_key = (grid, curve, pad, off)
    cpu_hit = _MORTON_CPU_CACHE.get(cpu_key)
    if cpu_hit is None:
        frames, height, width = grid
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        rem = linear - z * area
        y = rem // width
        x = rem - y * width
        ox, oy = off
        if curve == "2d_frame":
            code = (z << 42) | _part1by2(x + ox) | (_part1by2(y + oy) << 1)
        else:
            code = _part1by2(x + ox) | (_part1by2(y + oy) << 1) | (_part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        if pad:
            perm = torch.roll(perm, pad)
        inv = torch.argsort(perm)
        cpu_hit = (perm, inv)
        _MORTON_CPU_CACHE[cpu_key] = cpu_hit

    dev_key = (grid, curve, pad, off, str(device))
    dev_hit = _MORTON_DEV_CACHE.get(dev_key)
    if dev_hit is None:
        # Keep device cache bounded to avoid lingering memory across shape changes
        if len(_MORTON_DEV_CACHE) >= 8:
            _MORTON_DEV_CACHE.clear()
        dev_hit = (cpu_hit[0].to(device), cpu_hit[1].to(device))
        _MORTON_DEV_CACHE[dev_key] = dev_hit
    return dev_hit[0], dev_hit[1], cpu_hit[0]


def _morton_shift_rel_perms(
    grid: tuple[int, int, int],
    start: int,
    device: torch.device,
    curve: str = "2d_frame",
) -> tuple[tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]]:
    """Precompute 4-phase Swin-style relative permutations `(rel_fwd, rel_inv)` on `device`.

    Rotates Morton `(ox, oy)` across `(0,0), (4,4), (2,6), (6,2)` by `block_idx & 3`, ensuring
    every 256/512/768/1024 px tile seam sits inside the interior of an 8x8 Morton block in 3 of 4
    consecutive DiT layers (eliminates 50-layer stacked horizontal/vertical seams at y=512/1024).
    """
    pad = (-int(start)) % _SOL_BLOCK_SIZE
    key = (grid, curve, pad, str(device))
    hit = _MORTON_SHIFT_DEV_CACHE.get(key)
    if hit is not None:
        return hit

    # Base phase-0 perm/inv on CPU
    _morton_perm_aligned(grid, start, device, curve=curve, offset=(0, 0))
    perm0, inv0 = _MORTON_CPU_CACHE[(grid, curve, pad, (0, 0))]

    fwd_list: list[torch.Tensor | None] = [None]
    inv_list: list[torch.Tensor | None] = [None]
    for off in _MORTON_SHIFT_OFFSETS[1:]:
        cpu_key = (grid, curve, pad, off)
        pair = _MORTON_CPU_CACHE.get(cpu_key)
        if pair is None:
            frames, height, width = grid
            linear = torch.arange(frames * height * width, dtype=torch.int64)
            area = height * width
            z = linear // area
            rem = linear - z * area
            y = rem // width
            x = rem - y * width
            ox, oy = off
            if curve == "2d_frame":
                code = (z << 42) | _part1by2(x + ox) | (_part1by2(y + oy) << 1)
            else:
                code = _part1by2(x + ox) | (_part1by2(y + oy) << 1) | (_part1by2(z) << 2)
            perm_p = linear[torch.argsort(code)]
            if pad:
                perm_p = torch.roll(perm_p, pad)
            inv_p = torch.argsort(perm_p)
            pair = (perm_p, inv_p)
            _MORTON_CPU_CACHE[cpu_key] = pair
        perm_p, inv_p = pair
        # X_p[j] = X_0[inv0[perm_p[j]]], Y_0[i] = Y_p[inv_p[perm0[i]]]
        fwd_list.append(inv0[perm_p].to(device))
        inv_list.append(inv_p[perm0].to(device))

    if len(_MORTON_SHIFT_DEV_CACHE) >= 4:
        _MORTON_SHIFT_DEV_CACHE.clear()
    hit = (tuple(fwd_list), tuple(inv_list))
    _MORTON_SHIFT_DEV_CACHE[key] = hit
    return hit


def _permute_nhd_span_inplace(
    t: torch.Tensor,
    start: int,
    stop: int,
    idx: torch.Tensor,
    head_chunk: int = 2,
) -> None:
    """In-place token permutation of `[1, S, g, D]` over `start:stop` in small head chunks (0 peak VRAM impact)."""
    view = t[0, start:stop]
    g = view.shape[1]
    step = max(1, int(head_chunk))
    for h0 in range(0, g, step):
        h1 = min(h0 + step, g)
        view[:, h0:h1, :] = view[:, h0:h1, :].index_select(0, idx)


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


def _parse_dense_blocks_spec(spec: str, count: int) -> tuple[frozenset[int], float]:
    """Parse `sol_dense_blocks` like '0-1,48-49' or '48-49@0.35' -> `(block_indices, max_sigma)`.

    When `@<sigma>` is appended (e.g. `'48-49@0.35'`), dense overrides only trigger on low-sigma
    refinement steps (`sigma <= max_sigma`), skipping redundant O(S^2) dense passes on early
    high-noise steps (saves ~55-65s on 15s 1080p with zero seam regression).
    """
    s = "".join(str(spec or "").split())
    if not s:
        return frozenset(), 1.0
    max_sigma = 1.0
    if "@" in s:
        s_blocks, s_sig = s.rsplit("@", 1)
        try:
            max_sigma = float(s_sig)
        except ValueError as exc:
            raise ValueError(f"Invalid sol_dense_blocks sigma suffix {s_sig!r} in {spec!r}") from exc
        s = s_blocks
    if not s:
        return frozenset(), max_sigma
    out: set[int] = set()
    for part in s.split(","):
        if not part:
            continue
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if m is None:
            raise ValueError(f"Invalid sol_dense_blocks entry {part!r}; expected e.g. '0-1,48-49' or '48-49@0.35'")
        lo = int(m.group(1))
        hi = lo if m.group(2) is None else int(m.group(2))
        lo = lo if lo >= 0 else count + lo
        hi = hi if hi >= 0 else count + hi
        if lo > hi:
            lo, hi = hi, lo
        out.update(range(max(lo, 0), min(hi, count - 1) + 1))
    return frozenset(out), max_sigma


def _parse_dense_blocks(spec: str, count: int) -> frozenset[int]:
    """Parse block index/range spec like '0-1,48-49' or '-1' into absolute block indices."""
    blocks_set, _ = _parse_dense_blocks_spec(spec, count)
    return blocks_set


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

    use_tight_low_sigma = False
    cur_sigma: float | None = None
    sigmas = transformer_options.get("sigmas") if isinstance(transformer_options, dict) else None
    if sigmas is not None and torch.is_tensor(sigmas) and sigmas.numel() > 0:
        cur_sigma = float(sigmas.flatten()[0].item())
        if cur_sigma > float(sol_cfg.get("sol_max_sigma", 0.95)) + 1e-6:
            return None, None
        if cur_sigma < float(sol_cfg.get("sol_min_sigma", 0.0)) - 1e-6:
            if float(sol_cfg.get("sol_low_sigma_tau", 0.0)) > 0.0:
                use_tight_low_sigma = True
            else:
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

    if use_tight_low_sigma:
        eff_tau = float(sol_cfg.get("sol_low_sigma_tau", 0.90))
        eff_aug = max(128, int(sol_cfg.get("sol_token_aug", 64)))
        eff_dense = sol_cfg.get("sol_low_sigma_dense_blocks_set", frozenset())
    else:
        eff_tau = float(sol_cfg.get("sol_tau", 1.3))
        eff_aug = int(sol_cfg.get("sol_token_aug", 64))
        dense_max_sig = float(sol_cfg.get("sol_dense_blocks_max_sigma", 1.0))
        if cur_sigma is not None and cur_sigma > dense_max_sig + 1e-6:
            eff_dense = frozenset()
        else:
            eff_dense = sol_cfg.get("sol_dense_blocks_set", frozenset())

    sol_ctx = {
        "tau": eff_tau,
        "topk_ratio": float(sol_cfg.get("sol_topk_ratio", 0.0)),
        "token_aug": eff_aug,
        "sink_blocks": sink_blocks,
        "sink_q": sink_q,
        "dense_blocks": eff_dense,
    }
    return sol_ctx, morton_info


def _build_split_band_rope_scale(rope_hf_scale: float, n_freq: int, device: torch.device) -> torch.Tensor:
    """Build smooth split-band scale `[n_freq]` for spatial RoPE: high bands scaled, low bands 1.0."""
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


def _attend_group(
    q_g: torch.Tensor,
    k_g: torch.Tensor,
    make_v_fn,
    dtype: torch.dtype,
    sol_ctx: dict | None = None,
    block_idx: int | None = None,
    attn_temp_scale: float = 1.0,
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
        shift_inv = None
        shift_va = shift_vb = 0
        if block_idx is not None:
            rel_fwd_tuple = sol_ctx.get("shift_rel_fwd")
            if rel_fwd_tuple is not None:
                phase = int(block_idx) & 3
                rel_fwd = rel_fwd_tuple[phase]
                if rel_fwd is not None:
                    shift_va = int(sol_ctx["shift_va"])
                    shift_vb = int(sol_ctx["shift_vb"])
                    shift_inv = sol_ctx["shift_rel_inv"][phase]
                    _permute_nhd_span_inplace(q_g, shift_va, shift_vb, rel_fwd)
                    _permute_nhd_span_inplace(k_g, shift_va, shift_vb, rel_fwd)
                    _permute_nhd_span_inplace(v_g, shift_va, shift_vb, rel_fwd)
        sol_scale = (q_g.shape[-1] ** -0.5) * float(attn_temp_scale) if attn_temp_scale != 1.0 else None
        o_g = _ck.sol_attn(
            q_g,
            k_g,
            v_g,
            tau=sol_ctx["tau"],
            scale=sol_scale,
            topk_ratio=sol_ctx["topk_ratio"],
            sink_blocks=list(sol_ctx["sink_blocks"]),
            sink_q=list(sol_ctx["sink_q"]),
            tail=True,
            token_aug=sol_ctx["token_aug"],
        )
        del q_g, k_g, v_g
        if shift_inv is not None:
            _permute_nhd_span_inplace(o_g, shift_va, shift_vb, shift_inv)
        return o_g
    return _staged_sageattn_nhd(q_g, k_g, make_v_fn, dtype, attn_temp_scale=attn_temp_scale)


def _extract_lora_entries_from_patches(
    patches_list,
    weight_shape: tuple[int, ...],
) -> list[tuple[torch.Tensor, torch.Tensor, float]] | None:
    """Extract standard 2D linear LoRA `(B [N,r], A [r,K], eff_scale)` tuples from ComfyUI `patches`.

    Returns `None` if ANY patch is non-standard (DoRA, LoHa/LoKr, Tucker `mid`, `reshape`,
    `offset`, custom `function`, `strength_model != 1.0`, or non-LoRA `diff`/`set`), so the
    caller cleanly falls back to ComfyUI's stock weight-patching path with zero accuracy risk.
    """
    if not patches_list or len(weight_shape) != 2:
        return None
    out_dim, in_dim = int(weight_shape[0]), int(weight_shape[1])
    extracted: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for p in patches_list:
        if not isinstance(p, (tuple, list)) or len(p) < 5:
            return None
        strength, adapter, strength_model, offset, fn = p[0], p[1], p[2], p[3], p[4]
        if float(strength_model) != 1.0 or offset is not None or fn is not None:
            return None
        if getattr(adapter, "name", None) != "lora" or not hasattr(adapter, "weights"):
            return None
        weights = adapter.weights
        if not isinstance(weights, (tuple, list)) or len(weights) < 6:
            return None
        mat1, mat2, alpha, mid, dora_scale, reshape = weights[:6]
        if mid is not None or dora_scale is not None or reshape is not None:
            return None
        if not (torch.is_tensor(mat1) and torch.is_tensor(mat2)):
            return None
        if mat1.ndim != 2 or mat2.ndim != 2:
            return None
        if mat1.shape[0] != out_dim or mat2.shape[1] != in_dim or mat1.shape[1] != mat2.shape[0]:
            return None
        rank = int(mat2.shape[0])
        if rank <= 0:
            return None
        alpha_scale = (float(alpha) / float(rank)) if alpha is not None else 1.0
        eff_scale = float(strength) * alpha_scale
        if eff_scale != 0.0:
            extracted.append((mat1, mat2, eff_scale))
    return extracted


def _extract_bypass_lora_entries(
    linear,
    weight_shape: tuple[int, ...],
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Extract LoRA adapters attached via `LoraLoaderBypassModelOnly` (`BypassForwardHook`)."""
    if len(weight_shape) != 2:
        return []
    out_dim, in_dim = int(weight_shape[0]), int(weight_shape[1])
    out: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    fwd = getattr(linear, "forward", None)
    seen: set[int] = set()
    while fwd is not None:
        hook = getattr(fwd, "__self__", None)
        if hook is None or id(hook) in seen or not hasattr(hook, "adapter") or not hasattr(hook, "original_forward"):
            break
        seen.add(id(hook))
        adapter = hook.adapter
        mult = float(getattr(adapter, "multiplier", 1.0))
        if getattr(adapter, "name", None) == "lora" and hasattr(adapter, "weights"):
            weights = adapter.weights
            if isinstance(weights, (tuple, list)) and len(weights) >= 6:
                mat1, mat2, alpha, mid, dora_scale, reshape = weights[:6]
                if (
                    mid is None
                    and dora_scale is None
                    and reshape is None
                    and torch.is_tensor(mat1)
                    and torch.is_tensor(mat2)
                    and mat1.ndim == 2
                    and mat2.ndim == 2
                    and mat1.shape[0] == out_dim
                    and mat2.shape[1] == in_dim
                    and mat1.shape[1] == mat2.shape[0]
                    and mat2.shape[0] > 0
                ):
                    alpha_scale = (float(alpha) / float(mat2.shape[0])) if alpha is not None else 1.0
                    eff_scale = mult * alpha_scale
                    if eff_scale != 0.0:
                        out.append((mat1, mat2, eff_scale))
        elif hasattr(adapter, "lora_up") and hasattr(adapter, "lora_down") and getattr(adapter, "lora_mid", None) is None:
            mat1 = getattr(adapter.lora_up, "weight", None)
            mat2 = getattr(adapter.lora_down, "weight", None)
            if (
                torch.is_tensor(mat1)
                and torch.is_tensor(mat2)
                and mat1.ndim == 2
                and mat2.ndim == 2
                and mat1.shape[0] == out_dim
                and mat2.shape[1] == in_dim
                and mat1.shape[1] == mat2.shape[0]
                and mat2.shape[0] > 0
            ):
                alpha_val = float(adapter.alpha.item()) if torch.is_tensor(adapter.alpha) else float(adapter.alpha)
                eff_scale = mult * alpha_val / float(mat2.shape[0])
                if eff_scale != 0.0:
                    out.append((mat1, mat2, eff_scale))
        fwd = getattr(hook, "original_forward", None)
    return out


def _prepare_lora_packed(
    entries: list[tuple[torch.Tensor, torch.Tensor, float]] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack one or more low-rank `(B [N,r], A [r,K], scale)` tuples into a single `(B_cat [N,R], A_cat [R,K])`.

    Moves tensors to GPU before dtype casting (avoids slow CPU fp16->bf16) and folds per-adapter
    `scale` into the small `A` matrix so stacked LoRAs execute in ONE cuBLAS `addmm` with zero
    intermediate `[M, N]` allocations.
    """
    if not entries:
        return None
    b_list: list[torch.Tensor] = []
    a_list: list[torch.Tensor] = []
    for mat1, mat2, scale in entries:
        if mat1.device != device:
            b_dev = mat1.to(device=device, non_blocking=True)
        else:
            b_dev = mat1
        if b_dev.dtype != dtype:
            b_dev = b_dev.to(dtype=dtype)
        if not b_dev.is_contiguous():
            b_dev = b_dev.contiguous()

        if mat2.device != device:
            a_dev = mat2.to(device=device, non_blocking=True)
        else:
            a_dev = mat2
        if a_dev.dtype != dtype:
            a_dev = a_dev.to(dtype=dtype)
        if scale != 1.0:
            a_dev = (a_dev.float() * float(scale)).to(dtype=dtype)
        if not a_dev.is_contiguous():
            a_dev = a_dev.contiguous()

        b_list.append(b_dev)
        a_list.append(a_dev)

    if len(b_list) == 1:
        return b_list[0], a_list[0]
    return torch.cat(b_list, dim=1).contiguous(), torch.cat(a_list, dim=0).contiguous()


def _add_lora_inplace(
    out_2d: torch.Tensor,
    x_2d: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """In-place `out_2d += (x_2d @ A_cat^T) @ B_cat^T` via cuBLAS `addmm` (zero `[M, N]` temp)."""
    b_cat, a_cat = packed
    mid = F.linear(x_2d, a_cat)
    torch.addmm(out_2d, mid, b_cat.t(), beta=1.0, alpha=1.0, out=out_2d)
    del mid


def _add_gated_lora_mid_residual_inplace(
    x_resid: torch.Tensor,
    mid: torch.Tensor,
    b_cat: torch.Tensor,
    gate: torch.Tensor,
    gate_cast: torch.Tensor,
    sub_segs: list,
    gated_b_cache: dict[int, torch.Tensor],
) -> None:
    """In-place `x_resid += gate * (mid @ B_cat^T)` via gated `addmm` (zero `[M, N]` temp)."""
    dtype = x_resid.dtype
    for a, b, row in sub_segs:
        if b <= a:
            continue
        r_slice = x_resid[a:b]
        if isinstance(row, int) and r_slice.is_contiguous():
            b_gt = gated_b_cache.get(row)
            if b_gt is None:
                b_gt = (b_cat * gate_cast[row].unsqueeze(1)).contiguous().t()
                gated_b_cache[row] = b_gt
            torch.addmm(r_slice, mid[a:b], b_gt, beta=1.0, alpha=1.0, out=r_slice)
        else:
            lora_seg = F.linear(mid[a:b], b_cat)
            r_slice.addcmul_(lora_seg, _mod_row(gate, row, dtype))
            del lora_seg


def _eval_lora_delta(
    x_c: torch.Tensor,
    loras_dev: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> torch.Tensor:
    """Compute exact BF16 low-rank side-branch `sum_i scale_i * ((x_c @ A_i^T) @ B_i^T)`."""
    b0, a0, s0 = loras_dev[0]
    out = F.linear(F.linear(x_c, a0), b0)
    if s0 != 1.0:
        out.mul_(s0)
    for b_i, a_i, s_i in loras_dev[1:]:
        out.add_(F.linear(F.linear(x_c, a_i), b_i), alpha=s_i)
    return out


if _HAS_TRITON:
    @triton.jit
    def _swiglu_fused_kernel(
        InPtr, OutPtr,
        M, FFN,
        stride_im, stride_om,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        offs_k = col_blk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < FFN
        base_in = InPtr + row.to(tl.int64) * stride_im
        g = tl.load(base_in + offs_k, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(base_in + FFN + offs_k, mask=mask, other=0.0).to(tl.float32)
        out = (g * tl.sigmoid(g)) * u
        base_out = OutPtr + row.to(tl.int64) * stride_om
        tl.store(base_out + offs_k, out.to(OutPtr.dtype.element_ty), mask=mask)


def _swiglu_chunk(fc1_c: torch.Tensor) -> torch.Tensor:
    """Compute `silu(gate) * up` from `[M, 2*FFN]` in a single coalesced pass."""
    m, two_ffn = fc1_c.shape
    ffn = two_ffn // 2
    if _HAS_TRITON and fc1_c.is_cuda and fc1_c.is_contiguous():
        out = torch.empty((m, ffn), dtype=fc1_c.dtype, device=fc1_c.device)
        BLOCK_K = 1024
        grid = (m, (ffn + BLOCK_K - 1) // BLOCK_K)
        _swiglu_fused_kernel[grid](
            fc1_c, out, m, ffn, fc1_c.stride(0), out.stride(0), BLOCK_K=BLOCK_K, num_warps=4,
        )
        return out
    gate_part, up_part = fc1_c.chunk(2, dim=-1)
    return F.silu(gate_part).mul_(up_part)


def _get_linear_loras(
    linear,
    transformer_options: dict | None,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Resolve active BF16 low-rank LoRA entries for a linear module."""
    active = transformer_options.get("_chunked_sage_active_loras") if isinstance(transformer_options, dict) else None
    if isinstance(active, dict):
        hit = active.get(id(linear))
        if hit is not None:
            return hit
    raw_w = getattr(linear, "weight", None)
    if raw_w is not None and hasattr(raw_w, "shape"):
        return _extract_bypass_lora_entries(linear, tuple(raw_w.shape))
    return []


def _get_linear_rot_lora(
    linear,
    transformer_options: dict | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Resolve pre-rotated `(B [d_out, R_*], A_rot [R_*, d_in])` on `device`/`dtype` if active."""
    rot_map = transformer_options.get("_chunked_sage_rot_loras") if isinstance(transformer_options, dict) else None
    if not isinstance(rot_map, dict):
        return None
    pair = rot_map.get(id(linear))
    if pair is None:
        return None
    b_t, a_rot = pair
    if b_t.device != device or b_t.dtype != dtype:
        b_t = b_t.to(device=device, dtype=dtype, non_blocking=True)
    if not b_t.is_contiguous():
        b_t = b_t.contiguous()
    if a_rot.device != device or a_rot.dtype != dtype:
        a_rot = a_rot.to(device=device, dtype=dtype, non_blocking=True)
    if not a_rot.is_contiguous():
        a_rot = a_rot.contiguous()
    return b_t, a_rot


_LORA_OPT_CACHE: OrderedDict[
    tuple,
    tuple[list[tuple[torch.Tensor, torch.Tensor, float]], tuple[torch.Tensor, torch.Tensor] | None],
] = OrderedDict()
_LORA_OPT_CACHE_MAX = 1024


def _tensor_quick_sig(t: torch.Tensor) -> tuple:
    """Fast collision-resistant content signature for CPU/GPU LoRA matrices (avoids data_ptr ABA)."""
    flat = t.detach().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return (tuple(t.shape), str(t.dtype), 0)
    idx = (0, n // 5, (2 * n) // 5, (3 * n) // 5, (4 * n) // 5, n - 1)
    vals = tuple(round(float(flat[i].item()), 6) for i in idx)
    return (tuple(t.shape), str(t.dtype), vals)


def _optimize_lora_entries_for_mode(
    entries: list[tuple[torch.Tensor, torch.Tensor, float]],
    lora_fast_mode: str,
    allow_rot256: bool,
    max_rank: int = 64,
    energy_ratio: float = 0.992,
    compute_device: torch.device | str | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, float]], tuple[torch.Tensor, torch.Tensor] | None]:
    """Collapse multi-LoRA entries via QR-Core (if enabled) and pre-rotate `A_rot = A @ H_256`."""
    if not entries or lora_fast_mode == "legacy":
        return entries, None

    orig_dev = entries[0][0].device
    use_cpu_cache = orig_dev.type == "cpu"
    cache_key = None
    if use_cpu_cache:
        cache_key = (
            str(lora_fast_mode),
            bool(allow_rot256),
            int(max_rank),
            round(float(energy_ratio), 6),
            tuple((_tensor_quick_sig(b), _tensor_quick_sig(a), round(float(sc), 8)) for b, a, sc in entries),
        )
        cached = _LORA_OPT_CACHE.get(cache_key)
        if cached is not None:
            _LORA_OPT_CACHE.move_to_end(cache_key)
            return cached

    r_sum = sum(int(a.shape[0]) for _, a, _ in entries)
    if len(entries) > 1 or r_sum > max_rank:
        collapsed = qr_core_collapse_loras(
            entries,
            max_rank=max_rank,
            energy_ratio=energy_ratio,
            out_dtype=torch.bfloat16,
            compute_device=compute_device,
        )
        if collapsed is not None:
            b_tilde, a_tilde, _ = collapsed
            if b_tilde.device != orig_dev:
                b_tilde = b_tilde.to(device=orig_dev)
            if a_tilde.device != orig_dev:
                a_tilde = a_tilde.to(device=orig_dev)
            opt_entries = [(b_tilde, a_tilde, 1.0)]
        else:
            return [], None
    else:
        b0, a0, s0 = entries[0]
        b_tilde = b0.detach().to(device=orig_dev, dtype=torch.bfloat16).contiguous()
        a_tilde = (
            (a0.detach().to(device=orig_dev).float() * float(s0)).to(dtype=torch.bfloat16).contiguous()
            if float(s0) != 1.0
            else a0.detach().to(device=orig_dev, dtype=torch.bfloat16).contiguous()
        )
        opt_entries = [(b_tilde, a_tilde, 1.0)]

    rot_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    if allow_rot256 and lora_fast_mode in ("auto", "rotated_int8") and int(a_tilde.shape[1]) % 256 == 0:
        a_rot = pre_rotate_lora_down(a_tilde, group_size=256, mode="convrot")
        if a_rot.device != orig_dev:
            a_rot = a_rot.to(device=orig_dev)
        rot_pair = (b_tilde, a_rot)

    result = (opt_entries, rot_pair)
    if use_cpu_cache and cache_key is not None:
        _LORA_OPT_CACHE[cache_key] = result
        if len(_LORA_OPT_CACHE) > _LORA_OPT_CACHE_MAX:
            _LORA_OPT_CACHE.popitem(last=False)
    return result


def _make_chunked_run_blocks(
    orig_run_blocks_fn,
    sol_cfg: dict | None,
    static_loras: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] | None = None,
    rope_hf_scale: float = 1.0,
    attn_temp_scale: float = 1.0,
    lora_fast_mode: str = "auto",
    static_rot_loras: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
):
    """Wrap `MiniMaxH3Model._run_blocks` to manage BF16/Rotated-INT8 LoRA side-branches, Split-Band RoPE, & Sol-Attn Morton reordering."""
    static_loras = static_loras or {}
    static_rot_loras = static_rot_loras or {}
    lora_fast_mode = str(lora_fast_mode or "auto")
    rope_hf_scale = float(rope_hf_scale)
    attn_temp_scale = float(attn_temp_scale)
    dynamic_opt_cache: dict[
        tuple,
        tuple[list[tuple[torch.Tensor, torch.Tensor, float]], tuple[torch.Tensor, torch.Tensor] | None],
    ] = {}

    def chunked_sage_run_blocks(self, h, t_emb, mod_segments, layout, transformer_options):
        # 1. Intercept any remaining INT8 LoRA LowVramPatch / BypassForwardHook BEFORE
        #    `make_prefetch_queue` runs, so `pin_modules` never triggers 200-layer VBAR
        #    dequantize + B@A + Hadamard requantize on every step.
        active_loras: dict[int, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        active_rot_loras: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        stashed_lowvram: list[tuple[object, object]] = []
        stashed_wfn: list[tuple[object, list]] = []
        blocks = getattr(self, "blocks", None) or ()
        for idx, blk in enumerate(blocks):
            for suffix, lin in (
                ("attn.qkv_proj", blk.attn.qkv_proj),
                ("attn.out_proj", blk.attn.out_proj),
                ("mlp.fc1", blk.mlp.fc1),
                ("mlp.fc2", blk.mlp.fc2),
            ):
                key = f"diffusion_model.blocks.{idx}.{suffix}.weight"
                entries: list[tuple[torch.Tensor, torch.Tensor, float]] = []
                has_dynamic = False
                if key in static_loras:
                    entries.extend(static_loras[key])
                raw_w = getattr(lin, "weight", None)
                is_int8_q = isinstance(raw_w, qops.QuantizedTensor) and getattr(lin, "layout_type", None) is not None
                lowvram_fn = getattr(lin, "weight_lowvram_function", None)
                w_funcs = getattr(lin, "weight_function", ()) or ()
                if (
                    is_int8_q
                    and lowvram_fn is not None
                    and getattr(lowvram_fn, "is_lowvram_patch", False)
                    and len(w_funcs) == 0
                ):
                    lv_key = getattr(lowvram_fn, "key", None)
                    lv_patches = (getattr(lowvram_fn, "patches", None) or {}).get(lv_key)
                    lv_entries = _extract_lora_entries_from_patches(lv_patches, tuple(raw_w.shape))
                    if lv_entries is not None:
                        stashed_lowvram.append((lin, lowvram_fn))
                        lin.weight_lowvram_function = None
                        if lv_entries:
                            entries.extend(lv_entries)
                            has_dynamic = True
                elif (
                    is_int8_q
                    and lowvram_fn is None
                    and len(w_funcs) == 1
                    and getattr(w_funcs[0], "is_lowvram_patch", False)
                ):
                    lv_fn = w_funcs[0]
                    lv_key = getattr(lv_fn, "key", None)
                    lv_patches = (getattr(lv_fn, "patches", None) or {}).get(lv_key)
                    lv_entries = _extract_lora_entries_from_patches(lv_patches, tuple(raw_w.shape))
                    if lv_entries is not None:
                        stashed_wfn.append((lin, list(w_funcs)))
                        lin.weight_function = []
                        if lv_entries:
                            entries.extend(lv_entries)
                            has_dynamic = True
                if raw_w is not None and hasattr(raw_w, "shape"):
                    bypass_entries = _extract_bypass_lora_entries(lin, tuple(raw_w.shape))
                    if bypass_entries:
                        entries.extend(bypass_entries)
                        has_dynamic = True
                if entries:
                    if not has_dynamic and key in static_loras:
                        active_loras[id(lin)] = entries
                        if key in static_rot_loras:
                            active_rot_loras[id(lin)] = static_rot_loras[key]
                    elif lora_fast_mode != "legacy":
                        is_convrot_256 = (
                            suffix in ("attn.qkv_proj", "mlp.fc1")
                            and _is_int8_tensorwise(raw_w)
                            and bool(getattr(raw_w._params, "convrot", False))
                            and int(getattr(raw_w._params, "convrot_groupsize", 256)) == 256
                        )
                        sig = (
                            id(lin),
                            lora_fast_mode,
                            is_convrot_256,
                            tuple((id(b), id(a), round(float(sc), 7)) for b, a, sc in entries),
                        )
                        cached = dynamic_opt_cache.get(sig)
                        if cached is None:
                            if len(dynamic_opt_cache) >= 512:
                                dynamic_opt_cache.clear()
                            cached = _optimize_lora_entries_for_mode(
                                entries, lora_fast_mode, allow_rot256=is_convrot_256
                            )
                            dynamic_opt_cache[sig] = cached
                        opt_ents, rot_p = cached
                        if opt_ents:
                            active_loras[id(lin)] = opt_ents
                        if rot_p is not None:
                            active_rot_loras[id(lin)] = rot_p
                    else:
                        active_loras[id(lin)] = entries

        prev_loras = transformer_options.get("_chunked_sage_active_loras")
        prev_rot_loras = transformer_options.get("_chunked_sage_rot_loras")
        prev_fast_mode = transformer_options.get("_chunked_sage_lora_fast_mode")
        transformer_options["_chunked_sage_lora_fast_mode"] = lora_fast_mode
        if active_loras:
            transformer_options["_chunked_sage_active_loras"] = active_loras
        else:
            transformer_options.pop("_chunked_sage_active_loras", None)
        if active_rot_loras:
            transformer_options["_chunked_sage_rot_loras"] = active_rot_loras
        else:
            transformer_options.pop("_chunked_sage_rot_loras", None)

        prev_temp = transformer_options.get("_chunked_sage_attn_temp")
        if attn_temp_scale != 1.0:
            transformer_options["_chunked_sage_attn_temp"] = attn_temp_scale
        else:
            transformer_options.pop("_chunked_sage_attn_temp", None)

        # Optional Split-Band Spatial RoPE un-squishing for >1.35MP (e.g. native 1080p):
        # Scales ONLY high spatial frequencies (k=0..5, cosine taper k=6..7 -> 1.0) on video rows
        # va:vb, keeping low spatial bands (k=8..15), time axis, ref_img, and stereo audio 100% untouched.
        orig_rope_freqs_attr = None
        had_instance_rope = False
        if rope_hf_scale != 1.0 and hasattr(self, "rope_freqs"):
            segs = getattr(layout, "segments", None) or []
            vid_span_rope = next(((int(a), int(b)) for a, b, kind in segs if kind == "video"), None)
            if vid_span_rope is not None and vid_span_rope[1] > vid_span_rope[0]:
                r_va, r_vb = vid_span_rope
                had_instance_rope = "rope_freqs" in getattr(self, "__dict__", {})
                orig_rope_freqs_attr = self.__dict__.get("rope_freqs") if had_instance_rope else None
                base_rope_fn = self.rope_freqs

                def _split_band_rope_freqs(position_ids, dev):
                    angles = base_rope_fn(position_ids, dev)
                    if angles.ndim == 2 and angles.shape[0] >= r_vb and angles.shape[1] % 6 == 0:
                        nf = angles.shape[1] // 6  # 16 per axis for H3 (96 / 6)
                        half = nf * 3              # 48
                        g = _build_split_band_rope_scale(rope_hf_scale, nf, angles.device)
                        angles = angles.clone()
                        angles[r_va:r_vb, nf : 2 * nf].mul_(g)
                        angles[r_va:r_vb, 2 * nf : 3 * nf].mul_(g)
                        angles[r_va:r_vb, half + nf : half + 2 * nf].mul_(g)
                        angles[r_va:r_vb, half + 2 * nf : half + 3 * nf].mul_(g)
                    return angles

                self.rope_freqs = _split_band_rope_freqs

        s = h.shape[0]
        sol_ctx, morton_info = (
            _build_sol_context(layout, s, sol_cfg, transformer_options)
            if sol_cfg
            else (None, None)
        )
        prev_ctx = transformer_options.get("_chunked_sage_sol_ctx")
        active_morton = None
        run_layout = layout
        run_mod_segments = mod_segments

        if sol_ctx is not None:
            transformer_options["_chunked_sage_sol_ctx"] = sol_ctx
            if morton_info is not None:
                va, vb, grid = morton_info
                perm_dev, inv_dev, perm_cpu = _morton_perm_aligned(grid, va, h.device, curve="2d_frame")
                rel_fwd_dev, rel_inv_dev = _morton_shift_rel_perms(grid, va, h.device, curve="2d_frame")
                sol_ctx["shift_va"] = va
                sol_ctx["shift_vb"] = vb
                sol_ctx["shift_rel_fwd"] = rel_fwd_dev
                sol_ctx["shift_rel_inv"] = rel_inv_dev
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
        else:
            transformer_options.pop("_chunked_sage_sol_ctx", None)

        try:
            h = orig_run_blocks_fn(self, h, t_emb, run_mod_segments, run_layout, transformer_options)
        finally:
            if rope_hf_scale != 1.0 and hasattr(self, "rope_freqs"):
                if had_instance_rope:
                    self.rope_freqs = orig_rope_freqs_attr
                elif "rope_freqs" in getattr(self, "__dict__", {}):
                    del self.__dict__["rope_freqs"]
            for lin, lowvram_fn in stashed_lowvram:
                lin.weight_lowvram_function = lowvram_fn
            for lin, orig_wfn in stashed_wfn:
                lin.weight_function = orig_wfn
            if prev_temp is None:
                transformer_options.pop("_chunked_sage_attn_temp", None)
            else:
                transformer_options["_chunked_sage_attn_temp"] = prev_temp
            if prev_loras is None:
                transformer_options.pop("_chunked_sage_active_loras", None)
            else:
                transformer_options["_chunked_sage_active_loras"] = prev_loras
            if prev_rot_loras is None:
                transformer_options.pop("_chunked_sage_rot_loras", None)
            else:
                transformer_options["_chunked_sage_rot_loras"] = prev_rot_loras
            if prev_fast_mode is None:
                transformer_options.pop("_chunked_sage_lora_fast_mode", None)
            else:
                transformer_options["_chunked_sage_lora_fast_mode"] = prev_fast_mode
            if prev_ctx is None:
                transformer_options.pop("_chunked_sage_sol_ctx", None)
            else:
                transformer_options["_chunked_sage_sol_ctx"] = prev_ctx
            if active_morton is not None:
                va, vb, inv_dev = active_morton
                if torch.is_tensor(h) and h.ndim == 2 and h.shape[0] >= vb:
                    _permute_span_inplace(h, va, vb, inv_dev)
        return h

    chunked_sage_run_blocks._is_chunked_sage_run_blocks = True
    chunked_sage_run_blocks._orig_run_blocks_fn = orig_run_blocks_fn
    chunked_sage_run_blocks._chunked_sage_static_loras = static_loras
    chunked_sage_run_blocks._chunked_sage_static_rot_loras = static_rot_loras
    return chunked_sage_run_blocks


# Backwards-compatible alias for unit tests importing `_make_sol_run_blocks`
_make_sol_run_blocks = _make_chunked_run_blocks


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


def _staged_sageattn_nhd(
    q: torch.Tensor,
    k: torch.Tensor,
    get_v_fn,
    dtype: torch.dtype,
    attn_temp_scale: float = 1.0,
) -> torch.Tensor:
    """KJ-identical SageAttention2++ on NHD [1, S, g, D], deferring V projection until BF16 Q/K are freed."""
    head_dim = q.shape[-1]
    sm_scale = (head_dim ** -0.5) * float(attn_temp_scale)
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
    o_hnd = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, scale=sm_scale)
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


def _sol_attn_workspace_bytes(seq_len: int, group_heads: int, token_aug: int = 64) -> int:
    """Exact byte size of `comfy_kitchen.sol_attn` workspace (queries `_C.sol_attn_plan` on CPU, 0 VRAM)."""
    if _ck_cuda is not None and hasattr(_ck_cuda, "_C") and hasattr(_ck_cuda._C, "sol_attn_plan"):
        try:
            return int(_ck_cuda._C.sol_attn_plan(1, int(seq_len), int(group_heads), token_aug=int(token_aug))["total"])
        except Exception:
            pass
    n_blk = (int(seq_len) + 63) // 64
    return int(group_heads) * (int(seq_len) * 384 + n_blk * n_blk * 12 + n_blk * 4096)


def _uniform_head_chunk_candidates(heads: int, convrot_unit: int = 2) -> list[int]:
    """Ascending ConvRot-aligned `head_chunks` covering all distinct even max-group widths.

    Includes smooth intermediate ConvRot-256-aligned partitions (e.g. for 56 heads: `hc=10` ->
    `[6,6,6,6,6,6,6,6,4,4]` between `hc=7` (`g=8`) and `hc=14` (`g=4`)), avoiding the 2x jump
    from 7 to 14 on 15s 1080p while keeping shapes constant across all 50 blocks.
    """
    unit = max(1, int(convrot_unit))
    if heads % unit != 0:
        unit = 1
    n_units = max(1, heads // unit)
    seen_max_u: set[int] = set()
    cands: list[int] = []
    for hc in range(1, n_units + 1):
        max_u = (n_units + hc - 1) // hc
        if max_u not in seen_max_u:
            seen_max_u.add(max_u)
            cands.append(hc)
    return cands


def _max_group_heads_for_chunks(heads: int, n_groups: int, convrot_unit: int = 2) -> int:
    """Exact maximum group width produced by `_split_heads_even(heads, n_groups, convrot_unit*head_dim, head_dim)`."""
    n_groups = max(1, min(int(n_groups), int(heads)))
    unit = max(1, int(convrot_unit))
    if unit > 1 and heads % unit == 0 and n_groups <= heads // unit:
        n_units = heads // unit
        return ((n_units + n_groups - 1) // n_groups) * unit
    return (heads + n_groups - 1) // n_groups


def _activation_vram_budget_bytes(total_vram_bytes: int) -> int:
    """WDDM Sysmem-spill-free activation VRAM budget (6/8/10/12/16/24GB+): `B_act(V) = 0.88*V - 0.90 GiB`.

    Empirically bracketed on RTX 4060 8GB (Windows WDDM):
    - 10s 1080p hc=4 (`6.00 GiB` <= `6.14 GiB`): 0 spill, full TDP (`202s/it`);
    - 15s 1080p hc=14 (`5.20 GiB` <= `6.14 GiB`): 0 spill, full TDP (`7.1/8.0 GB`);
    - 15s 1080p hc=7 (`6.64 GiB` > `6.14 GiB`): triggers silent WDDM Sysmem fallback (`7.6/8.0 GB`, `29W/58°C` PCIe stall).
    """
    gib = 1024 ** 3
    v_gib = float(total_vram_bytes) / gib
    return int(max(2.0, 0.88 * v_gib - 0.90) * gib)


def _estimate_block_peak_bytes(
    seq_len: int,
    hidden: int,
    heads: int,
    head_dim: int,
    ffn: int,
    group_heads: int,
    token_chunk: int,
    eff_mode: str,
    use_sol: bool,
    sol_token_aug: int = 64,
    dtype_bytes: int = 2,
    lora_rank_sum: int = 0,
) -> tuple[int, int]:
    """Exact `(attn_peak_bytes, mlp_peak_bytes)` simultaneously alive in one INT8-ConvRot DiT block."""
    s = int(seq_len)
    # Resident across attention: x(bf16) + x_qdata(int8) + x_scale(fp32) + rope(~768B) + qkv_mid + out_mid
    base = s * (dtype_bytes * int(hidden) + int(hidden) + 4 + 768 + dtype_bytes * max(0, int(lora_rank_sum)))
    out_buf = (dtype_bytes * int(heads) * int(head_dim) * s) if eff_mode == "exact" else 0
    if use_sol:
        # Live simultaneously inside ck.sol_attn(q_g, k_g, v_g): q_g, k_g, v_g, o_g + sol_attn_plan workspace
        grp = 4 * dtype_bytes * int(group_heads) * int(head_dim) * s + _sol_attn_workspace_bytes(s, group_heads, sol_token_aug)
    else:
        # _staged_sageattn_nhd peak includes BF16 q,k + INT8 q,k (or INT8 q,k + BF16 v + FP8 v + BF16 o) + allocator headroom
        grp = (3 * dtype_bytes + 2) * int(group_heads) * int(head_dim) * s + 16 * int(group_heads) * s
    attn_peak = base + out_buf + grp
    # MLP peak inside _swiglu_chunk: x(bf16) + fc1_c[tc, 2*FFN] + act_c[tc, FFN]
    mlp_peak = dtype_bytes * int(hidden) * s + dtype_bytes * (3 * int(ffn)) * min(s, int(token_chunk))
    return attn_peak, mlp_peak


def _resolve_vram_plan(
    head_chunks: int,
    token_chunk: int,
    out_proj_mode: str,
    seq_len: int,
    hidden: int,
    heads: int,
    head_dim: int,
    ffn: int,
    device: torch.device,
    use_sol: bool = True,
    sol_token_aug: int = 64,
    dtype_bytes: int = 2,
    lora_rank_sum: int = 0,
    convrot_unit: int = 2,
) -> tuple[str, int, int, float, float, float]:
    """Deterministically solve `(eff_mode, eff_hc, eff_tc, attn_gib, mlp_gib, budget_gib)` from closed-form VRAM math.

    Constant across all 50 blocks (safe for `malloc_graph`).
    - `head_chunks == 0`: automatically chooses the fastest uniform ConvRot-aligned `hc` that fits `B_act(V)`.
    - `head_chunks > 0` in `out_proj_mode == "auto"`: keeps user's `hc` if it fits, else bumps only to the
      smallest (fastest) safe uniform divisor (e.g. 8GB 15s 1080p: `hc 4 -> 14`, keeping `tc=16384`).
    """
    gib = float(1024 ** 3)
    req_hc = int(head_chunks)
    req_tc = max(1024, int(token_chunk))
    candidates = _uniform_head_chunk_candidates(heads, convrot_unit)
    max_hc = candidates[-1]

    total_vram = 0
    if device.type == "cuda":
        try:
            total_vram = int(torch.cuda.get_device_properties(device).total_memory)
        except Exception:
            total_vram = 0

    if total_vram <= 0:
        eff_mode = out_proj_mode if out_proj_mode in ("exact", "streaming") else ("exact" if seq_len <= 64000 else "streaming")
        eff_hc = max(1, min(req_hc if req_hc > 0 else 7, heads))
        g_eff = _max_group_heads_for_chunks(heads, eff_hc, convrot_unit)
        ap, mp = _estimate_block_peak_bytes(
            seq_len, hidden, heads, head_dim, ffn, g_eff, req_tc, eff_mode, use_sol, sol_token_aug, dtype_bytes, lora_rank_sum
        )
        return eff_mode, eff_hc, req_tc, ap / gib, mp / gib, 0.0

    budget = _activation_vram_budget_bytes(total_vram)
    v_gib = total_vram / gib

    # 1. Resolve out_proj_mode
    exact_seq_cap = 48000 if v_gib < 12.0 else 64000
    if out_proj_mode in ("exact", "streaming"):
        eff_mode = out_proj_mode
    elif seq_len <= exact_seq_cap:
        eff_mode = "exact"
    elif v_gib >= 14.0:
        # Use exact on >=16GB GPUs whenever a reasonable hc (<= max(req_hc, 4)) fits the exact budget
        probe_hc = max(req_hc, 4) if req_hc > 0 else 4
        probe_hc = min((c for c in candidates if c >= probe_hc), default=max_hc)
        g_probe = _max_group_heads_for_chunks(heads, probe_hc, convrot_unit)
        ap_ex, _ = _estimate_block_peak_bytes(
            seq_len, hidden, heads, head_dim, ffn, g_probe, req_tc, "exact", use_sol, sol_token_aug, dtype_bytes, lora_rank_sum
        )
        eff_mode = "exact" if ap_ex <= budget else "streaming"
    else:
        eff_mode = "streaming"

    # 2. Solve minimum safe ConvRot-aligned head_chunks (exact ceiling max group width)
    # WDDM TDR & DynamicVRAM guard: on <=10GB GPUs with S>=16384, never launch all 56 heads at once (hc=1):
    # empirically at S=42,475, hc=2 (g=28) is 8% FASTER than hc=1 while using 1.70 GiB less reserved VRAM!
    min_safe_hc = 2 if (v_gib < 12.0 and seq_len >= 16384 and max_hc >= 2) else 1
    hc_safe = max_hc
    for cand in candidates:
        if cand < min_safe_hc:
            continue
        g_cand = _max_group_heads_for_chunks(heads, cand, convrot_unit)
        ap, _ = _estimate_block_peak_bytes(
            seq_len, hidden, heads, head_dim, ffn, g_cand, req_tc, eff_mode, use_sol, sol_token_aug, dtype_bytes, lora_rank_sum
        )
        if ap <= budget:
            hc_safe = cand
            break

    if req_hc <= 0:
        eff_hc = hc_safe
    elif out_proj_mode == "auto":
        if req_hc >= hc_safe:
            eff_hc = min(req_hc, heads)
        else:
            eff_hc = hc_safe
    else:
        eff_hc = min(req_hc, heads)

    # 3. Solve maximum safe token_chunk for MLP (never shrinks unless MLP actually exceeds budget)
    eff_tc = req_tc
    if req_hc <= 0 or out_proj_mode == "auto":
        x_bytes = dtype_bytes * int(hidden) * int(seq_len)
        bytes_per_tok = dtype_bytes * (3 * int(ffn))
        avail_mlp = budget - x_bytes
        if avail_mlp < bytes_per_tok * min(int(seq_len), eff_tc):
            tc_max = max(2048, (max(0, avail_mlp) // (bytes_per_tok * 1024)) * 1024)
            eff_tc = max(2048, min(eff_tc, int(tc_max)))

    g_final = _max_group_heads_for_chunks(heads, eff_hc, convrot_unit)
    ap_fin, mp_fin = _estimate_block_peak_bytes(
        seq_len, hidden, heads, head_dim, ffn, g_final, eff_tc, eff_mode, use_sol, sol_token_aug, dtype_bytes, lora_rank_sum
    )
    return eff_mode, eff_hc, eff_tc, ap_fin / gib, mp_fin / gib, budget / gib


def _resolve_out_proj_mode(mode: str, seq_len: int, device: torch.device) -> str:
    """Backwards-compatible wrapper around `_resolve_vram_plan`."""
    eff_mode, _, _, _, _, _ = _resolve_vram_plan(
        head_chunks=4, token_chunk=16384, out_proj_mode=mode, seq_len=seq_len,
        hidden=5376, heads=56, head_dim=128, ffn=14336, device=device, use_sol=True,
    )
    return eff_mode


def _resolve_safe_chunks(
    head_chunks: int,
    token_chunk: int,
    out_proj_mode: str,
    seq_len: int,
    heads: int,
    device: torch.device,
    hidden: int = 5376,
    head_dim: int = 128,
    ffn: int = 14336,
    use_sol: bool = True,
    sol_token_aug: int = 64,
    dtype_bytes: int = 2,
    lora_rank_sum: int = 0,
) -> tuple[int, int]:
    """Backwards-compatible wrapper returning `(eff_hc, eff_tc)` from the analytical VRAM solver."""
    _, eff_hc, eff_tc, _, _, _ = _resolve_vram_plan(
        head_chunks=head_chunks, token_chunk=token_chunk, out_proj_mode=out_proj_mode,
        seq_len=seq_len, hidden=hidden, heads=heads, head_dim=head_dim, ffn=ffn,
        device=device, use_sol=use_sol, sol_token_aug=sol_token_aug,
        dtype_bytes=dtype_bytes, lora_rank_sum=lora_rank_sum,
    )
    return eff_hc, eff_tc


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

def _prepare_rms_adaln_tables(
    norm_w: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold RMSNorm weight `norm_w` into AdaLN `scale` once per block (`[M_rows, D]`)."""
    if norm_w is not None:
        eff_scale = (norm_w.float() * (1.0 + scale.float()) - 1.0).to(dtype).contiguous()
    else:
        eff_scale = scale.to(dtype).contiguous()
    shift_cast = shift.to(dtype).contiguous()
    return eff_scale, shift_cast


def _rms_adaln_chunk(
    x_c: torch.Tensor,
    norm_shape,
    norm_w: torch.Tensor | None,
    eps: float,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eff_scale: torch.Tensor,
    shift_cast: torch.Tensor,
    sub_segs: list,
    use_fused_rms: bool = True,
) -> torch.Tensor:
    """Fused `_ck.rms_adaln` per uniform mod segment, falling back to `F.rms_norm + _mod_scale_shift`."""
    can_fused = (
        use_fused_rms
        and _ck is not None
        and hasattr(_ck, "rms_adaln")
        and x_c.is_cuda
        and x_c.dtype in (torch.bfloat16, torch.float16)
    )
    if can_fused:
        if len(sub_segs) == 1:
            a, b, row = sub_segs[0]
            if a == 0 and b == x_c.shape[0] and isinstance(row, int):
                return _ck.rms_adaln(x_c, eff_scale[row], shift_cast[row], eps)
        h_c = torch.empty_like(x_c)
        for a, b, row in sub_segs:
            if b <= a:
                continue
            if isinstance(row, int):
                h_c[a:b] = _ck.rms_adaln(x_c[a:b], eff_scale[row], shift_cast[row], eps)
            else:
                seg_h = F.rms_norm(x_c[a:b], norm_shape, norm_w, eps)
                seg_h.mul_(1.0 + _mod_row(scale, row, x_c.dtype)).add_(_mod_row(shift, row, x_c.dtype))
                h_c[a:b] = seg_h
        return h_c

    h_c = F.rms_norm(x_c, norm_shape, norm_w, eps)
    _mod_scale_shift(h_c, shift, scale, sub_segs)
    return h_c


def _int8_linear_residual_inplace(
    x_in: torch.Tensor,
    w_qdata: torch.Tensor,
    w_scale: torch.Tensor,
    ws_cutlass: torch.Tensor | None,
    bias_arg: torch.Tensor | None,
    bias_orig: torch.Tensor | None,
    convrot: bool,
    convrot_gs: int,
    x_resid: torch.Tensor,
    gate: torch.Tensor,
    gate_cast: torch.Tensor,
    sub_segs: list,
    input_act: str | None = None,
) -> None:
    """In-place gated residual `x_resid += gate * int8_linear(x_in, w)` fused into CUTLASS epilogue.

    Eliminates intermediate `[M, N]` projection buffers and `_mod_gate` HBM read-modify-writes
    for every contiguous uniform mod segment (100% bit-exact verified on SM89).
    """
    dtype = x_resid.dtype
    device = x_resid.device
    n_out, k_w = w_qdata.shape
    can_cutlass_base = (
        x_in.is_cuda
        and _ck_cuda is not None
        and ws_cutlass is not None
        and bias_arg is not None
        and hasattr(_ck_cuda._C, "cutlass_int8_dequant_residual")
        and not _ck_cuda._DISABLE_CUTLASS_INT8
        and _ck_cuda._cuda_device_supports_cutlass_int8_dequant(x_in)
        and x_in.dtype == dtype
        and dtype in (torch.bfloat16, torch.float16)
    )
    if can_cutlass_base:
        out_dtype_code = _ck_cuda.DTYPE_TO_CODE[dtype]
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
        fused_convrot_ok = (
            convrot
            and convrot_gs == 256
            and k_w % 256 == 0
            and 256 <= k_w <= _ck_cuda._CONVROT_FUSED_MAX_K
            and _ck_cuda._convrot_fused_shared_memory_fits(x_in, k_w, convrot_gs)
        )
        act_code = _ck_cuda._input_act_code(input_act)
        empty_act = _ck_cuda._empty_cuda_tensor(device, x_in.dtype)

        for a, b, row in sub_segs:
            m_seg = b - a
            if m_seg <= 0:
                continue
            r_slice = x_resid[a:b]
            use_epilogue = (
                isinstance(row, int)
                and m_seg > 2
                and (input_act in (None, "none") or fused_convrot_ok)
                and not _ck_cuda._prefer_cublas_int8_fallback(m_seg, n_out, k_w, x_in.get_device())
                and r_slice.is_contiguous()
                and (r_slice.data_ptr() % 16 == 0)
            )
            if use_epilogue:
                x_seg = x_in[a:b]
                if not x_seg.is_contiguous():
                    x_seg = x_seg.contiguous()
                q_seg = torch.empty((m_seg, k_w), dtype=torch.int8, device=device)
                s_seg = torch.empty((m_seg, 1), dtype=torch.float32, device=device)
                if convrot and fused_convrot_ok:
                    _ck_cuda._C.quantize_int8_rowwise_convrot64(
                        _ck_cuda._wrap_for_dlpack(x_seg),
                        _ck_cuda._wrap_for_dlpack(q_seg),
                        _ck_cuda._wrap_for_dlpack(s_seg),
                        convrot_gs,
                        False,
                        act_code,
                        0,
                        _ck_cuda._wrap_for_dlpack(empty_act),
                        0.0,
                        stream_ptr,
                    )
                else:
                    _quantize_int8_convrot_into(x_seg, q_seg, s_seg, convrot, convrot_gs)
                gate_vec = _ck_cuda._gemm_vector_arg(gate_cast[row], device, dtype)
                used = _ck_cuda._C.cutlass_int8_dequant_residual(
                    _ck_cuda._wrap_for_dlpack(q_seg),
                    _ck_cuda._wrap_for_dlpack(w_qdata),
                    _ck_cuda._wrap_for_dlpack(s_seg),
                    _ck_cuda._wrap_for_dlpack(ws_cutlass),
                    _ck_cuda._wrap_for_dlpack(bias_arg),
                    _ck_cuda._wrap_for_dlpack(gate_vec),
                    _ck_cuda._wrap_for_dlpack(r_slice),
                    _ck_cuda._wrap_for_dlpack(r_slice),
                    out_dtype_code,
                    stream_ptr,
                )
                del q_seg, s_seg
                if used:
                    continue

            # Fallback for tiny / non-uniform / non-CUTLASS segments
            proj_seg = _ck.int8_linear(
                x_in[a:b], w_qdata, w_scale, bias_orig, dtype,
                convrot=convrot, convrot_groupsize=convrot_gs, input_act=input_act,
            )
            r_slice.addcmul_(proj_seg, _mod_row(gate, row, dtype))
            del proj_seg
        return

    proj_c = _ck.int8_linear(
        x_in, w_qdata, w_scale, bias_orig, dtype,
        convrot=convrot, convrot_groupsize=convrot_gs, input_act=input_act,
    )
    _mod_gate(x_resid, gate, proj_c, sub_segs)
    del proj_c


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
    qkv_loras=None,
    out_loras=None,
    attn_temp_scale: float = 1.0,
    qkv_rot_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    out_merge: int = 1,
) -> None:
    """INT8 TensorWise ConvRot fast path: fused RMS-AdaLN + Rotated-INT8/BF16 LoRA side-branch + CUTLASS residual epilogue."""
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

    use_rot_qkv = qkv_convrot and qkv_gs == 256 and qkv_rot_pair is not None
    if use_rot_qkv:
        qkv_packed = None
        qkv_b_cat, qkv_a_rot = qkv_rot_pair
        qkv_a_cat = None
        qkv_mid = torch.empty((s, qkv_a_rot.shape[0]), dtype=dtype, device=device)
    else:
        qkv_a_rot = None
        qkv_packed = _prepare_lora_packed(qkv_loras, device, dtype)
        qkv_b_cat = qkv_packed[0] if qkv_packed is not None else None
        qkv_a_cat = qkv_packed[1] if qkv_packed is not None else None
        qkv_mid = (
            torch.empty((s, qkv_a_cat.shape[0]), dtype=dtype, device=device)
            if qkv_a_cat is not None
            else None
        )

    use_fused_rms = eff_mode == "streaming"
    with _cast_rmsnorm_once(block.norm1, x) as nw1:
        eff_scale1, shift1_cast = _prepare_rms_adaln_tables(nw1, scale_msa, shift_msa, dtype)
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            h_c = _rms_adaln_chunk(
                x[t0:t1], block.norm1.normalized_shape, nw1, block.norm1.eps,
                shift_msa, scale_msa, eff_scale1, shift1_cast, sub_segs,
                use_fused_rms=use_fused_rms,
            )
            if use_rot_qkv:
                _quantize_int8_convrot_into(h_c, x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
                del h_c
                rot_int8_lora_down(x_qdata[t0:t1], x_scale[t0:t1], qkv_a_rot, out=qkv_mid[t0:t1])
            else:
                if qkv_mid is not None:
                    qkv_mid[t0:t1] = F.linear(h_c, qkv_a_cat)
                _quantize_int8_convrot_into(h_c, x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
                del h_c

    # Release qkv `A` / `A_rot` matrix immediately; only `(qkv_mid [S,R], qkv_b_cat [21504,R])` needed below
    del qkv_a_cat, qkv_a_rot, qkv_packed

    def _add_qkv_lora_inplace(buf_2d: torch.Tensor, r0: int, r1: int) -> None:
        if qkv_mid is None:
            return
        torch.addmm(buf_2d, qkv_mid, qkv_b_cat[r0:r1].t(), beta=1.0, alpha=1.0, out=buf_2d)

    qw = mm.cast_to(attn.q_norm.weight, device=device) if rope_freqs is not None else None
    kw = mm.cast_to(attn.k_norm.weight, device=device) if rope_freqs is not None else None
    rot_dim = rope_freqs.shape[-3] * 2 if rope_freqs is not None else 0
    gate_msa_cast = gate_msa.to(dtype).contiguous()

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
                out_packed = _prepare_lora_packed(out_loras, device, dtype)
                out_b_cat = out_packed[0] if out_packed is not None else None
                out_a_cat = out_packed[1] if out_packed is not None else None
                out_mid = (
                    torch.zeros((s, out_a_cat.shape[0]), dtype=dtype, device=device)
                    if out_a_cat is not None
                    else None
                )
                ws_out_cutlass = None
                bias_out_arg = None
                bias_empty_arg = None
                if out_is_int8 and _ck_cuda is not None:
                    ws_tmp = _ck_cuda._int8_weight_scale_arg(w_out._params.scale, device)
                    ws_out_cutlass = ws_tmp if ws_tmp.numel() == hidden else ws_tmp.expand(hidden).contiguous()
                    bias_empty_arg = _ck_cuda._empty_cuda_tensor(device, dtype)
                    bias_out_arg = (
                        _ck_cuda._gemm_vector_arg(b_out, device, dtype)
                        if b_out is not None
                        else bias_empty_arg
                    )

                group_sizes = _split_heads_even(heads, head_chunks, out_gs if out_convrot else 0, head_dim)
                eff_out_merge = max(1, min(int(out_merge), len(group_sizes)))
                pending_os: list[torch.Tensor] = []
                pending_c0: int | None = None
                pending_c1: int = 0
                flushed_batches = 0

                def _flush_pending_out() -> None:
                    nonlocal pending_c0, pending_c1, flushed_batches
                    if not pending_os:
                        return
                    bc0, bc1 = int(pending_c0), int(pending_c1)
                    b_group = b_out if flushed_batches == 0 else None
                    b_group_arg = bias_out_arg if flushed_batches == 0 else bias_empty_arg
                    if out_is_int8:
                        w_out_g = w_out._qdata[:, bc0:bc1].contiguous()
                        out_scale = w_out._params.scale
                        if len(pending_os) == 1:
                            o_single = pending_os[0]
                            for t0 in range(0, s, token_chunk):
                                t1 = min(t0 + token_chunk, s)
                                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                                _int8_linear_residual_inplace(
                                    o_single[t0:t1], w_out_g, out_scale, ws_out_cutlass, b_group_arg, b_group,
                                    out_convrot, out_gs, x[t0:t1], gate_msa, gate_msa_cast, sub_segs,
                                )
                        else:
                            # Chunk-wise horizontal cat keeps scratch <= ~32 MiB while cutting 2.38 GiB
                            # residual `x` HBM read-modify-writes by 2x-3x and widening CUTLASS INT8 K!
                            for t0 in range(0, s, token_chunk):
                                t1 = min(t0 + token_chunk, s)
                                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                                o_cat = torch.cat([p[t0:t1] for p in pending_os], dim=1)
                                _int8_linear_residual_inplace(
                                    o_cat, w_out_g, out_scale, ws_out_cutlass, b_group_arg, b_group,
                                    out_convrot, out_gs, x[t0:t1], gate_msa, gate_msa_cast, sub_segs,
                                )
                                del o_cat
                        del w_out_g
                    else:
                        w_out_g = w_out[:, bc0:bc1]
                        if len(pending_os) == 1:
                            o_single = pending_os[0]
                            for t0 in range(0, s, token_chunk):
                                t1 = min(t0 + token_chunk, s)
                                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                                proj_c = F.linear(o_single[t0:t1], w_out_g, b_group)
                                _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                                del proj_c
                        else:
                            for t0 in range(0, s, token_chunk):
                                t1 = min(t0 + token_chunk, s)
                                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                                o_cat = torch.cat([p[t0:t1] for p in pending_os], dim=1)
                                proj_c = F.linear(o_cat, w_out_g, b_group)
                                del o_cat
                                _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                                del proj_c
                        del w_out_g
                    pending_os.clear()
                    pending_c0 = None
                    flushed_batches += 1

                hs = 0
                n_grp_total = len(group_sizes)
                for g_idx, g in enumerate(group_sizes):
                    he = hs + g
                    c0, c1 = hs * head_dim, he * head_dim
                    q_g_2d = _int8_gemm_prequant(
                        x_qdata, x_scale, qkv_qdata[c0:c1], _slice_scale(qkv_scale, c0, c1),
                        dtype, _slice_bias(b_qkv, c0, c1),
                    )
                    _add_qkv_lora_inplace(q_g_2d, c0, c1)
                    k_g_2d = _int8_gemm_prequant(
                        x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1],
                        _slice_scale(qkv_scale, inner + c0, inner + c1),
                        dtype, _slice_bias(b_qkv, inner + c0, inner + c1),
                    )
                    _add_qkv_lora_inplace(k_g_2d, inner + c0, inner + c1)
                    q_g, k_g = _apply_qk_norm_rope(
                        attn, q_g_2d.view(1, s, g, head_dim), k_g_2d.view(1, s, g, head_dim),
                        rope_freqs, qw, kw, rot_dim,
                    )

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        v_2d = _int8_gemm_prequant(
                            x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1],
                            _slice_scale(qkv_scale, 2 * inner + _c0, 2 * inner + _c1),
                            dtype, _slice_bias(b_qkv, 2 * inner + _c0, 2 * inner + _c1),
                        )
                        _add_qkv_lora_inplace(v_2d, 2 * inner + _c0, 2 * inner + _c1)
                        return v_2d.view(1, s, _g, head_dim)

                    o_g = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
                    del q_g, k_g, q_g_2d, k_g_2d
                    o_g_2d = o_g.view(s, g * head_dim)
                    del o_g
                    if out_mid is not None:
                        torch.addmm(out_mid, o_g_2d, out_a_cat[:, c0:c1].t(), beta=1.0, alpha=1.0, out=out_mid)
                    if pending_c0 is None:
                        pending_c0 = c0
                    pending_c1 = c1
                    pending_os.append(o_g_2d)
                    del o_g_2d
                    if len(pending_os) >= eff_out_merge or g_idx == n_grp_total - 1:
                        _flush_pending_out()
                    hs = he

                del x_qdata, x_scale, qkv_mid, qkv_b_cat, out_a_cat, out_packed
                # Apply out_proj LoRA up-projection ONCE via gated addmm directly into residual x
                if out_mid is not None:
                    gated_b_cache: dict[int, torch.Tensor] = {}
                    _add_gated_lora_mid_residual_inplace(
                        x, out_mid, out_b_cat, gate_msa, gate_msa_cast, mod_segments, gated_b_cache
                    )
                    del out_mid, out_b_cat, gated_b_cache
                return

    # Exact mode (Option 1): 100.00% bit-identical to stock KJ SageAttention2++
    group_sizes = _split_heads_even(heads, head_chunks, 0, head_dim)
    out = torch.empty((s, inner), dtype=dtype, device=device)
    out_nhd = out.view(1, s, heads, head_dim)
    hs = 0
    for g in group_sizes:
        he = hs + g
        c0, c1 = hs * head_dim, he * head_dim
        q_g_2d = _int8_gemm_prequant(
            x_qdata, x_scale, qkv_qdata[c0:c1], _slice_scale(qkv_scale, c0, c1),
            dtype, _slice_bias(b_qkv, c0, c1),
        )
        _add_qkv_lora_inplace(q_g_2d, c0, c1)
        k_g_2d = _int8_gemm_prequant(
            x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1],
            _slice_scale(qkv_scale, inner + c0, inner + c1),
            dtype, _slice_bias(b_qkv, inner + c0, inner + c1),
        )
        _add_qkv_lora_inplace(k_g_2d, inner + c0, inner + c1)
        q_g, k_g = _apply_qk_norm_rope(
            attn, q_g_2d.view(1, s, g, head_dim), k_g_2d.view(1, s, g, head_dim),
            rope_freqs, qw, kw, rot_dim,
        )

        def _make_v(_c0=c0, _c1=c1, _g=g):
            v_2d = _int8_gemm_prequant(
                x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1],
                _slice_scale(qkv_scale, 2 * inner + _c0, 2 * inner + _c1),
                dtype, _slice_bias(b_qkv, 2 * inner + _c0, 2 * inner + _c1),
            )
            _add_qkv_lora_inplace(v_2d, 2 * inner + _c0, 2 * inner + _c1)
            return v_2d.view(1, s, _g, head_dim)

        out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
        del q_g, k_g, q_g_2d, k_g_2d
        hs = he

    del x_qdata, x_scale, qkv_mid, qkv_b_cat

    with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
        out_is_int8 = _is_int8_tensorwise(w_out)
        out_packed = _prepare_lora_packed(out_loras, device, dtype)
        gated_out_cache: dict[int, torch.Tensor] = {}
        ws_out_cutlass = None
        bias_out_arg = None
        out_convrot = bool(getattr(w_out._params, "convrot", False)) if out_is_int8 else False
        out_gs = int(getattr(w_out._params, "convrot_groupsize", 256)) if out_is_int8 else 256
        if out_is_int8 and _ck_cuda is not None:
            ws_tmp = _ck_cuda._int8_weight_scale_arg(w_out._params.scale, device)
            ws_out_cutlass = ws_tmp if ws_tmp.numel() == hidden else ws_tmp.expand(hidden).contiguous()
            bias_out_arg = (
                _ck_cuda._gemm_vector_arg(b_out, device, dtype)
                if b_out is not None
                else _ck_cuda._empty_cuda_tensor(device, dtype)
            )
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            o_chunk = out[t0:t1]
            if out_is_int8:
                _int8_linear_residual_inplace(
                    o_chunk, w_out._qdata, w_out._params.scale, ws_out_cutlass, bias_out_arg, b_out,
                    out_convrot, out_gs, x[t0:t1], gate_msa, gate_msa_cast, sub_segs,
                )
            else:
                proj_c = _linear_with_cast_weight(o_chunk, w_out, b_out)
                _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                del proj_c
            if out_packed is not None:
                o_mid = F.linear(o_chunk, out_packed[1])
                _add_gated_lora_mid_residual_inplace(
                    x[t0:t1], o_mid, out_packed[0], gate_msa, gate_msa_cast, sub_segs, gated_out_cache
                )
                del o_mid
        del out_packed, gated_out_cache
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
    qkv_loras=None,
    out_loras=None,
    attn_temp_scale: float = 1.0,
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
    qkv_packed = _prepare_lora_packed(qkv_loras, device, dtype)

    use_fused_rms = eff_mode == "streaming"
    with _cast_rmsnorm_once(block.norm1, x) as nw1:
        eff_scale1, shift1_cast = _prepare_rms_adaln_tables(nw1, scale_msa, shift_msa, dtype)

        def _project_rows(r0: int, r1: int) -> torch.Tensor:
            w_slice = w_qkv[r0:r1]
            b_slice = b_qkv[r0:r1] if b_qkv is not None else None
            buf = torch.empty((s, r1 - r0), dtype=dtype, device=device)
            for t0 in range(0, s, token_chunk):
                t1 = min(t0 + token_chunk, s)
                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                h_c = _rms_adaln_chunk(
                    x[t0:t1], block.norm1.normalized_shape, nw1, block.norm1.eps,
                    shift_msa, scale_msa, eff_scale1, shift1_cast, sub_segs,
                    use_fused_rms=use_fused_rms,
                )
                out_c = F.linear(h_c, w_slice, b_slice)
                if qkv_packed is not None:
                    mid_c = F.linear(h_c, qkv_packed[1])
                    torch.addmm(out_c, mid_c, qkv_packed[0][r0:r1].t(), beta=1.0, alpha=1.0, out=out_c)
                    del mid_c
                buf[t0:t1] = out_c
                del h_c, out_c
            return buf

        if eff_mode == "streaming":
            with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
                if not isinstance(w_out, qops.QuantizedTensor):
                    out_packed = _prepare_lora_packed(out_loras, device, dtype)
                    hs = 0
                    for g_idx, g in enumerate(group_sizes):
                        he = hs + g
                        c0, c1 = hs * head_dim, he * head_dim
                        q_g = _project_rows(c0, c1).view(1, s, g, head_dim)
                        k_g = _project_rows(inner + c0, inner + c1).view(1, s, g, head_dim)
                        q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

                        def _make_v(_c0=c0, _c1=c1, _g=g):
                            return _project_rows(2 * inner + _c0, 2 * inner + _c1).view(1, s, _g, head_dim)

                        o_g = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
                        del q_g, k_g
                        o_g_2d = o_g.view(s, g * head_dim)
                        w_out_g = w_out[:, c0:c1]
                        b_group = b_out if g_idx == 0 else None
                        for t0 in range(0, s, token_chunk):
                            t1 = min(t0 + token_chunk, s)
                            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                            o_chunk = o_g_2d[t0:t1]
                            proj_c = F.linear(o_chunk, w_out_g, b_group)
                            if out_packed is not None:
                                mid_o = F.linear(o_chunk, out_packed[1][:, c0:c1])
                                torch.addmm(proj_c, mid_o, out_packed[0].t(), beta=1.0, alpha=1.0, out=proj_c)
                                del mid_o
                            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
                            del proj_c
                        del o_g, o_g_2d, w_out_g
                        hs = he
                    del qkv_packed, out_packed
                    return

        out = torch.empty((s, inner), dtype=dtype, device=device)
        out_nhd = out.view(1, s, heads, head_dim)
        hs = 0
        for g in group_sizes:
            he = hs + g
            c0, c1 = hs * head_dim, he * head_dim
            q_g = _project_rows(c0, c1).view(1, s, g, head_dim)
            k_g = _project_rows(inner + c0, inner + c1).view(1, s, g, head_dim)
            q_g, k_g = _apply_qk_norm_rope(attn, q_g, k_g, rope_freqs, qw, kw, rot_dim)

            def _make_v(_c0=c0, _c1=c1, _g=g):
                return _project_rows(2 * inner + _c0, 2 * inner + _c1).view(1, s, _g, head_dim)

            out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
            del q_g, k_g
            hs = he
    del qkv_packed

    with _cast_linear_once(attn.out_proj, device, dtype) as (w_out, b_out):
        out_packed = _prepare_lora_packed(out_loras, device, dtype)
        for t0 in range(0, s, token_chunk):
            t1 = min(t0 + token_chunk, s)
            sub_segs = _slice_mod_segments(mod_segments, t0, t1)
            o_chunk = out[t0:t1]
            proj_c = _linear_with_cast_weight(o_chunk, w_out, b_out)
            if out_packed is not None:
                _add_lora_inplace(proj_c, o_chunk, out_packed)
            _mod_gate(x[t0:t1], gate_msa, proj_c, sub_segs)
            del proj_c
        del out_packed
    del out, out_nhd


def _run_chunked_mlp_inplace(
    block,
    x: torch.Tensor,
    shift_mlp,
    scale_mlp,
    gate_mlp,
    mod_segments,
    token_chunk: int,
    transformer_options: dict | None = None,
    use_fused_rms: bool = True,
) -> None:
    """Fused token-chunked `rms_adaln + SwiGLU MLP + BF16 LoRA side-branch + CUTLASS residual epilogue`."""
    s, hidden = x.shape
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

    fc1_rot_pair = _get_linear_rot_lora(fc1, transformer_options, device, dtype)
    fc1_packed = None if fc1_rot_pair is not None else _prepare_lora_packed(_get_linear_loras(fc1, transformer_options), device, dtype)
    fc2_packed = _prepare_lora_packed(_get_linear_loras(fc2, transformer_options), device, dtype)
    gated_fc2_cache: dict[int, torch.Tensor] = {}
    gate_mlp_cast = gate_mlp.to(dtype).contiguous()

    with _cast_rmsnorm_once(block.norm2, x) as nw2:
        eff_scale2, shift2_cast = _prepare_rms_adaln_tables(nw2, scale_mlp, shift_mlp, dtype)
        if has_extra_scales:
            for t0 in range(0, s, token_chunk):
                t1 = min(t0 + token_chunk, s)
                sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                h_c = _rms_adaln_chunk(
                    x[t0:t1], block.norm2.normalized_shape, nw2, block.norm2.eps,
                    shift_mlp, scale_mlp, eff_scale2, shift2_cast, sub_segs,
                    use_fused_rms=use_fused_rms,
                )
                mlp_c = comfy.ops.linear_input_act(fc2, fc1(h_c), "swiglu")
                del h_c
                _mod_gate(x[t0:t1], gate_mlp, mlp_c, sub_segs)
                del mlp_c
            return

        with _cast_linear_once(fc1, device, dtype) as (w_fc1, b_fc1):
            with _cast_linear_once(fc2, device, dtype) as (w_fc2, b_fc2):
                fc1_is_int8 = _is_int8_tensorwise(w_fc1)
                fc1_convrot = bool(getattr(w_fc1._params, "convrot", False)) if fc1_is_int8 else False
                fc1_gs = int(getattr(w_fc1._params, "convrot_groupsize", 256)) if fc1_is_int8 else 256
                use_rot_fc1 = (
                    fc1_rot_pair is not None
                    and fc1_is_int8
                    and fc1_convrot
                    and fc1_gs == 256
                    and x.is_cuda
                    and _ck_cuda is not None
                )
                if fc1_rot_pair is not None and not use_rot_fc1:
                    # Non-ConvRot fallback: unpack unrotated side-branch
                    fc1_packed = _prepare_lora_packed(_get_linear_loras(fc1, transformer_options), device, dtype)

                q_fc1_buf = None
                s_fc1_buf = None
                mid_fc1_buf = None
                fc1_b_t = None
                fc1_a_rot = None
                if use_rot_fc1:
                    fc1_b, fc1_a_rot = fc1_rot_pair
                    fc1_b_t = fc1_b.t()
                    max_m = min(s, token_chunk)
                    q_fc1_buf = torch.empty((max_m, hidden), dtype=torch.int8, device=device)
                    s_fc1_buf = torch.empty((max_m, 1), dtype=torch.float32, device=device)
                    mid_fc1_buf = torch.empty((max_m, fc1_a_rot.shape[0]), dtype=dtype, device=device)

                fc2_is_int8 = _is_int8_tensorwise(w_fc2)
                fc2_convrot = bool(getattr(w_fc2._params, "convrot", False)) if fc2_is_int8 else False
                fc2_gs = int(getattr(w_fc2._params, "convrot_groupsize", 256)) if fc2_is_int8 else 256
                ws_fc2_cutlass = None
                bias_fc2_arg = None
                if fc2_is_int8 and _ck_cuda is not None:
                    ws_tmp = _ck_cuda._int8_weight_scale_arg(w_fc2._params.scale, device)
                    ws_fc2_cutlass = ws_tmp if ws_tmp.numel() == hidden else ws_tmp.expand(hidden).contiguous()
                    bias_fc2_arg = (
                        _ck_cuda._gemm_vector_arg(b_fc2, device, dtype)
                        if b_fc2 is not None
                        else _ck_cuda._empty_cuda_tensor(device, dtype)
                    )

                for t0 in range(0, s, token_chunk):
                    t1 = min(t0 + token_chunk, s)
                    sub_segs = _slice_mod_segments(mod_segments, t0, t1)
                    h_c = _rms_adaln_chunk(
                        x[t0:t1], block.norm2.normalized_shape, nw2, block.norm2.eps,
                        shift_mlp, scale_mlp, eff_scale2, shift2_cast, sub_segs,
                        use_fused_rms=use_fused_rms,
                    )
                    if use_rot_fc1:
                        m_c = t1 - t0
                        q_sl = q_fc1_buf[:m_c]
                        s_sl = s_fc1_buf[:m_c]
                        mid_sl = mid_fc1_buf[:m_c]
                        _quantize_int8_convrot_into(h_c, q_sl, s_sl, True, 256)
                        del h_c  # Freed BEFORE allocating [m_c, 28672] fc1_c & zero BF16 h_c re-read!
                        rot_int8_lora_down(q_sl, s_sl, fc1_a_rot, out=mid_sl)
                        fc1_c = _int8_gemm_prequant(
                            q_sl, s_sl, w_fc1._qdata, w_fc1._params.scale, dtype, b_fc1
                        )
                        torch.addmm(fc1_c, mid_sl, fc1_b_t, beta=1.0, alpha=1.0, out=fc1_c)
                    else:
                        fc1_c = _linear_with_cast_weight(h_c, w_fc1, b_fc1)
                        if fc1_packed is not None:
                            _add_lora_inplace(fc1_c, h_c, fc1_packed)
                        del h_c

                    if fc2_packed is not None:
                        # Single-pass fused Triton SwiGLU matching comfy_kitchen FP32 SiLU
                        act_c = _swiglu_chunk(fc1_c)
                        del fc1_c
                        if fc2_is_int8:
                            _int8_linear_residual_inplace(
                                act_c, w_fc2._qdata, w_fc2._params.scale, ws_fc2_cutlass,
                                bias_fc2_arg, b_fc2, fc2_convrot, fc2_gs,
                                x[t0:t1], gate_mlp, gate_mlp_cast, sub_segs, input_act=None,
                            )
                        else:
                            mlp_c = _linear_with_cast_weight(act_c, w_fc2, b_fc2)
                            _mod_gate(x[t0:t1], gate_mlp, mlp_c, sub_segs)
                            del mlp_c
                        fc2_mid = F.linear(act_c, fc2_packed[1])
                        del act_c
                        _add_gated_lora_mid_residual_inplace(
                            x[t0:t1], fc2_mid, fc2_packed[0], gate_mlp, gate_mlp_cast, sub_segs, gated_fc2_cache
                        )
                        del fc2_mid
                    elif fc2_is_int8:
                        _int8_linear_residual_inplace(
                            fc1_c, w_fc2._qdata, w_fc2._params.scale, ws_fc2_cutlass,
                            bias_fc2_arg, b_fc2, fc2_convrot, fc2_gs,
                            x[t0:t1], gate_mlp, gate_mlp_cast, sub_segs, input_act="swiglu",
                        )
                        del fc1_c
                    else:
                        mlp_c = comfy.ops.linear_input_act_(fc1_c, w_fc2, b_fc2, "swiglu")
                        del fc1_c
                        _mod_gate(x[t0:t1], gate_mlp, mlp_c, sub_segs)
                        del mlp_c
    del fc1_packed, fc1_rot_pair, fc2_packed, gated_fc2_cache


def _make_chunked_attn_forward(head_chunks: int, token_chunk: int, out_proj_mode: str = "auto"):
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
        sol_ctx = transformer_options.get("_chunked_sage_sol_ctx") if isinstance(transformer_options, dict) else None
        qkv_loras_sa = _get_linear_loras(self.qkv_proj, transformer_options)
        out_loras_sa = _get_linear_loras(self.out_proj, transformer_options)
        lora_r_sa = sum(int(a.shape[0]) for _, a, _ in qkv_loras_sa) + sum(int(a.shape[0]) for _, a, _ in out_loras_sa)
        n_groups, eff_token_chunk = _resolve_safe_chunks(
            n_groups, token_chunk, out_proj_mode, s, heads, device,
            hidden=int(x.shape[1]), head_dim=head_dim,
            use_sol=(sol_ctx is not None),
            sol_token_aug=int(sol_ctx["token_aug"]) if sol_ctx is not None else 64,
            dtype_bytes=x.element_size(),
            lora_rank_sum=lora_r_sa,
        )
        block_idx = transformer_options.get("block_index") if isinstance(transformer_options, dict) else None
        attn_temp_scale = float(transformer_options.get("_chunked_sage_attn_temp", 1.0)) if isinstance(transformer_options, dict) else 1.0

        qw = mm.cast_to(self.q_norm.weight, device=device) if rope_freqs is not None else None
        kw = mm.cast_to(self.k_norm.weight, device=device) if rope_freqs is not None else None
        rot_dim = rope_freqs.shape[-3] * 2 if rope_freqs is not None else 0
        qkv_rot_sa = _get_linear_rot_lora(self.qkv_proj, transformer_options, device, dtype)
        qkv_packed = _prepare_lora_packed(qkv_loras_sa, device, dtype)
        out_packed = _prepare_lora_packed(out_loras_sa, device, dtype)

        with _cast_linear_once(self.qkv_proj, device, dtype) as (w_qkv, b_qkv):
            if _is_int8_tensorwise(w_qkv) and x.is_cuda and _ck_cuda is not None:
                qkv_qdata = w_qkv._qdata
                qkv_scale = w_qkv._params.scale
                qkv_convrot = bool(getattr(w_qkv._params, "convrot", False))
                qkv_gs = int(getattr(w_qkv._params, "convrot_groupsize", 256))
                x_qdata = torch.empty((s, x.shape[1]), dtype=torch.int8, device=device)
                x_scale = torch.empty((s, 1), dtype=torch.float32, device=device)
                use_rot_sa = qkv_convrot and qkv_gs == 256 and qkv_rot_sa is not None
                if use_rot_sa:
                    qkv_b_sa, qkv_a_rot_sa = qkv_rot_sa
                    qkv_mid = torch.empty((s, qkv_a_rot_sa.shape[0]), dtype=dtype, device=device)
                else:
                    qkv_b_sa = qkv_packed[0] if qkv_packed is not None else None
                    qkv_a_rot_sa = None
                    qkv_mid = (
                        torch.empty((s, qkv_packed[1].shape[0]), dtype=dtype, device=device)
                        if qkv_packed is not None
                        else None
                    )
                for t0 in range(0, s, eff_token_chunk):
                    t1 = min(t0 + eff_token_chunk, s)
                    if use_rot_sa:
                        _quantize_int8_convrot_into(x[t0:t1], x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
                        rot_int8_lora_down(x_qdata[t0:t1], x_scale[t0:t1], qkv_a_rot_sa, out=qkv_mid[t0:t1])
                    else:
                        if qkv_mid is not None:
                            qkv_mid[t0:t1] = F.linear(x[t0:t1], qkv_packed[1])
                        _quantize_int8_convrot_into(x[t0:t1], x_qdata[t0:t1], x_scale[t0:t1], qkv_convrot, qkv_gs)
                del x, qkv_a_rot_sa

                def _add_sa_qkv_lora(buf_2d: torch.Tensor, r0: int, r1: int) -> None:
                    if qkv_mid is None:
                        return
                    torch.addmm(buf_2d, qkv_mid, qkv_b_sa[r0:r1].t(), beta=1.0, alpha=1.0, out=buf_2d)

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
                    q_2d = _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[c0:c1], _sc(c0, c1), dtype, _bs(c0, c1))
                    _add_sa_qkv_lora(q_2d, c0, c1)
                    k_2d = _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[inner + c0 : inner + c1], _sc(inner + c0, inner + c1), dtype, _bs(inner + c0, inner + c1))
                    _add_sa_qkv_lora(k_2d, inner + c0, inner + c1)
                    q_g, k_g = _apply_qk_norm_rope(self, q_2d.view(1, s, g, head_dim), k_2d.view(1, s, g, head_dim), rope_freqs, qw, kw, rot_dim)

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        v_2d = _int8_gemm_prequant(x_qdata, x_scale, qkv_qdata[2 * inner + _c0 : 2 * inner + _c1], _sc(2 * inner + _c0, 2 * inner + _c1), dtype, _bs(2 * inner + _c0, 2 * inner + _c1))
                        _add_sa_qkv_lora(v_2d, 2 * inner + _c0, 2 * inner + _c1)
                        return v_2d.view(1, s, _g, head_dim)

                    out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
                    del q_g, k_g, q_2d, k_2d
                    hs = he
                del x_qdata, x_scale, qkv_mid
            elif not isinstance(w_qkv, qops.QuantizedTensor):
                out = torch.empty((s, inner), dtype=dtype, device=device)
                out_nhd = out.view(1, s, heads, head_dim)
                qkv_mid = F.linear(x, qkv_packed[1]) if qkv_packed is not None else None
                hs = 0
                for g in _split_heads_even(heads, n_groups, 0, head_dim):
                    he = hs + g
                    c0, c1 = hs * head_dim, he * head_dim
                    q_2d = F.linear(x, w_qkv[c0:c1], b_qkv[c0:c1] if b_qkv is not None else None)
                    k_2d = F.linear(x, w_qkv[inner + c0 : inner + c1], b_qkv[inner + c0 : inner + c1] if b_qkv is not None else None)
                    if qkv_mid is not None:
                        torch.addmm(q_2d, qkv_mid, qkv_packed[0][c0:c1].t(), beta=1.0, alpha=1.0, out=q_2d)
                        torch.addmm(k_2d, qkv_mid, qkv_packed[0][inner + c0 : inner + c1].t(), beta=1.0, alpha=1.0, out=k_2d)
                    q_g, k_g = _apply_qk_norm_rope(self, q_2d.view(1, s, g, head_dim), k_2d.view(1, s, g, head_dim), rope_freqs, qw, kw, rot_dim)

                    def _make_v(_c0=c0, _c1=c1, _g=g):
                        v_2d = F.linear(x, w_qkv[2 * inner + _c0 : 2 * inner + _c1], b_qkv[2 * inner + _c0 : 2 * inner + _c1] if b_qkv is not None else None)
                        if qkv_mid is not None:
                            torch.addmm(v_2d, qkv_mid, qkv_packed[0][2 * inner + _c0 : 2 * inner + _c1].t(), beta=1.0, alpha=1.0, out=v_2d)
                        return v_2d.view(1, s, _g, head_dim)

                    out_nhd[:, :, hs:he] = _attend_group(q_g, k_g, _make_v, dtype, sol_ctx, block_idx, attn_temp_scale=attn_temp_scale)
                    del q_g, k_g, q_2d, k_2d
                    hs = he
                del x, qkv_mid
            else:
                # Non-INT8 quantized fallback (e.g. FP8/NVFP4): chunk QKV along tokens
                q_all = torch.empty((s, inner), dtype=dtype, device=device)
                k_all = torch.empty((s, inner), dtype=dtype, device=device)
                v_all = torch.empty((s, inner), dtype=dtype, device=device)
                for t0 in range(0, s, eff_token_chunk):
                    t1 = min(t0 + eff_token_chunk, s)
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
        del qkv_packed

        res = torch.empty((s, self.out_proj.out_features), dtype=dtype, device=device)
        with _cast_linear_once(self.out_proj, device, dtype) as (w_out, b_out):
            for t0 in range(0, s, eff_token_chunk):
                t1 = min(t0 + eff_token_chunk, s)
                o_c = out[t0:t1]
                r_c = _linear_with_cast_weight(o_c, w_out, b_out)
                if out_packed is not None:
                    _add_lora_inplace(r_c, o_c, out_packed)
                res[t0:t1] = r_c
                del r_c
        del out, out_nhd, out_packed
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
            eff_mode_ext = _resolve_out_proj_mode(out_proj_mode, s, device)
            _run_chunked_mlp_inplace(
                self, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, token_chunk,
                transformer_options=transformer_options,
                use_fused_rms=(eff_mode_ext == "streaming"),
            )
            return x

        req_groups = head_chunks
        if isinstance(transformer_options, dict) and transformer_options.get("minimax_head_chunks"):
            req_groups = max(req_groups, int(transformer_options["minimax_head_chunks"]))
        sol_ctx = transformer_options.get("_chunked_sage_sol_ctx") if isinstance(transformer_options, dict) else None
        block_idx = transformer_options.get("block_index") if isinstance(transformer_options, dict) else None
        attn_temp_scale = float(transformer_options.get("_chunked_sage_attn_temp", 1.0)) if isinstance(transformer_options, dict) else 1.0
        qkv_loras = _get_linear_loras(self.attn.qkv_proj, transformer_options)
        out_loras = _get_linear_loras(self.attn.out_proj, transformer_options)
        lora_rank_sum = sum(int(a.shape[0]) for _, a, _ in qkv_loras) + sum(int(a.shape[0]) for _, a, _ in out_loras)
        ffn_dim = int(self.mlp.fc1.out_features) // 2
        eff_mode, n_groups, eff_token_chunk, pred_attn_gib, pred_mlp_gib, budget_gib = _resolve_vram_plan(
            head_chunks=req_groups,
            token_chunk=token_chunk,
            out_proj_mode=out_proj_mode,
            seq_len=s,
            hidden=int(x.shape[1]),
            heads=int(self.attn.heads),
            head_dim=int(self.attn.head_dim),
            ffn=ffn_dim,
            device=device,
            use_sol=(sol_ctx is not None),
            sol_token_aug=int(sol_ctx["token_aug"]) if sol_ctx is not None else 64,
            dtype_bytes=x.element_size(),
            lora_rank_sum=lora_rank_sum,
        )
        out_merge = 1
        g_show = _max_group_heads_for_chunks(int(self.attn.heads), n_groups)
        if eff_mode == "streaming" and n_groups >= 4 and budget_gib > 0.0:
            o_grp_gib = (g_show * int(self.attn.head_dim) * x.element_size() * s) / float(1024 ** 3)
            slack_gib = budget_gib - pred_attn_gib - 0.45
            if o_grp_gib > 0.0 and slack_gib >= o_grp_gib:
                out_merge = min(3, 1 + int(slack_gib // o_grp_gib))
        if block_idx == 0 and (n_groups != req_groups or eff_token_chunk != token_chunk or out_merge > 1):
            hc_label = f"auto→{n_groups}" if req_groups <= 0 else f"{req_groups}→{n_groups}"
            tc_label = f"{token_chunk}→{eff_token_chunk}" if eff_token_chunk != token_chunk else f"{eff_token_chunk}"
            merge_label = f", out_merge={out_merge}" if out_merge > 1 else ""
            peak_with_merge = pred_attn_gib + ((out_merge - 1) * (g_show * int(self.attn.head_dim) * x.element_size() * s) / float(1024 ** 3))
            print(
                f"[SeamKit] 显存公式自适应 (预算 {budget_gib:.2f} GiB, S={s:,}, LoRA_R={lora_rank_sum}): "
                f"hc={hc_label} (g={g_show}), tc={tc_label}, {eff_mode}{merge_label} | "
                f"预计峰值 Attn {peak_with_merge:.2f} / MLP {pred_mlp_gib:.2f} GiB",
                flush=True,
            )

        has_qkv_extra_scales = (
            getattr(self.attn.qkv_proj, "pre_quant_scale", None) is not None
            or getattr(self.attn.qkv_proj, "input_scale", None) is not None
        )

        if not has_qkv_extra_scales:
            qkv_rot_pair = _get_linear_rot_lora(self.attn.qkv_proj, transformer_options, device, dtype)
            with _cast_linear_once(self.attn.qkv_proj, device, dtype) as (w_qkv, b_qkv):
                if _is_int8_tensorwise(w_qkv) and x.is_cuda and _ck_cuda is not None:
                    _run_chunked_attention_int8(
                        self, self.attn, x, shift_msa, scale_msa, gate_msa, mod_segments,
                        rope_freqs, w_qkv, b_qkv, n_groups, eff_token_chunk, eff_mode,
                        sol_ctx=sol_ctx, block_idx=block_idx,
                        qkv_loras=qkv_loras, out_loras=out_loras,
                        attn_temp_scale=attn_temp_scale,
                        qkv_rot_pair=qkv_rot_pair,
                        out_merge=out_merge,
                    )
                    handled = True
                elif not isinstance(w_qkv, qops.QuantizedTensor):
                    _run_chunked_attention_dense(
                        self, self.attn, x, shift_msa, scale_msa, gate_msa, mod_segments,
                        rope_freqs, w_qkv, b_qkv, n_groups, eff_token_chunk, eff_mode,
                        sol_ctx=sol_ctx, block_idx=block_idx,
                        qkv_loras=qkv_loras, out_loras=out_loras,
                        attn_temp_scale=attn_temp_scale,
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

        _run_chunked_mlp_inplace(
            self, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, eff_token_chunk,
            transformer_options=transformer_options,
            use_fused_rms=(eff_mode == "streaming"),
        )
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

    chunked_video_patch_proj_forward._orig_vpp_forward = orig_forward
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
                    default=0,
                    min=0,
                    max=56,
                    step=1,
                    tooltip=(
                        "注意力头分组数（0 = 按显存公式自适应当前显卡 6/8/10/12/16/24GB 与序列长度自动选最快安全档；"
                        "手动指定如 4/7 时，auto 模式下若超显存预算会自动升至最小安全均匀约数 1/2/4/7/14/28，按 2 头=256 ConvRot 对齐）。"
                    ),
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
                    tooltip="可选：强制保持稠密的层号（如 '0-1,48-49'；支持 '@sigma' 门限如 '48-49@0.35' 仅在低噪末步稠密消缝，15s 1080p 再省 ~60s）。",
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
                io.Float.Input(
                    "rope_hf_scale",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.02,
                    optional=True,
                    advanced=True,
                    tooltip="分频段空间 RoPE 高频解压缩倍数（1.0=原版关闭；1080p 推荐 1.32，仅放大视频高频空间频段 k=0..5 并平滑过渡，不动低频构图与音频对齐）。",
                ),
                io.Float.Input(
                    "attn_temp_scale",
                    default=1.0,
                    min=0.8,
                    max=1.3,
                    step=0.01,
                    optional=True,
                    advanced=True,
                    tooltip="长序列注意力 softmax 温度补偿倍数（1.0=原版关闭；1080p 154k token 推荐 1.08 抑制注意力熵稀释）。",
                ),
                io.Float.Input(
                    "sol_low_sigma_tau",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    advanced=True,
                    tooltip="当 sigma < sol_min_sigma 时的末步策略：0.0=退回100%稠密KJ Sage（864p/10s默认）；设为 0.90 则切 Tight-Sol（tau=0.90, aug≥128, 首尾各3层稠密），15s 1080p 末步提速 2.5×！",
                ),
                io.Combo.Input(
                    "lora_fast_mode",
                    options=["auto", "rotated_int8", "qr_collapsed", "legacy"],
                    default="auto",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "SeamKit v3.0 LoRA 旁路底层算子加速模式：\n"
                        "• auto / rotated_int8：自动执行 QR-Core 多 LoRA 正交坍缩（R≤64）+ Hadamard 旋转域直读 INT8×BF16 融合核，"
                        "qkv_proj 与 fc1 免除未旋转 BF16 激活重读（单步省 316 GB 显存带宽）；\n"
                        "• qr_collapsed：仅执行 QR-Core 正交坍缩，保留原版 BF16 激活输入；\n"
                        "• legacy：原版拼接 BF16 旁路。"
                    ),
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
        head_chunks: int = 7,
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
        rope_hf_scale: float = 1.0,
        attn_temp_scale: float = 1.0,
        sol_low_sigma_tau: float = 0.0,
        lora_fast_mode: str = "auto",
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

        head_chunks = max(0, min(int(head_chunks), int(blocks[0].attn.heads)))
        token_chunk_size = max(1024, int(token_chunk_size))
        eff_attn_mode = str(attention_mode)
        if eff_attn_mode == "sol_attn_sparse" and not _sol_attn_available:
            logger.warning(
                "MiniMaxH3ChunkedSageAttentionPatch: comfy_kitchen.sol_attn unavailable; "
                "falling back to dense_sage."
            )
            eff_attn_mode = "dense_sage"

        sol_dense_set, sol_dense_max_sigma = _parse_dense_blocks_spec(sol_dense_blocks, len(blocks))
        n_blk = len(blocks)
        sol_low_sigma_dense_set = sol_dense_set | frozenset(
            list(range(min(3, n_blk))) + list(range(max(0, n_blk - 3), n_blk))
        )
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
            "sol_dense_blocks_max_sigma": float(sol_dense_max_sigma),
            "sol_low_sigma_tau": float(sol_low_sigma_tau),
            "sol_low_sigma_dense_blocks_set": sol_low_sigma_dense_set,
            "sol_max_sigma": float(sol_max_sigma),
            "sol_min_sigma": float(sol_min_sigma),
        }

        # Offload upstream INT8-linear standard LoRA patches from `m.patches` to exact BF16
        # low-rank side-branches. Eliminates per-step 200-layer VBAR dequant/B@A/requant
        # (saves ~10-16 s/step + ~350 MiB AIMDO_CAST_BUFFER) and prevents 8-bit crushing of
        # low-strength sharpness LoRAs (e.g. minimax_h3_lms_v1.0_r64_r @ 0.15-0.20).
        static_loras: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        # If chaining ChunkedSage -> LoraLoader -> ChunkedSage on the SAME ModelPatcher lineage,
        # inherit the upstream ChunkedSage's static_loras (NEVER from a separate loaded branch!).
        prev_patched_rb = m.object_patches.get("diffusion_model._run_blocks")
        prev_rb_fn = prev_patched_rb.__func__ if isinstance(prev_patched_rb, types.MethodType) else prev_patched_rb
        if getattr(prev_rb_fn, "_is_chunked_sage_run_blocks", False):
            inherited = getattr(prev_rb_fn, "_chunked_sage_static_loras", None) or {}
            for k, v in inherited.items():
                static_loras[k] = list(v)

        patches_dict = getattr(m, "patches", None)
        popped_any = False
        if isinstance(patches_dict, dict) and patches_dict:
            for idx, block in enumerate(blocks):
                for suffix, lin in (
                    ("attn.qkv_proj", block.attn.qkv_proj),
                    ("attn.out_proj", block.attn.out_proj),
                    ("mlp.fc1", block.mlp.fc1),
                    ("mlp.fc2", block.mlp.fc2),
                ):
                    key = f"diffusion_model.blocks.{idx}.{suffix}.weight"
                    if key not in patches_dict:
                        continue
                    raw_w = getattr(lin, "weight", None)
                    if not (
                        isinstance(raw_w, qops.QuantizedTensor)
                        and getattr(lin, "layout_type", None) is not None
                        and not getattr(lin, "_full_precision_mm", False)
                    ):
                        continue
                    extracted = _extract_lora_entries_from_patches(patches_dict[key], tuple(raw_w.shape))
                    if extracted is not None:
                        patches_dict.pop(key, None)
                        popped_any = True
                        if extracted:
                            static_loras.setdefault(key, []).extend(extracted)
            if popped_any:
                m.patches_uuid = uuid.uuid4()

        eff_lora_fast_mode = str(lora_fast_mode or "auto")
        if eff_lora_fast_mode not in ("auto", "rotated_int8", "qr_collapsed", "legacy"):
            eff_lora_fast_mode = "auto"
        static_rot_loras: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if eff_lora_fast_mode != "legacy" and static_loras:
            qr_dev: torch.device = torch.device("cpu")
            if torch.cuda.is_available():
                try:
                    free_b, _ = torch.cuda.mem_get_info()
                    if free_b >= int(2.5 * (1024 ** 3)):
                        qr_dev = torch.device("cuda")
                except Exception:
                    qr_dev = torch.device("cpu")
            for idx, block in enumerate(blocks):
                for suffix, lin in (
                    ("attn.qkv_proj", block.attn.qkv_proj),
                    ("attn.out_proj", block.attn.out_proj),
                    ("mlp.fc1", block.mlp.fc1),
                    ("mlp.fc2", block.mlp.fc2),
                ):
                    key = f"diffusion_model.blocks.{idx}.{suffix}.weight"
                    if key not in static_loras or not static_loras[key]:
                        continue
                    raw_w = getattr(lin, "weight", None)
                    allow_rot256 = (
                        suffix in ("attn.qkv_proj", "mlp.fc1")
                        and _is_int8_tensorwise(raw_w)
                        and bool(getattr(raw_w._params, "convrot", False))
                        and int(getattr(raw_w._params, "convrot_groupsize", 256)) == 256
                    )
                    opt_ents, rot_pair = _optimize_lora_entries_for_mode(
                        static_loras[key],
                        eff_lora_fast_mode,
                        allow_rot256=allow_rot256,
                        compute_device=qr_dev,
                    )
                    static_loras[key] = opt_ents
                    if rot_pair is not None:
                        static_rot_loras[key] = rot_pair
            if qr_dev.type == "cuda":
                torch.cuda.empty_cache()

        attn_fwd = _make_chunked_attn_forward(head_chunks, token_chunk_size, str(out_proj_mode))
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

        orig_rb = prev_patched_rb if prev_patched_rb is not None else diffusion_model._run_blocks
        orig_rb_fn = (
            orig_rb.__func__
            if (isinstance(orig_rb, types.MethodType) and orig_rb.__self__ is diffusion_model)
            else type(diffusion_model)._run_blocks
        )
        # Never double-wrap our own `_run_blocks` (e.g. when another branch is still patched in VRAM)
        while getattr(orig_rb_fn, "_is_chunked_sage_run_blocks", False):
            orig_rb_fn = getattr(orig_rb_fn, "_orig_run_blocks_fn", type(diffusion_model)._run_blocks)
        rb_wrapped = _make_chunked_run_blocks(
            orig_rb_fn,
            sol_cfg if eff_attn_mode == "sol_attn_sparse" else None,
            static_loras=static_loras,
            rope_hf_scale=float(rope_hf_scale),
            attn_temp_scale=float(attn_temp_scale),
            lora_fast_mode=eff_lora_fast_mode,
            static_rot_loras=static_rot_loras,
        )
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
                orig_vpp_fwd = vpp.forward
                while hasattr(orig_vpp_fwd, "_orig_vpp_forward"):
                    orig_vpp_fwd = orig_vpp_fwd._orig_vpp_forward
                vpp_fwd = _make_chunked_video_patch_proj_forward(
                    orig_vpp_fwd, token_chunk_size, target_dtype
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
            "lora_fast_mode": eff_lora_fast_mode,
            "lora_side_branch_layers": len(static_loras),
            "lora_rot_int8_layers": len(static_rot_loras),
            "lora_rot_backend": get_rot_lora_active_backend() if static_rot_loras else "none",
            "arch": _cuda_archs[0] if _cuda_archs else "unknown",
            "sageplus_sm89": _sageplus_sm89_available,
            "sol_attn_available": _sol_attn_available,
        }

        _rope_str = ""
        if float(rope_hf_scale) != 1.0 or float(attn_temp_scale) != 1.0:
            _rope_str = f" | rope_hf={float(rope_hf_scale):.2f} temp={float(attn_temp_scale):.2f}"
        _low_str = f", low_tau={float(sol_low_sigma_tau):.2f}" if float(sol_low_sigma_tau) > 0.0 else ""
        print(
            f"[SeamKit] ChunkedSage: {len(blocks)} blocks | hc={head_chunks} "
            f"out_proj={out_proj_mode} attn={eff_attn_mode}"
            + (
                f" (tau={float(sol_tau):.2f}, aug={sol_token_aug}, sigma=[{float(sol_min_sigma):.2f},{float(sol_max_sigma):.2f}]{_low_str})"
                if eff_attn_mode == "sol_attn_sparse"
                else ""
            )
            + f" | bf16_lora={len(static_loras)}/200{_rope_str}",
            flush=True,
        )
        logger.info(
            "Applied MiniMaxH3ChunkedSageAttentionPatch (%d blocks, head_chunks=%d, "
            "out_proj_mode=%s, attn=%s, tau=%.2f, aug=%d, sink=%s, token_chunk=%d, "
            "chunk_io=%s, lora_side_branch=%d, arch=%s, sageplus=%s)",
            len(blocks),
            head_chunks,
            out_proj_mode,
            eff_attn_mode,
            float(sol_tau),
            sol_token_aug,
            sol_sink_mode,
            token_chunk_size,
            chunk_io_layers,
            len(static_loras),
            _cuda_archs[0] if _cuda_archs else "unknown",
            _sageplus_sm89_available,
        )
        return io.NodeOutput(m)
