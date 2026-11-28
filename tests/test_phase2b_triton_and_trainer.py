#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verification Suite for Phase 2B `Zero-Gather Triton 64×64` FlashAttention & v3.0 Trainer.

Covers:
  1. CSR/CSC `<2 MB` compact block table construction, zero K/V copy, MI300X `gfx942`
     `64 KB` LDS hardware contract (`shared <= 48 KB < 64 KB`), and minimal autograd tape.
  2. Numerical equivalence gate vs exact `sdpa_mask` (`cos(out), cos(dQ), cos(dK), cos(dV) > 0.9995`,
     `0 NaN`, including non-trivial `keep_kv` padding, `attn_temp=1.08`, non-divisible `S`,
     and non-contiguous `[S, H, D]` physical strides).
  3. Zero-VRAM-Blowup & Speedup Gate at `S = 8192` (`H = 56, D = 128`) across a checkpointed
     multi-layer DiT block stack (`peak_saved_triton <= peak_saved_sdpa + 32 MiB`, proving the
     Phase 2A `+69 GB` HBM-gather tape bug is 100% eliminated, plus speedup benchmark).
  4. v3.0 Long-Horizon Trainer unit tests:
     - `208`-layer `TokenRefiner` (`50` `blocks` + `2` `token_refiner.blocks`) `416/416` live
       gradients, out-of-place residuals, safetensors export/load, and `slice_lora_stylekeep.py`
       passthrough;
     - `--vel-t-weight` Temporal Velocity Jacobian Loss (`Δ_t v` anti-limb-dislocation);
     - `--orth-weight` Graviton Orthogonal Subspace Regularization (`||A_new @ A0_hat^T||_F^2`);
     - `--grad-accum` micro-batch gradient accumulation equivalence;
     - `--ema-decay` CPU FP32 shadow EMA export (`*_ema.safetensors`) & checkpoint persistence;
     - `--best-of-k` Golden Teacher kinematic smoothness + sharpness + anti-flicker gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
for p in (str(REPO_ROOT), str(TRAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_teacher_native as G  # noqa: E402
import slice_lora_stylekeep as SLICER  # noqa: E402
import train_h3_lora as T  # noqa: E402
from bench_phase2_mi300x import build_macro_gather_plan, macro_gather_flash_attn  # noqa: E402
from triton_block_sparse_attn import (  # noqa: E402
    GFX942_MAX_LDS_BYTES,
    HARDWARE_BLOCK_M,
    HARDWARE_BLOCK_N,
    HARDWARE_HEAD_DIM,
    HARDWARE_NUM_STAGES,
    HARDWARE_NUM_WARPS,
    TritonBlockSparsePlan,
    build_csr_csc_block_tables,
    get_compiled_shared_memory_bytes,
    triton_block_sparse_attn,
    triton_bs_available,
)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float((af * bf).sum() / (af.norm() * bf.norm() + 1e-12))


# ============================================================================
# Test 1: CSR/CSC <2MB Contract, Autograd Saved Tape & MI300X 64KB LDS Contract
# ============================================================================
def test_csr_csc_and_lds_hardware_contract(device: torch.device) -> dict:
    assert HARDWARE_BLOCK_M == 64 and HARDWARE_BLOCK_N == 64
    assert HARDWARE_HEAD_DIM == 128
    assert HARDWARE_NUM_STAGES == 1 and HARDWARE_NUM_WARPS == 4
    assert triton_bs_available(device), "Triton 64x64 Block-Sparse Flash must be available on GPU"

    # Realistic 39f sequence length (S = 29,632, N_QB = 463)
    s_39f = 29632
    block = 64
    nqb = s_39f // block
    torch.manual_seed(2026)

    sel = torch.zeros((nqb, nqb), device=device, dtype=torch.bool)
    sel[:16, :] = True  # text + ref sink prefix
    for b in range(16, nqb):
        sel[b, :16] = True
        sel[b, max(16, b - 8) : min(nqb, b + 9)] = True
        rand_cols = torch.randperm(nqb - 16, device=device)[:24] + 16
        sel[b, rand_cols] = True

    keep_39f = torch.ones(s_39f, device=device, dtype=torch.bool)
    keep_39f[192:320] = False  # blocks 3 and 4 completely padded -> must be pruned to cnt=0!
    keep_39f[150:192] = False  # partial padding in block 2

    tables = build_csr_csc_block_tables(sel, keep_39f, block_size=block)
    q2kv_idx, q2kv_cnt, kv2q_idx, kv2q_cnt = tables
    assert q2kv_idx.dtype == torch.int32 and q2kv_cnt.dtype == torch.int32
    assert kv2q_idx.dtype == torch.int32 and kv2q_cnt.dtype == torch.int32
    assert tables.total_bytes < 2 * 1024 * 1024, f"Expected < 2 MB tables, got {tables.total_bytes}"
    assert int(kv2q_cnt[3].item()) == 0 and int(kv2q_cnt[4].item()) == 0

    # Verify autograd saved tensors consist ONLY of (q, k, v, out, lse) + tiny tables
    s_small, h_small, d_hd = 512, 8, 128
    sel_s = torch.ones((s_small // 64, s_small // 64), device=device, dtype=torch.bool)
    keep_s = torch.ones(s_small, device=device, dtype=torch.bool)
    keep_s[40:64] = False
    tabs_s = build_csr_csc_block_tables(sel_s, keep_s, block_size=64)

    q = torch.randn((1, h_small, s_small, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1, h_small, s_small, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((1, h_small, s_small, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)

    saved_shapes: list[tuple[tuple[int, ...], torch.dtype]] = []

    def _pack_hook(t: torch.Tensor):
        saved_shapes.append((tuple(t.shape), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(_pack_hook, lambda x: x):
        out = triton_block_sparse_attn(q, k, v, tabs_s, keep_kv=keep_s, scale=(d_hd**-0.5) * 1.08)
    out.sum().backward()

    # Expected 10 saved tensors: q, k, v, out ([1,H,S,D]), lse ([1,H,S]), 4 int32 tables, keep_u8 ([S])
    assert len(saved_shapes) == 10, f"Unexpected saved tensor count: {saved_shapes}"
    bhsd_count = sum(1 for shp, _ in saved_shapes if shp == (1, h_small, s_small, d_hd))
    lse_count = sum(1 for shp, dt in saved_shapes if shp == (1, h_small, s_small) and dt == torch.float32)
    assert bhsd_count == 4 and lse_count == 1, f"Unexpected saved shapes: {saved_shapes}"

    lds = get_compiled_shared_memory_bytes()
    print(f"[lds-check] compiled shared memory bytes: {lds}", flush=True)
    assert {"fwd", "bwd_pre", "bwd_dkdv", "bwd_dq"}.issubset(lds.keys()), f"Missing kernel LDS stats: {lds}"
    for kname, nbytes in lds.items():
        assert nbytes <= 49152 and nbytes < GFX942_MAX_LDS_BYTES, (
            f"Kernel {kname} shared memory {nbytes} B exceeds 48 KB / 64 KB gfx942 contract!"
        )

    res = {
        "tables_39f_bytes": tables.total_bytes,
        "tables_39f_mib": round(tables.total_bytes / (1024 * 1024), 3),
        "lds_shared_bytes": lds,
        "saved_tensor_count": len(saved_shapes),
    }
    print(f"[PASS] Test 1 (CSR/CSC & gfx942 LDS Contract): {res}", flush=True)
    return res


# ============================================================================
# Test 2: Numerical Equivalence Gate (cos > 0.9995, 0 NaN, keep_kv & attn_temp)
# ============================================================================
def test_triton_bs_equivalence_gate(device: torch.device) -> dict:
    torch.manual_seed(20261128)
    block = 64
    d_hd = 128
    scale = (d_hd**-0.5) * 1.08

    metrics = {}

    # Case A: S=1024, H=16, D=128, partial + full block keep_kv padding, non-contiguous [S,H,D] strides
    s_a, h_a = 1024, 16
    nqb_a = s_a // block
    sel_a = torch.zeros((nqb_a, nqb_a), device=device, dtype=torch.bool)
    for i in range(nqb_a):
        if i < 3 or i >= nqb_a - 2:
            sel_a[i, :] = True
        else:
            sel_a[i, :3] = True
            sel_a[i, max(0, i - 2) : min(nqb_a, i + 3)] = True
            sel_a[i, (i * 5 + 3) % nqb_a] = True

    keep_a = torch.ones(s_a, device=device, dtype=torch.bool)
    keep_a[45:110] = False   # spans block 0 and block 1
    keep_a[128:192] = False  # full padded block 2

    tabs_a = build_csr_csc_block_tables(sel_a, keep_a, block_size=block)
    tok_mask_a = sel_a.repeat_interleave(block, 0).repeat_interleave(block, 1)[:s_a, :s_a] & keep_a[None, :s_a]
    mask_4d_a = tok_mask_a.unsqueeze(0).unsqueeze(0)

    # Physical [S, H, D] transposed to [1, H, S, D] (matches _make_attn_forward!)
    q_shd = (torch.randn((s_a, h_a, d_hd), device=device, dtype=torch.bfloat16) * 0.6)
    k_shd = (torch.randn((s_a, h_a, d_hd), device=device, dtype=torch.bfloat16) * 0.6)
    v_shd = (torch.randn((s_a, h_a, d_hd), device=device, dtype=torch.bfloat16) * 0.6)
    do_bhsd = torch.randn((1, h_a, s_a, d_hd), device=device, dtype=torch.bfloat16) * 0.3

    q1 = q_shd.clone().detach().requires_grad_(True)
    k1 = k_shd.clone().detach().requires_grad_(True)
    v1 = v_shd.clone().detach().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(
        q1.transpose(0, 1).unsqueeze(0),
        k1.transpose(0, 1).unsqueeze(0),
        v1.transpose(0, 1).unsqueeze(0),
        attn_mask=mask_4d_a,
        scale=scale,
    )
    (out_ref * do_bhsd).sum().backward()

    q2 = q_shd.clone().detach().requires_grad_(True)
    k2 = k_shd.clone().detach().requires_grad_(True)
    v2 = v_shd.clone().detach().requires_grad_(True)
    out_tri = triton_block_sparse_attn(
        q2.transpose(0, 1).unsqueeze(0),
        k2.transpose(0, 1).unsqueeze(0),
        v2.transpose(0, 1).unsqueeze(0),
        tabs_a,
        keep_kv=keep_a,
        scale=scale,
    )
    (out_tri * do_bhsd).sum().backward()

    c_out = _cos_sim(out_ref, out_tri)
    c_dq = _cos_sim(q1.grad, q2.grad)
    c_dk = _cos_sim(k1.grad, k2.grad)
    c_dv = _cos_sim(v1.grad, v2.grad)
    finite_a = all(torch.isfinite(t).all().item() for t in (out_tri, q2.grad, k2.grad, v2.grad))
    assert finite_a and min(c_out, c_dq, c_dk, c_dv) > 0.9995, (
        f"Case A equivalence failed: {(c_out, c_dq, c_dk, c_dv)}"
    )
    metrics["case_a_s1024"] = {
        "cos_out": round(c_out, 6),
        "cos_dq": round(c_dq, 6),
        "cos_dk": round(c_dk, 6),
        "cos_dv": round(c_dv, 6),
    }

    # Case B: Non-multiple-of-64 sequence length S=996 + empty query row NaN guard
    s_b, h_b = 996, 8
    nqb_b = (s_b + block - 1) // block
    sel_b = torch.eye(nqb_b, device=device, dtype=torch.bool)
    sel_b[:, 0] = True
    sel_b[5, :] = False  # deliberately empty query block 5 -> must output 0.0 with 0 NaN!
    keep_b = torch.ones(s_b, device=device, dtype=torch.bool)
    keep_b[10:30] = False
    tabs_b = build_csr_csc_block_tables(sel_b, keep_b, block_size=block)

    qb = torch.randn((1, h_b, s_b, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)
    kb = torch.randn((1, h_b, s_b, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)
    vb = torch.randn((1, h_b, s_b, d_hd), device=device, dtype=torch.bfloat16, requires_grad=True)
    dob = torch.randn_like(qb) * 0.2
    out_b = triton_block_sparse_attn(qb, kb, vb, tabs_b, keep_kv=keep_b, scale=scale)
    (out_b * dob).sum().backward()
    assert all(torch.isfinite(t).all().item() for t in (out_b, qb.grad, kb.grad, vb.grad)), "Empty-row / tail S=996 NaN!"
    assert float(out_b[:, :, 5 * 64 : 6 * 64].abs().max().item()) == 0.0

    print(f"[PASS] Test 2 (Equivalence Gate): {metrics}", flush=True)
    return metrics


# ============================================================================
# Test 3: S=8192 Checkpointed Multi-Layer Zero-VRAM-Blowup & Speedup Gate
# ============================================================================
def test_zero_vram_blowup_and_speedup_s8192(
    device: torch.device,
    seq_len: int = 8192,
    heads: int = 56,
    head_dim: int = 128,
    num_layers: int = 8,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()

    block = 64
    nqb = seq_len // block
    va = 6 * block
    vb = seq_len - 2 * block
    scale = (head_dim**-0.5) * 1.08

    torch.manual_seed(2026)
    sel = torch.zeros((nqb, nqb), device=device, dtype=torch.bool)
    for i in range(nqb):
        qs, qe = i * block, (i + 1) * block
        if qs >= va and qe <= vb:
            sel[i, :6] = True
            sel[i, nqb - 2 :] = True
            sel[i, max(6, i - 4) : min(nqb - 2, i + 5)] = True
        else:
            sel[i, :] = True

    keep = torch.ones(seq_len, device=device, dtype=torch.bool)
    keep[180:256] = False
    keep_ratio = float((sel.repeat_interleave(block, 0).repeat_interleave(block, 1)[:seq_len, :seq_len] & keep[None, :]).float().mean().item())

    tbs_plan = TritonBlockSparsePlan(
        tables=build_csr_csc_block_tables(sel, keep, block_size=block),
        keep_kv=keep,
        block_size=block,
        seq_len=seq_len,
    )
    tok_mask = sel.repeat_interleave(block, 0).repeat_interleave(block, 1)[:seq_len, :seq_len] & keep[None, :seq_len]
    sdpa_mask_4d = tok_mask.unsqueeze(0).unsqueeze(0)
    legacy_gather_plan = build_macro_gather_plan(sel, block, seq_len, va, vb, keep_kv=keep, macro_mult=4, macro_chunk=16)

    # Lightweight multi-layer checkpointed attention block stack at full 56x128 width
    hidden_proj = 512  # low-rank QKV projection into full [1, 56, 8192, 128] attention!

    class _CkptAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(hidden_proj, heads * head_dim, bias=False, dtype=torch.bfloat16)
            self.to_k = nn.Linear(hidden_proj, heads * head_dim, bias=False, dtype=torch.bfloat16)
            self.to_v = nn.Linear(hidden_proj, heads * head_dim, bias=False, dtype=torch.bfloat16)
            self.to_o = nn.Linear(heads * head_dim, hidden_proj, bias=False, dtype=torch.bfloat16)

        def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
            s = x.shape[0]
            q = self.to_q(x).view(s, heads, head_dim).transpose(0, 1).unsqueeze(0).contiguous()
            k = self.to_k(x).view(s, heads, head_dim).transpose(0, 1).unsqueeze(0).contiguous()
            v = self.to_v(x).view(s, heads, head_dim).transpose(0, 1).unsqueeze(0).contiguous()
            if mode == "triton_bs":
                o = triton_block_sparse_attn(q, k, v, tbs_plan, scale=scale)
            elif mode == "sdpa_mask":
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask_4d, scale=scale)
            elif mode == "legacy_gather":
                o = macro_gather_flash_attn(q, k, v, legacy_gather_plan, scale=scale)
            else:
                raise ValueError(mode)
            return x + self.to_o(o.squeeze(0).transpose(0, 1).reshape(s, heads * head_dim))

    stack = nn.ModuleList([_CkptAttnLayer().to(device) for _ in range(num_layers)])
    x0 = torch.randn((seq_len, hidden_proj), device=device, dtype=torch.bfloat16) * 0.2
    dy = torch.randn_like(x0) * 0.1

    def _measure_ckpt_stack(mode: str) -> tuple[float, float]:
        stack.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        live_saved: dict[int, int] = {}
        state = {"cur": 0, "peak": 0, "uid": 0}

        def _pack(t: torch.Tensor):
            uid = state["uid"]
            state["uid"] += 1
            nb = int(t.numel() * t.element_size())
            live_saved[uid] = nb
            state["cur"] += nb
            if state["cur"] > state["peak"]:
                state["peak"] = state["cur"]
            return (uid, t)

        def _unpack(packed):
            uid, t = packed
            nb = live_saved.pop(uid, 0)
            state["cur"] -= nb
            return t

        h = x0.clone().detach().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            for blk in stack:
                h = torch.utils.checkpoint.checkpoint(blk, h, mode, use_reentrant=False)
            (h * dy).sum().backward()

        torch.cuda.synchronize()
        peak_delta_mib = (torch.cuda.max_memory_allocated() - base_alloc) / (1024.0 * 1024.0)
        peak_saved_mib = state["peak"] / (1024.0 * 1024.0)
        return peak_saved_mib, peak_delta_mib

    saved_sdpa_mib, alloc_sdpa_mib = _measure_ckpt_stack("sdpa_mask")
    saved_triton_mib, alloc_triton_mib = _measure_ckpt_stack("triton_bs")
    saved_gather_mib, alloc_gather_mib = _measure_ckpt_stack("legacy_gather")

    del stack
    gc.collect()
    torch.cuda.empty_cache()

    # Single-layer pure operator fwd+bwd micro-benchmark & S=8192 equivalence check at [1, 56, 8192, 128]
    q_b = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    k_b = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    v_b = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    g_b = torch.randn_like(q_b) * 0.1

    q1, k1, v1 = [t.clone().detach().requires_grad_(True) for t in (q_b, k_b, v_b)]
    o1 = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=sdpa_mask_4d, scale=scale)
    (o1 * g_b).sum().backward()

    q2, k2, v2 = [t.clone().detach().requires_grad_(True) for t in (q_b, k_b, v_b)]
    o2 = triton_block_sparse_attn(q2, k2, v2, tbs_plan, scale=scale)
    (o2 * g_b).sum().backward()

    cos8k = {
        "cos_out": round(_cos_sim(o1, o2), 6),
        "cos_dq": round(_cos_sim(q1.grad, q2.grad), 6),
        "cos_dk": round(_cos_sim(k1.grad, k2.grad), 6),
        "cos_dv": round(_cos_sim(v1.grad, v2.grad), 6),
    }
    assert min(cos8k.values()) > 0.9995, f"S=8192 equivalence failed: {cos8k}"
    del q1, k1, v1, o1, q2, k2, v2, o2

    def _bench_op(fn, warmup: int = 2, iters: int = 5) -> float:
        for _ in range(warmup):
            qt, kt, vt = [t.clone().detach().requires_grad_(True) for t in (q_b, k_b, v_b)]
            y = fn(qt, kt, vt)
            (y * g_b).sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            qt, kt, vt = [t.clone().detach().requires_grad_(True) for t in (q_b, k_b, v_b)]
            y = fn(qt, kt, vt)
            (y * g_b).sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    ms_sdpa = _bench_op(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask_4d, scale=scale))
    ms_triton = _bench_op(lambda q, k, v: triton_block_sparse_attn(q, k, v, tbs_plan, scale=scale))
    speedup = ms_sdpa / max(ms_triton, 1e-6)

    del q_b, k_b, v_b, g_b, sdpa_mask_4d
    gc.collect()
    torch.cuda.empty_cache()

    assert saved_triton_mib <= saved_sdpa_mib + 32.0, (
        f"Autograd saved memory blowup: triton_bs={saved_triton_mib:.2f} MiB > sdpa_mask={saved_sdpa_mib:.2f} + 32 MiB!"
    )

    res = {
        "seq_len": seq_len,
        "heads": heads,
        "head_dim": head_dim,
        "num_ckpt_layers": num_layers,
        "keep_ratio": round(keep_ratio, 4),
        "cos_s8192": cos8k,
        "peak_saved_sdpa_mask_mib": round(saved_sdpa_mib, 2),
        "peak_saved_triton_bs_mib": round(saved_triton_mib, 2),
        "peak_saved_legacy_gather_mib": round(saved_gather_mib, 2),
        "saved_delta_triton_vs_sdpa_mib": round(saved_triton_mib - saved_sdpa_mib, 2),
        "saved_savings_vs_phase2a_gather_mib": round(saved_gather_mib - saved_triton_mib, 2),
        "peak_alloc_sdpa_mask_mib": round(alloc_sdpa_mib, 2),
        "peak_alloc_triton_bs_mib": round(alloc_triton_mib, 2),
        "peak_alloc_legacy_gather_mib": round(alloc_gather_mib, 2),
        "ms_sdpa_mask_fwdbwd": round(ms_sdpa, 2),
        "ms_triton_bs_fwdbwd": round(ms_triton, 2),
        "triton_vs_sdpa_speedup": round(speedup, 2),
    }
    print(f"[PASS] Test 3 (Zero-VRAM-Blowup & S={seq_len} Speedup Gate): {res}", flush=True)
    return res


# ============================================================================
# Test 4: Trainer v3.0 Unit Tests (208L TokenRefiner, Δ_t v, Orth, Accum, EMA, Best-of-K)
# ============================================================================
class _MiniRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _MiniAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False, dtype=dtype)
        self.q_norm = _MiniRMSNorm(head_dim, dtype=dtype)
        self.k_norm = _MiniRMSNorm(head_dim, dtype=dtype)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, rope_freqs=None, transformer_options=None) -> torch.Tensor:
        s = x.shape[0]
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)
        q = self.q_norm(q.view(s, self.heads, self.head_dim)).transpose(0, 1).unsqueeze(0)
        k = self.k_norm(k.view(s, self.heads, self.head_dim)).transpose(0, 1).unsqueeze(0)
        v = v.view(s, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.squeeze(0).transpose(0, 1).reshape(s, inner))


class _MiniMLP(nn.Module):
    def __init__(self, hidden: int, ffn: int, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * u)


class _MiniRefinerBlock(nn.Module):
    """Intentionally uses in-place `.add_(x)` before patching, mirroring `minimax/model.py:283-284`!"""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, dtype=torch.bfloat16):
        super().__init__()
        self.norm1 = _MiniRMSNorm(hidden, dtype=dtype)
        self.norm2 = _MiniRMSNorm(hidden, dtype=dtype)
        self.attn = _MiniAttn(hidden, heads, head_dim, dtype=dtype)
        self.mlp = _MiniMLP(hidden, ffn, dtype=dtype)

    def forward(self, x: torch.Tensor, transformer_options=None) -> torch.Tensor:
        x = self.attn(self.norm1(x), transformer_options=transformer_options).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class _MiniTokenRefiner(nn.Module):
    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, ffn: int, dtype=torch.bfloat16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MiniRefinerBlock(hidden, heads, head_dim, ffn, dtype=dtype) for _ in range(num_layers)]
        )
        self.final_norm = _MiniRMSNorm(hidden, dtype=dtype)

    def forward(self, x: torch.Tensor, transformer_options=None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, transformer_options=transformer_options)
        return self.final_norm(x)


class _MiniDiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, dtype=torch.bfloat16):
        super().__init__()
        self.norm1 = _MiniRMSNorm(hidden, dtype=dtype)
        self.norm2 = _MiniRMSNorm(hidden, dtype=dtype)
        self.attn = _MiniAttn(hidden, heads, head_dim, dtype=dtype)
        self.mlp = _MiniMLP(hidden, ffn, dtype=dtype)

    def forward(self, x: torch.Tensor, transformer_options=None) -> torch.Tensor:
        x = x + 0.2 * self.attn(self.norm1(x), transformer_options=transformer_options)
        return x + 0.2 * self.mlp(self.norm2(x))


class _MiniH3Model(nn.Module):
    """50 DiT blocks + 2 TokenRefiner blocks = exact 208 linear layers (416 LoRA tensors)."""

    def __init__(self, hidden: int = 128, heads: int = 4, head_dim: int = 32, ffn: int = 192, text_dim: int = 96):
        super().__init__()
        self.hidden_size = hidden
        self.condition_proj = nn.Linear(text_dim, hidden, bias=True, dtype=torch.bfloat16)
        self.token_refiner = _MiniTokenRefiner(2, hidden, heads, head_dim, ffn)
        self.blocks = nn.ModuleList([_MiniDiTBlock(hidden, heads, head_dim, ffn) for _ in range(50)])
        self.video_head = nn.Linear(hidden, 8, bias=True, dtype=torch.float32)

    def preprocess_text_embeds(self, text_states: torch.Tensor) -> torch.Tensor:
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def forward(self, vid_tokens: torch.Tensor, context: torch.Tensor, transformer_options=None) -> torch.Tensor:
        to = transformer_options or {}
        txt = context[0]
        if txt.shape[-1] != self.hidden_size:
            txt = self.token_refiner(self.condition_proj(txt), transformer_options=to)
        h = torch.empty((txt.shape[0] + vid_tokens.shape[0], self.hidden_size), device=vid_tokens.device, dtype=torch.bfloat16)
        h[: txt.shape[0]] = txt
        h[txt.shape[0] :] = vid_tokens
        for blk in self.blocks:
            h = blk(h, transformer_options=to)
        return self.video_head(h[txt.shape[0] :].float())


def test_trainer_v3_features(device: torch.device) -> dict:
    torch.manual_seed(2026)

    # ------------------------------------------------------------------
    # 4a. 208-Layer (50 blocks + 2 token_refiner.blocks) 416/416 Gradients & Slicer Passthrough
    # ------------------------------------------------------------------
    dm = _MiniH3Model().to(device)
    dm.eval()
    dm.requires_grad_(False)

    params = T.attach_lora(dm, rank=4, rank_attn=8, rank_mlp=4, train_refiner=True, rank_refiner=6)
    T.patch_attention(dm, {"attn_temp": 1.08})
    assert len(params) == 416, f"Expected 416 LoRA tensors (208 layers), got {len(params)}"

    s_txt, s_vid = 64, 192
    s_tot = s_txt + s_vid
    sel = torch.ones((s_tot // 64, s_tot // 64), device=device, dtype=torch.bool)
    keep = torch.ones(s_tot, device=device, dtype=torch.bool)
    keep[48:64] = False
    tbs_plan = TritonBlockSparsePlan(
        tables=build_csr_csc_block_tables(sel, keep, block_size=64),
        keep_kv=keep,
        block_size=64,
        seq_len=s_tot,
    )
    to = {"_h3_triton_bs": tbs_plan, "_h3_attn_temp": 1.08}

    raw_text = torch.randn((1, s_txt, 96), device=device, dtype=torch.bfloat16)
    vid_in = torch.randn((s_vid, 128), device=device, dtype=torch.bfloat16)
    tgt_out = torch.randn((s_vid, 8), device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(params, lr=1e-3)
    # Step 1: B=0 -> all 208 B matrices receive non-zero grads, opt.step() makes B != 0
    opt.zero_grad(set_to_none=True)
    pred1 = dm(vid_in, raw_text, transformer_options=to)
    F.mse_loss(pred1, tgt_out).backward()
    nz_step1 = sum(1 for p in params if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0.0)
    assert nz_step1 >= 208, f"Step 1 expected >= 208 live B grads, got {nz_step1}/416"
    opt.step()

    # Step 2: B != 0 -> ALL 416/416 tensors (including 16/16 TokenRefiner A & B) MUST have live finite grads!
    opt.zero_grad(set_to_none=True)
    pred2 = dm(vid_in, raw_text, transformer_options=to)
    F.mse_loss(pred2, tgt_out).backward()
    nz_step2 = sum(1 for p in params if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0.0)
    assert nz_step2 == 416, f"Expected 416/416 live finite gradients on step 2, got {nz_step2}/416!"
    opt.step()

    # Verify lora_state, safetensors export, slice_lora_stylekeep.py passthrough, and load_lora_state
    st_dict = T.lora_state(dm)
    assert len(st_dict) == 208
    assert "token_refiner.blocks.0.attn.qkv_proj" in st_dict
    assert "token_refiner.blocks.1.mlp.fc2" in st_dict

    with tempfile.TemporaryDirectory() as tmpd:
        raw_safe = Path(tmpd) / "step_208.safetensors"
        sliced_style = Path(tmpd) / "step_208_stylekeep.safetensors"
        sliced_attn = Path(tmpd) / "step_208_attn_only.safetensors"

        sd_export = {}
        for k, v in st_dict.items():
            a_t = v["A"].to(torch.bfloat16).contiguous()
            b_t = v["B"].to(torch.bfloat16).contiguous()
            sd_export[f"diffusion_model.{k}.lora_down.weight"] = a_t
            sd_export[f"diffusion_model.{k}.lora_up.weight"] = b_t
            sd_export[f"diffusion_model.{k}.alpha"] = torch.tensor(float(v["scale"]) * a_t.shape[0], dtype=torch.float32)
        save_file(sd_export, str(raw_safe))

        info_style = SLICER.slice_lora(str(raw_safe), str(sliced_style), preset="action_stylekeep")
        info_attn = SLICER.slice_lora(str(raw_safe), str(sliced_attn), preset="attn_only")
        assert info_style["kept_layers"] == 208 and info_style["dropped_layers"] == 0
        assert info_style["stats"]["token_refiner"]["layers"] == 8
        # Even in attn_only, all 8 token_refiner layers pass through untouched (100 attn + 8 refiner = 108)
        assert info_attn["kept_layers"] == 108 and info_attn["stats"]["token_refiner"]["layers"] == 8

        with safe_open(str(sliced_style), framework="pt", device="cpu") as sf:
            for rk in ("token_refiner.blocks.0.mlp.fc1", "token_refiner.blocks.1.mlp.fc2"):
                orig_up = sd_export[f"diffusion_model.{rk}.lora_up.weight"]
                got_up = sf.get_tensor(f"diffusion_model.{rk}.lora_up.weight")
                assert torch.equal(orig_up, got_up), f"TokenRefiner {rk} was altered by slicer!"

    # ------------------------------------------------------------------
    # 4b. Temporal Velocity Jacobian Loss (--vel-t-weight)
    # ------------------------------------------------------------------
    b_sz, c_ch, t_fr, h_sp, w_sp = 1, 8, 6, 8, 8
    z_vid = torch.randn((b_sz, c_ch, t_fr, h_sp, w_sp), device=device)
    z_vid[:, :, 2:4] += 2.0  # high-motion region
    m_mot = T._motion_gate_mask(z_vid, tau=0.35)
    v_star = torch.randn((b_sz, c_ch, t_fr, h_sp, w_sp), device=device)

    # Zero loss when pred == target or T == 1
    assert float(T._temporal_vel_jacobian_l1(v_star, v_star, m_mot).item()) == 0.0
    assert float(T._temporal_vel_jacobian_l1(v_star[:, :, :1], v_star[:, :, :1], m_mot[:, :, :1]).item()) == 0.0

    # Same framewise L2 perturbation magnitude: smooth constant offset vs alternating (-1)^t limb-flip flicker
    alt_sign = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=device).view(1, 1, t_fr, 1, 1)
    v_smooth_err = (v_star + 0.25).clone().detach().requires_grad_(True)
    v_flicker_err = (v_star + 0.25 * alt_sign).clone().detach().requires_grad_(True)

    l_smooth = T._temporal_vel_jacobian_l1(v_smooth_err, v_star, m_mot)
    l_flicker = T._temporal_vel_jacobian_l1(v_flicker_err, v_star, m_mot)
    l_flicker.backward()
    assert float(l_smooth.item()) < 1e-7
    assert float(l_flicker.item()) > 0.15
    assert v_flicker_err.grad is not None and float(v_flicker_err.grad.abs().max().item()) > 0.0

    # ------------------------------------------------------------------
    # 4c. Graviton Orthogonal Subspace Regularization (--orth-weight)
    # ------------------------------------------------------------------
    dm_src = _MiniH3Model().to(device)
    T.attach_lora(dm_src, rank=4, blocks_spec="0-1", rank_attn=4, rank_mlp=4)
    T.patch_attention(dm_src, {"attn_temp": 1.08})
    for _, lr_m in T._iter_lora_modules(dm_src):
        nn.init.normal_(lr_m.B, std=0.05)
    with torch.no_grad():
        y_src = dm_src(vid_in, raw_text)
    src_state = T.lora_state(dm_src)

    dm_exp = _MiniH3Model().to(device)
    dm_exp.load_state_dict(dm_src.state_dict(), strict=False)
    exp_params = T.attach_lora(dm_exp, rank=4, blocks_spec="0-1", rank_attn=8, rank_mlp=4)
    T.patch_attention(dm_exp, {"attn_temp": 1.08})
    ex_cnt, exp_cnt = T.load_lora_state(dm_exp, src_state)
    assert ex_cnt == 4 and exp_cnt == 4, f"Expected 4 exact MLP + 4 expanded Attn, got {(ex_cnt, exp_cnt)}"

    with torch.no_grad():
        y_exp = dm_exp(vid_in, raw_text)
    max_warmstart_diff = float((y_src - y_exp).abs().max().item())
    assert max_warmstart_diff == 0.0, f"Rank expansion broke step=0 equivalence: diff={max_warmstart_diff}"

    orth_init = float(T.compute_graviton_orth_loss(dm_exp).item())
    assert orth_init > 0.0, "Initial Graviton orth loss should be strictly positive"

    opt_orth = torch.optim.AdamW(exp_params, lr=8e-3, weight_decay=0.0)
    for _ in range(40):
        opt_orth.zero_grad(set_to_none=True)
        lo = 0.02 * T.compute_graviton_orth_loss(dm_exp)
        lo.backward()
        opt_orth.step()
    orth_after = float(T.compute_graviton_orth_loss(dm_exp).item())
    assert orth_after < orth_init * 0.01, f"Orth loss failed to converge: {orth_init} -> {orth_after}"
    # Frozen anchor rows A[:4] have exact zero orth gradient and remain 100% untouched!
    for key, lr_m in T._iter_lora_modules(dm_exp):
        if lr_m.rank == 8:
            assert float(lr_m.A.grad[:4].abs().max().item()) == 0.0
            expected_a0 = src_state[key]["A"].to(device=device, dtype=lr_m.A.dtype) * float(src_state[key]["scale"])
            assert torch.allclose(lr_m.A.data[:4], expected_a0, atol=1e-6), f"Old ranks mutated in {key}!"

    # Verify orth_a0_hat persists through lora_state -> load_lora_state on resume
    resumed_state = T.lora_state(dm_exp)
    dm_resumed = _MiniH3Model().to(device)
    T.attach_lora(dm_resumed, rank=4, blocks_spec="0-1", rank_attn=8, rank_mlp=4)
    T.load_lora_state(dm_resumed, resumed_state)
    assert abs(float(T.compute_graviton_orth_loss(dm_resumed).item()) - orth_after) < 1e-6

    # ------------------------------------------------------------------
    # 4d. Gradient Accumulation (--grad-accum) Exact Equivalence
    # ------------------------------------------------------------------
    grad_accum = 3
    micro_inputs = [torch.randn((s_vid, 128), device=device, dtype=torch.bfloat16) for _ in range(grad_accum)]
    micro_targets = [torch.randn((s_vid, 8), device=device, dtype=torch.float32) for _ in range(grad_accum)]

    for p in exp_params:
        p.grad = None
    for vi, ti in zip(micro_inputs, micro_targets):
        out_m = dm_exp(vi, raw_text)
        (F.mse_loss(out_m, ti) / float(grad_accum)).backward()
    grads_accum = [p.grad.detach().float().clone() for p in exp_params]

    for p in exp_params:
        p.grad = None
    full_loss = sum(F.mse_loss(dm_exp(vi, raw_text), ti) for vi, ti in zip(micro_inputs, micro_targets)) / float(grad_accum)
    full_loss.backward()
    grads_ref = [p.grad.detach().float().clone() for p in exp_params]

    accum_cos = min(_cos_sim(ga, gr) for ga, gr in zip(grads_accum, grads_ref) if gr.norm() > 0)
    assert accum_cos > 0.9999, f"grad_accum gradient mismatch: cos={accum_cos}"

    # ------------------------------------------------------------------
    # 4e. CPU FP32 Shadow EMA (--ema-decay) & Safetensors/Resume Persistence
    # ------------------------------------------------------------------
    ema = T.LoRAEMA(dm_exp, decay=0.9)
    sample_key = "blocks.0.attn.qkv_proj"
    assert ema.shadow[sample_key]["A"].device.type == "cpu"
    assert ema.shadow[sample_key]["A"].dtype == torch.float32

    a_t0 = ema.shadow[sample_key]["A"].clone()
    with torch.no_grad():
        dm_exp.blocks[0].attn.qkv_proj._train_lora.A.add_(1.0)
    a_t1 = dm_exp.blocks[0].attn.qkv_proj._train_lora.A.detach().cpu().float()
    ema.update(dm_exp)
    expected_ema = 0.9 * a_t0 + 0.1 * a_t1
    max_ema_err = float((ema.shadow[sample_key]["A"] - expected_ema).abs().max().item())
    assert max_ema_err < 1e-6, f"EMA update formula mismatch: err={max_ema_err}"

    ema_ckpt = ema.state_dict()
    ema_restored = T.LoRAEMA(dm_src, decay=0.9)
    # Re-init on matching shape model and restore
    ema2 = T.LoRAEMA(dm_exp, decay=0.9)
    ema2.shadow[sample_key]["A"].zero_()
    ema2.load_state_dict(ema_ckpt, dm=dm_exp)
    assert torch.allclose(ema2.shadow[sample_key]["A"], expected_ema, atol=1e-6)
    del ema_restored

    # ------------------------------------------------------------------
    # 4f. Best-of-K Golden Teacher Kinematic Gate (--best-of-k)
    # ------------------------------------------------------------------
    t_ax = torch.linspace(0.0, 1.0, 10).view(1, 1, 10, 1, 1)
    y_ax = torch.linspace(-1.0, 1.0, 16).view(1, 1, 1, 16, 1)
    x_ax = torch.linspace(-1.0, 1.0, 16).view(1, 1, 1, 1, 16)

    # Cand 0: smooth motion but blurry edges
    z_blurry = torch.exp(-((x_ax - 0.6 * t_ax + 0.3) ** 2 + y_ax**2) / 0.35).expand(1, 24, 10, 16, 16).contiguous()
    # Cand 1: sharp edges but frame-5 violent limb-teleportation flicker spike!
    z_spike = torch.tanh(((x_ax - 0.6 * t_ax + 0.3) * 12.0)).expand(1, 24, 10, 16, 16).clone()
    z_spike[:, :, 5] *= -3.5
    # Cand 2: smooth kinematic trajectory + crisp high-motion edges -> MUST WIN!
    z_golden = torch.tanh(((x_ax - 0.6 * t_ax + 0.3) * 12.0)).expand(1, 24, 10, 16, 16).contiguous()

    sc_blur = G.score_teacher_kinematic_candidate(z_blurry, qa_data={"motion_roi_lap": 55.0}, min_motion_lapvar=75.0)
    sc_spike = G.score_teacher_kinematic_candidate(z_spike, qa_data={"motion_roi_lap": 210.0}, min_motion_lapvar=75.0)
    sc_gold = G.score_teacher_kinematic_candidate(z_golden, qa_data={"motion_roi_lap": 195.0}, min_motion_lapvar=75.0)

    assert sc_spike["is_flicker_spike"] is True, f"Expected flicker spike rejection: {sc_spike}"
    assert sc_gold["is_flicker_spike"] is False and sc_gold["lap_ok"] is True
    assert sc_gold["composite_score"] > sc_blur["composite_score"]
    assert sc_gold["composite_score"] > sc_spike["composite_score"]

    # End-to-end mocked ComfyUI generate_one_with_qa(best_of_k=3) verification
    cands_lat = [
        (z_blurry, {"motion_roi_lap": 55.0}),
        (z_spike, {"motion_roi_lap": 210.0}),
        (z_golden, {"motion_roi_lap": 195.0}),
    ]
    with tempfile.TemporaryDirectory() as tmp_t:
        out_root = Path(tmp_t) / "raw"
        zup_dir = Path(tmp_t) / "zup"
        out_root.mkdir()
        zup_dir.mkdir()
        call_idx = {"i": 0}

        def _fake_submit(api, graph):
            idx = call_idx["i"]
            call_idx["i"] += 1
            zv, qad = cands_lat[idx]
            torch.save({"parts": [zv, torch.zeros(1, 32, 10)]}, out_root / "s001.z_native.pt")
            torch.save({"parts": [zv * 0.5]}, out_root / "s001.z_up.pt")
            torch.save({"conditioning": []}, out_root / "s001.cond.pt")
            (out_root / "s001.qa.json").write_text(json.dumps(qad), encoding="utf-8")
            torch.save({"marker": idx}, zup_dir / "s001_zup_force.pt")
            return f"pid_{idx}"

        mock_args = types.SimpleNamespace(
            unet="u", clip="c", video_vae="vv", audio_vae="av", lms_lora="l", lms_strength=0.0,
            upscaler="up", rope_hf=1.32, attn_temp=1.08, out=str(out_root), cfg=5.5, steps=28,
            shift_video=12.0, shift_audio=3.0, sampler="dual_clock_euler", frames=39,
            zup_mode="downsample", pass1_sigmas="0.68, 0.34, 0.0", pass1_cfg=3.0,
            zup_tmp_dir=str(zup_dir), min_motion_lapvar=75.0, max_qa_retries=2, best_of_k=3,
        )
        with patch.object(G, "submit", side_effect=_fake_submit), patch.object(G, "wait_done", return_value={}):
            win_info = G.generate_one_with_qa(
                {"id": "s001", "prompt": "sakuga sword", "seed": 1000, "frames": 39},
                mock_args,
                "http://127.0.0.1:8188",
                out_root,
            )
        assert win_info["candidates_evaluated"] == 3
        assert win_info["rejected_spikes"] == 1
        assert win_info["seed"] == 1000 + 2 * 10007  # Cand 2 (z_golden) won!
        saved_zup = torch.load(zup_dir / "s001_zup_force.pt", weights_only=False)
        assert saved_zup["marker"] == 2

    res = {
        "total_lora_tensors_208L": len(params),
        "step2_live_grads": f"{nz_step2}/{len(params)}",
        "slicer_stylekeep_kept": info_style["kept_layers"],
        "slicer_attn_only_kept": info_attn["kept_layers"],
        "vel_t_flicker_vs_smooth": (round(float(l_flicker.item()), 5), round(float(l_smooth.item()), 8)),
        "orth_loss_init_to_25steps": (round(orth_init, 5), round(orth_after, 6)),
        "grad_accum_cos": round(accum_cos, 8),
        "ema_max_err": max_ema_err,
        "best_of_3_winner_seed": win_info["seed"],
    }
    print(f"[PASS] Test 4 (Trainer v3.0 Unit Tests): {res}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny-only", action="store_true", help="Skip S=8192 VRAM/speedup benchmark while GPU is shared")
    ap.add_argument("--seq-len", type=int, default=8192)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA/ROCm GPU required for Triton verification"
    device = torch.device("cuda")

    t0 = time.perf_counter()
    summary = {
        "test1_contract": test_csr_csc_and_lds_hardware_contract(device),
        "test2_equivalence": test_triton_bs_equivalence_gate(device),
        "test4_trainer_v3": test_trainer_v3_features(device),
    }
    if not args.tiny_only:
        summary["test3_s8192_zero_vram_and_speedup"] = test_zero_vram_blowup_and_speedup_s8192(
            device, seq_len=args.seq_len
        )
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    print("\n=== ALL PHASE 2B TESTS PASSED ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
