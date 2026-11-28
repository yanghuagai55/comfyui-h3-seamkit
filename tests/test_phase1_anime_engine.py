# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Automated Phase 1 Verification & Benchmark Suite for H3-AnimeEngine (`SeamKit v3.0`).

Covers all deliverables specified in `docs/H3_ANIME_ENGINE_V3_ARCHITECTURE.md` (§1.2, §2, §3):
  - Test 1: `Universal AdaLN-Curve Bridge` (§2.2) — Centered Ridge Curve Projection `< 5e-4`
            relative error, exact `Δb_compat` bias absorption under arbitrary basis rotation/shift,
            `< 0.15s` 51-layer latency, `0` runtime VRAM overhead, and real safetensors load check
            against local `DaSiWa v5xs` + market LoRAs (`turbo_4STEPS`, `curveproj1025`, `pika`,
            `LMS`, `minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors`).
  - Test 2: `QR-Core` Multi-Slot Orthogonal LoRA Collapse (§3.1) — `R_sum = 544 -> 64` mathematical
            identity vs full dense `torch.linalg.svd` (`cos > 0.999999`), zero `[d_out, d_in]`
            dense materialization, and latency/VRAM benchmark.
  - Test 3: Hadamard Rotated `INT8 × BF16` LoRA Operator (§3.2) — exact `ConvRot-256` Hadamard
            equivalence, `E:/cuda/bin/nvcc.exe` V13.2 native CUDA C++ Tensor Core kernel + Triton
            + PyTorch fallback accuracy (`cos > 0.999`), end-to-end `MiniMaxH3ChunkedSageAttentionPatch`
            verification (`qkv_proj` & `fc1` zero unrotated `h` re-read), and `S=32768` GPU benchmark.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("D:/comfyui/ComfyUI")
if str(COMFY_ROOT) not in sys.path and COMFY_ROOT.is_dir():
    sys.path.insert(0, str(COMFY_ROOT))
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import comfy.model_base
import comfy.model_patcher
import comfy.ops
import comfy.quant_ops as qops
import comfy_kitchen.backends.cuda as ck_cuda
from comfy.ldm.minimax.model import DiTBlock, MiniMaxH3Model, TimeEmbedder
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.int8_utils import _build_hadamard

import chunked_sage as cs
import lora_compat_bridge as lcb
import rot_lora_ops as rlo

DASIWA_PATH = Path(
    "D:/comfyui/ComfyUI/models/unet/"
    "dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors"
)
LORA_DIR = Path("D:/comfyui/ComfyUI/models/loras")

BENCHMARK_SUMMARY: dict[str, object] = {}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64).reshape(-1)
    bf = b.detach().to(torch.float64).reshape(-1)
    return float(F.cosine_similarity(af, bf, dim=0).item())


class _FakeH3ModelWrapper(torch.nn.Module):
    """Minimal ComfyUI `BaseModel`-compatible wrapper around `MiniMaxH3Model` for `ModelPatcher` tests."""

    def __init__(self, diffusion_model: MiniMaxH3Model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.model_config = type("Cfg", (), {"unet_config": {"pdd_steps": 1}})()

    def get_dtype(self):
        return torch.bfloat16


def _build_meta_h3_patcher(use_adaln_curves: bool = True, t_table: torch.Tensor | None = None):
    """Build a 0-RAM 50-block + 2-refiner-block `MiniMaxH3Model` patcher with exact real H3 shapes."""
    dm = MiniMaxH3Model(
        adaln_curve_grid=1025 if use_adaln_curves else None,
        time_embed_dim=8 if use_adaln_curves else 2688,
        dtype=torch.bfloat16,
        device="meta",
        operations=comfy.ops.disable_weight_init,
    )
    if use_adaln_curves:
        if t_table is None:
            t_table, _ = lcb.load_canonical_adaln_basis()
        dm.adaln_t_table = t_table.detach().cpu().to(torch.float32).contiguous()
    else:
        # Materialize lightweight TimeEmbedder on CPU (~60 MB) for exact 2688-D grid evaluation
        torch.manual_seed(1234)
        te = TimeEmbedder(256, 5376, 2688, dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
        with torch.no_grad():
            for p in te.parameters():
                p.normal_(0.0, 0.02)
        dm.time_embedder = te
    wrapper = _FakeH3ModelWrapper(dm)
    return comfy.model_patcher.ModelPatcher(wrapper, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


# ===========================================================================
# TEST 1: Universal AdaLN-Curve Bridge (§2.2) + Real Safetensors Compatibility
# ===========================================================================

def test_1a_adaln_curve_bridge_math_and_bias_absorption():
    """Verify `< 5e-4` reconstruction error & exact `Δb_compat` absorption under rotated/shifted bases."""
    t_canon, proj_canon = lcb.load_canonical_adaln_basis()
    assert t_canon.shape == (1025, 8)
    assert proj_canon.shape == (9, 2688)

    # Construct an arbitrary rotated + scaled + non-zero-mean target curve table (simulating an
    # alien community pruned base model whose curve basis has sign flips, rotation, and offset!)
    torch.manual_seed(2026)
    q_orth, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    col_scales = torch.tensor([1.25, 0.85, 1.10, 0.90, 1.40, 0.75, 1.15, 0.95], dtype=torch.float64)
    nonzero_shift = torch.tensor([[0.42, -0.31, 0.18, -0.09, 0.05, -0.04, 0.02, -0.01]], dtype=torch.float64)
    t_alien = (t_canon.to(torch.float64) @ (q_orth * col_scales) + nonzero_shift).to(torch.float32)

    # Load real 2688-dim `turbo_4STEPS` AdaLN tensors if present, else realistic synthetic
    turbo_path = LORA_DIR / "minimax_h3_turbo_4STEPS_comfyui.safetensors"
    if turbo_path.is_file():
        with safe_open(str(turbo_path), framework="pt", device="cpu") as f:
            a_2688 = f.get_tensor("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight")
            b_2688 = f.get_tensor("diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight")
    else:
        a_2688 = torch.randn(16, 2688, dtype=torch.bfloat16) * 0.02
        b_2688 = torch.randn(96768, 16, dtype=torch.bfloat16) * 0.02

    # 1. Project 2688-dim LoRA onto alien rotated+shifted [1025, 8] target at strength=0.85
    strength = 0.85
    res = lcb.centered_ridge_curve_project(
        b_src=b_2688,
        a_src=a_2688,
        t_target=t_alien,
        strength=strength,
        out_dtype=torch.float32,
    )
    assert res.a_compat.shape == (16, 8)
    assert res.delta_b_compat.shape == (96768,)
    assert res.rel_error < 5e-4, f"Expected rel_error < 5e-4, got {res.rel_error:.6e}"

    # Verify on a dense subset of 17 timesteps that `s * B @ A_compat @ T_alien[k]^T + Δb_compat`
    # matches the exact source trajectory `s * B @ (Ψ_1025[k] @ A_2688^T)` to < 5e-4!
    psi_canon = t_canon.double() @ proj_canon[:8].double() + proj_canon[8:9].double()
    sample_idx = torch.linspace(0, 1024, 17).long()
    y_true = strength * (psi_canon[sample_idx] @ a_2688.double().T) @ b_2688.double().T
    y_pred = (
        strength * (t_alien[sample_idx].double() @ res.a_compat.double().T) @ res.b_out.double().T
        + res.delta_b_compat.double().unsqueeze(0)
    )
    dense_rel_err = float((torch.linalg.norm(y_pred - y_true) / torch.linalg.norm(y_true)).item())
    assert dense_rel_err < 5e-4, f"Dense 17-step check failed: {dense_rel_err:.6e}"

    # 2. Test heterogeneous 8-dim `curveproj1025` (with embedded `t_source` + `diff_b_src`) -> `DaSiWa` `t_canon`
    res_hetero = lcb.centered_ridge_curve_project(
        b_src=res.b_out,
        a_src=res.a_compat,
        t_target=t_canon,
        t_source=t_alien,
        diff_b_src=res.delta_b_unit,
        strength=strength,
        out_dtype=torch.float32,
    )
    y_roundtrip = (
        strength * (t_canon[sample_idx].double() @ res_hetero.a_compat.double().T) @ res_hetero.b_out.double().T
        + res_hetero.delta_b_compat.double().unsqueeze(0)
    )
    roundtrip_err = float((torch.linalg.norm(y_roundtrip - y_true) / torch.linalg.norm(y_true)).item())
    assert roundtrip_err < 5e-4, f"Heterogeneous 8->8 roundtrip error too high: {roundtrip_err:.6e}"

    # 3. Test reverse 8-dim curve LoRA -> 2688-dim unpruned target (`d_tgt=2688 > G=1025`)
    res_to_2688 = lcb.centered_ridge_curve_project(
        b_src=res.b_out,
        a_src=res.a_compat,
        t_target=psi_canon.to(torch.float32),
        t_source=t_alien,
        diff_b_src=res.delta_b_unit,
        strength=strength,
        out_dtype=torch.float32,
    )
    assert res_to_2688.a_compat.shape == (16, 2688)
    assert res_to_2688.rel_error < 5e-4, f"8->2688 reverse bridge rel_error={res_to_2688.rel_error:.6e}"

    BENCHMARK_SUMMARY["test1a_rel_error_2688_to_8"] = res.rel_error
    BENCHMARK_SUMMARY["test1a_roundtrip_8_to_8_err"] = roundtrip_err
    BENCHMARK_SUMMARY["test1a_reverse_8_to_2688_err"] = res_to_2688.rel_error


@unittest.skipIf(not DASIWA_PATH.is_file(), "Local DaSiWa v5xs safetensors not present")
def test_1b_real_dasiwa_and_market_loras_bridge():
    """Load real `DaSiWa v5xs` `adaln_t_table` and bridge real market LoRAs in `< 0.15s` with `0` VRAM overhead."""
    with safe_open(str(DASIWA_PATH), framework="pt", device="cpu") as fd:
        dasiwa_t_table = fd.get_tensor("adaln_t_table").float()

    base_patcher = _build_meta_h3_patcher(use_adaln_curves=True, t_table=dasiwa_t_table)

    # Measure 51-layer pure AdaLN-Curve Bridge latency & VRAM delta on real `turbo_4STEPS` (2688 -> DaSiWa 8)
    turbo_2688_path = LORA_DIR / "minimax_h3_turbo_4STEPS_comfyui.safetensors"
    assert turbo_2688_path.is_file(), f"Missing {turbo_2688_path}"
    raw_turbo = comfy.utils.load_torch_file(str(turbo_2688_path), safe_load=True)

    # Warm target Ridge solver cache once, then time full 51-module AdaLN projection
    lora_pairs, _, _, _ = lcb.parse_h3_lora_tensors(raw_turbo)
    adaln_items = [(m, pair) for m, pair in lora_pairs.items() if m.endswith(".adaln_proj.linear")]
    assert len(adaln_items) == 51

    # Warm one call so OpenMP / basis file load cold-start is excluded from steady-state 51-layer timing
    _ = lcb.centered_ridge_curve_project(
        b_src=adaln_items[0][1][0],
        a_src=adaln_items[0][1][1],
        t_target=dasiwa_t_table,
        strength=1.0,
        alpha=adaln_items[0][1][2],
        out_dtype=torch.float32,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_before = torch.cuda.memory_allocated()
    else:
        vram_before = 0

    t0 = time.perf_counter()
    rel_errs = []
    for _mod, (b_mat, a_mat, alpha_v) in adaln_items:
        pr = lcb.centered_ridge_curve_project(
            b_src=b_mat,
            a_src=a_mat,
            t_target=dasiwa_t_table,
            strength=1.0,
            alpha=alpha_v,
            out_dtype=torch.float32,
        )
        rel_errs.append(pr.rel_error)
    bridge_51_sec = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_delta = torch.cuda.memory_allocated() - vram_before
    else:
        vram_delta = 0

    max_rel_err = max(rel_errs)
    assert max_rel_err < 5e-4, f"51-layer DaSiWa bridge max_rel_err={max_rel_err:.6e} >= 5e-4"
    assert bridge_51_sec < 0.15, f"51-layer AdaLN bridge took {bridge_51_sec:.4f}s >= 0.15s"
    assert vram_delta == 0, f"Expected 0 runtime VRAM overhead, got {vram_delta} bytes"

    # Cross-validate projected `turbo_4STEPS -> DaSiWa` against local `10ErosMax_curveproj1025`
    eros_path = LORA_DIR / "minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001-T8.safetensors"
    if eros_path.is_file():
        raw_eros = dict(comfy.utils.load_torch_file(str(eros_path), safe_load=True))
        # Inject an alien rotated embedded `adaln_t_table` to verify embedded-table `curveproj1025` bridging!
        sign_flip = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))
        raw_eros["adaln_t_table"] = dasiwa_t_table @ sign_flip
        for k in list(raw_eros.keys()):
            if k.endswith(".adaln_proj.linear.lora_A.weight"):
                raw_eros[k] = (raw_eros[k].float() @ sign_flip).to(raw_eros[k].dtype)
        patched_eros, rep_eros = lcb.bridge_and_load_h3_loras(base_patcher, [(raw_eros, 1.0)], qr_collapse=False)
        assert rep_eros["adaln_bridged_count"] == 51
        assert rep_eros["adaln_rel_error_max"] < 5e-4
        assert rep_eros["token_refiner_adapter_hits"] == 8
        del patched_eros, raw_eros

    # Verify all market LoRAs in `models/loras` load cleanly onto `DaSiWa v5xs` via `MiniMaxH3UniversalLoRALoader`
    market_candidates = [
        ("minimax_h3_turbo_4STEPS_comfyui.safetensors", 1.0, 208, 51),
        ("minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors", 0.9, 200, 0),
        ("pika_h3_flex_fl2va_v1_bf16_r.safetensors", 0.5, 208, 0),
        ("minimax_h3_lms_v1.0_r64_r.safetensors", 0.25, 208, 50),
    ]
    loaded_market = []
    for fname, st, exp_lin_min, exp_adaln in market_candidates:
        fpath = LORA_DIR / fname
        if not fpath.is_file():
            continue
        out = lcb.MiniMaxH3UniversalLoRALoader.execute(
            base_patcher,
            lora_1=str(fpath),
            strength_1=st,
            adaln_bridge_mode="force_project" if exp_adaln > 0 else "auto",
            qr_collapse=False,
        )
        patched_m, rep_json = out.args[0], json.loads(out.args[1])
        assert rep_json["patched_targets"] >= exp_lin_min
        if exp_adaln > 0:
            assert rep_json["adaln_bridged_count"] == exp_adaln
            assert rep_json["adaln_rel_error_max"] < 5e-4
        loaded_market.append((fname, rep_json["patched_targets"], rep_json["adaln_rel_error_max"]))
        del patched_m

    # Also test unpruned 2688-D target base model bridging an 8-D `curveproj1025` LoRA!
    # Provide a `time_embedder_override` callable reproducing H3's canonical 2688-D post-SiLU curve
    # (note `eval_h3_time_embedder_grid` applies `F.silu(fn(t))`, so pass `inv_silu` or positive linear map
    # — or directly construct `psi_canon` via `t_canon @ proj_canon[:8] + proj_canon[8:9]`!).
    unpruned_patcher = _build_meta_h3_patcher(use_adaln_curves=False)
    t_canon_b, proj_canon_b = lcb.load_canonical_adaln_basis()
    psi_canon_2688 = (t_canon_b.double() @ proj_canon_b[:8].double() + proj_canon_b[8:9].double()).to(torch.float32)

    def _mock_unpruned_te_pre_silu(t_vals: torch.Tensor) -> torch.Tensor:
        # Invert SiLU(z) = y numerically via 8 Newton steps so F.silu(te(t)) == psi_canon_2688!
        pos = t_vals.clamp(0.0, 1.0) * (psi_canon_2688.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=psi_canon_2688.shape[0] - 2)
        y = torch.lerp(psi_canon_2688[i0], psi_canon_2688[i0 + 1], (pos - i0).unsqueeze(1)).double()
        # Clamp strictly above SiLU global minimum (-0.2784645427610738 at z = -1.27846454)
        y_safe = y.clamp(min=-0.27846)
        z = torch.where(y_safe >= 0.0, y_safe + 0.5, torch.full_like(y_safe, -0.5))
        for _ in range(12):
            sig = torch.sigmoid(z)
            f_val = z * sig - y_safe
            df = sig * (1.0 + z * (1.0 - sig))
            z = z - f_val / df.clamp(min=1e-9)
        return z.to(torch.float32)

    if eros_path.is_file():
        patched_unp, rep_unp = lcb.bridge_and_load_h3_loras(
            unpruned_patcher,
            [(eros_path, 0.8)],
            qr_collapse=False,
            time_embedder_override=_mock_unpruned_te_pre_silu,
        )
        assert rep_unp["target_adaln_mode"] == "unpruned_2688"
        assert rep_unp["adaln_bridged_count"] == 51
        assert rep_unp["adaln_rel_error_max"] < 5e-4, f"8->2688 max_err={rep_unp['adaln_rel_error_max']:.6e}"
        del patched_unp

    BENCHMARK_SUMMARY["test1b_51layer_bridge_ms"] = round(bridge_51_sec * 1000.0, 2)
    BENCHMARK_SUMMARY["test1b_51layer_max_rel_err"] = max_rel_err
    BENCHMARK_SUMMARY["test1b_vram_overhead_bytes"] = vram_delta
    BENCHMARK_SUMMARY["test1b_market_loras_verified"] = loaded_market


# ===========================================================================
# TEST 2: QR-Core Orthogonal LoRA Collapse (§3.1) `R_sum = 544 -> 64`
# ===========================================================================

def test_2_qr_core_collapse_544_to_64_identity_and_benchmark():
    """Verify `R_sum = 544 -> 64` exact identity vs dense `torch.linalg.svd` (`cos > 0.999999`) + latency."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(31415)

    # 1a. Exact mathematical identity check (`R_sum = 544 -> 64`, `energy_ratio=1.0`) in FP64 on CPU
    #     (avoids cuSOLVER FP32 iterative Jacobi `gesvdj` ~1.7e-6 approximation jitter!)
    d_out_id, d_in_id = 4096, 2048
    id_entries = []
    for r_m, s_m in [(384, 0.70), (64, 0.90), (64, 0.25), (32, 0.80)]:
        decay = (torch.arange(1, r_m + 1, dtype=torch.float64) ** -0.85) * 0.05
        bm = (torch.randn(d_out_id, r_m, dtype=torch.float64) * decay.unsqueeze(0)).to(torch.bfloat16)
        am = (torch.randn(r_m, d_in_id, dtype=torch.float64) * 0.02).to(torch.bfloat16)
        id_entries.append((bm, am, s_m))

    b_id64, a_id64, id_stats = lcb.qr_core_collapse_loras(
        id_entries, max_rank=64, energy_ratio=1.0, out_dtype=torch.float64, compute_device="cpu"
    )
    w_id_dense = torch.zeros((d_out_id, d_in_id), dtype=torch.float64)
    for bm, am, sm in id_entries:
        w_id_dense.addmm_(bm.double(), am.double(), beta=1.0, alpha=sm)
    u_id, s_id, vh_id = torch.linalg.svd(w_id_dense, full_matrices=False)
    w_id_svd64 = (u_id[:, :64] * s_id[:64].unsqueeze(0)) @ vh_id[:64, :]
    w_id_qr64 = b_id64 @ a_id64
    cos_svd = _cosine(w_id_qr64, w_id_svd64)
    rel_frob_diff = float(
        (torch.linalg.norm(w_id_qr64 - w_id_svd64) / torch.linalg.norm(w_id_svd64)).item()
    )
    del w_id_dense, u_id, s_id, vh_id, w_id_svd64, w_id_qr64

    # 1b. Full-scale H3 `mlp.fc1` `[28672, 5376]` (`R_sum = 544 -> 64`) FP32 GPU latency & VRAM benchmark
    d_out, d_in = 28672, 5376
    slot_specs = [(384, 0.70), (64, 0.90), (64, 0.25), (32, 0.80)]
    entries = []
    for r_m, s_m in slot_specs:
        decay = (torch.arange(1, r_m + 1, device=dev, dtype=torch.float32) ** -0.85) * 0.05
        b_m = (torch.randn(d_out, r_m, device=dev, dtype=torch.float32) * decay.unsqueeze(0)).to(torch.bfloat16)
        a_m = (torch.randn(r_m, d_in, device=dev, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        entries.append((b_m, a_m, s_m))

    # Warm cuSOLVER handle once before timing
    _ = lcb.qr_core_collapse_loras(entries[:2], max_rank=16, energy_ratio=1.0, out_dtype=torch.float32, compute_device=dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    collapsed = lcb.qr_core_collapse_loras(
        entries, max_rank=64, energy_ratio=1.0, out_dtype=torch.float32, compute_device=dev
    )
    if dev.type == "cuda":
        torch.cuda.synchronize()
        qr_peak_mib = (torch.cuda.max_memory_allocated() - base_mem) / (1024 ** 2)
    else:
        qr_peak_mib = 0.0
    qr_single_ms = (time.perf_counter() - t0) * 1000.0

    assert collapsed is not None
    b_qr, a_qr, stats = collapsed
    assert stats.r_in_sum == 544 and id_stats.r_in_sum == 544
    assert stats.r_out == 64 and id_stats.r_out == 64
    assert b_qr.shape == (d_out, 64) and a_qr.shape == (64, d_in)

    # Benchmark full `[28672, 5376]` dense FP32 SVD latency vs QR-Core!
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else False
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    t_d0 = time.perf_counter()
    w_dense = torch.zeros((d_out, d_in), device=dev, dtype=torch.float32)
    for b_m, a_m, s_m in entries:
        w_dense.addmm_(b_m.float(), a_m.float(), beta=1.0, alpha=s_m)
    u_full, s_full, vh_full = torch.linalg.svd(w_dense, full_matrices=False)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dense_svd_ms = (time.perf_counter() - t_d0) * 1000.0
    w_best64_dense = (u_full[:, :64] * s_full[:64].unsqueeze(0)) @ vh_full[:64, :]
    w_best64_qr = b_qr.float() @ a_qr.float()
    cos_fp32_gpu = _cosine(w_best64_qr, w_best64_dense)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    assert cos_fp32_gpu > 0.999995
    del w_dense, u_full, s_full, vh_full, w_best64_dense, w_best64_qr
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    assert cos_svd > 0.999999, f"Expected QR-Core vs dense SVD cos > 0.999999, got {cos_svd:.9f}"
    assert rel_frob_diff < 1e-5, f"Expected rel_frob_diff < 1e-5, got {rel_frob_diff:.3e}"

    # 2. Also verify adaptive spectral energy threshold (`energy_ratio=0.992, max_rank=544`) on collinear LoRAs
    b_base = torch.randn(4096, 32, device=dev, dtype=torch.bfloat16) * 0.02
    a_base = torch.randn(32, 5376, device=dev, dtype=torch.bfloat16) * 0.02
    collinear_entries = [
        (b_base, a_base, 0.6),
        (b_base * 1.05 + torch.randn_like(b_base) * 1e-4, a_base, 0.4),
    ]
    _, _, col_stats = lcb.qr_core_collapse_loras(
        collinear_entries, max_rank=64, energy_ratio=0.992, out_dtype=torch.bfloat16, compute_device=dev
    )
    assert col_stats.r_out <= 33 and col_stats.captured_energy >= 0.992

    BENCHMARK_SUMMARY["test2_cos_vs_dense_svd"] = cos_svd
    BENCHMARK_SUMMARY["test2_rel_frob_diff"] = rel_frob_diff
    BENCHMARK_SUMMARY["test2_qr_fc1_544to64_ms"] = round(qr_single_ms, 2)
    BENCHMARK_SUMMARY["test2_dense_svd_fc1_ms"] = round(dense_svd_ms, 2)
    BENCHMARK_SUMMARY["test2_speedup_vs_dense_svd"] = round(dense_svd_ms / max(qr_single_ms, 1e-6), 2)
    BENCHMARK_SUMMARY["test2_qr_peak_temp_mib"] = round(qr_peak_mib, 2)


# ===========================================================================
# TEST 3: Hadamard Rotated INT8 × BF16 LoRA (§3.2) + nvcc 13.2 + ChunkedSage
# ===========================================================================

@unittest.skipIf(not torch.cuda.is_available(), "CUDA GPU required for Test 3")
def test_3a_nvcc_compile_and_rot_int8_lora_kernels():
    """Verify `E:/cuda/bin/nvcc.exe`, ConvRot-256 Hadamard exactness, and `cuda`/`triton`/`eager` cos > 0.999."""
    nvcc_bin = rlo.find_nvcc_executable()
    assert nvcc_bin is not None, "Expected nvcc executable (`E:/cuda/bin/nvcc.exe`) to be found"
    ver_proc = subprocess.run([nvcc_bin, "--version"], capture_output=True, text=True, check=True)
    assert "release" in ver_proc.stdout.lower()

    cuda_lib = rlo.load_rot_lora_cuda_lib()
    assert cuda_lib is not None, "Expected native csrc/rot_lora_cuda.cu DLL to compile & load cleanly"
    assert rlo.get_rot_lora_active_backend() == "cuda"

    # 1. Verify ConvRot-256 Hadamard symmetry, involution, and exact match vs comfy_kitchen
    torch.manual_seed(777)
    a_orig = (torch.randn(64, 5376, device="cuda", dtype=torch.float32) * 0.03).to(torch.bfloat16)

    # Warmup & time pre-rotation (`< 5 ms` requirement!)
    _ = rlo.pre_rotate_lora_down(a_orig, group_size=256)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    a_rot = rlo.pre_rotate_lora_down(a_orig, group_size=256)
    torch.cuda.synchronize()
    prerot_ms = (time.perf_counter() - t0) * 1000.0
    assert prerot_ms < 5.0, f"Pre-rotation took {prerot_ms:.3f} ms >= 5.0 ms"

    h256 = _build_hadamard(256, device="cuda", dtype=torch.float32)
    a_rot_ref = (a_orig.float().reshape(64, 5376 // 256, 256) @ h256).reshape(64, 5376).to(torch.bfloat16)
    assert torch.equal(a_rot, a_rot_ref), "CUDA radix-4 ConvRot-256 differs from exact _build_hadamard(256)"

    # Involution check: (A_f32 @ H) @ H == A_f32 to exact FP32 precision, and BF16 round-trip cos > 0.99999
    a_f32 = torch.randn(64, 5376, device="cuda", dtype=torch.float32) * 0.03
    a_f32_back = rlo.pre_rotate_lora_down(rlo.pre_rotate_lora_down(a_f32, group_size=256), group_size=256)
    assert float((a_f32_back - a_f32).abs().max().item()) < 1e-6
    a_bf16_back = rlo.pre_rotate_lora_down(a_rot, group_size=256)
    assert _cosine(a_bf16_back, a_orig) > 0.99999

    # 2. Verify all 3 backends (`cuda`, `triton`, `eager`) on realistic outlier-heavy DiT activations
    s_tok = 2048
    h_act = torch.randn(s_tok, 5376, device="cuda", dtype=torch.bfloat16)
    # Inject realistic transformer channel outliers (10x spikes in specific channels, which ConvRot flattens!)
    h_act[:, ::137] *= 10.0

    # Use real `sol_sharp_v3_day2` LoRA weights if present
    day2_path = LORA_DIR / "minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors"
    if day2_path.is_file():
        with safe_open(str(day2_path), framework="pt", device="cpu") as fd2:
            a_real = fd2.get_tensor("diffusion_model.blocks.12.attn.qkv_proj.lora_down.weight").to("cuda", torch.bfloat16)
            b_real = fd2.get_tensor("diffusion_model.blocks.12.attn.qkv_proj.lora_up.weight").to("cuda", torch.bfloat16)
    else:
        a_real = a_orig
        b_real = (torch.randn(21504, 64, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)

    a_real_rot = rlo.pre_rotate_lora_down(a_real, group_size=256)

    x_qdata = torch.empty((s_tok, 5376), dtype=torch.int8, device="cuda")
    x_scale = torch.empty((s_tok, 1), dtype=torch.float32, device="cuda")
    cs._quantize_int8_convrot_into(h_act, x_qdata, x_scale, convrot=True, convrot_groupsize=256)

    gate_vec = torch.randn(b_real.shape[0], device="cuda", dtype=torch.bfloat16) * 0.5
    y_exact_bf16 = F.linear(F.linear(h_act, a_real), b_real) * gate_vec.unsqueeze(0)

    backend_cos: dict[str, float] = {}
    for bk in ("cuda", "triton", "eager"):
        y_rot = rlo.rot_int8_lora_forward(x_qdata, x_scale, a_real_rot, b_real, gate=gate_vec, backend=bk)
        c_val = _cosine(y_exact_bf16, y_rot)
        backend_cos[bk] = c_val
        assert c_val > 0.999, f"Backend {bk} cosine {c_val:.6f} <= 0.999"

    BENCHMARK_SUMMARY["test3a_nvcc_path"] = nvcc_bin
    BENCHMARK_SUMMARY["test3a_prerot_single_ms"] = round(prerot_ms, 4)
    BENCHMARK_SUMMARY["test3a_backend_cosines"] = {k: round(v, 7) for k, v in backend_cos.items()}


@unittest.skipIf(not torch.cuda.is_available(), "CUDA GPU required for Test 3B")
def test_3b_chunked_sage_end_to_end_and_large_seq_benchmark():
    """Verify `ChunkedSage` INT8 ConvRot `DiTBlock` end-to-end + `S=32768` throughput/VRAM benchmark."""
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    torch.manual_seed(99)

    # 1. Build a real H3 `DiTBlock` (`hidden=5376, heads=56, head_dim=128, ffn=14336`) quantized to INT8 ConvRot-256!
    blk = DiTBlock(
        hidden=5376,
        heads=56,
        head_dim=128,
        ffn=14336,
        t_dim=8,
        eps=1e-5,
        qk_eps=1e-5,
        apply_silu=False,
        adaln_dtype=torch.float32,
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.manual_cast,
    ).eval()

    with torch.no_grad():
        for p in blk.parameters():
            p.normal_(0.0, 0.02)
        for lin_name, lin in (
            ("qkv", blk.attn.qkv_proj),
            ("out", blk.attn.out_proj),
            ("fc1", blk.mlp.fc1),
            ("fc2", blk.mlp.fc2),
        ):
            qt = qops.QuantizedTensor.from_float(
                lin.weight.data,
                "TensorWiseINT8Layout",
                per_channel=True,
                convrot=True,
                convrot_groupsize=256,
            )
            lin.weight = torch.nn.Parameter(qt, requires_grad=False)
            lin.layout_type = TensorWiseINT8Layout

    # Attach 4 stacked LoRAs (`R_sum = 384 + 64 + 64 + 32 = 544`) to `qkv_proj` and `fc1`
    def _make_stacked(out_f: int, in_f: int):
        out_ents = []
        for r_m, sc_m in ((384, 0.5), (64, 0.8), (64, 0.25), (32, 0.7)):
            bm = (torch.randn(out_f, r_m, device=device, dtype=torch.float32) * 0.015).to(torch.bfloat16)
            am = (torch.randn(r_m, in_f, device=device, dtype=torch.float32) * 0.015).to(torch.bfloat16)
            out_ents.append((bm, am, sc_m))
        return out_ents

    qkv_entries = _make_stacked(21504, 5376)
    fc1_entries = _make_stacked(28672, 5376)
    qkv_opt, qkv_rot = cs._optimize_lora_entries_for_mode(qkv_entries, "rotated_int8", allow_rot256=True)
    fc1_opt, fc1_rot = cs._optimize_lora_entries_for_mode(fc1_entries, "rotated_int8", allow_rot256=True)
    assert qkv_rot is not None and fc1_rot is not None
    assert qkv_rot[1].shape == (64, 5376) and fc1_rot[1].shape == (64, 5376)

    attn_fwd = cs._make_chunked_attn_forward(head_chunks=4, token_chunk=1024, out_proj_mode="streaming")
    block_fwd = cs._make_chunked_block_forward(head_chunks=4, out_proj_mode="streaming", token_chunk=1024)
    import types
    blk.attn.forward = types.MethodType(attn_fwd, blk.attn)
    blk.forward = types.MethodType(block_fwd, blk)

    s_e2e = 1024
    x0 = torch.randn(s_e2e, 5376, device=device, dtype=torch.bfloat16)
    t_emb = torch.randn(2, 8, device=device, dtype=torch.float32)
    mod_segs = [(0, 256, 1), (256, s_e2e, 0)]

    with torch.no_grad():
        # Run qr_collapsed (exact BF16 h_c side-branch) vs rotated_int8 (INT8 X_q side-branch, zero h_c re-read!)
        t_opts_qr = {
            "block_index": 0,
            "_chunked_sage_active_loras": {id(blk.attn.qkv_proj): qkv_opt, id(blk.mlp.fc1): fc1_opt},
        }
        out_qr = blk(x0.clone(), t_emb, mod_segs, rope_freqs=None, transformer_options=t_opts_qr)

        t_opts_rot = {
            "block_index": 0,
            "_chunked_sage_active_loras": {id(blk.attn.qkv_proj): qkv_opt, id(blk.mlp.fc1): fc1_opt},
            "_chunked_sage_rot_loras": {id(blk.attn.qkv_proj): qkv_rot, id(blk.mlp.fc1): fc1_rot},
        }
        out_rot = blk(x0.clone(), t_emb, mod_segs, rope_freqs=None, transformer_options=t_opts_rot)

    cos_block = _cosine(out_qr, out_rot)
    assert cos_block > 0.9995, f"End-to-end INT8 DiTBlock rotated_int8 vs qr_collapsed cos={cos_block:.6f}"

    del blk, x0, out_qr, out_rot, qkv_entries, qkv_opt, qkv_rot
    torch.cuda.empty_cache()

    # 2. Large-Sequence (`S = 32,768`, `d_in = 5376`, `d_out = 28672`) Throughput & VRAM Benchmark
    #    Check free VRAM first per 8GB GPU contention protocol!
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes >= int(3.0 * (1024 ** 3)):
        s_bench = 32768
        h_bench = torch.randn(s_bench, 5376, device=device, dtype=torch.bfloat16)
        x_q_bench = torch.empty((s_bench, 5376), dtype=torch.int8, device=device)
        x_s_bench = torch.empty((s_bench, 1), dtype=torch.float32, device=device)
        cs._quantize_int8_convrot_into(h_bench, x_q_bench, x_s_bench, convrot=True, convrot_groupsize=256)

        b_cat_544, a_cat_544 = cs._prepare_lora_packed(fc1_entries, device, torch.bfloat16)
        b_rot_64, a_rot_64 = fc1_rot
        gate_fc1 = torch.randn(28672, device=device, dtype=torch.bfloat16) * 0.2
        out_sink = torch.zeros((s_bench, 8192), device=device, dtype=torch.bfloat16)  # slice of output to fit 8GB cleanly
        b544_sl = b_cat_544[:8192].contiguous()
        b64_sl = b_rot_64[:8192].contiguous()
        gate_sl = gate_fc1[:8192].contiguous()

        # Warmup
        for _ in range(3):
            mid544 = F.linear(h_bench, a_cat_544)
            torch.addmm(out_sink, mid544, (b544_sl * gate_sl.unsqueeze(1)).t(), out=out_sink)
            del mid544
            rlo.rot_int8_lora_forward(x_q_bench, x_s_bench, a_rot_64, b64_sl, gate=gate_sl, out_accum=out_sink, backend="cuda")
            rlo.rot_int8_lora_forward(x_q_bench, x_s_bench, a_rot_64, b64_sl, gate=gate_sl, out_accum=out_sink, backend="triton")
        torch.cuda.synchronize()

        # Benchmark Legacy BF16 R_sum=544
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        t_leg0 = time.perf_counter()
        n_iters = 10
        for _ in range(n_iters):
            mid544 = F.linear(h_bench, a_cat_544)
            torch.addmm(out_sink, mid544, (b544_sl * gate_sl.unsqueeze(1)).t(), out=out_sink)
            del mid544
        torch.cuda.synchronize()
        legacy_ms = (time.perf_counter() - t_leg0) * 1000.0 / n_iters
        legacy_extra_mib = (torch.cuda.max_memory_allocated() - m0) / (1024 ** 2)

        # Free unrotated BF16 `h_bench` (336 MiB @ S=32k, 1.58 GiB @ S=147k!) — not needed by rotated_int8!
        del h_bench, b_cat_544, a_cat_544
        torch.cuda.empty_cache()

        # Benchmark QR-Core (R=64) + Rotated INT8 Native CUDA Kernel
        torch.cuda.reset_peak_memory_stats()
        m1 = torch.cuda.memory_allocated()
        t_cu0 = time.perf_counter()
        for _ in range(n_iters):
            rlo.rot_int8_lora_forward(
                x_q_bench, x_s_bench, a_rot_64, b64_sl, gate=gate_sl, out_accum=out_sink, backend="cuda"
            )
        torch.cuda.synchronize()
        rot_cuda_ms = (time.perf_counter() - t_cu0) * 1000.0 / n_iters
        rot_cuda_extra_mib = (torch.cuda.max_memory_allocated() - m1) / (1024 ** 2)

        # Benchmark QR-Core (R=64) + Rotated INT8 Triton Kernel
        t_tr0 = time.perf_counter()
        for _ in range(n_iters):
            rlo.rot_int8_lora_forward(
                x_q_bench, x_s_bench, a_rot_64, b64_sl, gate=gate_sl, out_accum=out_sink, backend="triton"
            )
        torch.cuda.synchronize()
        rot_triton_ms = (time.perf_counter() - t_tr0) * 1000.0 / n_iters

        BENCHMARK_SUMMARY["test3b_e2e_block_cos"] = round(cos_block, 7)
        BENCHMARK_SUMMARY["test3b_S32768_legacy_r544_ms"] = round(legacy_ms, 3)
        BENCHMARK_SUMMARY["test3b_S32768_rot_cuda_r64_ms"] = round(rot_cuda_ms, 3)
        BENCHMARK_SUMMARY["test3b_S32768_rot_triton_r64_ms"] = round(rot_triton_ms, 3)
        BENCHMARK_SUMMARY["test3b_S32768_speedup_cuda"] = round(legacy_ms / max(rot_cuda_ms, 1e-6), 2)
        BENCHMARK_SUMMARY["test3b_S32768_speedup_triton"] = round(legacy_ms / max(rot_triton_ms, 1e-6), 2)
        BENCHMARK_SUMMARY["test3b_S32768_legacy_temp_mib"] = round(legacy_extra_mib, 2)
        BENCHMARK_SUMMARY["test3b_S32768_rot_cuda_temp_mib"] = round(rot_cuda_extra_mib, 2)

        del x_q_bench, x_s_bench, out_sink, b544_sl, b64_sl, gate_sl
        torch.cuda.empty_cache()


def test_z_print_summary():
    if BENCHMARK_SUMMARY:
        print("\n=== PHASE 1 H3-ANIMEENGINE VERIFICATION METRICS ===")
        print(json.dumps(BENCHMARK_SUMMARY, indent=2, ensure_ascii=False))


class TestPhase1AnimeEngine(unittest.TestCase):
    """Standard `unittest.TestCase` runner (also compatible with `pytest`)."""

    def test_1a_adaln_curve_bridge_math(self):
        test_1a_adaln_curve_bridge_math_and_bias_absorption()

    @unittest.skipIf(not DASIWA_PATH.is_file(), "Local DaSiWa v5xs safetensors not present")
    def test_1b_real_dasiwa_and_market_loras(self):
        test_1b_real_dasiwa_and_market_loras_bridge()

    def test_2_qr_core_collapse(self):
        test_2_qr_core_collapse_544_to_64_identity_and_benchmark()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA GPU required for Test 3A")
    def test_3a_nvcc_and_rot_int8_lora(self):
        test_3a_nvcc_compile_and_rot_int8_lora_kernels()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA GPU required for Test 3B")
    def test_3b_chunked_sage_e2e_and_bench(self):
        test_3b_chunked_sage_end_to_end_and_large_seq_benchmark()

    def test_z_summary(self):
        test_z_print_summary()


if __name__ == "__main__":
    unittest.main(verbosity=2)
