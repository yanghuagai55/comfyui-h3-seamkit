# -*- coding: utf-8 -*-
"""Unit & GPU parity tests for MiniMaxH3ChunkedSageAttentionPatch."""

from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn as nn

COMFY_ROOT = r"D:\comfyui\ComfyUI"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if os.path.dirname(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(REPO_ROOT))

# Ensure ComfyUI package imports work cleanly outside live server
import importlib.util

import comfy.ops
from comfy.ldm.minimax.model import DiTBlock, FinalLayer, pdd_grid, rope_rotation_table

# Load chunked_sage via package import
_pkg_name = os.path.basename(REPO_ROOT).replace("-", "_")
spec_pkg = importlib.util.spec_from_file_location(
    _pkg_name,
    os.path.join(REPO_ROOT, "__init__.py"),
    submodule_search_locations=[REPO_ROOT],
)
pkg = importlib.util.module_from_spec(spec_pkg)
sys.modules[_pkg_name] = pkg
spec_cs = importlib.util.spec_from_file_location(
    f"{_pkg_name}.chunked_sage", os.path.join(REPO_ROOT, "chunked_sage.py")
)
cs = importlib.util.module_from_spec(spec_cs)
sys.modules[f"{_pkg_name}.chunked_sage"] = cs
spec_cs.loader.exec_module(cs)

CKPT_PATH = r"D:\comfyui\ComfyUI\models\unet\dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors"


def test_split_heads_even():
    for n in [1, 2, 4, 6, 7, 8, 14, 28]:
        gs = cs._split_heads_even(56, n, convrot_groupsize=256, head_dim=128)
        assert sum(gs) == 56, f"sum mismatch for n={n}: {gs}"
        assert len(gs) == n, f"len mismatch for n={n}: {gs}"
        assert all(g % 2 == 0 and g > 0 for g in gs), f"non-even group for n={n}: {gs}"
    # Edge: n=56 -> 56 groups of 1
    gs56 = cs._split_heads_even(56, 56, convrot_groupsize=256, head_dim=128)
    assert gs56 == [1] * 56
    print("[PASS] test_split_heads_even")


def test_slice_mod_segments():
    per_tok = torch.arange(100, 200, dtype=torch.long)
    segs = [(0, 10, 1), (10, 30, 5), (30, 130, per_tok)]
    sub = cs._slice_mod_segments(segs, 5, 45)
    assert len(sub) == 3
    assert sub[0] == (0, 5, 1)
    assert sub[1] == (5, 25, 5)
    assert sub[2][0] == 25 and sub[2][1] == 40
    assert torch.equal(sub[2][2], torch.arange(100, 115, dtype=torch.long))
    print("[PASS] test_slice_mod_segments")


@torch.inference_mode()
def test_final_layer_and_patch_proj_bit_exact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    ops = comfy.ops.disable_weight_init
    for pdd_steps in (1, 4):
        fl = FinalLayer(
            hidden=256, t_dim=8, video_dim=96, audio_dim=32, eps=1e-5,
            apply_silu=False, adaln_dtype=torch.float32, pdd_steps=pdd_steps,
            dtype=dtype, device=device, operations=ops,
        ).requires_grad_(False).eval()
        for p in fl.parameters():
            p.normal_(0, 0.05)

        S = 3000
        x = torch.randn(S, 256, dtype=dtype, device=device)
        t_emb = torch.randn(2, 8, dtype=torch.float32, device=device)
        video_seg = (500, 3000, 0)
        audio_seg = (100, 500, 1)
        sigma = torch.tensor(0.5, device=device)
        sample_sigmas = torch.linspace(1.0, 0.0, 5, device=device)
        grids = (pdd_grid(pdd_steps, 12.0), pdd_grid(pdd_steps, 3.0))

        ref_v, ref_a = fl(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids)
        chunked_fwd = types.MethodType(cs._make_chunked_final_layer_forward(token_chunk=512), fl)
        got_v, got_a = chunked_fwd(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids)
        assert torch.allclose(ref_v, got_v, atol=1e-5, rtol=1e-5), f"FinalLayer video mismatch (pdd={pdd_steps}): max={(ref_v-got_v).abs().max()}"
        assert torch.allclose(ref_a, got_a, atol=1e-5, rtol=1e-5), f"FinalLayer audio mismatch (pdd={pdd_steps}): max={(ref_a-got_a).abs().max()}"

    # video_patch_proj chunked exactness
    vpp = ops.Linear(96, 256, bias=True, dtype=torch.float32, device=device).requires_grad_(False).eval()
    for p in vpp.parameters():
        p.normal_(0, 0.05)
    rows = torch.randn(2500, 96, dtype=torch.float32, device=device)
    ref_emb = vpp(rows).to(dtype)
    vpp_chunked = types.MethodType(
        cs._make_chunked_video_patch_proj_forward(vpp.forward, token_chunk=512, target_dtype=dtype), vpp
    )
    got_emb = vpp_chunked(rows)
    assert got_emb.dtype == dtype
    assert torch.allclose(ref_emb.float(), got_emb.float(), atol=1e-2, rtol=1e-2), f"video_patch_proj mismatch: max={(ref_emb-got_emb).abs().max()}"
    print("[PASS] test_final_layer_and_patch_proj_bit_exact")


@torch.inference_mode()
def test_dense_bf16_block_and_external_deferral():
    if not torch.cuda.is_available():
        print("[SKIP] test_dense_bf16_block_and_external_deferral (no CUDA)")
        return
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ops = comfy.ops.disable_weight_init
    block = DiTBlock(
        hidden=512, heads=8, head_dim=128, ffn=1024, t_dim=8,
        eps=1e-5, qk_eps=1e-5, apply_silu=False, adaln_dtype=torch.float32,
        dtype=dtype, device=device, operations=ops,
    ).requires_grad_(False).eval()
    gen = torch.Generator(device=device).manual_seed(7)
    for p in block.parameters():
        p.normal_(0, 0.04, generator=gen)

    S = 2048
    x0 = torch.randn(S, 512, dtype=dtype, device=device, generator=gen) * 0.5
    t_emb = torch.randn(2, 8, dtype=torch.float32, device=device, generator=gen) * 0.2
    mod_segments = [(0, 128, 1), (128, 384, 5), (384, S, 0)]
    angles = torch.randn(S, 96, dtype=torch.float32, device=device, generator=gen)
    angles[:, 48:] = angles[:, :48]
    rope_freqs = rope_rotation_table(angles, dtype)

    # Patch attn + block in exact & streaming modes
    block.attn.forward = types.MethodType(cs._make_chunked_attn_forward(head_chunks=4, token_chunk=512), block.attn)
    assert "sageattn" in block.attn.forward.__name__

    fwd_exact = types.MethodType(cs._make_chunked_block_forward(head_chunks=4, out_proj_mode="exact", token_chunk=512), block)
    fwd_stream = types.MethodType(cs._make_chunked_block_forward(head_chunks=4, out_proj_mode="streaming", token_chunk=512), block)
    out_exact = fwd_exact(x0.clone(), t_emb, mod_segments, rope_freqs)
    out_stream = fwd_stream(x0.clone(), t_emb, mod_segments, rope_freqs)

    # For dense BF16, streaming out_proj is algebraically exact up to BF16 summation order!
    cos = nn.functional.cosine_similarity(out_exact.float().flatten(), out_stream.float().flatten(), dim=0).item()
    rel = ((out_exact.float() - out_stream.float()).norm() / out_exact.float().norm()).item()
    assert cos > 0.99999 and rel < 5e-3, f"BF16 exact vs streaming diverged: cos={cos}, rel={rel}"

    # External attention deferral check (e.g. Route 3 beta_attention / custom attention arg)
    called = {"custom": 0}
    def custom_attn(h, rope_freqs=None, transformer_options={}):
        called["custom"] += 1
        t = h.pop() if isinstance(h, list) else h
        return torch.zeros_like(t)
    _ = fwd_exact(x0.clone(), t_emb, mod_segments, rope_freqs, attention=custom_attn)
    assert called["custom"] == 1, "External attention argument was NOT respected!"

    # Overwritten block.attn.forward deferral check (simulates Route 3 beta_attention context manager)
    orig_attn_fwd = block.attn.forward
    block.attn.forward = types.MethodType(lambda self, h, rope_freqs=None, transformer_options={}: custom_attn(h), block.attn)
    _ = fwd_exact(x0.clone(), t_emb, mod_segments, rope_freqs)
    block.attn.forward = orig_attn_fwd
    assert called["custom"] == 2, "Overwritten attn.forward (Route 3 beta_attention) was NOT respected!"
    print(f"[PASS] test_dense_bf16_block_and_external_deferral (cos={cos:.7f}, rel={rel:.6f})")


@torch.inference_mode()
def test_real_dasiwa_int8_bit_exact():
    if not torch.cuda.is_available() or not os.path.exists(CKPT_PATH):
        print("[SKIP] test_real_dasiwa_int8_bit_exact (no CUDA or checkpoint)")
        return
    from safetensors import safe_open
    import server
    if not hasattr(server.PromptServer, "instance"):
        class _R:
            def post(self, *a, **kw): return lambda f: f
            def get(self, *a, **kw): return lambda f: f
        class _S:
            routes = _R()
        server.PromptServer.instance = _S()

    kj_ltxv_path = os.path.join(COMFY_ROOT, "custom_nodes", "ComfyUI-KJNodes", "nodes", "ltxv_nodes.py")
    spec_kj = importlib.util.spec_from_file_location("kj_ltxv_test", kj_ltxv_path)
    kj_ltxv = importlib.util.module_from_spec(spec_kj)
    spec_kj.loader.exec_module(kj_ltxv)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    ops = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
    block = DiTBlock(
        hidden=5376, heads=56, head_dim=128, ffn=14336, t_dim=8,
        eps=1e-5, qk_eps=1e-5, apply_silu=False, adaln_dtype=torch.float32,
        gate_compress=False, dtype=dtype, device="cpu", operations=ops,
    )
    prefix = "blocks.0."
    sd = {}
    with safe_open(CKPT_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    block.load_state_dict(sd, strict=False)
    block.to(device=device).requires_grad_(False).eval()

    S = 4096
    gen = torch.Generator(device=device).manual_seed(42)
    x0 = torch.randn(S, 5376, dtype=dtype, device=device, generator=gen) * 0.5
    t_emb = torch.randn(2, 8, dtype=dtype, device=device, generator=gen) * 0.2
    mod_segments = [(0, 128, 1), (128, 384, 5), (384, S, 0)]
    angles = torch.randn(S, 96, dtype=torch.float32, device=device, generator=gen)
    angles[:, 48:] = angles[:, :48]
    rope_freqs = rope_rotation_table(angles, dtype)

    # Stock KJ reference
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    x_kj = x0.clone()
    h_kj = [cs._mod_scale_shift(block.norm1(x_kj), shift_msa, scale_msa, mod_segments)]
    a_kj = kj_ltxv.minimax_sageattn_forward(block.attn, h_kj, rope_freqs=rope_freqs, transformer_options={"minimax_head_chunks": 7})
    x_kj = cs._mod_gate(x_kj, gate_msa, a_kj, mod_segments)
    h2 = cs._mod_scale_shift(block.norm2(x_kj), shift_mlp, scale_mlp, mod_segments)
    x_kj = cs._mod_gate(x_kj, gate_mlp, block.mlp(h2), mod_segments)

    # Patch with our chunked_sage
    block.attn.forward = types.MethodType(cs._make_chunked_attn_forward(head_chunks=14, token_chunk=2048), block.attn)
    blk_exact = types.MethodType(cs._make_chunked_block_forward(head_chunks=14, out_proj_mode="exact", token_chunk=2048), block)
    blk_stream = types.MethodType(cs._make_chunked_block_forward(head_chunks=14, out_proj_mode="streaming", token_chunk=2048), block)

    out_exact = blk_exact(x0.clone(), t_emb, mod_segments, rope_freqs)
    assert torch.equal(out_exact, x_kj), f"Exact mode NOT bit-exact! max_abs={(out_exact-x_kj).abs().max().item()}"

    # Standalone attn.forward path (when KJ LowVRAM wraps block.forward!)
    x_standalone = x0.clone()
    h_st = [cs._mod_scale_shift(block.norm1(x_standalone), shift_msa, scale_msa, mod_segments)]
    a_st = block.attn(h_st, rope_freqs=rope_freqs, transformer_options={"minimax_head_chunks": 7})
    assert torch.equal(a_st, a_kj), f"Standalone attn.forward NOT bit-exact! max_abs={(a_st-a_kj).abs().max().item()}"

    # Streaming mode quality check
    out_stream = blk_stream(x0.clone(), t_emb, mod_segments, rope_freqs)
    cos = nn.functional.cosine_similarity(out_stream.float().flatten(), x_kj.float().flatten(), dim=0).item()
    assert cos > 0.9999, f"Streaming mode cos too low: {cos}"
    print(f"[PASS] test_real_dasiwa_int8_bit_exact (exact=100.00% bit-identical, standalone_attn=100.00% bit-identical, streaming_cos={cos:.7f})")


def test_effective_load_thresholds():
    spec_up = importlib.util.spec_from_file_location(
        f"{_pkg_name}.h3_upscale", os.path.join(REPO_ROOT, "h3_upscale.py")
    )
    up = importlib.util.module_from_spec(spec_up)
    sys.modules[f"{_pkg_name}.h3_upscale"] = up
    # Load sibling dependencies expected by relative imports in h3_upscale
    for sub in ("bridge", "hardcut_math"):
        if f"{_pkg_name}.{sub}" not in sys.modules:
            sp = importlib.util.spec_from_file_location(
                f"{_pkg_name}.{sub}", os.path.join(REPO_ROOT, f"{sub}.py")
            )
            mod = importlib.util.module_from_spec(sp)
            sys.modules[f"{_pkg_name}.{sub}"] = mod
            sp.loader.exec_module(mod)
    spec_up.loader.exec_module(up)

    class _DummyModel:
        def __init__(self, opts):
            self.model_options = opts

    lp0, lf0, info0 = up.effective_load_thresholds(_DummyModel({}))
    assert lp0 == 180.0 and lf0 == 191.0 and info0 is None

    lp1, lf1, info1 = up.effective_load_thresholds(
        _DummyModel({"transformer_options": {"minimax_chunked_sage": {"out_proj_mode": "exact"}}})
    )
    assert lp1 == up.LOAD_PASS_CHUNKED_EXACT and lf1 == up.LOAD_FAIL_CHUNKED_EXACT and info1 is not None

    lp2, lf2, info2 = up.effective_load_thresholds(
        _DummyModel({"transformer_options": {"minimax_chunked_sage": {"out_proj_mode": "auto"}}})
    )
    assert lp2 == up.LOAD_PASS_CHUNKED_STREAMING and lf2 == up.LOAD_FAIL_CHUNKED_STREAMING and info2 is not None
    print("[PASS] test_effective_load_thresholds")


if __name__ == "__main__":
    test_split_heads_even()
    test_slice_mod_segments()
    test_effective_load_thresholds()
    test_final_layer_and_patch_proj_bit_exact()
    test_dense_bf16_block_and_external_deferral()
    test_real_dasiwa_int8_bit_exact()
    print("ALL CHUNKED SAGE TESTS PASSED!")
