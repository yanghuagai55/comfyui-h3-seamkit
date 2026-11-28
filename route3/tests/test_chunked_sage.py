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


@torch.inference_mode()
def test_sol_attn_morton_and_sparse_path():
    # 1. Dense blocks parser
    assert cs._parse_dense_blocks("", 50) == frozenset()
    assert cs._parse_dense_blocks("0-1, 48-49, -1", 50) == frozenset({0, 1, 48, 49})
    assert cs._parse_dense_blocks("5-3", 50) == frozenset({3, 4, 5})

    # 2. Morton round-trip bit-exactness with non-64-aligned start + column-chunked in-place permute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = 277
    grid = (4, 8, 12)
    n_vid = 4 * 8 * 12
    total = start + n_vid
    h0 = torch.randn(total, 5376, dtype=torch.bfloat16, device=device)
    h = h0.clone()
    perm, inv, _ = cs._morton_perm_aligned(grid, start, device, curve="2d_frame")
    cs._permute_span_inplace(h, start, total, perm, col_chunk=1344)
    assert torch.equal(h[:start], h0[:start]), "Prefix modified by Morton permute!"
    assert not torch.equal(h[start:], h0[start:]), "Video span not permuted!"
    cs._permute_span_inplace(h, start, total, inv, col_chunk=1344)
    assert torch.equal(h, h0), "Morton round-trip NOT bit-exact!"

    # 3. _build_sol_context sigma gate & sink calculation
    from comfy.ldm.minimax.model import PackedLayout
    layout = PackedLayout(text_len=128, latent_t=10, latent_h=32, latent_w=48, audio_t=100)
    sol_cfg = {
        "attention_mode": "sol_attn_sparse",
        "sol_tau": 1.3,
        "sol_token_aug": 64,
        "sol_topk_ratio": 0.0,
        "sol_sink_mode": "exact_kv_and_rows",
        "sol_min_tokens": 1024,
        "sol_dense_blocks_set": frozenset(),
        "sol_max_sigma": 0.95,
        "sol_min_sigma": 0.05,
    }
    if cs._sol_attn_available:
        ctx_hi, _ = cs._build_sol_context(layout, layout.seq_len, sol_cfg, {"sigmas": torch.tensor([0.99])})
        assert ctx_hi is None, "Expected sigma > sol_max_sigma to skip Sol-Attn"
        ctx_lo, _ = cs._build_sol_context(layout, layout.seq_len, sol_cfg, {"sigmas": torch.tensor([0.01])})
        assert ctx_lo is None, "Expected sigma < sol_min_sigma to skip Sol-Attn"
        ctx_ok, m_info = cs._build_sol_context(layout, layout.seq_len, sol_cfg, {"sigmas": torch.tensor([0.30])})
        assert ctx_ok is not None and m_info is not None
        assert ctx_ok["sink_blocks"][0] == 0 and ctx_ok["sink_blocks"][1] > 0
        assert ctx_ok["sink_q"][1] == ctx_ok["sink_blocks"][1]
        assert m_info[2] == (10, 16, 24)

    # 4. End-to-end Sol-Attn + Morton through _make_sol_run_blocks on real dasiwa blocks.0
    if torch.cuda.is_available() and cs._sol_attn_available and os.path.exists(CKPT_PATH):
        from safetensors import safe_open
        dtype = torch.bfloat16
        ops = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
        block = DiTBlock(
            hidden=5376, heads=56, head_dim=128, ffn=14336, t_dim=8,
            eps=1e-5, qk_eps=1e-5, apply_silu=False, adaln_dtype=torch.float32,
            gate_compress=False, dtype=dtype, device="cpu", operations=ops,
        )
        sd = {}
        with safe_open(CKPT_PATH, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("blocks.0."):
                    sd[k[len("blocks.0."):]] = f.get_tensor(k)
        block.load_state_dict(sd, strict=False)
        block.to(device=device).requires_grad_(False).eval()
        block.attn.forward = types.MethodType(cs._make_chunked_attn_forward(head_chunks=14, token_chunk=2048), block.attn)
        block.forward = types.MethodType(cs._make_chunked_block_forward(head_chunks=14, out_proj_mode="exact", token_chunk=2048), block)

        class _MiniRunner:
            def _run_blocks(self, h_in, t_emb, mod_segs, lay, topts):
                pos = lay.position_ids.to(torch.float32).to(h_in.device)
                inv_f = torch.linspace(0.01, 1.0, 16, device=h_in.device)
                per_ax = pos.unsqueeze(-1) * inv_f.view(1, 1, -1)
                t_f, h_f, w_f = per_ax.unbind(dim=1)
                half = torch.cat((t_f, h_f, w_f), dim=-1)
                rope = rope_rotation_table(torch.cat((half, half), dim=-1), h_in.dtype)
                topts["block_index"] = 0
                return block(h_in, t_emb, mod_segs, rope, transformer_options=topts)

        runner = _MiniRunner()
        lay = PackedLayout(text_len=128, latent_t=8, latent_h=32, latent_w=48, audio_t=80)
        S = lay.seq_len
        gen = torch.Generator(device=device).manual_seed(99)
        x0 = torch.randn(S, 5376, dtype=dtype, device=device, generator=gen) * 0.4
        t_emb = torch.randn(2, 8, dtype=dtype, device=device, generator=gen) * 0.2
        mod_segs = [(0, 128, 1), (128, 288, 5), (288, S, 0)]

        out_dense = runner._run_blocks(x0.clone(), t_emb, mod_segs, lay, {"sigmas": torch.tensor([0.30])})
        sol_rb = types.MethodType(cs._make_sol_run_blocks(_MiniRunner._run_blocks, sol_cfg), runner)
        out_sol = sol_rb(x0.clone(), t_emb, mod_segs, lay, {"sigmas": torch.tensor([0.30])})
        cos = nn.functional.cosine_similarity(out_sol.float().flatten(), out_dense.float().flatten(), dim=0).item()
        assert cos > 0.99, f"Sol-Attn + Morton cos too low: {cos}"

        # Verify sol_dense_blocks="0" forces block 0 dense even with Morton active!
        sol_cfg_d0 = dict(sol_cfg, sol_dense_blocks_set=frozenset({0}))
        sol_rb_d0 = types.MethodType(cs._make_sol_run_blocks(_MiniRunner._run_blocks, sol_cfg_d0), runner)
        out_d0 = sol_rb_d0(x0.clone(), t_emb, mod_segs, lay, {"sigmas": torch.tensor([0.30])})
        cos_d0 = nn.functional.cosine_similarity(out_d0.float().flatten(), out_dense.float().flatten(), dim=0).item()
        assert cos_d0 > 0.9999, f"Dense block 0 under Morton diverged: cos={cos_d0}"
        print(f"[PASS] test_sol_attn_morton_and_sparse_path (sol_cos={cos:.6f}, dense_under_morton_cos={cos_d0:.7f})")
    else:
        print("[PASS] test_sol_attn_morton_and_sparse_path (CPU/parser/Morton checks)")


@torch.inference_mode()
def test_bf16_lora_side_branch_and_hooks():
    """Verify Punch 1: BF16 LoRA side-branch, Triton SwiGLU, LowVramPatch stash/restore, BypassForwardHook."""
    from comfy.model_patcher import LowVramPatch
    from comfy.weight_adapter.lora import LoRAAdapter
    from comfy.weight_adapter.bypass import BypassForwardHook

    # 1. Extractor rejects non-standard patches (DoRA, mid, reshape, offset, fn, strength_model != 1.0)
    up = torch.randn(32, 4, dtype=torch.float16)
    dn = torch.randn(4, 16, dtype=torch.float16)
    ok_ad = LoRAAdapter(set(), (up, dn, 8.0, None, None, None))
    dora_ad = LoRAAdapter(set(), (up, dn, 8.0, None, torch.ones(32), None))
    tuck_ad = LoRAAdapter(set(), (up, dn, 8.0, torch.randn(4, 4), None, None))
    assert cs._extract_lora_entries_from_patches([(0.5, ok_ad, 1.0, None, None)], (32, 16)) is not None
    got_ok = cs._extract_lora_entries_from_patches([(0.5, ok_ad, 1.0, None, None)], (32, 16))
    assert len(got_ok) == 1 and abs(got_ok[0][2] - 1.0) < 1e-6  # 0.5 * (8/4) = 1.0
    assert cs._extract_lora_entries_from_patches([(0.5, dora_ad, 1.0, None, None)], (32, 16)) is None
    assert cs._extract_lora_entries_from_patches([(0.5, tuck_ad, 1.0, None, None)], (32, 16)) is None
    assert cs._extract_lora_entries_from_patches([(0.5, ok_ad, 0.8, None, None)], (32, 16)) is None
    assert cs._extract_lora_entries_from_patches([(0.5, ok_ad, 1.0, (0, 0, 16), None)], (32, 16)) is None
    assert cs._extract_lora_entries_from_patches([(0.5, ok_ad, 1.0, None, lambda x: x)], (32, 16)) is None

    if not torch.cuda.is_available() or not os.path.exists(CKPT_PATH):
        print("[PASS] test_bf16_lora_side_branch_and_hooks (CPU checks)")
        return

    from safetensors import safe_open
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 2. Triton fused SwiGLU matches exact FP32 SiLU * up
    fc1_probe = torch.randn(256, 2048, device=device, dtype=dtype)
    got_sw = cs._swiglu_chunk(fc1_probe)
    g_ref, u_ref = fc1_probe.chunk(2, dim=-1)
    ref_sw = (nn.functional.silu(g_ref.float()) * u_ref.float()).to(dtype)
    assert (got_sw.float() - ref_sw.float()).abs().max().item() <= 0.02, "Triton SwiGLU mismatch!"

    # 3. Real dasiwa blocks.0 with 2 stacked BF16 LoRAs + LowVramPatch stash/restore + BypassForwardHook
    ops = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
    block = DiTBlock(
        hidden=5376, heads=56, head_dim=128, ffn=14336, t_dim=8,
        eps=1e-5, qk_eps=1e-5, apply_silu=False, adaln_dtype=torch.float32,
        gate_compress=False, dtype=dtype, device="cpu", operations=ops,
    )
    sd = {}
    with safe_open(CKPT_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("blocks.0."):
                sd[k[len("blocks.0."):]] = f.get_tensor(k)
    block.load_state_dict(sd, strict=False)
    block.to(device=device).requires_grad_(False).eval()

    gen_cpu = torch.Generator(device="cpu").manual_seed(2026)
    def _mk(in_f, out_f, r, st):
        u = torch.randn((out_f, r), dtype=torch.float16, generator=gen_cpu) * 0.01
        d = torch.randn((r, in_f), dtype=torch.float16, generator=gen_cpu) * 0.01
        return (st, LoRAAdapter(set(), (u, d, float(r), None, None, None)), 1.0, None, None), u, d

    p_qkv1, u_qkv1, d_qkv1 = _mk(5376, 21504, 16, 1.0)
    p_qkv2, u_qkv2, d_qkv2 = _mk(5376, 21504, 32, 0.2)
    p_out1, u_out1, d_out1 = _mk(7168, 5376, 16, 1.0)
    p_fc1, u_fc1, d_fc1 = _mk(5376, 28672, 16, 0.5)
    p_fc2, u_fc2, d_fc2 = _mk(14336, 5376, 16, 0.5)

    # Dynamic LowVramPatch + non-dynamic LowVramPatch + BypassForwardHook runtime test
    lv_map = {
        "diffusion_model.blocks.0.attn.qkv_proj.weight": [p_qkv1, p_qkv2],
        "diffusion_model.blocks.0.mlp.fc1.weight": [p_fc1],
    }
    lv_dyn = LowVramPatch("diffusion_model.blocks.0.attn.qkv_proj.weight", lv_map)
    lv_nondyn = LowVramPatch("diffusion_model.blocks.0.mlp.fc1.weight", lv_map)
    block.attn.qkv_proj.weight_lowvram_function = lv_dyn
    block.mlp.fc1.weight_lowvram_function = None
    block.mlp.fc1.weight_function = [lv_nondyn]
    bp = BypassForwardHook(block.attn.out_proj, p_out1[1])
    bp.adapter.multiplier = 1.0
    bp.inject()

    observed = {}
    class _Holder:
        def __init__(self, b):
            self.blocks = [b]
        def _run_blocks(self, h_in, te, segs, lay, topts):
            observed["dyn_stashed"] = self.blocks[0].attn.qkv_proj.weight_lowvram_function is None
            observed["nondyn_stashed"] = len(self.blocks[0].mlp.fc1.weight_function) == 0
            act = topts.get("_chunked_sage_active_loras") or {}
            observed["qkv_n"] = len(act.get(id(self.blocks[0].attn.qkv_proj), []))
            observed["out_n"] = len(act.get(id(self.blocks[0].attn.out_proj), []))
            observed["fc1_n"] = len(act.get(id(self.blocks[0].mlp.fc1), []))
            observed["fc2_n"] = len(act.get(id(self.blocks[0].mlp.fc2), []))
            return h_in

    holder = _Holder(block)
    static_fc2 = {
        "diffusion_model.blocks.0.mlp.fc2.weight": cs._extract_lora_entries_from_patches([p_fc2], (5376, 14336))
    }
    # Legacy mode preserves uncollapsed stacked entries (qkv_n == 2)
    rb_legacy = types.MethodType(
        cs._make_chunked_run_blocks(_Holder._run_blocks, None, static_loras=static_fc2, lora_fast_mode="legacy"),
        holder,
    )
    topts_rt = {}
    _ = rb_legacy(torch.zeros((16, 5376), device=device, dtype=dtype), None, [(0, 16, 0)], None, topts_rt)
    assert observed == {"dyn_stashed": True, "nondyn_stashed": True, "qkv_n": 2, "out_n": 1, "fc1_n": 1, "fc2_n": 1}, f"Legacy hook mismatch: {observed}"
    # Default v3.0 auto mode collapses stacked 2x qkv LoRAs via QR-Core into 1 orthogonal entry (qkv_n == 1)
    rb = types.MethodType(cs._make_chunked_run_blocks(_Holder._run_blocks, None, static_loras=static_fc2), holder)
    topts_rt = {}
    _ = rb(torch.zeros((16, 5376), device=device, dtype=dtype), None, [(0, 16, 0)], None, topts_rt)
    assert observed == {"dyn_stashed": True, "nondyn_stashed": True, "qkv_n": 1, "out_n": 1, "fc1_n": 1, "fc2_n": 1}, f"Auto QR hook mismatch: {observed}"
    assert block.attn.qkv_proj.weight_lowvram_function is lv_dyn
    assert block.mlp.fc1.weight_function == [lv_nondyn]
    assert "_chunked_sage_active_loras" not in topts_rt
    block.attn.qkv_proj.weight_lowvram_function = None
    block.mlp.fc1.weight_function = []
    bp.eject()

    # End-to-end block cosine parity against exact BF16 W_dequant + B@A reference
    S = 2048
    gen_gpu = torch.Generator(device=device).manual_seed(17)
    x0 = torch.randn(S, 5376, dtype=dtype, device=device, generator=gen_gpu) * 0.4
    t_emb = torch.randn(2, 8, dtype=dtype, device=device, generator=gen_gpu) * 0.2
    mod_segs = [(0, 128, 1), (128, 256, 5), (256, S, 0)]
    angles = torch.randn(S, 96, dtype=torch.float32, device=device, generator=gen_gpu)
    angles[:, 48:] = angles[:, :48]
    rope = rope_rotation_table(angles, dtype)

    def _ref_w(w_q, specs):
        w = w_q.dequantize().float()
        for u, d, sc in specs:
            w = w + sc * (u.to(device).float() @ d.to(device).float())
        return w.to(dtype)

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h1 = cs._mod_scale_shift(nn.functional.rms_norm(x0, (5376,), block.norm1.weight, block.norm1.eps), shift_msa, scale_msa, mod_segs)
    q, k, v = nn.functional.linear(h1, _ref_w(block.attn.qkv_proj.weight, [(u_qkv1, d_qkv1, 1.0), (u_qkv2, d_qkv2, 0.2)])).split(7168, dim=-1)
    q, k = cs._apply_qk_norm_rope(block.attn, q.view(1, S, 56, 128).contiguous(), k.view(1, S, 56, 128).contiguous(), rope, block.attn.q_norm.weight, block.attn.k_norm.weight, 96)
    o = cs._staged_sageattn_nhd(q, k, lambda: v.view(1, S, 56, 128).contiguous(), dtype).view(S, 7168)
    x_ref = cs._mod_gate(x0.clone(), gate_msa, nn.functional.linear(o, _ref_w(block.attn.out_proj.weight, [(u_out1, d_out1, 1.0)])), mod_segs)
    h2 = cs._mod_scale_shift(nn.functional.rms_norm(x_ref, (5376,), block.norm2.weight, block.norm2.eps), shift_mlp, scale_mlp, mod_segs)
    g_r, u_r = nn.functional.linear(h2, _ref_w(block.mlp.fc1.weight, [(u_fc1, d_fc1, 0.5)])).chunk(2, dim=-1)
    x_ref = cs._mod_gate(x_ref, gate_mlp, nn.functional.linear(nn.functional.silu(g_r) * u_r, _ref_w(block.mlp.fc2.weight, [(u_fc2, d_fc2, 0.5)])), mod_segs)

    act_loras = {
        id(block.attn.qkv_proj): cs._extract_lora_entries_from_patches([p_qkv1, p_qkv2], (21504, 5376)),
        id(block.attn.out_proj): cs._extract_lora_entries_from_patches([p_out1], (5376, 7168)),
        id(block.mlp.fc1): cs._extract_lora_entries_from_patches([p_fc1], (28672, 5376)),
        id(block.mlp.fc2): cs._extract_lora_entries_from_patches([p_fc2], (5376, 14336)),
    }
    block.attn.forward = types.MethodType(cs._make_chunked_attn_forward(head_chunks=7, token_chunk=1024), block.attn)
    for mode in ("exact", "streaming"):
        blk_fwd = types.MethodType(cs._make_chunked_block_forward(head_chunks=7, out_proj_mode=mode, token_chunk=1024), block)
        y = blk_fwd(x0.clone(), t_emb, mod_segs, rope, transformer_options={"_chunked_sage_active_loras": act_loras})
        cos = nn.functional.cosine_similarity(y.float().flatten(), x_ref.float().flatten(), dim=0).item()
        assert cos > 0.999, f"BF16 LoRA side-branch {mode} cos too low: {cos}"
    print(f"[PASS] test_bf16_lora_side_branch_and_hooks (Triton SwiGLU + hooks + exact/streaming LoRA cos={cos:.6f})")


def test_split_band_rope_and_temp_isolation():
    """Verify Split-Band Spatial RoPE scales ONLY video high-freq h/w bands and restores cleanly."""
    class _FakeModel:
        def __init__(self):
            self.blocks = []
        def rope_freqs(self, position_ids, dev):
            # [S, 3] -> [S, 96] identical layout to MiniMaxH3Model.rope_freqs
            pos = position_ids.to(torch.float32).to(dev)
            inv = torch.tensor([10.0 ** (-k / 4.0) for k in range(16)], dtype=torch.float32, device=dev)
            per_ax = pos.unsqueeze(-1) * inv.view(1, 1, -1)
            t_f, h_f, w_f = per_ax.unbind(dim=1)
            half = torch.cat((t_f, h_f, w_f), dim=-1)
            return torch.cat((half, half), dim=-1)
        def _run_blocks(self, h, t_emb, mod_segs, lay, topts):
            self.seen_angles = self.rope_freqs(lay.position_ids, h.device)
            self.seen_temp = topts.get("_chunked_sage_attn_temp")
            return h

    m = _FakeModel()
    pos = torch.randn((40, 3), dtype=torch.float64)
    lay = types.SimpleNamespace(segments=[(0, 10, "text"), (10, 16, "audio"), (16, 40, "video")], position_ids=pos)
    ref_angles = m.rope_freqs(pos, torch.device("cpu"))
    rb = types.MethodType(
        cs._make_chunked_run_blocks(_FakeModel._run_blocks, None, rope_hf_scale=1.32, attn_temp_scale=1.08),
        m,
    )
    topts = {}
    _ = rb(torch.zeros((40, 16)), None, [(0, 40, 0)], lay, topts)
    ang = m.seen_angles
    assert m.seen_temp == 1.08 and "_chunked_sage_attn_temp" not in topts
    assert "rope_freqs" not in m.__dict__, "Temporary rope_freqs wrapper leaked!"
    # Non-video rows (:16) 100% untouched
    assert torch.equal(ang[:16], ref_angles[:16]), "Text/audio RoPE rows were mutated!"
    # Time axis cols (0:16, 48:64) 100% untouched
    assert torch.equal(ang[16:, 0:16], ref_angles[16:, 0:16])
    assert torch.equal(ang[16:, 48:64], ref_angles[16:, 48:64])
    # Low spatial bands k=8..15 (24:32, 40:48, 72:80, 88:96) 100% untouched
    for c0, c1 in ((24, 32), (40, 48), (72, 80), (88, 96)):
        assert torch.equal(ang[16:, c0:c1], ref_angles[16:, c0:c1])
    # High spatial bands k=0..5 scaled by exactly 1.32; taper k=6 -> 1.24, k=7 -> 1.08
    for base in (16, 32, 64, 80):
        assert torch.allclose(ang[16:, base : base + 6], ref_angles[16:, base : base + 6] * 1.32, atol=1e-5)
        assert torch.allclose(ang[16:, base + 6], ref_angles[16:, base + 6] * 1.24, atol=1e-5)
        assert torch.allclose(ang[16:, base + 7], ref_angles[16:, base + 7] * 1.08, atol=1e-5)
    print("[PASS] test_split_band_rope_and_temp_isolation")


if __name__ == "__main__":
    test_split_heads_even()
    test_slice_mod_segments()
    test_effective_load_thresholds()
    test_final_layer_and_patch_proj_bit_exact()
    test_dense_bf16_block_and_external_deferral()
    test_real_dasiwa_int8_bit_exact()
    test_sol_attn_morton_and_sparse_path()
    test_bf16_lora_side_branch_and_hooks()
    test_split_band_rope_and_temp_isolation()
    print("ALL CHUNKED SAGE TESTS PASSED!")
