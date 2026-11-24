"""Unit tests for `sol_vae_decode.py` and Tight-Sol low-sigma routing in `chunked_sage.py`."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = Path(__file__).resolve().parents[4]
for p in (str(COMFY_ROOT), str(ROOT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load package module cleanly without requiring ComfyUI full server boot
_pkg_name = "comfyui_h3_seamkit_testpkg"
if _pkg_name not in sys.modules:
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(ROOT)]
    sys.modules[_pkg_name] = _pkg

import importlib  # noqa: E402

sol_vae_decode = importlib.import_module(f"{_pkg_name}.sol_vae_decode")
chunked_sage = importlib.import_module(f"{_pkg_name}.chunked_sage")


def test_frame_math():
    assert sol_vae_decode.sol_frames_for_latents(1) == 1
    assert sol_vae_decode.sol_frames_for_latents(5) == 17
    assert sol_vae_decode.sol_frames_for_latents(60) == 204
    assert sol_vae_decode.sol_frames_for_latents(62) == 209
    assert sol_vae_decode.sol_frames_for_latents(72) == 243
    assert sol_vae_decode.sol_frames_for_latents(107) == 362
    assert sol_vae_decode.sol_frames_for_latents(142) == 481
    print("[PASS] test_frame_math")


def test_chunk_plans():
    # 10s @ 1984x1120 (72 x 70 x 124 = 624,960 <= 630,000) -> single shot
    c10 = sol_vae_decode.plan_sol_vae_chunks(72, 70, 124)
    assert len(c10) == 1
    assert (c10[0].t0, c10[0].t1, c10[0].f_start, c10[0].f_end) == (0, 72, 0, 243)

    # 15s @ 1568x864 (107 x 54 x 98 = 566,244 <= 630,000) -> single shot
    c15_864 = sol_vae_decode.plan_sol_vae_chunks(107, 54, 98)
    assert len(c15_864) == 1

    # 15s @ 1920x1088 (107 x 68 x 120 = 873,120 > 630,000) -> 2 exact 5-latent-aligned chunks
    c15 = sol_vae_decode.plan_sol_vae_chunks(107, 68, 120)
    assert len(c15) == 2, f"expected 2 chunks, got {c15}"
    assert (c15[0].t0, c15[0].t1, c15[0].f_start, c15[0].f_end) == (0, 60, 0, 204)
    assert (c15[1].t0, c15[1].t1, c15[1].f_start, c15[1].f_end) == (45, 107, 153, 362)
    assert (c15[1].blend_start, c15[1].blend_end) == (170, 187)
    assert c15[1].t0 % 5 == 0
    assert (c15[0].t1 - c15[0].t0) * 68 * 120 <= 520_000
    assert (c15[1].t1 - c15[1].t0) * 68 * 120 <= 520_000

    # 20s @ 1920x1088 (142 x 68 x 120) -> 3 aligned chunks covering 481 frames
    c20 = sol_vae_decode.plan_sol_vae_chunks(142, 68, 120)
    assert len(c20) == 3, f"expected 3 chunks, got {c20}"
    assert c20[0].f_start == 0 and c20[-1].f_end == 481
    for ch in c20:
        assert ch.t0 % 5 == 0
    print(f"[PASS] test_chunk_plans (15s={c15}, 20s={len(c20)} chunks)")


class _MockNonCausalVAE:
    """Mock Sol-H3-VAE where frame pixel value encodes global latent block index."""

    def __init__(self, shift_per_chunk: float = 0.0):
        self.first_stage_model = types.SimpleNamespace(noncausal=True)
        self.calls: list[tuple[int, int, int]] = []
        self.shift_per_chunk = shift_per_chunk

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        _, _, t_lat, h_lat, w_lat = z.shape
        self.calls.append((int(t_lat), int(h_lat), int(w_lat)))
        n_frames = sol_vae_decode.sol_frames_for_latents(int(t_lat))
        # Read frame id stored in channel 0 of z (each 5 latents -> 17 frames)
        vals = torch.zeros((n_frames,), dtype=torch.float32)
        f = 0
        for b_start in range(0, int(t_lat), 5):
            b_len = min(5, int(t_lat) - b_start)
            bf = sol_vae_decode.sol_frames_for_latents(b_len)
            base_val = float(z[0, 0, b_start, 0, 0].item())
            vals[f : f + bf] = base_val + self.shift_per_chunk * (len(self.calls) - 1)
            f += bf
        return vals.view(1, n_frames, 1, 1, 1).expand(1, n_frames, 4, 4, 3).clone()


def test_chunked_decode_exact_and_blend():
    # Encode block index b into channel 0 at every block start
    z = torch.zeros((1, 24, 107, 68, 120), dtype=torch.float32)
    for b in range(22):
        if b * 5 < 107:
            z[0, 0, b * 5, :, :] = float(b)

    vae_exact = _MockNonCausalVAE(shift_per_chunk=0.0)
    out = sol_vae_decode.decode_video_sol_safe(vae_exact, z)
    assert out.shape == (362, 4, 4, 3)
    assert len(vae_exact.calls) == 2
    assert vae_exact.calls[0][0] == 60 and vae_exact.calls[1][0] == 62
    for b in range(21):
        assert torch.allclose(out[b * 17 : (b + 1) * 17], torch.full((17, 4, 4, 3), float(b)))
    assert torch.allclose(out[357:362], torch.full((5, 4, 4, 3), 21.0))

    # Now introduce a +1.0 chunk shift to verify exact midpoint linear crossfade on [170..187)
    vae_shift = _MockNonCausalVAE(shift_per_chunk=1.0)
    out_b = sol_vae_decode.decode_video_sol_safe(vae_shift, z)
    assert float(out_b[169, 0, 0, 0]) == 9.0
    assert float(out_b[187, 0, 0, 0]) == 12.0  # block 11 + 1.0 shift
    for k in range(17):
        expected = 10.0 + (k + 0.5) / 17.0
        assert abs(float(out_b[170 + k, 0, 0, 0]) - expected) < 1e-5
    print("[PASS] test_chunked_decode_exact_and_blend")


def test_sol_low_sigma_tau_routing():
    if not chunked_sage._sol_attn_available:
        print("[SKIP] test_sol_low_sigma_tau_routing (sol_attn unavailable)")
        return
    layout = types.SimpleNamespace(
        segments=[(0, 512, "cond"), (512, 512 + 16384, "video")],
        signature=(512, 16, 64, 64, 0),
        position_ids=torch.zeros((512 + 16384, 3)),
    )
    base_cfg = {
        "attention_mode": "sol_attn_sparse",
        "sol_tau": 1.3,
        "sol_token_aug": 64,
        "sol_topk_ratio": 0.0,
        "sol_sink_mode": "exact_kv_and_rows",
        "sol_min_tokens": 12288,
        "sol_dense_blocks_set": frozenset(),
        "sol_low_sigma_dense_blocks_set": frozenset({0, 1, 2, 47, 48, 49}),
        "sol_max_sigma": 0.95,
        "sol_min_sigma": 0.25,
    }
    # Mode 1: sol_low_sigma_tau == 0.0 -> sigma=0.18 returns None (100% Dense KJ Sage)
    cfg_dense = dict(base_cfg, sol_low_sigma_tau=0.0)
    ctx_d, _ = chunked_sage._build_sol_context(layout, 512 + 16384, cfg_dense, {"sigmas": torch.tensor([0.18])})
    assert ctx_d is None

    # Mode 2: sol_low_sigma_tau == 0.90 -> sigma=0.18 returns Tight-Sol (tau=0.90, aug>=128, 6 dense blocks)
    cfg_tight = dict(base_cfg, sol_low_sigma_tau=0.90)
    ctx_t, _ = chunked_sage._build_sol_context(layout, 512 + 16384, cfg_tight, {"sigmas": torch.tensor([0.18])})
    assert ctx_t is not None
    assert abs(ctx_t["tau"] - 0.90) < 1e-6
    assert ctx_t["token_aug"] == 128
    assert ctx_t["dense_blocks"] == frozenset({0, 1, 2, 47, 48, 49})

    # Normal high-sigma step (sigma=0.50) still uses regular sol_tau=1.3, aug=64, empty dense_blocks
    ctx_hi, _ = chunked_sage._build_sol_context(layout, 512 + 16384, cfg_tight, {"sigmas": torch.tensor([0.50])})
    assert ctx_hi is not None
    assert abs(ctx_hi["tau"] - 1.3) < 1e-6
    assert ctx_hi["token_aug"] == 64
    assert ctx_hi["dense_blocks"] == frozenset()
    print("[PASS] test_sol_low_sigma_tau_routing")


if __name__ == "__main__":
    test_frame_math()
    test_chunk_plans()
    test_chunked_decode_exact_and_blend()
    test_sol_low_sigma_tau_routing()
    print("ALL SOL VAE DECODE & TIGHT-SOL TESTS PASSED!")
