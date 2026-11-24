# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""VRAM-safe temporal-chunked AV decoder for MiniMax H3 + Sol-H3-VAE (FP16).

Sol-H3-VAE (`NonCausalViT3DDecoder`) decodes an entire clip at once and its
`decode_tiled` ignores tiling (`vae.py:989`). On 8GB GPUs, 10s @ 1080p
(72x70x124 = 625k tokens) fits single-shot, whereas 15s @ 1080p (107x68x120 =
873k tokens, estimated 8.46 GiB) OOMs and crashes on fallback.

Because `NonCausalViT3DDecoder` uses local 5-latent (17-frame) shifted temporal
windows, slicing the video latent on exact multiples of 5 latents with a 3-block
(15-latent / 51-frame) overlap and linear crossfade across the center guard block
keeps peak VRAM < 6.4 GiB (0 OOM) with visually seamless temporal continuity and
zero edits to ComfyUI core.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from comfy_api.latest import io

from .bridge import find_upstream_module

CATEGORY = "MiniMax H3 Hard Cut"

_SOL_BLOCK_LATENTS = 5
_SOL_FRAMES_PER_BLOCK = 17
_SOL_PATCH_T = 4
_SOL_FRAMES_TO_TRIM = 3
DEFAULT_MAX_SOL_TOKENS = 630_000


@dataclass(frozen=True)
class SolVAEChunk:
    """One 5-latent-aligned temporal decode chunk and its global frame mapping."""

    index: int
    t0: int
    t1: int
    f_start: int
    f_end: int
    blend_start: int
    blend_end: int


def sol_frames_for_latents(
    t_lat: int,
    *,
    block_size: int = _SOL_BLOCK_LATENTS,
    frames_per_block: int = _SOL_FRAMES_PER_BLOCK,
    patch_t: int = _SOL_PATCH_T,
    trim: int = _SOL_FRAMES_TO_TRIM,
) -> int:
    """Exact frame count produced by `NonCausalViT3DDecoder.iter_decode` for `t_lat` latents."""
    if t_lat <= 0:
        return 0
    if t_lat == 1:
        return 1
    full_blocks, rem = divmod(int(t_lat), int(block_size))
    tail = max(0, rem * int(patch_t) - int(trim)) if rem > 0 else 0
    return full_blocks * int(frames_per_block) + tail


def plan_sol_vae_chunks(
    t_lat: int,
    h_lat: int,
    w_lat: int,
    *,
    max_tokens: int = DEFAULT_MAX_SOL_TOKENS,
    max_chunk_latents: int = 0,
    overlap_blocks: int = 3,
    block_size: int = _SOL_BLOCK_LATENTS,
    frames_per_block: int = _SOL_FRAMES_PER_BLOCK,
    patch_t: int = _SOL_PATCH_T,
    trim: int = _SOL_FRAMES_TO_TRIM,
) -> list[SolVAEChunk]:
    """Plan 5-latent-aligned overlapping chunks for `NonCausalViT3DDecoder`.

    Returns a single chunk covering `[0, t_lat)` whenever the full clip fits within
    `max_tokens` (and `max_chunk_latents == 0`), adding zero overhead on 10s clips.
    """
    t_lat = int(t_lat)
    h_lat = max(1, int(h_lat))
    w_lat = max(1, int(w_lat))
    total_frames = sol_frames_for_latents(
        t_lat,
        block_size=block_size,
        frames_per_block=frames_per_block,
        patch_t=patch_t,
        trim=trim,
    )
    total_tokens = t_lat * h_lat * w_lat

    if max_chunk_latents <= 0 and total_tokens <= int(max_tokens):
        return [SolVAEChunk(0, 0, t_lat, 0, total_frames, 0, 0)]

    ov_blocks = max(1, int(overlap_blocks))
    if max_chunk_latents > 0:
        max_blocks = max(ov_blocks + 2, int(max_chunk_latents) // block_size)
    else:
        # Target ~82% of single-shot token ceiling so each chunk keeps generous 8GB headroom.
        target_tokens = max(block_size * h_lat * w_lat, int(int(max_tokens) * 0.82))
        max_blocks = max(ov_blocks + 2, (target_tokens // (h_lat * w_lat)) // block_size)

    max_lat = max_blocks * block_size
    if t_lat <= max_lat + (block_size - 1):
        return [SolVAEChunk(0, 0, t_lat, 0, total_frames, 0, 0)]

    stride_blocks = max(1, max_blocks - ov_blocks)
    stride_lat = stride_blocks * block_size
    # Blend across the center guard block of the overlap (1 full block inside both edges when ov_blocks>=3).
    guard_blocks = ov_blocks // 2

    chunks: list[SolVAEChunk] = []
    b0 = 0
    idx = 0
    while True:
        t0 = b0 * block_size
        remaining = t_lat - t0
        if remaining <= max_lat + (block_size - 1):
            t1 = t_lat
            is_last = True
        else:
            t1 = t0 + max_lat
            is_last = False

        f_start = b0 * frames_per_block
        f_len = sol_frames_for_latents(
            t1 - t0,
            block_size=block_size,
            frames_per_block=frames_per_block,
            patch_t=patch_t,
            trim=trim,
        )
        f_end = f_start + f_len

        if idx == 0:
            blend_start = 0
            blend_end = 0
        else:
            prev = chunks[-1]
            ov_start = f_start
            ov_end = prev.f_end
            # Center block inside [ov_start, ov_end)
            b_s = ov_start + guard_blocks * frames_per_block
            b_e = min(ov_end, b_s + frames_per_block)
            if b_e <= b_s:
                b_s = ov_start
                b_e = min(ov_end, ov_start + frames_per_block)
            blend_start = b_s
            blend_end = b_e

        chunks.append(SolVAEChunk(idx, t0, t1, f_start, f_end, blend_start, blend_end))
        if is_last:
            break
        b0 += stride_blocks
        idx += 1

    return chunks


def decode_video_sol_safe(
    video_vae,
    video_latent: torch.Tensor,
    *,
    max_sol_tokens: int = DEFAULT_MAX_SOL_TOKENS,
    max_chunk_latents: int = 0,
    overlap_blocks: int = 3,
    soft_empty_cache_fn=None,
) -> torch.Tensor:
    """Decode `[B, 24, T, H, W]` video latent safely, chunking `Sol-H3-VAE` when over budget.

    Returns ComfyUI IMAGE tensor `[F, H_pix, W_pix, 3]` FP32 in `[0, 1]` on CPU.
    """
    fsm = getattr(video_vae, "first_stage_model", None)
    is_noncausal = bool(getattr(fsm, "noncausal", False))
    if (
        not is_noncausal
        or video_latent.ndim != 5
        or video_latent.shape[0] != 1
        or video_latent.shape[2] <= _SOL_BLOCK_LATENTS * 2
    ):
        images = video_vae.decode(video_latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images

    _, _, t_lat, h_lat, w_lat = (int(x) for x in video_latent.shape)
    chunks = plan_sol_vae_chunks(
        t_lat,
        h_lat,
        w_lat,
        max_tokens=max_sol_tokens,
        max_chunk_latents=max_chunk_latents,
        overlap_blocks=overlap_blocks,
    )
    if len(chunks) == 1:
        images = video_vae.decode(video_latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images

    total_frames = chunks[-1].f_end
    t_start = time.perf_counter()
    print(
        f"[SeamKit] Sol-H3-VAE 安全分块解码: T_lat={t_lat} ({h_lat}×{w_lat}, {t_lat * h_lat * w_lat:,} tok) "
        f"-> {len(chunks)} 段 (总计 {total_frames}f, 重叠 {overlap_blocks} 块={overlap_blocks * _SOL_FRAMES_PER_BLOCK}f)",
        flush=True,
    )

    out: torch.Tensor | None = None
    for ch in chunks:
        c_t0 = time.perf_counter()
        sub_z = video_latent[:, :, ch.t0 : ch.t1].contiguous()
        part = video_vae.decode(sub_z)
        if part.ndim == 5:
            part = part.reshape(-1, *part.shape[-3:])
        part = part.to(device="cpu", dtype=torch.float32)

        expected_f = ch.f_end - ch.f_start
        if part.shape[0] != expected_f:
            raise RuntimeError(
                f"Sol-H3-VAE chunk {ch.index} [{ch.t0}:{ch.t1}] decoded {part.shape[0]} frames, "
                f"expected {expected_f}"
            )

        if out is None:
            out = torch.empty(
                (total_frames, part.shape[1], part.shape[2], part.shape[3]),
                dtype=torch.float32,
                device="cpu",
            )

        if ch.index == 0:
            out[ch.f_start : ch.f_end].copy_(part)
        else:
            loc_b0 = ch.blend_start - ch.f_start
            loc_b1 = ch.blend_end - ch.f_start
            n_blend = ch.blend_end - ch.blend_start
            for k in range(n_blend):
                w = (k + 0.5) / float(n_blend)
                out[ch.blend_start + k].mul_(1.0 - w).add_(part[loc_b0 + k], alpha=w)
            out[ch.blend_end : ch.f_end].copy_(part[loc_b1:])

        c_dt = time.perf_counter() - c_t0
        blend_str = f", 融合 [{ch.blend_start}..{ch.blend_end - 1}]" if ch.index > 0 else ""
        print(
            f"[SeamKit]   Sol-VAE 段 {ch.index + 1}/{len(chunks)}: lat[{ch.t0}:{ch.t1}] -> "
            f"f[{ch.f_start}..{ch.f_end - 1}] ({expected_f}f{blend_str}) | {c_dt:.1f}s",
            flush=True,
        )
        del part, sub_z
        if soft_empty_cache_fn is not None:
            soft_empty_cache_fn()

    assert out is not None
    print(
        f"[SeamKit] Sol-H3-VAE 安全分块解码完成: {total_frames}f @ {out.shape[2]}×{out.shape[1]} "
        f"| 总耗时 {time.perf_counter() - t_start:.1f}s",
        flush=True,
    )
    return out


class MiniMaxH3SafeAVDecode(io.ComfyNode):
    """Drop-in replacement for `MiniMaxH3AVDecode` with 8GB-safe chunked `Sol-H3-VAE` decode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SafeAVDecode",
            display_name="MiniMax H3 Safe AV Decode (Sol-VAE 8GB)",
            category=CATEGORY,
            description=(
                "MiniMax H3 音视频安全解码器（兼容原版 INT8 VAE 与 Sol-H3-VAE FP16）：\n"
                "• 10s 1080p 及以下直接原速单遍解码（零额外开销）；\n"
                "• 15s/20s 1080p 使用 Sol-H3-VAE 时自动按 5-latent 块网格分 2~3 段解码，"
                "在重叠中段保护块平滑过渡，8GB 显卡彻底告别 Sol-VAE OOM 崩溃。"
            ),
            inputs=[
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Latent.Input("av_latent"),
                io.Int.Input(
                    "max_sol_tokens",
                    default=DEFAULT_MAX_SOL_TOKENS,
                    min=200_000,
                    max=2_000_000,
                    step=10_000,
                    advanced=True,
                    tooltip="Sol-H3-VAE 单遍解码最大 token 阈值（默认 630,000：10s 1080p 单遍直出，15s 1080p 自动分 2 段安全解码）。",
                ),
                io.Int.Input(
                    "max_chunk_latents",
                    default=0,
                    min=0,
                    max=120,
                    step=5,
                    advanced=True,
                    tooltip="手动指定每段 latent 长度（5 的倍数；0 = 按 max_sol_tokens 自动计算）。",
                ),
                io.Int.Input(
                    "overlap_blocks",
                    default=3,
                    min=1,
                    max=6,
                    step=1,
                    advanced=True,
                    tooltip="相邻分块重叠的 5-latent 块数（默认 3 块 = 15 latent = 51 帧，前后各留 1 块缓冲并在中间块线性过渡）。",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video_vae,
        audio_vae,
        av_latent: dict,
        max_sol_tokens: int = DEFAULT_MAX_SOL_TOKENS,
        max_chunk_latents: int = 0,
        overlap_blocks: int = 3,
    ) -> io.NodeOutput:
        import comfy.model_management as mm
        from comfy_extras.nodes_audio import vae_decode_audio

        core_mod = find_upstream_module("core", require_substring="minimax-h3-audio-T8")
        if core_mod is None or not hasattr(core_mod, "nested_av_parts"):
            raise RuntimeError("upstream comfyui-minimax-h3-audio-T8 core.nested_av_parts not found")

        video, audio = core_mod.nested_av_parts(av_latent)
        images = decode_video_sol_safe(
            video_vae,
            video,
            max_sol_tokens=int(max_sol_tokens),
            max_chunk_latents=int(max_chunk_latents),
            overlap_blocks=int(overlap_blocks),
            soft_empty_cache_fn=mm.soft_empty_cache,
        )
        decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})

        video_latent = {k: v for k, v in av_latent.items() if k not in {"samples", "noise_mask"}}
        audio_latent = video_latent.copy()
        video_latent["samples"] = video
        audio_latent["samples"] = audio
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            video_mask, audio_mask = masks.unbind()
            video_latent["noise_mask"] = video_mask
            audio_latent["noise_mask"] = audio_mask

        return io.NodeOutput(images, decoded_audio, video_latent, audio_latent)
