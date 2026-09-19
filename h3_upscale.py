"""Minimal MiniMax H3 two-pass upscale executor: hard cut + UNEQUAL window lengths.

We reuse the verified "atomic" pieces from the upstream pack (the learned 3D
upscaler, the sampler rebind, the per-window conditioning re-anchor, the append)
by resolving their modules at runtime — those pieces are exactly what must stay
byte-compatible with H3 and are pointless to fork.

What is OURS is the windowing layer: when the plan carries
`hardcut.segment_frames` (explicit cut frames), it splits there, so window
lengths can differ (e.g. 70 frames then 130). The equal-stride path is kept as a
fallback. Upstream upgrades can never touch this, because we do not import the
upstream *executor*, only its learned-upscaler / sampler / conditioning helpers.
"""

from __future__ import annotations

import json

import torch

import comfy.nested_tensor

from comfy_api.latest import io

from .bridge import PLAN_TYPE_STRING, find_upstream_module
from .hardcut_math import FPS, FRAME_GRID

CATEGORY = "MiniMax H3/HardCut"
PLAN_TYPE = io.Custom(PLAN_TYPE_STRING)

PLAN_SCHEMA_LOW_SIGMA_V3 = "t8.minimax_h3.chunked_two_pass.low_sigma.v3"

# ComfyUI loads custom-node modules with absolute-path names, so upstream cannot
# be imported with a fixed statement; resolve it lazily and cache the objects.
_upstream_cache: dict = {}


def _upstream(suffix: str, *, require: str | None = None):
    key = (suffix, require)
    module = _upstream_cache.get(key)
    if module is None:
        module = find_upstream_module(suffix, require_substring=require)
        if module is None:
            raise RuntimeError(
                f"upstream H3 upscale pack not loaded in this process ({suffix}); "
                "load comfyui-minimax-h3-audio before running"
            )
        _upstream_cache[key] = module
    return module


def _core():
    return _upstream("chunked_two_pass_upscale_advanced")


def _learned():
    return _upstream("learned_latent_upscale_advanced")


def _sampling():
    return _upstream("sampling", require="minimax-h3-audio-T8")


def _snap_boundary(frame: int, video_tokens: int, step: int = 1):
    """Frame -> (token, frame) at the nearest token edge, `step` tokens apart.

    A window can only start on a token edge, but NOT only every 5th one: the
    token->frame map is `[1,4,4,4,4]` repeating, so edges land on
    0,1,5,9,13,17,18,22,... — i.e. every token is a legal boundary and the real
    resolution is 1-4 frames.  `step=5` reproduces the old 17-frame grid (kept
    for callers that want it); `step=1` uses every edge.
    """
    core = _core()
    choices = [
        (token, core.frames_for_tokens(token))
        for token in range(0, int(video_tokens) + 1, max(1, int(step)))
    ]
    return min(choices, key=lambda item: abs(item[1] - int(frame)))


def explicit_segments(
    video_tokens: int, total_frames: int, cut_frames, token_step: int = 1
) -> list:
    """Explicit cut-frame list -> UNEQUAL window bounds.

    Each boundary is snapped to the nearest TOKEN edge (`token_step=1` = finest,
    1-4 frame resolution; `token_step=5` = the coarse 17-frame grid).
    `cut_frames` is in frames; 0 and the clip end are implied and must not be
    listed.

    Every cut is validated against the grid *by the number the caller wrote*,
    before any bounds are built, so a rejected cut is always reported with that
    number (rather than with the derived boundary next to it).
    """
    core = _core()
    snapped: list[tuple[int, int, int]] = []
    for raw in cut_frames:
        if raw is None:
            continue
        f = int(raw)
        if f <= 0:
            continue
        if f >= total_frames:
            raise ValueError(
                f"cut frame {f} is not inside the clip (0, {total_frames})"
            )
        token, frame = _snap_boundary(f, video_tokens, token_step)
        if token <= 0:
            raise ValueError(
                f"cut frame {f} snaps back onto the clip start (token 0), which would "
                "leave the first window empty. Keep cuts at least a few frames in — "
                "with token_step=1 the usable edges start at frame 1."
            )
        snapped.append((f, token, frame))

    snapped = sorted(set(snapped))
    for (a, ta, _fa), (b, tb, _fb) in zip(snapped, snapped[1:]):
        if ta == tb:
            raise ValueError(
                f"cuts {a} and {b} snap onto the same 17-frame grid point (token {ta}), "
                "which would leave the window between them empty. Space cuts at least "
                "17 frames apart."
            )

    last_start = max([0] + [frame for _f, _t, frame in snapped])
    tail_frames = int(total_frames) - last_start
    if snapped and tail_frames < FRAME_GRID:
        raise ValueError(
            f"the last cut snaps to frame {last_start}, leaving only a {tail_frames}-frame "
            f"tail ({tail_frames / FPS:.3f}s) — less than one 17-frame block, which the "
            "sampler cannot build a window from. Move the last cut earlier (a tail of "
            "about 2 s is a safe choice)."
        )

    bounds = [0] + [f for f, _t, _fr in snapped] + [int(total_frames)]
    segments = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == 0:
            start_token, start_frame = 0, 0
        else:
            start_token, start_frame = core._snap_frame(lo, video_tokens)
        if hi >= total_frames:  # exact clip end, never snapped away
            end_token, end_frame = video_tokens, total_frames
        else:
            end_token, end_frame = core._snap_frame(hi, video_tokens)
        if end_token <= start_token:  # defensive: every case is caught above
            raise ValueError(
                f"window {i} collapses to {end_token} token(s) (frames {start_frame}.."
                f"{end_frame}); cuts must sit on distinct 17-frame grid points, away "
                "from both the clip start and the clip end."
            )
        segments.append((start_token, start_frame, end_token, end_frame))
    return segments


def _upscale_chunk(learned, chunk_latent, plan):
    upscaled, _audio, _mask, _report = learned.learned_upscale_h3_av_latent(
        chunk_latent,
        plan["model_name"],
        "target_dimensions",
        2.0,
        1.0,
        int(plan["target_width"]),
        int(plan["target_height"]),
        "honor_dimensions_exp",
        2.0,
        plan["precision"],
        plan["release_policy"],
    )
    return upscaled["samples"].tensors[0]


def _sample_fullframe(
    core,
    sampling,
    chunk_video,
    chunk_audio,
    conditioning,
    model,
    noise,
    sampler,
    sigmas,
    negative,
    cfg,
    chunk_noise_video,
    chunk_noise_audio,
):
    """Full-frame, single tile: one sample pass, no tile / overlap / fade."""
    height, width = chunk_video.shape[-2:]
    video_mask = torch.ones(
        (1, 1, 1, height, width), dtype=torch.float32, device=chunk_video.device
    )
    audio_mask = torch.ones_like(chunk_audio)
    piece = {
        "samples": comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
    }
    piece_sampler = sampling.rebind_dual_clock_sampler(model, piece, sampler)
    prepared_noise = None
    if chunk_noise_video is not None:
        prepared_noise = comfy.nested_tensor.NestedTensor(
            (
                chunk_noise_video.contiguous(),
                chunk_noise_audio.to(
                    device=chunk_video.device, dtype=chunk_audio.dtype
                ).contiguous(),
            )
        )
    sampled = core.sample_piece(
        piece,
        conditioning,
        model,
        noise,
        piece_sampler,
        sigmas,
        negative,
        cfg,
        prepared_noise=prepared_noise,
    )
    return sampled.tensors[0]


def _hunt_shot_changes(video, sens: float = 2.0, win: int = 2) -> list:
    """First-pass latent -> the tokens where the model changed shots (no VAE).

    The upscaler never sees the prompt, so where the model actually cut is only
    observable in the latent we were handed.  Diffing the token axis and scoring
    each token against its own neighbourhood finds a cut even when the scene is
    busy — a fight moves every token, so a global threshold is useless.
    Returns a list of `(token_index, ratio)`; the change sits between that token
    and the next one.
    """
    v = video.detach().float()
    if v.ndim != 5 or v.shape[2] < 3:
        return []
    d = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
    n = int(d.numel())
    hits = []
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        med = float(d[lo:hi].median())
        if med <= 0:
            continue
        ratio = float(d[i]) / med
        if ratio >= sens:
            hits.append((i, ratio))
    # Keep the strongest when neighbours fire together (a cut smears over 1-2 tokens)
    hits.sort(key=lambda x: -x[1])
    kept = []
    for i, r in hits:
        if all(abs(i - j) > 1 for j, _ in kept):
            kept.append((i, r))
    return sorted(kept)


def execute(
    model,
    conditioning,
    latent,
    noise,
    sampler,
    sigmas,
    plan,
    negative=None,
    cfg: float = 1.0,
    auto_seam_hunt: bool = False,
    auto_seam_sensitivity: int = 20,
):
    core = _core()
    learned = _learned()
    sampling = _sampling()

    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA_LOW_SIGMA_V3:
        raise ValueError(
            "plan must be a low-sigma v3 plan (wire it from MiniMaxH3HardCutPlan)"
        )

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("expected a nested MiniMax H3 AV latent")
    video, audio = samples.tensors
    if (
        video.ndim != 5
        or video.shape[1] != 24
        or audio.ndim != 4
        or audio.shape[1:3] != (32, 2)
    ):
        raise ValueError("unexpected MiniMax H3 AV latent shapes")
    if video.shape[0] != 1:
        raise ValueError("chunked two-pass currently supports batch 1")

    frame_count = core.frames_for_tokens(int(video.shape[2]))
    (
        global_video_noise,
        global_audio_noise,
        _noise_report,
    ) = core._build_global_target_av_noise(noise, latent, video, audio, plan)

    # ---- optional: put the window boundary just BEFORE the model's own cut ----
    # The upscaler never sees the prompt, so the model's shot change is only
    # observable in the latent we are holding.  Measuring it here costs no VAE
    # decode, and placing the split before it keeps the seam on continuous
    # content — the model performs the cut itself, inside the window.
    planned = [int(c) for c in ((plan.get("hardcut") or {}).get("segment_frames") or [])]
    tolerance = max(0, int((plan.get("hardcut") or {}).get("seam_tolerance", 17)))
    segment_frames = planned or None
    seam_hunt = None
    if auto_seam_hunt:
        hits = _hunt_shot_changes(
            video, sens=max(0.5, float(auto_seam_sensitivity) / 10.0)
        )
        tokens, accepted, rejected = [], [], []
        for idx, ratio in hits:
            change_frame = core.frames_for_tokens(int(idx) + 1)  # between idx and idx+1
            # The token edge right before the change - not the old 17-frame grid.
            # token edges run 0,1,5,9,13,17,18,22,... so this lands within 1-4
            # frames of the change instead of up to 8 frames away.
            token = max(1, int(idx))
            if token >= int(video.shape[2]):
                continue
            # Sanity gate: the hunt can fire on a hard action beat.  Only trust it
            # when the change lands within `seam_tolerance` frames of a cut that
            # #56 planned; anything farther is reported and dropped.
            if planned:
                nearest = min(planned, key=lambda c: abs(c - change_frame))
                distance = abs(nearest - change_frame)
                if distance > tolerance:
                    rejected.append({
                        "between_tokens": int(idx),
                        "change_frame": change_frame,
                        "nearest_planned_cut": nearest,
                        "distance": distance,
                        "ratio": round(float(ratio), 2),
                    })
                    continue
            if token not in tokens:
                tokens.append(token)
                accepted.append({
                    "between_tokens": int(idx),
                    "change_frame": change_frame,
                    "boundary_token": token,
                    "boundary_frame": core.frames_for_tokens(token),
                    "ratio": round(float(ratio), 2),
                })
            if len(tokens) >= 9:
                break
        if tokens:
            segment_frames = [core.frames_for_tokens(t) for t in tokens]
        seam_hunt = {
            "sensitivity": round(max(0.5, float(auto_seam_sensitivity) / 10.0), 2),
            "tolerance_frames": tolerance,
            "planned_cuts": planned,
            "accepted": accepted,
            "rejected": rejected,
            "boundary_tokens": tokens,
            "boundary_frames": [core.frames_for_tokens(t) for t in tokens],
        }

    # ---- windowing: explicit (possibly unequal) first, then the equal paths ----
    if segment_frames:
        segments = explicit_segments(
            int(video.shape[2]), frame_count, segment_frames
        )
    elif plan.get("temporal_strategy") == "full_clip_safe":
        segments = [(0, 0, int(video.shape[2]), frame_count)]
    else:
        segments, frame_count = core.compute_temporal_segments(
            int(video.shape[2]),
            int(plan["temporal_chunk_frames"]),
            int(plan["temporal_overlap_frames"]),
        )

    accumulated = None
    segment_reports = []
    for start_token, start_frame, end_token, end_frame in segments:
        chunk_video = video[:, :, start_token:end_token].contiguous()
        audio_start = round(start_frame * core.FRAME_RESCALE)
        audio_end = min(audio.shape[-1], round(end_frame * core.FRAME_RESCALE))
        chunk_audio = audio[..., audio_start:audio_end].contiguous()
        chunk_latent = {
            "samples": comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))
        }

        chunk_video = _upscale_chunk(learned, chunk_latent, plan)
        chunk_conditioning = core.reanchor_conditioning(
            conditioning, start_frame, end_frame, tuple(chunk_video.shape[-2:])
        )
        # Hard cut: windows are independent (zero overlap), so no
        # anchor_conditioning against `accumulated` — upstream only anchors when
        # there is locked overlap, and anchoring at a hard boundary indexes one
        # token past the previous window and raises.

        chunk_noise_video = (
            global_video_noise[:, :, start_token:end_token]
            if global_video_noise is not None
            else None
        )
        chunk_noise_audio = (
            global_audio_noise[..., audio_start:audio_end]
            if global_audio_noise is not None
            else None
        )

        sampled = _sample_fullframe(
            core,
            sampling,
            chunk_video,
            chunk_audio,
            chunk_conditioning,
            model,
            noise,
            sampler,
            sigmas,
            negative,
            cfg,
            chunk_noise_video,
            chunk_noise_audio,
        )
        accumulated = core._append_video(accumulated, sampled, start_token)
        segment_reports.append(
            {
                "index": len(segment_reports),
                "frames": [start_frame, end_frame],
                "tokens": [start_token, end_token],
                "length_frames": end_frame - start_frame,
            }
        )

    output = {"samples": comfy.nested_tensor.NestedTensor((accumulated, audio))}
    report = {
        "schema": "h3.hardcut.upscale.v1",
        "status": "completed",
        "segment_count": len(segments),
        "unequal_lengths": bool(segment_frames),
        "lengths": [r["length_frames"] for r in segment_reports],
        "segments": segment_reports,
    }
    if seam_hunt is not None:
        report["seam_hunt"] = seam_hunt
    return output, json.dumps(report)


class MiniMaxH3HardCutUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutUpscale",
            display_name="MiniMax H3 Hard-Cut Upscale (Two-Pass)",
            description=(
                "Learned latent upscale per temporal window, appended back-to-back "
                "with zero overlap (a hard cut). Supports UNEQUAL window lengths: when "
                "the plan carries `hardcut.segment_frames` (explicit cut frames) it "
                "splits there instead of using an equal stride. Audio passes through. "
                "Full-frame only - spatial tiling, inherited masks and the dual-clock "
                "research paths of the upstream node are intentionally omitted."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                PLAN_TYPE.Input("plan"),
                io.Conditioning.Input("negative", optional=True),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Boolean.Input(
                    "auto_seam_hunt",
                    default=False,
                    tooltip=(
                        "自动找切镜、把窗口边界挪到它前面（**不用 VAE，直接在 latent 上算**）。\n"
                        "二采看不到提示词，模型究竟在哪一帧换镜头只有 latent 知道。\n"
                        "开启后：在时间轴找 latent 的突变（局部邻域法，剧烈动作也不会被淹），\n"
                        "把窗口边界放在突变**之前**最近的 17 帧网格点上 —— 缝就落在内容连续处，"
                        "剪辑由模型自己在窗口内完成。\n"
                        "会覆盖 plan 里的 segment_frames；报告里给出检测到的 token 与最终边界。"
                    ),
                ),
                io.Int.Input(
                    "auto_seam_sensitivity",
                    default=20,
                    min=5,
                    max=80,
                    step=1,
                    tooltip=(
                        "检测灵敏度（×0.1）。某个 token 的变化超过「自己邻域中位数的 N/10 倍」"
                        "才算一次切镜 —— 20 = 2.0 倍。调小＝更敏感（可能多切），调大＝更保守。"
                    ),
                ),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, **kwargs):
        output, report = execute(**kwargs)
        return io.NodeOutput(output, report)
