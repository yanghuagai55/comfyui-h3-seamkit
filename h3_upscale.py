# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
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
import os

import torch
import comfy.model_management

import comfy.nested_tensor

from comfy_api.latest import io

from .bridge import PLAN_TYPE_STRING, find_upstream_module
from .hardcut_math import FPS, FRAME_GRID, LOAD_FAIL

CATEGORY = "MiniMax H3/SeamKit"
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


def find_calm_boundaries(
    profile,
    planned,
    aligned,
    frame_count: int,
    *,
    window: int = 34,
    overlap_frames: int = 17,
    seam_tolerance: int = 17,
    min_sep: int = 34,
    grid: int = FRAME_GRID,
    policy: str = "calm_overlap",
    abstain_below: float = 0.0,
    calm_min_quality: float = 0.0,
    too_quiet_below: float = 0.0,
    calm_min_gain: float = 0.0,
):
    """Decide, per planned cut, WHERE to actually put the window boundary.

    A cut whose hunt entry landed ON the measured turn is trustworthy (the
    boundary sits on the model's own cut) -> hard-cut there, overlap 0.

    Anything else (hunt rejected it, or the 17-frame grid forced the boundary
    away from the turn) means we are about to cut through continuous content, so
    instead of cutting at the plan we look for the CALMEST exclusive frame
    within `window` of it: the token whose
    global latent-change score is lowest, i.e. the least eventful moment.  The
    boundary goes there and that seam gets `overlap_frames` of anchored prefix,
    so the sampler continues from the previous window instead of starting cold.

    Returns (boundaries, overlaps, notes) - one entry per planned cut - where
    `boundaries` are frame numbers and `overlaps` the per-seam overlap in frames
    (0 = hard cut).  Never returns an empty boundary: an unsearchable cut falls
    back to its planned frame.
    """
    # global score per token; the profile rows are (idx, local, global)
    score = {}
    jerk_by_idx = {}
    for row in profile or ():
        try:
            idx, glob = int(row[0]), float(row[2])
            # column 3 is the |d3| (jerk) ratio; rank by the WORSE of the two so
            # a spot that is quiet in value terms but busy in jerk terms (a
            # "drifter": steady motion, high velocity, low jerk - or the reverse)
            # does not win.  Falls back to the first difference alone when an
            # older profile has only three columns.
            jerk = float(row[3]) if len(row) > 3 else glob
        except (TypeError, IndexError, ValueError):
            continue
        score[idx] = max(glob, jerk)
        jerk_by_idx[idx] = jerk
    entry_by_cut = {}
    for e in aligned or ():
        try:
            entry_by_cut[int(e.get("planned_cut"))] = e
        except (TypeError, ValueError):
            continue

    # ---- abstain: a quantile can RANK but cannot say "nothing to do here" ----
    # (idea from MAINodes' Jerk Oracle `abstain_below`).  If the jerk profile has
    # almost no contrast, every candidate is equally unremarkable and moving a
    # boundary buys nothing; keep the plan untouched and say so.
    if abstain_below > 0.0 and jerk_by_idx:
        _vals = [v for v in jerk_by_idx.values() if v > 0]
        if _vals:
            _mean = sum(_vals) / len(_vals)
            _contrast = (max(_vals) / _mean) if _mean > 0 else 1.0
            if _contrast < abstain_below:
                # The clip is too flat to be worth SEARCHING, but the seams
                # still exist and a hard cut through continuous content breaks
                # it.  Same rule as everywhere else: keep the plan, anchor.
                return (
                    [int(c) for c in planned],
                    [int(overlap_frames)] * len(planned),
                    [f"clip too flat for a search (jerk contrast {_contrast:.2f} < "
                     f"{abstain_below:.2f}) -> keep every planned cut, "
                     f"overlap {int(overlap_frames)}f"],
                )

    boundaries, overlaps, notes = [], [], []
    for cut in planned:
        cut = int(cut)
        entry = entry_by_cut.get(cut) or {}
        b = entry.get("boundary_frame")
        # refs to keep clear of: everything already emitted + the cuts still to
        # come (so two seams can never collapse onto the same frame - that used
        # to merge two windows and double the load)
        pending_cuts = [int(c) for c in planned if int(c) != cut]
        refs = boundaries + pending_cuts
        # The deviation that decides "hard cut or anchored overlap" is how far
        # the FINAL boundary had to sit from where the model ACTUALLY turned.
        # A boundary may only sit on an exclusive frame (17k), so this residual
        # is the grid quantisation of the turn: 0 when the turn itself is on the
        # grid, up to 8 when it falls halfway between two legal frames.  A small
        # residual means the seam lands on the model's own cut -> hard cut; a
        # large one means the cut would sit mid-shot, where the two windows
        # render the same content differently -> anchored overlap instead.
        #
        # Do NOT fall back to |boundary - planned|: that is a different quantity
        # (plan drift, a multiple of 17), and comparing it against the same
        # threshold silently turned clean hits into overlaps at random - see the
        # note in _align_to_profile (2026-09-23).
        _m = entry.get("measured_turn_frame")
        if b is not None and _m is not None:
            _dev = abs(int(b) - int(_m))
            _dev_src = f"boundary {b} vs measured turn {_m}"
        else:
            # No measured turn -> we cannot claim the boundary sits on the
            # model's cut, so do not hard-cut.  Fall through to the calm search
            # and let that seam get an anchored overlap.
            _dev, _dev_src = None, None
        if (
            b is not None
            and _dev is not None
            and _dev <= int(seam_tolerance)
            and all(abs(int(b) - int(o)) >= int(min_sep) for o in refs)
        ):
            boundaries.append(int(b))
            overlaps.append(0)
            notes.append(
                f"cut {cut}: hard cut, residual {_dev}f <= {int(seam_tolerance)}f "
                f"({_dev_src})"
            )
            continue

        # candidates: exclusive frames inside the window, away from the others
        # window 0 = keep the boundary where the plan put it and only switch
        # that seam to an anchored overlap (no search at all)
        lo, hi = max(grid, cut - int(window)), min(frame_count - grid, cut + int(window))
        cands = [f for f in range(lo, hi + 1, grid) if f % grid == 0]
        if int(window) == 0 and cut % grid:
            cands = []   # off-grid plan with no room to snap
        cands = [f for f in cands if all(abs(f - int(o)) >= int(min_sep) for o in refs)]
        if not cands:
            # Same rule as the gates: with nothing suitable to move to, do not
            # break the content - keep the planned cut and anchor it.
            boundaries.append(cut)
            overlaps.append(int(overlap_frames))
            notes.append(
                f"cut {cut}: no calm candidate in window -> keep plan, "
                f"overlap {int(overlap_frames)}f"
            )
            continue

        def tok_of(frame):
            # frames here are exclusive anchors: frame 17k starts token 5k
            # (FRAME_PER_TOKEN=(1,4,4,4,4) puts a 1-frame token every 17 frames)
            return int(frame) // int(grid) * 5

        if policy == "jerk_hardcut":
            # Opposite bet to the calm search: put the seam where the picture is
            # ALREADY moving hardest.  Two reasons.  Motion masks a cut, and
            # high jerk is exactly where MAINodes measured the model gives up and
            # smears - so the sharpness step between two windows is smallest
            # between two already-soft frames.  Hard cut, no anchor: continuity
            # is not expected here, concealment is.
            # Score a 3-token window so a single spike does not win over a
            # sustained burst.
            def burst(f, _grid=grid):
                t = tok_of(f)
                return sum(
                    jerk_by_idx.get(t + k, 0.0)
                    for k in (-5, 0, 5)          # one token either side, same phase
                )

            peak = max(cands, key=burst)
            if burst(peak) <= 0.0:
                boundaries.append(cut)
                overlaps.append(0)
                notes.append(
                    f"cut {cut}: jerk_hardcut found no jerk anywhere -> keep plan (hard cut)"
                )
                continue
            boundaries.append(int(peak))
            overlaps.append(0)
            notes.append(
                f"cut {cut}: hunt unreliable ({b}) -> JERK peak {peak} "
                f"(burst {burst(peak):.2f}), hard cut (no overlap)"
            )
            continue

        best = min(cands, key=lambda f: score.get(tok_of(f), float("inf")))
        best_score = score.get(tok_of(best), float("inf"))
        if best_score == float("inf"):
            boundaries.append(cut)
            overlaps.append(int(overlap_frames))
            notes.append(
                f"cut {cut}: profile has no score here -> keep plan, "
                f"overlap {int(overlap_frames)}f"
            )
            continue
        # ---- per-seam quality gate ------------------------------------------
        # score is normalised so that 1.0 = the clip's own median, i.e. "as
        # eventful as usual".  A move is only worth making if the best frame in
        # range is actually calm; otherwise we would be sliding the boundary to
        # a spot that is merely the least-bad one and then adding an anchored
        # overlap THERE - the one combination with no defence: alignment blends
        # two independently generated versions of a busy frame, which reads as
        # ghosting.  Better to stay put and hard cut.
        # ---- too quiet ----------------------------------------------------
        # A near-zero score means the neighbouring frames are almost identical,
        # which sounds ideal - but H3's latent is four frames per token, so an
        # almost-static stretch decodes as 'three frames identical, one frame
        # nudged'.  That ratchet is invisible under motion and glaring in a
        # still: putting a seam in the stillest spot shows it off.  A slow but
        # CONTINUOUS move hides it.  So the very calmest frames are a trap too.
        if too_quiet_below > 0.0 and best_score < float(too_quiet_below):
            # No good spot to move to.  Do NOT hard cut: a hard cut between two
            # windows that were never told about each other breaks the content
            # itself, which is worse than a visible seam.  Anchor instead - the
            # overlap at least carries the previous window's frames across.
            boundaries.append(cut)
            overlaps.append(int(overlap_frames))
            notes.append(
                f"cut {cut}: calmest frame is nearly static (score {best_score:.3f} "
                f"< {float(too_quiet_below):.3f}), no good spot to move to "
                f"-> keep plan, overlap {int(overlap_frames)}f"
            )
            continue
        # D6: a fixed absolute gate made the decision hinge on ~0.09 of score,
        # because on a busy clip the best candidate sits right at the clip
        # median (score 1.0) - so two nearly identical candidates landed on
        # opposite sides and produced visibly different cuts.  What actually
        # matters is the GAIN over staying put, so require a relative
        # improvement: moving must beat the planned frame by `calm_min_gain`.
        _plan_score = score.get(tok_of(cut), float("inf"))
        _gain = (
            (_plan_score - best_score) / _plan_score
            if _plan_score not in (0.0, float("inf")) else 0.0
        )
        if calm_min_gain > 0.0 and _gain < float(calm_min_gain):
            boundaries.append(cut)
            overlaps.append(int(overlap_frames))
            notes.append(
                f"cut {cut}: best candidate only {_gain*100:.0f}% calmer than the "
                f"planned frame (< calm_min_gain {float(calm_min_gain)*100:.0f}%) "
                f"-> not worth moving, keep plan, overlap {int(overlap_frames)}f"
            )
            continue
        if calm_min_quality > 0.0 and best_score > float(calm_min_quality):
            boundaries.append(cut)
            overlaps.append(int(overlap_frames))
            notes.append(
                f"cut {cut}: no calm frame within {window}f (best {best_score:.2f} "
                f"> calm_min_quality {float(calm_min_quality):.2f}) "
                f"-> keep plan, overlap {int(overlap_frames)}f"
            )
            continue
        boundaries.append(int(best))
        overlaps.append(int(overlap_frames))
        notes.append(
            f"cut {cut}: hunt unreliable (boundary {b}, ratio {entry.get('ratio')}) "
            f"-> calm frame {best} (score {score.get(tok_of(best)):.2f}), "
            f"overlap {int(overlap_frames)}f"
        )
    return boundaries, overlaps, notes


def explicit_segments(
    video_tokens: int, total_frames: int, cut_frames, token_step: int = 1,
    overlap_frames: int = 0, overlap_per_cut=None,
) -> list:
    """Explicit cut-frame list -> UNEQUAL window bounds.

    Each boundary is snapped to the nearest TOKEN edge (`token_step=1` = finest,
    1-4 frame resolution; `token_step=5` = the coarse 17-frame grid).
    `cut_frames` is in frames; 0 and the clip end are implied and must not be
    listed.

    `overlap_frames` > 0 keeps every CUT where it is but moves each window's
    START back by that many frames, so window i+1 re-reads the tail of window i.
    That overlap is what lets the sampler anchor its first token on the previous
    window's output (see `anchor_conditioning`) instead of starting cold at a
    hard boundary.  It must stay below the shortest window or the window would
    collapse onto itself.

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

    # Use the frames the snap above produced, not the raw numbers - and keep
    # snapping with `_snap_boundary` (token_step aware).  Re-snapping through the
    # coarse `core._snap_frame` here used to throw the fine grid away again and
    # pull every boundary back onto a multiple of 17.
    bounds = [0] + [fr for _f, _t, fr in snapped] + [int(total_frames)]
    segments = []
    ov_default = max(0, int(overlap_frames))
    # per-seam overlap: one entry per cut (i.e. per interior boundary); a shorter
    # list falls back to `overlap_frames`, a longer one is truncated
    ov_list = [max(0, int(x)) for x in (overlap_per_cut or [])]
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        ov = ov_list[i - 1] if 0 < i <= len(ov_list) else ov_default
        # overlap: keep the cut (hi) where it is, pull this window's start back
        if i > 0 and ov:
            lo = max(0, lo - ov)
        if lo == 0:
            start_token, start_frame = 0, 0
        else:
            start_token, start_frame = _snap_boundary(lo, video_tokens, token_step)
        if hi >= total_frames:  # exact clip end, never snapped away
            end_token, end_frame = video_tokens, total_frames
        else:
            end_token, end_frame = _snap_boundary(hi, video_tokens, token_step)
        if end_token <= start_token:  # defensive: every case is caught above
            raise ValueError(
                f"window {i} collapses to {end_token} token(s) (frames {start_frame}.."
                f"{end_frame}); cuts must sit on distinct 17-frame grid points, away "
                "from both the clip start and the clip end."
                + (f" overlap_frames={ov} is too large for this window."
                   if ov else "")
            )
        segments.append((start_token, start_frame, end_token, end_frame))
    if ov_default and len(segments) > 1:
        # the sampler needs the start to land strictly inside the previous
        # window, i.e. overlap < shortest window length
        shortest = min(e - s for _st, s, _et, e in segments)
        if ov_default >= shortest:
            raise ValueError(
                f"overlap_frames={ov_default} must be smaller than the shortest "
                f"window ({shortest} frames)"
            )
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


def _dump_av_latent(path, video, audio=None, meta=None):
    """诊断用（dump_latents=true 时）：AV latent 落盘（fp16 contiguous + 元数据）。

    绝不打断采样：任何 dump 失败只打日志。**CPU 优先**——先 .cpu() 再转 fp16，
    全程不在 GPU 上分配新显存（aimdo 的内存计划之外绝不碰显存，教训：2026-09-25
    21:03 进程在 W2 加载时无声死亡，当日 WER 有三个 LiveKernelEvent 141/117）。
    """
    try:
        import os

        os.makedirs(str(os.path.dirname(str(path))) or ".", exist_ok=True)
        payload = {
            "video": video.detach().cpu().to(torch.float16).contiguous(),
        }
        if audio is not None:
            payload["audio"] = audio.detach().cpu().to(torch.float16).contiguous()
        if meta:
            payload["meta"] = meta
        torch.save(payload, str(path))
        print(f"[HardCut]   dump -> {path}", flush=True)
    except Exception as _dump_exc:  # pragma: no cover - 诊断路径绝不打断采样
        print(f"[HardCut]   dump FAILED ({path}): {_dump_exc}", flush=True)


def _redenoise_seam_windows(
    core,
    sampling,
    accumulated,
    audio,
    conditioning,
    model,
    noise,
    sampler,
    sigmas,
    negative,
    cfg,
    seam_marks,
    window_tokens: int,
    lock_tokens: int,
    only_frames,
    global_video_noise,
    global_audio_noise,
):
    """Route ① / E-3: re-denoise a short window around each seam, both
    temporal ends locked (denoise_mask 0), the middle free (1).

    The locked ends are re-injected every sampler step by KSamplerX0Inpaint
    at VISUAL_COND_TIMESTEP (0.999) - i.e. the model sees them as near-clean
    keyframes on BOTH sides of the seam while it regenerates the middle.  The
    transition is generated by the model, not blended across.

    The scheme is RePaint's (Lugmayr et al., CVPR 2022, arXiv:2201.09865): the
    known region is re-conditioned on EVERY step instead of once at t=0, so the
    model always solves the free region under a known boundary.
    IDEA only - no RePaint code is used or copied, and none is needed: ComfyUI
    core already implements this, and these are what the call chain reaches
    (`KSamplerX0Inpaint` in comfy/samplers.py, `MiniMaxH3.scale_latent_inpaint`
    in comfy/model_base.py).  Said explicitly because the RePaint repository is
    CC BY-NC-SA 4.0 (non-commercial, share-alike), which would NOT be compatible
    with this pack's GPL-3.0-or-later if any of its code were in here.
    Verified end-to-end on the real comfy path (CPU, no weights) in
    _hardcut_work/seamfix/e1_temporal_mask_test.py: locked tokens converge to
    the published latent exactly (err ~2e-7).
    """
    entries = []
    total_tokens = int(accumulated.shape[2])
    accumulated = accumulated.clone()
    for seam_token in seam_marks:
        seam_token = int(seam_token)
        if only_frames is not None:
            frame_here = core.frames_for_tokens(seam_token)
            if all(abs(frame_here - int(f)) > FRAME_GRID for f in only_frames):
                continue
        half = max(1, int(window_tokens) // 2)
        w0 = max(0, seam_token - half)
        w1 = min(total_tokens, w0 + int(window_tokens))
        w0 = max(0, w1 - int(window_tokens))
        span = w1 - w0
        if span <= 2 * int(lock_tokens):
            entries.append({"seam_token": seam_token, "skipped": "window too small"})
            continue
        window = accumulated[:, :, w0:w1].contiguous()
        height, width = window.shape[-2:]
        f0 = core.frames_for_tokens(w0)
        f1 = core.frames_for_tokens(w1)
        a0 = round(f0 * core.FRAME_RESCALE)
        a1 = min(audio.shape[-1], max(a0 + 1, round(f1 * core.FRAME_RESCALE)))
        chunk_audio = audio[..., a0:a1].contiguous()

        # temporal mask: 0 = keep published latent (locked), 1 = generate
        video_mask = torch.ones(
            1, 1, span, height, width, dtype=torch.float32, device=window.device
        )
        video_mask[:, :, : int(lock_tokens)] = 0.0
        video_mask[:, :, span - int(lock_tokens):] = 0.0
        audio_mask = torch.ones_like(chunk_audio)
        piece = {
            "samples": comfy.nested_tensor.NestedTensor((window, chunk_audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        }
        piece_sampler = sampling.rebind_dual_clock_sampler(model, piece, sampler)
        prepared_noise = None
        if global_video_noise is not None:
            prepared_noise = comfy.nested_tensor.NestedTensor(
                (
                    global_video_noise[:, :, w0:w1].contiguous(),
                    global_audio_noise[..., a0:a1].to(
                        device=window.device, dtype=chunk_audio.dtype
                    ).contiguous(),
                )
            )
        chunk_conditioning = core.reanchor_conditioning(
            conditioning, f0, f1, (height, width)
        )
        sampled = core.sample_piece(
            piece,
            chunk_conditioning,
            model,
            noise,
            piece_sampler,
            sigmas,
            negative,
            cfg,
            prepared_noise=prepared_noise,
        )
        new_video = sampled.tensors[0].to(
            device=accumulated.device, dtype=accumulated.dtype
        )
        accumulated[:, :, w0:w1] = new_video
        entries.append(
            {
                "seam_token": seam_token,
                "window_tokens": [w0, w1],
                "window_frames": [f0, f1],
                "lock_tokens": int(lock_tokens),
            }
        )
        print(
            f"[HardCut]   seam {seam_token} re-denoised: window {w0}:{w1} tok "
            f"({f0}:{f1} f), lock {lock_tokens} tok each side",
            flush=True,
        )
    return accumulated, entries


def _anchor_conditioning_multi(
    core, conditioning, previous_video, start_frame, strength, tokens
):
    """Route ② / E-2: anchor the window on the previous output's first `tokens`
    tokens instead of exactly one (upstream `anchor_conditioning` slices
    `token:token+1`).  The keyframe format is natively multi-token - upstream
    `_trim_keyframe` walks every latent token and its FRAME_PER_TOKEN span -
    so this is a slice-width change only.  StreamingT2V's finding: single-frame
    conditioning is what makes chunk transitions inconsistent.
    """
    token = core.tokens_for_frames(int(start_frame))
    span = max(1, min(int(tokens), int(previous_video.shape[2]) - token))
    anchor = {
        "resolved_frame_index": 0,
        "latent": previous_video[:, :, token : token + span].contiguous(),
    }
    output = []
    for tensor, metadata in conditioning:
        updated = dict(metadata)
        keyframes = [
            keyframe
            for keyframe in updated.get("minimax_keyframes", [])
            if keyframe.get("resolved_frame_index") != 0
            or keyframe.get("latent") is None
        ]
        updated["minimax_keyframes"] = [anchor, *keyframes]
        updated["minimax_visual_cond_noise_aug"] = max(
            0.0, min(1.0, float(strength))
        )
        output.append([tensor, updated])
    return output


def _camera_compensate(v, max_shift: int = 3):
    """Align each token to its predecessor by the integer (dy, dx) latent shift
    that minimises their mean absolute difference, accumulating along the clip,
    so a steady pan or truck reads as stillness and only motion AGAINST the
    camera survives into the differences.

    IDEA re-implemented (no code copied) from MAINodes (matlowai,
    GPL-3.0-or-later - same licence as this pack).  Their note is the whole
    reason this exists: "the documented cause of panrun's over-dilation
    (124 -> 345 frames) was the pan itself scoring as jerk."  Our prompt
    templates offer 14 camera moves, so without this a truck or arc shot is
    read as violent motion and the calm search walks away from perfectly good
    boundaries.

    Edges wrap (torch.roll); at <= max_shift latent cells on a ~64-cell frame
    that is a border effect, not a signal.
    """
    T = int(v.shape[2])
    if T < 2 or max_shift <= 0:
        return v
    frames = [v[:, :, 0]]
    dy = dx = 0
    for t in range(1, T):
        prev = frames[-1]
        cur = v[:, :, t]
        best, best_shift = None, (dy, dx)
        for sy in range(dy - max_shift, dy + max_shift + 1):
            for sx in range(dx - max_shift, dx + max_shift + 1):
                cand = torch.roll(cur, (sy, sx), dims=(-2, -1))
                err = float((cand - prev).abs().mean())
                if best is None or err < best:
                    best, best_shift = err, (sy, sx)
        dy, dx = best_shift
        frames.append(torch.roll(cur, (dy, dx), dims=(-2, -1)))
    return torch.stack(frames, dim=2)


def _reduce_hw(x, mode: str = "mean"):
    """Collapse (1, C, T, h, w) -> (T,).  `mean` matches the community default;
    `max` / `top-decile` keep a small hot region from being averaged away."""
    if mode == "max":
        return x.amax(dim=(0, 1, 3, 4))
    if str(mode).startswith("top"):
        flat = x.flatten(-2)                       # (1, C, T, h*w)
        k = max(1, int(flat.shape[-1]) // 10)
        return flat.topk(k, dim=-1).values.mean(dim=(0, 1, 3))
    return x.mean(dim=(0, 1, 3, 4))


def _latent_change_profile(video, win: int = 2, compensate: bool = False,
                           reduce_mode: str = "mean", persistence: bool = True) -> list:
    """Every token boundary scored TWICE - no threshold anywhere.

    `[(token_index, local_ratio, global_ratio), ...]` for all boundaries; the
    change sits between that token and the next one.

    * `local_ratio`  = d[i] / median(d[i-win : i+win+1]) - does this boundary
      stand out from ITS OWN neighbourhood (the original score).
    * `global_ratio` = d[i] / median(d) - the whole clip is the baseline.

    Measured 2026-09-20 (planned cut 68, seam landed on 85 = 17 frames off;
    report top_candidates `[[85, 2.04], [68, 1.68], [51, 1.62]]`): the LOCAL
    score is actively misleading AT a turn, because a smeared turn raises the
    median of the very neighbourhood it sits in - the turn's own score drops,
    while a quiet stretch two grids away wins on a tiny wobble.  The global
    score has no such blind spot (same clip in the pixel domain: turn 2.72x
    median, quiet stretch 1.25x).  Callers therefore RANK by `global_ratio`
    and keep `local_ratio` only to describe how eventful the window is.
    """
    v = video.detach().float()
    if v.ndim != 5 or v.shape[2] < 3:
        return []
    if compensate:
        v = _camera_compensate(v)
    d = _reduce_hw((v[:, :, 1:] - v[:, :, :-1]).abs(), reduce_mode)
    n = int(d.numel())
    gmed = float(d.median())
    if gmed <= 0:
        return []
    # ---- third difference (jerk) -------------------------------------------
    # IDEA (re-implemented, no code copied) from MAINodes' H3 Jerk Oracle
    # (matlowai, GPL-3.0-or-later - same licence as this pack), which ranks
    # tokens by |d3| instead of |d1|.  Reason, in their words and measurements:
    # the value-domain first difference is contaminated by motion energy - a
    # textured object passing a location makes the values there pulse, and a
    # pulse has large differences of EVERY order even at constant velocity
    # (they measured corr(|d1|, |d3|) = 0.96-0.98 on real clips).  |d3| measures
    # how abruptly the motion CHANGES, which is closer to what "calm" means.
    if v.shape[2] >= 4:
        j3 = _reduce_hw(
            (v[:, :, 3:] - 3.0 * v[:, :, 2:-1] + 3.0 * v[:, :, 1:-2] - v[:, :, :-3]).abs(),
            reduce_mode,
        )
        # centre-align onto d's (n) grid: leading + trailing edge pad.
        # (F.pad with mode="replicate" rejects 1-D tensors, so grow it by hand.)
        j3 = torch.cat([j3[:1], j3, j3[-1:]])
        if int(j3.numel()) < n:
            j3 = torch.cat([j3, j3[-1:].expand(n - int(j3.numel()))])
        elif int(j3.numel()) > n:
            j3 = j3[:n]
    else:
        j3 = torch.zeros_like(d)
    jmed = float(j3.median())
    if jmed <= 0:
        jmed = 1.0

    # ---- persistence (IDEA from PERSIST, arXiv:2608.29287) ----------------
    # Their reformulation of shot-boundary detection: a frame is a real boundary
    # only when its local change evidence comes with a PERSISTENT update of the
    # clip's latent state, not a transient excursion that "returns to the
    # surrounding trend".  Flicker, hand-held shake, motion blur, occlusion and a
    # texture passing a spot all produce equally sharp local change without a new
    # shot - and our d1/d3 cannot tell those apart from a real cut.
    #
    # Hand-rolled from the paper's idea (the paper trains a FiLM-conditioned
    # classifier; we use a plain window-mean difference).  No code copied.
    #   before = mean latent state of the tokens BEFORE the change
    #   after  = mean latent state of the tokens AFTER it
    #   real cut  -> two different steady states      -> large
    #   flicker   -> after falls back toward before   -> small
    half = max(1, int(win) * 5)          # one token = 5 slots; same phase either side
    pers = [0.0] * n
    if persistence and v.shape[2] >= 4:
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half)
            if lo >= i or hi <= i + 1:
                continue
            _before = v[:, :, lo:i].mean(dim=2)
            _after = v[:, :, i + 1:hi].mean(dim=2)
            _den = 0.5 * (float(_before.abs().mean()) + float(_after.abs().mean())) + 1e-6
            pers[i] = float((_after - _before).abs().mean()) / _den
        _pmax = max(pers) if pers else 0.0
        if _pmax > 0:
            pers = [p / _pmax for p in pers]     # normalise to the clip's own max

    out = []
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        med = float(d[lo:hi].median())
        if med <= 0:
            continue
        out.append((i, float(d[i]) / med, float(d[i]) / gmed,
                    float(j3[i]) / jmed, pers[i]))
    return out

def _hunt_shot_changes(video, sens: float = 2.0, win: int = 2) -> list:
    """First-pass latent -> the tokens where the model changed shots (no VAE).
    [DEPRECATED - kept for reference, the executor no longer calls this.]
    A fixed-ratio threshold never fires on the real turn in a fight (every
    token moves); superseded by `_latent_change_profile`, which scores each
    token locally AND against the whole clip and then snaps the peak onto
    the exclusive-frame grid.  Do not wire this back in without re-measuring.

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


# A weak global score means the window holds no real shot change - only a
# wobble or a busy-motion stretch - so the cut stays where the plan put it.
# Measured 2026-09-20 on a 15s render: real turns score 2.0-2.5, a false one
# (motion peak mistaken for a turn) scored 1.49 and moved a cut 17 frames.
# Measured on the convrot weights: real turns score 1.82-1.91, a false one (motion peak) scored 1.51.  1.6 splits them with margin - 1.8 sat right on
# top of the real turns and risked dropping them too.
FLAT_RATIO = 1.6     # below this a tolerance window counts as featureless
# ★ 硬切的转镜置信门（2026-09-26，exp_4v10a_00086 @51 事故）：真实的 latent
#   变化不一定是转镜——实测 v4 真转镜（pan->lock）~2.2x，而已锁定镜头内的
#   姿势跳变 ~1.7x；对后者硬切会发布用户可见的无动机跳切。低于此值的变化
#   照实记录但拒绝硬切，由 calm 搜索 / 锚定 overlap 藏接缝。n=2 标定。
HUNT_MIN_CUT_RATIO = 2.0
# One 17-frame block carries 5 tokens: 1 exclusive (a single frame) + 4 shared
# (each covering 4 frames).  So a tolerance below ~4 frames cannot even reach
# the neighbouring candidate row - worth saying out loud in the log.
GRID_FRAMES_PER_TOKEN = 4


def _align_to_profile(profile, planned, tolerance: int, video_tokens: int,
                      min_persistence: float = 0.0, search_window: int = 0):
    """Per planned cut: the strongest latent change in the window, then SNAP it
    onto the exclusive-frame grid the window start requires.

    ★ Why snap instead of argmax over exclusive frames (measured 2026-09-20,
    symptom: "the seam is ~16 frames off"): `_latent_change_profile`
    normalises every token by its OWN +-2 neighbourhood, so the two shoulders
    of a turn RAISE the median right where the turn is and LOWER that token's
    score, while a quiet stretch two grids away scores high on a tiny wobble.
    Choosing the best exclusive frame therefore walked away from the turn with
    the plan as prior: planned 68, real turn ~66 -> chosen 85 with ratio 2.04
    while the turn itself only scored 1.68.  The turn is the signal; the
    exclusive frame is only a constraint on where the WINDOW may start.  So
    find the peak anywhere in the window first, then snap THAT to the closest
    multiple of FRAME_GRID.

    Both sides only ever emit exclusive-frame boundaries - a shared-group head
    corrupts the window (measured 2026-09-19/20, same seed: start 68 clean,
    start 69 corrupted 4 frames).

    Returns `(aligned, boundary_tokens)`; `aligned` carries the per-cut report
    including the top candidates, so the latent's behaviour near the cut stays
    visible instead of a bare accept/reject.
    """
    core = _core()
    total_frames = core.frames_for_tokens(int(video_tokens))

    def frame_of(idx):
        # the change sits between token `idx` and `idx+1`
        return core.frames_for_tokens(int(idx) + 1)

    def token_of(frame):
        for t in range(1, int(video_tokens)):
            if core.frames_for_tokens(t) == int(frame):
                return t
        return None

    # The SEARCH radius and the RESIDUAL threshold are different things.  The
    # window must be wide enough to actually see the model's turn (measured
    # drift runs 1-17 frames, and 78 vs a planned 85 is a typical case); the
    # threshold decides whether, once measured, the boundary is trustworthy.
    # Sharing one number made a small tolerance blind the detector entirely.
    _win = int(search_window) if int(search_window) > 0 else max(int(tolerance), 34)

    aligned, boundary_tokens = [], []
    _diagnostics = []
    for cut in planned:
        # index, not unpack: the profile grew columns (|d3|, persistence)
        window = [
            (int(row[0]), row[1], float(row[2]),
             float(row[4]) if len(row) > 4 else 1.0)
            for row in profile
            if 0 < int(row[0]) + 1 < int(video_tokens)
            and abs(frame_of(int(row[0])) - cut) <= _win
        ]
        if not window:
            # Silent before.  A tolerance narrower than one token cell (17
            # frames / 5 tokens = ~4 frames per step) leaves almost no candidate
            # rows, and combined with the persistence and flat-ratio gates the
            # hunt quietly finds nothing at all - which reads as "the detector
            # is broken" rather than "the window was too narrow".
            # ★ This used to call `notes.append`, and `notes` exists nowhere in
            # this function or the module: an empty window raised
            # NameError('notes') and took the node down.  Reachable whenever
            # `_latent_change_profile` skips a row (it drops tokens whose
            # neighbourhood median is 0, i.e. a run of perfectly static tokens),
            # so the cut's own row can be missing.  Same class as the
            # `_SkipProbe` crash - a name that is never defined at all, which
            # check_undefined.py cannot see (2026-09-23).
            # Functionally this cut is already covered downstream: the caller
            # treats "no entry / no boundary_frame" as "keep the planned frame".
            _diagnostics.append(
                f"[HardCut]   cut {cut}: no candidate token within {tolerance}f "
                f"(one token spans ~{GRID_FRAMES_PER_TOKEN}f) -> increase "
                f"seam_tolerance_frames or the hunt will find nothing here"
            )
            continue
        # Rank by local_change x persistence.  The global score is still the
        # local-change term (the LOCAL variant suppresses a turn, see above); the
        # persistence term is what separates a real cut from a sharp transient.
        # The 0.25 floor keeps the old ordering if persistence is unavailable
        # (older profiles, or the switch off) so behaviour degrades, not breaks.
        window.sort(key=lambda x: -(x[2] * (0.25 + 0.75 * x[3])))
        peak_idx, peak_local, peak_ratio, peak_pers = window[0][:4]
        peak_frame = frame_of(peak_idx)
        top = [
            [frame_of(i), round(float(gr), 2), round(float(pr), 2)]
            for i, _lr, gr, pr in window[:3]
        ]
        # ---- false-positive gate (PERSIST) ---------------------------------
        # A sharp local change is not a shot change on its own: a flicker, a
        # shake, or a texture sweeping past produce the same bump.  What a real
        # cut does is leave the latent state somewhere ELSE and keep it there.
        # Below the floor we decline the detection and let the plan (and the
        # calm search / hard cut) stand instead.
        if min_persistence > 0.0 and peak_pers < float(min_persistence):
            aligned.append({
                "planned_cut": cut,
                "moved": False,
                "note": (f"local change is real but not persistent "
                         f"(persistence {peak_pers:.2f} < {float(min_persistence):.2f}) "
                         f"-> treated as a false positive, cut not accepted"),
                "top_candidates": top,
            })
            continue
        note = None

        if peak_ratio < FLAT_RATIO:
            aligned.append({
                "planned_cut": cut,
                "moved": False,
                "note": (f"latent is flat near the cut "
                         f"(peak ratio {peak_ratio:.2f} < {FLAT_RATIO})"),
                "top_candidates": top,
            })
            continue

        # ---- turn-confidence gate (RATIO) ----------------------------------
        # 见 HUNT_MIN_CUT_RATIO：真实变化 ≠ 转镜。低于置信门的变化照实记录、
        # 拒绝硬切 —— 计划位置保留，calm 搜索 / 锚定 overlap 藏接缝。
        if peak_ratio < HUNT_MIN_CUT_RATIO:
            aligned.append({
                "planned_cut": cut,
                "moved": False,
                "note": (f"local change is real but below turn confidence "
                         f"(ratio {peak_ratio:.2f} < {HUNT_MIN_CUT_RATIO:.2f}) "
                         f"-> kept the plan, anchored overlap"),
                "top_candidates": top,
            })
            continue

        # ★ snap the measured turn onto the exclusive-frame grid.  NEAREST wins:
        # nudging the seam a frame or two towards the turn is harmless, walking
        # to another grid point is what produced the 16-frame miss.
        final_frame = peak_frame
        if peak_frame % FRAME_GRID:
            lo = (peak_frame // FRAME_GRID) * FRAME_GRID
            hi = lo + FRAME_GRID
            snapped = lo if (peak_frame - lo) <= (hi - peak_frame) else hi
            snapped = max(FRAME_GRID, min(snapped, total_frames - FRAME_GRID))
            if snapped != peak_frame:
                note = "; ".join(x for x in (
                    note,
                    f"snapped the measured turn {peak_frame} -> {snapped} "
                    f"(nearest exclusive frame)",
                ) if x)
                final_frame = snapped

        token = token_of(final_frame)
        if token is None:
            # unreachable for multiples of FRAME_GRID - kept as a guard
            token, final_frame = int(peak_idx) + 1, peak_frame
        if token not in boundary_tokens:
            boundary_tokens.append(token)
        entry = {
            "planned_cut": cut,
            "moved": final_frame != cut,
            "boundary_token": token,
            "boundary_frame": final_frame,
            "ratio": round(float(peak_ratio), 2),
            "local_ratio": round(float(peak_local), 2),
            "top_candidates": top,
        }
        # ★ ALWAYS report the measured turn - not only when the snap moved the
        # frame.  The calm search decides hard-cut vs anchored-overlap from how
        # far the boundary had to sit from the TURN; while this key was written
        # conditionally, every peak that already landed on a multiple of 17 left
        # it absent, the gate fell back to the PLAN distance, and the same
        # situation (plan 85, turn 68 vs turn 66) flipped between an anchored
        # overlap and a hard cut purely because of where the argmax fell
        # (2026-09-23).
        entry["measured_turn_frame"] = peak_frame
        if note:
            entry["note"] = note
        aligned.append(entry)
    for _line in _diagnostics:
        print(_line, flush=True)
    return aligned, boundary_tokens



def _hunt_log_line(entry) -> str:
    """Format one per-cut hunt result for the console.

    Separated from the executor so it is unit-testable (see
    `_hardcut_work/seamfix/e4_hardcut_policy_test.py`, cases L1-L4).  The console
    is the ONLY window into where the seams landed - a 15s render runs for
    minutes - and a declined cut used to print `ratio=None` with no reason,
    which is indistinguishable between "the persistence gate fired", "the flat
    gate fired" and "the window held no candidate at all".  The note also
    carries the snap line for an ACCEPTED cut, which is what explains
    `measured=73 -> boundary=68` (2026-09-23).
    """
    moved = entry.get("moved")
    planned_cut = entry.get("planned_cut")
    boundary = entry.get("boundary_frame")
    if boundary is None:
        text = (f"cut planned={planned_cut} -> boundary=None "
                f"(NOT accepted, moved={moved}")
    else:
        offset = int(boundary) - int(planned_cut)
        text = (f"cut planned={planned_cut} -> boundary={boundary} "
                f"(offset={offset:+d}f, moved={moved}")
    if entry.get("measured_turn_frame") is not None:
        text += f", measured={entry['measured_turn_frame']}"
    text += f", ratio={entry.get('ratio')}"
    if entry.get("note"):
        text += f", why: {entry['note']}"
    return f"[HardCut]   {text})"


def release_text_encoders() -> list:
    """Unload CLIP / text-encoder models, keep the diffusion model loaded.

    On a 32 GB machine the H3 stack is DiT (19.5 GB) + Qwen3-VL TE (14.6 GB);
    keeping both parked is what overflows aimdo's host buffer.  The second pass
    never uses the TE, and here its conditioning is already built, so the TE can
    go.  Unlike model_management.unload_all_models() this leaves the DiT alone,
    so the second pass does not have to reload 19.5 GB.
    """
    freed = []
    try:
        import comfy.model_management as _mm
        loaded = list(getattr(_mm, "current_loaded_models", []) or [])
    except Exception:
        return freed
    for entry in loaded:
        try:
            from .nodes_unload import _looks_like_text_encoder
        except Exception:
            break
        if not _looks_like_text_encoder(entry):
            continue
        inner = getattr(entry, "model", None)
        name = type(inner).__name__ if inner is not None else "?"
        try:
            entry.model_unload()
            freed.append(name)
        except Exception:
            pass
    return freed


# ---- seam re-denoise auto gate（策略树 §4.1 B1 前置门 v1，TOKEN_RESEARCH.md）----
# 证据链：
#   * E-3(00068)：静缝重去噪有效（187 台阶 5.30x -> 1.72x）；闹处/细节密处
#     （85/272）重去噪**注入伪纹理**。
#   * 00078：静止锁定窗的锚定 overlap 本来就不可见（187/272 零重影）；
#     运动窗的锚定 overlap 磨糊（00076/68 背景 -32%）—— 但 E-3 说闹处恰是
#     重去噪会注入的地方。
#   * ⇒ v1 门控：跳过"已证实危险"的闹缝，其余放行。静缝是否也该省掉
#     （00078 显示纯 overlap 已干净）交给标定跑片决定，不拍脑袋。
# 度量：缝邻域的 latent 变化分 —— 与 calm 搜索排序同口径（每 token 取
# max(global, jerk)），除以全片中位 -> busy 度。这是 §4.1 像素域 B1
# （锚间高频一致性）的 latent 域替身，零额外计算；标定后若相关性差再升级。
REDENOISE_BUSY_SKIP_RATIO = 1.5


def gate_seam_redenoise(profile, seam_marks,
                        busy_skip_ratio: float = REDENOISE_BUSY_SKIP_RATIO,
                        span: int = 2):
    """对每个 overlap 缝算锚定窗 busy 度，闹缝跳过重去噪。

    `seam_marks` 是累积时间轴的 token 位；`profile` 行 = (idx, local, glob,
    jerk[, pers])，score = max(glob, jerk)（与 find_calm_boundaries 同口径）。

    返回 (eligible_tokens, info)，info 每缝一条 {"token", "busy_ratio",
    "gated"}。**profile 为空 -> 全部放行**：测量不到时不静默改变语义
    （静默少干活=条件写入式陷阱），由调用方打日志提醒。
    """
    marks = [int(t) for t in seam_marks or []]
    if not marks:
        return [], []
    score = {}
    for row in profile or ():
        try:
            idx, glob = int(row[0]), float(row[2])
            jerk = float(row[3]) if len(row) > 3 else glob
        except (TypeError, IndexError, ValueError):
            continue
        score[idx] = max(glob, jerk)
    if not score:
        return list(marks), [
            {"token": t, "busy_ratio": None, "gated": False, "why": "no score"}
            for t in marks
        ]
    vals = sorted(score.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    eligible, info = [], []
    for t in marks:
        nb = [score[i] for i in range(t - span, t + span + 1) if i in score]
        nb_mean = sum(nb) / len(nb) if nb else 0.0
        ratio = (nb_mean / med) if med > 1e-9 else 0.0
        gated = bool(ratio >= busy_skip_ratio)
        info.append({"token": t, "busy_ratio": round(ratio, 3), "gated": gated})
        if not gated:
            eligible.append(t)
    return eligible, info


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
    show_memory_log: bool = True,
    seam_redenoise: bool = False,
    seam_redenoise_frames: str = "",
    seam_redenoise_gate: str = "off",
    seam_window_tokens: int = 10,
    seam_lock_tokens: int = 3,
    anchor_tokens: int = 1,
    dump_latents: bool = False,
    dump_dir: str = "",
    upscale_pad_tokens: int = 0,
    seam_blend: bool = False,
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
    # 注：一采缓存与显存护栏**不放在这里**。
    #   本机 OOM 发生在 execution.py:306 process_inputs（搬权重阶段），采样没开始 -> 本节点那时还没执行，
    #   写在这里等于死代码。现分别由 `MiniMaxH3AVLatentSave`（放在一采之后、二采之前）与
    #   `MiniMaxH3VRamGuard`（放在加载完大模型之后）承担。

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

    # ---- optional: put the window boundary ON the model's own cut ----
    # The upscaler never sees the prompt, so the model's shot change is only
    # observable in the latent we are holding.  Measuring it here costs no VAE
    # decode, and placing the split before it keeps the seam on continuous
    # content — the model performs the cut itself, inside the window.
    planned = [int(c) for c in ((plan.get("hardcut") or {}).get("segment_frames") or [])]
    tolerance = max(0, int((plan.get("hardcut") or {}).get("seam_tolerance", 17)))
    segment_frames = planned or None
    seam_hunt = None
    profile = None  # 门控在 hunt 关着时也要能问；None -> gate 全放行并打日志
    aligned, boundary_tokens = None, None
    if planned and isinstance(plan.get("hardcut"), dict):
        # ★ 解耦（2026-09-26）：profile 与对齐分析**总是跑** —— calm 搜索和
        #   熔化 gate 都吃它们的输出，不该被 auto_seam_hunt 连坐。
        #   auto_seam_hunt 现在只控制一件事：**是否按测得的转镜移动切点/接受
        #   硬切**。关掉 = 分析照跑，但全部切点回退计划位置 + 锚定 overlap
        #   （教训 exp_4v10a_00086 @51：hunt 把锁定镜头内的姿势跳变误判为
        #   转镜，ratio 1.69 切出用户可见的跳切）。
        # Threshold-based detection died on busy footage: a fight moves every
        # token, so the real shot change never clears a fixed ratio while a
        # hard action beat does (measured: the only hit was the finale, 119
        # frames away from the cut).  The planned cut is already a strong
        # prior, so instead of hunting hits and gating them, take the
        # STRONGEST latent change within the tolerance window of each planned
        # cut.  No threshold; the top candidates go into the report so the
        # latent's behaviour near the cut is visible at last.
        profile = _latent_change_profile(
            video,
            compensate=bool((plan.get("hardcut") or {}).get("profile_camera_compensate", False)),
            reduce_mode=str((plan.get("hardcut") or {}).get("profile_reduce", "mean")),
            persistence=bool((plan.get("hardcut") or {}).get("hunt_persistence", True)),
        )
        aligned, boundary_tokens = _align_to_profile(
            profile, planned, tolerance, int(video.shape[2]),
            min_persistence=float(
                (plan.get("hardcut") or {}).get("hunt_min_persistence", 0.0)
            ),
            search_window=int((plan.get("hardcut") or {}).get("hunt_search_window", 0)),
        )
        if not auto_seam_hunt:
            # hunt 关（用户手动选择）：全部已接受的切点转成 declined 形态
            # （与 persistence/flat 门同构），回退计划位置 + 锚定 overlap。
            boundary_tokens = []
            for entry in aligned:
                if entry.get("boundary_token") is not None:
                    entry["moved"] = False
                    entry["note"] = "; ".join(
                        x
                        for x in (
                            entry.get("note"),
                            f"hunt off -> declined the measured turn "
                            f"{entry.get('measured_turn_frame')} (anchored overlap)",
                        )
                        if x
                    )
                    entry.pop("boundary_token", None)
                    entry.pop("boundary_frame", None)
                    entry.pop("measured_turn_frame", None)
        if boundary_tokens:
            boundary_tokens.sort()
            cand = [core.frames_for_tokens(t) for t in boundary_tokens]
            # ★ A cut the hunt did not accept (flat window) must fall back to
            # the PLANNED frame, not disappear: dropping it merges two windows
            # into one and blows the load (measured: 85/187/272 -> only 85/187
            # kept -> segments [85,102,175], last one 175f instead of 90f ->
            # OOM on the first run).
            # "unaccepted" means THIS cut's own entry carries no boundary - NOT
            # "its planned frame is absent from the boundary list".  A cut the
            # hunt MOVED (85 -> 68) is accepted; falling back for it as well
            # added a duplicate boundary and split off a 17-frame sliver
            # (measured: boundaries [68,85,170,187,272] -> segments
            # [68,17,85,17,85,90], i.e. six samples instead of three).
            extra = []
            for cut in planned:
                entry = next(
                    (e for e in aligned
                     if int(e.get("planned_cut", -1)) == int(cut)),
                    None,
                )
                if entry is None or entry.get("boundary_frame") is None:
                    extra.append(int(cut))
            if extra:
                seam_hunt_note = (
                    f"hunt found no usable change near {extra}; keeping the planned cut(s)"
                )
            cand = sorted({int(c) for c in cand} | set(extra))
            tail = frame_count - cand[-1]
            if tail < FRAME_GRID:
                seam_hunt_note = (
                    f"hunted boundary {cand[-1]} leaves a {tail}-frame tail; keeping the plan"
                )
            else:
                segment_frames = cand
                seam_hunt_note = None
        else:
            seam_hunt_note = (
                "hunt off -> all planned cuts kept as anchored overlaps"
                if not auto_seam_hunt
                else "no latent change found within tolerance of any planned cut"
            )
        seam_hunt = {
            "tolerance_frames": tolerance,
            "planned_cuts": planned,
            "aligned": aligned,
            "boundary_tokens": boundary_tokens,
            "boundary_frames": [core.frames_for_tokens(t) for t in boundary_tokens],
        }
        if seam_hunt_note:
            seam_hunt["note"] = seam_hunt_note

        # ---- jerk profile digest -------------------------------------------
        # Why: a high-jerk peak may sit on the FALLING side of a burst.  Motion
        # that is violent enough makes the model give up and smear, and the
        # smear flattens frame-to-frame differences, so |d3| can fall off again
        # past the peak.  If that happens, the chosen boundary is "the edge of
        # the burst" rather than "the messiest frame" - and whether the seam is
        # hidden then depends on the masking still being there.  These numbers
        # let us see the curve's shape instead of arguing about it.
        try:
            # only 17k (token 5k) is a legal boundary, so only those are worth
            # reporting - a peak on a shared token cannot be cut at anyway
            _jr = sorted(
                ((int(r[0]), float(r[3])) for r in profile
                 if len(r) > 3 and int(r[0]) % 5 == 0),
                key=lambda kv: -kv[1],
            )[:6]
            if _jr:
                seam_hunt["jerk_peaks"] = [
                    {"token": t, "frame": int(t) // 5 * FRAME_GRID,
                     "ratio": round(r, 2)}
                    for t, r in _jr
                ]
                _med_t = max(1, len(profile) // 2)
                seam_hunt["jerk_shape_note"] = (
                    "top-6 |d3| ratios on exclusive frames only (token 5k -> frame 17k), "
                    "the only frames a boundary may sit on. A peak followed by a sharp "
                    "fall means the burst is being smeared; a flat top means sustained motion."
                )
                seam_hunt["profile_len"] = len(profile)
        except Exception as _pe:
            seam_hunt["jerk_peaks_error"] = str(_pe)

    # ---- adaptive: if the hunt could not vouch for a cut, move that boundary to
    # the CALMEST frame nearby and give that seam an anchored overlap, instead of
    # cutting through continuous content.  Runs whenever a plan exists - the
    # profile/alignment are computed unconditionally (decoupled from
    # auto_seam_hunt, 2026-09-26).
    calm_boundaries = None
    # Log lines are buffered and emitted in READING order at the end, not in
    # code order: the flow is params -> per-cut detection -> decisions ->
    # fallback config -> summary, while the code computes the plan first, hunts
    # second and decides third.  Printing as we go put the decisions above the
    # detections they came from.
    _log_buf = {"cfg": [], "hunt": [], "calm": [], "overlap": [], "sum": []}
    calm_overlaps = None
    if planned and isinstance(plan.get("hardcut"), dict):
        _hc = plan["hardcut"]
        if _hc.get("auto_calm_search"):
            calm_boundaries, calm_overlaps, _calm_notes = find_calm_boundaries(
                profile, planned, aligned, frame_count,
                window=int(_hc.get("calm_search_window", 34)),
                overlap_frames=int(_hc.get("calm_overlap_frames", 17)),
                seam_tolerance=int(_hc.get("seam_tolerance", 17)),
                policy=str(_hc.get("calm_policy", "calm_overlap")),
                abstain_below=float(_hc.get("calm_abstain_below", 0.0)),
                calm_min_quality=float(_hc.get("calm_min_quality", 0.0)),
                too_quiet_below=float(_hc.get("calm_too_quiet_below", 0.0)),
                calm_min_gain=float(_hc.get("calm_min_gain", 0.0)),
            )
            # ---- load guard (external review D2) ---------------------------
            # auto_plan sized the PLANNED windows.  The search just moved the
            # boundaries, and a longer window costs more second-pass memory
            # (load = longest_frames x canvas_mp), so recheck and undo any move
            # that breaks the anchor.  Otherwise the calm search can quietly
            # create the oversized window that produces tail artefacts.
            _mp = float(_hc.get("canvas_mp") or 0.0) or 1.5
            _total = int(frame_count)
            for _i in range(len(calm_boundaries)):
                _lo = max(0, int(calm_boundaries[_i]) - int(
                    calm_overlaps[_i] if _i < len(calm_overlaps) else 0))
                _hi = int(calm_boundaries[_i + 1]) if _i + 1 < len(calm_boundaries) else _total
                _frames = max(0, _hi - _lo)
                if _frames * _mp < float(LOAD_FAIL):
                    continue
                _plan_cut = int(planned[_i]) if _i < len(planned) else int(calm_boundaries[_i])
                if _plan_cut == int(calm_boundaries[_i]):
                    continue                      # already the plan; nothing to undo
                _log_buf["calm"].append(
                    f"[HardCut]   calm: cut {_plan_cut}: move to "
                    f"{int(calm_boundaries[_i])} would make segment {_i} "
                    f"{_frames}f x {_mp:.2f}MP = {_frames * _mp:.0f} >= "
                    f"{float(LOAD_FAIL):.0f} -> reverted to the planned frame"
                )
                calm_boundaries[_i] = _plan_cut
                if _i < len(calm_overlaps):
                    calm_overlaps[_i] = 0    # a reverted seam is a plain cut
            segment_frames = calm_boundaries
            for _n in _calm_notes:
                _log_buf["calm"].append(f"[HardCut]   calm: {_n}")

    # ---- free the text encoder before sampling ----
    # The conditioning above is fully built, and nothing downstream of this point
    # touches the TE: the DiT stays loaded, the encoder's 14.6 GB (and the aimdo
    # host buffer it occupies) goes back.  Disable with
    # plan.hardcut.unload_text_encoder_before_sampling = false.
    if bool((plan.get("hardcut") or {}).get("unload_text_encoder_before_sampling", True)):
        _freed_te = release_text_encoders()
        if _freed_te:
            print(f"[HardCut]   TE unloaded before sampling: {', '.join(_freed_te)}",
                  flush=True)

    # ---- windowing: explicit (possibly unequal) first, then the equal paths ----
    # overlap tokens: when > 0 each window's START is pulled back so the sampler
    # can anchor its first token on the previous window (see anchor_conditioning
    # below).  The knob is free to set, but the value that actually reaches the
    # sampler is clamped below one full window - an overlap of a whole window
    # would leave nothing new to generate.
    # Unequal windows + anchored prefixes give every segment a different shape,
    # so PyTorch's caching allocator cannot reuse the previous block and the
    # pool fragments upward run after run.  Releasing the cached blocks between
    # segments keeps the footprint flat; the models stay loaded (this is
    # soft_empty_cache, never unload_all_models).
    _empty_between = bool((plan.get("hardcut") or {}).get("empty_cache_between_segments", True))
    ov_input = max(0, int(plan.get("temporal_overlap_frames", 0)))
    _chunk = int(plan.get("temporal_chunk_frames") or 0)
    ov_tokens = ov_input
    if _chunk > 0 and ov_input >= _chunk:
        ov_tokens = max(0, _chunk - 1)
    locked_overlap = max(
        0, min(int(plan.get("locked_overlap_tokens", ov_tokens)), ov_tokens)
    )
    if ov_input:
        # This is the FALLBACK only: when the calm search runs it decides each
        # seam individually (calm_overlaps), and a hard cut there is 0.  Say so,
        # otherwise the line reads as if every seam got the global value.
        _log_buf["overlap"].append(
            f"[HardCut]   overlap: fallback {ov_input}f -> {ov_tokens}f "
            f"(chunk {_chunk}f), locked {locked_overlap}f"
            + ("   [clamped below one window]" if ov_tokens != ov_input else "")
            + ("   | per-seam values below take precedence"
               if calm_overlaps is not None else "")
        )
    if segment_frames:
        segments = explicit_segments(
            int(video.shape[2]), frame_count, segment_frames,
            overlap_frames=ov_tokens,
            overlap_per_cut=calm_overlaps,
        )
    elif plan.get("temporal_strategy") == "full_clip_safe":
        segments = [(0, 0, int(video.shape[2]), frame_count)]
    else:
        segments, frame_count = core.compute_temporal_segments(
            int(video.shape[2]),
            _chunk,
            ov_tokens,
        )

    # ---- log the cut decisions NOW ----
    # The report only lands when the whole second pass is finished, and a 15s
    # render takes minutes; without this you stare at a black console while the
    # thing decides where to split.  Never let logging break a run.
    try:
        lengths = [int(e) - int(s) for _st, s, _et, e in segments]
        longest = max(lengths) if lengths else 0
        # the FINAL boundaries are what the windows actually ended on: every
        # segment's end frame except the last one.  Reporting only the hunt's
        # accepted frames was misleading - a cut that fell back to its planned
        # frame (because the anchor killed the latent change) showed up as
        # "boundary=None" while the windows were split exactly on it.
        boundaries = [int(e) for _st, _sf, _et, e in segments][:-1]
        hunted = [int(f) for f in (seam_hunt or {}).get("boundary_frames") or []]
        # Three states, not two.  '' = hunt accepted a real turn here; '+' = the
        # calm search moved it; '*' = untouched, on the planned frame.  The old
        # version labelled calm moves as '*' too, which read as "nothing was
        # found here" even though the boundary had clearly moved.
        _moved = set()
        if calm_boundaries is not None:
            for _pc, _cb in zip(planned or [], calm_boundaries):
                if int(_pc) != int(_cb):
                    _moved.add(int(_cb))

        def _tag(_b):
            if _b in hunted:
                return ""
            return "+" if _b in _moved else "*"

        src_tag = ", ".join(f"{b}{_tag(b)}" for b in boundaries)
        _log_buf["sum"].append(
            f"[HardCut] planned_cuts={planned or '-'} "
            f"boundary_frames=[{src_tag or '-'}] "
            f"segments={len(segments)} lengths={lengths} longest={longest}f"
            + ("   ('+' = moved by the search, '*' = on the planned frame)"
               if any(b not in hunted for b in boundaries) else "")
        )
        if seam_hunt:
            _log_buf["cfg"].append(
                f"[HardCut]   tolerance={seam_hunt.get('tolerance_frames')}f "
                f"hunt={'on' if auto_seam_hunt else 'off'}"
            )
            for entry in seam_hunt.get("aligned") or []:
                _log_buf["hunt"].append(_hunt_log_line(entry))
            if seam_hunt.get("note"):
                _log_buf["hunt"].append(f"[HardCut]   note: {seam_hunt['note']}")
        for _section in ("cfg", "hunt", "calm", "overlap", "sum"):
            for _line in _log_buf[_section]:
                print(_line, flush=True)
    except Exception as _log_exc:  # pragma: no cover - logging must never fail the run
        print(f"[HardCut] log error: {_log_exc}", flush=True)

    accumulated = None
    segment_reports = []
    prev_end_frame = None
    seam_marks = []
    for start_token, start_frame, end_token, end_frame in segments:
        # how much this window re-reads from the published output: > 0 only when
        # a seam asked for an anchored prefix (the calm search sets it per cut)
        seg_overlap = max(0, (prev_end_frame or 0) - int(start_frame)) if prev_end_frame is not None else 0
        # ── 上采样时间 padding（熔化修复 ③，2026-09-26）─────────────────
        # 3D learned upscaler 逐窗独立处理，分块尾的时间感受野只有单侧
        # → 尾部 ~8 帧（= 下一窗的锚定源）轻度软化（0.85-0.9x，UP_W1 判别实证），
        #   二采采样再加深为 0.67x 的"锁入口脱焦带"。
        # 修法：upscale 输入向两侧各借 pad_t 个一采 token（全片连续、无接缝），
        #   上采样后裁回窗口 own 范围 —— 尾部拿到双侧上下文，与片尾不熔一致。
        # 音频直通不经 upscaler：采样用的 chunk_audio 保持窗口 own 范围不变。
        pad_t = max(0, int(upscale_pad_tokens))
        pt_lo = max(0, start_token - pad_t)
        pt_hi = min(video.shape[2], end_token + pad_t)
        chunk_video = video[:, :, pt_lo:pt_hi].contiguous()
        audio_start = round(start_frame * core.FRAME_RESCALE)
        audio_end = min(audio.shape[-1], round(end_frame * core.FRAME_RESCALE))
        chunk_audio = audio[..., audio_start:audio_end].contiguous()
        if pad_t:
            pf_lo = core.frames_for_tokens(pt_lo)
            pf_hi = core.frames_for_tokens(pt_hi)
            pa_s = round(pf_lo * core.FRAME_RESCALE)
            pa_e = min(audio.shape[-1], max(pa_s + 1, round(pf_hi * core.FRAME_RESCALE)))
            pad_audio = audio[..., pa_s:pa_e].contiguous()
            chunk_latent = {
                "samples": comfy.nested_tensor.NestedTensor((chunk_video, pad_audio))
            }
        else:
            chunk_latent = {
                "samples": comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))
            }

        chunk_video = _upscale_chunk(learned, chunk_latent, plan)
        if pad_t:
            _trim_lo = start_token - pt_lo          # 头部要裁掉的 padding token 数
            _n_tok = end_token - start_token        # 窗口 own token 数
            chunk_video = chunk_video[:, :, _trim_lo:_trim_lo + _n_tok].contiguous()
        chunk_conditioning = core.reanchor_conditioning(
            conditioning, start_frame, end_frame, tuple(chunk_video.shape[-2:])
        )
        # Overlap > 0: pin this window's FIRST token on the previous window's
        # output (upstream inserts it as minimax_keyframes[0] and applies
        # anchor_strength as a noise-aug factor), so the sampler continues from
        # what was already generated instead of starting cold at a hard cut.
        # With overlap 0 there is nothing for it to anchor to - upstream raises
        # "previous chunk does not reach the next chunk anchor" - so it stays off.
        if seg_overlap and accumulated is not None:
            if int(anchor_tokens) > 1:
                chunk_conditioning = _anchor_conditioning_multi(
                    core,
                    chunk_conditioning,
                    accumulated,
                    start_frame,
                    float(plan.get("anchor_strength", 0.999)),
                    int(anchor_tokens),
                )
            else:
                chunk_conditioning = core.anchor_conditioning(
                    chunk_conditioning,
                    accumulated,
                    start_frame,
                    float(plan.get("anchor_strength", 0.999)),
                )

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
        if dump_latents:
            _w = len(segment_reports)
            _dump_av_latent(
                os.path.join(dump_dir, f"upscaled_W{_w}.pt"),
                chunk_video,
                chunk_audio,
                {
                    "stage": "upscaled_input",
                    "window": _w,
                    "tokens": [int(start_token), int(end_token)],
                    "frames": [int(start_frame), int(end_frame)],
                },
            )
            _sampled_tensors = getattr(sampled, "tensors", None)
            _dump_av_latent(
                os.path.join(dump_dir, f"window_W{_w}.pt"),
                _sampled_tensors[0] if _sampled_tensors is not None else sampled,
                _sampled_tensors[1] if _sampled_tensors is not None and len(_sampled_tensors) > 1 else None,
                {
                    "stage": "sampled_output",
                    "window": _w,
                    "tokens": [int(start_token), int(end_token)],
                    "frames": [int(start_frame), int(end_frame)],
                },
            )
        # Blend or freeze?  Both are upstream paths over the SAME overlap frames:
        #   _append_video                  -> linear crossfade across the overlap
        #   _append_video_guarded_overlap  -> keep the published frames verbatim
        # We defaulted to the frozen one to avoid ghosting when both sides are
        # moving.  But freezing means the boundary is still a hard join, which
        # reads as a cut - so it is now a switch, and blending is available for
        # material where ghosting is not a risk (still or slow shots).
        if seg_overlap and not (bool(seam_blend) or bool((plan.get("hardcut") or {}).get("seam_blend", False))):
            # guarded overlap: the first `locked_overlap` tokens stay exactly as
            # the previous window published them, the remaining `transition`
            # tokens take this window's fresh sample, then the rest is appended.
            # upstream counts TOKENS here, we hold a frame count: 17 frames = 5
            # tokens (a token covers 4 frames, plus one exclusive frame per block)
            _locked_tokens = (
                max(0, min(int(locked_overlap), seg_overlap)) if locked_overlap
                else seg_overlap
            ) * 5 // FRAME_GRID
            accumulated, _ov, _tr = core._append_video_guarded_overlap(
                accumulated, sampled, start_token, _locked_tokens
            )
            # the published seam (previous output meets this window's fresh
            # sample) sits where the frozen guard ends
            seam_marks.append(int(start_token) + int(_locked_tokens))
            print(f"[HardCut]   window {len(segment_reports)}: anchored prefix "
                  f"{seg_overlap} frames at {start_frame} (frozen)", flush=True)
        else:
            if seg_overlap:
                # crossfade smears the join across the whole overlap; centre
                # any later seam window on the middle of that blend
                seam_marks.append(
                    int(start_token) + (seg_overlap * 5 // FRAME_GRID) // 2
                )
                print(f"[HardCut]   window {len(segment_reports)}: crossfaded overlap "
                      f"{seg_overlap} frames at {start_frame}", flush=True)
            accumulated = core._append_video(accumulated, sampled, start_token)
        prev_end_frame = int(end_frame)
        # The probe is optional; when it is off, simply do not run it.  (An
        # earlier version raised a private exception to skip the block and
        # caught it below - but the class was never defined anywhere, so
        # turning the switch off took the whole node down with a NameError.
        # Plain control flow has no such failure mode.)
        if show_memory_log:
            try:
                _free, _total = torch.cuda.mem_get_info()
                print(
                    f"[HardCut]   seg {len(segment_reports)} done: "
                    f"alloc {torch.cuda.memory_allocated()/2**30:.2f}GB  "
                    f"reserved {torch.cuda.memory_reserved()/2**30:.2f}GB  "
                    f"device-free {_free/2**30:.2f}GB",
                    flush=True,
                )
                if _empty_between:
                    comfy.model_management.soft_empty_cache()
                    _free2, _ = torch.cuda.mem_get_info()
                    if _free2 - _free > 32 * 2**20:
                        print(f"[HardCut]   cache released: +{(_free2-_free)/2**30:.2f}GB free",
                              flush=True)
            except Exception as _mm_exc:
                print(f"[HardCut]   mem probe skipped: {_mm_exc}", flush=True)
        segment_reports.append(
            {
                "index": len(segment_reports),
                "frames": [start_frame, end_frame],
                "tokens": [start_token, end_token],
                "length_frames": end_frame - start_frame,
            }
        )

    seam_redenoise_entries = None
    redenoise_gate_info = None
    if seam_redenoise and seam_marks:
        if seam_redenoise_gate == "auto":
            # 策略树 B1 前置门 v1：闹缝跳过（E-3 实测闹处重去噪注入伪纹理）
            eligible, redenoise_gate_info = gate_seam_redenoise(profile, seam_marks)
            _kept = [t for t in seam_marks if t in set(eligible)]
            _skip = [t for t in seam_marks if t not in set(eligible)]
            print(
                f"[HardCut] seam re-denoise AUTO gate "
                f"(busy >= median x {REDENOISE_BUSY_SKIP_RATIO:g} -> skip): "
                f"redenoise {_kept} / skip {_skip}",
                flush=True,
            )
            for g in redenoise_gate_info:
                print(
                    f"[HardCut]   seam tok {g['token']}: busy {g['busy_ratio']} -> "
                    + ("SKIP（闹处重去噪会注入伪纹理，E-3 实测）" if g["gated"]
                       else "redenoise"),
                    flush=True,
                )
            if not profile:
                print(
                    "[HardCut]   ⚠ 无 latent profile（auto_seam_hunt 关着？）"
                    "-> busy 度测不了，全部放行（保持旧语义，不静默少干活）",
                    flush=True,
                )
            if (seam_redenoise_frames or "").strip():
                print("[HardCut]   note: auto 门控下 seam_redenoise_frames 被忽略",
                      flush=True)
            seam_marks = _kept
        else:
            only_frames = None
            raw = (seam_redenoise_frames or "").strip()
            if raw:
                only_frames = []
                for part in raw.replace(";", ",").split(","):
                    part = part.strip()
                    if part:
                        only_frames.append(int(part))
        if seam_marks:
            print(
                f"[HardCut] seam re-denoise: {len(seam_marks)} seam(s) at tokens "
                f"{seam_marks}, window {seam_window_tokens} tok, lock {seam_lock_tokens} tok",
                flush=True,
            )
            accumulated, seam_redenoise_entries = _redenoise_seam_windows(
                core,
                sampling,
                accumulated,
                audio,
                conditioning,
                model,
                noise,
                sampler,
                sigmas,
                negative,
                cfg,
                seam_marks,
                int(seam_window_tokens),
                int(seam_lock_tokens),
                None,
                global_video_noise,
                global_audio_noise,
            )
        else:
            print("[HardCut] seam re-denoise: 门控后没有剩余缝，跳过重去噪",
                  flush=True)
    elif seam_redenoise:
        print("[HardCut] seam re-denoise requested but no interior seams recorded", flush=True)

    if dump_latents:
        _dump_av_latent(
            os.path.join(dump_dir, "accumulated.pt"),
            accumulated,
            audio,
            {
                "stage": "assembled_final",
                "anchor_tokens": int(anchor_tokens),
                "frame_grid": int(FRAME_GRID),
                "segments": [
                    [int(s), int(sf), int(e), int(ef)]
                    for s, sf, e, ef in segments
                ],
                "note": "seed 见工作流 noise 节点；帧<->token 映射用 core.tokens_for_frames / frames_for_tokens",
            },
        )
    output = {"samples": comfy.nested_tensor.NestedTensor((accumulated, audio))}
    report = {
        "schema": "h3.hardcut.upscale.v1",
        "status": "completed",
        "segment_count": len(segments),
        "unequal_lengths": bool(segment_frames),
        "calm_boundaries": list(calm_boundaries) if calm_boundaries else None,
        "calm_overlaps": list(calm_overlaps) if calm_overlaps else None,
        "lengths": [r["length_frames"] for r in segment_reports],
        "segments": segment_reports,
    }
    if seam_hunt is not None:
        report["seam_hunt"] = seam_hunt
    if seam_redenoise_entries is not None:
        report["seam_redenoise"] = seam_redenoise_entries
    if redenoise_gate_info is not None:
        report["seam_redenoise_gate"] = {
            "mode": "auto",
            "busy_skip_ratio": REDENOISE_BUSY_SKIP_RATIO,
            "seams": redenoise_gate_info,
        }
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
                        "开启后：在时间轴找 latent 的突变：每个 token 同时按**自身邻域**与**全片基准**打分，\n"
                        "取全片基准最高者 —— 转镜自己的肩峰会抬高邻域中位数，只看局部反而会把它压下去，\n"
                        "结果被两格之外安静区的偶然波动抢走（实测：计划 68、真转镜 ~66，却选中 85）。\n"
                        "再把该突变**吸附到最近的独占帧（17k）**上：窗口起点必须是独占帧，\n"
                        "否则第二个窗口首 token 会崩（实测同 seed：起点 68 正常 / 起点 69 崩）。\n"
                        "报告给出 measured_turn_frame（测得转镜帧）与 boundary_frame（最终边界）。\n"
                        "会覆盖 plan 里的 segment_frames。"
                    ),
                ),
                io.Boolean.Input(
                    "show_memory_log",
                    default=True,
                    tooltip=(
                        "每个分块采样结束后打印真实显存：alloc（真正占用）/ reserved（池）"
                        "/ device-free（卡上剩余），并在段间释放缓存后报告释放量。"
                        "用来区分「显存逐段上涨」是哪一类：alloc 涨=有引用没放，"
                        "reserved 涨=显存池碎片，只有 device-free 掉=pinned/其它进程。"
                        "关掉只保留必要的进度日志。"
                    ),
                ),
                io.Boolean.Input(
                    "seam_redenoise",
                    default=False,
                    tooltip=(
                        "E-3 实验（路线①缝窗重去噪）：所有分块采样完之后，对每个缝取一个"
                        "短窗（默认 10 token ≈ 34 帧，缝居中），两端锁定为已发布 latent"
                        "（denoise_mask 时间维 0），中段自由重去噪。锁端在每一步被以"
                        "0.999 键帧语义注入（KSamplerX0Inpaint 原生 RePaint 机制），"
                        "过渡由模型在双侧上下文里自己补出——是生成，不是混合。"
                        "默认关闭，行为与旧版完全一致。"
                    ),
                ),
                io.String.Input(
                    "seam_redenoise_frames",
                    default="",
                    tooltip=(
                        "只重去噪指定帧附近的缝（逗号分隔，如 \"187\" 或 \"187,254\"）。"
                        "留空 = 全部内缝。帧号按 17 帧 token 网格吸附到最近的缝。"
                    ),
                ),
                io.Int.Input(
                    "seam_window_tokens",
                    default=10,
                    min=4,
                    max=30,
                    tooltip="缝窗宽度（token，1 token ≈ 3.4 帧），缝居中。",
                ),
                io.Int.Input(
                    "seam_lock_tokens",
                    default=3,
                    min=1,
                    max=10,
                    tooltip="缝窗两端各锁定多少 token（≈10 帧/侧）为已发布 latent。",
                ),
                io.Int.Input(
                    "anchor_tokens",
                    default=1,
                    min=1,
                    max=5,
                    tooltip=(
                        "E-2 实验（路线②多 token 锚定）：overlap 锚定的 keyframe 宽度。"
                        "1 = 上游原版（单 token，StreamingT2V 点名批评的做法）；"
                        "2/5 = 把锚扩到前块输出的前 2/5 个 token（≈7/17 帧）。"
                        "keyframe 格式原生支持多 token，仅切片宽度不同。"
                        "需 overlap > 0 才生效（无 overlap 无锚可锚）。"
                    ),
                ),
                io.Combo.Input(
                    "seam_redenoise_gate",
                    options=["off", "auto"],
                    default="off",
                    tooltip=(
                        "每缝门控（策略树 §4.1 B1 前置门 v1，见 docs/TOKEN_RESEARCH.md）。\n"
                        "off = 手动模式（现状）：seam_redenoise 开 = 全部 overlap 缝重去噪，"
                        "可用 seam_redenoise_frames 挑缝。\n"
                        "auto = 按锚定窗 busy 度自动挑缝：缝邻域 latent 变化分"
                        "（与 calm 搜索同口径 max(global, jerk)，±2 token ≈ ±7 帧）"
                        "≥ 全片中位 × 1.5 的缝跳过 —— E-3 实测闹处重去噪会注入伪纹理；"
                        "其余缝重去噪（静处有效：E-3 缝 187 台阶 5.30x→1.72x）。\n"
                        "需 auto_seam_hunt 开着才有测量；关着退化为全部放行并打日志。\n"
                        "报告 seam_redenoise_gate.seams 有每缝 busy 度，供标定阈值用。"
                    ),
                ),
                io.Boolean.Input(
                    "dump_latents",
                    default=False,
                    optional=True,
                    tooltip=(
                        "诊断用：把二采中间 latent 落盘（每窗 upscale 输入 + 每窗采样输出 + "
                        "最终装配结果，fp16），供离线解码实验台使用（熔化机制判决 V0-V4）。"
                        "默认关，正常跑零开销、零数值影响。"
                    ),
                ),
                io.String.Input(
                    "dump_dir",
                    default="D:\\comfyui\\_hardcut_work\\latent_dump\\",
                    optional=True,
                    tooltip="dump_latents 的输出目录（建议带运行标签，如 ...\\latent_dump\\A2_20260925\\）。",
                ),
                io.Int.Input(
                    "upscale_pad_tokens",
                    default=3,
                    min=0,
                    max=8,
                    optional=True,
                    tooltip=(
                        "熔化修复（2026-09-26 验收：00085 熔化区 0.76/0.57x → 0.98/0.99x，"
                        "用户目视确认达成目标）：上采样分块时间重叠。3D upscaler 逐窗独立处理，"
                        "分块尾时间感受野单侧 → 尾部 ~8 帧轻度软，二采再加深为锁入口脱焦带。"
                        "此参数让 upscale 输入向两侧各借 N 个一采 token（全片连续无接缝），"
                        "上采样后裁回窗口范围。默认 3（≈10 帧，已验收）；0=旧行为。"
                    ),
                ),
                io.Boolean.Input(
                    "seam_blend",
                    default=False,
                    optional=True,
                    tooltip=(
                        "锚定缝用 latent 域线性交叉淡化（普通 overlap）替代冻结式："
                        "重叠 17 帧内两窗 latent 线性溶解后再解码。静止/慢速镜头适用"
                        "（无重影风险，背景跳变摊成柔和过渡）；运动镜头会重影。"
                        "★ 放在本节点（下游）而在规划节点上——改这里**不会击穿一采缓存**"
                        "（规划节点在一采上游链里，改它的开关会触发重采样）。"
                    ),
                ),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, **kwargs):
        output, report = execute(**kwargs)
        return io.NodeOutput(output, report)
