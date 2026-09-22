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

import torch
import comfy.model_management

import comfy.nested_tensor

from comfy_api.latest import io

from .bridge import PLAN_TYPE_STRING, find_upstream_module
from .hardcut_math import FPS, FRAME_GRID

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
):
    """Decide, per planned cut, WHERE to actually put the window boundary.

    A cut whose hunt entry carries a boundary close to the plan is trustworthy
    (the model really turns there) -> keep it and hard-cut (overlap 0).

    Anything else (hunt rejected it, or it points far away) means we are about
    to cut through continuous content, so instead of cutting at the plan we look
    for the CALMEST exclusive frame within `window` of it: the token whose
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
                return (
                    [int(c) for c in planned],
                    [0] * len(planned),
                    [f"clip too flat for a search (jerk contrast {_contrast:.2f} < "
                     f"{abstain_below:.2f}) -> keep every planned cut as a hard cut"],
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
        # the FINAL boundary lands from where the model ACTUALLY turned - not how
        # far the model drifted from the plan.  The hunt already snaps the
        # boundary onto the exclusive-frame grid, so a 15-frame plan error can
        # end up 2 frames off (plan 187, real turn 202 -> boundary 204).
        # Only when that residual exceeds seam_tolerance do we stop trusting the
        # hard cut and switch the seam to an anchored overlap.
        _m = entry.get("measured_turn_frame")
        if b is not None and _m is not None:
            _dev = abs(int(b) - int(_m))
            _dev_src = f"boundary {b} vs measured {_m}"
        elif b is not None:
            # no measured turn reported: fall back to the plan distance
            _dev = abs(int(b) - cut)
            _dev_src = f"boundary {b} vs plan {cut}"
        else:
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
            boundaries.append(cut)
            overlaps.append(0)
            notes.append(f"cut {cut}: no calm candidate in window -> keep plan (hard cut)")
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
        if score.get(tok_of(best), float("inf")) == float("inf"):
            boundaries.append(cut)
            overlaps.append(0)
            notes.append(f"cut {cut}: profile has no score here -> keep plan (hard cut)")
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


def _align_to_profile(profile, planned, tolerance: int, video_tokens: int):
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

    aligned, boundary_tokens = [], []
    for cut in planned:
        # index, not unpack: the profile grew columns (|d3|, persistence)
        window = [
            (int(row[0]), row[1], float(row[2]),
             float(row[4]) if len(row) > 4 else 1.0)
            for row in profile
            if 0 < int(row[0]) + 1 < int(video_tokens)
            and abs(frame_of(int(row[0])) - cut) <= tolerance
        ]
        if not window:
            continue
        # Rank by local_change x persistence.  The global score is still the
        # local-change term (the LOCAL variant suppresses a turn, see above); the
        # persistence term is what separates a real cut from a sharp transient.
        # The 0.25 floor keeps the old ordering if persistence is unavailable
        # (older profiles, or the switch off) so behaviour degrades, not breaks.
        window.sort(key=lambda x: -(x[2] * (0.25 + 0.75 * x[3])))
        peak_idx, peak_local, peak_ratio, _peak_pers = window[0][:4]
        peak_frame = frame_of(peak_idx)
        top = [
            [frame_of(i), round(float(gr), 2), round(float(pr), 2)]
            for i, _lr, gr, pr in window[:3]
        ]
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
        if final_frame != peak_frame:
            entry["measured_turn_frame"] = peak_frame
        if note:
            entry["note"] = note
        aligned.append(entry)
    return aligned, boundary_tokens



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

    # ---- optional: put the window boundary ON the model's own cut ----
    # The upscaler never sees the prompt, so the model's shot change is only
    # observable in the latent we are holding.  Measuring it here costs no VAE
    # decode, and placing the split before it keeps the seam on continuous
    # content — the model performs the cut itself, inside the window.
    planned = [int(c) for c in ((plan.get("hardcut") or {}).get("segment_frames") or [])]
    tolerance = max(0, int((plan.get("hardcut") or {}).get("seam_tolerance", 17)))
    segment_frames = planned or None
    seam_hunt = None
    if auto_seam_hunt:
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
            profile, planned, tolerance, int(video.shape[2])
        )
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
            seam_hunt_note = "no latent change found within tolerance of any planned cut"
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
            _jr = sorted(
                ((int(r[0]), float(r[3])) for r in profile if len(r) > 3),
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
                    "top-6 |d3| ratios, token->frame via token//5*17. "
                    "A peak followed by a sharp fall means the burst is being "
                    "smeared (motion overload); a flat top means sustained motion."
                )
                seam_hunt["profile_len"] = len(profile)
        except Exception as _pe:
            seam_hunt["jerk_peaks_error"] = str(_pe)

    # ---- adaptive: if the hunt could not vouch for a cut, move that boundary to
    # the CALMEST frame nearby and give that seam an anchored overlap, instead of
    # cutting through continuous content.  Needs the hunt's profile, so it only
    # runs when auto_seam_hunt is on.
    calm_boundaries = None
    calm_overlaps = None
    if auto_seam_hunt and planned and isinstance(plan.get("hardcut"), dict):
        _hc = plan["hardcut"]
        if _hc.get("auto_calm_search"):
            calm_boundaries, calm_overlaps, _calm_notes = find_calm_boundaries(
                profile, planned, aligned, frame_count,
                window=int(_hc.get("calm_search_window", 34)),
                overlap_frames=int(_hc.get("calm_overlap_frames", 17)),
                seam_tolerance=int(_hc.get("seam_tolerance", 17)),
                policy=str(_hc.get("calm_policy", "calm_overlap")),
                abstain_below=float(_hc.get("calm_abstain_below", 0.0)),
            )
            segment_frames = calm_boundaries
            for _n in _calm_notes:
                print(f"[HardCut]   calm: {_n}", flush=True)

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
        print(
            f"[HardCut]   overlap: fallback {ov_input}f -> {ov_tokens}f "
            f"(chunk {_chunk}f), locked {locked_overlap}f"
            + ("   [clamped below one window]" if ov_tokens != ov_input else "")
            + ("   | per-seam values below take precedence"
               if calm_overlaps is not None else ""),
            flush=True,
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
        src_tag = ", ".join(
            f"{b}{'' if b in hunted else '*'}" for b in boundaries
        )
        print(
            f"[HardCut] planned_cuts={planned or '-'} "
            f"boundary_frames=[{src_tag or '-'}] "
            f"segments={len(segments)} lengths={lengths} longest={longest}f"
            + ("   (* = kept on the planned frame, hunt found no turn there)"
               if any(b not in hunted for b in boundaries) else ""),
            flush=True,
        )
        if seam_hunt:
            print(
                f"[HardCut]   tolerance={seam_hunt.get('tolerance_frames')}f "
                f"hunt={'on' if auto_seam_hunt else 'off'}",
                flush=True,
            )
            for entry in seam_hunt.get("aligned") or []:
                moved = entry.get("moved")
                planned_cut = entry.get("planned_cut")
                boundary = entry.get("boundary_frame")
                if boundary is None:
                    head = (f"[HardCut]   cut planned={planned_cut} -> boundary=None "
                            f"(NOT accepted, moved={moved}")
                else:
                    offset = int(boundary) - int(planned_cut)
                    head = (f"[HardCut]   cut planned={planned_cut} -> "
                            f"boundary={boundary} (offset={offset:+d}f, moved={moved}")
                tail = (
                    (f", measured={entry.get('measured_turn_frame')}"
                     if entry.get("measured_turn_frame") is not None else "")
                    + f", ratio={entry.get('ratio')})"
                )
                print(head + tail, flush=True)
            if seam_hunt.get("note"):
                print(f"[HardCut]   note: {seam_hunt['note']}", flush=True)
    except Exception as _log_exc:  # pragma: no cover - logging must never fail the run
        print(f"[HardCut] log error: {_log_exc}", flush=True)

    accumulated = None
    segment_reports = []
    prev_end_frame = None
    for start_token, start_frame, end_token, end_frame in segments:
        # how much this window re-reads from the published output: > 0 only when
        # a seam asked for an anchored prefix (the calm search sets it per cut)
        seg_overlap = max(0, (prev_end_frame or 0) - int(start_frame)) if prev_end_frame is not None else 0
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
        # Overlap > 0: pin this window's FIRST token on the previous window's
        # output (upstream inserts it as minimax_keyframes[0] and applies
        # anchor_strength as a noise-aug factor), so the sampler continues from
        # what was already generated instead of starting cold at a hard cut.
        # With overlap 0 there is nothing for it to anchor to - upstream raises
        # "previous chunk does not reach the next chunk anchor" - so it stays off.
        if seg_overlap and accumulated is not None:
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
        if seg_overlap:
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
            print(f"[HardCut]   window {len(segment_reports)}: anchored prefix "
                  f"{seg_overlap} frames at {start_frame}", flush=True)
        else:
            accumulated = core._append_video(accumulated, sampled, start_token)
        prev_end_frame = int(end_frame)
        try:
            if not show_memory_log:
                raise _SkipProbe()
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
        except _SkipProbe:
            pass
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
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, **kwargs):
        output, report = execute(**kwargs)
        return io.NodeOutput(output, report)
