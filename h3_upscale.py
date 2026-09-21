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
    deviation_threshold: int = 17,
    min_sep: int = 34,
    grid: int = FRAME_GRID,
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
    for row in profile or ():
        try:
            idx, _local, glob = int(row[0]), row[1], float(row[2])
        except (TypeError, IndexError, ValueError):
            continue
        score[idx] = glob
    entry_by_cut = {}
    for e in aligned or ():
        try:
            entry_by_cut[int(e.get("planned_cut"))] = e
        except (TypeError, ValueError):
            continue

    boundaries, overlaps, notes = [], [], []
    for cut in planned:
        cut = int(cut)
        entry = entry_by_cut.get(cut) or {}
        b = entry.get("boundary_frame")
        if b is not None and abs(int(b) - cut) <= int(deviation_threshold):
            boundaries.append(int(b))
            overlaps.append(0)
            notes.append(f"cut {cut}: hunt reliable (boundary {b}) -> hard cut")
            continue

        # candidates: exclusive frames inside the window, away from the others
        lo, hi = max(grid, cut - int(window)), min(frame_count - grid, cut + int(window))
        cands = [f for f in range(lo, hi + 1, grid) if f % grid == 0]
        cands = [
            f for f in cands
            if all(abs(f - other) >= int(min_sep) for other in planned if int(other) != cut)
        ]
        if not cands:
            boundaries.append(cut)
            overlaps.append(0)
            notes.append(f"cut {cut}: no calm candidate in window -> keep plan (hard cut)")
            continue

        def tok_of(frame):
            # frames here are exclusive anchors: frame 17k starts token 5k
            # (FRAME_PER_TOKEN=(1,4,4,4,4) puts a 1-frame token every 17 frames)
            return int(frame) // int(grid) * 5

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


def _latent_change_profile(video, win: int = 2) -> list:
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
    d = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
    n = int(d.numel())
    gmed = float(d.median())
    if gmed <= 0:
        return []
    out = []
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        med = float(d[lo:hi].median())
        if med <= 0:
            continue
        out.append((i, float(d[i]) / med, float(d[i]) / gmed))
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
        window = [
            (idx, lr, gr) for idx, lr, gr in profile
            if 0 < int(idx) + 1 < int(video_tokens)
            and abs(frame_of(idx) - cut) <= tolerance
        ]
        if not window:
            continue
        # rank by the GLOBAL score: the local one suppresses the turn itself
        window.sort(key=lambda x: -x[2])
        peak_idx, peak_local, peak_ratio = window[0]
        peak_frame = frame_of(peak_idx)
        top = [[frame_of(i), round(float(gr), 2)] for i, _lr, gr in window[:3]]
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
        profile = _latent_change_profile(video)
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
                deviation_threshold=int(_hc.get("deviation_threshold", 17)),
            )
            segment_frames = calm_boundaries
            for _n in _calm_notes:
                print(f"[HardCut]   calm: {_n}", flush=True)

    # ---- windowing: explicit (possibly unequal) first, then the equal paths ----
    # overlap tokens: when > 0 each window's START is pulled back so the sampler
    # can anchor its first token on the previous window (see anchor_conditioning
    # below).  The knob is free to set, but the value that actually reaches the
    # sampler is clamped below one full window - an overlap of a whole window
    # would leave nothing new to generate.
    ov_input = max(0, int(plan.get("temporal_overlap_frames", 0)))
    _chunk = int(plan.get("temporal_chunk_frames") or 0)
    ov_tokens = ov_input
    if _chunk > 0 and ov_input >= _chunk:
        ov_tokens = max(0, _chunk - 1)
    locked_overlap = max(
        0, min(int(plan.get("locked_overlap_tokens", ov_tokens)), ov_tokens)
    )
    if ov_input:
        print(
            f"[HardCut]   overlap: requested {ov_input}f -> effective {ov_tokens}f "
            f"(chunk {_chunk}f), locked {locked_overlap}f"
            + ("   [clamped below one window]" if ov_tokens != ov_input else ""),
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
            accumulated, _ov, _tr = core._append_video_guarded_overlap(
                accumulated, sampled, start_token,
                max(0, min(int(locked_overlap), seg_overlap)) if locked_overlap else seg_overlap,
            )
            print(f"[HardCut]   window {len(segment_reports)}: anchored prefix "
                  f"{seg_overlap} frames at {start_frame}", flush=True)
        else:
            accumulated = core._append_video(accumulated, sampled, start_token)
        prev_end_frame = int(end_frame)
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
                io.Int.Input(
                    "auto_seam_sensitivity",
                    default=20,
                    min=5,
                    max=80,
                    step=1,
                    tooltip=(
                        "（已停用，保留兼容）旧版用固定阈值检测（20 = 2.0 倍），在打斗类"
                        "内容上会把真转镜漏掉、反而抓到动作重击。现改为：在每个计划切点的"
                        "容差窗内直接取 latent 变化最强的 token 作为边界 —— 无阈值。"
                        "低于 1.1 倍视为平坦，保持原计划边界。"
                    ),
                ),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, **kwargs):
        output, report = execute(**kwargs)
        return io.NodeOutput(output, report)
