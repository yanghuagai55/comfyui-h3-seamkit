# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Hard-cut planning math for MiniMax H3 two-pass chunked upscale.

Pure Python, no torch / numpy dependency — safe to run standalone:

    python hardcut_math.py 8 4.25
    python hardcut_math.py 15 5,10 --mp 1.5

Core idea
---------
The H3 second pass can only split time into **equal-length windows** (the
segment stride is constant), so a "hard cut" is expressed as:

    temporal_overlap_frames = 0        -> zero blending between windows
    temporal_chunk_frames   = L (17n)  -> each window is L frames long

With overlap = 0 every window is sampled completely independently and the
results are appended back-to-back with no cross-fade, no latent endpoint
shift and no colour mix.  The join is a cut, exactly like a film edit.

That reframes the seam from a defect into intended grammar — provided the
prompt tells the model to *cut* there (see the R2V prompt engineering doc).

Grid facts (must not be broken)
-------------------------------
* Frames of a clip must be 17n + 5  (24 fps, FRAME_PER_TOKEN=(1,4,4,4,4))
* `temporal_chunk_frames % 17 == 0` and `temporal_overlap_frames % 17 == 0`
* `temporal_overlap_frames < temporal_chunk_frames`
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

FPS = 24
FRAME_GRID = 17          # window boundaries must be multiples of this
FRAME_OFFSET = 5         # clip length must be 17n + 5
MAX_SECONDS = 15.0       # requested ceiling for this helper
MIN_TAIL_SECONDS = 2.0   # a trailing window shorter than this is not worth a cut

# 8 GB empirical load anchors (frames x megapixels of the second-pass canvas)
# 2026-09-20 re-measured WITH the quality LoRAs enabled: 191 now OOMs, ~180 is
# the practical ceiling.  (The older 210 / 236.2 anchors were measured on the
# bare model - keep them in mind when you disable the LoRAs again.)
LOAD_PASS = 180.0        # known-good with quality LoRAs on
LOAD_FAIL = 191.0        # measured CUDA OOM with the LoRAs on
# (imported by h3_upscale.py so the calm search can re-check the guard after it
#  moves a boundary - see D2 in CODE_REVIEW_20260922.md)


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------

def frames_for_seconds(seconds: float) -> int:
    """Round a duration up onto the 17n+5 frame grid (matches H3 conditioning)."""
    base = int(round(float(seconds) * FPS))
    return base + (FRAME_OFFSET - base % FRAME_GRID) % FRAME_GRID


def seconds_for_frames(frames: int) -> float:
    return frames / FPS


def resolution_for(
    megapixels: float,
    width_ratio: float,
    height_ratio: float,
    multiple: int = 32,
) -> tuple[int, int]:
    """Megapixels + aspect ratio -> (width, height), pixel-for-pixel identical to
    ComfyUI's own `ResolutionSelector` (`comfy_extras/nodes_resolution.py`).

    Kept dependency-free so the auto node can replace that node without asking
    the caller to pass a canvas size.
    """
    multiple = max(1, int(multiple or 1))
    total = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total / (float(width_ratio) * float(height_ratio)))
    width = round(width_ratio * scale / multiple) * multiple
    height = round(height_ratio * scale / multiple) * multiple
    return int(width), int(height)


def timecode(frame: int) -> str:
    """Frame -> `MM:SS.mmm` exactly as the official R2V prompt template wants."""
    total_ms = int(round(frame / FPS * 1000))
    mm, rest = divmod(total_ms, 60_000)
    ss, ms = divmod(rest, 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


_ANY_STAMP_RE = re.compile(r"(\d{1,3}):(\d{2})\.(\d{3})")


def shift_shot_times(prompt: str, shift_frames: int) -> tuple[str, list[dict]]:
    """Move every declared shot change by `shift_frames` frames, wherever it is quoted.

    The model does not turn exactly where it is told: measured offsets run from
    -10 to +1 frames and depend on the content and the seed.  The executor, on the
    other hand, can only split on the token grid (1-4 frame granularity), so
    asking the model for a slightly different time is the one lever with a reach
    finer than the grid — and it belongs here rather than in the prompt text, so
    the model that *writes* the prompt never has to be taught plugin mechanics.

    Only stamps matching a declared `[Shot N]` cut are moved; in-shot beats
    (e.g. "By 00:06.500 Takagi has been driven back") keep their values.  Every
    occurrence of a matched stamp is rewritten, because the summary, the shot
    line and the soundscape all quote the same string.

    Returns `(new_prompt, moved)`; `moved` is `[]` when nothing changed.
    """
    shift = int(shift_frames or 0)
    if not shift or not prompt:
        return prompt, []
    declared = {
        int(round(shot["cut_seconds"] * FPS))
        for shot in parse_shots(prompt)
        if shot.get("cut_seconds")
    }
    if not declared:
        return prompt, []

    moved: list[dict] = []
    out = prompt
    for literal in dict.fromkeys(m.group(0) for m in _ANY_STAMP_RE.finditer(prompt)):
        minutes, seconds, millis = _ANY_STAMP_RE.match(literal).groups()
        old_frame = int(
            round((int(minutes) * 60 + int(seconds) + int(millis) / 1000.0) * FPS)
        )
        if old_frame not in declared:
            continue
        new_frame = old_frame + shift
        if new_frame <= 0:
            continue
        new = timecode(new_frame)
        if new == literal:
            continue
        out = out.replace(literal, new)
        moved.append(
            {"old": literal, "new": new, "old_frame": old_frame, "new_frame": new_frame}
        )
    if not moved:
        return prompt, []
    moved.sort(key=lambda row: row["old_frame"])
    return out, moved


# --------------------------------------------------------------------------
# segmentation (mirrors T8 compute_temporal_segments for overlap = 0)
# --------------------------------------------------------------------------

def segments_for(total_frames: int, chunk: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame), ...] for the equal-stride windowing."""
    if chunk <= 0 or overlap < 0 or chunk <= overlap:
        raise ValueError("chunk must be positive and larger than overlap")
    hop = chunk - overlap
    out: list[tuple[int, int]] = []
    index = 0
    while True:
        start = index * hop
        end = min(start + chunk, total_frames)
        out.append((start, end))
        if start + chunk >= total_frames:
            break
        index += 1
        if index > 512:                                   # paranoia guard
            raise RuntimeError("segmentation did not terminate")
    return out


def segment_lengths(total_frames: int, chunk: int, overlap: int = 0) -> list[int]:
    return [e - s for s, e in segments_for(total_frames, chunk, overlap)]


def cut_frames(total_frames: int, chunk: int, overlap: int = 0) -> list[int]:
    """Frames at which a new window starts = the hard-cut positions."""
    return [s for s, _ in segments_for(total_frames, chunk, overlap)][1:]


def reachable_cuts(total_frames: int) -> list[tuple[int, float, int]]:
    """Every legal single cut: (chunk, cut_seconds, segment_count)."""
    rows = []
    for chunk in range(FRAME_GRID, total_frames, FRAME_GRID):
        segs = segments_for(total_frames, chunk)
        if len(segs) < 2:
            continue
        rows.append((chunk, seconds_for_frames(chunk), len(segs)))
    return rows


# --------------------------------------------------------------------------
# chunk solving
# --------------------------------------------------------------------------

def parse_cut_points(text: str | Iterable[float]) -> list[float]:
    """`"4.25, 8.5"` or `[4.25, 8.5]` -> [4.25, 8.5] (deduped, sorted, positive)."""
    if text is None:
        return []
    if isinstance(text, str):
        raw = [p for p in text.replace(";", ",").split(",") if p.strip()]
        try:
            vals = [float(p) for p in raw]
        except ValueError as exc:
            raise ValueError(f"cut points must be numbers separated by commas: {text!r}") from exc
    else:
        vals = [float(v) for v in text]
    vals = sorted({v for v in vals if v > 0})
    return vals


def solve_chunk(
    total_frames: int,
    targets: Sequence[float],
    min_tail_frames: int | None = None,
) -> dict:
    """Pick the legal window length whose cuts land closest to `targets`.

    The window stride is constant, so the n-th cut always sits at n * chunk.
    We therefore minimise the summed frame error between `targets` and
    `(n * chunk)/FPS` over every legal chunk, restricted to chunks that
    produce exactly len(targets) + 1 windows and leave a usable tail.
    """
    n_cuts = len(targets)
    if n_cuts == 0:
        raise ValueError("at least one cut point is required")
    if min_tail_frames is None:
        min_tail_frames = int(round(MIN_TAIL_SECONDS * FPS))

    wanted_segments = n_cuts + 1
    best = None
    for chunk in range(FRAME_GRID, total_frames, FRAME_GRID):
        segs = segments_for(total_frames, chunk)
        if len(segs) != wanted_segments:
            continue
        if segs[-1][1] - segs[-1][0] < min_tail_frames:
            continue
        error = 0.0
        for n, target in enumerate(targets, start=1):
            error += abs(target - seconds_for_frames(n * chunk))
        candidate = (error, chunk, segs)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        # No window count matches — report the widest legal option instead of failing.
        raise ValueError(
            f"cannot produce {wanted_segments} windows for {total_frames} frames "
            f"({seconds_for_frames(total_frames):.2f}s); use fewer cut points"
        )

    error, chunk, segs = best
    actual = [seconds_for_frames(c) for c in cut_frames(total_frames, chunk)]
    return {
        "chunk": chunk,
        "segments": segs,
        "actual_cuts": actual,
        "targets": list(targets),
        "deviations_frames": [
            round((a - t) * FPS, 2) for a, t in zip(actual, targets)
        ],
        "max_error_seconds": error,
    }


def enumerate_plans(
    total_frames: int,
    min_tail_frames: int | None = None,
    per_count: int = 3,
) -> list[dict]:
    """Every legal windowing option, grouped by segment count.

    The reachable cut positions are discrete (multiples of 17 frames), so when
    a caller has a *target* time but not a fixed segment count, showing the
    menu beats chasing a value that cannot be hit.
    """
    if min_tail_frames is None:
        min_tail_frames = int(round(MIN_TAIL_SECONDS * FPS))
    by_count: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
    for chunk in range(FRAME_GRID, total_frames, FRAME_GRID):
        segs = segments_for(total_frames, chunk)
        tail = segs[-1][1] - segs[-1][0]
        if tail < min_tail_frames:
            continue
        by_count.setdefault(len(segs), []).append((chunk, segs))

    rows: list[dict] = []
    for count in sorted(by_count):
        candidates = by_count[count]
        # Prefer even divisions: tail length closest to the window length.
        candidates.sort(
            key=lambda item: (item[1][-1][1] - item[1][-1][0]) / item[0], reverse=True
        )
        for chunk, segs in candidates[:per_count]:
            rows.append(
                {
                    "segments": count,
                    "chunk": chunk,
                    "cuts": [seconds_for_frames(s) for s, _ in segs][1:],
                    "lengths": [e - s for s, e in segs],
                    "load_frames": max(e - s for s, e in segs),
                }
            )
    return rows


# --------------------------------------------------------------------------
# load estimate
# --------------------------------------------------------------------------

def estimate_load(
    segments: Sequence[tuple[int, int]], canvas_mp: float, overlap: int = 0
) -> dict:
    """Second-pass peak load = longest window (frames) x canvas megapixels.

    With overlap > 0 every window also re-reads `overlap` frames of the one
    before it, so the sampler's real work is longest + overlap frames.
    """
    longest = max(e - s for s, e in segments) + max(0, int(overlap))
    load = longest * float(canvas_mp)
    if load <= LOAD_PASS:
        verdict = "SAFE"
    elif load < LOAD_FAIL:
        verdict = "BORDERLINE"
    else:
        verdict = "LIKELY-OOM"
    return {
        "longest_window": longest,
        "load": load,
        "verdict": verdict,
        "vs_pass_anchor": load / LOAD_PASS,
    }


def max_canvas_mp(segments: Sequence[tuple[int, int]], overlap: int = 0) -> float:
    """Largest second-pass canvas that still fits the known-good load anchor."""
    longest = max(e - s for s, e in segments) + max(0, int(overlap))
    return LOAD_PASS / longest


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

CUT_SLOTS = 4            # how many cut inputs the plan node offers


def cuts_from_inputs(slots: Iterable[float] | None) -> list[float]:
    """Cut slots -> the times that really ask for a cut.

    A slot that is negative (the `-1` the node ships with) or zero means "no cut
    here", so an 8 s clip with a single cut is `[4.25, -1, -1, -1]`.
    """
    out: list[float] = []
    for value in slots or ():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            out.append(round(number, 3))
    return sorted(set(out))


def single_window_chunk(total_frames: int) -> int:
    """Smallest legal chunk that yields exactly ONE window — i.e. no cut."""
    return max(FRAME_GRID, math.ceil(total_frames / FRAME_GRID) * FRAME_GRID)


def snap_frame_grid(frames: Iterable[float] | None) -> list[int]:
    """Snap explicit cut frames onto the 17-frame token grid, ascending, deduped.

    Matches the executor's `_snap_frame` (nearest `frames_for_tokens(5k) = 17k`).

    NOTE: the executor's real boundary resolution is FINER than this — a window
    may start on ANY token edge (0,1,5,9,13,17,18,..., resolution 1-4 frames).
    New code should use `snap_token_edge`, which mirrors the executor exactly.
    """
    out = sorted(
        {
            max(FRAME_GRID, round(float(f) / FRAME_GRID) * FRAME_GRID)
            for f in frames or ()
            if f is not None and float(f) > 0
        }
    )
    return out


_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # H3 temporal compression, one 17f block


def frames_for_token_count(tokens: int) -> int:
    """Frames covered by the first `tokens` tokens (H3 [1,4,4,4,4] pattern)."""
    per = _FRAME_PER_TOKEN
    full, rest = divmod(max(0, int(tokens)), len(per))
    return full * sum(per) + sum(per[:rest])


def snap_token_edge(
    frames: Iterable[float] | None, video_tokens: int | None = None
) -> list[int]:
    """Snap cut frames onto the nearest TOKEN edge — mirrors the executor.

    Same math as `h3_upscale._snap_boundary`: token `t` covers
    `frames_for_token_count(t)` frames, so edges sit at 0,1,5,9,13,17,18,...
    (1-4 frame resolution).  Ties resolve to the LOWER edge, exactly like the
    executor's `min()` over ascending tokens.  A window boundary can only sit
    on these edges, so this is the honest prediction of where a cut lands.
    """
    vals = [float(f) for f in frames or () if f is not None and float(f) > 0]
    if not vals:
        return []
    if video_tokens is None:
        need = max(int(v) for v in vals)
        video_tokens = 1
        while frames_for_token_count(video_tokens) < need:
            video_tokens += 1
    choices = [
        (t, frames_for_token_count(t)) for t in range(0, int(video_tokens) + 1)
    ]
    out = sorted({min(choices, key=lambda c: abs(c[1] - v))[1] for v in vals})
    return [f for f in out if f > 0]


def chunk_ladder(
    total_frames: int,
    base_chunk: int,
    canvas_mp: float,
    span: int = 4,
    chunk_step: int = 0,
    min_tail_frames: int | None = None,
) -> list[dict]:
    """`chunk_step` -> one row per `base + step x 17` frames.

    The window stride slides with the chunk, so every step re-places the cuts.
    Steps that leave an unusable tail are kept and flagged instead of hidden —
    seeing *why* a step is unavailable is the point of the table.
    """
    if min_tail_frames is None:
        min_tail_frames = int(round(MIN_TAIL_SECONDS * FPS))
    rows: list[dict] = []
    for step in range(-span, span + 1):
        chunk = base_chunk + step * FRAME_GRID
        if chunk < FRAME_GRID:
            continue
        segs = segments_for(total_frames, chunk)
        tail = segs[-1][1] - segs[-1][0]
        longest = max(e - s for s, e in segs)
        rows.append(
            {
                "step": step,
                "chunk": chunk,
                "chunk_seconds": seconds_for_frames(chunk),
                "windows": len(segs),
                "cut_frames": [s for s, _ in segs][1:],
                "cuts": [seconds_for_frames(s) for s, _ in segs][1:],
                "lengths": [e - s for s, e in segs],
                "longest": longest,
                "tail": tail,
                "tail_ok": tail >= min_tail_frames,
                "load": longest * canvas_mp,
                "selected": step == chunk_step,
            }
        )
    return rows


def plan_hard_cut(
    total_seconds: float,
    cut_points: str | Iterable[float] = "",
    canvas_mp: float = 1.0,
    chunk_step: int = 0,
    segment_frames: Iterable[float] | None = None,
    overlap: int = 0,
) -> dict:
    """Full plan: duration, windowing, cuts, load.

    Two modes:

    * **explicit frames** (`segment_frames` given) — UNEQUAL window lengths.  Each
      frame is snapped to the nearest TOKEN edge (1-4 frame resolution, same as
      the executor's `_snap_boundary`) and used as a window boundary, so the
      first shot can be 68 frames and the next 124.  `chunk` then just means the
      longest window (used for the load estimate).
    * **equal stride** (default) — `cut_points` (seconds) solve a constant window
      length, and `chunk_step` walks it in 17-frame steps.
    """
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if total_seconds > MAX_SECONDS:
        raise ValueError(
            f"total_seconds {total_seconds} exceeds the supported ceiling "
            f"of {MAX_SECONDS}s"
        )

    total_frames = frames_for_seconds(total_seconds)
    targets = parse_cut_points(cut_points)
    chunk_step = int(chunk_step or 0)
    min_tail = int(round(MIN_TAIL_SECONDS * FPS))

    # Fine token-edge snapping (1-4f resolution), same as the executor — the
    # old 17-frame coarse grid silently moved legal edges like 115 onto 119.
    seg_frames = snap_token_edge(segment_frames)
    if seg_frames:
        seg_frames = [f for f in seg_frames if 0 < f < total_frames]
        if seg_frames:
            bounds = [0] + seg_frames + [total_frames]
            segs = [
                (bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            ]
            segs = [(s, e) for s, e in segs if e > s]
            frames = seg_frames
            actual = [seconds_for_frames(f) for f in frames]
            tail = segs[-1][1] - segs[-1][0] if segs else total_frames
            chunk = max((e - s for s, e in segs), default=total_frames)
            base_chunk = chunk
            cut_mode = "frames"
            deviations = []
            ladder: list[dict] = []
            alternatives: list[dict] = []
        else:
            seg_frames = []
    if not seg_frames:
        if targets:
            base_chunk = solve_chunk(total_frames, targets)["chunk"]
            cut_mode = "requested"
        else:
            base_chunk = single_window_chunk(total_frames)
            cut_mode = "none"
        chunk = max(FRAME_GRID, base_chunk + chunk_step * FRAME_GRID)
        segs = segments_for(total_frames, chunk)
        frames = cut_frames(total_frames, chunk)
        actual = [seconds_for_frames(f) for f in frames]
        tail = (total_frames - frames[-1]) if frames else total_frames
        deviations = (
            [round((a - t) * FPS, 2) for a, t in zip(actual, targets)]
            if len(actual) == len(targets)
            else []
        )
        ladder = chunk_ladder(total_frames, base_chunk, canvas_mp, 4, chunk_step, min_tail)
        alternatives = enumerate_plans(total_frames)

    return {
        "total_frames": total_frames,
        "total_seconds": seconds_for_frames(total_frames),
        "chunk": chunk,
        "chunk_seconds": seconds_for_frames(chunk),
        "base_chunk": base_chunk,
        "chunk_step": chunk_step,
        "cut_mode": cut_mode,
        "cut_slots": CUT_SLOTS,
        "segment_frames": seg_frames,
        "overlap": max(0, int(overlap)),
        "segments": segs,
        "segment_lengths": [e - s for s, e in segs],
        "cut_frames": frames,
        "actual_cuts": actual,
        "requested_cuts": targets,
        "deviations_frames": deviations,
        "tail_frames": tail,
        "tail_seconds": seconds_for_frames(tail),
        "tail_ok": tail >= min_tail,
        "min_tail_frames": min_tail,
        "load": estimate_load(segs, canvas_mp, overlap),
        "max_canvas_mp": max_canvas_mp(segs, overlap),
        "canvas_mp": float(canvas_mp),
        "alternatives": alternatives,
        "ladder": ladder,
        "overlap_used": max(0, int(overlap)),
    }


def shot_prefixes(cuts_seconds: Sequence[float]) -> list[str]:
    """`[Shot N] At MM:SS.mmm, ` prefixes for the R2V detailed_description."""
    out = []
    for n, sec in enumerate(cuts_seconds, start=2):
        out.append(f"[Shot {n}] At {timecode(int(round(sec * FPS)))}, ")
    return out


def uniform_cut_frames(total_frames: int, n: int) -> list[int]:
    """`n` evenly-spaced cut frames, snapped to the 17-frame grid.

    The i-th cut sits as close as possible to `round(i * total / n)`, which is
    what minimises the variance of the window lengths (负载均衡).
    """
    cuts = []
    for i in range(1, n):
        ideal = round(i * total_frames / n)
        frame = round(ideal / FRAME_GRID) * FRAME_GRID
        frame = max(FRAME_GRID, frame)
        if cuts and frame <= cuts[-1]:
            frame = cuts[-1] + FRAME_GRID
        if frame >= total_frames:
            break
        cuts.append(frame)
    return cuts


def auto_plan(
    total_seconds: float,
    chunk_step: int,
    canvas_mp: float = 1.0,
    max_segments: int = 10,
    overlap: int = 0,
) -> dict:
    """Fully automatic cut planning from a chunk STEP (17-frame 档位).

    `chunk_step` is how many 17-frame blocks each window may span, so
    `chunk_frames = chunk_step * 17` and the per-window cap in seconds is
    `chunk_step * 17 / 24 = chunk_step * 0.708s`.

    Goals, in priority order:
      1. fewest segments (段数最少),
      2. smallest variance of window lengths (方差最小; ties are arbitrary),
      3. EVERY window (including the last) <= chunk_frames,
      4. second-pass load below the OOM anchor (能跑).

    Tries n = 1..`max_segments` and returns the first workable split; windows
    are placed as evenly as the 17-frame grid allows.  Raises `ValueError` when
    no split fits (切不出来).
    """
    total_frames = frames_for_seconds(total_seconds)
    chunk_frames = max(FRAME_GRID, int(chunk_step) * FRAME_GRID)
    total_seconds_actual = seconds_for_frames(total_frames)
    max_segment_seconds = seconds_for_frames(chunk_frames)

    for n in range(1, max_segments + 1):
        cuts = uniform_cut_frames(total_frames, n)
        bounds = [0] + cuts + [total_frames]
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        segs = [(s, e) for s, e in segs if e > s]
        if len(segs) != n:
            continue
        lengths = [e - s for s, e in segs]
        longest = max(lengths)  # 每一段（含末段）都要 ≤ chunk
        if longest > chunk_frames:
            continue
        # The executor builds windows out of 17-frame blocks, so a tail shorter
        # than one block cannot be sampled.  Planning it here would only move the
        # failure into the upscale node (and it is what makes a tiny chunk_step
        # like 1 unusable, which is the honest answer).
        if lengths[-1] < FRAME_GRID:
            continue
        load = longest * canvas_mp
        if load >= LOAD_FAIL:
            continue
        mean = sum(lengths) / len(lengths)
        variance = sum((ln - mean) ** 2 for ln in lengths) / len(lengths)
        actual_cuts = [seconds_for_frames(f) for f in cuts]
        return {
            "total_frames": total_frames,
            "total_seconds": total_seconds_actual,
            "segments": n,
            "cut_frames": cuts,
            "segment_frames": cuts,
            "segments_list": segs,
            "lengths": lengths,
            "longest": longest,
            "actual_cuts": actual_cuts,
            "chunk_step": int(chunk_step),
            "chunk_frames": chunk_frames,
            "max_segment_seconds": max_segment_seconds,
            "max_frames": chunk_frames,
            "load": estimate_load(segs, canvas_mp, overlap),
            "max_canvas_mp": max_canvas_mp(segs, overlap),
            "canvas_mp": float(canvas_mp),
            "variance": variance,
            "cut_mode": "frames",
        }

    raise ValueError(
        f"切不出来：{total_seconds_actual:.2f}s（{total_frames} 帧）在「每段 ≤ "
        f"{chunk_frames} 帧 = {max_segment_seconds:.3f}s（chunk_step {int(chunk_step)}）"
        f"、尾段 ≥ {FRAME_GRID} 帧、负载 SAFE」下，{max_segments} 段内无解。"
        "加大 chunk_step，或降画布 MP。"
        + (
            "（chunk_step 太小时，末段总会剩下不足一个 17 帧块，执行器建不出窗口。）"
            if int(chunk_step) * FRAME_GRID < 3 * FRAME_GRID
            else ""
        )
    )


def auto_prompt(plan: dict, style_lead: str = "", action_outline: str = "") -> str:
    """Six-section R2V prompt skeleton with every `[Shot N]` timestamp filled in.

    The caller only writes the plot/character/lighting placeholders — the shot
    count, the timestamps and the retention coverage are already correct.
    """
    cuts = plan["actual_cuts"]
    n_shots = len(cuts) + 1
    stamps = [timecode(int(round(c * FPS))) for c in cuts]
    total = plan["total_seconds"]
    shot_list = ", ".join(f"[Shot {i}]" for i in range(1, n_shots + 1))

    lines = []
    lines.append("subject_definitions:")
    lines.append("<Subject 1> is <角色/物体> in <Picture 1>, <外形特征>.")
    lines.append("<Subject 2> is <场景> in <Picture 2>, <特征>.")
    lines.append("")
    lines.append("summary:")
    cap = plan.get("max_segment_seconds")
    cap_txt = f", with every segment no longer than {cap:.3f} seconds" if cap else ""
    if stamps:
        cut_txt = ", and ".join(f"a hard cut at {s}" for s in stamps)
        shot_txt = f"a {n_shots}-shot sequence with {cut_txt}{cap_txt}"
    else:
        shot_txt = f"a single-shot sequence{cap_txt}"
    lead = (
        f"[reference generation] The target video is a {total:.3f}-second clip with "
        f"native stereo sound, executed as {shot_txt}"
    )
    lines.append(
        (lead + (f", {action_outline}." if action_outline.strip() else "."))
    )
    lines.append("")
    lines.append("retention_analysis:")
    lines.append(
        f"<Subject 1> (appears in {shot_list}): fully_copy - <要保持的具体特征>."
    )
    lines.append(
        f"<Subject 2> (appears in {shot_list}): fully_copy - <环境/光线/构图>."
    )
    lines.append("")
    lines.append("detailed_description:")
    lines.append(style_lead.strip() or "<一两句定调：风格/画质/色调 —— 放在 [Shot 1] 之前>.")
    lines.append("[Shot 1] <构图 + 主体位置与动作 + 环境光线 + 镜头运动 + 声音>.")
    for i, stamp in enumerate(stamps, start=2):
        lines.append(f"[Shot {i}] At {stamp}, <新机位/新景别>, <同上六要素>.")
    lines.append("")
    lines.append("overall_soundscape:")
    lines.append(
        "Throughout the video, <环境音 + 物理音效>. The sound bed carries across the "
        "cut(s) without interruption — the video cuts, the ambience does not."
    )
    lines.append("")
    lines.append("non_diegetic_music:")
    lines.append("<配乐描述，或 N/A>")
    return "\n".join(lines)


def format_auto_report(plan: dict, note: str = "") -> str:
    """Human-readable summary of an automatic split."""
    lines = ["=== H3 HARD-CUT AUTO PLAN ==="]
    lines.append(
        f"duration      : {plan['total_seconds']:.3f}s  ({plan['total_frames']} frames, 17n+5)"
    )
    lines.append(
        f"max segment   : {plan['max_segment_seconds']:.3f}s  ->  {plan['max_frames']} frames "
        f"(17-grid); EVERY window (tail included) stays <= this"
    )
    lines.append(
        f"segments      : {plan['segments']}  (variance {plan['variance']:.1f} — lower = more even)"
    )
    lines.append("windows:")
    for i, (s, e) in enumerate(plan["segments_list"]):
        lines.append(
            f"  #{i:<2} frames [{s:>4} -> {e:>4})  {e - s:>4}f  "
            f"{seconds_for_frames(s):>7.3f}s -> {seconds_for_frames(e):>7.3f}s"
        )
    if plan["cut_frames"]:
        lines.append(
            "cut points    : "
            + ", ".join(
                f"{c:.3f}s (frame {f})"
                for c, f in zip(plan["actual_cuts"], plan["cut_frames"])
            )
        )
    else:
        lines.append("cut points    : (none) — single window")
    load = plan["load"]
    lines.append(
        f"load estimate : {load['longest_window']}f x {plan['canvas_mp']:.3f}MP "
        f"= {load['load']:.1f}  ({load['verdict']}, {load['vs_pass_anchor']:.2f}x pass anchor)"
    )
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def suggest_chunk(total_frames: int, canvas_mp: float, max_windows: int = 6) -> dict | None:
    """Fewest windows that fit the load anchor, longest window inside that budget.

    Fewest windows means fewest cuts to describe in the prompt, and the longest
    window that still fits is the least aggressive answer for that count.
    Returns None when even `max_windows` windows cannot fit.
    """
    min_tail = int(round(MIN_TAIL_SECONDS * FPS))
    for count in range(2, max_windows + 1):
        best = None
        for chunk in range(FRAME_GRID, total_frames, FRAME_GRID):
            segs = segments_for(total_frames, chunk)
            if len(segs) != count:
                continue
            if segs[-1][1] - segs[-1][0] < min_tail:
                continue
            longest = max(e - s for s, e in segs)
            if longest * canvas_mp > LOAD_PASS:
                continue
            if best is None or chunk > best[0]:
                best = (chunk, segs)
        if best:
            chunk, segs = best
            longest = max(e - s for s, e in segs)
            return {
                "windows": count,
                "chunk": chunk,
                "segments": segs,
                "cuts": [seconds_for_frames(s) for s, _ in segs][1:],
                "longest": longest,
                "load": longest * canvas_mp,
            }
    return None


# Durations worth listing as ready-made manual settings.
REFERENCE_SECONDS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _manual_entry_lines(plan: dict) -> list[str]:
    """Every value needed to reproduce this plan by hand in the pack's own node."""
    chunk = plan["chunk"]
    total = plan["total_frames"]
    lines = ["--- MANUAL ENTRY (same values in the upscaler pack's own plan node) ---"]
    _ov_manual = int(plan.get("temporal_overlap_frames") or 0)
    lines.append(
        f"  temporal_chunk_frames    = {chunk}"
        f"        (17 x {chunk // FRAME_GRID};  node range 17..3600, step 17)"
    )
    lines.append(
        f"  temporal_overlap_frames  = {_ov_manual}"
        + ("   (zero = hard cut)"
           if not _ov_manual else
           "   (anchored prefix: each window starts back and pins its first token)")
    )
    lines.append(
        "  temporal_strategy        = guarded_overlap_exp   (this is what enables windowing)"
    )
    lines.append("  spatial_strategy         = full_frame_safe")
    width, height = plan.get("target_width"), plan.get("target_height")
    if width and height:
        lines.append(
            f"  target_width / height    = {width} x {height}"
            "     (must match the HIGH conditioning)"
        )
    else:
        lines.append(
            f"  second-pass canvas       = {plan['canvas_mp']:.3f} MP"
            "   (keep #48.target_width/height wired to #36)"
        )
    lines.append(
        f"  precision / release      = {plan.get('precision', 'bf16')}"
        f" / {plan.get('release_policy', 'clear_after')}"
    )
    lines.append(
        f"  anchor_strength          = {plan.get('anchor_strength', 0.999)}"
        "   (no effect at overlap 0)"
    )
    lines.append(
        f"  one window, no cut       : chunk >= {total}"
        "   (or temporal_strategy = full_clip_safe)"
    )
    lines.append(
        "  note: -1 belongs to THIS node's cut slots only (means: no cut). The"
    )
    lines.append(
        "  pack's own plan node has no -1 convention - its chunk range starts at 17."
    )
    return lines


def _reference_lines(canvas_mp: float, current_seconds: float) -> list[str]:
    """Ready-made chunk/overlap per duration, so other lengths need no guesswork."""
    lines = [
        "--- other durations (same canvas, overlap 0, fewest cuts that fit) ---",
        "  dur    frames  chunk   cut points                          longest  load",
    ]
    current_frames = frames_for_seconds(current_seconds)
    for seconds in REFERENCE_SECONDS:
        frames = frames_for_seconds(seconds)
        picked = suggest_chunk(frames, canvas_mp)
        if not picked:
            lines.append(
                f"  {seconds:>3}s   {frames:>5}   --      no legal split at this canvas"
            )
            continue
        cuts = ", ".join(f"{c:.3f}s" for c in picked["cuts"])
        mark = "  <- this" if frames == current_frames else ""
        lines.append(
            f"  {seconds:>3}s   {frames:>5}   {picked['chunk']:<7} {cuts:<35}"
            f" {picked['longest']:<8} {picked['load']:.1f}{mark}"
        )
    lines.append(
        "  (table = fewest-cuts suggestion only; the plan above is what will run,"
    )
    lines.append(
        "   and any row of the reachable menu below is equally legal)"
    )
    return lines


def _window_block(plan: dict) -> list[str]:
    """One line per window: which frames, how long, and which second it covers."""
    if plan.get("cut_mode") == "frames":
        lines = [
            f"windows       : {len(plan['segments'])} UNEQUAL windows "
            f"(longest {plan['chunk']}f = {plan['chunk_seconds']:.3f}s)"
        ]
    else:
        lines = [
            f"windows       : {len(plan['segments'])} window(s) of {plan['chunk']}f "
            f"({plan['chunk_seconds']:.3f}s) each; the last is the tail"
        ]
    starts = set(plan["cut_frames"])
    for i, (s, e) in enumerate(plan["segments"]):
        mark = ""
        if s in starts:
            mark = f"  <- CUT #{plan['cut_frames'].index(s) + 1}"
        lines.append(
            f"  #{i:<2} frames [{s:>4} -> {e:>4})  {e - s:>4}f  "
            f"{seconds_for_frames(s):>7.3f}s -> {seconds_for_frames(e):>7.3f}s{mark}"
        )
    return lines


def _cut_block(plan: dict) -> list[str]:
    """Where it cuts: seconds, frame number and the R2V timecode."""
    if not plan["actual_cuts"]:
        return ["cut points    : (none) - single window, no hard cut"]
    lines = [f"cut points    : {len(plan['actual_cuts'])} hard cut(s)"]
    for n, (sec, frame) in enumerate(zip(plan["actual_cuts"], plan["cut_frames"]), 1):
        asked = ""
        if n <= len(plan["requested_cuts"]):
            asked = f"   (requested {plan['requested_cuts'][n - 1]:.3f}s)"
        lines.append(
            f"  #{n}  frame {frame:>4}  @ {sec:>7.3f}s  {timecode(frame)}{asked}"
        )
    return lines


def _ladder_lines(plan: dict) -> list[str]:
    base = plan["base_chunk"]
    step = plan["chunk_step"]
    lines = [
        f"--- chunk ladder (base {base}f = {base // FRAME_GRID} x 17; "
        f"chunk_step {step:+d} -> {plan['chunk']}f; one step = 17 frames) ---",
        "  step  chunk  wins  cut times (sec @ frame)                     "
        "longest   load  note",
    ]
    for row in plan["ladder"]:
        pairs = list(zip(row["cuts"], row["cut_frames"]))
        cuts = ", ".join(f"{c:.3f}@{f}" for c, f in pairs[:4])
        if len(pairs) > 4:
            cuts += ", ..."
        if not cuts:
            cuts = "(none)"
        if not row["tail_ok"]:
            note = f"tail {row['tail']}f < {plan['min_tail_frames']}f"
        elif row["load"] <= LOAD_PASS:
            note = "SAFE"
        elif row["load"] < LOAD_FAIL:
            note = "borderline"
        else:
            note = "OOM risk"
        mark = "  <= selected" if row["selected"] else ""
        lines.append(
            f"  {row['step']:>+3}  {row['chunk']:>5}  {row['windows']:>4}  "
            f"{cuts:<46} {row['longest']:>7} {row['load']:>6.1f}  {note}{mark}"
        )
    return lines


def format_report(plan: dict, note: str = "") -> str:
    lines = []
    lines.append("=== H3 HARD-CUT PLAN ===")
    lines.append(
        f"duration      : {plan['total_seconds']:.3f}s  ({plan['total_frames']} frames, 17n+5)"
    )
    is_frames = plan.get("cut_mode") == "frames"
    if is_frames:
        frames = ", ".join(f"{f}f" for f in plan["segment_frames"])
        lines.append(
            f"cut frames    : {frames}   (UNEQUAL windows, snapped to 17-frame grid)"
        )
    elif plan["requested_cuts"]:
        asked = ", ".join(f"{c:.3f}s" for c in plan["requested_cuts"])
        lines.append(
            f"cut inputs    : {len(plan['requested_cuts'])} of {plan['cut_slots']} slots "
            f"used -> {asked}   (-1 = no cut)"
        )
    else:
        lines.append(
            f"cut inputs    : all {plan['cut_slots']} slots = -1  ->  no cut, one window"
        )
    if is_frames:
        lines.append(
            f"longest win   : {plan['chunk']}f = {plan['chunk_seconds']:.3f}s  /  0"
            f"   <- UNEQUAL windows; chunk above = the longest one"
        )
    else:
        lines.append(
            f"chunk/overlap : {plan['chunk']}f = {plan['chunk_seconds']:.3f}s  /  0"
            f"   <- window length ({plan['chunk'] // FRAME_GRID} x 17 frames) / HARD CUT"
        )
    lines += _window_block(plan)
    lines += _cut_block(plan)
    if not is_frames and plan["requested_cuts"]:
        if plan["deviations_frames"]:
            dev = ", ".join(f"{d:+.2f}f" for d in plan["deviations_frames"])
            lines.append(f"vs requested  : {dev}  (frame deviation per cut)")
        else:
            lines.append(
                f"vs requested  : cut count changed — {len(plan['requested_cuts'])} "
                f"requested, {len(plan['actual_cuts'])} produced"
            )
    lines.append(
        f"tail window   : {plan['tail_frames']}f = {plan['tail_seconds']:.3f}s  "
        f"({'OK' if plan['tail_ok'] else 'TOO SHORT'}, needs >= "
        f"{plan['min_tail_frames']}f)"
    )
    load = plan["load"]
    lines.append(
        f"load estimate : {load['longest_window']}f x {plan['canvas_mp']:.3f}MP "
        f"= {load['load']:.1f}  ({load['verdict']}, {load['vs_pass_anchor']:.2f}x pass anchor)"
    )
    lines.append(
        f"canvas ceiling: <= {plan['max_canvas_mp']:.3f} MP keeps this plan in the safe band"
    )
    if not is_frames:
        lines.append("")
        lines += _ladder_lines(plan)
    lines.append("")
    lines += _manual_entry_lines(plan)
    lines.append("")
    lines.append("--- prompt timestamps (paste into detailed_description) ---")
    lines.append("style lead sentence goes here, before [Shot 1].")
    lines.append("[Shot 1] <opening shot description>")
    for prefix, (s, e) in zip(
        shot_prefixes(plan["actual_cuts"]), plan["segments"][1:]
    ):
        lines.append(f"{prefix}<new shot from {seconds_for_frames(s):.3f}s>")
    lines.append("")
    lines += _reference_lines(plan["canvas_mp"], plan["total_seconds"])
    alts = plan.get("alternatives") or []
    if alts:
        lines.append("")
        lines.append(
            f"--- full reachable menu at {plan['total_seconds']:.3f}s "
            "(cuts are multiples of 17 frames) ---"
        )
        current = None
        for row in alts:
            if row["segments"] != current:
                current = row["segments"]
                lines.append(f"  {current} windows / {current - 1} cut(s):")
            cuts = ", ".join(f"{c:.3f}s" for c in row["cuts"])
            lines.append(
                f"     chunk {row['chunk']:<4} cuts @ {cuts:<30} "
                f"frames {row['lengths']}"
            )
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# prompt validation
# --------------------------------------------------------------------------
#
# The hard cut only reads as an intentional edit when the *prompt* asks the
# model for a shot change on the very frame the executor splits on.  Two things
# therefore have to agree: the wording (no "single continuous take") and the
# timestamps (every cut reachable as n x chunk frames, chunk a multiple of 17).
# Catching that here costs a second; catching it after a 30 minute render does
# not.

PROMPT_SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# Wording that tells the model NOT to cut — directly opposed to overlap = 0.
NO_CUT_PATTERNS = (
    (r"\bno\s+cuts?\b", "says there is no cut"),
    (r"\bwithout\s+(?:any\s+)?cuts?\b", "says without a cut"),
    (r"\bnever\s+cuts?\b", "says the shot never cuts"),
    (r"\bno\s+edits?\b", "says there is no edit"),
    (r"(?:one|single|a)\s+continuous\s+take", "'continuous take' wording"),
    (r"\bunbroken\s+(?:take|move|shot|sequence)\b", "'unbroken' wording"),
    (r"\bin\s+one\s+take\b", "'in one take' wording"),
    (r"\bno\s+shot\s+changes?\b", "says there is no shot change"),
)

_SECTION_RE = re.compile(r"^([a-z_]+)\s*:", re.MULTILINE)
_SHOT_RE = re.compile(
    r"\[\s*Shot\s+(\d+)\s*\]\s*(.*?)(?=\[\s*Shot\s+\d+\s*\]|\Z)", re.DOTALL
)
_LEAD_STAMP_RE = re.compile(r"^At\s+(\d{1,3}):(\d{2})\.(\d{3})\s*,", re.IGNORECASE)
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*second", re.IGNORECASE)

MAX_LISTED = 8          # never dump an unbounded error list into the node UI


def _tail_after(prompt: str, header: str, stop: str | None = None) -> str:
    """Text of one `header:` section, optionally cut off before the next one."""
    if header not in prompt:
        return ""
    chunk = prompt.split(header, 1)[1]
    if stop and stop in chunk:
        chunk = chunk.split(stop, 1)[0]
    return chunk


def parse_shots(prompt: str) -> list[dict]:
    """Every `[Shot N]` block inside `detailed_description`, with its cut stamp.

    Deliberately scoped to that one section: `retention_analysis` also writes
    `[Shot 1]` when it lists the shots a subject survives, and counting those as
    shots would invent shots that do not exist.
    """
    section = _tail_after(prompt or "", "detailed_description:", "overall_soundscape:")
    rows = []
    for match in _SHOT_RE.finditer(section):
        body = match.group(2).strip()
        lead = _LEAD_STAMP_RE.match(body)
        rows.append(
            {
                "index": int(match.group(1)),
                "body": body,
                "cut_seconds": (
                    int(lead.group(1)) * 60 + int(lead.group(2)) + int(lead.group(3)) / 1000.0
                )
                if lead
                else None,
            }
        )
    return rows


def geometry_from_plan(plan: dict | None) -> dict:
    """Pull the window geometry out of a T8 plan (or our own hard-cut plan).

    The executor reads the plan field by field, so the plan is the only honest
    source for *where* it will cut: with `hop = chunk - overlap` the cuts sit at
    `n * hop`.  Our own plan node additionally stashes the clip length under
    `hardcut.geometry`, which is what lets the shot-count check run at all.
    """
    if not isinstance(plan, dict):
        return {}
    chunk = int(plan.get("temporal_chunk_frames") or 0)
    overlap = int(plan.get("temporal_overlap_frames") or 0)
    width = int(plan.get("target_width") or 0)
    height = int(plan.get("target_height") or 0)
    extra = plan.get("hardcut") or {}
    geo = extra.get("geometry") or {}
    total_frames = geo.get("total_frames")
    return {
        "chunk": chunk or None,
        "overlap": overlap,
        "hop": (chunk - overlap) if chunk else None,
        "canvas_mp": (width * height / 1e6) if (width and height) else None,
        "total_frames": int(total_frames) if total_frames else None,
        "total_seconds": (int(total_frames) / FPS) if total_frames else None,
        "cut_seconds": list(geo.get("cut_seconds") or []),
        "segment_frames": list(geo.get("segment_frames") or []),
    }


def validate_prompt(
    prompt: str,
    total_seconds: float | None = None,
    requested_cuts: str | Iterable[float] = "",
    canvas_mp: float | None = None,
    check_load: bool = True,
    chunk_frames: int | None = None,
    overlap_frames: int = 0,
    overlap_anchored: bool = False,
    segment_frames: Iterable[float] | None = None,
    loose: bool = False,
) -> dict:
    """Check that a hand-written R2V prompt really describes the plan's hard cut.

    `total_seconds`, `canvas_mp` and `chunk_frames` can all be left out: without
    them the checks that need geometry are skipped or fall back to what the
    prompt itself claims.  Feed them from the plan (see `geometry_from_plan`)
    and the check gets exact.

    Returns a dict with `ok`, `errors`, `warnings`, the parsed geometry and a
    `report` string.  `ok` is False as soon as a real contradiction is found.
    """
    prompt = prompt or ""
    errors: list[str] = []
    warnings: list[str] = []

    # 官方 ref guide：retention 标签分两族，两族都合法（2026-09-26 订正：旧版只认
    # 音频族，把官方画面族的 fully_preserved 误报成"非官方标签"）。
    _OFFICIAL_RETENTION = (
        # 画面类（<Subject> / <Picture> / <Video>）
        "fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference",
        # 音频类（<Audio>）
        "fully_copy", "partially_copy", "reference",
    )
    _TAG_SHAPE = re.compile(r"^(?:fully|partially|attribute|weak)_[a-z]{2,}$")
    _ret = re.search(r"retention_analysis\s*:(.*?)(?=\n[a-z_]+\s*:|\Z)", prompt, re.S | re.I)
    if _ret:
        _tags = set(re.findall(r"\b([a-z_]{3,})\b", _ret.group(1)))
        _bad = sorted(
            t for t in _tags
            if _TAG_SHAPE.match(t) and t not in _OFFICIAL_RETENTION
        )
        if _bad:
            warnings.append(
                "retention_analysis uses unofficial tag(s) "
                + ", ".join(_bad)
                + " - 画面类: fully_preserved / partially_preserved / attribute_transfer / "
                "weak_reference；音频类: fully_copy / partially_copy / reference / weak_reference"
            )
    info: dict = {"canvas_mp": float(canvas_mp) if canvas_mp else None}

    # ---- 1. section skeleton -------------------------------------------
    headers = [m.group(1) for m in _SECTION_RE.finditer(prompt)]
    known = [h for h in headers if h in PROMPT_SECTIONS]
    missing = [s for s in PROMPT_SECTIONS if s not in set(known)]
    if missing:
        # `loose` = the caller has stopped relying on the prompt for boundary
        # placement (auto_calm_search / overlap), so a free-form prompt is
        # legitimate and this is advice, not a contradiction.
        _m = ("prompt is missing required section header(s): "
              + ", ".join(f"{s}:" for s in missing))
        (warnings if loose else errors).append(_m)
    duplicated = sorted({h for h in known if known.count(h) > 1})
    if duplicated:
        warnings.append("section header(s) appear more than once: " + ", ".join(duplicated))

    # ---- 2. shots -------------------------------------------------------
    shots = parse_shots(prompt)
    indices = [s["index"] for s in shots]
    info["shots"] = indices
    if not shots:
        _m = ("no '[Shot N]' marker found — the executor will cut, but nothing in the "
              "prompt asks the model for a shot change")
        (warnings if loose else errors).append(_m)
    elif indices != list(range(1, len(indices) + 1)):
        errors.append(
            f"[Shot N] numbering must run consecutively from 1; found {indices}"
        )

    # ---- 3. wording that forbids cutting --------------------------------
    hits: list[str] = []
    for pattern, why in NO_CUT_PATTERNS:
        for match in re.finditer(pattern, prompt, re.IGNORECASE):
            line = prompt[: match.start()].count("\n") + 1
            hits.append(f"line {line}: {why} — '{match.group(0)}'")
    if hits:
        extra = f"  (+{len(hits) - MAX_LISTED} more)" if len(hits) > MAX_LISTED else ""
        _m = (
            "wording contradicts a hard cut (the model is told not to cut while the "
            "executor splits) -> " + "; ".join(hits[:MAX_LISTED]) + extra
        )
        if overlap_anchored:
            # 与上面时间戳那段同一个道理：开着锚定 overlap / calm 搜索时，窗口边界是
            # 「从上一窗继续」的缝合处，不是硬断点 —— 所以「一镜到底 / 不换镜」的措辞
            # 是**对的**（模型本来就不该换镜，是采样器跨过去），不该判错。
            # 修之前这条不看 overlap_anchored，于是永远报错，和同一份报告里的 W3
            # （「plan overlap 是 17 帧，是混合不是硬切」）自相矛盾。
            warnings.append(
                _m + "  — acceptable: overlap/calm-search is on, so the window boundary is "
                "stitched with an anchored overlap rather than a hard break; a prompt that "
                "describes one continuous take is correct in that case."
            )
        else:
            errors.append(_m)

    # ---- 4. timestamps --------------------------------------------------
    cut_secs: list[float] = []
    missing_stamps = 0
    for shot in shots[1:]:
        if shot["cut_seconds"] is None:
            missing_stamps += 1
            errors.append(
                f"[Shot {shot['index']}] does not start with 'At MM:SS.mmm,' — the "
                "official template requires a timestamp on every shot after the first"
            )
        else:
            cut_secs.append(shot["cut_seconds"])
    if shots and shots[0]["cut_seconds"] not in (None, 0.0):
        warnings.append(
            f"[Shot 1] carries a timestamp ({shots[0]['cut_seconds']:.3f}s); the "
            "official template omits it on the opening shot"
        )
    if any(b <= a for a, b in zip(cut_secs, cut_secs[1:])):
        errors.append(
            "shot timestamps must strictly increase; found "
            + ", ".join(f"{c:.3f}s" for c in cut_secs)
        )
    info["cut_seconds"] = cut_secs

    # ---- 5. clip length: argument, else what the prompt claims ----------
    declared = [float(x) for x in _DURATION_RE.findall(prompt)]
    declared_seconds = declared[0] if declared else None
    info["declared_seconds"] = declared_seconds
    if total_seconds is None:
        total_seconds = declared_seconds
        if total_seconds is None:
            warnings.append(
                "clip length unknown — the prompt never says 'N-second' and no plan "
                "was supplied, so the shot-count and last-window checks are skipped"
            )
    elif declared_seconds is not None and abs(declared_seconds - float(total_seconds)) > 1e-6:
        # Info, not a warning: a round number in the prompt vs the 17n+5 grid is
        # expected and harmless, and a warning here made callers rewrite prompts
        # that were already fine.
        info["declared_seconds_note"] = (
            f"prompt says '{declared_seconds:g}-second', plan is "
            f"{float(total_seconds):g}s (snapped to the 17n+5 frame grid)"
        )
    info["total_seconds"] = float(total_seconds) if total_seconds else None
    total_frames = frames_for_seconds(total_seconds) if total_seconds else None
    info["total_frames"] = total_frames

    # ---- 6. geometry: window length from the plan, else from the prompt --
    requested = parse_cut_points(requested_cuts)
    info["requested"] = requested
    seed = requested or cut_secs
    seg_frames = snap_token_edge(segment_frames) if segment_frames else []
    if seg_frames and total_frames:
        seg_frames = [f for f in seg_frames if 0 < f < total_frames]
    info["segment_frames"] = seg_frames
    chunk = int(chunk_frames) if chunk_frames else None
    if not chunk and seed:
        seed_frames = [int(round(c * FPS)) for c in seed]
        estimate = sum(f / n for n, f in enumerate(seed_frames, start=1)) / len(seed_frames)
        chunk = max(FRAME_GRID, int(round(estimate / FRAME_GRID)) * FRAME_GRID)
    overlap = max(0, int(overlap_frames or 0))
    hop = (chunk - overlap) if chunk else None
    info.update(chunk=chunk, overlap=overlap, hop=hop)
    if overlap:
        warnings.append(
            f"plan overlap is {overlap} frames, not 0 — this plan blends at the join "
            "instead of hard cutting; the timing checks below use hop = chunk - overlap"
        )

    truth: list[int] = []
    segs: list[tuple[int, int]] = []
    if seg_frames:
        # UNEQUAL windows: the plan carries explicit cut frames, so the executor
        # splits exactly there and no constant stride applies.
        truth = seg_frames
        if total_frames:
            bounds = [0] + seg_frames + [total_frames]
            segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
            segs = [(s, e) for s, e in segs if e > s]
    elif hop is not None and hop <= 0:
        errors.append(
            f"window stride is {hop} frames — the plan has nothing to cut with"
        )
    elif hop is not None:
        # what the executor will really do
        if total_frames:
            segs = segments_for(total_frames, chunk, overlap)
            truth = cut_frames(total_frames, chunk, overlap)
        else:
            truth = [n * hop for n in range(1, len(cut_secs) + 1)]
    info["segments"] = segs
    info["actual_cuts"] = [seconds_for_frames(f) for f in truth]

    # The three timing checks below must hold in BOTH windowing modes.
    if seg_frames or (hop is not None and hop > 0):
        origin = (
            "hardcut.segment_frames"
            if seg_frames
            else ("the plan" if requested else f"hop {hop}")
        )
        # Window boundaries are HARD breaks: overlap is 0, so no information
        # crosses them — each one must therefore be a shot change the prompt
        # declares, or the model keeps shooting the old scene where the executor
        # has already started a new window.  The reverse is perfectly legal: a
        # prompt may declare MORE timestamps than the executor cuts, and those
        # extra ones are shot changes the model performs *inside* one window
        # (exactly how the first pass does multi-shot clips in a single run).
        if cut_secs:
            prompt_frames = [int(round(sec * FPS)) for sec in cut_secs]
            missing = []
            early = []
            for f in truth:
                if any(abs(f - p) <= 1 for p in prompt_frames):
                    continue
                later = [p for p in prompt_frames if p > f]
                if later and min(later) - f <= FRAME_GRID:
                    # "Early" boundary: the executor retreats BEFORE the next
                    # declared shot, within one 17-frame block.  The seam then
                    # falls inside the OLD shot and the model performs the shot
                    # change inside window 2 — the same contract as an
                    # equal-window split.  This is the only way to keep the new
                    # shot whole when the measured turn lands on a shared-token
                    # frame: a window boundary can never sit there, so
                    # retreating to the previous token edge is the fix.
                    early.append((f, min(later)))
                else:
                    missing.append(f)
            if early:
                info["early_cuts"] = [
                    {"frame": f, "next_declared": p} for f, p in early
                ]
            for frame in missing[:MAX_LISTED]:
                _msg = (
                    f"executor boundary frame {frame} ({timecode(frame)}) has no matching "
                    "'At MM:SS.mmm' in the prompt. Declared: "
                    + (", ".join(timecode(p) for p in prompt_frames) or "(none)")
                )
                if overlap_anchored:
                    # The seam is stitched with an anchored overlap, so the window
                    # boundary no longer has to coincide with a model shot change:
                    # the sampler continues from the previous window and the
                    # boundary may sit in continuous content on purpose.
                    warnings.append(
                        _msg + "  — acceptable: overlap/calm-search is on, so the seam is "
                        "anchored instead of relying on a model cut here."
                    )
                else:
                    errors.append(
                        _msg + "  — a window boundary is a hard break (no context crosses "
                        "it), so the model has to be told to change shot at that exact frame."
                    )
            extra = [
                p for p in prompt_frames if not any(abs(p - t) <= 1 for t in truth)
            ]
            if extra and not missing:
                # Legal by design: the executor only hard-splits on its own
                # boundaries, and these extra timestamps are shot changes the
                # model performs inside one window.  Recorded as info (shown as a
                # plain line in the report), never as a warning — a warning here
                # made callers edit perfectly valid prompts.
                info["in_window_cuts"] = [seconds_for_frames(p) for p in extra]
            elif not missing and not extra and any(
                int(round(sec * FPS)) != t for sec, t in zip(cut_secs, truth)
            ):
                warnings.append(
                    "shot timestamps sit within one frame of the real cut frames "
                    "(rounding only, harmless)"
                )

        if total_frames:
            tail = total_frames - truth[-1] if truth else total_frames
            if tail < int(round(MIN_TAIL_SECONDS * FPS)):
                warnings.append(
                    f"last window is only {tail} frames "
                    f"({seconds_for_frames(tail):.2f}s) — a cut that late is barely "
                    "worth it"
                )

    # ---- 7. plan wired but the prompt declares nothing ------------------
    if requested and not cut_secs and not missing_stamps:
        _m = (f"the plan cuts {len(requested)} time(s) "
              f"({', '.join(f'{c:.3f}s' for c in requested)}) but the prompt declares no "
              "'At MM:SS.mmm' timestamp at all")
        (warnings if loose else errors).append(_m)
    elif cut_secs and not requested:
        warnings.append(
            "no plan was supplied, so the prompt was judged on its own — wire "
            "MiniMaxH3HardCutPlan.plan in to catch prompt/plan drift"
        )

    # ---- 8. load --------------------------------------------------------
    if check_load and canvas_mp and info.get("segments"):
        load = estimate_load(info["segments"], canvas_mp)
        info["load"] = load
        if load["verdict"] == "LIKELY-OOM":
            errors.append(
                f"load {load['load']:.1f} ({load['longest_window']}f x "
                f"{float(canvas_mp):.3f}MP) is past the measured OOM anchor "
                f"{LOAD_FAIL:.1f}; canvas <= {max_canvas_mp(info['segments']):.3f} MP or "
                "one more cut would fit"
            )
        elif load["verdict"] == "BORDERLINE":
            warnings.append(
                f"load {load['load']:.1f} is above the known-good anchor "
                f"{LOAD_PASS:.1f}; canvas <= {max_canvas_mp(info['segments']):.3f} MP is "
                "safer"
            )

    # ---- 9. retention_analysis coverage ---------------------------------
    retention = _tail_after(prompt, "retention_analysis:", "detailed_description:")
    if retention.strip() and shots:
        mentioned = {int(x) for x in re.findall(r"\[Shot\s+(\d+)\]", retention)}
        expected = set(range(1, len(shots) + 1))
        if mentioned and mentioned != expected:
            warnings.append(
                f"retention_analysis mentions shots {sorted(mentioned)} but the video "
                f"has {sorted(expected)} — list every shot a subject must survive"
            )
    elif not retention.strip():
        warnings.append("retention_analysis section looks empty")

    # ---- 10. nearest legal windings -------------------------------------
    if total_frames and shots:
        want = max(1, len(shots) - 1)
        info["menu"] = [
            row for row in enumerate_plans(total_frames) if row["segments"] == want + 1
        ][:3]

    result = {
        "ok": not errors,
        "status": "OK" if not errors else "ERROR",
        "errors": errors,
        "warnings": warnings,
        **info,
    }
    result["report"] = format_validation(result)
    return result


def format_validation(result: dict) -> str:
    lines = ["=== H3 HARD-CUT PROMPT CHECK ==="]
    lines.append(
        f"status        : {'OK' if result.get('ok') else 'ERROR'}   "
        f"({len(result.get('errors') or [])} error(s), "
        f"{len(result.get('warnings') or [])} warning(s))"
    )
    total_frames = result.get("total_frames")
    if total_frames:
        lines.append(
            f"clip          : {result['total_seconds']:.3f}s -> {total_frames} frames (17n+5)"
        )
    elif result.get("declared_seconds"):
        lines.append(
            f"clip          : prompt says {result['declared_seconds']:g}s "
            "(no plan length available)"
        )
    if result.get("declared_seconds_note"):
        lines.append("clip note     : " + result["declared_seconds_note"])
    if result.get("shots"):
        lines.append(
            "shots found   : " + ", ".join(f"[Shot {i}]" for i in result["shots"])
        )
    cuts = result.get("cut_seconds") or []
    if cuts:
        lines.append(
            "prompt cuts   : "
            + ", ".join(
                f"{c:.3f}s (frame {int(round(c * FPS))}, {timecode(int(round(c * FPS)))})"
                for c in cuts
            )
        )
    in_window = result.get("in_window_cuts") or []
    if in_window:
        lines.append(
            "in-window cuts: "
            + ", ".join(
                f"{c:.3f}s ({timecode(int(round(c * FPS)))})" for c in in_window
            )
            + "  -> shot changes the MODEL performs inside one window (legal: the "
            "executor only hard-splits on its own boundaries)"
        )
    early = result.get("early_cuts") or []
    if early:
        parts = [
            f"frame {e['frame']} ({timecode(e['frame'])}), "
            f"{e['next_declared'] - e['frame']}f before the declared "
            f"{timecode(e['next_declared'])}"
            for e in early
        ]
        lines.append(
            "early cuts    : "
            + "; ".join(parts)
            + "  -> legal: boundary sits inside the OLD shot, so the turn is "
            "sampled whole inside window 2 and the seam hides in a slow-moving "
            "region (amplitude scales with sigma0)"
        )
    if result.get("chunk"):
        chunk = result["chunk"]
        segs = result.get("segments") or []
        if result.get("segment_frames"):
            head = (
                f"executor grid : UNEQUAL windows, lengths {[e - s for s, e in segs]}"
                f", overlap {result.get('overlap', 0)}"
            )
        else:
            head = (
                f"executor grid : window {chunk}f = {chunk / FPS:.3f}s "
                f"({chunk // FRAME_GRID} x {FRAME_GRID}), overlap {result.get('overlap', 0)}"
            )
            if segs:
                head += f" -> {len(segs)} window(s), lengths {[e - s for s, e in segs]}"
        lines.append(head)
        if segs:
            starts = {s for s, _ in segs[1:]}
            for i, (s, e) in enumerate(segs):
                mark = "  <- CUT" if s in starts else ""
                lines.append(
                    f"  #{i:<2} frames [{s:>4} -> {e:>4})  {e - s:>4}f  "
                    f"{s / FPS:>7.3f}s -> {e / FPS:>7.3f}s{mark}"
                )
        if result.get("actual_cuts"):
            lines.append(
                "executor cuts : "
                + ", ".join(
                    f"{c:.3f}s (frame {int(round(c * FPS))})"
                    for c in result["actual_cuts"]
                )
            )
    if result.get("load"):
        load = result["load"]
        lines.append(
            f"load estimate : {load['longest_window']}f x "
            f"{float(result.get('canvas_mp') or 0.0):.3f}MP = {load['load']:.1f}  "
            f"({load['verdict']}, {load['vs_pass_anchor']:.2f}x pass anchor)"
        )

    errors = result.get("errors") or []
    warnings = result.get("warnings") or []
    if errors:
        lines.append("")
        lines.append("--- ERRORS (fix these before running) ---")
        for i, item in enumerate(errors, start=1):
            lines.append(f"  [E{i}] {item}")
    if warnings:
        lines.append("")
        lines.append("--- WARNINGS ---")
        for i, item in enumerate(warnings, start=1):
            lines.append(f"  [W{i}] {item}")

    menu = result.get("menu") or []
    if menu:
        lines.append("")
        lines.append(
            "--- reachable windings for this clip (cuts are multiples of 17 frames) ---"
        )
        for row in menu:
            lines.append(
                f"  chunk {row['chunk']:<4} {row['segments']} windows, cuts @ "
                + ", ".join(f"{c:.3f}s" for c in row["cuts"])
            )

    if errors:
        lines.append("")
        lines.append(
            "hint: use the reachable windings above — a cut time that is not listed "
            "there cannot be produced, because windows share one constant stride."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _int_list(text: str) -> list[int] | None:
    """`"68,136"` / `"68;136"` -> [68, 136]; empty -> None."""
    if not text or not text.strip():
        return None
    out: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("usage: python hardcut_math.py <total_seconds> [cut_points] [--mp M] [--step N]")
        print("       python hardcut_math.py --check <prompt.txt> --total 8 [--cuts 4.25] [--mp 1.5]")
        print("                                       [--frames 68,136] [--chunk 136] [--step 0]")
        return 2

    if argv[1] == "--check":
        if len(argv) < 3:
            print("ERROR: --check needs a prompt file path")
            return 2
        text = open(argv[2], encoding="utf-8").read()
        total = 8.0
        cuts = ""
        mp = 1.5
        step = 0
        frames = ""
        if "--total" in argv:
            total = float(argv[argv.index("--total") + 1])
        if "--cuts" in argv:
            cuts = argv[argv.index("--cuts") + 1]
        if "--mp" in argv:
            mp = float(argv[argv.index("--mp") + 1])
        if "--step" in argv:
            step = int(argv[argv.index("--step") + 1])
        if "--frames" in argv:
            frames = argv[argv.index("--frames") + 1]

        seg_frames = _int_list(frames)
        chunk = None
        if seg_frames:
            # Same derivation the plan node uses, so the geometry line, the window
            # list and the load check match a real run instead of being guessed.
            try:
                chunk = plan_hard_cut(total, "", mp, 0, seg_frames)["chunk"]
            except ValueError:
                chunk = None
        if "--chunk" in argv:
            chunk = int(argv[argv.index("--chunk") + 1])

        result = validate_prompt(
            text,
            total,
            cuts,
            mp,
            chunk_frames=chunk,
            overlap_frames=step,
            segment_frames=seg_frames,
        )
        print(result["report"])
        return 0 if result["ok"] else 1

    total = float(argv[1])
    cuts = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else ""
    mp = 1.0
    if "--mp" in argv:
        mp = float(argv[argv.index("--mp") + 1])
    step = 0
    if "--step" in argv:
        step = int(argv[argv.index("--step") + 1])
    try:
        plan = plan_hard_cut(total, cuts, mp, step)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(format_report(plan))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
