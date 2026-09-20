#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-hardcut contributors
# -*- coding: utf-8 -*-
"""V2 pixel-domain seam repair for hard-cut clips.

Pipeline position: AFTER decode (VAE domain).  A window seam reads as a
one-frame pop: fine detail / lighting jumps between the last frame of a
window and the first frame of the next.  We blend the frames on both
sides toward each other so the pop spreads into the local motion --
frame count, duration and audio are preserved exactly.

Usage:
  python repair_seam.py clip.mp4 --locate                 # blind-detect candidates
  python repair_seam.py clip.mp4 --seams 67,101           # seam sits between S and S+1
  python repair_seam.py clip.mp4 --seams 67 --strength 0.3 --mode both
  python repair_seam.py clip.mp4 --seams 67 --audio clip-audio.mp4

--seams S : the pop sits BETWEEN frame S and frame S+1 (S = last frame of
            the earlier window).  With --mode both (default) both frames
            are pulled toward each other by `strength`.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np


def find_ffmpeg():
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe and os.path.isfile(exe):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


def probe_audio(path, ffmpeg):
    """True if `path` carries an audio stream (parses `ffmpeg -i` stderr)."""
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path],
            capture_output=True, text=True, errors="replace",
        )
        return "Audio:" in (proc.stderr or "")
    except Exception:
        return False


def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    return frames


def gray_series(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]


def frame_diffs(gray):
    return np.array(
        [float(np.abs(gray[i + 1] - gray[i]).mean()) for i in range(len(gray) - 1)],
        dtype=np.float64,
    )


def locate(diffs, k):
    """Local maxima that stand k x above the clip-wide median."""
    med = float(np.median(diffs))
    if med <= 0:
        return []
    out = []
    for i, d in enumerate(diffs):
        if d < k * med:
            continue
        lo, hi = max(0, i - 2), min(len(diffs), i + 3)
        if d >= max(diffs[lo:hi]):
            out.append((i, round(d / med, 2)))
    return out


def blend_pair(a, b, strength, mode):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if mode == "prev":  # only the earlier frame leans forward
        return [(1.0 - strength) * a + strength * b, b]
    if mode == "next":  # only the later frame leans back
        return [a, strength * a + (1.0 - strength) * b]
    # both: pull the pair together -> peak jump drops by (1 - 2*strength)
    return [
        (1.0 - strength) * a + strength * b,
        strength * a + (1.0 - strength) * b,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("video")
    ap.add_argument("--seams", default="", help="comma list; pop sits between S and S+1")
    ap.add_argument("--replace", default="",
                    help="S:patch.mp4 - overwrite frames S..S+n-1 with the "
                         "frames of patch.mp4 (e.g. an H3 ref2v bridge clip); "
                         "repeatable, comma separated")
    ap.add_argument("--dissolve", default="",
                    help="S:E,... - replace corrupted frames S..E with a "
                         "linear dissolve from frame S-1 to frame E+1.  "
                         "Zero-generation fix: the span reads as a dissolve "
                         "cut, which is legitimate film grammar.")
    ap.add_argument("--fuse", default="",
                    help="S:E:patch.mp4 - blend the patch frames into frames "
                         "S..E with a strength ramp instead of a hard replace. "
                         "Use it to feather a generated bridge into the main "
                         "clip when both versions show the same content.")
    ap.add_argument("--fuse-side", choices=["after", "before"], default="after",
                    help="after (default): patch weight is STRONGEST at frame "
                         "S (right after the hard cut) and fades toward E - "
                         "'five frames after the cut, strength high to low'. "
                         "before: mirrored - weight rises toward E (strongest "
                         "on the side facing the shot change).")
    ap.add_argument("--fuse-min", type=float, default=0.0,
                    help="patch weight at the weak end of the ramp")
    ap.add_argument("--fuse-max", type=float, default=1.0,
                    help="patch weight at the strong end of the ramp")
    ap.add_argument("--locate", action="store_true", help="detect candidates only")
    ap.add_argument("--k", type=float, default=4.0, help="locate threshold (x median)")
    ap.add_argument("--strength", type=float, default=0.3,
                    help="blend factor per frame (0..0.5)")
    ap.add_argument("--mode", choices=["both", "prev", "next"], default="both")
    ap.add_argument("--audio", default="", help="external audio source when the "
                    "clip itself has no track")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    frames = read_frames(args.video)
    gray = gray_series(frames)
    diffs = frame_diffs(gray)
    print(f"frames={len(frames)}  diff median={np.median(diffs):.2f}  "
          f"diff max={diffs.max():.2f}")

    if args.locate or not (args.seams or args.dissolve or args.replace
                           or args.fuse):
        cands = locate(diffs, args.k)
        print("candidates (pop between S and S+1, x median):")
        for i, ratio in cands:
            print(f"  S={i:<5} diff={diffs[i]:7.2f}  ({ratio}x)")
        if not (args.seams or args.dissolve or args.replace or args.fuse):
            print("give --seams / --dissolve / --replace / --fuse to repair; "
                  "nothing written")
            return

    seams = [int(s) for s in args.seams.replace(";", ",").split(",") if s.strip()]
    strength = min(0.5, max(0.0, args.strength))
    fixed = 0
    dissolve_fixed = 0

    if args.dissolve:
        for spec in args.dissolve.replace(";", ",").split(","):
            if ":" not in spec:
                print(f"skip {spec!r}: expected S:E")
                continue
            s_str, e_str = spec.split(":", 1)
            s, e = int(s_str), int(e_str)
            if not (0 < s <= e < len(frames) - 1):
                print(f"skip dissolve {s}:{e}: anchors would fall outside the clip")
                continue
            left = frames[s - 1].astype(np.float32)
            right = frames[e + 1].astype(np.float32)
            steps = e - s + 2
            for j, fi in enumerate(range(s, e + 1)):
                alpha = (j + 1) / steps
                frames[fi] = ((1.0 - alpha) * left + alpha * right).astype(np.uint8)
            print(f"dissolved frames {s}..{e}: {e - s + 1} frames replaced by a "
                  f"linear blend of f{s - 1} -> f{e + 1} "
                  f"(alpha {1 / steps:.2f} .. {(steps - 1) / steps:.2f})")
            dissolve_fixed += 1

    if args.fuse:
        fmin = min(1.0, max(0.0, args.fuse_min))
        fmax = min(1.0, max(fmin, args.fuse_max))
        for spec in args.fuse.replace(";", ",").split(","):
            parts = spec.split(":", 2)  # maxsplit: Windows drive letters live in paths
            if len(parts) != 3:
                print(f"skip {spec!r}: expected S:E:patch.mp4")
                continue
            s, e = int(parts[0]), int(parts[1])
            patch_path = parts[2].strip()
            ext = os.path.splitext(patch_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                img = cv2.imread(patch_path)
                patch = [img] if img is not None else []
            else:
                patch = read_frames(patch_path)
            n_span = e - s + 1
            if not patch or len(patch) != n_span:
                print(f"skip fuse {s}:{e}: patch has {len(patch)} frames, "
                      f"span needs {n_span}")
                continue
            h, w = frames[0].shape[:2]
            if patch[0].shape[:2] != (h, w):
                print(f"skip fuse {s}:{e}: patch size mismatch")
                continue
            if s < 1 or e >= len(frames) - 1:
                print(f"skip fuse {s}:{e}: ramp anchors fall outside the clip")
                continue
            for j, fi in enumerate(range(s, e + 1)):
                t = j / (n_span - 1) if n_span > 1 else 1.0
                wp = (fmax - (fmax - fmin) * t) if args.fuse_side == "after" \
                    else (fmin + (fmax - fmin) * t)
                pf = patch[j].astype(np.float32)
                mf = frames[fi].astype(np.float32)
                frames[fi] = ((1.0 - wp) * mf + wp * pf).astype(np.uint8)
            print(f"fused frames {s}..{e} ({args.fuse_side}, patch weight "
                  f"{fmax:.2f} -> {fmin:.2f}): frame count untouched")
            fixed += 1

    if args.replace:
        total = len(frames)
        for spec in args.replace.replace(";", ",").split(","):
            if ":" not in spec:
                print(f"skip {spec!r}: expected S:patch.mp4")
                continue
            s_str, patch_path = spec.split(":", 1)
            s = int(s_str)
            patch_path = patch_path.strip()
            ext = os.path.splitext(patch_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                img = cv2.imread(patch_path)
                if img is None:
                    print(f"skip S={s}: cannot read {patch_path}")
                    continue
                patch = [img]
                kind = "single image"
            else:
                patch = read_frames(patch_path)
                kind = f"{len(patch)} frames"
            h, w = frames[0].shape[:2]
            if patch[0].shape[:2] != (h, w):
                print(f"skip S={s}: patch size {patch[0].shape[1]}x"
                      f"{patch[0].shape[0]} != clip {w}x{h}")
                continue
            if s < 0 or s + len(patch) > total:
                print(f"skip S={s}: {len(patch)} frames would run past the clip end")
                continue
            for j, pf in enumerate(patch):
                frames[s + j] = pf
            print(f"replaced frames {s}..{s + len(patch) - 1} "
                  f"({kind} from {os.path.basename(patch_path)})")
        base, ext = os.path.splitext(args.video)
        out = args.out or f"{base}_repaired{ext or '.mp4'}"
        tmp = out + ".tmp.mp4"
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        ffmpeg = find_ffmpeg()
        audio_src = args.audio or args.video
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", tmp]
        if probe_audio(audio_src, ffmpeg):
            cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium",
                "-pix_fmt", "yuv420p", out]
        subprocess.run(cmd, check=True)
        os.remove(tmp)
        check = cv2.VideoCapture(out)
        n = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        print(f"written {out}: output frames={n} (input {total}) -> "
              f"{'OK' if n == total else 'FRAME COUNT MISMATCH'}")
        return

    for s in seams:
        if not (0 <= s < len(frames) - 1):
            print(f"skip S={s}: outside 0..{len(frames) - 2}")
            continue
        before = float(np.abs(gray[s + 1] - gray[s]).mean())
        a, b = blend_pair(frames[s], frames[s + 1], strength, args.mode)
        frames[s], frames[s + 1] = a.astype(np.uint8), b.astype(np.uint8)
        after = float(np.abs(
            cv2.cvtColor(frames[s + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            - cv2.cvtColor(frames[s], cv2.COLOR_BGR2GRAY).astype(np.float32)
        ).mean())
        print(f"seam S={s}: jump {before:.2f} -> {after:.2f} ({after / before:.0%})")
        fixed += 1
    if not (fixed or dissolve_fixed):
        return

    base, ext = os.path.splitext(args.video)
    out = args.out or f"{base}_repaired{ext or '.mp4'}"
    tmp = out + ".tmp.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                             24.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    ffmpeg = find_ffmpeg()
    audio_src = args.audio or args.video
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", tmp]
    if probe_audio(audio_src, ffmpeg):
        cmd += ["-i", audio_src,
                "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", out]
    subprocess.run(cmd, check=True)
    os.remove(tmp)

    check = cv2.VideoCapture(out)
    n = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    print(f"written {out}: {fixed} seam(s) repaired, output frames={n} "
          f"(input {len(frames)}) -> {'OK' if n == len(frames) else 'FRAME COUNT MISMATCH'}")


if __name__ == "__main__":
    main()
