# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Seam/flicker forensics for a rendered clip.

Usage:
    python check_cut_flicker.py <clip.mp4> [--cuts 119,238,323]

Prints (1) the strongest frame-to-frame changes - where the model really
switched shots - and (2) an alternation test around each expected cut
(flicker shows as a period-2 pattern: odd frames alike, even frames
alike).  If the strongest changes sit far from your cuts, the executor
boundaries are landing inside a shot: raise seam_tolerance_frames.
"""
import io as _io
import sys

import cv2
import numpy as np

sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
V = sys.argv[1]
CUTS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[2] == "--cuts" else []
cap = cv2.VideoCapture(V)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
n = len(frames)
print(f"clip: {n} frames ({n/24:.3f}s)")
gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
small = [g[::4, ::4] for g in gray]


def d(i, j):
    return float(np.abs(small[i] - small[j]).mean())


diffs = np.array([d(i - 1, i) for i in range(1, n)])
med = float(np.median(diffs))
print(f"frame-diff median {med:.2f}  max {diffs.max():.2f} @ frame {int(diffs.argmax())+1}")
print()
print("top-8 change frames (boundary i-1|i):")
order = np.argsort(-diffs)[:8]
for i in sorted(order):
    print(f"  boundary {i+1:>4}|{i+2:<4} diff {diffs[i]:7.2f}  ({diffs[i]/med:5.2f}x)")
print()

# expected cuts (segment length 102) -> boundaries 102|103, 204|205, 306|307
print("=== neighbourhood of each cut: alternation test ===")
print("  (flicker = |f(i)-f(i+2)| << |f(i)-f(i+1)|  -> period-2 alternation)")
for c in (CUTS or [102, 204, 306]):
    print(f"\n-- boundary {c}|{c+1} --")
    print("   frame | d(prev) | d(next) | d(i,i+2) | alt-ratio | luma")
    for i in range(max(1, c - 4), min(n - 2, c + 5)):
        dp = d(i - 1, i)
        dn = d(i, i + 1)
        d2 = d(i, i + 2)
        ratio = d2 / dn if dn > 1e-6 else 0.0
        luma = float(gray[i].mean())
        flag = "  <-- ALTERNATING" if ratio < 0.55 else ""
        print(f"   {i:>5} | {dp:7.2f} | {dn:7.2f} | {d2:8.2f} | {ratio:9.2f} | {luma:6.1f}{flag}")
