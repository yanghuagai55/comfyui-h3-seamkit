"""成片硬切点分析：找画面突变的真实位置，和理论切点比对。

用法：
    python analyze_cut.py <mp4> [--cuts 4.958,8.500] [--sheet 100,130]

输出：fps/帧数/中位帧差、突变点排名、每个理论切点的实测偏移、
      以及（--sheet 时）一张接触表 PNG，方便肉眼复核。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import cv2
import numpy as np

FPS_FALLBACK = 24.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cuts", default="", help="理论切点，秒，逗号分隔")
    ap.add_argument("--sheet", default="", help="导出接触表：起帧,止帧")
    ap.add_argument("--fps", type=float, default=0.0, help="覆盖容器里的 fps")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("打不开视频: %s" % args.video)
        return 2
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n < 3:
        print("帧数太少 (%d)" % n)
        return 2

    # 灰度缩小后逐帧平均绝对差：对"画面整体变化"敏感，对噪点不敏感
    gs = [
        cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        for f in frames
    ]
    diffs = [float(np.abs(gs[i] - gs[i - 1]).mean()) for i in range(1, n)]
    med = statistics.median(diffs)

    print("=" * 66)
    print("文件      : %s" % os.path.basename(args.video))
    print("分辨率    : %dx%d" % (frames[0].shape[1], frames[0].shape[0]))
    print("fps       : %.3f  帧数: %d  时长: %.3fs" % (fps, n, n / fps))
    print("中位帧差  : %.2f   （超过它的 2.5 倍就算一次画面突变）" % med)
    print()

    spikes = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:10]
    spikes = sorted(i for i in spikes if diffs[i] > med * 2.5)
    print("--- 画面突变点（> 2.5× 中位）---")
    if not spikes:
        print("  （没有——整片没有明显切换）")
    for i in spikes:
        print("  帧 %3d  时间 %6.3fs  差值 %6.2f  = %.1f× 中位"
              % (i + 1, (i + 1) / fps, diffs[i], diffs[i] / med))

    if args.cuts:
        print()
        print("--- 理论切点 vs 实测突变 ---")
        cuts = [float(s) for s in args.cuts.replace(";", ",").split(",") if s.strip()]
        for c in cuts:
            want = int(round(c * fps))
            # Only count real spikes — a plain neighbouring frame would report a
            # meaningless "offset 0" and hide the fact that nothing happened here.
            near = [(abs(i + 1 - want), i) for i in range(len(diffs))
                    if abs(i + 1 - want) <= 30 and diffs[i] > med * 2.5]
            if not near:
                print("  理论 %.3fs (帧 %3d): 附近 30 帧内【没有突变】"
                      " — 模型没在这里切镜" % (c, want))
                continue
            dist, i = min(near)
            print("  理论 %.3fs (帧 %3d)  →  最近突变 帧 %3d (%.3fs)  偏移 %+d 帧  幅度 %.1f×"
                  % (c, want, i + 1, (i + 1) / fps, (i + 1) - want, diffs[i] / med))

    if args.sheet:
        a, b = (int(x) for x in args.sheet.split(","))
        sel = list(range(a, min(b, n) + 1))
        cols = 6
        rows = (len(sel) + cols - 1) // cols
        sheet = np.zeros((135 * rows, 240 * cols, 3), np.uint8)
        for k, fi in enumerate(sel):
            r, c = divmod(k, cols)
            img = cv2.resize(frames[fi], (240, 135))
            cv2.putText(img, "f%d %.2fs" % (fi, fi / fps), (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            sheet[r * 135:(r + 1) * 135, c * 240:(c + 1) * 240] = img
        out = os.path.join(os.path.dirname(args.video) or ".", "sheet_%d-%d.png" % (a, b))
        cv2.imwrite(out, sheet)
        print("\n接触表: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
