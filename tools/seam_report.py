# -*- coding: utf-8 -*-
"""接缝标准测量工具（单机自检 + A/B 对照）。

用法
----
    # ① 单片自检：给缝的位置，判"台阶看不看得见"
    python seam_report.py D:/共享/MiniMaxH3/exp_4v10a_00075.mp4 --seams 51

    # ② 自动找可疑台阶（不知道缝在哪时用）
    python seam_report.py <video> --auto

    # ③ A/B 对照：两份成片，判"重去噪有没有注入伪纹理"
    python seam_report.py <关> <开> --seams 85,187,272 --window 12

判据（阈值是**用本机已知好坏的片子标定**出来的，不是拍的）
------------------------------------------------------
单片自检对每条缝给两个数：

  1. **局部倍数** = 该帧台阶 / ±12 帧中位   —— 抓"安静处的台阶"
  2. **全片百分位** = 该台阶在全片所有台阶里的百分位 —— 抓"闹处的离群点"

  两个数各管一头，缺一个都会漏。标定样本：
      00069 缝187  倍数 5.30 / 百分位 72%  -> 可见（坏）  ← 只有倍数抓得到
      00072 缝 68  倍数 3.55 / 百分位 99%  -> 可见（坏）  ← 只有百分位抓得到
      00070 缝187  倍数 1.72 / 百分位 37%  -> 干净
      00075 缝 51  倍数 1.36 / 百分位 89%  -> 干净（用户目视确认看不出）

  判定：倍数 < 2.0 且 百分位 < 95%  -> 干净
        否则                        -> 可疑（出一帧拼图人工看一眼）
        倍数 >= 3.0 或 百分位 >= 99% -> 明显可见

⚠️ **锐度变化必须结合缝的类型判读**：硬切落在真转镜上时，锐度大幅变化是**内容该有的变化**（例：00075 缝 51 锐度 +54%，那是"运动模糊 -> 锁定变清晰"，干净）。
   只有在**连续内容里**出现大锐度变化，才怀疑是重去噪注入了伪纹理。

⚠️ **单片自检测不出"注入伪纹理"** —— 那必须靠 A/B 对照（同一份一采、只差开关）比较自由区锐度。
   这是本工具的第 ③ 种用法。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

CLEAN_RATIO = 2.0      # 局部倍数门槛
CLEAN_PCT = 95.0       # 全片百分位门槛
BAD_RATIO = 3.0
BAD_PCT = 99.0

# ★ 绝对台阶下限（级）。来源：视觉复核报告 SEAM_VISUAL_REVIEW_20260924.md §6
#   静止段相邻帧差可低到 0.07~0.7 级，此时 1 级变化会算出 3.5~4.6 倍 -> 全部误报。
#   台阶低于此值：肉眼不可能看见，直接判干净，**不看倍率**。
#   ⚠️ 但下限不是免检：00070 缝 85 就是"数字干净、眼睛看到一条线"，目视终审不能省。
STEP_FLOOR = 2.0

# ★ 17k「独占帧」是结构性尖峰（同上报告 §5）：17 帧 token 组的首帧比邻居少一次
#   时间维平均 -> 天生更锐(+27~35%)且更暗(-0.5~0.8 级)。硬切边界只能落在 17k 上，
#   所以任何硬切缝都必然带这个基线。判读时必须先扣掉：邻域中位数里排除 17k，
#   --auto 也不再把 17k 当作候选缝。
def is_grid_frame(f, grid=17):
    """f 是 17 的整数倍帧（窗口边界只能落在这里）"""
    return f % grid == 0


def load(path):
    cap = cv2.VideoCapture(str(path))
    F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    cap.release()
    if not F:
        raise SystemExit(f"读不到帧: {path}")
    return F


def luma_stack(F):
    return np.array(
        [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)
         @ np.array([0.2126, 0.7152, 0.0722], np.float32) for f in F]
    )


def steps(lum):
    return np.abs(np.diff(lum, axis=0)).mean(axis=(1, 2))


def sharp(lum):
    return np.array([cv2.Laplacian(g, cv2.CV_32F).var() for g in lum])


def evaluate(lum, d1, lap, seam):
    i = int(seam) - 1
    if not (0 <= i < len(d1)):
        return None
    lo, hi = max(0, i - 12), min(len(d1), i + 13)
    # ★ 邻域中位数里排除 17k 帧：它们是结构性尖峰，会把中位数抬上去、把真台阶压下去
    idxs = [k for k in range(lo, hi) if k != i and not is_grid_frame(k + 1)]
    nb = d1[idxs] if len(idxs) >= 3 else np.concatenate([d1[lo:i], d1[i + 1:hi]])
    med = float(np.median(nb))
    step = float(d1[i])
    ratio = step / med if med > 1e-6 else 99.0
    pct = float((d1 < step).mean() * 100)
    pre = float(lap[max(0, i - 6):i + 1].mean())
    post = float(lap[i + 1:min(len(lap), i + 8)].mean())
    dsharp = (post - pre) / pre * 100 if pre > 1e-6 else 0.0
    return step, med, ratio, pct, pre, post, dsharp


def verdict(ratio, pct, step, hard_cut=False):
    if hard_cut:
        # ★ 硬切落在 17k「独占帧」上：台阶百分位高、锐度大涨都是**结构基线**
        #   （17k 天生更锐/更暗，且内容在此真变了）—— 只按局部倍数判。
        #   依据：SEAM_VISUAL_REVIEW_20260924.md §5（17k 基线）+ 00075 缝51/00078 缝51 实测
        if ratio < CLEAN_RATIO:
            return f"干净（硬切于 17k，倍数 {ratio:.2f}x；百分位/锐度为结构基线，豁免）"
        if ratio < BAD_RATIO:
            return f"可疑（看图确认；硬切 {ratio:.2f}x）"
        return f"明显可见（硬切 {ratio:.2f}x）"
    if step < STEP_FLOOR:
        return f"干净（台阶 {step:.2f} 级 < {STEP_FLOOR}）"
    if ratio >= BAD_RATIO or pct >= BAD_PCT:
        return "明显可见"
    if ratio < CLEAN_RATIO and pct < CLEAN_PCT:
        return "干净"
    return "可疑（看图确认）"


def auto_suspects(d1, k=5):
    """按"局部倍数"排，找出最可疑的台阶（排除首尾）。"""
    cand = []
    for i in range(2, len(d1) - 2):
        f = i + 1
        if is_grid_frame(f):          # ★ 17k 是结构性尖峰，不是候选缝
            continue
        if d1[i] < STEP_FLOOR:        # ★ 台阶太小，肉眼不可能看见
            continue
        lo, hi = max(0, i - 12), min(len(d1), i + 13)
        idxs = [k for k in range(lo, hi) if k != i and not is_grid_frame(k + 1)]
        nb = d1[idxs] if len(idxs) >= 3 else np.concatenate([d1[lo:i], d1[i + 1:hi]])
        med = float(np.median(nb))
        if med <= 1e-6:
            continue
        cand.append((float(d1[i] / med), f, float(d1[i])))
    cand.sort(reverse=True)
    return cand[:k]


def _strip(F, lo, hi, scale, crop=None, label=None):
    tiles = []
    for i in range(lo, hi):
        f = F[i]
        if crop is not None:  # (y0,y1,x0,x1)
            y0, y1, x0, x1 = crop
            f = f[y0:y1, x0:x1]
        t = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
    return np.hstack(tiles)


def sheets(F, seam, stem: Path, n=7):
    """一次出三张看图素材（给能看图的人/模型复核用）：
        1) 全帧条带        —— 看整体跳变、构图、亮度
        2) 中心 2 倍放大   —— 看纹理/锐度/细节是否一致
        3) 相邻帧差分伪彩  —— 看"突变"到底落在哪一帧、多大范围
    """
    lo = max(0, seam - n)
    hi = min(len(F), seam + n)
    H, W = F[0].shape[:2]
    # 中心 40% 区域
    cy0, cy1 = int(H * 0.30), int(H * 0.70)
    cx0, cx1 = int(W * 0.30), int(W * 0.70)

    outs = []
    # 1 全帧
    p = stem.with_name(f"{stem.name}_seam{seam}_full.png")
    cv2.imwrite(str(p), _strip(F, lo, hi, 0.5))
    outs.append(p)
    # 2 中心放大
    p2 = stem.with_name(f"{stem.name}_seam{seam}_zoom.png")
    cv2.imwrite(str(p2), _strip(F, lo, hi, 1.0, crop=(cy0, cy1, cx0, cx1)))
    outs.append(p2)
    # 3 差分伪彩（相邻帧亮度差，放大对比）
    diffs = []
    for i in range(lo, hi - 1):
        a = cv2.cvtColor(F[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(F[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = np.clip(np.abs(b - a) * 6.0, 0, 255).astype(np.uint8)
        d = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        t = cv2.resize(d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{i}->{i+1}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        diffs.append(t)
    p3 = stem.with_name(f"{stem.name}_seam{seam}_diff.png")
    cv2.imwrite(str(p3), np.hstack(diffs))
    outs.append(p3)
    return outs


def main():
    ap = argparse.ArgumentParser(description="接缝标准测量")
    ap.add_argument("video", nargs="+", help="一个视频=自检；两个=A/B 对照")
    ap.add_argument("--seams", default="", help="缝的帧号，逗号分隔")
    ap.add_argument("--auto", action="store_true", help="自动找最可疑的台阶")
    ap.add_argument("--window", type=int, default=12, help="A/B 时自由区半径（帧）")
    ap.add_argument("--sheet", action="store_true", help="出看图素材（全帧/中心放大/差分）")
    ap.add_argument("--hard-cut", default="", help="这些缝是 hunt 接受的硬切（残差小）：豁免 17k 结构基线（百分位/锐度）")
    ap.add_argument("--outdir", default=r"D:\comfyui\_hardcut_work\seam_sheets",
                    help="看图素材输出目录（默认 _hardcut_work\\seam_sheets）")
    args = ap.parse_args()

    paths = args.video
    seams = [int(s) for s in args.seams.split(",") if s.strip()]
    hard_cuts = {int(s) for s in args.hard_cut.split(",") if s.strip()}

    print("=" * 78)
    print(f"接缝测量  |  局部倍数门槛 {CLEAN_RATIO}  |  全片百分位门槛 {CLEAN_PCT}%")
    print("=" * 78)

    data = []
    for p in paths:
        F = load(p)
        lum = luma_stack(F)
        d1 = steps(lum)
        lap = sharp(lum)
        data.append((p, F, lum, d1, lap))
        print(f"\n文件: {os.path.basename(p)}   {len(F)} 帧   "
              f"全片 d1 中位 {np.median(d1):.2f}   最大 {d1.max():.2f}")

    if args.auto and not seams:
        print("\n★ 自动找最可疑的台阶（按局部倍数）：")
        for r, f, s in auto_suspects(data[0][3]):
            print(f"   帧{f:>5}  倍数 {r:5.2f}x  台阶 {s:6.2f}")
        return 0

    if not seams:
        print("\n（没给 --seams；加 --auto 可以自动找）")
        return 0

    print()
    hdr = f"{'缝':>6} {'台阶':>8} {'局部中位':>9} {'倍数':>7} {'百分位':>8} {'锐度前':>8} {'锐度后':>8} {'锐度变化':>9}  判定"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for p, F, lum, d1, lap in data:
        rows = []
        for s in seams:
            e = evaluate(lum, d1, lap, s)
            if e is None:
                continue
            step, med, ratio, pct, pre, post, dsharp = e
            rows.append((s, ratio, pct))
            grid = " [17k]" if is_grid_frame(s) else ""
            print(f"{s:>6} {step:>8.2f} {med:>9.2f} {ratio:>7.2f} {pct:>7.0f}% "
                  f"{pre:>8.1f} {post:>8.1f} {dsharp:>8.1f}%  "
                  f"{verdict(ratio, pct, step, hard_cut=(s in hard_cuts))}{grid}")
        results[p] = rows

        if args.sheet:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            stem = outdir / Path(p).stem     # 默认落到工作区，别弄脏输出目录
            for s in seams:
                for out in sheets(F, s, stem):
                    print(f"   看图素材 -> {out}")

    if len(paths) == 2:
        print("\n★ A/B 对照（自由区锐度/亮度，判重去噪有没有注入伪纹理）：")
        a, b = data[0], data[1]
        for s in seams:
            i = s - 1
            lo, hi = max(0, i - args.window), min(len(a[3]), i + args.window + 1)
            la_a = float(a[4][lo:hi].mean())
            la_b = float(b[4][lo:hi].mean())
            lu_a = float(a[2][lo:hi].mean())
            lu_b = float(b[2][lo:hi].mean())
            print(f"   缝{s:>4} ±{args.window} 帧：锐度 {la_a:8.1f} -> {la_b:8.1f}  "
                  f"({(la_b - la_a) / la_a * 100 if la_a else 0:+6.1f}%)   "
                  f"亮度 {lu_a:6.2f} -> {lu_b:6.2f} ({lu_b - lu_a:+5.2f})")
        print("   判读：锐度**大涨（>+15%）** = 疑似注入伪纹理；锐度大跌 = 被磨糊。")
        print("   ⚠️ 若该缝是 hunt 接受的硬切（残差小），锐度变化属内容变化，不适用此判读。")

    print("\n提示：判定为「可疑」时，用 --sheet 出拼图人工确认 —— 数字会漏，图不会。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
