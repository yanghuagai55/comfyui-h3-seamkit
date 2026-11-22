# -*- coding: utf-8 -*-
"""活跃区掩码：全局运镜必须被补偿掉，局部运动/静立人物必须被抓住"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, r"D:/comfyui/ComfyUI")

from route3.masks import active_mask, mask_stats, dilate
from route3.splitstream_attn import route3_plan

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(n, ok, d=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + n + ("  " + d if d else ""))
    if not ok:
        FAIL.append(n)


def shift_img(x, dy, dx):
    C, H, W = x.shape
    ys = torch.arange(H, device=x.device, dtype=x.dtype) - dy
    xs = torch.arange(W, device=x.device, dtype=x.dtype) - dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.stack([(gx + 0.5) / W * 2 - 1, (gy + 0.5) / H * 2 - 1], -1).unsqueeze(0)
    return torch.nn.functional.grid_sample(x.unsqueeze(0), g, align_corners=False,
                                           padding_mode="border").squeeze(0)


torch.manual_seed(0)
T, C, H, W = 8, 3, 96, 128
tex = torch.rand(C, H, W, device=DEV) * 0.6 + 0.2
pan = torch.stack([shift_img(tex, 2 * t, 3 * t) for t in range(T)])

m1, i1 = active_mask(pan, ring=1)
s1 = mask_stats(m1)
check("纯全局运镜 → 活跃区很小（压缩不被吃掉）", s1["mean"] < 0.20,
      "mean=%.1f%% thr=%.4f" % (100 * s1["mean"], i1["thr"]))

r0, c0 = 40, 50
withb = pan.clone()
for t in range(T):
    # 局部物体：屏幕位置固定（不跟镜头走）→ 相对背景在运动
    withb[t, :, r0:r0 + 16, c0:c0 + 16] += 0.8

m2, _ = active_mask(withb, ring=1)
s2 = mask_stats(m2)
check("全局运镜 + 局部物体 → 只抓局部", 0.03 < s2["mean"] < 0.45,
      "mean=%.1f%%" % (100 * s2["mean"]))
ry, rx = (r0 + 8) // 2, (c0 + 8) // 2
check("局部物体确实被标成活跃",
      bool(m2[4, max(0, ry - 2):ry + 2, max(0, rx - 2):rx + 2].any()),
      "probe@(%d,%d)" % (ry, rx))

d = torch.zeros(2, 8, 8, dtype=torch.bool, device=DEV)
d[:, 3, 3] = True
check("dilate r=1 扩到 3x3", int(dilate(d, 1)[0].sum()) == 9, "sum=%d" % dilate(d, 1)[0].sum())

mp = torch.nn.functional.adaptive_avg_pool2d(m2.float().unsqueeze(1), (23, 40)).squeeze(1) > 0.5
big = torch.zeros(37, 23, 40, dtype=torch.bool, device=DEV)
big[:T] = mp
p = route3_plan(0, 37, 46, 80, 5, fg_mask=big.reshape(37, -1))
check("掩码能直接喂给 plan", p.fg_local.numel() + p.bg_local.numel() <= p.video_rows,
      "kcpt=%d/%d" % (p.fg_local.numel() + p.bg_local.numel(), p.video_rows))

# ---- 静立人物保底：微动能量远小于画面里真正在动的东西 ----
torch.manual_seed(1)
T2, H2, W2 = 10, 96, 128
bg = torch.full((C, H2, W2), 0.5, device=DEV)
frames2 = bg.unsqueeze(0).repeat(T2, 1, 1, 1)
for t in range(T2):
    frames2[t, :, 10:40, 90:120] += (t % 2) * 0.35          # 窗帘：大幅闪烁
    frames2[t, :, 50:80, 40:70] += 0.02 * ((t % 3) - 1)     # 静立人物：微抖

hm_only, _ = active_mask(frames2, ring=1, micro=False)
with_micro, _ = active_mask(frames2, ring=1, micro=True)


def cov(m, sl):
    return float(m[:, sl[0], sl[1]].float().mean())


person = (slice(50 // 2, 80 // 2), slice(40 // 2, 70 // 2))  # 掩码在 patch 网格（1/2 分辨率）
check("单一阈值会漏掉静立人物", cov(hm_only, person) < 0.5,
      "覆盖率 %.0f%%" % (100 * cov(hm_only, person)))
check("微动保底把静立人物捞回来", cov(with_micro, person) > 0.9,
      "覆盖率 %.0f%%" % (100 * cov(with_micro, person)))
check("保底不会把整帧都点亮", float(with_micro.float().mean()) < 0.45,
      "总覆盖 %.0f%%" % (100 * float(with_micro.float().mean())))

print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
sys.exit(1 if FAIL else 0)
