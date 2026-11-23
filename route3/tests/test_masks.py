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

# ---- 12fps 动漫隔帧桥接 + 切镜绝不串镜测试 ----
from route3.masks import _temporal_bridge_same_shot
m_twos = torch.zeros(7, 8, 8, dtype=torch.bool)
# 镜头1 (0..3)：隔帧 1, 0, 1, 0；镜头2 (4..6)：全 0（t=3->4 为镜头硬切）
m_twos[0, 2:5, 2:5] = True
m_twos[2, 2:5, 2:5] = True
m_twos[4, 2:5, 2:5] = True  # 若不防切镜，t=3 会被 t=2 & t=4 误桥接！
fr_twos = torch.zeros(7, 3, 8, 8)
fr_twos[:4] = 0.1
fr_twos[4:] = 0.9
bridged = _temporal_bridge_same_shot(m_twos, fr_twos)
check("12fps 动漫隔帧空洞 (t=1) 被同镜头闭运算填平", bool(bridged[1, 3, 3].item()) is True)
check("硬切边界 (t=3->4) 绝不跨镜桥接掩码", bool(bridged[3, 3, 3].item()) is False)

# ---- 保守空间降采样 + patch 级 ring 回归（消灭运动指尖/发梢被 avg_pool>0.5 啃出缺口）----
from route3.mask_provider import MaskProvider
fr_thin = torch.zeros(6, 3, 64, 64, device=DEV)
fg_thin = torch.zeros(6, 64, 64, dtype=torch.bool, device=DEV)
fg_thin[:, 20:22, 20:22] = True   # 极细 2x2px 指尖（仅占 16x16 patch 的 1/64 << 50%）
m_thin0, _ = active_mask(fr_thin, fg_mask=fg_thin, ring=0, patch_grid=(16, 16), micro=False)
m_thin1, _ = active_mask(fr_thin, fg_mask=fg_thin, ring=1, patch_grid=(16, 16), micro=False)
check("极细指尖降采样绝不被 >0.5 均值阈值啃掉 (ring=0 仍保留)", bool(m_thin0[0, 5, 5].item()) is True)
check("ring=1 在 patch 网格上真实外扩 3x3 保护环", int(m_thin1[0].sum().item()) == 9,
      "sum=%d" % int(m_thin1[0].sum().item()))
prov_thin = MaskProvider(m_thin0.cpu())
g_thin = prov_thin.for_grid(latent_t=2, lat_h=16, lat_w=16).view(2, 8, 8)
check("MaskProvider.for_grid 保守空间 max_pool 绝不啃掉细指尖", bool(g_thin[0, 2, 2].item()) is True)

# ---- 硬切镜不炸 95% 假前景 + 同镜头展开绝不跨镜串帧 ----
from route3.masks import _expand_temporal_mask, _detect_frame_cuts
fr_cut = torch.zeros(10, 3, 32, 32, device=DEV)
fr_cut[:5] = 0.1
fr_cut[5:] = 0.9
m_cut, _ = active_mask(fr_cut, fg_mask=None, ring=0, patch_grid=(16, 16), micro=False)
check("硬切镜当帧运动尖峰已屏蔽（不炸 95% 假前景）", float(m_cut[5].float().mean()) < 0.05,
      "cut_cov=%.2f" % float(m_cut[5].float().mean()))
cuts_det = _detect_frame_cuts(fr_cut)
sub_m = torch.zeros(3, 8, 8, dtype=torch.bool)
sub_m[0, :3, :3] = True     # idx=0 (镜头1)
sub_m[1, 5:, 5:] = True     # idx=5 (镜头2)
sub_m[2, 5:, 5:] = True     # idx=9 (镜头2)
exp_m = _expand_temporal_mask(sub_m, [0, 5, 9], 10, cuts=cuts_det)
check("抽样掩码按镜头硬切隔离展开：镜头1尾帧 t=4 绝不吸附镜头2(t=5)",
      bool(exp_m[4, 1, 1].item()) is True and bool(exp_m[4, 6, 6].item()) is False)

print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
sys.exit(1 if FAIL else 0)
