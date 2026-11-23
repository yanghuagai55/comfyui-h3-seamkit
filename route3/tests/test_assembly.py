# -*- coding: utf-8 -*-
"""法 A / 法 B 的 A/B：已知平移下，法 B 必须明显优于线性插值"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, r"D:/comfyui/ComfyUI")
from route3.assembly import (estimate_tile_flow, warp, reconstruct_moving,
                             reconstruct_static, lerp_residual, flow_residual,
                             detect_latent_cuts, cut_pin_tokens,
                             token_frame_center, motion_pin_tokens)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []
def check(n, ok, d=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + n + ("  " + d if d else ""))
    if not ok: FAIL.append(n)

def shift_img(x, dy, dx):
    C, H, W = x.shape
    g = torch.zeros(1, H, W, 2, device=x.device, dtype=x.dtype)
    ys = torch.arange(H, device=x.device, dtype=x.dtype) - dy
    xs = torch.arange(W, device=x.device, dtype=x.dtype) - dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g[0, ..., 0] = (gx + 0.5) / W * 2 - 1
    g[0, ..., 1] = (gy + 0.5) / H * 2 - 1
    return torch.nn.functional.grid_sample(x.unsqueeze(0), g, align_corners=False,
                                           padding_mode="border").squeeze(0)

torch.manual_seed(0)
H, W, C = 48, 80, 4
tex = torch.randn(C, H, W, device=DEV) + torch.sin(
    torch.arange(W, device=DEV).float() * 0.9)[None, None, :] * 3
D = torch.randn(C, H, W, device=DEV) * 0.3        # 细节残差
shift = (2, 4)

f0, f1, f2 = tex, shift_img(tex, *[s // 2 for s in shift]), shift_img(tex, *shift)
d0, d2 = D, shift_img(D, *shift)
z = torch.stack([f0, f1, f2])

gt1 = f1 + shift_img(D, *[s // 2 for s in shift])

fl = estimate_tile_flow(z[0], z[2], tiles=(6, 8), search=5)
med = fl.reshape(-1, 2).median(0).values
check("分块位移估计 ≈ (2,4)", abs(float(med[0]) - 2) < 1.5 and abs(float(med[1]) - 4) < 1.5,
      "median=(%.1f, %.1f)" % (med[0], med[1]))

delta_keys = {0: d0, 2: d2}
out_flow = reconstruct_moving(z, delta_keys, [0, 2], method="flow", tiles=(6, 8), search=5)
out_lerp = reconstruct_moving(z, delta_keys, [0, 2], method="lerp")
e_flow = (out_flow[1] - gt1).abs().mean().item()
e_lerp = (out_lerp[1] - gt1).abs().mean().item()
check("法 B 误差 < 法 A", e_flow < e_lerp * 0.6,
      "flow=%.4f lerp=%.4f (%.0f%% 更低)" % (e_flow, e_lerp, 100 * (1 - e_flow / max(e_lerp, 1e-9))))

st = reconstruct_static(torch.stack([f0, f1, f2]), f0)
check("静机位档：整片铺同一张高清底", bool((st[2] - f0).abs().max().item() < 1e-6))

m = torch.zeros(3, H, W, dtype=torch.bool, device=DEV); m[:, 10:30, 20:50] = True
st2 = reconstruct_static(torch.stack([f0, f1, f2]), f0, fg_mask=m)
check("静机位档：前景保留原帧",
      bool((st2[2][:, 10:30, 20:50] - f2[:, 10:30, 20:50]).abs().max().item() < 1e-6))

# 尾帧外推（模拟 T=37 / s_bg=5 → anchors 止于 35，t=36 必须顺流外推而不是退回糊底）
f3 = shift_img(tex, 3, 6)
gt3 = f3 + shift_img(D, 3, 6)
z4 = torch.stack([f0, f1, f2, f3])
out4 = reconstruct_moving(z4, delta_keys, [0, 2], method="flow", tiles=(6, 8), search=5)
e_tail_flow = (out4[3] - gt3).abs().mean().item()
e_tail_raw = (f3 - gt3).abs().mean().item()
check("尾帧外推（t > anchors[-1]）显著优于退回糊底", e_tail_flow < e_tail_raw * 0.5,
      "tail_flow=%.4f raw_blur=%.4f" % (e_tail_flow, e_tail_raw))

# 湿地面(低纹理) 紧挨 红砖墙(强纹理) 反例测试：绝不能把砖墙残差 warp 到平坦地面上！
C24 = 24
wall_floor_a = torch.zeros(C24, H, W, device=DEV)
wall_floor_b = torch.zeros(C24, H, W, device=DEV)
# 上半部：强纹理红砖墙；下半部：近乎平坦的湿地面 + 极微小噪声
wall_tex = torch.randn(C24, H // 2, W, device=DEV) * 2.0
wall_floor_a[:, :H // 2, :] = wall_tex
wall_floor_b[:, :H // 2, :] = wall_tex
wall_floor_a[:, H // 2:, :] = torch.randn(C24, H // 2, W, device=DEV) * 0.01
wall_floor_b[:, H // 2:, :] = torch.randn(C24, H // 2, W, device=DEV) * 0.01
fl_wf, conf_wf = estimate_tile_flow(wall_floor_a, wall_floor_b, tiles=(6, 8), search=4, return_conf=True)
# 砖墙上有巨大残差 (+5.0)，地面残差为 0；门控必须挡住地面 tile，不让 +5.0 污块贴到地面！
d_wall = torch.zeros(C24, H, W, device=DEV)
d_wall[:, :H // 2, :] = 5.0
res_mid = flow_residual(d_wall, d_wall, 0.5, fl_wf, conf=conf_wf)
floor_blotch = float(res_mid[:, H // 2 + 2:, :].abs().max())
check("湿地面低纹理门控：绝不把红砖墙残差贴到平坦地面", floor_blotch < 1e-4,
      "floor_max_blotch=%.2e floor_valid=%d" % (floor_blotch, int((conf_wf[3:] >= 0.55).sum())))

# 跨镜头硬切防污染测试（复现今晚 10s 双镜片 f119 跨镜蓝天糊进镜头1的真 bug）：
# tokens 0..5 为镜头1（底图 -2.0，残差 +0.1），tokens 6..9 为镜头2（底图 +2.0，残差 +10.0）
# anchors = [0, 3, 6, 9]：t=4,5 处于镜头1尾部，绝不能借用 t=6（镜头2）的 +10.0 残差！
z_cut = torch.zeros(10, 4, 16, 16, device=DEV)
z_cut[:6] = -2.0
z_cut[6:] = +2.0
d_cut = {
    0: torch.full((4, 16, 16), 0.1, device=DEV),
    3: torch.full((4, 16, 16), 0.1, device=DEV),
    6: torch.full((4, 16, 16), 10.0, device=DEV),
    9: torch.full((4, 16, 16), 10.0, device=DEV),
}
rec_cut = reconstruct_moving(z_cut, d_cut, [0, 3, 6, 9], method="lerp", cut_protect=True)
leak_shot1 = (rec_cut[5] - (-1.9)).abs().max().item()
leak_shot2 = (rec_cut[7] - (12.0)).abs().max().item()
check("跨切镜自动断链：镜头2残差绝不泄漏到镜头1尾帧 (t=5)", leak_shot1 < 1e-5 and leak_shot2 < 1e-5,
      "shot1_leak=%.2e shot2_err=%.2e" % (leak_shot1, leak_shot2))

# I/P 相位隔离测试（复现 2026-11-23 绿幕边缘光晕与暗边污影）：
# 设 I-token (t%5==0) 残差为 +10.0，P-token (t%5!=0) 残差为 +1.0：
# - 任何被删 P-token (如 t=4, t=6) 绝不能混入 I-token 的 +10.0（否则炸暗边/亮晕）
# - 任何被删 I-token (如 s_bg=4 下 t=5) 绝不能混入 P-token 的 +1.0！
z_ip = torch.zeros(12, 4, 16, 16, device=DEV)
anc_ip = [0, 2, 7, 10, 11]  # I-锚: 0, 10 (t=5 被删); P-锚: 2, 7, 11 (t=4, 6 被删)
d_ip = {t: torch.full((4, 16, 16), 10.0 if t % 5 == 0 else 1.0, device=DEV) for t in anc_ip}
rec_ip = reconstruct_moving(z_ip, d_ip, anc_ip, method="lerp", phase_split=True)
err_i_pure = (rec_ip[5] - 10.0).abs().max().item()
err_p_pure = max((rec_ip[4] - 1.0).abs().max().item(), (rec_ip[6] - 1.0).abs().max().item())
check("I/P 相位严格隔离：I↔I 与 P↔P 互不污染 (消灭绿幕 17k 帧边缘光晕)",
      err_i_pure < 1e-5 and err_p_pure < 1e-5,
      "i_err=%.2e p_err=%.2e" % (err_i_pure, err_p_pure))

# 运动前景残差屏蔽测试（复现 2026-11-23 exp_4v10a_00004 f054/f055 运动手臂残影）：
# 在 t0=2 时区域 A=[2:6,2:6] 是纯 BG（残差 +1.0），在 t1=7 时手臂移入区域 A（FG，残差 +20.0）；
# 区域 B=[10:14,10:14] 在 t0=2 与 t1=7 均被 FG 占据（残差 +20.0），而在中间帧 t=4 露出背景。
# 期望：t=4 上区域 A 必须 100% 取干净侧 t0=+1.0（绝不混入 +20.0），区域 B 归 0.0（绝不贴 +20.0）！
z_fg = torch.zeros(10, 4, 16, 16, device=DEV)
anc_fg = [0, 2, 5, 7, 9]
fm_fg = torch.zeros(10, 16, 16, dtype=torch.bool, device=DEV)
fm_fg[7, 2:6, 2:6] = True          # 区域 A 仅在 t=7 被手臂占
fm_fg[2, 10:14, 10:14] = True      # 区域 B 在 t=2 和 t=7 都被占
fm_fg[7, 10:14, 10:14] = True
d_fg = {}
for t in anc_fg:
    dt = torch.full((4, 16, 16), 1.0, device=DEV)
    if t == 7:
        dt[:, 2:6, 2:6] = 20.0
    if t in (2, 7):
        dt[:, 10:14, 10:14] = 20.0
    d_fg[t] = dt
rec_fg = reconstruct_moving(z_fg, d_fg, anc_fg, method="lerp", phase_split=True, fg_mask=fm_fg)
err_area_a = (rec_fg[4, :, 3:5, 3:5] - 1.0).abs().max().item()
err_area_b = (rec_fg[4, :, 11:13, 11:13] - 0.0).abs().max().item()
check("reconstruct_moving 前景残差屏蔽：运动手臂 δ 绝不泄漏到露出背景 (消灭 f054 残影)",
      err_area_a < 1e-5 and err_area_b < 1e-5,
      "areaA_err=%.2e areaB_err=%.2e" % (err_area_a, err_area_b))

# 亚像素位移精修回归（推镜 <0.5 latent 格 = <8 像素，旧版误锁死 0.0）
f_sub = shift_img(tex, 0.0, 0.35)
fl_sub = estimate_tile_flow(tex, f_sub, tiles=(4, 4), search=3)
med_sub_x = float(fl_sub[..., 1].median().item())
check("亚像素推镜位移估计 (真值 +0.35 格 ≈ 5.6px，不再被误锁 0.0)",
      0.18 < med_sub_x < 0.52, "est_dx=%.3f" % med_sub_x)

# 相位感知硬切检测 + 半跨镜 token (t=17) 检测（复现 exp_4v10a_lsd_00001 f058..065 爆炸）
# 模拟 H3 真实 I/P 分布跳变（每 t%5==0 加 +1.5 偏置），镜头1=0..16，t=17 为半跨镜(50%镜1+50%镜2)，镜头2=18..36
z_straddle = torch.zeros(37, 4, 16, 16, device=DEV)
shot1_base = torch.randn(4, 16, 16, device=DEV)
shot2_base = torch.randn(4, 16, 16, device=DEV) + 2.0
for t in range(37):
    if t < 17:
        z_straddle[t] = shot1_base + 0.03 * t
    elif t == 17:
        z_straddle[t] = 0.5 * (shot1_base + 0.03 * 16) + 0.5 * shot2_base
    else:
        z_straddle[t] = shot2_base + 0.02 * (t - 18)
    if t % 5 == 0:
        z_straddle[t] = z_straddle[t] * 1.35 + 1.2   # 强烈 I-token 分布跳变
cuts_st = detect_latent_cuts(z_straddle)
pins_st = cut_pin_tokens(cuts_st, 37, radius=1)
check("相位感知硬切检测：免疫 100%% I↔P 跳变并精准抓出半跨镜 t=17 与 t=18",
      cuts_st == [17, 18] and all(x in pins_st for x in (16, 17, 18, 19)),
      "cuts=%s pins=%s" % (cuts_st, pins_st))

# 2026-11-23 沙发角与格子毛巾体积呼吸回归 1：真实 24fps 帧中心坐标 token_frame_center
check("token_frame_center 精确映射 1 帧 I-token 与 4 帧 P-token 帧中心",
      token_frame_center(0) == 0.0 and token_frame_center(1) == 2.5
      and token_frame_center(4) == 14.5 and token_frame_center(5) == 17.0
      and token_frame_center(6) == 19.5)

# 2026-11-23 沙发角与格子毛巾体积呼吸回归 2：前景掩码紧邻外缘（如左侧木质扶手外缘）绝不掉入 3x3 膨胀真空死区
z_adj = torch.zeros(6, 4, 16, 16, device=DEV)
fm_adj = torch.zeros(6, 16, 16, dtype=torch.bool, device=DEV)
fm_adj[:, :, :4] = True  # x=0..3 为 FG，x=4（紧邻扶手外缘！）为纯 BG
d_adj = {0: torch.full((4, 16, 16), 2.5, device=DEV), 1: torch.full((4, 16, 16), 2.5, device=DEV),
         4: torch.full((4, 16, 16), 2.5, device=DEV), 5: torch.full((4, 16, 16), 2.5, device=DEV)}
rec_adj = reconstruct_moving(z_adj, d_adj, [0, 1, 4, 5], method="lerp", fg_mask=fm_adj)
adj_edge_err = (rec_adj[2, :, :, 4] - 2.5).abs().max().item()
check("前景掩码紧邻外缘 (x=4) 保持完整背景残差 δ（消灭 t=13..14 扶手外缘 -4.2px 塌缩死区）",
      adj_edge_err < 1e-5, "adj_edge_err=%.2e" % adj_edge_err)

# 2026-11-23 沙发角与格子毛巾体积呼吸回归 3：motion_pin_tokens 高速推镜段自动加密中点 P-锚，静止段 0 额外锚点
z_fast_slow = torch.zeros(17, 4, 32, 32, device=DEV)
base_fs = torch.randn(4, 32, 32, device=DEV) + torch.sin(torch.arange(32, device=DEV).float() * 0.7)[None, None, :] * 3.0
for t in range(17):
    if t <= 8:
        # 镜头1 (t=0..8)：高速横移/推镜（每 token 位移 1.0 格，3 token 跨度位移 3.0 格 >> 0.75）
        z_fast_slow[t] = shift_img(base_fs, 0.0, float(t) * 1.0)
    else:
        # 镜头2 (t=9..16)：固定机位（0 位移）
        z_fast_slow[t] = base_fs
mpins = motion_pin_tokens(z_fast_slow, s_bg=3, cuts=[9], flow_thr=0.75)
check("motion_pin_tokens：高速运镜段 (t=0..8) 自动钉住双丢帧中点 P-锚 (2,6)，静止段 (t=9..16) 零冗余",
      all(p in mpins for p in (2, 6)) and all(p < 9 for p in mpins),
      "motion_pins=%s" % mpins)

# 2026-11-23 沙发角与格子毛巾体积呼吸回归 4：前镜高速推镜 (t=0..16) + 半跨镜切点 (t=17) + 后镜固定机位 (t=18..36)
# 绝不可将前镜末个 P-token t=16 误判为切点（否则会切断 t=13,14 与 t=16 的连接并漏钉 13,14）
torch.manual_seed(42)
z_asym = torch.zeros(37, 24, 32, 32, device=DEV)
s1_pat = torch.randn(24, 32, 32, device=DEV) + torch.sin(torch.arange(32, device=DEV).float() * 0.6)[None, None, :] * 2.5
s2_pat = torch.randn(24, 32, 32, device=DEV) * 1.4 - 1.5
frame_ctr = [token_frame_center(t) for t in range(37)]
for t in range(37):
    fc = frame_ctr[t]
    if t <= 16:
        # Shot 1: 高速横移推镜（跨 I-token 的 14->16 帧距为 5 帧，比常规 4 帧大 25%）
        z_asym[t] = shift_img(s1_pat, 0.0, fc * 0.25) + 0.012 * fc
    elif t == 17:
        # t=17: 25% Shot 1 + 75% Shot 2 半跨镜混合
        z_s1_end = shift_img(s1_pat, 0.0, frame_ctr[16] * 0.25) + 0.012 * frame_ctr[16]
        z_asym[t] = 0.30 * z_s1_end + 0.70 * s2_pat
    else:
        # Shot 2: 固定机位微动（步进噪底远低于 Shot 1，拉低全片中位数 med1）
        z_asym[t] = s2_pat + 0.008 * (t - 18)
    if t % 5 == 0:
        z_asym[t] = z_asym[t] * 1.25 + 0.8
cuts_asym = detect_latent_cuts(z_asym)
mpins_asym = motion_pin_tokens(z_asym, s_bg=3, cuts=cuts_asym, flow_thr=0.75)
check("非对称快慢镜跨镜切点检测：精准检出 cuts=[17, 18]（绝不误报 16）且 motion_pins 完整覆盖 t=13,14",
      cuts_asym == [17, 18] and 16 not in cuts_asym and 13 in mpins_asym and 14 in mpins_asym,
      "cuts_asym=%s mpins_asym=%s" % (cuts_asym, mpins_asym))

# 2026-11-23 Iteration 4 回归 1：周期性百叶窗纹理防 ±4 格整周期误跳
torch.manual_seed(7)
H_p, W_p = 80, 112
xs_p = torch.arange(W_p, device=DEV, dtype=torch.float32)[None, None, :]
ys_p = torch.arange(H_p, device=DEV, dtype=torch.float32)[None, :, None]
bg_per = torch.randn(4, H_p, W_p, device=DEV) * 0.2 + torch.sin(xs_p * 0.25) * 0.1
# 在中央放一片垂直百叶窗周期条纹（周期=4格）
per_a = bg_per.clone()
per_a[:, 20:50, 30:70] = torch.sin(xs_p[:, :, 30:70] * (2.0 * 3.14159265 / 4.0)) * 2.0 + 0.05 * torch.randn(4, 30, 40, device=DEV)
per_b = shift_img(per_a, 0.0, 0.5)
# 在百叶窗中心加微小噪声让 +4.5 与 +0.5 几乎等分，但外围与粗尺度一致为 +0.5
per_b[:, 20:50, 30:70] += 0.02 * torch.randn(4, 30, 40, device=DEV)
fl_per = estimate_tile_flow(per_a, per_b, tiles=(10, 14), search=4)
max_jump = float((fl_per[..., 1] - 0.5).abs().max().item())
check("周期纹理防跳变（10x14 粗尺度+邻域共识）：百叶窗条纹绝不跳 ±4 格",
      max_jump < 0.6, "max_err=%.3f" % max_jump)

# 2026-11-23 Iteration 4 回归 2：motion_pin_tokens 前景隔离（静止背景+中心大幅移动人物不误加 pin）
torch.manual_seed(11)
z_static_bg_mov_fg = torch.zeros(17, 4, 48, 64, device=DEV)
bg_s2 = torch.randn(4, 48, 64, device=DEV) + torch.sin(torch.arange(64, device=DEV).float() * 0.5)[None, None, :] * 2.0
fg_s2 = torch.zeros(17, 48, 64, dtype=torch.bool, device=DEV)
fg_s2[:, 8:40, 16:48] = True  # 中心 33% 前景人物
fg_tex = torch.randn(4, 48, 64, device=DEV) * 3.0
for t in range(17):
    z_static_bg_mov_fg[t] = bg_s2.clone()
    fg_moved = shift_img(fg_tex, 0.0, float((t % 5) - 2) * 1.5)
    z_static_bg_mov_fg[t, :, 8:40, 16:48] = fg_moved[:, 8:40, 16:48]
mpins_fg_masked = motion_pin_tokens(z_static_bg_mov_fg, s_bg=3, cuts=[], flow_thr=0.75, fg_mask=fg_s2)
check("motion_pin_tokens 前景隔离：静止机位+中心大幅移动人物返回 0 冗余 pin，保持压缩率",
      mpins_fg_masked == [], "mpins=%s" % mpins_fg_masked)

# 2026-11-23 Iteration 4 回归 3：运镜下边缘 FG 残差 (+25) 绝不经 bicubic warp 渗漏进邻近 BG
z_mvfg = torch.zeros(6, 4, 32, 32, device=DEV)
pat_mv = torch.randn(4, 32, 32, device=DEV) + torch.sin(torch.arange(32, device=DEV).float() * 0.5)[None, None, :] * 3.0
for t in range(6):
    z_mvfg[t] = shift_img(pat_mv, 0.0, float(t) * 1.0)
fm_mvfg = torch.zeros(6, 32, 32, dtype=torch.bool, device=DEV)
fm_mvfg[:, :, :10] = True  # x=0..9 为 FG
d_mvfg = {}
for t in (0, 1, 4, 5):
    dt = torch.full((4, 32, 32), 1.0, device=DEV)
    dt[:, :, :10] = 25.0  # 强前景残差
    d_mvfg[t] = dt
rec_mvfg = reconstruct_moving(z_mvfg, d_mvfg, [0, 1, 4, 5], method="flow", tiles=(4, 4), search=4, fg_mask=fm_mvfg)
# 检查 t=2（被删 P-token）在紧邻前景外缘 x=10..12 的背景残差是否被 +25.0 污染
bg_edge_max = float((rec_mvfg[2, :, 4:-4, 10:14] - z_mvfg[2, :, 4:-4, 10:14] - 1.0).abs().max().item())
check("运镜流场防前景残差渗漏：FG(+25) 经 bicubic warp 绝不污染紧邻 BG 外缘 (x=10..13)",
      bg_edge_max < 0.15, "bg_edge_err=%.4f" % bg_edge_max)

print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
sys.exit(1 if FAIL else 0)
