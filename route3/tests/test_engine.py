# -*- coding: utf-8 -*-
"""Route 3 engine 自检：无压缩时必须与原生 forward 逐位一致

用法：D:/comfyui/comfyenv/python.exe route3/tests/test_engine.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, r"D:/comfyui/ComfyUI")

import comfy.ops
import comfy.model_management as mm
from comfy.ldm.minimax.model import MiniMaxH3Model
from route3.engine import splitstream_forward, fill_dropped

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    print("torch", torch.__version__, "device", DEV)
    torch.manual_seed(0)
    m = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1,
                       num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                       text_dim=64, timestep_input_dim=256, time_embed_hidden_size=256,
                       time_embed_dim=128, dtype=torch.float32, device=DEV,
                       operations=comfy.ops.disable_weight_init).to(DEV).eval()
    # disable_weight_init 分配出的参数在全新 CUDA 页上全为 0，必须显式填充非零权重，否则是在比 0==0！
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 1 and ("norm" in name or "scale" in name):
                p.fill_(1.0)
                p.add_(torch.randn_like(p) * 0.05)
            else:
                p.normal_(0.0, 0.05)
    # ComfyUI 实际加载的权重是 requires_grad=False；带 grad 时融合 RoPE 直接拒绝执行
    m.requires_grad_(False)
    mm.in_training = False

    T, H, W = 6, 8, 8
    x = torch.randn(1, 24, T, H, W, device=DEV)
    a = torch.randn(1, 32, 2, 20, device=DEV)
    ctx = torch.randn(1, 7, 64, device=DEV)
    ts = torch.tensor([500.0], device=DEV)

    # 人物掩码：中间 1/2 高 x 1/2 宽
    fm = torch.zeros(T, H // 2, W // 2, dtype=torch.bool, device=DEV)
    fm[:, H // 8: H // 8 + H // 4, W // 8: W // 8 + W // 4] = True
    fm = fm.reshape(T, -1)

    with torch.no_grad():
        ref = m([x.clone(), a.clone()], ts, ctx, {}, minimax_payload={})
        # s_bg=1：每帧都是锚点 → keep 是按 FG/BG 重排的置换（非恒等），但一行不删。
        # 注意力对行序等变，输出必须仍然逐位一致。
        got, plan, _ = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=1, beta=None,
            fg_mask=fm, return_plan=True)

    d0 = (ref[0] - got[0]).abs().max().item()
    d1 = (ref[1] - got[1]).abs().max().item()
    ref_mag = ref[0].abs().max().item()
    check("非零权重基线有效 (ref_max > 0.1)", ref_mag > 0.1, "ref_max=%.3e" % ref_mag)
    check("s_bg=1 + 非平凡重排：视频与原生 forward 逐位一致", d0 < 1e-5, "max_diff=%.2e" % d0)
    check("s_bg=1 + 非平凡重排：音频一致", d1 < 1e-5, "max_diff=%.2e" % d1)
    check("s_bg=1 一行不删（fg+bg == 全部，且 bg>0）",
          int(plan.fg_local.numel() + plan.bg_local.numel()) == plan.video_rows
          and int(plan.bg_local.numel()) > 0,
          "bg=%d fg=%d rows=%d" % (plan.bg_local.numel(), plan.fg_local.numel(), plan.video_rows))

    L = len(m.blocks)
    beta = torch.full((L, getattr(m, "heads", 2), 5), 0.5, device=DEV)
    with torch.no_grad():
        got2, plan2, _ = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=3, beta=beta,
            fg_mask=fm, return_plan=True)
    check("s_bg=3 输出有限且形状正确",
          torch.isfinite(got2[0]).all().item() and got2[0].shape == x.shape,
          "plan=%s" % plan2.to_dict())
    kept = plan2.fg_local.numel() + plan2.bg_local.numel()
    check("s_bg=3 确实降了 token", kept < plan2.video_rows,
          "保持 %d/%d，降幅 %.0f%%" % (kept, plan2.video_rows,
                                      100 * (1 - kept / plan2.video_rows)))
    check("相位长度对齐压缩后序列",
          plan2.full_phase().numel() == plan2.prefix_end + int(plan2.fg_local.numel()) + int(plan2.bg_local.numel()),
          "S=%d" % plan2.full_phase().numel())

    # 验证 full_velocity=True 在 Flow Matching 下通过 x0_hat 插值完全消去被删帧独立噪声 ε(t)，
    # 且严格遵守 I/P 相位隔离（P-token 绝不混入 p=0 I-token！）
    with torch.no_grad():
        sigma_test = 0.5
        ts_half = torch.tensor([sigma_test * 1000.0], device=DEV)
        T_ip = 10
        x_clean = torch.zeros(1, 24, T_ip, H, W, device=DEV)
        eps_indep = torch.randn_like(x_clean) * 2.0
        x_noisy = (1.0 - sigma_test) * x_clean + sigma_test * eps_indep
        fm_ip = torch.zeros(T_ip, (H // 2) * (W // 2), dtype=torch.bool, device=DEV)
        fm_ip[:, -4:] = True
        v_out = splitstream_forward(
            m, [x_noisy.clone(), a.clone()], ts_half, ctx, {}, {}, s_bg=3, beta=None,
            fg_mask=fm_ip, full_velocity=True,
        )[0]
        # s_bg=3 相位感知锚点（T=10）：[0, 1, 4, 5, 8, 9]（I-锚=0,5；P-锚=1,4,8,9）
        # 被删 P-token t=6 必须严格等于 P-锚 t=4 与 t=8 的同池线性插值（绝不混入 I-锚 t=5！）
        x0_hat = x_noisy - sigma_test * v_out
        expected_t6_bg = 0.5 * x0_hat[:, :, 4, :2, :2] + 0.5 * x0_hat[:, :, 8, :2, :2]
        err_x0 = (x0_hat[:, :, 6, :2, :2] - expected_t6_bg).abs().max().item()
    check("full_velocity 按 x0_hat 同池(P↔P)插值完全消去 ε(t) 且绝不混入 I-token", err_x0 < 1e-5,
          "err_x0=%.2e" % err_x0)

    # 修复 A + B1 回归：
    #   A: P 池首帧 t=1 被删（prev < 0），必须严格外推取右侧 P-锚 t=2，绝不读全 0 空行（消灭 f001..f004 紫边环）
    #   B1: 运动前景手臂占据 t=7 某 patch（+50.0），传入 source_mask_2d=仅纯 BG 锚点，
    #       绝不能污染被删 BG 帧 t=1,3,6,8（全部严格等于 1.0，0% 混入 +50.0）！
    with torch.no_grad():
        T_mv, P_mv = 10, 2
        v_mv = torch.zeros(T_mv, P_mv, 4, device=DEV)
        km_mv = torch.zeros(T_mv, P_mv, dtype=torch.bool, device=DEV)
        bg_mv = torch.zeros(T_mv, P_mv, dtype=torch.bool, device=DEV)
        for ta in (0, 2, 5, 7, 9):
            km_mv[ta, :] = True
            bg_mv[ta, :] = True
            v_mv[ta, :, :] = 10.0 if (ta % 5 == 0) else 1.0
        # t=7 的 p=0 被运动前景手臂占据：仍在 keep 中，但不是纯 BG 源！
        bg_mv[7, 0] = False
        v_mv[7, 0, :] = 50.0
        filled_mv = fill_dropped(v_mv.reshape(T_mv * P_mv, 4), km_mv,
                                 phase_split=True, source_mask_2d=bg_mv).view(T_mv, P_mv, 4)
        err_t1_lead = (filled_mv[1, 1] - 1.0).abs().max().item()
        err_mv_p = max(
            (filled_mv[1, 0] - 1.0).abs().max().item(),
            (filled_mv[3, 0] - 1.0).abs().max().item(),
            (filled_mv[6, 0] - 1.0).abs().max().item(),
            (filled_mv[8, 0] - 1.0).abs().max().item(),
        )
        err_mv_fg = (filled_mv[7, 0] - 50.0).abs().max().item()
    check("P 池首帧被删 (t=1, prev<0) 严格取右侧锚点 t=2 绝不读 0 行", err_t1_lead < 1e-5,
          "err_t1=%.2e" % err_t1_lead)
    check("source_mask_2d 纯 BG 隔离：运动前景手臂 (t=7) 绝不泄漏到被删 BG 帧 (t=1,3,6,8)",
          err_mv_p < 1e-5 and err_mv_fg < 1e-5,
          "bg_leak=%.2e fg_keep_err=%.2e" % (err_mv_p, err_mv_fg))

    # 复现 2026-11-23 exp_4v10a_00007 截图白方块事故：
    # 镜头1 (t=0..9) 列 p=0 是亮白窗户 (+100.0, 纯 BG)；硬切 t=10 后，镜头2 (t=10..19) 列 p=0 是人物蓝发 (-5.0, FG)，
    # 仅 t=13 偶发 1 帧掩码空洞。fill_dropped 必须严格按硬切断链 + max_src_gap 封顶，绝不能把镜头1的 +100.0 白窗贴到 t=13！
    with torch.no_grad():
        T_c, P_c = 20, 2
        v_c = torch.zeros(T_c, P_c, 4, device=DEV)
        km_c = torch.zeros(T_c, P_c, dtype=torch.bool, device=DEV)
        bg_c = torch.zeros(T_c, P_c, dtype=torch.bool, device=DEV)
        for ta in (0, 1, 4, 5, 8, 9):
            km_c[ta, :] = True; bg_c[ta, :] = True; v_c[ta, :, :] = 100.0
        for t2 in range(10, 20):
            if t2 in (10, 12, 15, 16, 19):
                km_c[t2, 1] = True; bg_c[t2, 1] = True; v_c[t2, 1, :] = 1.0
            if t2 != 13:
                km_c[t2, 0] = True; bg_c[t2, 0] = False; v_c[t2, 0, :] = -5.0
        f_cut_explicit = fill_dropped(v_c.reshape(T_c * P_c, 4), km_c, source_mask_2d=bg_c, cuts=[10]).view(T_c, P_c, 4)
        f_cut_auto = fill_dropped(v_c.reshape(T_c * P_c, 4), km_c, source_mask_2d=bg_c, cuts=None).view(T_c, P_c, 4)
        err_c1 = (f_cut_explicit[13, 0] - (-5.0)).abs().max().item()
        err_c2 = (f_cut_auto[13, 0] - (-5.0)).abs().max().item()
    check("fill_dropped 跨硬切断链：镜头1白窗 (+100) 绝不跨镜传送到镜头2发丝空洞 (t=13)",
          err_c1 < 1e-5 and err_c2 < 1e-5,
          "explicit_err=%.2e auto_err=%.2e" % (err_c1, err_c2))

    # 验证 beta_scale 与 beta_max 精确生效：
    # (a) beta=2.0, beta_scale=0.5, beta_max=0.0  ===  beta=1.0
    # (b) beta=2.0, beta_scale=1.0, beta_max=0.75 ===  beta=0.75
    with torch.no_grad():
        g_scale = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=3,
            beta=2.0, fg_mask=fm, beta_scale=0.5, beta_max=0.0)[0]
        g_ref10 = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=3,
            beta=1.0, fg_mask=fm)[0]
        d_scale = (g_scale - g_ref10).abs().max().item()

        g_clamp = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=3,
            beta=torch.full((L, getattr(m, "heads", 2), 5), 2.0, device=DEV),
            fg_mask=fm, beta_scale=1.0, beta_max=0.75)[0]
        g_ref75 = splitstream_forward(
            m, [x.clone(), a.clone()], ts, ctx, {}, {}, s_bg=3,
            beta=torch.full((L, getattr(m, "heads", 2), 5), 0.75, device=DEV),
            fg_mask=fm)[0]
        d_clamp = (g_clamp - g_ref75).abs().max().item()
    check("beta_scale=0.5 精确缩放标量 beta", d_scale < 1e-5, "max_diff=%.2e" % d_scale)
    check("beta_max=0.75 精确钳位 3D 张量 beta", d_clamp < 1e-5, "max_diff=%.2e" % d_clamp)

    # Pass-1 引导的流场残差轨迹回归：推镜移动背景时，传入 route3_p1_up 必须保住被删帧的真实空间位置
    # （旧版 fill_dropped(x0_full) 丢掉 z_p1_up 会把中间帧拉成 5px 双曝光重影）
    with torch.no_grad():
        T_p1, H_p1, W_p1 = 10, 16, 16
        base_img = torch.zeros(1, 24, 1, H_p1, W_p1, device=DEV)
        base_img[:, :, :, 6:10, 2:6] = 5.0  # 鲜明背景物体
        p1_seq = torch.cat([torch.roll(base_img, shifts=t, dims=-1) for t in range(T_p1)], dim=2)
        eps_p1 = torch.randn_like(p1_seq) * 0.2
        sig_p1 = 0.5
        x_p1 = (1.0 - sig_p1) * p1_seq + sig_p1 * eps_p1
        fm_p1 = torch.zeros(T_p1, (H_p1 // 2) * (W_p1 // 2), dtype=torch.bool, device=DEV)
        fm_p1[:, 0] = True   # 极小 FG，确保 BG 被压缩
        fcache = {}
        v_p1 = splitstream_forward(
            m, [x_p1.clone(), a.clone()], torch.tensor([sig_p1 * 1000.0], device=DEV), ctx,
            {"route3_p1_up": p1_seq, "route3_flow_cache": fcache, "route3_cuts": []}, {},
            s_bg=3, beta=None, fg_mask=fm_p1, full_velocity=True,
        )[0]
        v_raw = splitstream_forward(
            m, [x_p1.clone(), a.clone()], torch.tensor([sig_p1 * 1000.0], device=DEV), ctx,
            {"route3_cuts": []}, {},
            s_bg=3, beta=None, fg_mask=fm_p1, full_velocity=True,
        )[0]
        x0_p1 = x_p1 - sig_p1 * v_p1
        x0_raw = x_p1 - sig_p1 * v_raw
        # 被删 P-token t=6（介于 P-锚 t=4 与 t=8 之间）：物体真值在 x=[8:12]，
        # 旧版 lerp 会把物体拆成 x=[6:10] 与 x=[10:14] 的双曝光，新版必须紧贴 p1_seq[6]！
        err_p1 = (x0_p1[:, :, 6, 6:10, 8:12] - p1_seq[:, :, 6, 6:10, 8:12]).abs().mean().item()
        err_raw = (x0_raw[:, :, 6, 6:10, 8:12] - p1_seq[:, :, 6, 6:10, 8:12]).abs().mean().item()
    check("Pass-1 引导流场残差轨迹：推镜被删帧 (t=6) 精准锁住真实运镜位置且缓存光流",
          err_p1 < err_raw * 0.5 and len(fcache) > 0,
          "err_p1=%.3f err_raw=%.3f cache_entries=%d" % (err_p1, err_raw, len(fcache)))

    print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
