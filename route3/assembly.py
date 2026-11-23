# -*- coding: utf-8 -*-
"""Route 3 · 中间帧背景还原（法 A / 法 B v3）

任务书要求 Stage 1 做 A/B：

  法 A（线性插值）    z_bg(j) = z_p1_up(j) + lerp(δ(k0), δ(k1), frac)
  法 B（24通道多门控双向目标锚定 warp + 局部高频能量守恒，主推）
                      先用全 24 通道 latent 估双向分块位移（含短基线 t0->j / t1->j 直达配准），
                      再用「纹理能量门控 + 零位移裕度 + 前后向一致性 + 局部高斯邻域流场修补」
                      拦截低纹理漂移（如湿地面误贴红砖）同时杜绝纹理区跨块撕裂，
                      可信移动块用 bicubic 双向 warp + 局部高频偶极能量恢复保持边缘体积与锐度，
                      低纹理或静止块退回法 A 零重采样软化。

两者都保留 —— 法 A 当对照组。静机位档（s_bg=0 / T_bg=1）则只用于纯固定机位/绿幕，
把二采重绘好的第 0 帧高清背景铺满背景区。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["token_frame_center", "estimate_tile_flow", "flow_to_grid", "warp",
           "lerp_residual", "flow_residual", "detect_latent_cuts", "cut_pin_tokens",
           "motion_pin_tokens", "reconstruct_static", "reconstruct_moving"]


def _as_chw(x):
    return x.unsqueeze(0) if x.dim() == 2 else x


def token_frame_center(t: int) -> float:
    """返回 H3 latent token t 在 24fps 像素帧时间轴上的精确中心坐标。

    H3 3D Video VAE: FRAME_PER_TOKEN = (1, 4, 4, 4, 4)，每 5 token = 17 帧：
      * p = 0 (t % 5 == 0)：1 帧独立 I-token，坐落在第 17*g 帧（中心 17*g + 0.0）
      * p = 1..4          ：4 帧因果 P-token，覆盖 [17*g + 1 + 4*(p-1) .. 17*g + 4*p]（中心 17*g + 2.5 + 4*(p-1)）
    使用真实帧中心而非 token 下标 t，可彻底消除跨 I-token 的 P 锚区间上 1.5 帧（~1.8px）的系统性相位超前/滞后！
    """
    ti = int(t)
    g = ti // 5
    p = ti % 5
    if p == 0:
        return float(g * 17)
    return float(g * 17 + 2.5 + 4 * (p - 1))


@torch.no_grad()
def _match_tiles(A, B, tiles, search, zero_margin=0.002):
    """全通道（如 24ch latent）向量化分块归一化互相关匹配 + 抛物线亚像素精修（~2ms/对）。

    A, B: [C, H, W] float
    返回 (flow [Th, Tw, 2], score [Th, Tw], tex_std [Th, Tw])
    """
    C, H, W = A.shape
    th = max(1, min(int(tiles[0]), H))
    tw = max(1, min(int(tiles[1]), W))
    bh, bw = max(1, H // th), max(1, W // tw)
    h_use, w_use = th * bh, tw * bw
    s = int(search)
    K = 2 * s + 1

    # [th, tw, C, bh, bw]
    pa = A[:, :h_use, :w_use].view(C, th, bh, tw, bw).permute(1, 3, 0, 2, 4).contiguous()
    pa_c = pa - pa.mean(dim=(-2, -1), keepdim=True)
    tex = pa_c.pow(2).mean(dim=(2, 3, 4)).sqrt()
    na = pa_c.reshape(th, tw, -1).norm(dim=-1) + 1e-6

    Bp = F.pad(B.unsqueeze(0), (s, s, s, s), mode="replicate")[0]
    scores = torch.empty(K, K, th, tw, device=A.device, dtype=A.dtype)
    for iy, dy in enumerate(range(-s, s + 1)):
        y0 = dy + s
        for ix, dx in enumerate(range(-s, s + 1)):
            x0 = dx + s
            pb = Bp[:, y0:y0 + h_use, x0:x0 + w_use].view(C, th, bh, tw, bw).permute(1, 3, 0, 2, 4)
            pb_c = pb - pb.mean(dim=(-2, -1), keepdim=True)
            nb = pb_c.reshape(th, tw, -1).norm(dim=-1) + 1e-6
            scores[iy, ix] = (pa_c * pb_c).sum(dim=(2, 3, 4)) / (na * nb)

    sc0 = scores[s, s]                                    # [th, tw] (0, 0) 零位移得分
    flat_idx = scores.reshape(K * K, th, tw).argmax(dim=0)
    best_iy = flat_idx // K                               # [th, tw]
    best_ix = flat_idx % K
    best_sc = scores.reshape(K * K, th, tw).gather(0, flat_idx.unsqueeze(0)).squeeze(0)

    # 零位移裕度门控：非零位移必须明显优于零位移才采用，杜绝平坦区乱跳
    use_zero = ((best_iy != s) | (best_ix != s)) & ((best_sc - sc0) < float(zero_margin))
    best_iy = torch.where(use_zero, torch.full_like(best_iy, s), best_iy)
    best_ix = torch.where(use_zero, torch.full_like(best_ix, s), best_ix)
    best_sc = torch.where(use_zero, sc0, best_sc)

    # 抛物线亚像素精修（包含 0.15~0.49 latent 格 = 2.4~7.8 像素的推镜小位移！用 denom < -1e-4 防平坦区噪点除零）
    dy_f = (best_iy - s).to(A.dtype)
    dx_f = (best_ix - s).to(A.dtype)
    if s >= 1:
        ti_g = torch.arange(th, device=A.device)[:, None].expand(th, tw)
        tj_g = torch.arange(tw, device=A.device)[None, :].expand(th, tw)
        im_y = (best_iy - 1).clamp(0, K - 1)
        ip_y = (best_iy + 1).clamp(0, K - 1)
        sy_m = scores[im_y, best_ix, ti_g, tj_g]
        sy_p = scores[ip_y, best_ix, ti_g, tj_g]
        denom_y = (sy_m - 2.0 * best_sc + sy_p)
        ok_y = (best_iy > 0) & (best_iy < K - 1) & (denom_y < -1e-4) & (~use_zero) & (best_sc >= 0.45)
        sub_y = (0.5 * (sy_m - sy_p) / denom_y.clamp(max=-1e-4)).clamp(-0.5, 0.5)
        dy_f = dy_f + torch.where(ok_y, sub_y, torch.zeros_like(sub_y))

        im_x = (best_ix - 1).clamp(0, K - 1)
        ip_x = (best_ix + 1).clamp(0, K - 1)
        sx_m = scores[best_iy, im_x, ti_g, tj_g]
        sx_p = scores[best_iy, ip_x, ti_g, tj_g]
        denom_x = (sx_m - 2.0 * best_sc + sx_p)
        ok_x = (best_ix > 0) & (best_ix < K - 1) & (denom_x < -1e-4) & (~use_zero) & (best_sc >= 0.45)
        sub_x = (0.5 * (sx_m - sx_p) / denom_x.clamp(max=-1e-4)).clamp(-0.5, 0.5)
        dx_f = dx_f + torch.where(ok_x, sub_x, torch.zeros_like(sub_x))

    flow = torch.stack([dy_f, dx_f], dim=-1)
    return flow, best_sc.clamp(min=0.0), tex


@torch.no_grad()
def estimate_tile_flow(a, b, tiles=(6, 8), search=4, return_conf=False,
                       fb_check=True, fb_thr=1.35, tex_ratio=0.25, zero_margin=0.002):
    """估计把 a 对齐到 b 的稳健分块位移（全通道 + 纹理门控 + 前后向一致性 + 局部高斯邻域修补）。

    a, b : [C,H,W] 或 [H,W]
    返回  flow [Th,Tw,2]（或 return_conf=True 时返回 (flow, conf [Th,Tw])）

    关键区分（2026-11-23 沙发角与格子毛巾体积呼吸修复）：
      1) 真正极低纹理平坦区（~tex_ok，如平坦湿地面）：conf 严格置 0，让 flow_residual 退回固定坐标 lerp，
         绝不把紧邻红砖墙的残差贴到湿地面上；
      2) 有纹理区（tex_ok == True，如推镜中的沙发角/格子毛巾）因跨相位或遮挡边界偶发未过 fb_ok / sc_ab 时：
         不再粗暴把 conf 置 0 退回零速 lerp（那会导致同一个毛巾左半块走 0 速 lerp、右半块走 -15px warp 而发生跨块手风琴挤压！），
         而是用周围可信邻块的局部高斯加权流场平滑修补（保留径向推镜视差梯度）并继承邻域置信度！
    """
    A, B = _as_chw(a).float(), _as_chw(b).float()
    H, W = A.shape[-2:]
    f_ab, sc_ab, tex_a = _match_tiles(A, B, tiles, search, zero_margin=zero_margin)
    th, tw = f_ab.shape[:2]
    bh, bw = max(1, H // th), max(1, W // tw)

    # 1) 纹理能量门控：低于本帧中位纹理 25% 的极平坦块标记为不可信（湿地面保护）
    tex_med = float(tex_a.median())
    tex_ok = tex_a >= max(1e-4, float(tex_ratio) * tex_med)

    # 2) 前后向一致性门控：若 a->b 误跳到红砖墙，则 b->a 会停在红砖墙自身，环路误差 >> 1 格
    fb_ok = torch.ones(th, tw, dtype=torch.bool, device=A.device)
    if fb_check:
        f_ba, _, _ = _match_tiles(B, A, (th, tw), search, zero_margin=zero_margin)
        for i in range(th):
            for j in range(tw):
                dy = float(f_ab[i, j, 0])
                dx = float(f_ab[i, j, 1])
                ti = max(0, min(th - 1, int(round(i + dy / max(1.0, float(bh))))))
                tj = max(0, min(tw - 1, int(round(j + dx / max(1.0, float(bw))))))
                err_y = dy + float(f_ba[ti, tj, 0])
                err_x = dx + float(f_ba[ti, tj, 1])
                if (err_y * err_y + err_x * err_x) ** 0.5 > float(fb_thr):
                    fb_ok[i, j] = False

    valid = tex_ok & fb_ok & (sc_ab >= 0.48)

    # 3) 空间一致性 + 粗尺度共识防跳门控：
    # 拦截百叶窗/木地板/格子布等周期性重复纹理上的 ±4 格整周期误匹配（它们能同时通过正反向一致性！）
    if th >= 3 and tw >= 3 and int(valid.sum()) >= 4:
        spatial_ok = torch.ones_like(valid)
        if min(th, tw) >= 8:
            c_th, c_tw = max(4, th // 2), max(5, tw // 2)
            fl_c = estimate_tile_flow(
                A, B, tiles=(c_th, c_tw), search=search, return_conf=False,
                fb_check=fb_check, fb_thr=fb_thr, tex_ratio=tex_ratio, zero_margin=zero_margin,
            )
            fl_c_up = F.interpolate(
                fl_c.permute(2, 0, 1).unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
            c_mag = float(fl_c_up.norm(dim=-1).median().item())
            c_thr = max(1.65, 1.15 * c_mag)
            spatial_ok = spatial_ok & ((f_ab - fl_c_up).norm(dim=-1) <= c_thr)
        glob_med = f_ab[valid].median(dim=0).values
        mag_med = float(f_ab[valid].norm(dim=-1).median().item())
        nb_thr = max(1.65, 1.25 * mag_med)
        for i in range(th):
            for j in range(tw):
                if not bool(valid[i, j]) or not bool(spatial_ok[i, j]):
                    continue
                i0, i1 = max(0, i - 1), min(th, i + 2)
                j0, j1 = max(0, j - 1), min(tw, j + 2)
                sub_v = valid[i0:i1, j0:j1]
                sub_f = f_ab[i0:i1, j0:j1][sub_v]
                ref_f = sub_f.median(dim=0).values if sub_f.shape[0] >= 3 else glob_med
                if float((f_ab[i, j] - ref_f).norm().item()) > nb_thr:
                    spatial_ok[i, j] = False
        if int((valid & spatial_ok).sum()) >= 2:
            valid = valid & spatial_ok

    conf = torch.where(valid, sc_ab, torch.zeros_like(sc_ab))

    # 被拒收块的位移修补：优先用局部高斯邻域可信块插值（保留推镜径向视差），回落全局中位位移
    flow = f_ab.clone()
    if bool((~valid).any()):
        if bool(valid.any()):
            med = f_ab[valid].median(dim=0).values
            med_sc = float(sc_ab[valid].median().item())
            yy, xx = torch.meshgrid(
                torch.arange(th, device=A.device, dtype=A.dtype),
                torch.arange(tw, device=A.device, dtype=A.dtype),
                indexing="ij",
            )
            w_base = valid.to(A.dtype) * sc_ab.clamp(min=0.1)
            for i in range(th):
                for j in range(tw):
                    if not bool(valid[i, j]):
                        dist2 = (yy - float(i)) ** 2 + (xx - float(j)) ** 2
                        w = w_base * torch.exp(-0.5 * dist2 / (1.6 ** 2))
                        w_sum = float(w.sum().item())
                        if w_sum > 1e-5:
                            flow[i, j, 0] = (w * f_ab[:, :, 0]).sum() / w_sum
                            flow[i, j, 1] = (w * f_ab[:, :, 1]).sum() / w_sum
                            nb_sc = float(((w * sc_ab).sum() / w_sum).item())
                        else:
                            flow[i, j] = med
                            nb_sc = med_sc
                        # 若该块自身具备充足纹理（tex_ok == True），赋予邻域继承置信度，防止纹理物体跨块断成 0 速 lerp！
                        if bool(tex_ok[i, j]):
                            conf[i, j] = nb_sc * 0.90
        else:
            flow.zero_()

    if return_conf:
        pos_c = conf[conf > 0]
        denom_c = float(pos_c.median().item()) if pos_c.numel() > 0 else float(conf.median().item())
        conf = conf / (denom_c + 1e-6)
    return (flow, conf) if return_conf else flow


def flow_to_grid(flow, H, W):
    """分块位移 → 双线性平滑上采样为 grid_sample 采样网格（[1,H,W,2]，已归一化到 [-1,1]），消除推镜块边界缝。

    warp 的语义：输出在位置 p 处取输入 p + flow(p) 的值（后向采样）。
    """
    ys = torch.arange(H, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(W, device=flow.device, dtype=flow.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    fl_up = F.interpolate(
        flow.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0]
    sy = gy + fl_up[0]
    sx = gx + fl_up[1]
    nx = (sx + 0.5) / W * 2 - 1
    ny = (sy + 0.5) / H * 2 - 1
    return torch.stack([nx, ny], dim=-1).unsqueeze(0)


def warp(x, flow, mode="bilinear"):
    """x: [C,H,W] → 按 flow 重采样。位移是「a 在 b 里的位置」，所以采样用负号。"""
    C, H, W = x.shape
    g = flow_to_grid(-flow, H, W)
    return F.grid_sample(x.unsqueeze(0), g, mode=mode, align_corners=False,
                         padding_mode="border").squeeze(0)


def lerp_residual(da, db, frac):
    """法 A：固定坐标线性插值。"""
    return (1.0 - frac) * da + frac * db


def _restore_local_hf_energy(out, wa, wb, da, db, frac, max_gain=1.30):
    """补偿被删帧在 bicubic 亚像素重采样 + 双帧加权混合中的高频边缘偶极衰减（~25%~30%）。

    在推镜场景下，二采残差 δ = x0_hat - z_p1_up 包含用于雕塑 1.35MP 沙发扶手厚度与毛巾褶皱弧度的
    高频边缘偶极信号。若被删帧的高频振幅因双帧平均而衰减 30%，被删帧就会向一采粗体积回弹 30%，
    形成「锚点帧紧致 ↔ 被删帧膨胀/塌缩」的体积呼吸！
    通过匹配重采样前的局部高频包络能量（增益严格钳位在 [1.0, max_gain]），保持全序列边缘偶极强度恒定。
    """
    if out.ndim != 3 or min(out.shape[-2], out.shape[-1]) < 5:
        return out
    low_out = F.avg_pool2d(out.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    hf_out = out - low_out
    low_a = F.avg_pool2d(da.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    low_b = F.avg_pool2d(db.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    hf_a_energy = F.avg_pool2d((da - low_a).pow(2).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].sqrt()
    hf_b_energy = F.avg_pool2d((db - low_b).pow(2).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].sqrt()
    # 同时考虑对齐后 wa/wb 的局部高频能量上限，防止在未对齐处过冲
    low_wa = F.avg_pool2d(wa.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    low_wb = F.avg_pool2d(wb.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    hf_wa_e = F.avg_pool2d((wa - low_wa).pow(2).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].sqrt()
    hf_wb_e = F.avg_pool2d((wb - low_wb).pow(2).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].sqrt()

    target_e = (1.0 - frac) * torch.minimum(hf_a_energy, hf_wa_e * 1.18) + frac * torch.minimum(hf_b_energy, hf_wb_e * 1.18)
    cur_e = F.avg_pool2d(hf_out.pow(2).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].sqrt()
    gain = ((target_e + 1e-4) / (cur_e + 1e-4)).clamp(1.0, float(max_gain))
    return low_out + hf_out * gain


def flow_residual(da, db, frac, flow_ab, conf=None, conf_thr=0.55, mode="bicubic",
                  return_warped=False, flow_a=None, flow_b=None, restore_hf=True):
    """法 B v3：双向 bicubic warp（支持直达目标帧短基线 flow_a / flow_b）+ 高频偶极能量守恒；
    低纹理（如湿地面）或几乎静止的 tile 自动退回 lerp 零软化。
    """
    lerp_out = lerp_residual(da, db, frac)
    fl_a = (flow_ab * frac) if flow_a is None else flow_a
    fl_b = (flow_ab * (frac - 1.0)) if flow_b is None else flow_b

    # 整帧几乎静止时直接走 lerp，零重采样高频衰减（0.15 latent 格 ≈ 2.4 像素，放行亚像素推镜）
    flow_mag = flow_ab.pow(2).sum(dim=-1).sqrt()
    if flow_a is not None and flow_b is not None and flow_a.shape == flow_ab.shape:
        flow_mag = torch.maximum(flow_mag, flow_a.norm(dim=-1) + flow_b.norm(dim=-1))
    if float(flow_mag.max()) < 0.15:
        return (lerp_out, da, db) if return_warped else lerp_out

    wa = warp(da, fl_a, mode=mode)
    wb = warp(db, fl_b, mode=mode)
    out = (1.0 - frac) * wa + frac * wb
    if restore_hf:
        out = _restore_local_hf_energy(out, wa, wb, da, db, frac)

    # tile 级门控：极低纹理平坦区（如湿地面 conf==0）保持绝对零泄漏，其余移动交界平滑过渡消除块状接缝
    H, W = da.shape[-2:]
    th, tw = flow_ab.shape[:2]
    ti = (torch.arange(H, device=flow_ab.device) * th / H).long().clamp(0, th - 1)
    tj = (torch.arange(W, device=flow_ab.device) * tw / W).long().clamp(0, tw - 1)
    moving_tile = flow_mag >= 0.15
    good_tile = moving_tile if conf is None else (moving_tile & (conf >= float(conf_thr)))
    tex_keep = (torch.ones(1, H, W, dtype=torch.bool, device=da.device)
                if conf is None else (conf > 0)[ti][:, tj].unsqueeze(0))
    good_soft = F.interpolate(
        good_tile.to(da.dtype)[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0].clamp(0.0, 1.0)
    good_w = torch.where(tex_keep, good_soft, torch.zeros_like(good_soft))
    blended = good_w * out + (1.0 - good_w) * lerp_out
    if return_warped:
        return blended, good_w * wa + (1.0 - good_w) * da, good_w * wb + (1.0 - good_w) * db
    return blended


def detect_latent_cuts(z_p1_up, cut_ratio=2.1, cut_floor=0.15, phase_aware=True):
    """在 latent 序列 [T, C, H, W] 上检测镜头硬切位置（返回新镜头的起始 token 下标列表）。

    关键修正（2026-11-23 exp_4v10a 24ch 真实潜空间跨镜 t=17 检出修复）：
      * H3 每 5 token 有 1 个独立 I-token (t%5==0)，与 P-token 分布不同；默认 phase_aware=True 只在纯 P-pool 内检测；
      * 24 通道高分辨率 latent 含有大量高频空间纹理与局部运镜差分（拉高像素级 L1 中位数 med1/med2，曾导致 task-1569 漏检 t=17）；
        当空间尺寸 >= 16 时先做 4×4 空间均值池化滤除局部高频运镜噪底，让全画幅切镜跳变信噪比提升 2.5 倍；
      * 2 步差分阈值 thr2 同时以 2.35*med1 与 1.65*med2 约束，精准捕获落在 4 帧 P-token 内部的半跨镜切点（如 t=17）。
    """
    T = int(z_p1_up.shape[0])
    if T < 4:
        return []
    zf = z_p1_up.float()
    if zf.ndim == 4 and min(int(zf.shape[-2]), int(zf.shape[-1])) >= 16:
        zf = F.avg_pool2d(zf, kernel_size=4, stride=4)
    z_std = float(zf.std().item())
    if z_std < 1e-6:
        return []
    eff_floor = float(cut_floor) * max(z_std, 1e-3)

    if phase_aware and T >= 6:
        p_idx = [t for t in range(T) if t % 5 != 0]
        if len(p_idx) >= 3:
            zp = zf[p_idx]
            # 同时结合空间均值与空间中位数（抑制单镜头内局部人物大幅摆头造成的假阳切镜，保留全画幅真切镜）
            diff1_hw = (zp[1:] - zp[:-1]).abs().mean(dim=1).flatten(1)
            d1_raw = 0.5 * diff1_hw.mean(dim=1) + 0.5 * diff1_hw.median(dim=1).values
            # 跨 I-token 的 P 对间距为 5 帧（非跨 I 为 4 帧），按 /1.25 折算回同时间步距防止误报
            step_scale = torch.tensor(
                [1.25 if (p_idx[k + 1] - p_idx[k]) > 1 else 1.0 for k in range(len(p_idx) - 1)],
                device=d1_raw.device, dtype=d1_raw.dtype,
            )
            d1 = d1_raw / step_scale
            med1 = float(d1.median().item())
            thr1 = max(float(cut_ratio) * max(med1, 1e-6), eff_floor)
            n_d1 = len(d1)
            cuts = set()
            for k in range(n_d1):
                l_s1 = d1[max(0, k - 4):k]
                r_s1 = d1[k + 1:min(n_d1, k + 5)]
                m_l1 = float(l_s1.median().item()) if l_s1.numel() > 0 else med1
                m_r1 = float(r_s1.median().item()) if r_s1.numel() > 0 else med1
                thr1_k = max(thr1, 1.65 * max(m_l1, m_r1))
                if float(d1[k].item()) > thr1_k:
                    a, b = p_idx[k], p_idx[k + 1]
                    if b - a == 1:
                        cuts.add(b)
                    else:
                        # 中间夹着 I-token ti = a + 1：优先用同池 I 距离判断 ti 属于哪一镜
                        ti = a + 1
                        if ti - 5 >= 0 and ti + 5 < T:
                            dl = float((zf[ti] - zf[ti - 5]).abs().mean().item())
                            dr = float((zf[ti + 5] - zf[ti]).abs().mean().item())
                            cuts.add(b if dl <= dr else ti)
                        elif ti - 5 >= 0:
                            dl = float((zf[ti] - zf[ti - 5]).abs().mean().item())
                            cuts.add(b if dl <= eff_floor else ti)
                        else:
                            dl = float((zf[ti] - zf[a]).abs().mean().item())
                            dr = float((zf[b] - zf[ti]).abs().mean().item())
                            cuts.add(b if dl <= dr else ti)

            # 2 步跨镜过渡检测：硬切落在 4 帧 token p_idx[k+1] 内部时，左右 d1 各吃一半但 d2 剧烈飙升
            if len(p_idx) >= 4:
                diff2_hw = (zp[2:] - zp[:-2]).abs().mean(dim=1).flatten(1)
                d2_raw = 0.5 * diff2_hw.mean(dim=1) + 0.5 * diff2_hw.median(dim=1).values
                step2_scale = torch.tensor(
                    [1.125 if (p_idx[k + 2] - p_idx[k]) > 2 else 1.0 for k in range(len(p_idx) - 2)],
                    device=d2_raw.device, dtype=d2_raw.dtype,
                )
                d2 = d2_raw / step2_scale
                med2 = float(d2.median().item())
                thr2 = max(
                    min(float(cut_ratio) * max(med2, 1e-6), 2.35 * max(med1, 1e-6)),
                    1.65 * max(med2, 1e-6),
                    1.25 * eff_floor,
                )
                sub_thr = max(1.35 * max(med1, 1e-6), 0.55 * eff_floor)
                # 计算每一对 (d1[k], d1[k+1]) 相对于左右各自镜头局部运动基线 (med_L, med_R) 的跳变倍数：
                # 防止前镜高速推镜 (Shot 1, d1≈0.16) + 后镜固定机位 (Shot 2, d1≈0.08) 时，
                # 前镜末步 d1[11] (14->16) 因高于全片中位数 med1 而与真切点 d1[12] (16->17) 误配对把 16 误判为切点
                n_d1 = len(d1)
                r2_scores = []
                loc_ok = []
                for k in range(len(d2)):
                    l_slice = d1[max(0, k - 4):k]
                    r_slice = d1[k + 2:min(n_d1, k + 6)]
                    med_l = float(l_slice.median().item()) if l_slice.numel() > 0 else med1
                    med_r = float(r_slice.median().item()) if r_slice.numel() > 0 else med1
                    v_l = float(d1[k].item())
                    v_r = float(d1[k + 1].item())
                    ratio_l = v_l / max(med_l, 1e-6)
                    ratio_r = v_r / max(med_r, 1e-6)
                    r2_scores.append(min(ratio_l, ratio_r))
                    loc_ok.append(
                        v_l > max(sub_thr, 1.45 * med_l)
                        and v_r > max(sub_thr, 1.45 * med_r)
                    )
                for k in range(len(d2)):
                    v2 = float(d2[k].item())
                    s_cur = r2_scores[k]
                    is_peak = (
                        (k == 0 or s_cur >= r2_scores[k - 1])
                        and (k + 1 >= len(d2) or s_cur >= r2_scores[k + 1])
                    )
                    if is_peak and v2 > thr2 and loc_ok[k]:
                        cuts.add(p_idx[k + 1])
                        cuts.add(p_idx[k + 2])
            return sorted(c for c in cuts if 0 < c < T)

    d = (zf[1:] - zf[:-1]).abs().mean(dim=(1, 2, 3))
    med = float(d.median().item())
    thr = max(float(cut_ratio) * max(med, 1e-6), eff_floor)
    return [int(i + 1) for i in range(T - 1) if float(d[i].item()) > thr]


def cut_pin_tokens(cuts, T: int, radius: int = 1) -> list[int]:
    """把每个硬切点及其 ±radius 邻近过渡 token 钉为全速率锚点，避免因果 3D VAE 组内跨镜污染。"""
    if not cuts or int(T) <= 0:
        return []
    r = max(0, int(radius))
    pins = set()
    for c in cuts:
        ci = int(c)
        for t in range(ci - r, ci + r + 1):
            if 0 <= t < int(T):
                pins.add(t)
    return sorted(pins)


@torch.no_grad()
def motion_pin_tokens(z_p1_up, s_bg: int = 3, cuts=None, flow_thr: float = 0.75, fg_mask=None) -> list[int]:
    """在高速运镜镜头段（如快速推拉摇移 Shot 1）自动加密双丢帧跨度的 P-锚点。

    物理机制（2026-11-23 沙发角与毛巾体积跳变根治 + 前景掩码隔离）：
      * 在 s_bg=3 下，常规 P-锚点间距为 3 个 P-token（跨 I-token 时达 13 像素帧，连续丢弃 2 个 4 帧 P-token = 8 帧！）；
      * 若传入 fg_mask，计算全画幅运镜中位数时严格剔除前景主体占比较高的 tile（fg_frac >= 0.35），
        防止固定机位特写镜头（如 Shot 2 中心人物大幅转头/端杯）误抬全画幅中位数而浪费压缩率；
      * 固定机位镜头（Shot 2 或绿幕，纯 BG 位移 < 0.75）返回 0 个额外锚点，保持满额压缩；
      * 中速运镜（0.75 <= med_mag < 0.8625）钉住中点 P-锚；
      * 高速推镜/横移镜头（med_mag >= 1.15 * flow_thr = 0.8625 格，实测 Shot 1 真实 24ch latent 达 ~0.95..1.85 格）全量钉住该跨度内全部 P-token，
        彻底消除近景三维沙发角与格子毛巾透视形变呼吸！
    """
    from .splitstream_attn import bg_anchor_indices

    T = int(z_p1_up.shape[0])
    if T < 5 or int(s_bg) <= 1 or int(s_bg) >= T:
        return []
    cut_list = sorted(set(int(c) for c in (cuts if cuts is not None else detect_latent_cuts(z_p1_up)) if 0 < int(c) < T))
    cut_pins = cut_pin_tokens(cut_list, T, radius=1)
    base_anchors = bg_anchor_indices(T, int(s_bg), phase_aware=True, pin_tokens=cut_pins)
    bounds = [0] + cut_list + [T]

    fg_lat = None
    if fg_mask is not None:
        fm = torch.as_tensor(fg_mask, device=z_p1_up.device, dtype=torch.float32)
        if fm.ndim == 2 and fm.shape[0] == T:
            H_z, W_z = int(z_p1_up.shape[-2]), int(z_p1_up.shape[-1])
            gh_f, gw_f = max(1, H_z // 2), max(1, W_z // 2)
            if fm.shape[1] == gh_f * gw_f:
                fm = fm.view(T, gh_f, gw_f).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        if fm.ndim == 3 and fm.shape[0] == T:
            if fm.shape[-2:] != z_p1_up.shape[-2:]:
                fm = F.interpolate(fm.unsqueeze(1), size=z_p1_up.shape[-2:], mode="nearest").squeeze(1)
            fg_lat = fm > 0.5

    high_thr = float(flow_thr) * 1.15
    extra_pins = set()
    for si in range(len(bounds) - 1):
        lo, hi = bounds[si], bounds[si + 1]
        p_anc = [t for t in base_anchors if lo <= t < hi and (t % 5 != 0)]
        if len(p_anc) < 2:
            continue
        for k in range(len(p_anc) - 1):
            t0, t1 = p_anc[k], p_anc[k + 1]
            p_mids = [j for j in range(t0 + 1, t1) if j % 5 != 0]
            if len(p_mids) < 2:
                continue
            fl = estimate_tile_flow(z_p1_up[t0], z_p1_up[t1], tiles=(6, 8), search=4)
            fl_mag = fl.norm(dim=-1)
            if fg_lat is not None:
                fg_pair = (fg_lat[t0] | fg_lat[t1]).float()
                fg_tile = F.adaptive_avg_pool2d(fg_pair[None, None], fl_mag.shape)[0, 0]
                bg_tiles = fg_tile < 0.35
                vals = fl_mag[bg_tiles] if int(bg_tiles.sum()) >= 6 else fl_mag.flatten()
            else:
                vals = fl_mag.flatten()
            med_mag = float(vals.median().item())
            if med_mag >= high_thr:
                for pm in p_mids:
                    extra_pins.add(int(pm))
            elif med_mag >= float(flow_thr):
                extra_pins.add(int(p_mids[(len(p_mids) - 1) // 2]))
    return sorted(extra_pins)


def reconstruct_moving(z_p1_up, delta_keys, anchor_idx, method="lerp",
                       tiles=(6, 8), search=4, cut_protect=True, phase_split=True,
                       fg_mask=None, cuts=None, flow_cache=None):
    """按关键帧残差还原全部帧的背景（自动按镜头硬切分段 + 严格分离 I/P token + 屏蔽前景残差污染）。

    z_p1_up    : [T,C,H,W] 一采上采样后的全帧底图（自带 24fps 运镜）
    delta_keys : {anchor_t: δ [C,H,W]}  二采在锚点帧真正画出来的高清细节增量
    anchor_idx : 锚点帧下标（升序）
    method     : "lerp"（默认主推，<1ms）| "flow"（法 B v3：短基线目标配准 + 高频偶极守恒）| "none"
    cut_protect: True = 自动检测镜头硬切并断开插值区间，杜绝跨镜叠影与方框污染
    phase_split: True = 严格分离 H3 的 1 帧 I-token (t%5==0) 与 4 帧 P-token (t%5!=0)
    fg_mask    : [T,H,W] bool 或 None（若提供，自动屏蔽锚点帧上被人物占据位置的残差 δ，
                 防止运动手臂/发丝在 t0/t1 的前景残差被贴到中间帧新露出的绿幕/背景边缘！）
    cuts       : 可选预设硬切点列表（非 None 时直接使用，避免重复检测）
    flow_cache : 可选字典 {(t0, t1): (flow, conf)}，跨去噪步复用 z_p1_up 位移场，零重复开销
    返回 [T,C,H,W]
    """
    T, C, H, W = z_p1_up.shape
    out = z_p1_up.clone()
    anchors_all = sorted(int(t) for t in anchor_idx if 0 <= int(t) < T)
    for t0 in anchors_all:
        out[t0] = out[t0] + delta_keys[t0]
    if method == "none" or not anchors_all:
        return out

    # 高分辨率潜空间自动加密分块网格（108x196 下用 10x14 细网格隔离沙发角与背景餐椅视差）
    eff_tiles = (10, 14) if (tuple(tiles) == (6, 8) and H >= 72 and W >= 120) else tiles

    # 构建每帧纯背景有效掩码（注意：不再在 fg_mask 外额外做 3x3 max_pool2d 膨胀！
    # 因为 fg_mask 本身已含 ring=1 保护环，额外膨胀会在 fg_mask 外缘制造一圈「既被 DiT 丢弃、又被 clean_bg 归零」
    # 的 16px 真空死区，曾导致 Shot 1 t=13..14 左侧木质沙发扶手外缘 δ 被归零、宽度塌缩 -4.2px！）
    clean_bg = None
    if fg_mask is not None:
        fm_f = fg_mask.to(device=z_p1_up.device, dtype=torch.float32).unsqueeze(1)
        if fm_f.shape[-2:] != (H, W):
            fm_f = F.interpolate(fm_f, size=(H, W), mode="nearest")
        clean_bg = ~(fm_f > 0.5)  # [T, 1, H, W]

    def _get_flow(ta, tb):
        key = (int(ta), int(tb), tuple(eff_tiles), int(search))
        if flow_cache is not None and key in flow_cache:
            fl_c, cf_c = flow_cache[key]
            return fl_c.to(z_p1_up.device), cf_c.to(z_p1_up.device)
        fl, cf = estimate_tile_flow(z_p1_up[ta], z_p1_up[tb], tiles=eff_tiles,
                                    search=search, return_conf=True)
        if flow_cache is not None:
            flow_cache[key] = (fl.detach().cpu(), cf.detach().cpu())
        return fl, cf

    def _warp_single_gated(delta, fl, conf, conf_thr=0.55):
        flow_mag = fl.pow(2).sum(dim=-1).sqrt()
        if float(flow_mag.max()) < 0.15:
            return delta
        warped = warp(delta, fl, mode="bicubic")
        H_d, W_d = delta.shape[-2:]
        th_d, tw_d = fl.shape[:2]
        ti = (torch.arange(H_d, device=fl.device) * th_d / H_d).long().clamp(0, th_d - 1)
        tj = (torch.arange(W_d, device=fl.device) * tw_d / W_d).long().clamp(0, tw_d - 1)
        good_tile = (flow_mag >= 0.15) & (conf >= float(conf_thr))
        return torch.where(good_tile[ti][:, tj].unsqueeze(0), warped, delta)

    def _prepare_clean_delta(d, t_anc):
        if clean_bg is None:
            return d
        cb = clean_bg[t_anc]  # [1, H, W]
        if bool(cb.all()):
            return d
        # 用 1 格邻域纯 BG 均值填补锚点 FG 边界格（避免 bicubic 4x4 核在 warp 时把前景残差抽进背景外缘）
        d_clean = torch.where(cb, d, torch.zeros_like(d))
        cb_f = cb.to(d.dtype).unsqueeze(0)
        num = F.avg_pool2d(d_clean.unsqueeze(0), kernel_size=3, stride=1, padding=1, divisor_override=1)[0]
        den = F.avg_pool2d(cb_f, kernel_size=3, stride=1, padding=1, divisor_override=1)[0]
        edge_fill = torch.where(den > 0.5, num / den.clamp(min=1.0), torch.zeros_like(d))
        return torch.where(cb, d, edge_fill)

    def _warped_clean(t_anc, fl=None):
        if clean_bg is None:
            return None
        cb = clean_bg[t_anc]
        if fl is None or float(fl.pow(2).sum(dim=-1).max().sqrt().item()) < 0.15:
            return cb
        return cb & (warp(cb.to(fl.dtype), fl, mode="bilinear") > 0.85)

    def _blend_pair_clean(d0_eff, d1_eff, frac, t0, t1, res_both, v0=None, v1=None):
        """若提供了 fg_mask：只从纯净背景锚点借残差（一侧被人物占了就 100% 取另一侧，两侧都占了取 0）。"""
        if clean_bg is None:
            return res_both
        v0 = clean_bg[t0] if v0 is None else v0   # [1, H, W]
        v1 = clean_bg[t1] if v1 is None else v1   # [1, H, W]
        zero = torch.zeros_like(res_both)
        # 优先双净 -> 仅左净 -> 仅右净 -> 双占归 0
        only0 = v0 & (~v1)
        only1 = v1 & (~v0)
        both = v0 & v1
        out_r = torch.where(both, res_both, zero)
        out_r = torch.where(only0, d0_eff, out_r)
        out_r = torch.where(only1, d1_eff, out_r)
        return out_r

    def _single_clean(d, t_anc, fl=None):
        if clean_bg is None:
            return d
        v = _warped_clean(t_anc, fl)
        return torch.where(v, d, torch.zeros_like(d))

    use_frame_time = bool(phase_split and T >= 5)

    def _time_coord(t_idx: int) -> float:
        return token_frame_center(t_idx) if use_frame_time else float(t_idx)

    def _fill_pool(target_frames, pool_anchors):
        """只用 pool_anchors 里的同类型锚点还原 target_frames。"""
        if not pool_anchors:
            return
        anc_set = set(pool_anchors)
        for k in range(len(pool_anchors) - 1):
            t0, t1 = pool_anchors[k], pool_anchors[k + 1]
            d0, d1 = delta_keys[t0], delta_keys[t1]
            mids = [j for j in target_frames if t0 < j < t1 and j not in anc_set]
            if not mids:
                continue
            tau0, tau1 = _time_coord(t0), _time_coord(t1)
            denom_tau = max(1e-5, float(tau1 - tau0))
            if method == "flow":
                d0_src = _prepare_clean_delta(d0, t0)
                d1_src = _prepare_clean_delta(d1, t1)
                fl_01, conf_01 = _get_flow(t0, t1)
                for j in mids:
                    frac = (_time_coord(j) - tau0) / denom_tau
                    # 直达目标帧 j 的短基线双向配准 (t0->j 与 t1->j) + 长基线约束 (t0->t1)：
                    # 消除长跨度线性匀速假设误差与跨 I-token 时间步长失配
                    fl_0j, cf_0j = _get_flow(t0, j)
                    fl_1j, cf_1j = _get_flow(t1, j)
                    fl_a = 0.75 * fl_0j + 0.25 * (fl_01 * frac)
                    fl_b = 0.75 * fl_1j + 0.25 * (fl_01 * (frac - 1.0))
                    conf_eff = torch.maximum(conf_01, 0.5 * (cf_0j + cf_1j))
                    r, d0_w, d1_w = flow_residual(
                        d0_src, d1_src, frac, fl_01, conf=conf_eff, return_warped=True,
                        flow_a=fl_a, flow_b=fl_b, restore_hf=True,
                    )
                    v0_w = _warped_clean(t0, fl_a)
                    v1_w = _warped_clean(t1, fl_b)
                    out[j] = out[j] + _blend_pair_clean(d0_w, d1_w, frac, t0, t1, r, v0=v0_w, v1=v1_w)
            else:
                for j in mids:
                    frac = (_time_coord(j) - tau0) / denom_tau
                    r = lerp_residual(d0, d1, frac)
                    out[j] = out[j] + _blend_pair_clean(d0, d1, frac, t0, t1, r)
        t_first = pool_anchors[0]
        for j in target_frames:
            if j < t_first and j not in anc_set:
                if method == "flow":
                    fl, conf = _get_flow(t_first, j)
                    d = _warp_single_gated(_prepare_clean_delta(delta_keys[t_first], t_first), fl, conf)
                    out[j] = out[j] + _single_clean(d, t_first, fl=fl)
                else:
                    d = delta_keys[t_first]
                    out[j] = out[j] + _single_clean(d, t_first)
        t_last = pool_anchors[-1]
        for j in target_frames:
            if j > t_last and j not in anc_set:
                if method == "flow":
                    fl, conf = _get_flow(t_last, j)
                    d = _warp_single_gated(_prepare_clean_delta(delta_keys[t_last], t_last), fl, conf)
                    out[j] = out[j] + _single_clean(d, t_last, fl=fl)
                else:
                    d = delta_keys[t_last]
                    out[j] = out[j] + _single_clean(d, t_last)

    if cuts is not None:
        cut_list = sorted(set(int(c) for c in cuts if 0 < int(c) < T))
    else:
        cut_list = detect_latent_cuts(z_p1_up) if cut_protect else []
    bounds = [0] + cut_list + [T]

    for seg_i in range(len(bounds) - 1):
        seg_lo, seg_hi = bounds[seg_i], bounds[seg_i + 1]
        anchors = [t for t in anchors_all if seg_lo <= t < seg_hi]
        if not anchors:
            continue
        i_anc = [t for t in anchors if t % 5 == 0]
        p_anc = [t for t in anchors if t % 5 != 0]
        if phase_split and (seg_hi - seg_lo) >= 5 and i_anc and p_anc:
            i_frames = [t for t in range(seg_lo, seg_hi) if t % 5 == 0]
            p_frames = [t for t in range(seg_lo, seg_hi) if t % 5 != 0]
            _fill_pool(i_frames, i_anc)
            _fill_pool(p_frames, p_anc)
        else:
            _fill_pool(list(range(seg_lo, seg_hi)), anchors)
    return out


def reconstruct_static(z_p1_up, bg_hd, fg_mask=None):
    """静机位专享档（s_bg=0 / T_bg=1）：单帧二采高清重绘 + 背景静态锁屏（非动态背景重绘）。

    bg_hd   : [C,H,W] 已重绘好的高清背景
    fg_mask : [T,H,W] bool 或 None
    """
    T, C, H, W = z_p1_up.shape
    out = z_p1_up.clone()
    for t in range(T):
        if fg_mask is None:
            out[t] = bg_hd
        else:
            m = fg_mask[t].unsqueeze(0)
            out[t] = torch.where(m, z_p1_up[t], bg_hd)
    return out
