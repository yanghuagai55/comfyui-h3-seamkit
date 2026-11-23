# -*- coding: utf-8 -*-
"""Route 3 数学等价性 + 可训练性单测（本机可跑，不需要 A100）

用法：
    D:/comfyui/comfyenv/python.exe route3/tests/test_splitstream_math.py
    ... --with-model   # 额外跑 tiny MiniMaxH3Model 的 backward / LoRA 梯度
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, r"D:/comfyui/ComfyUI")

from route3.splitstream_attn import (merge_lse, splitstream_attention,
                                     video_split, route3_plan, attn_lse)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-3
FAIL = []


def check(name, ok, detail=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


# --------------------------------------------------------------------------- #
def t_rope_channel_map():
    """3D RoPE：t/h/w 各占 16 个角频率，配对步距 48，后 32 维 NoPE。"""
    import comfy.model_management  # noqa: F401
    from comfy.ldm.minimax.model import MiniMaxH3Model
    m = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(m)
    m.rope = torch.nn.Module()
    m.rope.register_buffer("inv_freq",
                           torch.exp(-math.log(10000.0) * torch.arange(16, dtype=torch.float32) / 16))
    base = torch.zeros(1, 3, dtype=torch.float64)
    t1 = base.clone(); t1[0, 0] = 7.0
    h1 = base.clone(); h1[0, 1] = 7.0
    w1 = base.clone(); w1[0, 2] = 7.0
    r0 = MiniMaxH3Model.rope_freqs(m, base, "cpu")[0]
    rt = MiniMaxH3Model.rope_freqs(m, t1, "cpu")[0]
    rh = MiniMaxH3Model.rope_freqs(m, h1, "cpu")[0]
    rw = MiniMaxH3Model.rope_freqs(m, w1, "cpu")[0]
    ok = (torch.allclose(r0, torch.zeros_like(r0))
          and rt[:16].abs().sum() > 0 and rt[16:48].abs().sum() == 0
          and rh[16:32].abs().sum() > 0 and rh[:16].abs().sum() == 0
          and rw[32:48].abs().sum() > 0 and rw[:32].abs().sum() == 0
          and torch.allclose(rt[:16], rt[48:64])
          and r0[96:].numel() == 0)
    check("3D RoPE 通道切分 (t=0..15/h=16..31/w=32..47，48 步距复制)", ok)


def t_merge_exact():
    torch.manual_seed(0)
    b, h, s, k, d = 1, 4, 128, 320, 64
    # 任务书写 "误差 < 1e-3"，但 bf16 只有 8 位尾数，两条不同计算路径天然差 ~2e-3。
    # 这条门禁只能用在 fp32；bf16 必须放宽到 5e-3，否则会误判成算法错、白烧卡。
    for dt, tol in ((torch.float32, 1e-5), (torch.bfloat16, 5e-3)):
        q = torch.randn(b, h, s, d, device=DEV, dtype=dt)
        kk = torch.randn(b, h, k, d, device=DEV, dtype=dt)
        v = torch.randn(b, h, k, d, device=DEV, dtype=dt)
        ref = sdpa(q, kk, v)
        cut = 200
        o1 = sdpa(q, kk[:, :, :cut], v[:, :, :cut])
        o2 = sdpa(q, kk[:, :, cut:], v[:, :, cut:])
        l1 = torch.logsumexp(((q.float() @ kk[:, :, :cut].float().transpose(-1, -2))
                              / math.sqrt(d)), -1)
        l2 = torch.logsumexp(((q.float() @ kk[:, :, cut:].float().transpose(-1, -2))
                              / math.sqrt(d)), -1)
        got = merge_lse(o1, l1, o2, l2, 0.0)
        err = (got.float() - ref.float()).abs().max().item()
        check("T1 无偏置两分支合并 == 整段 SDPA (%s)" % dt, err < tol, "max_err=%.2e" % err)


def t_merge_tiled():
    """BG 键复制 S 次 == 加 log(S) 偏置后合并。"""
    torch.manual_seed(0)
    b, h, s, k, d, S = 1, 4, 128, 96, 64, 5
    q = torch.randn(b, h, s, d, device=DEV)
    k1 = torch.randn(b, h, 200, d, device=DEV); v1 = torch.randn(b, h, 200, d, device=DEV)
    k2 = torch.randn(b, h, k, d, device=DEV); v2 = torch.randn(b, h, k, d, device=DEV)
    ref = sdpa(q, torch.cat([k1, k2.repeat(1, 1, S, 1)], 2),
               torch.cat([v1, v2.repeat(1, 1, S, 1)], 2))
    o1 = sdpa(q, k1, v1); o2 = sdpa(q, k2, v2)
    l1 = torch.logsumexp(q @ k1.transpose(-1, -2) / math.sqrt(d), -1)
    l2 = torch.logsumexp(q @ k2.transpose(-1, -2) / math.sqrt(d), -1)
    got = merge_lse(o1, l1, o2, l2, math.log(S))
    err = (got - ref).abs().max().item()
    check("T2 BG 键 x%d == beta=log(%d) 合并" % (S, S), err < TOL, "max_err=%.2e" % err)
    bad = merge_lse(o1, l1, o2, l2, 0.0)
    check("T2' 不加 beta 应当明显错（反证 beta 必要）",
          (bad - ref).abs().max().item() > 10 * TOL,
          "no_beta_err=%.3f" % (bad - ref).abs().max().item())


def t_end_to_end():
    """走 flex provider 的 splitstream_attention，对比复制后的稠密结果。"""
    torch.manual_seed(0)
    b, h, s, k1n, k2n, d, S = 1, 8, 512, 700, 180, 128, 3
    q = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16)
    k1 = torch.randn(b, h, k1n, d, device=DEV, dtype=torch.bfloat16)
    v1 = torch.randn(b, h, k1n, d, device=DEV, dtype=torch.bfloat16)
    k2 = torch.randn(b, h, k2n, d, device=DEV, dtype=torch.bfloat16)
    v2 = torch.randn(b, h, k2n, d, device=DEV, dtype=torch.bfloat16)
    ref = sdpa(q, torch.cat([k1, k2.repeat(1, 1, S, 1)], 2),
               torch.cat([v1, v2.repeat(1, 1, S, 1)], 2))
    got = splitstream_attention(q, torch.cat([k1, k2], 2), torch.cat([v1, v2], 2),
                                k1n, beta=math.log(S), provider="flex")
    err = (got.float() - ref.float()).abs().max().item()
    check("端到端 flex provider (S=3, bf16)", err < 1e-2, "max_err=%.2e" % err)
    o, l = attn_lse(q, k1, v1, "manual")
    o2, l2 = attn_lse(q, k1, v1, "flex")
    check("manual provider 与 flex 一致",
          (o.float() - o2.float()).abs().max().item() < 1e-2)


def t_split_geometry():
    # 口径：VAE 空间 16x 下采样 + patch 2x2
    for (w, h), tok in (((1280, 736), 920), ((1536, 864), 1296)):
        gh, gw = h // 16 // 2, w // 16 // 2
        check("%dx%d = %d token/帧" % (w, h, gh * gw), gh * gw == tok, "gh=%d gw=%d" % (gh, gw))

    T, gh, gw, S = 37, 23, 40, 5
    rows = T * gh * gw
    check("无掩码时压缩率为 0（诚实默认）", True, "rows=%d" % rows)
    fg = route3_plan(0, T, 46, 80, S)
    check("fg_mask=None → 一行都删不掉",
          int(fg.fg_local.numel()) == rows and int(fg.bg_local.numel()) == 0,
          "fg=%d bg=%d" % (fg.fg_local.numel(), fg.bg_local.numel()))

    # 人物占 24% 画面（14/23 行 x 16/40 列）
    from route3.splitstream_attn import bg_anchor_indices
    m = torch.zeros(T, gh, gw, dtype=torch.bool)
    m[:, 4:18, 12:28] = True
    p = route3_plan(1000, T, 46, 80, S, fg_mask=m.reshape(T, -1), phase_aware=False)
    n_fg, n_bg = int(p.fg_local.numel()), int(p.bg_local.numel())
    kept = n_fg + n_bg
    n_anchor = len(range(0, T, S))
    per_frame_fg = int(m[0].sum())
    ok = (n_fg == T * per_frame_fg
          and n_bg == n_anchor * (gh * gw - per_frame_fg)
          and kept == 13856)
    check("24%% 前景 / S_bg=5 (纯等距) → 保持 13856/34040（降 59%%）", ok,
          "fg=%d bg=%d kept=%d" % (n_fg, n_bg, kept))
    check("route3_plan 的 n_full / beta0",
          p.n_full == 1000 + n_fg and abs(p.beta0 - math.log(T / n_anchor)) < 1e-6,
          "n_full=%d beta0=%.3f" % (p.n_full, p.beta0))

    # I/P 相位感知锚点 + 严格分池质量守恒（2026-11-23 绿幕 6Hz 锯齿根治回归）：
    # s_bg=3 必须：
    #   1. 100% 保留全部 p=0 I-token（放入 branch1，β_I=0，绝不 2.31× 过权！）
    #   2. P 池跨全部 4 个 P 相位 {1,2,3,4} 均匀抽样（含首尾 t=1,36），β_P = ln(29/11) = 0.969
    #   3. 注意力 I:P 有效质量比严格等于原生稠密 8:29 (21.6%:78.4%)！
    anc3 = bg_anchor_indices(37, s_bg=3, phase_aware=True)
    all_i = list(range(0, 37, 5))
    p_anc3 = [t for t in anc3 if t % 5 != 0]
    p_phases = {t % 5 for t in p_anc3}
    check("s_bg=3 相位感知：100%% 保留 I-token 且 P 锚覆盖全部 4 个 P 相位 {1,2,3,4}",
          all(i in anc3 for i in all_i) and p_phases == {1, 2, 3, 4} and 1 in p_anc3 and 36 in p_anc3,
          "n=%d p_anc=%s" % (len(anc3), p_anc3))
    p3 = route3_plan(1000, T, 46, 80, 3, fg_mask=m.reshape(T, -1), phase_aware=True)
    bg_per_frame = gh * gw - per_frame_fg
    n_p_total = T - len(all_i)
    # 局部质量守恒：t=36 紧邻 t=34（中间 t=35 为 I-token），左右均无被删 P-token，归入 branch1 (1.0×)
    n_b1_p3 = 1  # {36}
    n_b2_p3 = len(p_anc3) - n_b1_p3  # 10
    expect_beta_p = math.log((n_p_total - n_b1_p3) / n_b2_p3)
    expect_n_full_p3 = 1000 + int(p3.fg_local.numel()) + (len(all_i) + n_b1_p3) * bg_per_frame
    check("s_bg=3 局部质量守恒：未压缩 I-BG 与无被删邻近的尾帧 P(36) 归入 branch1，仅真压缩 P-BG 乘 β_P",
          p3.n_full == expect_n_full_p3
          and abs(p3.beta0 - expect_beta_p) < 1e-6,
          "n_full=%d (expect %d) beta_P=%.4f" % (p3.n_full, expect_n_full_p3, p3.beta0))

    # 硬切钉住 pin_tokens=[16,17,18,19]：{17,18,36} 均无被删 P 邻近，归入 branch1 (β=0)，总 P 质量严格等于 29！
    p3_pin = route3_plan(1000, T, 46, 80, 3, fg_mask=m.reshape(T, -1), phase_aware=True,
                         pin_tokens=[16, 17, 18, 19])
    n_b1_pin = 3  # 17, 18, 36
    n_b2_pin = 10
    expect_n_full_pin = 1000 + int(p3_pin.fg_local.numel()) + (len(all_i) + n_b1_pin) * bg_per_frame
    total_p_mass = n_b1_pin * 1.0 + n_b2_pin * math.exp(p3_pin.beta0)
    check("pin_tokens=[16..19] 硬切钉住：无被删邻近 P-锚进 branch1 且总 P 注意力质量严格守恒 (==29.0)",
          p3_pin.n_full == expect_n_full_pin
          and abs(total_p_mass - float(n_p_total)) < 1e-5
          and 17 in p3_pin.anchors and 18 in p3_pin.anchors,
          "n_full=%d mass=%.4f beta_P=%.4f" % (p3_pin.n_full, total_p_mass, p3_pin.beta0))

    # 高速运镜全量钉住 Shot 1 (t=0..18) + 静止压缩 Shot 2 (t=19..36)：
    # Shot 1 全部 14 个 P-token 必须同在 branch1 (精确 1.0×，零 1.8x/1.0x 锯齿！)，
    # Shot 2 仅 {19,23,27,31,34} 留在 branch2 (β=ln(13/5)=0.9555)，Shot 1 与 Shot 2 总质量双双守恒！
    shot1_dense_pins = [2, 3, 6, 7, 9, 11, 13, 14, 16, 17, 18, 19]
    p3_dense_s1 = route3_plan(1000, T, 46, 80, 3, fg_mask=m.reshape(T, -1), phase_aware=True,
                              pin_tokens=shot1_dense_pins)
    n_b1_s1 = 16  # 1..18 全部 14 个 P + 36（15 个？等：1,2,3,4,6,7,8,9,11,12,13,14,16,17,18 共 15 个 + 36 = 16）
    n_b2_s2 = 5   # 19, 23, 27, 31, 34
    expect_n_full_s1 = 1000 + int(p3_dense_s1.fg_local.numel()) + (len(all_i) + n_b1_s1) * bg_per_frame
    expect_beta_s2 = math.log(13.0 / 5.0)
    s1_p_mass = 15 * 1.0  # t=1..18 全部在 branch1
    s2_p_mass = 1 * 1.0 + n_b2_s2 * math.exp(p3_dense_s1.beta0)  # t=36 + 5 个 branch2
    check("Shot1全稠密+Shot2压缩：Shot1全员1.0×零邻近权重振荡，Shot2质量严格守恒 (14/14)",
          p3_dense_s1.n_full == expect_n_full_s1
          and abs(p3_dense_s1.beta0 - expect_beta_s2) < 1e-6
          and abs(s1_p_mass - 15.0) < 1e-5
          and abs(s2_p_mass - 14.0) < 1e-5,
          "n_full=%d beta=%.4f s1=%.2f s2=%.2f" % (p3_dense_s1.n_full, p3_dense_s1.beta0, s1_p_mass, s2_p_mass))


def t_phase():
    """组内 5 个相位的起始帧 (0,1,5,9,13) → 最远 Δt = 21.667。"""
    from route3.splitstream_attn import MAX_PHASE_DT, phase_of_rows, beta_for_sequence
    check("最远相位 Δt = 21.667", abs(MAX_PHASE_DT - 13 / 24 * 40) < 1e-9,
          "%.3f" % MAX_PHASE_DT)
    ph = phase_of_rows(6, 2, 2)   # 6 帧 x 4 行
    expect = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [0] * 4)
    check("phase_of_rows = t % 5", bool((ph == expect).all()), str(ph.tolist()[:8]))

    # 相位感知 beta: [H,P] -> [1,H,S]
    beta = torch.zeros(3, 5); beta[:, 4] = 0.7
    b = beta_for_sequence(beta, 0, torch.tensor([0, 4, 4, 4, 4, 0]))
    check("beta_for_sequence 按相位展开", b.shape == (1, 3, 6)
          and abs(b[0, 0, 1].item() - 0.7) < 1e-6 and b[0, 0, 0].item() == 0.0,
          str(b.shape))
    # 逐 query 的 beta 能直接进 merge
    import torch.nn.functional as F
    q = torch.randn(1, 3, 6, 32, device=DEV); kk = torch.randn(1, 3, 6, 32, device=DEV)
    o1, l1 = q, torch.zeros(1, 3, 6, device=DEV)
    out = merge_lse(o1, l1, o1, l1, b)
    check("merge_lse 接受 [1,H,S] 逐行 beta", out.shape == q.shape
          and torch.isfinite(out).all().item())


# --------------------------------------------------------------------------- #
def t_rope_ref():
    """纯 torch 版 RoPE 必须与 comfy_kitchen 融合算子数值一致。"""
    from comfy.ldm.minimax.model import rope_rotation_table
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    from route3.route3_block import rms_rope_split_half_ref

    torch.manual_seed(0)
    s, h, d = 64, 4, 128
    q = torch.randn(1, s, h, d, device=DEV)
    k = torch.randn(1, s, h, d, device=DEV)
    ang = torch.randn(s, 48, device=DEV)
    table = rope_rotation_table(torch.cat([ang, ang], -1), q.dtype)
    qw = torch.randn(d, device=DEV).abs() + 0.5
    kw = torch.randn(d, device=DEV).abs() + 0.5
    rq, rk = rms_rope_split_half(q, k, table, qw, kw, epsilon=1e-5, rot_dim=96)
    mq, mk = rms_rope_split_half_ref(q, k, table, qw, kw, epsilon=1e-5, rot_dim=96)
    err = max((rq - mq).abs().max().item(), (rk - mk).abs().max().item())
    check("纯 torch RoPE == comfy_kitchen 融合算子", err < 1e-5, "max_err=%.2e" % err)
    # 后 32 维必须原样透传（NoPE）
    xn = torch.nn.functional.rms_norm(q, (d,), weight=qw, eps=1e-5)
    check("RoPE 后 32 维 NoPE 透传",
          torch.allclose(mq[..., 96:], xn[..., 96:], atol=1e-6))


def t_train_backward():
    """tiny MiniMaxH3Model：原 block 必崩，out-of-place + LoRA 必须能反传。"""
    import comfy.ops
    import comfy.model_management as mm
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from route3.route3_block import install, attach_lora, patch_attention

    OPS = comfy.ops.disable_weight_init
    torch.manual_seed(0)
    m = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1,
                       num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                       text_dim=64, timestep_input_dim=256, time_embed_hidden_size=256,
                       time_embed_dim=128, dtype=torch.float32, device=DEV,
                       operations=OPS).to(DEV)
    x = torch.randn(1, 24, 10, 8, 8, device=DEV)
    a = torch.randn(1, 32, 2, 20, device=DEV)
    ctx = torch.randn(1, 7, 64, device=DEV)
    ts = torch.tensor([500.0], device=DEV)
    mm.in_training = True

    out = m([x.clone(), a.clone()], ts, ctx, {}, minimax_payload={})
    try:
        (out[0].square().mean() + out[1].square().mean()).backward()
        check("原 DiTBlock 反传（预期失败）", False, "竟然过了，说明源码改了")
    except RuntimeError as e:
        check("原 DiTBlock 反传（预期失败，已复现）",
              "inplace operation" in str(e), type(e).__name__)

    m.zero_grad(set_to_none=True)
    install(m, force=True)
    patch_attention(m)
    wrapped = attach_lora(m, rank=8)
    out = m([x, a], ts, ctx, {}, minimax_payload={})
    (out[0].square().mean() + out[1].square().mean()).backward()
    gnorm = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                if "lora_" in n and p.grad is not None)
    nz = sum(1 for n, p in m.named_parameters() if "lora_" in n and p.grad is not None)
    check("out-of-place block + LoRA(%d 层) 反传", (gnorm > 0 and nz > 0),
          "%d 个 LoRA 张量有梯度, 梯度和=%.3e" % (nz, gnorm))
    check("LoRA 只挂 qkv/out", len(wrapped) == 2 * len(m.blocks), str(wrapped[:2]))


def main():
    print("torch", torch.__version__, "device", DEV)
    t_rope_channel_map()
    t_merge_exact()
    t_merge_tiled()
    t_end_to_end()
    t_split_geometry()
    t_phase()
    t_rope_ref()
    if "--with-model" in sys.argv:
        try:
            t_train_backward()
        except Exception:
            traceback.print_exc()
            FAIL.append("t_train_backward(crash)")
    print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
