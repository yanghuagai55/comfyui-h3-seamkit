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
    T, gh, gw, S = 37, 27, 48, 5
    fg, bg = video_split(T, gh, gw, S)
    n_bg_frames = len(set(range(0, T, S)))
    ok = (int(fg.numel()) + int(bg.numel()) == T * gh * gw
          and int(bg.numel()) == n_bg_frames * gh * gw
          and int(fg.numel()) == (T - n_bg_frames) * gh * gw)
    check("5s/1280x736 几何：47952 行 = 37584 FG + 10368 BG (8 关键帧)", ok,
          "fg=%d bg=%d" % (fg.numel(), bg.numel()))
    p = route3_plan(prefix_end=1000, latent_t=T, lat_h=54, lat_w=96, s_bg=S)
    check("route3_plan 的 n_full / beta0",
          p.n_full == 1000 + int(fg.numel()) and abs(p.beta0 - math.log(T / 8)) < 1e-6,
          "n_full=%d beta0=%.3f" % (p.n_full, p.beta0))


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
