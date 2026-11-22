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
from route3.engine import splitstream_forward

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
    check("s_bg=1 + 非平凡重排：视频与原生 forward 逐位一致", d0 < 1e-6, "max_diff=%.2e" % d0)
    check("s_bg=1 + 非平凡重排：音频一致", d1 < 1e-6, "max_diff=%.2e" % d1)
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
          plan2.full_phase().numel() == plan2.n_full + int(plan2.bg_local.numel()),
          "S=%d" % plan2.full_phase().numel())

    print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
