# SPDX-License-Identifier: GPL-3.0-or-later
"""训练冒烟：验证「补丁后 = 原模型」（LoRA=0 稠密）以及稀疏分支可反传。

用法：
  python train/smoke_train.py --base <bf16.safetensors> --cache /content/work/teacher_cache \
      --comfy /content/ComfyUI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h3_lora as T  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--record", default="")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--neighbors", type=int, default=2)
    ap.add_argument("--topk", type=float, default=0.25)
    args = ap.parse_args()

    device = torch.device("cuda")
    mm, comfy_sd, h3 = T.bootstrap(args.comfy)
    T.patch_out_of_place(h3)
    dm = T.load_dm(comfy_sd, args.base, device)

    groups = T.scan_cache(Path(args.cache))
    if not groups:
        raise SystemExit("teacher cache 为空")
    key = sorted(groups)[0]
    recs = groups[key]
    rec = next((r for r in recs if args.record and args.record in r["path"].name), recs[0])
    print(f"[smoke] group={key.split('|')[0]} n={len(recs)} record={rec['path'].name}", flush=True)

    group = T._build_group(h3, key, recs, args.block_size, args.neighbors)
    sigma = args.sigma
    sigma_a = h3.time_shift_sigma(sigma, 12.0, 3.0)
    gen = torch.Generator(device=device).manual_seed(1234)
    batch = T.build_batch(rec, group, sigma, sigma_a, device, gen)
    ts = torch.tensor([sigma * 1000.0], device=device)
    layout = group.layout
    layout.position_ids = batch["pos"]
    payload = batch["payload"]

    with torch.no_grad():
        ctx_ref = dm.preprocess_text_embeds(batch["text"])
        t0 = time.time()
        ref = dm(batch["x"], ts, ctx_ref, transformer_options={}, minimax_payload=payload)
        dt_ref = time.time() - t0
    print(f"[smoke] reference dense forward {dt_ref:.1f}s", flush=True)

    params = T.attach_lora(dm, 32)
    cfg = {"rope_hf": 1.0, "attn_temp": 1.0, "sparse": False, "topk": 0.0, "topk_max": 0.0,
           "ckpt": False, "grad_clamp": 10}
    T.patch_attention(dm, {"attn_temp": 1.0})
    T.patch_run_blocks(dm, h3, cfg)
    with torch.no_grad():
        out = dm(batch["x"], ts, ctx_ref, transformer_options={}, minimax_payload=payload)
    cos_v = _cos(out[0], ref[0])
    cos_a = _cos(out[1], ref[1])
    print(f"[smoke] patched dense vs native  cos_video={cos_v:.6f} cos_audio={cos_a:.6f}", flush=True)
    ok = cos_v > 0.999 and cos_a > 0.999

    # 稀疏 + 反传
    cfg.update({"rope_hf": 1.32, "attn_temp": 1.08, "sparse": True, "topk": args.topk,
                "topk_max": max(args.topk, 0.30), "ckpt": True})
    layout.position_ids = batch["pos"]
    to = {"_h3_plan": group, "_h3_keep_kv": batch["keep"]}
    t0 = time.time()
    out = dm(batch["x"], ts, ctx_ref, transformer_options=to, minimax_payload=payload)
    loss = -out[0].float().mean() - out[1].float().mean()
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    dt_sparse = time.time() - t0
    finite = bool(torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all())
    print(f"[smoke] sparse fwd+bwd {dt_sparse:.1f}s grads={len(grads)}/{len(params)} finite={finite} "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    if not grads:
        ok = False
        print("[smoke] LoRA 无梯度：checkpoint + 冻结基座的组合有问题", flush=True)

    print("[smoke] PASS" if ok else "[smoke] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
