# SPDX-License-Identifier: GPL-3.0-or-later
"""训练 ckpt -> ComfyUI LoRA safetensors（3060/4060 直接用 LoraLoaderModelOnly 加载）。

用法：
  python train/finalize_lora.py --ckpt /content/work/lora_run/last.pt \
      --out /content/work/lora_run/minimax_h3_ref2va_sparse_r32.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lora = ck.get("lora") or {}
    if not lora:
        raise SystemExit(f"{args.ckpt} 里没有 lora 权重")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    from safetensors.torch import save_file

    sd: dict[str, torch.Tensor] = {}
    dead = []
    for key, val in sorted(lora.items()):
        down = val["A"].to(dtype).contiguous()
        up = val["B"].to(dtype).contiguous()
        rank = down.shape[0]
        alpha = float(val.get("scale", 1.0)) * rank
        prefix = f"diffusion_model.{key}"
        sd[f"{prefix}.lora_down.weight"] = down
        sd[f"{prefix}.lora_up.weight"] = up
        sd[f"{prefix}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        if float(up.float().abs().max()) < 1e-6:
            dead.append(key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out), metadata={"format": "pt", "step": str(ck.get("step", 0))})
    total = sum(t.numel() for t in sd.values())
    print(f"[finalize] {len(lora)} layers, {total / 1e6:.1f}M tensors -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB) step={ck.get('step')}", flush=True)
    if dead:
        print(f"[finalize] 仍有 {len(dead)} 层 B=0（未训练到): {dead[:5]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
