#!/usr/bin/env python3
"""Offline layer-wise LoRA weight slicer for MiniMax-H3 (style-preserving surgery).

Background
----------
When a sparse-sharpness LoRA distilled from an official/realistic teacher is loaded onto
a stylized base model (e.g. DaSiWa thin-line 2D anime), high-frequency (<80px) brushstroke
and outline-thickness priors live predominantly in early/late MLP layers (`blocks 0..4` and
`blocks 32..49`), whereas `attn` (`qkv_proj`/`out_proj`) handles cross-frame sparse routing
and reference-image fine-line transfer (convex Value combinations, zero texture hallucination).

Presets
-------
- `stylekeep`:
    * attn.qkv_proj / attn.out_proj (all 50 blocks): 1.00x
    * mlp.fc1 / mlp.fc2: blocks 0..4 = 0.25x, blocks 5..31 = 0.45x, blocks 32..49 = 0.15x
- `attn_only`:
    * attn.qkv_proj / attn.out_proj (all 50 blocks): 1.00x
    * mlp.fc1 / mlp.fc2: omitted cleanly (halves LoRA size & skips 100 MLP BF16 GEMMs!)

Usage
-----
    python train/slice_lora_stylekeep.py \\
        --input  D:\\comfyui\\ComfyUI\\models\\loras\\minimax_h3_sol_sharp_v2_s2_step165.safetensors \\
        --output D:\\comfyui\\ComfyUI\\models\\loras\\minimax_h3_sol_sharp_v2_final_stylekeep.safetensors \\
        --preset stylekeep
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_KEY_RE = re.compile(
    r"^(?P<prefix>.*blocks\.(?P<block>\d+)\.(?P<mod>attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)))\.(?P<part>alpha|lora_down\.weight|lora_up\.weight)$"
)


def _layer_scale(block_idx: int, mod: str, preset: str, attn_scale: float, mlp_early: float, mlp_mid: float, mlp_late: float) -> float:
    if mod.startswith("attn."):
        return attn_scale
    if preset == "attn_only":
        return 0.0
    if block_idx <= 4:
        return mlp_early
    if block_idx <= 31:
        return mlp_mid
    return mlp_late


def slice_lora(
    input_path: str,
    output_path: str,
    preset: str = "stylekeep",
    attn_scale: float = 1.0,
    mlp_early: float = 0.25,
    mlp_mid: float = 0.45,
    mlp_late: float = 0.15,
) -> dict:
    out_tensors: dict[str, torch.Tensor] = {}
    stats = defaultdict(lambda: {"orig_norm": 0.0, "new_norm": 0.0, "layers": 0})
    meta: dict[str, str] = {}

    with safe_open(input_path, framework="pt", device="cpu") as f:
        if f.metadata():
            meta.update(f.metadata())
        keys = list(f.keys())

        # Group by layer prefix so we can verify completeness and cleanly drop 0-scale layers
        groups: dict[str, dict[str, str]] = defaultdict(dict)
        passthrough: list[str] = []
        for k in keys:
            m = _KEY_RE.match(k)
            if not m:
                passthrough.append(k)
                continue
            groups[m.group("prefix")][m.group("part")] = k

        for k in passthrough:
            out_tensors[k] = f.get_tensor(k)

        kept_layers = 0
        dropped_layers = 0
        for prefix, parts in sorted(groups.items()):
            m = _KEY_RE.match(next(iter(parts.values())))
            assert m is not None
            bidx = int(m.group("block"))
            mod = m.group("mod")
            scale = _layer_scale(bidx, mod, preset, attn_scale, mlp_early, mlp_mid, mlp_late)

            bucket = (
                "attn_all"
                if mod.startswith("attn.")
                else ("mlp_0_4" if bidx <= 4 else ("mlp_5_31" if bidx <= 31 else "mlp_32_49"))
            )

            up_t = f.get_tensor(parts["lora_up.weight"])
            down_t = f.get_tensor(parts["lora_down.weight"])
            alpha_t = f.get_tensor(parts["alpha"]) if "alpha" in parts else None

            # Effective Frobenius norm of (alpha/r) * (U @ D) via fast rank-r identity:
            # ||U D||_F = sqrt(tr((U^T U)(D D^T)))
            u32 = up_t.float()
            d32 = down_t.float()
            rank = d32.shape[0]
            alpha_val = float(alpha_t.item()) if alpha_t is not None else float(rank)
            eff_mul = alpha_val / float(rank)
            orig_frob = float(torch.sqrt(torch.clamp(((u32.T @ u32) * (d32 @ d32.T)).sum(), min=0.0)).item()) * eff_mul
            stats[bucket]["orig_norm"] += orig_frob
            stats[bucket]["layers"] += 1

            if scale == 0.0:
                dropped_layers += 1
                continue

            kept_layers += 1
            if scale == 1.0:
                new_up = up_t
            else:
                new_up = (u32 * scale).to(up_t.dtype)

            nu32 = new_up.float()
            new_frob = float(torch.sqrt(torch.clamp(((nu32.T @ nu32) * (d32 @ d32.T)).sum(), min=0.0)).item()) * eff_mul
            stats[bucket]["new_norm"] += new_frob

            if torch.isnan(new_up).any() or torch.isinf(new_up).any():
                raise ValueError(f"Non-finite values produced for {parts['lora_up.weight']}")

            out_tensors[parts["lora_down.weight"]] = down_t
            out_tensors[parts["lora_up.weight"]] = new_up
            if alpha_t is not None:
                out_tensors[parts["alpha"]] = alpha_t

    meta["seamkit_slice_preset"] = preset
    meta["seamkit_slice_source"] = os.path.basename(input_path)
    meta["seamkit_slice_scales"] = (
        f"attn={attn_scale},mlp_0_4={0.0 if preset == 'attn_only' else mlp_early},"
        f"mlp_5_31={0.0 if preset == 'attn_only' else mlp_mid},"
        f"mlp_32_49={0.0 if preset == 'attn_only' else mlp_late}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out_tensors, output_path, metadata=meta)

    size_mib = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[slice] {preset} -> {output_path}")
    print(f"        kept={kept_layers} layers ({len(out_tensors)} tensors), dropped={dropped_layers} layers, size={size_mib:.2f} MiB")
    for bucket in ("attn_all", "mlp_0_4", "mlp_5_31", "mlp_32_49"):
        s = stats[bucket]
        ratio = (s["new_norm"] / s["orig_norm"]) if s["orig_norm"] > 0 else 0.0
        print(f"        {bucket:9s} ({s['layers']:3d}L): orig_frob={s['orig_norm']:7.2f} -> new_frob={s['new_norm']:7.2f} ({ratio:.3f}x)")

    return {
        "output": output_path,
        "kept_layers": kept_layers,
        "dropped_layers": dropped_layers,
        "tensors": len(out_tensors),
        "size_mib": size_mib,
        "stats": dict(stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline layer-wise stylekeep LoRA slicer for MiniMax-H3.")
    ap.add_argument("--input", required=True, help="Source .safetensors LoRA path")
    ap.add_argument("--output", required=True, help="Destination .safetensors LoRA path")
    ap.add_argument("--preset", choices=["stylekeep", "attn_only", "custom"], default="stylekeep")
    ap.add_argument("--attn-scale", type=float, default=1.0)
    ap.add_argument("--mlp-early", type=float, default=0.25, help="MLP scale for blocks 0..4")
    ap.add_argument("--mlp-mid", type=float, default=0.45, help="MLP scale for blocks 5..31")
    ap.add_argument("--mlp-late", type=float, default=0.15, help="MLP scale for blocks 32..49")
    args = ap.parse_args()

    slice_lora(
        input_path=args.input,
        output_path=args.output,
        preset=args.preset,
        attn_scale=args.attn_scale,
        mlp_early=args.mlp_early,
        mlp_mid=args.mlp_mid,
        mlp_late=args.mlp_late,
    )


if __name__ == "__main__":
    main()
