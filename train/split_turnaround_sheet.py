# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic 3-view character turnaround sheet splitter for H3-AnimeEngine v3.0.

Detects inter-view vertical whitespace valleys via core-body ink projection, traces
minimum-energy Dynamic Programming (DP) vertical seams so protruding wings, tails,
hair curls, and dress trains are never clipped and adjacent views never bleed in,
removes margin text / logos / color swatches / border frames via connected-component
filtering, and exports clean padded single-view crops (`<char>_front.png`,
`<char>_side.png`, `<char>_back.png`, plus weapon-aware crops for Grok's Demonic Axe)
into `D:/comfyui/ComfyUI/input/veo_cast/`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class SheetSpec:
    filename: str
    slug: str
    style_mode: str  # "25d_illust" or "2d_cel_sakuga"
    has_left_weapon: bool = False


DEFAULT_SHEET_SPECS: tuple[SheetSpec, ...] = (
    SheetSpec("GPT娘龙族少女三视图设定图.png", "gpt_dragon", "25d_illust", False),
    SheetSpec("Kimi星月风角色设定集.png", "kimi_star", "25d_illust", False),
    SheetSpec("克劳德酱三视图与细节设定稿.png", "claude_chan", "25d_illust", False),
    SheetSpec("双子座星辉猫娘设定图.png", "gemini_cat", "25d_illust", False),
    SheetSpec("哥特吸血鬼角色设计图.png", "grok_vampire", "2d_cel_sakuga", True),
    SheetSpec("ChatGPT 图像 2026年11月16日 16_48_43.png", "deepseek_maid", "2d_cel_sakuga", False),
    SheetSpec("ChatGPT 图像 2026年11月16日 16_48_57.png", "deepseek_chibi", "2d_cel_sakuga", False),
)


def compute_ink_and_grad(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return foreground ink darkness above faint grid noise and local gradient magnitude."""
    arr = rgb.astype(np.float32)
    # Darkness relative to white background; subtract 16.0 so faint pastel grid lines (~244+) vanish
    raw_dark = 255.0 - np.min(arr, axis=-1)
    sat = np.max(arr, axis=-1) - np.min(arr, axis=-1)
    ink = np.maximum(raw_dark - 16.0, 0.0) + 0.35 * np.maximum(sat - 12.0, 0.0)

    gray = np.mean(arr, axis=-1)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    grad = np.hypot(gx, gy)
    grad = np.maximum(grad - 6.0, 0.0)
    return ink, grad


def find_valley_anchor(
    ink: np.ndarray,
    x_min_frac: float,
    x_max_frac: float,
    y_top_frac: float = 0.12,
    y_bot_frac: float = 0.84,
) -> int:
    """Find the cleanest vertical whitespace valley column in [x_min_frac*W, x_max_frac*W]."""
    h, w = ink.shape
    y0, y1 = int(round(h * y_top_frac)), int(round(h * y_bot_frac))
    x0, x1 = int(round(w * x_min_frac)), int(round(w * x_max_frac))
    col_profile = np.mean(ink[y0:y1, :], axis=0)
    # Smooth with a 15-px box filter to prefer wide whitespace valleys over 1-px gaps
    kernel = np.ones(15, dtype=np.float32) / 15.0
    smooth = np.convolve(col_profile, kernel, mode="same")
    sub = smooth[x0:x1]
    return int(x0 + int(np.argmin(sub)))


def trace_dp_vertical_seam(
    ink: np.ndarray,
    grad: np.ndarray,
    x_anchor: int,
    half_window_frac: float = 0.09,
    max_step: int = 3,
    drift_weight: float = 0.14,
) -> np.ndarray:
    """Trace a top-to-bottom minimum-energy seam anchored near `x_anchor`."""
    h, w = ink.shape
    win = max(24, int(round(w * half_window_frac)))
    xa = max(0, x_anchor - win)
    xb = min(w, x_anchor + win + 1)
    sw = xb - xa

    xs = np.arange(xa, xb, dtype=np.float32)[None, :]
    ink_sub = ink[:, xa:xb]
    energy = (
        ink_sub * 60.0
        + (ink_sub > 5.0).astype(np.float32) * 2500.0
        + grad[:, xa:xb] * 30.0
        + drift_weight * np.abs(xs - float(x_anchor))
    )

    dp = np.empty((h, sw), dtype=np.float32)
    back = np.zeros((h, sw), dtype=np.int16)
    dp[0] = energy[0]

    shifts = range(-max_step, max_step + 1)
    for y in range(1, h):
        prev = dp[y - 1]
        best_cost = np.full(sw, np.inf, dtype=np.float32)
        best_prev = np.zeros(sw, dtype=np.int16)
        for dx in shifts:
            step_pen = 0.30 * float(dx * dx)
            if dx < 0:
                cand = prev[: sw + dx] + step_pen
                target_slice = slice(-dx, sw)
                src_idx = np.arange(0, sw + dx, dtype=np.int16)
            elif dx > 0:
                cand = prev[dx:] + step_pen
                target_slice = slice(0, sw - dx)
                src_idx = np.arange(dx, sw, dtype=np.int16)
            else:
                cand = prev
                target_slice = slice(0, sw)
                src_idx = np.arange(0, sw)
                src_idx = src_idx.astype(np.int16)

            cur_best = best_cost[target_slice]
            better = cand < cur_best
            cur_best[better] = cand[better]
            best_cost[target_slice] = cur_best
            cur_prev = best_prev[target_slice]
            cur_prev[better] = src_idx[better]
            best_prev[target_slice] = cur_prev

        dp[y] = energy[y] + best_cost
        back[y] = best_prev

    seam = np.empty(h, dtype=np.int32)
    idx = int(np.argmin(dp[h - 1]))
    seam[h - 1] = xa + idx
    for y in range(h - 1, 0, -1):
        idx = int(back[y, idx])
        seam[y - 1] = xa + idx
    return seam


def apply_panel_seams(
    rgb: np.ndarray,
    left_seam: np.ndarray | None,
    right_seam: np.ndarray | None,
) -> np.ndarray:
    """Copy `rgb` and paint pixels outside [left_seam, right_seam) pure white."""
    h, w, _ = rgb.shape
    out = rgb.copy()
    xs = np.arange(w, dtype=np.int32)[None, :]
    outside = np.zeros((h, w), dtype=bool)
    if left_seam is not None:
        outside |= xs < left_seam[:, None]
    if right_seam is not None:
        outside |= xs >= right_seam[:, None]
    out[outside] = 255
    return out


def pad_tight_crop(work: np.ndarray, mask: np.ndarray, pad: int = 36) -> Image.Image:
    """Whiten outside `mask`, crop tightly to visible foreground, and center on 8px-aligned white canvas."""
    struct = ndimage.generate_binary_structure(2, 2)
    halo = ndimage.binary_dilation(mask, structure=struct, iterations=6)
    cleaned = work.copy()
    cleaned[~halo] = 255

    vis = halo & ((255.0 - np.min(cleaned.astype(np.float32), axis=-1)) > 6.0)
    ys, xs = np.where(vis)
    if ys.size == 0:
        ys, xs = np.where(mask)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = cleaned[y0:y1, x0:x1]

    ch, cw, _ = crop.shape
    out_h = ((ch + 2 * pad + 7) // 8) * 8
    out_w = ((cw + 2 * pad + 7) // 8) * 8
    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    oy = (out_h - ch) // 2
    ox = (out_w - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw] = crop
    return Image.fromarray(canvas, mode="RGB")


def isolate_primary_mask(
    panel_rgb: np.ndarray,
    dil_iters: int = 2,
    include_nearby: bool = True,
) -> np.ndarray:
    """Return hole-filled boolean mask of the main character/object inside `panel_rgb`."""
    h, w, _ = panel_rgb.shape
    work = panel_rgb.copy()
    # Strip 4-px outer sheet frame artifact lines on left/right/bottom edges
    work[:, :4] = 255
    work[:, max(0, w - 4) :] = 255
    work[max(0, h - 4) :, :] = 255

    ink, grad = compute_ink_and_grad(work)
    fg = (ink > 7.5) | (grad > 10.0)

    border_strip = np.zeros((h, w), dtype=bool)
    border_strip[: int(h * 0.022), :] = True
    border_strip[int(h * 0.985) :, :] = True
    border_strip[:, : int(w * 0.018)] = True
    border_strip[:, int(w * 0.990) :] = True

    struct = ndimage.generate_binary_structure(2, 2)
    thin_line = fg & border_strip & (~ndimage.binary_opening(fg, structure=np.ones((5, 5))))
    fg_core = fg & (~thin_line)

    bridged = ndimage.binary_dilation(fg_core, structure=struct, iterations=dil_iters)
    labels, num = ndimage.label(bridged, structure=struct)
    if num == 0:
        raise RuntimeError("No foreground component detected in panel")

    comp_weights = np.zeros(num + 1, dtype=np.float64)
    for cid in range(1, num + 1):
        comp_weights[cid] = float(np.sum(ink[labels == cid]))

    main_cid = int(np.argmax(comp_weights))
    main_mask = labels == main_cid
    keep_region = main_mask.copy()

    if include_nearby:
        near_main = ndimage.binary_dilation(main_mask, structure=struct, iterations=2)
        for cid in range(1, num + 1):
            if cid == main_cid:
                continue
            comp = labels == cid
            if not np.any(comp & near_main):
                continue
            ys, xs = np.where(comp)
            if float(xs.max()) / w < 0.12 and (float(ys.max()) / h < 0.28 or float(ys.min()) / h > 0.92):
                continue
            if float(xs.max()) / w < 0.072:
                continue
            keep_region |= comp

    # Fill all interior holes inside the character/weapon silhouette so pale skin,
    # smiling mouths, sclera, and white dress highlights are 100% preserved losslessly!
    keep_region = ndimage.binary_fill_holes(keep_region)
    return keep_region


def clean_and_crop_single_character(panel_rgb: np.ndarray, pad: int = 36) -> Image.Image:
    cleaned_panel = panel_rgb.copy()
    cleaned_panel[:, :4] = 255
    cleaned_panel[:, max(0, cleaned_panel.shape[1] - 4) :] = 255
    mask = isolate_primary_mask(cleaned_panel, dil_iters=2, include_nearby=True)
    return pad_tight_crop(cleaned_panel, mask, pad=pad)


def process_sheet(spec: SheetSpec, src_dir: Path, out_dir: Path, pad: int = 36) -> dict:
    src_path = src_dir / spec.filename
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing turnaround sheet: {src_path}")

    img = Image.open(src_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    h, w, _ = rgb.shape
    ink, grad = compute_ink_and_grad(rgb)

    v1_anchor = find_valley_anchor(ink, 0.31, 0.49)
    v2_anchor = find_valley_anchor(ink, 0.57, 0.76)

    seam1 = trace_dp_vertical_seam(ink, grad, v1_anchor, half_window_frac=0.09)
    seam2 = trace_dp_vertical_seam(ink, grad, v2_anchor, half_window_frac=0.11)

    front_panel = apply_panel_seams(rgb, None, seam1)
    side_panel = apply_panel_seams(rgb, seam1, seam2)
    back_panel = apply_panel_seams(rgb, seam2, None)

    outputs: dict[str, list[int]] = {}

    if spec.has_left_weapon:
        # Separate left Demonic Axe (`y in [0.34H, 0.955H]`) from Grok's front body via dedicated DP seam
        w_anchor = int(round(w * 0.118))
        w_seam = trace_dp_vertical_seam(ink, grad, w_anchor, half_window_frac=0.045, max_step=3, drift_weight=0.08)
        body_only_panel = apply_panel_seams(front_panel, w_seam, None)
        axe_only_panel = apply_panel_seams(front_panel, None, w_seam)
        # Strip top-left title/color swatches (`y < 0.34H`) and bottom text (`y > 0.955H`) from axe panel
        axe_only_panel[: int(h * 0.345), :] = 255
        axe_only_panel[int(h * 0.952) :, :] = 255

        body_mask = isolate_primary_mask(body_only_panel, dil_iters=2, include_nearby=True)
        axe_mask = isolate_primary_mask(axe_only_panel, dil_iters=2, include_nearby=False)

        front_with_axe = pad_tight_crop(front_panel, body_mask | axe_mask, pad=pad)
        front_body = pad_tight_crop(body_only_panel, body_mask, pad=pad)
        axe_only = pad_tight_crop(axe_only_panel, axe_mask, pad=pad)

        for suffix, im in (
            ("front", front_with_axe),
            ("front_with_axe", front_with_axe),
            ("front_body", front_body),
            ("axe", axe_only),
        ):
            name = f"{spec.slug}_{suffix}.png"
            im.save(out_dir / name, format="PNG", compress_level=6)
            outputs[name] = [im.width, im.height]
    else:
        front_im = clean_and_crop_single_character(front_panel, pad=pad)
        name = f"{spec.slug}_front.png"
        front_im.save(out_dir / name, format="PNG", compress_level=6)
        outputs[name] = [front_im.width, front_im.height]

    side_im = clean_and_crop_single_character(side_panel, pad=pad)
    back_im = clean_and_crop_single_character(back_panel, pad=pad)

    for suffix, im in (("side", side_im), ("back", back_im)):
        name = f"{spec.slug}_{suffix}.png"
        im.save(out_dir / name, format="PNG", compress_level=6)
        outputs[name] = [im.width, im.height]

    return {
        "slug": spec.slug,
        "source": spec.filename,
        "style_mode": spec.style_mode,
        "sheet_size": [w, h],
        "valley_anchors_x": [int(v1_anchor), int(v2_anchor)],
        "seam1_range_x": [int(seam1.min()), int(seam1.max())],
        "seam2_range_x": [int(seam2.min()), int(seam2.max())],
        "outputs": outputs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="D:/工作空间/创作/veo", help="Directory containing source turnaround sheets")
    ap.add_argument(
        "--out-dir",
        default="D:/comfyui/ComfyUI/input/veo_cast",
        help="Destination directory inside ComfyUI/input",
    )
    ap.add_argument("--pad", type=int, default=36, help="Clean white border padding in pixels")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in DEFAULT_SHEET_SPECS:
        info = process_sheet(spec, src_dir=src_dir, out_dir=out_dir, pad=args.pad)
        manifest.append(info)
        print(
            f"[split] {spec.slug:15s} ({info['sheet_size'][0]}x{info['sheet_size'][1]}) "
            f"seams=({info['seam1_range_x']}, {info['seam2_range_x']}) -> "
            + ", ".join(f"{k}:{v[0]}x{v[1]}" for k, v in info["outputs"].items())
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[split] Done! Exported {sum(len(m['outputs']) for m in manifest)} clean crops -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
