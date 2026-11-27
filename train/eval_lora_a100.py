# SPDX-License-Identifier: GPL-3.0-or-later
"""快速视频 A/B 验收脚本（v3 升级：原生兼容 AMD MI300X ROCm `sdpa_mask`、异构 rank 64/32 与运动峰值 ROI 锐度评测）。

用法：
  python3 train/eval_lora_a100.py \
      --base /content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
      --vae /content/local_models/vae/minimax_h3_sol_vae_fp16.safetensors \
      --cache /content/work/teacher_cache \
      --lora /content/work/lora_run_v3/steps/step_000220.safetensors \
      --eval-frames 39 --out /content/work/eval_v3_s220 \
      --hf-repo coolwindtang/minimax-h3-teacher-cache --hf-eval-prefix eval_v3_s220 --upload-hf
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import train_h3_lora as T  # noqa: E402


def _inspect_lora_ranks(lora_path: Path) -> tuple[int, int]:
    from safetensors.torch import load_file

    sd = load_file(str(lora_path), device="cpu")
    r_attn = 32
    r_mlp = 32
    for k, v in sd.items():
        if k.endswith(".lora_down.weight"):
            if ".attn." in k:
                r_attn = int(v.shape[0])
            elif ".mlp." in k:
                r_mlp = int(v.shape[0])
    return r_attn, r_mlp


def _load_lora_into_dm(dm, lora_path: Path, scale_mult: float) -> int:
    from safetensors.torch import load_file

    sd = load_file(str(lora_path), device="cpu")
    loaded = 0
    for i, blk in enumerate(dm.blocks):
        for name in T.LORA_TARGETS:
            owner = blk.attn if name in T.ATTN_TARGETS else blk.mlp
            lin = getattr(owner, name)
            lora = getattr(lin, "_train_lora", None)
            if lora is None:
                continue
            prefix = f"diffusion_model.blocks.{i}.{'attn' if name in T.ATTN_TARGETS else 'mlp'}.{name}"
            dk, uk, ak = f"{prefix}.lora_down.weight", f"{prefix}.lora_up.weight", f"{prefix}.alpha"
            if dk in sd and uk in sd:
                lora.A.data.copy_(sd[dk].to(lora.A.device, lora.A.dtype))
                lora.B.data.copy_(sd[uk].to(lora.B.device, lora.B.dtype))
                rank = sd[dk].shape[0]
                alpha = float(sd[ak].item()) if ak in sd else float(rank)
                lora.scale = (alpha / float(rank)) * float(scale_mult)
                loaded += 1
            else:
                lora.B.data.zero_()
                lora.scale = 0.0
    return loaded


def _zero_lora(dm) -> None:
    for blk in dm.blocks:
        for name in T.LORA_TARGETS:
            owner = blk.attn if name in T.ATTN_TARGETS else blk.mlp
            lora = getattr(getattr(owner, name), "_train_lora", None)
            if lora is not None:
                lora.B.data.zero_()
                lora.scale = 0.0


@torch.no_grad()
def run_pass2_euler(dm, h3, rec: dict, group, sigmas: list[float], device: torch.device, seed: int) -> torch.Tensor:
    data = torch.load(rec["path"], map_location="cpu", weights_only=False)
    z_up = data["z_up"].to(device=device, dtype=torch.float32)
    z_audio = data["audio"].to(device=device, dtype=torch.float32)
    cond_text, _ = data["cond"][0]

    real_len = int(cond_text.shape[1])
    delta = group.text_pad - real_len
    text = torch.empty(1, group.text_pad, cond_text.shape[-1], dtype=torch.bfloat16)
    text[0, :real_len] = cond_text[0].to(torch.bfloat16)
    if group.text_pad > real_len:
        text[0, real_len:] = cond_text[0, -1].to(torch.bfloat16)
    text = text.to(device)

    tags = torch.ones(group.text_pad, dtype=torch.long)
    if rec["tags"] is not None:
        tags[:real_len] = rec["tags"][:real_len].to(torch.long)
    keep = torch.ones(group.S, dtype=torch.bool)
    keep[real_len:group.text_pad] = False
    keep = keep.to(device)

    pos = group.layout.position_ids.clone()
    if delta:
        pos[group.text_pad:, 0] -= float(delta)

    payload = {}
    if rec["tags"] is not None:
        payload["text_token_tags"] = tags.to(device)
    if rec["keyframes"]:
        payload["keyframes"] = rec["keyframes"]
    if rec["refs"]:
        payload["refs"] = rec["refs"]
    payload["cond_video_latents"] = [k["latent"] for k in (rec["keyframes"] or []) if k.get("latent") is not None] + \
                                    [r["latent"] for r in (rec["refs"] or []) if r.get("latent") is not None]
    aud = [r["audio_latent"] for r in (rec["refs"] or []) if r.get("audio_latent") is not None]
    if aud:
        payload["cond_audio_latents"] = aud
    payload["seed"] = seed
    payload["audio_scale"] = 1.0
    payload["layout"] = group.layout

    gen = torch.Generator(device=device).manual_seed(seed)
    eps_v = torch.randn(z_up.shape, generator=gen, device=device, dtype=torch.float32)
    eps_a = torch.randn(z_audio.shape, generator=gen, device=device, dtype=torch.float32)

    s0 = float(sigmas[0])
    sa0 = h3.time_shift_sigma(s0, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else s0
    x_v = (1.0 - s0) * z_up + s0 * eps_v
    x_a = (1.0 - sa0) * z_audio + sa0 * eps_a

    layout = group.layout
    old_pos = layout.position_ids
    layout.position_ids = pos
    try:
        ctx = dm.preprocess_text_embeds(text)
        sched = list(sigmas) + [0.0]
        for idx in range(len(sigmas)):
            s_cur = float(sched[idx])
            s_nxt = float(sched[idx + 1])
            sa_cur = h3.time_shift_sigma(s_cur, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else s_cur
            sa_nxt = h3.time_shift_sigma(s_nxt, 12.0, 3.0) if (s_nxt > 0 and hasattr(h3, "time_shift_sigma")) else s_nxt
            torch.manual_seed(seed + idx)
            to = {"_h3_plan": group, "_h3_keep_kv": keep}
            out = dm([x_v.to(torch.bfloat16), x_a.to(torch.bfloat16)],
                     torch.tensor([s_cur * 1000.0], device=device), ctx,
                     transformer_options=to, minimax_payload=payload)
            v_v = out[0].float()
            v_a = out[1].float()
            x_v = x_v + (s_nxt - s_cur) * v_v
            x_a = x_a + (sa_nxt - sa_cur) * v_a
    finally:
        layout.position_ids = old_pos

    return x_v.cpu()


def _decode_video(comfy_sd, vae_path: str, z_vid: torch.Tensor) -> np.ndarray:
    import comfy.utils

    sd = comfy.utils.load_torch_file(str(vae_path))
    vae = comfy_sd.VAE(sd=sd)
    with torch.no_grad():
        imgs = vae.decode(z_vid)
    if imgs.ndim == 5:
        imgs = imgs[0]
    arr = (imgs.float().clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    del vae, sd
    torch.cuda.empty_cache()
    return arr


def _frame_metrics(vid: np.ndarray, ref: np.ndarray | None = None) -> dict:
    """计算全局 LapVar@1080、峰值运动帧 Motion-ROI LapVar、Halo(99-1) 与 PSNR。"""
    n = vid.shape[0]
    idxs = sorted({min(n - 1, max(0, int(n * p))) for p in (0.25, 0.5, 0.75)})
    laps, halos, psnrs = [], [], []
    for i in idxs:
        g = (0.299 * vid[i, :, :, 0] + 0.587 * vid[i, :, :, 1] + 0.114 * vid[i, :, :, 2]).astype(np.float32)
        lap = (
            -4.0 * g[1:-1, 1:-1]
            + g[:-2, 1:-1] + g[2:, 1:-1]
            + g[1:-1, :-2] + g[1:-1, 2:]
        )
        laps.append(float(np.var(lap)))
        p99, p01 = np.percentile(lap, [99.0, 1.0])
        halos.append(float(p99 - p01))
        if ref is not None and ref.shape == vid.shape:
            mse = float(np.mean((vid[i].astype(np.float32) - ref[i].astype(np.float32)) ** 2))
            psnrs.append(50.0 if mse < 1e-6 else float(10.0 * math.log10((255.0 ** 2) / mse)))

    # 额外计算运动峰值帧的前景 ROI 锐度（直击 00:07 大动作痛点）
    src_for_mot = ref if (ref is not None and ref.shape == vid.shape) else vid
    if src_for_mot.shape[0] >= 2:
        f_diffs = np.mean(np.abs(src_for_mot[1:].astype(np.float32) - src_for_mot[:-1].astype(np.float32)), axis=(1, 2, 3))
        bf = int(np.argmax(f_diffs))
        pix_diff = np.mean(np.abs(src_for_mot[bf + 1].astype(np.float32) - src_for_mot[bf].astype(np.float32)), axis=-1)
        fm = vid[bf + 1]
    else:
        bf = 0
        pix_diff = np.ones(vid.shape[1:3], dtype=np.float32)
        fm = vid[0]
    gm = (0.299 * fm[:, :, 0] + 0.587 * fm[:, :, 1] + 0.114 * fm[:, :, 2]).astype(np.float32)
    lap_m = -4.0 * gm[1:-1, 1:-1] + gm[:-2, 1:-1] + gm[2:, 1:-1] + gm[1:-1, :-2] + gm[1:-1, 2:]
    lh, lw = lap_m.shape
    bh, bw = max(1, lh // 32), max(1, lw // 32)
    ch, cw = bh * 32, bw * 32
    dc = pix_diff[1:1 + ch, 1:1 + cw].reshape(bh, 32, bw, 32).mean(axis=(1, 3))
    lb = lap_m[:ch, :cw].reshape(bh, 32, bw, 32)
    roi_px = lb.transpose(0, 2, 1, 3)[dc >= float(np.percentile(dc, 75.0))]
    motion_lap = float(np.var(roi_px)) if roi_px.size > 0 else float(np.var(lap_m))

    out = {
        "lap1080": round(float(np.mean(laps)), 2),
        "motion_roi_lap": round(motion_lap, 2),
        "peak_frame": bf + 1 if n >= 2 else 0,
        "halo": round(float(np.mean(halos)), 2),
    }
    if psnrs:
        out["psnr_vs_gt"] = round(float(np.mean(psnrs)), 2)
    return out


def _save_strip(panels: list[tuple[str, np.ndarray, dict]], out_jpg: Path) -> None:
    from PIL import Image, ImageDraw

    w_tile, h_tile, bar = 640, 360, 42
    canvas = Image.new("RGB", (w_tile * len(panels), (h_tile + bar) * 2), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    peak_idx = panels[-1][2].get("peak_frame", panels[0][1].shape[0] // 2)
    for col, (label, vid, met) in enumerate(panels):
        f_idx = min(vid.shape[0] - 1, max(0, int(peak_idx)))
        mid = vid[f_idx]
        im = Image.fromarray(mid)
        full = im.resize((w_tile, h_tile), Image.Resampling.LANCZOS)
        H, W = mid.shape[:2]
        cy, cx = H // 2, W // 2
        y0, x0 = max(0, cy - h_tile // 2), max(0, cx - w_tile // 2)
        crop = Image.fromarray(mid[y0:y0 + h_tile, x0:x0 + w_tile])

        x_off = col * w_tile
        canvas.paste(full, (x_off, bar))
        canvas.paste(crop, (x_off, bar * 2 + h_tile))
        txt = f"{label} | MotLap={met['motion_roi_lap']} Lap={met['lap1080']} Halo={met['halo']}"
        if "psnr_vs_gt" in met:
            txt += f" PSNR={met['psnr_vs_gt']}dB"
        draw.text((x_off + 10, 12), txt, fill=(245, 245, 245))
        draw.text((x_off + 10, bar + h_tile + 12), f"{label} (Peak-Motion Frame #{f_idx} 1:1 Crop)", fill=(180, 230, 255))
    canvas.save(out_jpg, quality=93)


def _save_mp4(frames: np.ndarray, out_mp4: Path, fps: int = 24) -> None:
    H, W = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out_mp4),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert p.stdin is not None
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--vae", default="/content/local_models/vae/minimax_h3_sol_vae_fp16.safetensors")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--lora", default="", help="本地 LoRA .safetensors 路径")
    ap.add_argument("--lora-hf", default="lora_run_v3/steps/step_000220.safetensors")
    ap.add_argument("--hf-repo", default="coolwindtang/minimax-h3-teacher-cache")
    ap.add_argument("--token-file", default="/content/work/hftoken.txt")
    ap.add_argument("--out", default="/content/work/eval_v3")
    ap.add_argument("--sigmas", default="0.85,0.54,0.20", help="对齐 4060 二采 ManualSigmas 三锚点")
    ap.add_argument("--topk", type=float, default=0.22)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--attn-backend", choices=["auto", "flex", "sdpa_mask"], default="auto")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--eval-frames", type=int, default=39, help="抽取的帧数桶（22 / 39 / 73；0=不限）")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--upload-hf", action="store_true")
    ap.add_argument("--hf-eval-prefix", default="eval_v3")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    lora_path = Path(args.lora) if args.lora else None
    if lora_path is None or not lora_path.is_file():
        from huggingface_hub import hf_hub_download
        from hf_shards import read_token

        got = hf_hub_download(args.hf_repo, args.lora_hf, token=read_token(args.token_file))
        lora_path = Path(got)
        print(f"[eval] 从 HF 拉取 LoRA -> {lora_path}", flush=True)

    mm, comfy_sd, h3 = T.bootstrap(args.comfy)
    dm = T.load_dm(comfy_sd, args.base, device)
    r_attn, r_mlp = _inspect_lora_ranks(lora_path)
    T.attach_lora(dm, rank=r_mlp, rank_attn=r_attn, rank_mlp=r_mlp)
    T.patch_attention(dm, {"attn_temp": 1.08})
    eff_backend = T.resolve_attn_backend(args.attn_backend)
    cfg = {
        "rope_hf": 1.32, "attn_temp": 1.08, "sparse": True,
        "attn_backend": eff_backend,
        "topk": args.topk, "topk_max": args.topk, "ckpt": False, "grad_clamp": 0.0,
        "mask_jitter": 0.0,
    }
    T.patch_run_blocks(dm, h3, cfg)

    groups = T.scan_cache(Path(args.cache))
    if not groups:
        raise SystemExit("teacher_cache 为空，请先拉取云端分片")
    plan_groups = {k: T._build_group(h3, k, v, args.block_size, 2) for k, v in groups.items()}

    # 优先选高动态 sakuga/act 样本（1 Ref2VA + 1 T2VA）
    picked = []
    for want_action in (True, False):
        for want_ref in (True, False):
            for g in plan_groups.values():
                for rec in g.records:
                    if args.eval_frames > 0 and int(rec["meta"].get("frames", 22)) != args.eval_frames:
                        continue
                    if bool(rec.get("is_action")) != want_action:
                        continue
                    if bool(rec.get("refs")) == want_ref and all(r["path"] != rec["path"] for r, _ in picked):
                        picked.append((rec, g))
                        break
                if len(picked) >= args.samples:
                    break
            if len(picked) >= args.samples:
                break
        if len(picked) >= args.samples:
            break
    picked = picked[: args.samples]
    if not picked:
        raise SystemExit(f"未在 teacher_cache 中找到 frames={args.eval_frames} 的样本（可设 --eval-frames 0）")

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    print(
        f"[eval] lora={lora_path.name} rank_attn={r_attn} rank_mlp={r_mlp} backend={eff_backend} "
        f"topk={args.topk} sigmas={sigmas} samples={[r['path'].stem for r, _ in picked]}",
        flush=True,
    )

    latents_cache = {}
    for rec, g in picked:
        sid = rec["path"].stem
        data = torch.load(rec["path"], map_location="cpu", weights_only=False)
        latents_cache[sid] = {"z_up": data["z_up"], "teacher_gt": data["video"]}

        _zero_lora(dm)
        t0 = time.time()
        latents_cache[sid]["base_no_lora"] = run_pass2_euler(dm, h3, rec, g, sigmas, device, args.seed)
        print(f"  [{sid}] base_no_lora done ({time.time() - t0:.1f}s)", flush=True)

        for mult, tag in ((0.70, "lora_070"), (1.00, "lora_100")):
            n_loaded = _load_lora_into_dm(dm, lora_path, mult)
            t0 = time.time()
            latents_cache[sid][tag] = run_pass2_euler(dm, h3, rec, g, sigmas, device, args.seed)
            print(f"  [{sid}] {tag} (layers={n_loaded}) done ({time.time() - t0:.1f}s)", flush=True)

    del dm
    torch.cuda.empty_cache()

    summary = {
        "lora": str(lora_path.name),
        "rank_attn": r_attn,
        "rank_mlp": r_mlp,
        "sigmas": sigmas,
        "topk": args.topk,
        "samples": {},
    }
    produced_files: list[Path] = []
    for sid, lats in latents_cache.items():
        vids = {k: _decode_video(comfy_sd, args.vae, z) for k, z in lats.items()}
        gt = vids["teacher_gt"]
        order = ["z_up", "base_no_lora", "lora_070", "lora_100", "teacher_gt"]
        mets = {k: _frame_metrics(vids[k], None if k == "teacher_gt" else gt) for k in order}
        summary["samples"][sid] = mets
        print(f"[metrics {sid}] " + json.dumps(mets, ensure_ascii=False), flush=True)

        strip = out_dir / f"compare_{sid}.jpg"
        _save_strip([(k, vids[k], mets[k]) for k in order], strip)
        produced_files.append(strip)

        from PIL import Image

        tri = []
        for fi in range(gt.shape[0]):
            row = [
                np.asarray(Image.fromarray(vids[k][fi]).resize((640, 360), Image.Resampling.LANCZOS))
                for k in ("base_no_lora", "lora_070", "lora_100")
            ]
            tri.append(np.concatenate(row, axis=1))
        mp4 = out_dir / f"sidebyside_{sid}.mp4"
        _save_mp4(np.stack(tri, axis=0), mp4)
        produced_files.append(mp4)

    met_path = out_dir / "eval_metrics.json"
    met_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    produced_files.append(met_path)

    if args.upload_hf:
        from huggingface_hub import HfApi
        from hf_shards import read_token

        api = HfApi(token=read_token(args.token_file))
        for fp in produced_files:
            dst = f"{args.hf_eval_prefix}/{fp.name}"
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=dst, repo_id=args.hf_repo, repo_type="model")
            print(f"[hf-eval] uploaded -> {dst}", flush=True)

    print(f"[eval done] files in {out_dir}: {[p.name for p in produced_files]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
