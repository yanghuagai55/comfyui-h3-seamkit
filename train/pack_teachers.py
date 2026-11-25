# SPDX-License-Identifier: GPL-3.0-or-later
"""把 `gen_teacher_native.py` 的原始落盘压成单条训练样本（一文件一样本）。

原始目录（`gen_teacher_native.py` 直接写在 out_dir 根下）：
  <id>.z_native.pt   # 原生 1080p AV 真值：parts = [video, audio]
  <id>.z_up.pt       # 480p VAE 往返 + 2x 上采样糊底：parts = [video]
  <id>.cond.pt       # 二采 conditioning（text fp32 + minimax_token_tags / refs）

输出：
  <out>/<id>.pt
  {
    "id", "video", "audio",        # fp16 CPU，原生真值
    "z_up",                        # fp16 CPU，与 video 同 shape
    "cond",                        # [[text_fp16, meta], ...]
    "meta": {"frames","task","seed","latent_t","lat_h","lat_w","audio_t"}
  }
  <out>/_manifest.json             # 按 shape 分桶的清单（训练脚本据此建 bucket）

用法：
  python train/pack_teachers.py --raw /content/work/native --out /content/work/teacher_cache \
      --prompts /content/work/prompts/prompts_run.jsonl --tar
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import torch

TAGS = ("z_native", "z_up", "cond")


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _parts(payload, key, want: int | None = None) -> list[torch.Tensor]:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"missing '{key}'")
    parts = [t for t in value if torch.is_tensor(t)] if isinstance(value, (list, tuple)) else [value]
    if want is not None and len(parts) != want:
        raise ValueError(f"{key}: expected {want} parts, got {len(parts)}")
    for i, t in enumerate(parts):
        if t.numel() == 0:
            raise ValueError(f"{key}[{i}] is empty")
        if not torch.isfinite(t).all():
            raise ValueError(f"{key}[{i}] has NaN/Inf")
    return parts


def _read_prompts(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                out[str(rec["id"])] = rec
    return out


def pack_one(raw_dir: Path, key: str, prompts: dict[str, dict]) -> dict:
    missing = [f"{key}.{tag}.pt" for tag in TAGS if not (raw_dir / f"{key}.{tag}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing}")

    native = _load(raw_dir / f"{key}.z_native.pt")
    up = _load(raw_dir / f"{key}.z_up.pt")
    cond = _load(raw_dir / f"{key}.cond.pt")

    video, audio = _parts(native, "parts", want=2)
    (z_up,) = _parts(up, "parts", want=1)
    if tuple(video.shape) != tuple(z_up.shape):
        raise ValueError(f"{key}: z_native video {tuple(video.shape)} != z_up {tuple(z_up.shape)}")
    if video.ndim != 5:
        raise ValueError(f"{key}: video latent must be 5D [B,C,T,H,W], got {tuple(video.shape)}")

    conditioning = cond.get("conditioning")
    if not conditioning:
        raise ValueError(f"{key}: empty conditioning")
    packed_cond = []
    for item in conditioning:
        text, meta = item[0], item[1]
        if not torch.is_tensor(text) or text.ndim != 3:
            raise ValueError(f"{key}: bad text tensor {type(text)}")
        # fp32 -> bf16：与 fp16 同为 2 字节（落盘减半），但指数范围 3.4e38 彻底避免 Qwen3-VL 文本隐态 >65504 溢出成 Inf
        t_bf16 = text.detach().to(torch.bfloat16).contiguous()
        if not torch.isfinite(t_bf16).all():
            raise ValueError(f"{key}: text conditioning has NaN/Inf")
        packed_cond.append([t_bf16, meta])

    prompt = prompts.get(key, {})
    meta = {
        "frames": int(prompt.get("frames") or 0),
        "task": prompt.get("task") or native.get("task") or "",
        "seed": int(prompt.get("seed") or 0),
        "latent_t": int(video.shape[2]),
        "lat_h": int(video.shape[3]),
        "lat_w": int(video.shape[4]),
        "audio_t": int(audio.shape[-1]),
        "n_cond": len(packed_cond),
        "refs": len((packed_cond[0][1] or {}).get("minimax_refs") or []),
        "text_len": int(packed_cond[0][0].shape[1]),
    }
    return {
        "id": key,
        "video": video.detach().to(torch.float16).contiguous(),
        "audio": audio.detach().to(torch.float16).contiguous(),
        "z_up": z_up.detach().to(torch.float16).contiguous(),
        "cond": packed_cond,
        "meta": meta,
    }


def _write_tar(out_root: Path, files: list[Path]) -> Path | None:
    tar_path = out_root / "teacher_cache_v1.tar.zst"
    try:
        import zstandard  # type: ignore
    except Exception:
        print("[pack] zstandard 不可用，改用 gzip tar（.tar.gz）", flush=True)
        tar_path = out_root / "teacher_cache_v1.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
        return tar_path

    import io as _io

    cctx = zstandard.ZstdCompressor(level=10, threads=4)
    with tar_path.open("wb") as raw_fh, cctx.stream_writer(raw_fh, closefd=False) as zfh:
        with tarfile.open(fileobj=zfh, mode="w|") as tf:
            for p in files:
                info = tarfile.TarInfo(p.name)
                info.size = p.stat().st_size
                info.mtime = int(p.stat().st_mtime)
                with p.open("rb") as fh:
                    tf.addfile(info, fh)
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--tar", action="store_true", help="额外打 teacher_cache_v1.tar.zst")
    ap.add_argument("--keep", action="store_true", help="保留 out 目录里已有的单样本文件")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(Path(args.prompts) if args.prompts else None)

    keys = sorted(
        p.name[: -len(".z_native.pt")]
        for p in raw_root.glob("*.z_native.pt")
        if p.name != "smoke_t2va_0001.z_native.pt"
    )
    print(f"[pack] {len(keys)} candidates, prompts={len(prompts)}", flush=True)

    manifest: list[dict] = []
    buckets: dict[str, dict] = {}
    ok = fail = 0
    for i, key in enumerate(keys, 1):
        try:
            rec = pack_one(raw_root, key, prompts)
            path = out_root / f"{key}.pt"
            if not (args.keep and path.is_file()):
                torch.save(rec, path)
            mb = path.stat().st_size / 1e6
            m = rec["meta"]
            bkey = f"{m['latent_t']}x{m['lat_h']}x{m['lat_w']}"
            b = buckets.setdefault(bkey, {"frames": m["frames"], "latent_t": m["latent_t"],
                                          "count": 0, "audio_t": m["audio_t"]})
            b["count"] += 1
            b["frames"] = max(b["frames"], m["frames"])
            manifest.append({"id": key, "file": path.name, "mb": round(mb, 2), **m})
            ok += 1
            print(f"[{i}/{len(keys)}] {key}: {mb:.1f} MB  {bkey} frames={m['frames']}", flush=True)
        except Exception as exc:
            fail += 1
            print(f"[{i}/{len(keys)}] {key}: FAIL {exc}", flush=True)

    total_gb = sum(m["mb"] for m in manifest) / 1024.0
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": ok,
        "failed": fail,
        "total_gb": round(total_gb, 3),
        "buckets": buckets,
        "samples": manifest,
    }
    (out_root / "_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pack] done ok={ok} fail={fail} total={total_gb:.2f} GB buckets={list(buckets)}", flush=True)

    if args.tar and ok:
        files = [out_root / m["file"] for m in manifest] + [out_root / "_manifest.json"]
        tar_path = _write_tar(out_root, files)
        if tar_path is not None:
            print(f"[pack] tar -> {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
