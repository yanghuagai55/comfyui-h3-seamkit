# SPDX-License-Identifier: GPL-3.0-or-later
"""开机硬门槛：不是 A100 / RTX PRO 6000 直接退出（不做 T4/L4 降级）。

用法：
  python train/preflight.py --role teacher
  python train/preflight.py --role train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ALLOWED = ("A100", "RTX PRO 6000", "RTX 6000")


def _sh(cmd: str) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or "").strip()
    except Exception as exc:
        return 1, repr(exc)


def _fail(msg: str, snapshot: dict, path: Path) -> int:
    snapshot["ok"] = False
    snapshot["error"] = msg
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    print(f"snapshot -> {path}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["teacher", "train"], required=True)
    ap.add_argument("--min-vram-gb", type=float, default=40.0)
    ap.add_argument("--min-disk-gb", type=float, default=0.0)
    ap.add_argument("--require-hf", action="store_true")
    ap.add_argument("--snapshot", default="")
    args = ap.parse_args()

    root = Path("/content/work")
    out = Path(args.snapshot) if args.snapshot else root / "logs" / f"env_snapshot_{args.role}.json"
    snap = {
        "role": args.role,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
    }

    rc, gpu = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    snap["nvidia_smi"] = gpu
    if rc != 0 or not gpu:
        return _fail("nvidia-smi 不可用（没有 GPU 或驱动异常）", snap, out)
    first = gpu.splitlines()[0]
    name = first.split(",")[0].strip()
    snap["gpu_name"] = name
    if not any(tag.lower() in name.lower() for tag in ALLOWED):
        return _fail(f"GPU 不在白名单（{name}；只允许 A100 / RTX PRO 6000）", snap, out)

    try:
        vram_mib = float(first.split(",")[1].strip().split()[0])
    except Exception:
        vram_mib = 0.0
    snap["vram_mib"] = vram_mib
    if vram_mib < args.min_vram_gb * 1024:
        return _fail(f"显存不足：{vram_mib:.0f} MiB < {args.min_vram_gb:.0f} GiB", snap, out)
    snap["vram_mode"] = "bf16_80g" if vram_mib >= 70 * 1024 else "int8_40g"

    rc, free = _sh("df -B1 /content | tail -1 | awk '{print $4}'")
    try:
        free_gb = int(free) / 1e9
    except Exception:
        free_gb = 0.0
    snap["content_free_gb"] = round(free_gb, 1)
    min_disk = args.min_disk_gb or (120.0 if args.role == "teacher" else 30.0)
    if free_gb < min_disk:
        return _fail(f"/content 可用空间 {free_gb:.1f} GB < {min_disk:.0f} GB", snap, out)

    try:
        import torch
        snap["torch"] = torch.__version__
        snap["cuda"] = torch.version.cuda
        snap["compute_capability"] = list(torch.cuda.get_device_capability())
        if not torch.cuda.is_available():
            return _fail("torch.cuda.is_available() = False", snap, out)
        if tuple(torch.cuda.get_device_capability()) < (8, 0):
            return _fail("compute capability < 8.0", snap, out)
    except Exception as exc:
        return _fail(f"torch 导入失败：{exc}", snap, out)

    try:
        import comfy_kitchen as ck
        snap["comfy_kitchen"] = True
        snap["comfy_kitchen_sol_attn"] = bool(
            hasattr(ck, "sol_attn") and hasattr(ck, "sol_attn_is_available") and ck.sol_attn_is_available()
        )
        snap["comfy_kitchen_backends"] = list(getattr(ck, "list_backends", lambda: {})() or {})
    except Exception as exc:
        snap["comfy_kitchen"] = False
        snap["comfy_kitchen_error"] = str(exc)[:300]

    try:
        import sageattention
        snap["sageattention"] = True
        snap["sageattention_version"] = getattr(sageattention, "__version__", "unknown")
    except Exception as exc:
        snap["sageattention"] = False
        snap["sageattention_error"] = str(exc)[:200]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    snap["hf_token_env"] = bool(token)
    snap["hf_token_file"] = token_file.is_file()
    if args.require_hf and not (token or token_file.is_file()):
        return _fail("缺少 HF_TOKEN（私有一包推拉需要）", snap, out)

    snap["ok"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    print(f"PREFLIGHT OK ({args.role}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
