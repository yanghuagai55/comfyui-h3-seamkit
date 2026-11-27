# SPDX-License-Identifier: GPL-3.0-or-later
"""开机硬门槛（兼容 NVIDIA CUDA 与魔搭 AMD MI300X/MI308X ROCm）：不符合大显存门槛立即退出。

用法：
  python train/preflight.py --role teacher --min-vram-gb 70
  python train/preflight.py --role train --min-vram-gb 70 --work-root /mnt/workspace/h3_v3/work
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

ALLOWED = (
    "A100",
    "H100",
    "H200",
    "RTX PRO 6000",
    "RTX 6000",
    "MI300",
    "MI308",
    "MI325",
    "MI355",
    "Instinct",
    "AMD",
    "gfx942",
)


def _sh(cmd: str) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or "").strip()
    except Exception as exc:
        return 1, repr(exc)


def _default_work_root() -> Path:
    if Path("/mnt/workspace").is_dir() and not Path("/content/work").exists():
        return Path("/mnt/workspace/h3_v3/work")
    return Path("/content/work")


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
    ap.add_argument("--work-root", default="")
    ap.add_argument("--model-root", default="/content/local_models")
    ap.add_argument("--require-hf", action="store_true")
    ap.add_argument("--snapshot", default="")
    args = ap.parse_args()

    root = Path(args.work_root) if args.work_root else _default_work_root()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(args.snapshot) if args.snapshot else root / "logs" / f"env_snapshot_{args.role}.json"
    snap: dict = {
        "role": args.role,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "work_root": str(root),
    }

    # 1. PyTorch + CUDA/ROCm 硬件检测（统一走 torch.cuda，兼容 NVIDIA CUDA 与 AMD ROCm HIP）
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda"] = torch.version.cuda
        snap["hip"] = getattr(torch.version, "hip", None)
        snap["backend"] = "rocm" if snap["hip"] else "cuda"
        if not torch.cuda.is_available():
            return _fail("torch.cuda.is_available() = False（未检测到可用 CUDA/ROCm GPU）", snap, out)
        props = torch.cuda.get_device_properties(0)
        name = str(torch.cuda.get_device_name(0) or props.name or "").strip()
        arch_name = str(getattr(props, "gcnArchName", "") or "")
        vram_mib = float(props.total_memory) / (1024.0 * 1024.0)
        cap = tuple(torch.cuda.get_device_capability(0))
        snap["gpu_name"] = name
        if arch_name:
            snap["gcn_arch"] = arch_name
        snap["vram_mib"] = round(vram_mib, 1)
        snap["compute_capability"] = list(cap)
    except Exception as exc:
        return _fail(f"PyTorch GPU 检测失败：{exc}", snap, out)

    rc_nv, nv_out = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if rc_nv == 0 and nv_out:
        snap["nvidia_smi"] = nv_out
    else:
        rc_rocm, rocm_out = _sh("rocm-smi --showproductname --showdriverversion")
        if rc_rocm == 0 and rocm_out:
            snap["rocm_smi"] = rocm_out[:800]

    full_id = f"{name} {arch_name}".lower()
    if not any(tag.lower() in full_id for tag in ALLOWED):
        return _fail(f"GPU 不在白名单（{name} / {arch_name}；允许 A100 / H100 / PRO 6000 / AMD MI300X/MI308X）", snap, out)

    if vram_mib < args.min_vram_gb * 1024.0:
        return _fail(f"显存不足：{vram_mib:.0f} MiB < {args.min_vram_gb:.0f} GiB", snap, out)
    if cap < (8, 0):
        return _fail(f"compute capability {cap} < (8, 0)", snap, out)

    if vram_mib >= 150 * 1024:
        snap["vram_mode"] = "bf16_192g_rocm" if snap["backend"] == "rocm" else "bf16_192g"
    elif vram_mib >= 70 * 1024:
        snap["vram_mode"] = "bf16_80g"
    else:
        snap["vram_mode"] = "int8_40g"

    # 2. 磁盘检查（分别统计持久化工作目录与大模型目录）
    usage_work = shutil.disk_usage(root)
    free_work_gb = usage_work.free / 1e9
    snap["work_free_gb"] = round(free_work_gb, 1)
    model_dir = Path(args.model_root)
    if model_dir.exists():
        snap["model_root_free_gb"] = round(shutil.disk_usage(model_dir).free / 1e9, 1)
    min_disk = args.min_disk_gb if args.min_disk_gb > 0 else 20.0
    if free_work_gb < min_disk:
        return _fail(f"工作目录 {root} 可用空间 {free_work_gb:.1f} GB < {min_disk:.0f} GB", snap, out)

    # 3. 可选加速扩展状态快照（ROCm 下无 comfy_kitchen/sageattention 属正常现象）
    try:
        import comfy_kitchen as ck

        snap["comfy_kitchen"] = True
        snap["comfy_kitchen_sol_attn"] = bool(
            hasattr(ck, "sol_attn") and hasattr(ck, "sol_attn_is_available") and ck.sol_attn_is_available()
        )
    except Exception as exc:
        snap["comfy_kitchen"] = False
        snap["comfy_kitchen_note"] = str(exc)[:200]

    try:
        import sageattention

        snap["sageattention"] = True
        snap["sageattention_version"] = getattr(sageattention, "__version__", "unknown")
    except Exception as exc:
        snap["sageattention"] = False
        snap["sageattention_note"] = str(exc)[:200]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    snap["hf_token_env"] = bool(token)
    snap["hf_token_file"] = token_file.is_file()
    if args.require_hf and not (token or token_file.is_file()):
        return _fail("缺少 HF_TOKEN（私有仓推拉需要）", snap, out)

    snap["ok"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    print(f"PREFLIGHT OK ({args.role}, backend={snap['backend']}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
