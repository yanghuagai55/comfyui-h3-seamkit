# SPDX-License-Identifier: GPL-3.0-or-later
"""一键开机脚本：硬门槛 → 拉云端分片 → 教师续跑 → 无缝开训。

  python train/boot.py --mode auto --repo <user>/<repo> \
      --prompts /content/work/prompts/prompts_run.jsonl \
      --deadline-utc 15:30 --teacher-budget-min 240 \
      --then-train --train-steps 300

  --mode teacher   只跑教师（带 shard 上传）
  --mode train     只跑训练（--pull-shards --auto-resume）
  --mode auto      教师跑完/到点后，若还剩 ≥ --train-min 分钟直接开训
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
COMFY_DIR = Path("/content/ComfyUI")
TOKEN_DEFAULT = "/content/work/hftoken.txt"


def _log(msg: str) -> None:
    print(f"[boot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _api_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ensure_teacher_nodes() -> None:
    """新号仅解包 seamkit_train.tgz（不含根 __init__.py）时，自动补一个最小节点注册入口。"""
    pkg_init = TRAIN_DIR.parent / "__init__.py"
    if pkg_init.is_file():
        return
    pkg_init.write_text(
        "from comfy_api.latest import ComfyExtension\n"
        "from .train.teacher_capture import (\n"
        "    MiniMaxH3ConditioningDump,\n"
        "    MiniMaxH3LatentDownsample2x,\n"
        "    MiniMaxH3TeacherLatentDump,\n"
        "    MiniMaxH3TeacherProbe,\n"
        ")\n"
        "from .train.dense_rope_patch import MiniMaxH3DenseRopePatch\n\n"
        "class H3TrainOnlyExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [\n"
        "            MiniMaxH3TeacherProbe,\n"
        "            MiniMaxH3ConditioningDump,\n"
        "            MiniMaxH3TeacherLatentDump,\n"
        "            MiniMaxH3LatentDownsample2x,\n"
        "            MiniMaxH3DenseRopePatch,\n"
        "        ]\n\n"
        "def comfy_entrypoint():\n"
        "    return H3TrainOnlyExtension()\n",
        encoding="utf-8",
    )
    _log(f"已补写训练专用节点入口 -> {pkg_init}")


def ensure_comfy(api: str, wait_s: int = 240) -> bool:
    _ensure_teacher_nodes()
    if _api_ok(api):
        return True
    _log("ComfyUI 未响应，拉起 main.py ...")
    log = open("/content/comfyui.log", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                     cwd=str(COMFY_DIR), stdout=log, stderr=log, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _api_ok(api):
            _log(f"ComfyUI 就绪（{time.time() - t0:.0f}s）")
            return True
        time.sleep(3)
    _log("ComfyUI 启动超时")
    return False


def preflight(role: str, min_vram: float) -> bool:
    rc = subprocess.call([sys.executable, str(TRAIN_DIR / "preflight.py"),
                          "--role", role, "--min-vram-gb", str(min_vram)])
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "train", "auto"], default="auto")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", default="teacher_cache_v1")
    ap.add_argument("--token-file", default=TOKEN_DEFAULT)
    ap.add_argument("--prompts", default="/content/work/prompts/prompts_run.jsonl")
    ap.add_argument("--out", default="/content/work/native")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--train-out", default="/content/work/lora_run")
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--deadline-utc", default="")
    ap.add_argument("--teacher-budget-min", type=float, default=0.0)
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--then-train", action="store_true")
    ap.add_argument("--train-min", type=float, default=35.0)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--min-vram-gb", type=float, default=70.0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--frames", default="", help="只训这些帧数桶，如 22 / 22,39")
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--lr", type=float, default=0.0, help="0=用 train_h3_lora 默认值(2e-4)；Phase B 建议 8e-5")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--sigma-min", type=float, default=0.0)
    ap.add_argument("--sigma-max", type=float, default=0.0)
    ap.add_argument("--sigmas", default="0.85,0.54,0.20")
    ap.add_argument("--anchor-prob", type=float, default=0.0)
    ap.add_argument("--lf-base-anchor", action="store_true")
    ap.add_argument("--lf-kernel", type=int, default=5)
    ap.add_argument("--init-lora", default="")
    ap.add_argument("--init-lora-hf", default="")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-clamp", type=float, default=10.0)
    ap.add_argument("--hf-prefix", default="lora_run")
    args = ap.parse_args()

    if not Path(args.token_file).is_file():
        _log(f"缺少 HF token 文件 {args.token_file}")
        return 2
    if Path(args.token_file).read_text(encoding="utf-8").strip() == "":
        _log(f"HF token 文件为空 {args.token_file}")
        return 2

    if not args.skip_preflight and not preflight("train" if args.mode == "train" else "teacher", args.min_vram_gb):
        _log("preflight 失败，退出（不做 T4/L4 降级）")
        return 3

    # 训练号：先把云端分片拉到本地 cache
    sys.path.insert(0, str(TRAIN_DIR))
    from hf_shards import ShardStore

    store = ShardStore(args.repo, args.prefix, args.token_file)
    if args.mode == "train":
        store.pull_all(args.cache, "/content/work/shard_dl")
        cmd = [sys.executable, str(TRAIN_DIR / "train_h3_lora.py"),
               "--comfy", str(COMFY_DIR),
               "--base", args.base, "--cache", args.cache, "--out", args.train_out,
               "--phase", args.phase, "--steps", str(args.train_steps),
               "--save-every", str(args.save_every), "--log-every", str(args.log_every),
               "--optim", "adamw8bit", "--grad-clamp", str(args.grad_clamp), "--auto-resume",
               "--pull-shards", "--hf-repo", args.repo, "--hf-prefix", args.hf_prefix,
               "--shard-prefix", args.prefix,
               "--token-file", args.token_file]
        if args.frames:
            cmd += ["--frames", args.frames]
        if args.lr > 0:
            cmd += ["--lr", str(args.lr)]
        if args.warmup > 0:
            cmd += ["--warmup", str(args.warmup)]
        if args.sigma_min > 0:
            cmd += ["--sigma-min", str(args.sigma_min)]
        if args.sigma_max > 0:
            cmd += ["--sigma-max", str(args.sigma_max)]
        if args.phase == "B" or args.anchor_prob > 0:
            cmd += ["--sigmas", args.sigmas]
        if args.anchor_prob > 0:
            cmd += ["--anchor-prob", str(args.anchor_prob)]
        if args.lf_base_anchor:
            cmd += ["--lf-base-anchor", "--lf-kernel", str(args.lf_kernel)]
        if args.init_lora:
            cmd += ["--init-lora", args.init_lora]
        if args.init_lora_hf:
            cmd += ["--init-lora-hf", args.init_lora_hf]
        _log("train: " + " ".join(cmd))
        return subprocess.call(cmd)

    if not ensure_comfy(args.api):
        return 4

    cmd = [sys.executable, str(TRAIN_DIR / "run_teacher.py"),
           "--repo", args.repo, "--prefix", args.prefix, "--token-file", args.token_file,
           "--prompts", args.prompts, "--out", args.out, "--cache", args.cache,
           "--shard-size", str(args.shard_size), "--api", args.api,
           "--base", args.base, "--train-out", args.train_out]
    if args.deadline_utc:
        cmd += ["--deadline-utc", args.deadline_utc]
    if args.teacher_budget_min:
        cmd += ["--budget-min", str(args.teacher_budget_min)]
    if args.mode == "auto" and args.then_train:
        cmd += ["--then-train", "--train-min", str(args.train_min), "--train-steps", str(args.train_steps)]
    _log("teacher: " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
