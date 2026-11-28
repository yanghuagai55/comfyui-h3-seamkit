# SPDX-License-Identifier: GPL-3.0-or-later
"""一键开机与每日接力总入口（v3 兼容魔搭 AMD MI300X 192GB ROCm 与 NVIDIA CUDA）。

典型两段接力用法：
  # Day 1（约 4.5~5 小时出首版效果）：跑前 20 条 39f 零残影教师 + 无损扩秩热启训 220 步
  python3 train/boot.py --mode auto --repo coolwindtang/minimax-h3-teacher-cache \
      --prefix teacher_cache_v3_action \
      --train-shard-prefixes teacher_cache_v1,teacher_cache_v3_action \
      --prompts train/prompts_sakuga_action_32.jsonl --limit 20 \
      --clip qwen3vl_32b_minimax_h3_bf16.safetensors \
      --teacher-budget-min 150 --then-train --train-steps 220 \
      --init-lora-hf final/minimax_h3_sol_sharp_v2_final.safetensors

  # Day 2+（每日接力续训）：自动跳过已完成的 20 条，补齐剩余 12 条 73f 教师，并从 step 220 续训至 420 步
  python3 train/boot.py --mode auto --repo coolwindtang/minimax-h3-teacher-cache \
      --prefix teacher_cache_v3_action \
      --train-shard-prefixes teacher_cache_v1,teacher_cache_v3_action \
      --prompts train/prompts_sakuga_action_32.jsonl \
      --clip qwen3vl_32b_minimax_h3_bf16.safetensors \
      --teacher-budget-min 140 --then-train --train-steps 420
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
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


def ensure_comfy(comfy_dir: Path, api: str, log_path: Path, wait_s: int = 240) -> bool:
    _ensure_teacher_nodes()
    if _api_ok(api):
        return True
    _log(f"ComfyUI 未响应，从 {comfy_dir} 拉起 main.py ...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--bf16-unet"],
        cwd=str(comfy_dir), stdout=log, stderr=log, env=env, start_new_session=True,
    )
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _api_ok(api):
            _log(f"ComfyUI 就绪（{time.time() - t0:.0f}s）")
            return True
        time.sleep(3)
    _log("ComfyUI 启动超时")
    return False


def preflight(role: str, min_vram: float, work_root: str, model_root: str) -> bool:
    cmd = [
        sys.executable, str(TRAIN_DIR / "preflight.py"),
        "--role", role, "--min-vram-gb", str(min_vram),
        "--model-root", model_root,
    ]
    if work_root:
        cmd += ["--work-root", work_root]
    return subprocess.call(cmd) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "train", "auto"], default="auto")
    ap.add_argument("--repo", default="", help="HF 私有仓；留空=纯本地模式（魔搭等 HF 不可达环境，产物只落本地/NAS）")
    ap.add_argument("--prefix", default="teacher_cache_v3_action")
    ap.add_argument("--train-shard-prefixes", default="",
                    help="训练时拉取的分片前缀（逗号分隔，如 teacher_cache_v1,teacher_cache_v3_action）")
    ap.add_argument("--token-file", default=TOKEN_DEFAULT)
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--work-root", default="/content/work")
    ap.add_argument("--model-root", default="/content/local_models")
    ap.add_argument("--prompts", default=str(TRAIN_DIR / "prompts_sakuga_action_32.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="本次最多生成前 N 条样本（首日 5h 设 20，接力设 0）")
    ap.add_argument("--out", default="/tmp/h3_native_raw")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--unet", default="minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--clip", default="qwen3vl_32b_minimax_h3_bf16.safetensors")
    ap.add_argument("--video-vae", default="minimax_h3_sol_vae_fp16.safetensors")
    ap.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    ap.add_argument("--upscaler", default="h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors")
    ap.add_argument("--teacher-steps", type=int, default=28)
    ap.add_argument("--teacher-cfg", type=float, default=5.5)
    ap.add_argument("--zup-mode", choices=["hybrid", "real_pass1", "downsample"], default="hybrid")
    ap.add_argument("--min-motion-lapvar", type=float, default=75.0)
    ap.add_argument("--best-of-k", type=int, default=1,
                    help="Best-of-K 黄金教师门禁：每条提示词运行 K 个种子并严选运动学平滑+高速锐度最优 Top-1（推荐 3）")
    ap.add_argument("--train-out", default="/content/work/lora_run_v3")
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--deadline-utc", default="")
    ap.add_argument("--teacher-budget-min", type=float, default=0.0)
    ap.add_argument("--shard-size", type=int, default=5)
    ap.add_argument("--then-train", action="store_true")
    ap.add_argument("--train-min", type=float, default=35.0)
    ap.add_argument("--train-steps", type=int, default=220)
    ap.add_argument("--min-vram-gb", type=float, default=70.0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--frames", default="22,39,73")
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--rank-attn", type=int, default=64)
    ap.add_argument("--rank-mlp", type=int, default=32)
    ap.add_argument("--train-refiner", action="store_true", help="开启 TokenRefiner 8 层训练（完整 208 层）")
    ap.add_argument("--rank-refiner", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sigma-min", type=float, default=0.14)
    ap.add_argument("--sigma-max", type=float, default=0.75)
    ap.add_argument("--sigmas", default="0.75,0.54,0.20")
    ap.add_argument("--anchor-prob", type=float, default=0.35)
    ap.add_argument("--lf-base-anchor", action="store_true", default=True)
    ap.add_argument("--no-lf-base-anchor", dest="lf_base_anchor", action="store_false")
    ap.add_argument("--lf-kernel", type=int, default=5)
    ap.add_argument("--motion-gate-tau", type=float, default=0.35)
    ap.add_argument("--grad-xy-weight", type=float, default=0.15)
    ap.add_argument("--vel-t-weight", type=float, default=0.20)
    ap.add_argument("--orth-weight", type=float, default=0.02)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--action-boost", type=float, default=2.5)
    ap.add_argument("--optim", choices=["auto", "adamw", "adamw8bit"], default="auto")
    ap.add_argument("--attn-backend", choices=["auto", "flex", "triton_bs", "sdpa_mask"], default="auto")
    ap.add_argument("--init-lora", default="")
    ap.add_argument("--init-lora-hf", default="")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-clamp", type=float, default=10.0)
    ap.add_argument("--hf-prefix", default="lora_run_v3")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.repo and (not Path(args.token_file).is_file()
                      or not Path(args.token_file).read_text(encoding="utf-8").strip()):
        _log(f"缺少或空的 HF token 文件: {args.token_file}")
        return 2
    if not args.repo:
        _log("本地模式：未指定 --repo，教师分片与断点只落本地磁盘（不自愈上云）")

    if not args.skip_preflight and not preflight(
        "train" if args.mode == "train" else "teacher",
        args.min_vram_gb, args.work_root, args.model_root,
    ):
        _log("preflight 未通过，中止执行")
        return 3

    shard_prefixes = args.train_shard_prefixes.strip() or args.prefix
    if args.mode == "train":
        cmd = [
            sys.executable, str(TRAIN_DIR / "train_h3_lora.py"),
            "--comfy", args.comfy,
            "--base", args.base, "--cache", args.cache, "--out", args.train_out,
            "--phase", args.phase, "--steps", str(args.train_steps),
            "--rank-attn", str(args.rank_attn), "--rank-mlp", str(args.rank_mlp),
            "--lr", str(args.lr), "--warmup", str(args.warmup),
            "--sigma-min", str(args.sigma_min), "--sigma-max", str(args.sigma_max),
            "--sigmas", args.sigmas, "--anchor-prob", str(args.anchor_prob),
            "--motion-gate-tau", str(args.motion_gate_tau),
            "--grad-xy-weight", str(args.grad_xy_weight),
            "--vel-t-weight", str(args.vel_t_weight),
            "--orth-weight", str(args.orth_weight),
            "--grad-accum", str(args.grad_accum),
            "--ema-decay", str(args.ema_decay),
            "--action-boost", str(args.action_boost),
            "--optim", args.optim, "--attn-backend", args.attn_backend, "--block-size", "64",
            "--save-every", str(args.save_every), "--log-every", str(args.log_every),
            "--grad-clamp", str(args.grad_clamp), "--auto-resume",
        ]
        if args.train_refiner:
            cmd += ["--train-refiner", "--rank-refiner", str(args.rank_refiner)]
        if args.repo:
            cmd += ["--pull-shards", "--hf-repo", args.repo, "--hf-prefix", args.hf_prefix,
                    "--shard-prefix", shard_prefixes, "--token-file", args.token_file]
        if args.frames:
            cmd += ["--frames", args.frames]
        if args.lf_base_anchor:
            cmd += ["--lf-base-anchor", "--lf-kernel", str(args.lf_kernel)]
        if args.init_lora:
            cmd += ["--init-lora", args.init_lora]
        if args.init_lora_hf:
            cmd += ["--init-lora-hf", args.init_lora_hf]
        _log("train: " + " ".join(cmd))
        return subprocess.call(cmd)

    comfy_log = Path(args.work_root) / "logs" / "comfyui.log"
    if not ensure_comfy(Path(args.comfy), args.api, comfy_log):
        return 4

    cmd = [
        sys.executable, str(TRAIN_DIR / "run_teacher.py"),
        "--repo", args.repo, "--prefix", args.prefix, "--token-file", args.token_file,
        "--comfy", args.comfy, "--prompts", args.prompts, "--out", args.out, "--cache", args.cache,
        "--unet", args.unet, "--clip", args.clip,
        "--video-vae", args.video_vae, "--audio-vae", args.audio_vae, "--upscaler", args.upscaler,
        "--steps", str(args.teacher_steps), "--cfg", str(args.teacher_cfg),
        "--zup-mode", args.zup_mode, "--min-motion-lapvar", str(args.min_motion_lapvar),
        "--best-of-k", str(args.best_of_k),
        "--shard-size", str(args.shard_size), "--api", args.api,
        "--base", args.base, "--train-out", args.train_out, "--train-hf-prefix", args.hf_prefix,
        "--train-shard-prefixes", shard_prefixes, "--train-frames", args.frames,
        "--rank-attn", str(args.rank_attn), "--rank-mlp", str(args.rank_mlp),
        "--vel-t-weight", str(args.vel_t_weight), "--orth-weight", str(args.orth_weight),
        "--grad-accum", str(args.grad_accum), "--ema-decay", str(args.ema_decay),
        "--lr", str(args.lr), "--optim", args.optim, "--attn-backend", args.attn_backend,
    ]
    if args.train_refiner:
        cmd += ["--train-refiner", "--rank-refiner", str(args.rank_refiner)]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.init_lora:
        cmd += ["--init-lora", args.init_lora]
    if args.init_lora_hf:
        cmd += ["--init-lora-hf", args.init_lora_hf]
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
