# SPDX-License-Identifier: GPL-3.0-or-later
"""Colab 端纯训练看门狗：只在 train_h3_lora.py 意外退出时按锁定参数自动续训。

用法（Colab 后台）：
  nohup python3 train/train_watchdog.py --steps 600 --frames 22 \
      --hf-repo coolwindtang/minimax-h3-teacher-cache --hf-prefix lora_run \
      > /content/work/lora_run/logs/train_watchdog.log 2>&1 &

安全护栏：
  1. 训练进程活着 → 什么都不做。
  2. 已达到目标 steps 或超过 --deadline-utc → 正常退出。
  3. 进程消失 → 用固定锁定参数（真稀疏 flex + --grad-clamp 10 + --auto-resume + --hf-repo）重启。
  4. 连续 3 次在 180 秒内闪退 → 自动停机，绝不死循环烧卡。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
COMFY_DIR = Path("/content/ComfyUI")
REPO_DEFAULT = "coolwindtang/minimax-h3-teacher-cache"


def log(msg: str) -> None:
    print(f"[train-wd {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_training_running() -> bool:
    rc = subprocess.run(["pgrep", "-f", r"train_h3_lora[.]py"], capture_output=True, text=True)
    return rc.returncode == 0 and bool(rc.stdout.strip())


def current_step(out_dir: Path) -> int:
    metrics = out_dir / "logs" / "metrics.jsonl"
    if not metrics.is_file():
        return 0
    last = 0
    try:
        for line in metrics.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last = max(last, int(obj.get("step", 0)))
            except Exception:
                continue
    except Exception:
        pass
    return last


def parse_deadline(text: str) -> datetime | None:
    if not text.strip():
        return None
    hh, mm = (int(x) for x in text.strip().split(":"))
    now = datetime.now(timezone.utc)
    dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return dt if dt > now else dt + timedelta(days=1)


def launch_train(args) -> None:
    out_dir = Path(args.out)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_sparse.log"
    cmd = [
        sys.executable, str(TRAIN_DIR / "train_h3_lora.py"),
        "--comfy", str(COMFY_DIR),
        "--base", args.base,
        "--cache", args.cache,
        "--out", args.out,
        "--phase", args.phase,
        "--steps", str(args.steps),
        "--save-every", str(args.save_every),
        "--log-every", str(args.log_every),
        "--optim", "adamw8bit",
        "--grad-clamp", str(args.grad_clamp),
        "--auto-resume",
        "--hf-repo", args.hf_repo,
        "--hf-prefix", args.hf_prefix,
        "--token-file", args.token_file,
    ]
    if args.frames:
        cmd += ["--frames", args.frames]
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[train-wd {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] auto-resume launch\n")
        fh.flush()
        subprocess.Popen(cmd, cwd=str(TRAIN_DIR.parent), stdout=fh, stderr=fh, start_new_session=True)
    log("已自动拉起续训: " + " ".join(cmd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--out", default="/content/work/lora_run")
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--frames", default="22")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-clamp", type=float, default=10.0)
    ap.add_argument("--hf-repo", default=REPO_DEFAULT)
    ap.add_argument("--hf-prefix", default="lora_run")
    ap.add_argument("--token-file", default="/content/work/hftoken.txt")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--deadline-utc", default="")
    ap.add_argument("--max-rapid-restarts", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    deadline = parse_deadline(args.deadline_utc)
    log(f"armed: target_steps={args.steps} frames={args.frames} interval={args.interval}s "
        f"deadline={deadline.isoformat() if deadline else 'none'}")

    rapid_restarts = 0
    last_launch_ts = 0.0

    while True:
        now = datetime.now(timezone.utc)
        if deadline is not None and now >= deadline:
            log("到达 deadline_utc，看门狗退出")
            return 0

        step = current_step(out_dir)
        if step >= args.steps:
            log(f"已达成目标步数 step={step}/{args.steps}，看门狗退出")
            return 0

        if is_training_running():
            if last_launch_ts and (time.time() - last_launch_ts) > 300:
                rapid_restarts = 0
            time.sleep(args.interval)
            continue

        now_s = time.time()
        if last_launch_ts and (now_s - last_launch_ts) < 240:
            rapid_restarts += 1
        else:
            rapid_restarts = 1

        if rapid_restarts > args.max_rapid_restarts:
            log(f"ALERT: 连续 {rapid_restarts} 次短时间闪退，停止自动重启以避免死循环烧卡！")
            return 1

        log(f"检测到训练进程不在（当前 step={step}/{args.steps}，连续重启计数={rapid_restarts}/{args.max_rapid_restarts}）")
        launch_train(args)
        last_launch_ts = time.time()
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    sys.exit(main())
