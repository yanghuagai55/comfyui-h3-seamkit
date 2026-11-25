# SPDX-License-Identifier: GPL-3.0-or-later
"""Colab 端常驻 watchdog：教师循环/ComfyUI 掉线自动拉起。

  nohup python train/watchdog.py --deadline-utc 14:50 --interval 300 \
      > /content/work/watchdog.log 2>&1 &

规则（每 interval 秒）：
  1. 训练进程在跑 → 什么都不做（训练阶段独占显存）。
  2. 教师进程不在，且未过 deadline → 用同一套参数重启 run_teacher.py。
  3. 教师进程在但 ComfyUI /system_stats 连续 3 次不通 → 重新拉起 main.py。
  4. 过了 deadline + 10 分钟 → 退出。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
REPO = "coolwindtang/minimax-h3-teacher-cache"


def log(msg: str) -> None:
    print(f"[wd {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pgrep(pattern: str) -> bool:
    rc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return rc.returncode == 0 and bool(rc.stdout.strip())


def api_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def deadline_at(text: str) -> datetime:
    hh, mm = (int(x) for x in text.split(":"))
    now = datetime.now(timezone.utc)
    dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return dt if dt > now else dt + timedelta(days=1)


def start_comfy() -> None:
    logf = open("/content/comfyui.log", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                     cwd="/content/ComfyUI", stdout=logf, stderr=logf, start_new_session=True)
    log("ComfyUI 重新拉起")


def start_teacher(args) -> None:
    cmd = [
        sys.executable, str(TRAIN_DIR / "run_teacher.py"),
        "--repo", args.repo, "--prefix", args.prefix, "--token-file", args.token_file,
        "--prompts", args.prompts, "--out", args.out, "--cache", args.cache,
        "--shard-size", str(args.shard_size), "--api", args.api,
        "--base", args.base, "--train-out", args.train_out,
        "--deadline-utc", args.deadline_utc, "--then-train",
        "--train-min", str(args.train_min), "--train-steps", str(args.train_steps),
    ]
    logf = open("/content/work/teacher_loop.log", "a", encoding="utf-8")
    subprocess.Popen(cmd, cwd=str(TRAIN_DIR.parent), stdout=logf, stderr=logf, start_new_session=True)
    log("run_teacher 重新拉起: " + " ".join(cmd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--prefix", default="teacher_cache_v1")
    ap.add_argument("--token-file", default="/content/work/hftoken.txt")
    ap.add_argument("--prompts", default="/content/work/prompts/prompts_run.jsonl")
    ap.add_argument("--out", default="/content/work/native")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--train-out", default="/content/work/lora_run")
    ap.add_argument("--train-min", type=float, default=40.0)
    ap.add_argument("--train-steps", type=int, default=250)
    args = ap.parse_args()

    deadline = deadline_at(args.deadline_utc)
    log(f"armed: deadline={deadline.isoformat()} interval={args.interval}s")
    fails = 0
    while True:
        now = datetime.now(timezone.utc)
        if now > deadline + timedelta(minutes=10):
            log("past deadline, exit")
            return 0
        training = pgrep(r"train_h3_lora[.]py")
        teacher = pgrep(r"run_teacher[.]py")
        if training:
            fails = 0
            time.sleep(args.interval)
            continue
        if not teacher:
            if now <= deadline:
                log("teacher 进程不在，重启")
                start_teacher(args)
                time.sleep(30)
            else:
                log("teacher 已结束且无训练进程（可能时间不足跳过）")
        else:
            if api_ok(args.api):
                fails = 0
            else:
                fails += 1
                log(f"ComfyUI 无响应 ({fails}/3)")
                if fails >= 3:
                    start_comfy()
                    fails = 0
                    time.sleep(40)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
