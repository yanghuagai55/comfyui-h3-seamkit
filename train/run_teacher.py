# SPDX-License-Identifier: GPL-3.0-or-later
"""教师生成 + 每 10 条打 shard 上传 + 剩余时间无缝开训。

流程
----
1. 从私有 HF Repo 拉 `manifest.json`，`done_ids` 里的提示词直接跳过（换号续跑）。
2. 本地已完整的 raw（上次中断留下的）先打包上传，再继续生成。
3. 每凑满 `--shard-size` 条：`pack_one` → `shard_XXX.tar.zst`（.tmp 原子改名）→
   上传 shard → 更新 manifest；成功后删掉 raw 释放磁盘。
4. 到 `--budget-min` 或 `--deadline-utc` 停止，冲掉不满一片的尾巴。
5. `--then-train`：还剩 ≥ `--train-min` 分钟就 subprocess 起 `train_h3_lora.py
   --auto-resume --pull-shards`，把这个号的积分直接烧进 Phase A。

用法：
  python train/run_teacher.py --repo <user>/<repo> --prompts /content/work/prompts/prompts_run.jsonl \
      --out /content/work/native --cache /content/work/teacher_cache --shard-size 10 \
      --budget-min 250 --deadline-utc 15:30 --then-train --train-steps 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import gen_teacher_native as G  # noqa: E402
from hf_shards import DEFAULT_PREFIX, DEFAULT_TOKEN_FILE, ShardStore, make_shard  # noqa: E402
from pack_teachers import pack_one  # noqa: E402

RAW_TAGS = ("z_native", "z_up", "cond")


def _log(msg: str) -> None:
    print(f"[teacher {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _raw_ready(out_root: Path, key: str) -> bool:
    return all((out_root / f"{key}.{t}.pt").is_file() for t in RAW_TAGS)


def _drop_raw(out_root: Path, key: str) -> int:
    n = 0
    for t in RAW_TAGS:
        p = out_root / f"{key}.{t}.pt"
        if p.is_file():
            p.unlink()
            n += 1
    zup_tmp = Path("/content/work/zup_tmp") / f"{key}_zup_force.pt"
    if zup_tmp.is_file():
        zup_tmp.unlink()
        n += 1
    return n


def _deadline_utc(text: str) -> datetime | None:
    if not text:
        return None
    hh, mm = (int(x) for x in text.split(":"))
    now = datetime.now(timezone.utc)
    dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if dt <= now:
        dt += timedelta(days=1)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    # 生成
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="/content/work/native")
    ap.add_argument("--cache", default="/content/work/teacher_cache")
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--budget-min", type=float, default=0.0)
    ap.add_argument("--deadline-utc", default="")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--unet", default=G.DEFAULT_UNET)
    ap.add_argument("--clip", default=G.DEFAULT_CLIP)
    ap.add_argument("--video-vae", default=G.DEFAULT_VIDEO_VAE)
    ap.add_argument("--audio-vae", default=G.DEFAULT_AUDIO_VAE)
    ap.add_argument("--upscaler", default=G.DEFAULT_UPSCALER)
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    # 云端
    ap.add_argument("--repo", default="")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    # 无缝开训
    ap.add_argument("--then-train", action="store_true")
    ap.add_argument("--train-min", type=float, default=35.0, help="至少剩这么多分钟才开训")
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--train-frames", default="22", help="无缝开训时使用的帧数桶（80GB 建议 22 或 22,39；96GB 可 22,39,73）")
    ap.add_argument("--train-out", default="/content/work/lora_run")
    ap.add_argument("--base", default="/content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    args = ap.parse_args()

    out_root = Path(args.out)
    cache = Path(args.cache)
    out_root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "shard_gen_log.jsonl"

    samples = []
    with open(args.prompts, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    prompts = {str(s["id"]): s for s in samples}

    store = ShardStore(args.repo, args.prefix, args.token_file) if args.repo else None
    done = store.done_ids() if store else set()
    pending = [s for s in samples if str(s["id"]) not in done]
    _log(f"prompts={len(samples)} cloud_done={len(done)} pending={len(pending)}")

    # 上次中断留下的完整 raw 先入队
    ready = [str(s["id"]) for s in pending if _raw_ready(out_root, str(s["id"]))]
    queue = [s for s in pending if str(s["id"]) not in ready]
    _log(f"adopt raw={len(ready)} to_generate={len(queue)}")

    gen_args = types.SimpleNamespace(
        unet=args.unet, clip=args.clip, video_vae=args.video_vae, audio_vae=args.audio_vae,
        lms_lora=G.DEFAULT_LMS, lms_strength=0.0, upscaler=args.upscaler,
        rope_hf=args.rope_hf, attn_temp=args.attn_temp, out=str(out_root),
        cfg=args.cfg, steps=args.steps, shift_video=12.0, shift_audio=3.0,
        sampler="dual_clock_euler", frames=args.frames,
    )

    started = time.time()
    deadline = _deadline_utc(args.deadline_utc)
    shard_idx = len((store.fetch_manifest()["shards"] if store else []))

    def minutes_left() -> float:
        left = float("inf")
        if args.budget_min:
            left = min(left, args.budget_min - (time.time() - started) / 60.0)
        if deadline is not None:
            left = min(left, (deadline - datetime.now(timezone.utc)).total_seconds() / 60.0)
        return left

    def flush(ids: list[str]) -> int:
        nonlocal shard_idx
        ids = [i for i in ids if (cache / f"{i}.pt").is_file() or _raw_ready(out_root, i)]
        if not ids:
            return 0
        packed = []
        for key in ids:
            if (cache / f"{key}.pt").is_file():
                packed.append(key)
                continue
            try:
                rec = pack_one(out_root, key, prompts)
                tmp = cache / f"{key}.pt.tmp"
                import torch

                torch.save(rec, tmp)
                tmp.replace(cache / f"{key}.pt")
                packed.append(key)
            except Exception as exc:
                _log(f"pack {key} FAIL: {exc}")
        if not packed:
            return 0
        if store is None:
            _log(f"no repo, {len(packed)} packed locally only")
            for key in packed:
                _drop_raw(out_root, key)
            return len(packed)
        name = f"shard_{shard_idx:03d}.tar.zst"
        tar = make_shard(cache, packed, name)
        if tar is None:
            return 0
        try:
            store.push_shard(tar, packed)
            shard_idx += 1
            for key in packed:
                _drop_raw(out_root, key)
            tar.unlink(missing_ok=True)
            _log(f"shard {name} ok ({len(packed)} samples), cache={len(list(cache.glob('*.pt')))}")
        except Exception as exc:
            _log(f"upload {name} FAIL（保留本地 tar/raw，下次重试）: {exc}")
            return 0
        return len(packed)

    ok = fail = 0
    cursor = 0
    while ready and len(ready) >= args.shard_size:
        flush(ready[: args.shard_size])
        ready = ready[args.shard_size:]

    for sample in queue:
        if minutes_left() <= 0:
            _log(f"budget/deadline reached, stop at {ok} generated, {len(ready)} waiting to flush")
            break
        key = str(sample["id"])
        if _raw_ready(out_root, key):
            ready.append(key)
        else:
            rec = {"id": key, "task": sample.get("task"), "frames": sample.get("frames")}
            t0 = time.time()
            try:
                graph = G.build_graph(sample, gen_args)
                pid = G.submit(args.api, graph)
                G.wait_done(args.api, pid, timeout=args.timeout)
                missing = [p.name for p in G.expected(out_root, key) if not p.is_file()]
                if missing:
                    raise RuntimeError(f"missing outputs: {missing}")
                rec.update({"status": "ok", "sec": round(time.time() - t0, 1), "prompt_id": pid})
                ok += 1
                ready.append(key)
                _log(f"{key} OK {rec['sec']}s  ({len(ready)}/{args.shard_size})")
            except Exception as exc:
                fail += 1
                rec.update({"status": "fail", "error": str(exc)[:2000]})
                _log(f"{key} FAIL {exc}")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if len(ready) >= args.shard_size:
            flush(ready[: args.shard_size])
            ready = ready[args.shard_size:]

    if ready:
        flush(ready)

    total_cloud = len(store.fetch_manifest()["shards"]) if store else 0
    _log(f"teacher done: ok={ok} fail={fail} local={len(list(cache.glob('*.pt')))} shards={total_cloud}")
    (out_root / "teacher_summary.json").write_text(json.dumps({
        "ok": ok, "fail": fail, "cloud_shards": total_cloud,
        "local_samples": len(list(cache.glob("*.pt"))),
        "elapsed_min": round((time.time() - started) / 60.0, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 无缝开训 ----------------
    if args.then_train and args.train_steps > 0:
        left = minutes_left()
        _log(f"then-train: minutes_left={left:.0f} (need {args.train_min:.0f})")
        if left >= args.train_min:
            # 训练要独占 40GB 基座：先让 ComfyUI 卸模型（否则 60GB 占用直接 OOM）
            try:
                import urllib.request

                req = urllib.request.Request(
                    args.api.rstrip("/") + "/free",
                    data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=60)
                _log("ComfyUI 已卸载模型，释放显存")
                time.sleep(8)
            except Exception as exc:
                _log(f"ComfyUI /free 调用失败（继续，训练前会再抢显存）: {exc}")
            cmd = [
                sys.executable, str(TRAIN_DIR / "train_h3_lora.py"),
                "--comfy", "/content/ComfyUI",
                "--base", args.base, "--cache", str(cache), "--out", args.train_out,
                "--phase", "A", "--steps", str(args.train_steps), "--save-every", "50",
                "--log-every", "5", "--optim", "adamw8bit", "--auto-resume", "--grad-clamp", "10",
            ]
            if args.train_frames:
                cmd += ["--frames", args.train_frames]
            if args.repo:
                cmd += ["--pull-shards", "--hf-repo", args.repo, "--hf-prefix", "lora_run",
                        "--shard-prefix", args.prefix, "--token-file", args.token_file]
            _log("launch: " + " ".join(cmd))
            return subprocess.call(cmd)
        _log("剩余时间不足，跳过开训")
    return 0


if __name__ == "__main__":
    sys.exit(main())
