# SPDX-License-Identifier: GPL-3.0-or-later
"""教师轨迹批量生成（在 ComfyUI 进程外，走 /prompt HTTP API）。

为什么不在节点里循环
--------------------
一个 prompt 一条图，ComfyUI 自己负责权重换入换出、VAE 释放、OOM 重试；
我们只做三件事：改参数 -> 提交 -> 检查落盘。断连后 `--resume` 按目录跳过。

用法（远端 /content/ComfyUI 已起，127.0.0.1:8188）：
  python train/gen_teachers.py --prompts train/prompts_teacher.jsonl \
      --out /content/work/teacher_raw --limit 1 --frames 22

产物（每个样本一个目录）：
  <out>/<id>/<id>.step0_s0.8200.pt   # x / v（探针）
  <out>/<id>/<id>.cond.pt            # 二采 conditioning
  <out>/<id>/<id>.z_up.pt            # 上采样后的二采输入
  <out>/<id>/<id>.z_clean.pt         # 3 步稠密教师终点
  <out>/_gen_log.jsonl               # 每样本一行状态
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "route3" / "workflows" / "local_dualclock_api.json"

# 与 4060 部署一致的默认值；可用命令行覆盖
DEFAULT_UNET = "minimax_h3_ref2va_pruned_bf16.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_ACCEL_LORA = "pika_h3_flex_fl2va_v1_bf16_r.safetensors"
DEFAULT_LMS = "minimax_h3_lms_v1.0_r64_r.safetensors"
DEFAULT_UPSCALER = "h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"


def _node(graph, nid):
    if nid not in graph:
        raise KeyError(f"template missing node {nid}")
    return graph[nid]


def _drop(graph, *nids):
    for nid in nids:
        graph.pop(nid, None)


def build_graph(sample, args) -> dict:
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    # 去掉解码/预览/8GB 低显存节点；教师只要 latent
    _drop(graph, "16", "77", "57", "60", "34", "35", "106")

    frames = int(sample.get("frames") or args.frames)
    seconds = frames / 24.0
    task = str(sample.get("task") or ("ref2va" if sample.get("ref_image") else "t2va")).lower()
    key = str(sample["id"])

    _node(graph, "31")["inputs"]["unet_name"] = args.unet
    _node(graph, "26")["inputs"]["clip_name"] = args.clip
    _node(graph, "20")["inputs"]["vae_name"] = args.video_vae
    _node(graph, "12")["inputs"]["vae_name"] = args.audio_vae
    _node(graph, "25")["inputs"]["noise_seed"] = int(sample.get("seed") or 0)

    # LoRA 链：底模 -> LMS（两采共用）-> 一采再挂 pika 加速；二采纯稠密
    graph["112"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"lora_name": args.lms_lora, "strength_model": float(args.lms_strength), "model": ["31", 0]},
    }
    _node(graph, "22")["inputs"].update(
        {"lora_name": args.accel_lora, "strength_model": float(args.accel_strength), "model": ["112", 0]}
    )
    _node(graph, "17")["inputs"]["model"] = ["22", 0]
    _node(graph, "19")["inputs"]["model"] = ["17", 0]

    def _sage_patch(model_link):
        return {
            "class_type": "MiniMaxH3ChunkedSageAttentionPatch",
            "inputs": {
                "model": model_link,
                "head_chunks": int(args.head_chunks),
                "out_proj_mode": "exact",
                "token_chunk_size": 8192,
                "chunk_io_layers": True,
                "attention_mode": "dense_sage",
                "sol_tau": 1.3,
                "sol_token_aug": 64,
                "sol_topk_ratio": 0.0,
                "sol_sink_mode": "exact_kv_and_rows",
                "sol_min_tokens": 12288,
                "sol_dense_blocks": "",
                "sol_max_sigma": 0.95,
                "sol_min_sigma": 0.0,
                "rope_hf_scale": float(args.rope_hf),
                "attn_temp_scale": float(args.attn_temp),
                "sol_low_sigma_tau": 0.0,
            },
        }

    graph["110"] = _sage_patch(["19", 0])   # 一采（含 pika）
    graph["114"] = _sage_patch(["112", 0])  # 二采（纯底模 + LMS，教师轨迹）

    # 一采：4 步；二采：3 步 ManualSigmas（教师全稠密）
    _node(graph, "30")["inputs"].update({"steps": 4, "shift_video": 12, "shift_audio": 3, "model": ["110", 0]})
    _node(graph, "38")["inputs"].update({"steps": 3, "shift_video": 6.5, "shift_audio": 3, "model": ["200", 0]})
    graph["39"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.82, 0.50, 0.18, 0.0"}}

    # 探针只挂二采；一采走 110 原样
    graph["200"] = {
        "class_type": "MiniMaxH3TeacherProbe",
        "inputs": {
            "model": ["114", 0],
            "out_dir": args.out,
            "key": key,
            "sigmas": "0.82, 0.50, 0.18",
            "sigma_tol": 0.01,
            "enabled": True,
        },
    }
    graph["201"] = {
        "class_type": "MiniMaxH3ConditioningDump",
        "inputs": {"conditioning": ["41", 0], "out_dir": args.out, "key": key, "enabled": True},
    }
    graph["202"] = {
        "class_type": "MiniMaxH3TeacherLatentDump",
        "inputs": {"latent": ["40", 2], "out_dir": args.out, "key": key, "tag": "z_up", "enabled": True},
    }
    graph["203"] = {
        "class_type": "MiniMaxH3TeacherLatentDump",
        "inputs": {"latent": ["40", 0], "out_dir": args.out, "key": key, "tag": "z_clean", "enabled": True},
    }

    _node(graph, "40")["inputs"].update(
        {
            "model": ["38", 0],
            "conditioning": ["201", 0],
            "latent": ["93", 1],
            "noise": ["25", 0],
            "sampler": ["38", 1],
            "sigmas": ["39", 0],
            "plan": ["102", 0],
            "pass2_plan": ["107", 0],
        }
    )

    _node(graph, "102")["inputs"].update(
        {
            "total_seconds": round(seconds, 6),
            "target_segment_seconds": 16.0,
            "first_megapixels": 0.5,
            "second_megapixels": 2.0,
            "aspect_ratio": "16:9 (Widescreen)",
            "green_screen_mode": False,
            "use_cache": False,
            "boundary_search_frames": 0,
        }
    )
    _node(graph, "107")["inputs"].update(
        {
            "model_name": args.upscaler,
            "seam_mode": "auto",
            "overlap_frames": 0,
            "seam_blend": False,
            "cfg": 1,
            "route3_enabled": False,
            "upscale_pad_tokens": 3,
            "precision": "bf16",
            "dump_latents": False,
        }
    )

    # 提示词
    _node(graph, "9")["inputs"]["value"] = sample["prompt"]

    # 条件：Ref2VA 带参考图，T2VA 不带
    for nid in ("8", "41"):
        cond = _node(graph, nid)["inputs"]
        if task.startswith("ref"):
            cond["task_type"] = "Ref2VA"
            cond["ref_images.ref_image_0"] = ["3", 0]
        else:
            cond["task_type"] = "T2VA"
            cond.pop("ref_images.ref_image_0", None)
    if task.startswith("ref"):
        if not sample.get("ref_image"):
            raise ValueError(f"{key}: Ref2VA sample missing ref_image")
        _node(graph, "3")["inputs"]["image"] = sample["ref_image"]
    else:
        _drop(graph, "3")

    return graph


def _post(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def submit(api, graph):
    res = _post(api.rstrip("/") + "/prompt", {"prompt": graph})
    if "prompt_id" not in res:
        raise RuntimeError(f"submit failed: {res}")
    return res["prompt_id"]


def wait_done(api, prompt_id, timeout=1200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            hist = _get(api.rstrip("/") + f"/history/{prompt_id}")
        except urllib.error.URLError:
            time.sleep(2)
            continue
        if prompt_id in hist:
            item = hist[prompt_id]
            status = item.get("status") or {}
            if status.get("status_str") == "error" or status.get("completed") is False:
                msgs = status.get("messages") or []
                raise RuntimeError(f"prompt error: {json.dumps(msgs, ensure_ascii=False)[:2000]}")
            if status.get("completed", True):
                return item
        time.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} not finished in {timeout}s")


def expected_files(out_root: Path, key: str):
    d = out_root / key
    return [
        d / f"{key}.step0_s0.8200.pt",
        d / f"{key}.step1_s0.5000.pt",
        d / f"{key}.step2_s0.1800.pt",
        d / f"{key}.cond.pt",
        d / f"{key}.z_up.pt",
        d / f"{key}.z_clean.pt",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=22, choices=[22, 39, 73, 243])
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unet", default=DEFAULT_UNET)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    ap.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    ap.add_argument("--accel-lora", default=DEFAULT_ACCEL_LORA)
    ap.add_argument("--accel-strength", type=float, default=1.0)
    ap.add_argument("--lms-lora", default=DEFAULT_LMS)
    ap.add_argument("--lms-strength", type=float, default=0.18)
    ap.add_argument("--upscaler", default=DEFAULT_UPSCALER)
    ap.add_argument("--head-chunks", type=int, default=1)
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_gen_log.jsonl"

    samples = []
    with open(args.prompts, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]

    ok = fail = skip = 0
    for i, sample in enumerate(samples, 1):
        key = str(sample["id"])
        files = expected_files(out_root, key)
        if args.resume and all(p.is_file() for p in files):
            skip += 1
            print(f"[{i}/{len(samples)}] {key}: skip (complete)", flush=True)
            continue
        try:
            graph = build_graph(sample, args)
            if args.dry_run:
                Path(out_root / f"{key}.graph.json").write_text(
                    json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[{i}/{len(samples)}] {key}: dry-run graph written", flush=True)
                ok += 1
                continue
            t0 = time.time()
            pid = submit(args.api, graph)
            wait_done(args.api, pid, timeout=args.timeout)
            missing = [p.name for p in files if not p.is_file()]
            if missing:
                raise RuntimeError(f"missing outputs: {missing}")
            elapsed = time.time() - t0
            ok += 1
            print(f"[{i}/{len(samples)}] {key}: OK {elapsed:.1f}s", flush=True)
            rec = {"id": key, "status": "ok", "sec": round(elapsed, 2), "prompt_id": pid,
                   "task": sample.get("task"), "frames": sample.get("frames")}
        except Exception as exc:
            fail += 1
            print(f"[{i}/{len(samples)}] {key}: FAIL {exc}", flush=True)
            rec = {"id": key, "status": "fail", "error": str(exc)[:2000],
                   "task": sample.get("task"), "frames": sample.get("frames")}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"done: ok={ok} fail={fail} skip={skip}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
