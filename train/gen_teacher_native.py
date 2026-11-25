# SPDX-License-Identifier: GPL-3.0-or-later
"""原生 1080p 满血教师：单遍 24 步 CFG=5.0 全稠密（官方 BF16 底模）。

与旧两遍教师不同，这里直接生成 **1920×1088 原生真值 z_native**，然后在同一张图里
做「VAE 解码 → 960×544 → 编码 → h3_upscaler 2.0×」合成真实糊底 z_up。训练目标：

    x_σ = (1-σ)·z_up + σ·ε,    v* = (x_σ - z_native) / σ

产物（每样本）：
  <out>/<id>/<id>.z_native.pt   # 原生 AV 潜空间（video + audio）
  <out>/<id>/<id>.z_up.pt       # 480p→2× 糊底（video）
  <out>/<id>/<id>.cond.pt       # 二采 conditioning（文本 + 参考图 + 音频条件）

用法：
  python train/gen_teacher_native.py --prompts ... --out /content/work/native \
      --steps 24 --cfg 5.0 --budget-min 330
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_UNET = "minimax_h3_ref2va_pruned_bf16.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_sol_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_LMS = "minimax_h3_lms_v1.0_r64_r.safetensors"
DEFAULT_UPSCALER = "h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"

EMPTY_PROMPT = " "


def _sage(model_link, args):
    return {
        "class_type": "MiniMaxH3DenseRopePatch",
        "inputs": {
            "model": model_link,
            "rope_hf_scale": float(args.rope_hf),
            "attn_temp_scale": float(args.attn_temp),
            "chunk_io_layers": True,
        },
    }


def build_graph(sample, args) -> dict:
    frames = int(sample.get("frames") or args.frames)
    task = str(sample.get("task") or ("ref2va" if sample.get("ref_image") else "t2va")).lower()
    key = str(sample["id"])
    width, height = 1920, 1088

    g: dict = {}
    g["31"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}}
    g["26"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "minimax", "device": "default"}}
    g["20"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.video_vae}}
    g["12"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.audio_vae}}
    g["9"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": sample["prompt"]}}
    g["10"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": EMPTY_PROMPT}}

    # LoRA：lms_strength=0（默认）时**不接 LMS**，教师真值保持官方 BF16 满血输出；
    # 只有显式 >0 才插 LoraLoaderModelOnly（4060 推理仍可另叠 LMS，但教师不背这个偏置）。
    if float(args.lms_strength) > 0.0:
        g["112"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": args.lms_lora, "strength_model": float(args.lms_strength), "model": ["31", 0]},
        }
        model_link = ["112", 0]
    else:
        model_link = ["31", 0]
    g["110"] = _sage(model_link, args)

    g["25"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(sample.get("seed") or 0)}}

    common = {
        "audio_mode": "native",
        "audio_denoise_strength": 1,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "ref_image_size": "max",
        "reference_video_policy": "official_2_to_15s",
        "allow_above_reference_area": True,
        "clip": ["26", 0],
        "video_vae": ["20", 0],
        "audio_vae": ["12", 0],
        "width": width,
        "height": height,
        "length": frames,
    }
    # 正向条件
    pos = dict(common)
    pos.update({"task_type": "Ref2VA" if task.startswith("ref") else "T2VA",
                "strict_prompt_tags": True, "prompt": ["9", 0]})
    if task.startswith("ref"):
        g["3"] = {"class_type": "LoadImage", "inputs": {"image": sample["ref_image"]}}
        pos["ref_images.ref_image_0"] = ["3", 0]
    g["8"] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": pos}

    # 负向条件（CFG）：空提示词、不带参考图
    neg = dict(common)
    neg.update({"task_type": "T2VA", "strict_prompt_tags": False, "prompt": ["10", 0]})
    g["108"] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": neg}

    g["29"] = {
        "class_type": "CFGGuider",
        "inputs": {"model": ["110", 0], "positive": ["8", 0], "negative": ["108", 0], "cfg": float(args.cfg)},
    }
    g["30"] = {
        "class_type": "MiniMaxH3DualClockSamplerT8",
        "inputs": {
            "steps": int(args.steps),
            "shift_video": float(args.shift_video),
            "shift_audio": float(args.shift_audio),
            "sampler_name": args.sampler,
            "scheduler": "native_flow",
            "model": ["110", 0],
            "av_latent": ["8", 1],
        },
    }
    g["93"] = {
        "class_type": "MiniMaxH3FirstPassSampler",
        "inputs": {
            "noise": ["25", 0],
            "guider": ["29", 0],
            "sampler": ["30", 1],
            "sigmas": ["30", 2],
            "latent_image": ["8", 1],
        },
    }

    # 落盘：z_native / cond
    g["204"] = {
        "class_type": "MiniMaxH3TeacherLatentDump",
        "inputs": {"latent": ["93", 0], "out_dir": args.out, "key": key, "tag": "z_native", "enabled": True},
    }
    g["205"] = {
        "class_type": "MiniMaxH3ConditioningDump",
        "inputs": {"conditioning": ["8", 0], "out_dir": args.out, "key": key, "enabled": True},
    }

    # 合成糊底：潜空间 2x 降采 -> 480p VAE 往返 -> 2D 上采样
    g["211"] = {
        "class_type": "MiniMaxH3LatentDownsample2x",
        "inputs": {"latent": ["93", 0], "enabled": True},
    }
    g["206"] = {"class_type": "VAEDecode", "inputs": {"samples": ["211", 0], "vae": ["20", 0]}}
    g["208"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["206", 0], "vae": ["20", 0]}}
    g["209"] = {
        "class_type": "MinimaxH3LatentUpscaler3D",
        "inputs": {
            "latent": ["208", 0],
            "model_name": args.upscaler,
            # COMFY_DYNAMICCOMBO_V3：选中的 key 直接作为输入值，嵌套字段用 `<id>.<field>`
            "mode": "scale by multiplier",
            "mode.scale": 2.0,
            "align": 32,
            "enable_temporal_chunking": True,
            "force_unload": False,
            "device": "cuda",
            "precision": "bf16",
        },
    }
    g["210"] = {
        "class_type": "MiniMaxH3TeacherLatentDump",
        "inputs": {"latent": ["209", 0], "out_dir": args.out, "key": key, "tag": "z_up", "enabled": True},
    }
    # 强制输出端：ComfyUI 以 output_node 为执行根；AVLatentSave 是已验证的 output 节点，
    # 串在 dump 之后可保证 z_up 分支（含 dump）一定被调度。
    g["212"] = {
        "class_type": "MiniMaxH3AVLatentSave",
        "inputs": {
            "latent": ["210", 0],
            "key": f"{key}_zup_force",
            "overwrite": True,
            "cache_path": "/content/work/zup_tmp",
        },
    }
    return g


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


def wait_done(api, prompt_id, timeout=2400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            hist = _get(api.rstrip("/") + f"/history/{prompt_id}")
        except urllib.error.URLError:
            time.sleep(3)
            continue
        if prompt_id in hist:
            item = hist[prompt_id]
            status = item.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"prompt error: {json.dumps(status.get('messages'), ensure_ascii=False)[:3000]}")
            if status.get("completed", True):
                return item
        time.sleep(3)
    raise TimeoutError(f"prompt {prompt_id} not finished in {timeout}s")


def expected(out_root: Path, key: str):
    # dump 节点直接写在 out_dir 根下（<out>/<key>.<tag>.pt），不是每样本子目录
    return [out_root / f"{key}.z_native.pt", out_root / f"{key}.z_up.pt", out_root / f"{key}.cond.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--sampler", default="dual_clock_euler")
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--budget-min", type=float, default=0.0, help="0=不限制；到点停止并保留已完成样本")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unet", default=DEFAULT_UNET)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    ap.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    ap.add_argument("--lms-lora", default=DEFAULT_LMS)
    ap.add_argument("--lms-strength", type=float, default=0.0,
                    help="0 = 不叠 LMS（默认，教师=官方 BF16 纯净上限）")
    ap.add_argument("--upscaler", default=DEFAULT_UPSCALER)
    ap.add_argument("--head-chunks", type=int, default=1)
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_gen_log.jsonl"

    # SM80（A100）没有 FP4 单元：nvfp4 CLIP 会走反量化权重路径，能跑但更慢。
    # 已实测 A100-SXM4-80GB 正常加载；若出现 dtype/算子报错，换
    # `--clip qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（同仓，32B int8 ≈ 32GB）。
    try:
        import torch as _torch

        if _torch.cuda.is_available() and "nvfp4" in args.clip.lower():
            cap = _torch.cuda.get_device_capability()
            if cap < (9, 0):
                print(
                    f"[warn] nvfp4 CLIP on SM{cap[0]}{cap[1]}：无原生 FP4，走 dequant 路径（已实测可跑）",
                    flush=True,
                )
    except Exception:
        pass

    samples = []
    with open(args.prompts, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]

    started = time.time()
    ok = fail = skip = 0
    for i, sample in enumerate(samples, 1):
        if args.budget_min and (time.time() - started) / 60.0 >= args.budget_min:
            print(f"[budget] {args.budget_min} min reached, stop at {i - 1}/{len(samples)}", flush=True)
            break
        key = str(sample["id"])
        files = expected(out_root, key)
        if args.resume and all(p.is_file() for p in files):
            skip += 1
            print(f"[{i}/{len(samples)}] {key}: skip (complete)", flush=True)
            continue
        rec = {"id": key, "task": sample.get("task"), "frames": sample.get("frames")}
        try:
            graph = build_graph(sample, args)
            if args.dry_run:
                (out_root / f"{key}.graph.json").write_text(
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
            sec = time.time() - t0
            ok += 1
            rec.update({"status": "ok", "sec": round(sec, 1), "prompt_id": pid})
            print(f"[{i}/{len(samples)}] {key}: OK {sec:.1f}s", flush=True)
        except Exception as exc:
            fail += 1
            rec.update({"status": "fail", "error": str(exc)[:2500]})
            print(f"[{i}/{len(samples)}] {key}: FAIL {exc}", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"done: ok={ok} fail={fail} skip={skip} elapsed={(time.time()-started)/60:.1f}min", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
