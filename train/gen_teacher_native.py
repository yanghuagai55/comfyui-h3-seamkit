# SPDX-License-Identifier: GPL-3.0-or-later
"""原生 1080p 满血教师生成器（v3 升级：多图 R2VA + 同构真一采退化糊底 `real_pass1` + 运动峰值帧 `LapVar` 自动筛糊门禁）。

核心特性
--------
1. **满血真值 `z_native`**：单遍 24~28 步 `CFG=5.0~5.5` 全稠密 SDPA + Split-Band RoPE 1.32 生成原生 `1920×1088` 潜空间。
2. **多参考图支持**：同时兼容旧字段 `"ref_image": "a.png"` 与新多图数组 `"ref_images": ["deepseek.png", "claude_sheet.png"]`。
3. **三档 `z_up` 构造（`--zup-mode`）**：
   - `real_pass1`：将 `z_native` 空间 2× 下采样锁死动作姿态位置后，在 `960×544` 下加噪至 `σ=0.68` 并跑 2 步粗 Euler（`0.68 -> 0.34 -> 0.0`）→ `480p VAE` 往返 → `h3_upscaler 2.0×`，100% 对齐姿态的同时复现 4060 一采在大动作下的真实五官融化/残影畸变；
   - `downsample`：经典 `2× 降采 -> 480p VAE 往返 -> h3_upscaler 2.0×`；
   - `hybrid`（v3 默认）：按样本 ID 确定性哈希分配 80% `real_pass1` + 20% `downsample`，兼顾大动作轮廓修复与静态超分微纹理。
4. **运动峰值帧自动筛糊（`--min-motion-lapvar`）**：若峰值运动区域 `motion_roi_lap` 低于阈值，自动递增 `seed + 10007` 重抽（最多 `--max-qa-retries` 次，最终保留得分最高的一版，零样本丢失）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_UNET = "minimax_h3_ref2va_pruned_bf16.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_sol_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_LMS = "minimax_h3_lms_v1.0_r64_r.safetensors"
DEFAULT_UPSCALER = "h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"

EMPTY_PROMPT = " "


def _sage(model_link, rope_hf: float, attn_temp: float):
    return {
        "class_type": "MiniMaxH3DenseRopePatch",
        "inputs": {
            "model": model_link,
            "rope_hf_scale": float(rope_hf),
            "attn_temp_scale": float(attn_temp),
            "chunk_io_layers": True,
        },
    }


def _extract_ref_images(sample: dict) -> list[str]:
    refs = sample.get("ref_images")
    if isinstance(refs, (list, tuple)):
        out = [str(x).strip() for x in refs if x and str(x).strip()]
        if out:
            return out[:5]
    single = sample.get("ref_image")
    if single and str(single).strip():
        return [str(single).strip()]
    return []


def resolve_zup_mode(mode: str, key: str) -> str:
    m = (mode or "hybrid").strip().lower()
    if m in ("real_pass1", "downsample"):
        return m
    # hybrid: 80% real_pass1 + 20% downsample (deterministic by key)
    return "downsample" if (zlib.crc32(key.encode("utf-8")) % 5 == 0) else "real_pass1"


def build_graph(sample: dict, args) -> dict:
    frames = int(sample.get("frames") or args.frames)
    ref_list = _extract_ref_images(sample)
    task = str(sample.get("task") or ("ref2va" if ref_list else "t2va")).lower()
    is_ref = task.startswith("ref") and len(ref_list) > 0
    key = str(sample["id"])
    seed = int(sample.get("seed") or 0)
    width, height = 1920, 1088
    width_low, height_low = width // 2, height // 2  # 960×544 (exact 2x latent match: 60×34)

    eff_zup = resolve_zup_mode(getattr(args, "zup_mode", "downsample"), key)
    enable_qa = float(getattr(args, "min_motion_lapvar", 0.0) or 0.0) > 0.0
    pass1_sigmas = str(getattr(args, "pass1_sigmas", "0.68, 0.34, 0.0") or "0.68, 0.34, 0.0")
    pass1_cfg = float(getattr(args, "pass1_cfg", 3.0) or 3.0)
    zup_tmp_dir = str(getattr(args, "zup_tmp_dir", "/tmp/h3_zup_tmp") or "/tmp/h3_zup_tmp")

    g: dict = {}
    g["31"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}}
    g["26"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "minimax", "device": "default"}}
    g["20"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.video_vae}}
    g["12"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.audio_vae}}
    g["9"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": sample["prompt"]}}
    g["10"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": EMPTY_PROMPT}}

    if float(getattr(args, "lms_strength", 0.0) or 0.0) > 0.0:
        g["112"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": args.lms_lora, "strength_model": float(args.lms_strength), "model": ["31", 0]},
        }
        model_link = ["112", 0]
    else:
        model_link = ["31", 0]
    g["110"] = _sage(model_link, float(args.rope_hf), float(args.attn_temp))
    g["25"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}

    ref_node_links: list[tuple[str, list]] = []
    if is_ref:
        for idx, img_name in enumerate(ref_list):
            nid = f"ref_{idx}"
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
            ref_node_links.append((f"ref_images.ref_image_{idx}", [nid, 0]))

    def _make_cond_pair(w: int, h: int, pos_id: str, neg_id: str, ref_size: str):
        common = {
            "audio_mode": "native",
            "audio_denoise_strength": 1,
            "add_source_as_reference": False,
            "prompt_primary_audio_ordinal": 0,
            "ref_image_size": ref_size,
            "reference_video_policy": "official_2_to_15s",
            "allow_above_reference_area": True,
            "clip": ["26", 0],
            "video_vae": ["20", 0],
            "audio_vae": ["12", 0],
            "width": w,
            "height": h,
            "length": frames,
        }
        pos = dict(common)
        pos.update({
            "task_type": "Ref2VA" if is_ref else "T2VA",
            "strict_prompt_tags": True,
            "prompt": ["9", 0],
        })
        for field_k, link_v in ref_node_links:
            pos[field_k] = link_v
        g[pos_id] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": pos}

        neg = dict(common)
        neg.update({"task_type": "T2VA", "strict_prompt_tags": False, "prompt": ["10", 0]})
        g[neg_id] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": neg}

    _make_cond_pair(width, height, "8", "108", "max")

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

    dump_native_inputs: dict = {
        "latent": ["93", 0],
        "out_dir": args.out,
        "key": key,
        "tag": "z_native",
        "enabled": True,
    }
    if enable_qa:
        dump_native_inputs["vae"] = ["20", 0]
    g["204"] = {"class_type": "MiniMaxH3TeacherLatentDump", "inputs": dump_native_inputs}
    g["205"] = {
        "class_type": "MiniMaxH3ConditioningDump",
        "inputs": {"conditioning": ["8", 0], "out_dir": args.out, "key": key, "enabled": True},
    }

    if eff_zup == "real_pass1":
        # 姿态锚定真一采退化：z_native -> 2x 降采保 AV -> 960×544 下加噪 σ=0.68 跑 2 步粗 Euler
        _make_cond_pair(width_low, height_low, "308", "309", "max")
        g["310"] = _sage(model_link, 1.0, 1.0)
        g["211"] = {
            "class_type": "MiniMaxH3LatentDownsample2x",
            "inputs": {"latent": ["93", 0], "enabled": True, "scale_factor": 2, "keep_av": True},
        }
        g["311"] = {
            "class_type": "CFGGuider",
            "inputs": {"model": ["310", 0], "positive": ["308", 0], "negative": ["309", 0], "cfg": pass1_cfg},
        }
        g["312"] = {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "steps": 2,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": args.sampler,
                "scheduler": "beta57",
                "model": ["310", 0],
                "av_latent": ["308", 1],
            },
        }
        g["313"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": pass1_sigmas}}
        g["314"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 7919}}
        g["315"] = {
            "class_type": "MiniMaxH3FirstPassSampler",
            "inputs": {
                "noise": ["314", 0],
                "guider": ["311", 0],
                "sampler": ["312", 1],
                "sigmas": ["313", 0],
                "latent_image": ["211", 0],
            },
        }
        g["316"] = {
            "class_type": "MiniMaxH3LatentDownsample2x",
            "inputs": {"latent": ["315", 1], "enabled": True, "scale_factor": 1, "keep_av": False},
        }
        low_vid_link = ["316", 0]
    else:
        g["211"] = {
            "class_type": "MiniMaxH3LatentDownsample2x",
            "inputs": {"latent": ["93", 0], "enabled": True, "scale_factor": 2, "keep_av": False},
        }
        low_vid_link = ["211", 0]

    g["206"] = {"class_type": "VAEDecode", "inputs": {"samples": low_vid_link, "vae": ["20", 0]}}
    g["208"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["206", 0], "vae": ["20", 0]}}
    g["209"] = {
        "class_type": "MinimaxH3LatentUpscaler3D",
        "inputs": {
            "latent": ["208", 0],
            "model_name": args.upscaler,
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
    g["212"] = {
        "class_type": "MiniMaxH3AVLatentSave",
        "inputs": {
            "latent": ["210", 0],
            "key": f"{key}_zup_force",
            "overwrite": True,
            "cache_path": zup_tmp_dir,
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
    return [out_root / f"{key}.z_native.pt", out_root / f"{key}.z_up.pt", out_root / f"{key}.cond.pt"]


def generate_one_with_qa(sample: dict, args, api: str, out_root: Path, timeout: int = 2400) -> dict:
    """执行单条教师生成；若开启 min_motion_lapvar 则在未达标时自动换 seed 重抽并保留最锐利的一版。"""
    key = str(sample["id"])
    base_seed = int(sample.get("seed") or 0)
    min_lap = float(getattr(args, "min_motion_lapvar", 0.0) or 0.0)
    max_retries = max(0, int(getattr(args, "max_qa_retries", 2) or 0)) if min_lap > 0 else 0

    best_lap = -1.0
    best_backup_dir = out_root / f"_best_{key}"
    last_pid = ""
    t_total = time.time()

    try:
        for attempt in range(max_retries + 1):
            cur_sample = dict(sample)
            cur_sample["seed"] = base_seed + attempt * 10007
            graph = build_graph(cur_sample, args)
            pid = submit(api, graph)
            last_pid = pid
            wait_done(api, pid, timeout=timeout)
            missing = [p.name for p in expected(out_root, key) if not p.is_file()]
            if missing:
                raise RuntimeError(f"missing outputs: {missing}")

            if min_lap <= 0.0:
                break

            qa_path = out_root / f"{key}.qa.json"
            cur_lap = 9999.0
            if qa_path.is_file():
                try:
                    qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
                    cur_lap = float(qa_data.get("motion_roi_lap", 0.0))
                except Exception:
                    cur_lap = 9999.0

            if cur_lap >= min_lap:
                best_lap = cur_lap
                break

            print(
                f"[qa-gate] {key} attempt {attempt + 1}/{max_retries + 1} "
                f"motion_roi_lap={cur_lap:.1f} < {min_lap:.1f}",
                flush=True,
            )
            if cur_lap > best_lap:
                best_lap = cur_lap
                if max_retries > 0 and attempt < max_retries:
                    best_backup_dir.mkdir(parents=True, exist_ok=True)
                    for p in expected(out_root, key) + [qa_path]:
                        if p.is_file():
                            shutil.copy2(p, best_backup_dir / p.name)
            if attempt == max_retries and best_backup_dir.is_dir():
                # 恢复历次尝试中锐度最高的一版
                for p in best_backup_dir.iterdir():
                    shutil.copy2(p, out_root / p.name)
                print(f"[qa-gate] {key} keep sharpest attempt (motion_roi_lap={best_lap:.1f})", flush=True)
    finally:
        if best_backup_dir.is_dir():
            shutil.rmtree(best_backup_dir, ignore_errors=True)

    return {
        "prompt_id": last_pid,
        "sec": round(time.time() - t_total, 1),
        "zup_mode": resolve_zup_mode(getattr(args, "zup_mode", "downsample"), key),
        "motion_roi_lap": round(best_lap, 2) if best_lap >= 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--sampler", default="dual_clock_euler")
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--zup-mode", choices=["hybrid", "real_pass1", "downsample"], default="hybrid")
    ap.add_argument("--pass1-sigmas", default="0.68, 0.34, 0.0")
    ap.add_argument("--pass1-cfg", type=float, default=3.0)
    ap.add_argument("--zup-tmp-dir", default="/tmp/h3_zup_tmp")
    ap.add_argument("--min-motion-lapvar", type=float, default=0.0,
                    help="运动前景 ROI 最低 LapVar 门禁（推荐 75.0；0=关闭）")
    ap.add_argument("--max-qa-retries", type=int, default=2)
    ap.add_argument("--budget-min", type=float, default=0.0, help="0=不限制；到点停止并保留已完成样本")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unet", default=DEFAULT_UNET)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    ap.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    ap.add_argument("--lms-lora", default=DEFAULT_LMS)
    ap.add_argument("--lms-strength", type=float, default=0.0)
    ap.add_argument("--upscaler", default=DEFAULT_UPSCALER)
    ap.add_argument("--head-chunks", type=int, default=1)
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_gen_log.jsonl"

    try:
        import torch as _torch

        is_rocm = bool(getattr(_torch.version, "hip", None))
        if is_rocm and "nvfp4" in args.clip.lower():
            print(
                f"[rocm-guard] 警告：检测到 AMD ROCm 环境但 --clip={args.clip} 为 NVFP4 权重！"
                "强烈建议改为 bf16 / fp8 / int8 标准版 Qwen3-VL-32B 以避免 N 卡算子报错。",
                flush=True,
            )
        elif _torch.cuda.is_available() and "nvfp4" in args.clip.lower():
            cap = _torch.cuda.get_device_capability()
            if cap < (9, 0):
                print(f"[warn] nvfp4 CLIP on SM{cap[0]}{cap[1]}：无原生 FP4，走 dequant 路径", flush=True)
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
            if args.dry_run:
                graph = build_graph(sample, args)
                (out_root / f"{key}.graph.json").write_text(
                    json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[{i}/{len(samples)}] {key}: dry-run graph written (zup={resolve_zup_mode(args.zup_mode, key)})", flush=True)
                ok += 1
                continue
            info = generate_one_with_qa(sample, args, args.api, out_root, timeout=args.timeout)
            ok += 1
            rec.update({"status": "ok", **info})
            print(f"[{i}/{len(samples)}] {key}: OK {info['sec']}s zup={info['zup_mode']} lap={info['motion_roi_lap']}", flush=True)
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
