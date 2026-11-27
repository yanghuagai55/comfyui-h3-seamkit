# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""教师轨迹采集与动态锐度质检组件（仅训练用，不影响正常出片，保持原 4 个节点 ID 不变）。

* `MiniMaxH3TeacherProbe`       —— 旁路观察二采 (sigma, x, v)。
* `MiniMaxH3ConditioningDump`   —— 把二采完整 conditioning 原样落盘。
* `MiniMaxH3TeacherLatentDump`  —— 把 AV latent 按 `<key>.<tag>.pt` 落盘；若可选接入 `vae`，
                                   同步解码运动峰值 5-latent 窗并写出 `<key>.qa.json`
                                   （含运动 ROI `motion_roi_lap` 锐度，供教师自动筛糊重试）。
* `MiniMaxH3LatentDownsample2x` —— 视频潜空间 H/W 下采样（支持 `scale_factor=1` 纯提取视频分量，
                                   以及 `keep_av=True` 保留联合 AV NestedTensor 供真一采 `real_pass1` 重采样）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import comfy.nested_tensor
import folder_paths
from comfy_api.latest import io


CATEGORY = "MiniMax H3 Distill"
DEFAULT_DIRNAME = "h3_teacher_raw"


def _resolve_dir(out_dir: str) -> Path:
    raw = (out_dir or "").strip()
    root = Path(raw).expanduser() if raw else Path(folder_paths.get_output_directory()) / DEFAULT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_key(key: str) -> str:
    text = (key or "").strip() or "unnamed"
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)[:80]


def _split_tensors(value):
    """NestedTensor -> [普通张量]；普通张量 -> [它自己]。"""
    if hasattr(value, "is_nested") and getattr(value, "is_nested", False):
        return list(value.tensors)
    if hasattr(value, "unbind") and not torch.is_tensor(value):
        return list(value.unbind())
    return [value]


def _plain_cpu(tensors) -> list:
    out = []
    for t in tensors:
        if not torch.is_tensor(t):
            continue
        out.append(t.detach().to("cpu", torch.float16).contiguous())
    return out


def _sigma_of(timestep) -> float:
    if torch.is_tensor(timestep):
        if timestep.numel() == 0:
            return float("nan")
        return float(timestep.reshape(-1)[0].item())
    try:
        return float(timestep)
    except (TypeError, ValueError):
        return float("nan")


def compute_motion_peak_qa(vae, video_latent: torch.Tensor) -> dict:
    """解码运动能量最高的单个 5-latent 窗（<=17 帧），计算全局与运动前景 ROI 的 LapVar@1080。"""
    if video_latent.ndim != 5 or video_latent.shape[2] < 1:
        return {"global_lap": 0.0, "motion_roi_lap": 0.0, "peak_latent_t": 0}
    t_len = int(video_latent.shape[2])
    if t_len >= 2:
        lat_diff = (video_latent[:, :, 1:].float() - video_latent[:, :, :-1].float()).abs().mean(dim=(0, 1, 3, 4))
        peak_t = int(lat_diff.argmax().item())
    else:
        peak_t = 0
    t0 = max(0, min(t_len - 5, (peak_t // 5) * 5)) if t_len >= 5 else 0
    t1 = min(t_len, t0 + 5)
    z_win = video_latent[:, :, t0:t1].detach()

    with torch.no_grad():
        imgs = vae.decode(z_win)
    if imgs.ndim == 5:
        imgs = imgs[0]
    arr = (imgs.float().clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    n_f = arr.shape[0]
    if n_f >= 2:
        f_diffs = np.mean(np.abs(arr[1:].astype(np.float32) - arr[:-1].astype(np.float32)), axis=(1, 2, 3))
        best_f = int(np.argmax(f_diffs))
        pix_diff = np.mean(np.abs(arr[min(n_f - 1, best_f + 1)].astype(np.float32) - arr[best_f].astype(np.float32)), axis=-1)
        frame = arr[min(n_f - 1, best_f + 1)]
    else:
        best_f = 0
        pix_diff = np.ones(arr.shape[1:3], dtype=np.float32)
        frame = arr[0]

    gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.float32)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    global_lap = float(np.var(lap))

    # 按 32×32 块选取运动能量最高的前 25% 区域计算 LapVar（避免暗色静止背景拉低动作主体锐度评分）
    lh, lw = lap.shape
    bh, bw = max(1, lh // 32), max(1, lw // 32)
    crop_h, crop_w = bh * 32, bw * 32
    diff_crop = pix_diff[1:1 + crop_h, 1:1 + crop_w].reshape(bh, 32, bw, 32).mean(axis=(1, 3))
    lap_blocks = lap[:crop_h, :crop_w].reshape(bh, 32, bw, 32)
    thresh = float(np.percentile(diff_crop, 75.0))
    mask = diff_crop >= thresh
    roi_pixels = lap_blocks.transpose(0, 2, 1, 3)[mask]
    motion_roi_lap = float(np.var(roi_pixels)) if roi_pixels.size > 0 else global_lap

    return {
        "peak_latent_t": peak_t,
        "window_latents": [t0, t1],
        "peak_local_frame": best_f,
        "global_lap": round(global_lap, 2),
        "motion_roi_lap": round(motion_roi_lap, 2),
        "max_block_diff": round(float(diff_crop.max()), 2),
    }


class MiniMaxH3TeacherProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TeacherProbe",
            display_name="MiniMax H3 Teacher Probe (dense x/v)",
            description=(
                "旁路记录二采每一步的 (sigma, x, v)。速度按 Euler 原式 "
                "v=(x-denoised)/sigma 计算，与采样器更新完全一致。\n"
                "**只接在二采 model 链上**；接错到一采会把 4 步一采也录进来（有 sigma 过滤兜底）。\n"
                "不改变模型数值：wrapper 只是观察，返回值原样透传。"
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "out_dir",
                    default="",
                    tooltip="落盘目录。留空 = `<ComfyUI 输出目录>/h3_teacher_raw/`。",
                ),
                io.String.Input(
                    "key",
                    default="sample_0000",
                    tooltip="样本标识；文件名前缀。一个样本一个 key。",
                ),
                io.String.Input(
                    "sigmas",
                    default="0.82, 0.50, 0.18",
                    tooltip="只记录这些 sigma（逗号分隔）。二采 ManualSigmas 的前三个节点值。",
                ),
                io.Float.Input(
                    "sigma_tol",
                    default=0.01,
                    min=0.0001,
                    max=0.2,
                    step=0.0001,
                    advanced=True,
                    tooltip="sigma 匹配容差。默认 0.01 足够（ManualSigmas 是精确值）。",
                ),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, out_dir, key, sigmas, sigma_tol, enabled):
        if not enabled:
            return io.NodeOutput(model)

        targets = []
        for part in str(sigmas).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                targets.append(float(part))
            except ValueError:
                pass
        if not targets:
            raise ValueError(f"MiniMaxH3TeacherProbe: 无法解析 sigmas={sigmas!r}")

        root = _resolve_dir(out_dir)
        stem = _safe_key(key)
        state = {"n": 0}

        def _wrap(model_function, kwargs):
            result = model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            try:
                sigma = _sigma_of(kwargs.get("timestep"))
                if sigma != sigma or sigma <= 1e-6:
                    return result
                if not any(abs(sigma - t) <= float(sigma_tol) for t in targets):
                    return result

                x_parts = _split_tensors(kwargs["input"])
                d_parts = _split_tensors(result)
                if len(x_parts) != len(d_parts):
                    print(
                        f"[Distill] probe skip: x parts {len(x_parts)} != denoised parts {len(d_parts)}",
                        flush=True,
                    )
                    return result

                idx = state["n"]
                state["n"] += 1
                payload = {
                    "key": stem,
                    "sigma": float(sigma),
                    "step": int(idx),
                    "cond_or_uncond": list(kwargs.get("cond_or_uncond") or []),
                    "x": _plain_cpu(x_parts),
                    "v": _plain_cpu(
                        [
                            (xi.float() - di.float()) / float(sigma)
                            for xi, di in zip(x_parts, d_parts)
                        ]
                    ),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                path = root / f"{stem}.step{idx}_s{float(sigma):.4f}.pt"
                torch.save(payload, path)
                print(
                    f"[Distill] probe {path.name}  shapes={[list(t.shape) for t in payload['x']]}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Distill] probe error (ignored): {exc}", flush=True)
            return result

        patched = model.clone()
        patched.model_options = dict(getattr(patched, "model_options", {}) or {})
        patched.set_model_unet_function_wrapper(_wrap)
        print(f"[Distill] teacher probe armed: dir={root} key={stem} sigmas={targets}", flush=True)
        return io.NodeOutput(patched)


class MiniMaxH3ConditioningDump(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ConditioningDump",
            display_name="MiniMax H3 Conditioning Dump (teacher)",
            description=(
                "把二采 conditioning 原样落盘（文本特征 + 参考图 keyframe + 音频条件），"
                "直通节点，不改任何数值。训练时用同一份条件重放前向。"
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("out_dir", default=""),
                io.String.Input("key", default="sample_0000"),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, out_dir, key, enabled):
        if not enabled:
            return io.NodeOutput(conditioning)
        root = _resolve_dir(out_dir)
        stem = _safe_key(key)
        try:
            torch.save(
                {
                    "key": stem,
                    "conditioning": conditioning,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                root / f"{stem}.cond.pt",
            )
            print(f"[Distill] conditioning -> {stem}.cond.pt", flush=True)
        except Exception as exc:
            print(f"[Distill] conditioning dump failed (ignored): {exc}", flush=True)
        return io.NodeOutput(conditioning)


class MiniMaxH3TeacherLatentDump(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TeacherLatentDump",
            display_name="MiniMax H3 Teacher Latent Dump",
            description=(
                "把 AV latent 按 `<key>.<tag>.pt` 落盘（z_up / z_native / 任意中间量）；"
                "若可选接入 `vae`，同步对运动峰值帧写出 `<key>.qa.json` 锐度审计。直通节点，不改数值。"
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input("latent"),
                io.String.Input("out_dir", default=""),
                io.String.Input("key", default="sample_0000"),
                io.String.Input("tag", default="latent"),
                io.Boolean.Input("enabled", default=True),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, out_dir, key, tag, enabled, vae=None):
        if not enabled:
            return io.NodeOutput(latent)
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            print("[Distill] latent dump skip: no 'samples'", flush=True)
            return io.NodeOutput(latent)
        root = _resolve_dir(out_dir)
        stem = _safe_key(key)
        safe_tag = _safe_key(tag)
        parts = _split_tensors(samples)
        try:
            torch.save(
                {
                    "key": stem,
                    "tag": safe_tag,
                    "parts": _plain_cpu(parts),
                    "shapes": [list(t.shape) for t in parts],
                    "nested": bool(getattr(samples, "is_nested", False)),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                root / f"{stem}.{safe_tag}.pt",
            )
            print(f"[Distill] latent -> {stem}.{safe_tag}.pt  shapes={[list(t.shape) for t in parts]}", flush=True)
            if vae is not None and len(parts) >= 1 and torch.is_tensor(parts[0]) and parts[0].ndim == 5:
                qa = compute_motion_peak_qa(vae, parts[0])
                qa["key"] = stem
                qa["tag"] = safe_tag
                qa_path = root / f"{stem}.qa.json"
                qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[Distill-QA] {stem}: motion_roi_lap={qa['motion_roi_lap']} "
                    f"global_lap={qa['global_lap']} peak_t={qa['peak_latent_t']}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Distill] latent dump/qa failed (ignored): {exc}", flush=True)
        return io.NodeOutput(latent)


class MiniMaxH3LatentDownsample2x(io.ComfyNode):
    """视频潜空间空间下采样 / 分量提取节点。

    - `scale_factor=2, keep_av=False`（默认）：与旧版完全一致，输出单视频 480p 潜变量供 VAE 往返；
    - `scale_factor=2, keep_av=True`：输出 `[video_480p, audio]` NestedTensor，供 `real_pass1` 同构图粗采样退化；
    - `scale_factor=1, keep_av=False`：从 `real_pass1` 输出的联合 AV NestedTensor 中无损剥离纯视频分量送入 `VAEDecode`。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentDownsample2x",
            display_name="MiniMax H3 Latent Downsample 2x (teacher z_up)",
            description="视频潜空间 H/W 平均池化或分量剥离；支持保留 AV NestedTensor 供真一采退化分支使用。",
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("latent"),
                io.Boolean.Input("enabled", default=True),
                io.Int.Input("scale_factor", default=2, min=1, max=4, step=1, advanced=True),
                io.Boolean.Input("keep_av", default=False, advanced=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, enabled, scale_factor: int = 2, keep_av: bool = False):
        if not enabled:
            return io.NodeOutput(latent)
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            return io.NodeOutput(latent)
        nested = bool(getattr(samples, "is_nested", False))
        video = samples.tensors[0] if nested else samples
        rest = list(samples.tensors[1:]) if nested else []
        orig_dtype = video.dtype
        sf = max(1, int(scale_factor))
        if sf > 1:
            v = video.float()
            if v.ndim == 5:
                v = F.avg_pool3d(v, kernel_size=(1, sf, sf), stride=(1, sf, sf))
            elif v.ndim == 4:
                v = F.avg_pool2d(v, kernel_size=sf, stride=sf)
            v = v.to(orig_dtype)
        else:
            v = video
        print(f"[Distill] downsample(sf={sf}, keep_av={keep_av}) {list(video.shape)} -> {list(v.shape)}", flush=True)
        if keep_av and nested and rest:
            return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor([v, *rest])})
        return io.NodeOutput({"samples": v})


NODES = [
    MiniMaxH3TeacherProbe,
    MiniMaxH3ConditioningDump,
    MiniMaxH3TeacherLatentDump,
    MiniMaxH3LatentDownsample2x,
]
