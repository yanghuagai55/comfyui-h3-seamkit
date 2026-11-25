# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""教师轨迹采集三件套（仅训练用，不影响正常出片）。

为什么放在 ComfyUI 节点里，而不是另写一套采样器
------------------------------------------------
二采的 conditioning（参考图 keyframe、音频 latent、双时钟噪声）由 T8 节点在
运行期才拼好，自己复刻一遍必然漂移。这里只做**旁路观察**：

* `MiniMaxH3TeacherProbe`  —— MODEL 进、MODEL 出。用 ComfyUI 官方的
  `model_function_wrapper` 钩子包住二采的每一次去噪调用，记录
  `(sigma, x, denoised)`；速度 `v = (x - denoised) / sigma` 与 Euler 更新
  完全同式，不需要猜符号。放在**二采 model 链**上，一采不受影响。
* `MiniMaxH3ConditioningDump` —— CONDITIONING 进、出。把二采的完整条件
  （Qwen3-VL 文本特征 + 参考图 keyframe latent + 音频条件）原样落盘，
  训练时才能重放同一次前向。
* `MiniMaxH3TeacherLatentDump` —— LATENT 进、出。存二采的 `upscaled_latent`
  （z_up）和最终 `z_teacher_clean`。

落盘格式刻意保持“普通张量 + pickle”，训练脚本用 `torch.load(weights_only=False)`
直接读；单个 22f/1088p 样本约 6–10 MB。
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
            # ComfyUI 的钩子签名：wrapper(model.apply_model, {"input","timestep","c",...})
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
            except Exception as exc:  # 观察失败绝不能打断采样
                print(f"[Distill] probe error (ignored): {exc}", flush=True)
            return result

        patched = model.clone()
        # clone() 的 model_options 可能是浅拷贝；显式再拷一层，避免污染一采共用链
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
                "把 AV latent 按 `<key>.<tag>.pt` 落盘（z_up / z_teacher_clean / 任意中间量）。"
                "直通节点，不改数值。"
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
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, out_dir, key, tag, enabled):
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
        except Exception as exc:
            print(f"[Distill] latent dump failed (ignored): {exc}", flush=True)
        return io.NodeOutput(latent)


NODES = [
    MiniMaxH3TeacherProbe,
    MiniMaxH3ConditioningDump,
    MiniMaxH3TeacherLatentDump,
]


class MiniMaxH3LatentDownsample2x(io.ComfyNode):
    """把 1080p 视频潜空间按 H/W 2× 平均池化到 480p（T 不变）。

    用途：教师 z_up 合成。先在潜空间降到 480p，再走 VAE 解码/编码，
    比「1080p 解码 → 像素缩放 → 编码」快约 4 倍，且保留 VAE 的软化特征。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentDownsample2x",
            display_name="MiniMax H3 Latent Downsample 2x (teacher z_up)",
            description="视频潜空间 H/W 2× 平均池化；AV 嵌套只处理视频分量。直通式，不改其它分量。",
            category=CATEGORY,
            is_experimental=True,
            inputs=[io.Latent.Input("latent"), io.Boolean.Input("enabled", default=True)],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, enabled):
        if not enabled:
            return io.NodeOutput(latent)
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            return io.NodeOutput(latent)
        nested = bool(getattr(samples, "is_nested", False))
        video = samples.tensors[0] if nested else samples
        orig_dtype = video.dtype
        v = video.float()
        if v.ndim == 5:
            v = F.avg_pool3d(v, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        elif v.ndim == 4:
            v = F.avg_pool2d(v, kernel_size=2, stride=2)
        v = v.to(orig_dtype)
        print(f"[Distill] downsample2x {list(video.shape)} -> {list(v.shape)}", flush=True)
        return io.NodeOutput({"samples": v})


NODES.append(MiniMaxH3LatentDownsample2x)
