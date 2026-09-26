# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""一采采样节点：官方 `SamplerCustomAdvanced` 的薄壳 + 一采缓存 + 显存护栏。

为什么要有这个节点
------------------
* 本机实测一采**不是逐位可复现的**（同 seed 跨会话，测得的转镜帧能差 9 帧），
  而"让采样变确定"在 8GB 上走不通（关 Sage 会 OOM、钉 MATH 是 O(n²)）。
* 唯一干净的 A/B = **冻结一采**。但独立 Save 节点放在一采后面，一采永远先跑，跳不过去。
* ⇒ 把缓存装进**一采节点本身**：同一个节点两种模式，图不用改，切 `use_cache` 即可。

实现方式（**不是 fork**）
------------------------
逐行复刻官方 `SamplerCustomAdvanced.execute`
（comfy_extras/nodes_custom_sampler.py:1041-1075）的流程 ——
`fix_empty_latent_channels` -> `guider.sample(noise.generate_noise(...), ...)` ->
`process_latent_out` —— 只在外面包了：缓存读取（采样前）与缓存保存（采样后），
外加显存护栏（在 `guider.sample` 的权重搬运**之前**打印，赶得上 OOM）。
签名与官方完全一致，可直接替换。
"""

from __future__ import annotations

import comfy.sample
import comfy.model_management
import latent_preview
import comfy.utils
from comfy_api.latest import io

from .nodes_latent_cache import (
    _file_stem,
    fingerprint_from_node_inputs,
    load_av_latent,
    save_av_latent,
    zero_audio_part,
)
from .nodes_guard import _attn_forward_name


def _release_upstream_audio(latent_image) -> int:
    """丢掉上游（#8/#41 那条线）送进来的音频分量，返回被丢弃的元素数。

    HIT 时 `latent_image` 是**上游 conditioning 节点已经产出**的联合 AV latent ——
    它的音频就是我们要扔掉的那份"前面采样出来的声音"。这里：
      * 若为 nested 且恰好两分量（视频+音频），**就地清零音频**并返回其元素数；
        上游张量此后若无人再引用即可被回收（我们不再持有它）。
      * 单分量（纯视频）或形状不符：原样返回 0，不猜、不动。
    永不抛异常。
    """
    n_video, n_audio = zero_audio_part(
        (latent_image or {}).get("samples") if isinstance(latent_image, dict) else None,
        label="上游 latent_image",
    )
    return n_audio


def _cache_hit(key: str, fp: str, path: str | None = None):
    """指纹一致才命中；读不到/不一致返回 None。`path` 非空 = 从该目录读。"""
    try:
        samples, meta, name = load_av_latent(key, path=path)
    except ValueError:
        # ★ miss 必须留痕：2026-09-24 跑 B 时 use_cache 没开/没存过，
        #   全程无日志线索，用户以为"节点没变化"。没命中要说出来。
        where = (path or "").strip() or "<默认输出目录>/seamkit_latent_cache"
        print(
            f"[SeamKit] 一采缓存 MISS（在 {where} 没有 key={key!r} 的存档），照常采样"
            "（本次采样结束后会写入缓存，下一次同 key 且上游没变才会 HIT）",
            flush=True,
        )
        return None
    old = (meta.get("fingerprint") or "")
    if fp and old and old != fp:
        print(
            f"[SeamKit] 一采缓存指纹不一致 -> 不使用（存档 {old[:12]} vs 当前 {fp[:12]}），照常采样",
            flush=True,
        )
        return None
    return samples, meta, name


class MiniMaxH3FirstPassSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FirstPassSampler",
            display_name="MiniMax H3 First Pass (cacheable)",
            description=(
                "官方 `SamplerCustomAdvanced` 的薄壳 + 一采缓存 + 显存护栏。\n"
                "输入输出签名与官方完全一致，可**直接替换**一采的 SamplerCustomAdvanced。\n\n"
                "use_cache 是**总开关**：\n"
                "  开：同 key 且上游指纹一致 -> **直接读缓存、跳过采样**（省一次一采，"
                "二采输入逐位相同，A/B 才干净）；没命中 -> 照常采样并保存缓存。\n"
                "      HIT 时**按设计清零音频**：一采把音频与视频一起采样并写入缓存（第一次）；"
                "之后每次运行，上游音频链路照样先跑完才进本节点——缓存里那份是旧音频，"
                "不清零会与本次上游音频混流。故缓存音频与上游送入的音频双双清零，"
                "流向下游的音频恒为零（二采拿到「干净视频 + 静音」）。设计行为，不是 bug。\n"
                "  关：**纯采样器** —— 不算指纹、不读不写缓存，行为与官方 "
                "SamplerCustomAdvanced 一致（仅保留显存护栏打印，纯诊断不影响数值）。\n\n"
                "缓存目录可用 `cache_path` 改到任意盘（留空 = 输出目录下的 "
                "`seamkit_latent_cache/`）。"
            ),
            category="MiniMax H3 Hard Cut",
            is_experimental=True,
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Boolean.Input(
                    "use_cache",
                    optional=True,
                    tooltip=(
                        "总开关。开：命中缓存就跳过采样（读回冻结的一采），"
                        "没命中就采样并写缓存。\n"
                        "关：纯采样 —— 不算指纹、不读不写缓存（= 官方采样器行为）。"
                    ),
                ),
                io.String.Input(
                    "cache_key",
                    optional=True,
                    tooltip="缓存标识。建议带来源，如 `seed342114_s5_cam_v4`。换 key = 换缓存。",
                ),
                io.Boolean.Input(
                    "require_sage_patch",
                    optional=True,
                    tooltip="开：没检测到 KJNodes 显存优化补丁就打醒目警告（8GB 卡强烈建议开）。",
                ),
                io.String.Input(
                    "cache_path",
                    optional=True,
                    tooltip=(
                        "缓存目录。留空 = 默认 `<输出目录>/seamkit_latent_cache/`。\n"
                        "填绝对路径即可指到任意盘（本机例：`D:\\共享\\seamkit_latent_cache`）。\n"
                        "HIT 从这个目录读、MISS 后往这个目录写；与独立 Load/Save 节点\n"
                        "共用同一份目录时，三边填一致即可互相复用缓存。"
                    ),
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[
                io.Latent.Output("output"),
                io.Latent.Output("denoised_output"),
                io.Boolean.Output("cache_hit"),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        use_cache: bool = False,
        cache_key: str = "run1",
        require_sage_patch: bool = True,
        cache_path: str = "",
    ):
        import torch

        # ── 0. 总开关：use_cache=False = 纯采样器 ─────────────────────────
        #    不算指纹（指纹要走完整上游子图 + sha256）、不读不写缓存。
        fp = summary = None
        key = ""
        if use_cache:
            _hidden = getattr(cls, "hidden", None)
            fp, summary = fingerprint_from_node_inputs(
                getattr(_hidden, "prompt", None), getattr(_hidden, "unique_id", None)
            )
            key = (cache_key or "").strip() or "run1"

        # ── 1. 显存护栏：必须在 guider.sample 之前打，赶得上权重搬运阶段的 OOM ──
        name = _attn_forward_name(guider.model_patcher)
        if name and "sageattn" in name:
            print(f"[SeamKit] 一采: 显存优化补丁在位（{name}）", flush=True)
        else:
            print(
                f"[SeamKit] 一采: 注意力实现 = {name!r}"
                + (
                    "\n[SeamKit] ⚠⚠ 8GB 卡上少了显存优化补丁 -> 峰值显存压不住 -> 一采就 OOM\n"
                    "          请确认 `MiniMax H3 Mem Eff Sage Attention Patch` 节点在图上、"
                    "mode=0、输出接进 model 链。"
                    if require_sage_patch and name
                    else ""
                ),
                flush=True,
            )

        # ── 2. 缓存读取（在采样之前）──────────────────────────────────────
        if use_cache:
            hit = _cache_hit(key, fp, path=cache_path)
            if hit is not None:
                samples, meta, fname = hit
                # 加载缓存 = 音频按设计清零（用户 2026-09-25 定调并确认语义，勿改行为）：
                #   · 一采把音频与视频一起采样、一起写入缓存（第一次）
                #   · 之后每次运行，上游音频链路照样跑完才进本节点（第二次同理）
                #   · 缓存里那份音频是"上一次采样"的旧音频——不清零就会与本次上游
                #     送入的音频混流；所以缓存音频与上游送入的音频**双双清零**，
                #     保证 HIT 路径流向下游的音频恒为静音（二采拿到「干净视频 + 静音」）。
                #   · 这是设计行为，不是 bug；也不省显存（音频 latent ≈ 53 KB）。
                n_v, n_a = zero_audio_part(samples, label="一采缓存")
                dropped = _release_upstream_audio(latent_image)
                print(
                    f"[SeamKit] 一采缓存 HIT（跳过采样）: {fname}  fp={fp[:12]}  "
                    f"saved_at={meta.get('saved_at')}\n"
                    f"[SeamKit]         音频按设计清零（防旧音频与本次上游音频混流）：\n"
                    f"[SeamKit]           · 一采把音频与视频一起采样并写入缓存（第一次）；"
                    f"之后每次运行，上游音频链路照样先跑完才进本节点\n"
                    f"[SeamKit]           · 缓存里那份是上次采样的旧音频 → 清零（{n_a} 元素）；"
                    f"上游 latent_image 送入的音频 → 丢弃（{dropped} 元素）\n"
                    f"[SeamKit]           · ⇒ HIT 路径流向下游的音频恒为静音"
                    f"（二采拿「干净视频 + 静音」；成片音频由上游/后续节点决定）"
                    f"——设计行为，不是 bug，也不省显存（音频 latent ≈ 53 KB）",
                    flush=True,
                )
                out = {"samples": samples}
                return io.NodeOutput(out, dict(out), True)

        # ── 3. 官方采样流程（逐行照抄 SamplerCustomAdvanced.execute）──────
        latent = latent_image
        li = latent["samples"]
        latent = latent.copy()
        li = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, li,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = li

        noise_mask = latent.get("noise_mask")

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent), li, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0 = x0_output["x0"]
            if samples.is_nested and not x0.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes)
                )
            x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        # ── 4. 缓存保存（采样之后；失败不影响生成）────────────────────────
        if use_cache:
            print(
                save_av_latent(samples, key, False, fp, summary, tag="一采", path=cache_path),
                flush=True,
            )

        return io.NodeOutput(out, out_denoised, False)
