# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""comfyui-h3-seamkit

Single-window Chunked-QKV SageAttention2++ & Sol-Attn sparse acceleration,
zero-RAM-OOM streaming RelayVSR saver, Sol-H3-VAE safe chunked decoding,
self-distillation teacher/LoRA utilities, and Route 3 split-stream nodes
for MiniMax H3.

Active Nodes (18 total):
  Core Inference & Two-Pass Chain (MiniMax H3 Hard Cut):
    * `MiniMaxH3HardCutFirstPassPlan`     — 上游规划：片长/分辨率/分段/一采缓存
    * `MiniMaxH3HardCutPass2Plan`         — 下游二采规划（精简版，已适配单窗与 Route 3）
    * `MiniMaxH3FirstPassSampler`         — 一采可缓存采样器（内置显存补丁护栏）
    * `MiniMaxH3HardCutUpscale`           — 二采执行器（单窗/多窗分块 + Route 3 透传）
    * `MiniMaxH3ChunkedSageAttentionPatch` — 分块 QKV + BF16 LoRA 旁路 + Sol-Attn + 四相 Morton + Split-Band RoPE
    * `MiniMaxH3AVLatentSave`             — 冻结联合 AV latent 存档
    * `MiniMaxH3AVLatentLoad`             — 冻结联合 AV latent 读档（0 秒跳过一采保音频）
    * `MiniMaxH3SafeAVDecode`             — Sol-H3-VAE / INT8 VAE 安全分块音视频解码
    * `MiniMaxH3StreamingVSRSave`         — 流式 RelayVSR Flash 解码直出 H.264 MP4（32GB 内存零 OOM）

  Self-Distillation Utilities (MiniMax H3 Distill):
    * `MiniMaxH3TeacherProbe`             — 教师采样探针
    * `MiniMaxH3ConditioningDump`         — 教师条件导出
    * `MiniMaxH3TeacherLatentDump`        — 教师潜空间导出
    * `MiniMaxH3LatentDownsample2x`       — `z_up` 潜空间 2x 下采样
    * `MiniMaxH3DenseRopePatch`           — 教师端稠密 SDPA + Split-Band RoPE 补丁

  Route 3 Acceleration Chain (MiniMaxH3/route3):
    * `MiniMaxH3Route3AutoMask`           — 一采直出自动前景掩码（绿幕/实景/动漫通用）
    * `MiniMaxH3Route3Apply`              — 二采前景全速率 + 背景关键帧双流引擎
    * `MiniMaxH3Route3Assemble`           — 背景高频残差还原（自动切镜断链 + 软羽化）
    * `MiniMaxH3Route3Probe`              — 50 层输出替换误差探针（科研标定用）
"""

from comfy_api.latest import ComfyExtension

from .chunked_sage import MiniMaxH3ChunkedSageAttentionPatch
from .h3_upscale import MiniMaxH3HardCutUpscale
from .nodes import (
    MiniMaxH3HardCutFirstPassPlan,
    MiniMaxH3HardCutPass2Plan,
)
from .nodes_firstpass import MiniMaxH3FirstPassSampler
from .nodes_latent_cache import MiniMaxH3AVLatentLoad, MiniMaxH3AVLatentSave
from .sol_vae_decode import MiniMaxH3SafeAVDecode
from .vsr_stream_save import MiniMaxH3StreamingVSRSave
from .train.teacher_capture import (
    MiniMaxH3ConditioningDump,
    MiniMaxH3LatentDownsample2x,
    MiniMaxH3TeacherLatentDump,
    MiniMaxH3TeacherProbe,
)
from .train.dense_rope_patch import MiniMaxH3DenseRopePatch
from .route3.nodes import (
    MiniMaxH3Route3Apply,
    MiniMaxH3Route3Assemble,
    MiniMaxH3Route3AutoMask,
    MiniMaxH3Route3Probe,
)

__all__ = ["comfy_entrypoint"]


class H3HardCutExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3HardCutFirstPassPlan,
            MiniMaxH3HardCutPass2Plan,
            MiniMaxH3FirstPassSampler,
            MiniMaxH3HardCutUpscale,
            MiniMaxH3ChunkedSageAttentionPatch,
            MiniMaxH3AVLatentSave,
            MiniMaxH3AVLatentLoad,
            MiniMaxH3SafeAVDecode,
            MiniMaxH3StreamingVSRSave,
            MiniMaxH3TeacherProbe,
            MiniMaxH3ConditioningDump,
            MiniMaxH3TeacherLatentDump,
            MiniMaxH3LatentDownsample2x,
            MiniMaxH3DenseRopePatch,
            MiniMaxH3Route3AutoMask,
            MiniMaxH3Route3Apply,
            MiniMaxH3Route3Assemble,
            MiniMaxH3Route3Probe,
        ]


def comfy_entrypoint():
    return H3HardCutExtension()
