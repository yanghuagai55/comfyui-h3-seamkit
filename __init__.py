# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""comfyui-h3-seamkit

Two-pass hard-cut planning, cacheable first-pass sampling, and Route 3
split-stream acceleration for MiniMax H3.

Active Nodes (11 total — cleaned up from 21 legacy/pixel-repair nodes):
  Core Two-Pass Chain (MiniMax H3 Hard Cut):
    * `MiniMaxH3HardCutFirstPassPlan` — 上游规划：片长/分辨率/分段/一采缓存
    * `MiniMaxH3HardCutPass2Plan`     — 下游二采规划（精简版，已适配 Route 3）
    * `MiniMaxH3FirstPassSampler`     — 一采可缓存采样器（内置显存补丁护栏）
    * `MiniMaxH3HardCutUpscale`       — 二采执行器（不等长分块 + 多窗 Route 3 透传）
    * `MiniMaxH3ChunkedSageAttentionPatch` — 分块 QKV + KJ SageAttention2++ 低显存补丁（8GB 单窗 10s）
    * `MiniMaxH3AVLatentSave`         — 冻结联合 AV latent 存档
    * `MiniMaxH3AVLatentLoad`         — 冻结联合 AV latent 读档（0 秒跳过一采保音频）

  Route 3 Acceleration Chain (MiniMaxH3/route3):
    * `MiniMaxH3Route3AutoMask`       — 一采直出自动前景掩码（绿幕/实景/动漫通用）
    * `MiniMaxH3Route3Apply`          — 二采前景全速率 + 背景关键帧双流引擎
    * `MiniMaxH3Route3Assemble`       — 背景高频残差还原（自动切镜断链 + 软羽化）
    * `MiniMaxH3Route3Probe`          — 50 层输出替换误差探针（科研标定用）
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
            MiniMaxH3Route3AutoMask,
            MiniMaxH3Route3Apply,
            MiniMaxH3Route3Assemble,
            MiniMaxH3Route3Probe,
        ]


def comfy_entrypoint():
    return H3HardCutExtension()
