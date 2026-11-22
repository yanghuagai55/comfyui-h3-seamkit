# -*- coding: utf-8 -*-
"""把 Route 3 引擎挂进 ComfyUI 的采样流程（ModelPatcher 级）

用法（自定义节点里）：
    from route3.patch import install
    m = install(model, s_bg=5, beta=beta, fg_mask_fn=my_mask_fn)
    return (m,)

`fg_mask_fn(latent_t, lat_h, lat_w) -> [latent_t, lat_h//2*lat_w//2] bool or None`
—— 一采/二采的 latent 尺寸不同，掩码要按形状现算；返回 None 表示不压缩
（此时本模块退化成直通，用于云端等价性 smoke）。

⚠ 两个必须在上机前确认的前置条件（本机无法代验）：
1. `comfy.model_prefetch.malloc_graph_enabled(device)` 必须是 False。
   我们绕开了 `MiniMaxH3Model.forward` 里的 `malloc_graph_begin/end`，动态显存图一旦开着
   就会踩到未初始化的分配器。启动加 `--disable-comfy-compiler` 即可关掉。
2. `denoise_mask` / `audio_denoise_mask` 非空时直接回落到原生 forward —— 那条路径的
   逐行时间步我们还没搬。ref2va 若用到 mask，压缩会静默失效（会计数并告警）。
"""

from __future__ import annotations

import logging
import types

import torch

from .engine import splitstream_forward

__all__ = ["install", "malloc_graph_active", "Route3State"]


def malloc_graph_active(device) -> bool:
    try:
        import comfy.model_prefetch as mp
        return bool(mp.malloc_graph_enabled(device))
    except Exception:
        return False


class Route3State:
    def __init__(self):
        self.calls = 0
        self.compressed = 0
        self.passthrough = 0
        self.last_plan = None
        self.warned = set()

    def note(self, kind, msg):
        if kind not in self.warned:
            self.warned.add(kind)
            logging.warning("[route3] " + msg)

    def summary(self):
        return {"calls": self.calls, "compressed": self.compressed,
                "passthrough": self.passthrough, "last_plan": self.last_plan}


def install(model_patcher, s_bg=5, beta=None, fg_mask_fn=None, state=None,
            strict_malloc=True, use_beta=True):
    """返回装了 Route 3 forward 的 ModelPatcher（clone，不改原对象）。"""
    from comfy.ldm.minimax.model import time_shift_sigma

    m = model_patcher.clone()
    dm = m.get_model_object("diffusion_model")
    orig = dm.forward
    st = state or Route3State()

    if strict_malloc and malloc_graph_active(torch.device("cuda")):
        logging.info("[route3] malloc graph 已开启 —— 由本模块按原生 forward 的方式包裹，"
                     "不需要加 --disable-comfy-compiler")

    def fwd(self, x, timestep, context, transformer_options={}, minimax_payload=None,
            denoise_mask=None, audio_denoise_mask=None, **kw):
        st.calls += 1
        if denoise_mask is not None or audio_denoise_mask is not None:
            st.passthrough += 1
            st.note("mask", "带 denoise_mask 的步回落原生 forward（此时不压缩）")
            return orig(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload,
                        denoise_mask=denoise_mask,
                        audio_denoise_mask=audio_denoise_mask, **kw)
        if not torch.is_tensor(x[0]) or x[0].shape[0] != 1:
            st.passthrough += 1
            return orig(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload,
                        denoise_mask=denoise_mask,
                        audio_denoise_mask=audio_denoise_mask, **kw)

        lat_t, lat_h, lat_w = x[0].shape[2], x[0].shape[3], x[0].shape[4]
        mask = fg_mask_fn(lat_t, lat_h, lat_w) if fg_mask_fn is not None else None

        # 与 MiniMaxH3Model.forward 完全对齐的 audio_scale 换算（兼容 ModelSamplingAV 与 dual_clock_euler）
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        x_in = x
        if scale != 1.0:
            shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
            shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x_in = [x[0], audio_src * carry]

        # 和原生 forward 一样：malloc graph 开着就 begin/end + 拷进预分配缓冲
        import comfy.model_prefetch as _mp
        mg = _mp.malloc_graph_enabled(x_in[0].device)
        if mg:
            buf = [torch.empty_like(x_in[0]), torch.empty_like(x_in[1])]
            _mp.malloc_graph_begin(x_in[0].device)
        out, plan, _ = splitstream_forward(
            self, x_in, timestep, context, transformer_options, minimax_payload,
            s_bg=s_bg, beta=beta, fg_mask=mask, full_velocity=True,
            return_plan=True, use_beta=use_beta)
        st.last_plan = plan.to_dict()
        if mg:
            buf[0].copy_(out[0])
            buf[1].copy_(out[1])
            del out
            _mp.malloc_graph_end()
            out = buf
        if scale != 1.0:
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])
        if mask is not None and s_bg != 1:
            st.compressed += 1
        else:
            st.passthrough += 1
        return out

    m.add_object_patch("diffusion_model.forward", types.MethodType(fwd, dm))
    return m
