# -*- coding: utf-8 -*-
"""Route 3 的三个 ComfyUI 节点

* `MiniMaxH3Route3Apply`    —— 把引擎挂到 model 上（只压你接的那条链路）
* `MiniMaxH3Route3Probe`    —— 第一次前向时顺手量 RoPE 衰减，写 json
* `MiniMaxH3Route3Assemble` —— 法 A / 法 B：用关键帧残差把中间帧背景还原

接法（API 格式里就是改一个 inputs 引用）：
    UNET/LoRA 链 → Route3Apply → CFGGuider.model → 二采 sampler
一采那条链路不接 = 一采保持原生，只压二采。
"""

from __future__ import annotations

import json
import logging
import os

import torch

from comfy_api.latest import io

from route3.assembly import reconstruct_moving
from route3.mask_provider import load_mask
from route3.patch import Route3State, install

CATEGORY = "MiniMaxH3/route3"


class MiniMaxH3Route3Apply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Apply",
            display_name="Route3 · Split-Stream (apply to model)",
            category=CATEGORY,
            description=(
                "把「前景全速率 + 背景关键帧」的双流引擎挂到这条模型链上。"
                "掩码来自 Oracle 视频（.npz），会按当前 latent 网格自动适配。"
                "只接二采 = 一采保持原生。"
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input("mask_path", default="", tooltip="mask .npz 路径；留空 = 直通（等价性 smoke）"),
                io.Int.Input("s_bg", default=5, min=1, max=64, step=1),
                io.String.Input("beta_path", default="", tooltip="splitstream_gates.pt；留空用 beta0=log(S_bg)"),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, mask_path="", s_bg=5, beta_path="", enabled=True):
        if not enabled:
            print("[route3] disabled - pass through", flush=True)
            return io.NodeOutput(model)

        provider = load_mask(mask_path) if mask_path and os.path.exists(mask_path) else None
        if mask_path and provider is None:
            logging.warning("[route3] mask not found: %s -> 直通（不压缩）", mask_path)

        beta = None
        if beta_path and os.path.exists(beta_path):
            blob = torch.load(beta_path, map_location="cpu")
            beta = blob.get("beta", blob) if isinstance(blob, dict) else blob
            print("[route3] beta loaded %s" % (tuple(beta.shape) if torch.is_tensor(beta) else beta), flush=True)

        fn = (lambda lt, lh, lw: provider.for_grid(lt, lh, lw)) if provider else None
        st = Route3State()
        m = install(model, s_bg=s_bg, beta=beta, fg_mask_fn=fn, state=st, strict_malloc=False)
        print("[route3] attached: s_bg=%d mask=%s" % (s_bg, "yes" if provider else "no"), flush=True)
        return io.NodeOutput(m)


class MiniMaxH3Route3Probe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Probe",
            display_name="Route3 · RoPE decay probe",
            category=CATEGORY,
            description="第一次前向时抽 8 头/层量注意力质量随 Δt 的分布，写 probe_attn_stats.json，然后直通。",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("out_path", default="/content/h3_route3/artifacts/probe_attn_stats.json"),
                io.Int.Input("n_query", default=8, min=1, max=64),
                io.Int.Input("layers", default=0, tooltip="0 = 全部层；>0 = 只测前 N 层"),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, out_path, n_query=8, layers=0):
        import types
        from route3.probe_attn import probe_attention, summarize, dump

        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        orig = dm.forward
        state = {"busy": False, "done": False}

        def fwd(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kw):
            if state["busy"] or state["done"] or denoise_mask is not None:
                return orig(x, timestep, context, transformer_options,
                            minimax_payload=minimax_payload, **kw)
            state["busy"] = True
            try:
                lay = list(range(len(self.blocks)))[:layers] if layers else None
                p = probe_attention(self, x, timestep, context, transformer_options,
                                    minimax_payload, n_query=n_query, layers=lay)
                s = summarize(p)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                dump(p, s, out_path)
                print("[route3] probe -> %s  verdict=%s median=%.3f"
                      % (out_path, s["verdict"], s["overall_median"]), flush=True)
                state["done"] = True
            except Exception as e:  # 探针失败绝不拖垮出片
                logging.warning("[route3] probe failed: %r", e)
                state["done"] = True
            finally:
                state["busy"] = False
            return orig(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kw)

        m.add_object_patch("diffusion_model.forward", types.MethodType(fwd, dm))
        return io.NodeOutput(m)


class MiniMaxH3Route3Assemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Assemble",
            display_name="Route3 · Background reassembly (法A / 法B)",
            category=CATEGORY,
            description=(
                "用二采在关键帧上真正画出的细节残差 δ = z_p2 - z_p1_up，"
                "把中间帧背景还原。method=flow 用分块位移场双向 warp（主推），"
                "lerp 是固定坐标对照组，none 表示只用采样器自己的结果。"
            ),
            inputs=[
                io.Latent.Input("latent_p1_up", tooltip="一采上采样后的 latent（全帧）"),
                io.Latent.Input("latent_p2", tooltip="二采采样器输出的 latent"),
                io.Int.Input("s_bg", default=5, min=1, max=64),
                io.Combo.Input("method", options=["flow", "lerp", "none"], default="flow"),
                io.Boolean.Input("only_bg", default=False,
                                 tooltip="True = 只改非前景区（需要掩码，先留 False 方便对比）"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent_p1_up, latent_p2, s_bg=5, method="flow", only_bg=False):
        a = latent_p1_up["samples"]
        b = latent_p2["samples"]
        if a.shape != b.shape:
            raise ValueError("两路 latent 形状不一致: %s vs %s" % (tuple(a.shape), tuple(b.shape)))
        T = a.shape[2]
        anchors = list(range(0, T, max(1, s_bg)))
        delta = {t: (b[:, :, t] - a[:, :, t]) for t in anchors}
        out = reconstruct_moving(a, delta, anchors, method=method if method != "none" else "none")
        return io.NodeOutput({"samples": out})


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Route3Apply": MiniMaxH3Route3Apply,
    "MiniMaxH3Route3Probe": MiniMaxH3Route3Probe,
    "MiniMaxH3Route3Assemble": MiniMaxH3Route3Assemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Route3Apply": "Route3 · Split-Stream (apply to model)",
    "MiniMaxH3Route3Probe": "Route3 · RoPE decay probe",
    "MiniMaxH3Route3Assemble": "Route3 · Background reassembly (法A/法B)",
}
