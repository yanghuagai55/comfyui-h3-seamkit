# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""早期护栏节点：接 MODEL、直通，在采样**之前**把致命配置问题喊出来。

为什么必须独立成一个节点、而且要插在模型加载之后
------------------------------------------------
本机（RTX 4060 Laptop 8GB）跑 MiniMax H3 时，OOM 发生在
`execution.py:306 process_inputs` —— 也就是**把模型权重搬进显存的阶段**，
采样根本没开始。所以：

* 把检测写在**二采执行器（MiniMaxH3HardCutUpscale）里是死代码** —— 它那时还没执行；
* 检测必须放在**拿到 model、但还没开始重活**的位置 —— 也就是「加载完大模型节点之后」。

它检查什么
----------
KJNodes 的 `MiniMaxH3MemoryEfficientSageAttentionPatch` 会把
`diffusion_model.blocks[*].attn.forward` 换成 `minimax_sageattn_forward`。
没换 = 峰值显存压不住 = 一采起步就 CUDA out of memory。

这个补丁**不依赖 `--use-sage-attention`**（那边只看 sageattention 模块能否 import），
但名字里带 "Sage"，换配置时极易被误当"配套"旁路掉，而且**失效时完全静默**：
既不报错，也没有任何日志 —— 之前两次 OOM 都是这么来的。

只警告，不阻断（有些场景可能故意不用）。
"""

from __future__ import annotations

from comfy_api.latest import io


def _attn_forward_name(model) -> str:
    try:
        dm = model.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if not blocks:
            return ""
        fwd = getattr(getattr(blocks[0], "attn", None), "forward", None)
        return getattr(fwd, "__name__", "") or ""
    except Exception:
        return ""


class MiniMaxH3VRamGuard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VRamGuard",
            display_name="MiniMax H3 VRAM Guard",
            description=(
                "早期护栏（直通：MODEL 进、MODEL 出）。\n"
                "插在「加载完大模型 / LoRA」之后、采样器之前 —— 因为本机 OOM 发生在\n"
                "`process_inputs` 搬权重那一步，写在二采执行器里根本轮不到执行。\n"
                "检查 KJNodes 的显存优化注意力补丁是否在位；不在就打醒目警告。只警告不阻断。"
            ),
            category="MiniMax H3 Hard Cut",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "require_sage_patch",
                    default=True,
                    tooltip=(
                        "开：没检测到显存优化补丁就警告（8GB 卡强烈建议开）。\n"
                        "关：只把检测到的注意力实现名打出来，不警告。"
                    ),
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, require_sage_patch: bool):
        name = _attn_forward_name(model)
        if name and "sageattn" in name:
            print(f"[SeamKit] VRAM Guard: 显存优化注意力补丁在位（{name}）", flush=True)
            return io.NodeOutput(model)

        msg = (
            f"[SeamKit] VRAM Guard: 注意力实现 = {name!r}"
            if name
            else "[SeamKit] VRAM Guard: 读不到 blocks[0].attn.forward（可能不是 MiniMax H3 模型）"
        )
        print(msg, flush=True)
        if require_sage_patch and name:
            print(
                "[SeamKit] ⚠⚠ 未检测到 KJNodes 的 MiniMax H3 显存优化注意力补丁\n"
                "          8GB 卡上少了它 -> 峰值显存压不住 -> 一采起步就 CUDA out of memory\n"
                "          （本机已因此失败两次，失败点一模一样：cast_to_gathered -> copy_from -> OOM）\n"
                "          请确认工作流里的 `MiniMax H3 Mem Eff Sage Attention Patch` 节点：\n"
                "            1) 在图上  2) mode = 0（没被 mute/bypass）  3) 输出接进 model 链\n"
                "          它不依赖 --use-sage-attention —— 关了旗标也要留着它。",
                flush=True,
            )
        return io.NodeOutput(model)


NODES = [MiniMaxH3VRamGuard]
