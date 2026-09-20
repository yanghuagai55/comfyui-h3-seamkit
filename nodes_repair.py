"""Pixel-domain seam repair nodes (V2): blend / fuse / dissolve on IMAGE batches.

These live AFTER the AV decoder (#16 `frames` IMAGE output) and before the
video save, operating on the decoded frame sequence directly:

* `MiniMaxH3SeamBlend`    - soften the one-frame pop at a window seam
* `MiniMaxH3SeamFuse`     - feather the 5-frame bridge clip into the main
                            frames with a strength ramp (bridge input is
                            OPTIONAL: when the bridge group is bypassed the
                            node passes the main frames through untouched)
* `MiniMaxH3SeamDissolve` - replace a corrupted span with a linear dissolve
                            between its healthy neighbours

All of them preserve frame count - the audio track wired into the video
save node is never affected.
"""

import torch

from comfy_api.latest import io, ComfyExtension

CATEGORY = "h3_hardcut/repair"


# --------------------------------------------------------------------------
# pure kernels (unit-testable without ComfyUI)
# --------------------------------------------------------------------------

def blend_pair(a: torch.Tensor, b: torch.Tensor, strength: float, mode: str):
    """The seam pop spread across the two frames around it."""
    if mode == "prev":  # only the earlier frame leans forward
        return (1.0 - strength) * a + strength * b, b
    if mode == "next":  # only the later frame leans back
        return a, strength * a + (1.0 - strength) * b
    # both: peak jump drops by (1 - 2*strength)
    return (
        (1.0 - strength) * a + strength * b,
        strength * a + (1.0 - strength) * b,
    )


def fuse_ramp(n: int, side: str, fmin: float, fmax: float) -> torch.Tensor:
    """Patch weight per bridge frame: after = strong at frame 0 fading out;
    before = mirrored.  n == 1 -> the single frame takes the strong end."""
    t = torch.linspace(0.0, 1.0, n) if n > 1 else torch.zeros(1)
    if side == "after":
        return fmax - (fmax - fmin) * t
    return fmin + (fmax - fmin) * t


def fuse_frames(
    main: torch.Tensor,
    bridge: torch.Tensor,
    start: int,
    side: str,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Feather `bridge` into `main` starting at frame `start` (in place safe:
    returns a new tensor).  When bridge is None the main frames pass through."""
    if bridge is None or bridge.numel() == 0:
        return main
    out = main.clone()
    n = min(int(bridge.shape[0]), int(main.shape[0]) - start)
    if n <= 0:
        return out
    w = fuse_ramp(n, side, fmin, fmax).view(n, *([1] * (main.ndim - 1)))
    span = out[start : start + n]
    out[start : start + n] = (1.0 - w) * span + w * bridge[:n].to(span.dtype)
    return out


def dissolve_span(
    frames: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """Replace frames start..end with a linear dissolve from frame start-1
    to frame end+1 (both must be inside the clip)."""
    out = frames.clone()
    left = out[start - 1]
    right = out[end + 1]
    n = end - start + 2
    for j, fi in enumerate(range(start, end + 1)):
        alpha = (j + 1) / n
        out[fi] = (1.0 - alpha) * left + alpha * right
    return out


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

class MiniMaxH3SeamBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeamBlend",
            display_name="MiniMax H3 Seam Blend (repair)",
            category="h3_hardcut/repair",
            inputs=[
                io.Image.Input("images", tooltip="解码后的整片帧序列（#16 frames）"),
                io.Int.Input(
                    "seam_frame", default=67, min=1,
                    tooltip="缝在 seam_frame 与 seam_frame+1 之间 —— 从 #40 report "
                            "的 boundary_frame 读（边界 68 = 缝 67|68 → 填 67）",
                ),
                io.Float.Input(
                    "strength", default=0.0, min=0.0, max=0.5, step=0.01,
                    tooltip="0 = 不处理；0.3 ≈ 峰值跳变降 40%（实测）",
                ),
                io.Combo.Input(
                    "mode", options=["both", "prev", "next"], default="both",
                    tooltip="both = 两帧互相靠拢（峰值降幅最大）；prev/next 单侧",
                ),
            ],
            outputs=[io.Image.Output("images")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, seam_frame: int, strength: float, mode: str):
        if strength <= 0:
            return io.NodeOutput(images)
        s = int(seam_frame)
        if not (0 <= s < images.shape[0] - 1):
            raise ValueError(f"seam_frame {s} outside 0..{images.shape[0] - 2}")
        a, b = blend_pair(images[s], images[s + 1], float(strength), mode)
        out = images.clone()
        out[s], out[s + 1] = a, b
        return io.NodeOutput(out)


class MiniMaxH3SeamFuse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeamFuse",
            display_name="MiniMax H3 Bridge Fuse (repair)",
            category="h3_hardcut/repair",
            inputs=[
                io.Image.Input("images", tooltip="解码后的整片帧序列（#16 frames）"),
                io.Image.Input(
                    "bridge", optional=True,
                    tooltip="桥段分支 VAEDecode 的 5 帧输出。未连接（桥段组 bypass）"
                            "时本节点直通主片 —— 不影响正常出片",
                ),
                io.Int.Input(
                    "start_frame", default=69, min=0,
                    tooltip="桥段第一帧覆盖的主片帧号（= 崩帧起点，从 #40 report 读）",
                ),
                io.Combo.Input(
                    "side", options=["after", "before"], default="after",
                    tooltip="after = 桥段权重贴崩帧起点最强、沿帧递减"
                            "（硬切后五帧，强度高到弱）；before 镜像",
                ),
                io.Float.Input("fuse_min", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("fuse_max", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("images")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, bridge, start_frame: int, side: str,
                fuse_min: float, fuse_max: float):
        out = fuse_frames(images, bridge, int(start_frame), side,
                          float(fuse_min), float(fuse_max))
        return io.NodeOutput(out)


class MiniMaxH3SeamDissolve(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeamDissolve",
            display_name="MiniMax H3 Seam Dissolve (repair)",
            category="h3_hardcut/repair",
            inputs=[
                io.Image.Input("images", tooltip="解码后的整片帧序列（#16 frames）"),
                io.Int.Input(
                    "start_frame", default=69, min=1,
                    tooltip="崩坏区间起点（该帧及之后到 end_frame 被叠化替换）",
                ),
                io.Int.Input("end_frame", default=72, min=1, tooltip="崩坏区间末帧"),
            ],
            outputs=[io.Image.Output("images")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, start_frame: int, end_frame: int):
        s, e = int(start_frame), int(end_frame)
        if not (0 < s <= e < images.shape[0] - 1):
            raise ValueError(
                f"dissolve span {s}..{e} needs anchors at {s - 1} and {e + 1} "
                f"inside 0..{images.shape[0] - 1}"
            )
        return io.NodeOutput(dissolve_span(images, s, e))


class MiniMaxH3RepairExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SeamBlend,
            MiniMaxH3SeamFuse,
            MiniMaxH3SeamDissolve,
        ]


def comfy_entrypoint():
    return MiniMaxH3RepairExtension()
