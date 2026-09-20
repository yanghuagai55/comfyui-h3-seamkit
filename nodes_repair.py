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
* `MiniMaxH3InfoBuffer`   - stash the main frames + both reports between the
                            render chain and the repair branch; parses the
                            #40 report for the cut/redraw frames (auto) or
                            takes hand-filled values (manual mode); the
                            images always pass through untouched

All of them preserve frame count - the audio track wired into the video
save node is never affected.
"""

import json

import torch

from comfy_api.latest import io, ComfyExtension

CATEGORY = "h3_hardcut/repair"


# --------------------------------------------------------------------------
# pure kernels (unit-testable without ComfyUI)
# --------------------------------------------------------------------------

def parse_boundary(report_str):
    """Extract the window boundary frame from a #40 report (JSON string).

    Priority: seam_hunt.aligned[0].boundary_frame (hunt moved the cut) ->
    segments[0].frames[1] (the planned boundary).  Returns
    (boundary_frame or None, source_tag or None).
    """
    if not report_str:
        return None, None
    try:
        data = json.loads(report_str)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    aligned = (data.get("seam_hunt") or {}).get("aligned") or []
    for entry in aligned:
        bf = entry.get("boundary_frame")
        if bf is not None:
            return int(bf), "hunt"
    segs = data.get("segments") or []
    if segs:
        frames = segs[0].get("frames") or []
        if len(frames) >= 2:
            return int(frames[1]), "plan"
    return None, None


def load_video_frames(path: str, start: int = 0, max_frames: int = 0):
    """Decode a video file into a ComfyUI IMAGE batch [B, H, W, C] float 0-1.

    Used by the InfoBuffer's manual mode: point it at an existing clip and the
    whole repair chain runs in seconds instead of re-generating the video.
    """
    import numpy as np

    try:
        import av  # PyAV
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyAV is required to load a video in the InfoBuffer (pip install av)"
        ) from exc

    frames = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index < int(start):
                continue
            if max_frames and len(frames) >= int(max_frames):
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no frames decoded from {path} (start={start})")
    batch = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(batch)


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
    def execute(cls, images: torch.Tensor, seam_frame=67, strength=0.0, mode="both"):
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
    def execute(cls, images: torch.Tensor, bridge=None, start_frame=69, side="after",
                fuse_min=0.0, fuse_max=1.0):
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


class MiniMaxH3InfoBuffer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3InfoBuffer",
            display_name="MiniMax H3 Info Buffer (repair)",
            category="h3_hardcut/repair",
            inputs=[
                io.Image.Input(
                    "images", optional=True,
                    tooltip="主片帧序列（#16 frames）——原样直通。**未连接时改用下面的\n"
                            "video_path 从磁盘视频加载**（快速验证：只 bypass #16 即可，\n"
                            "生成链整条不跑，几秒出修复片）",
                ),
                io.String.Input(
                    "report_upscale", optional=True, default="",
                    tooltip="#40 的 report（JSON）——auto 模式从这里解析切点/边界帧",
                ),
                io.String.Input(
                    "report_auto", optional=True, default="",
                    tooltip="#56 的 report——备用解析源（优先级低于 #40）",
                ),
                io.Combo.Input(
                    "source_mode", options=["auto", "manual"], default="auto",
                    tooltip="auto = 从 report 自动解析切点与重绘起点，下面的手填值不生效；\n"
                            "manual = 用手填值（视频起点、切点都可手选）。\n"
                            "auto 解析不到时自动回退手填值并在 report 里注明",
                ),
                io.Int.Input(
                    "manual_cut_frame", default=69, min=1,
                    tooltip="手动切点帧（manual 模式生效）——缝在它与前一帧之间",
                ),
                io.Int.Input(
                    "manual_redraw_start", default=68, min=0,
                    tooltip="手动重绘起点帧（manual 模式生效）——桥段切片/覆盖从这帧开始",
                ),
                io.String.Input(
                    "video_path", default="",
                    tooltip="★ 快速验证用：填一个成片视频的完整路径（如\n"
                            "D:\\共享\\MiniMaxH3\\exp_4v10a_00043.mp4），\n"
                            "本节点就从这个文件加载帧序列，不再要上游的 #16 输出。\n"
                            "配合「只 bypass #16」几秒就能走完修复链看效果",
                ),
                io.Int.Input(
                    "video_start", default=0, min=0,
                    tooltip="从视频的第几帧开始取（0 = 从头）",
                ),
                io.Int.Input(
                    "video_frames", default=0, min=0,
                    tooltip="取多少帧（0 = 全部）",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="主片帧序列原样直通"),
                io.Int.Output("cut_frame", tooltip="切点帧（缝在 cut_frame-1 与 cut_frame 之间）"),
                io.Int.Output("seam_frame", tooltip="= cut_frame-1，接 SeamBlend.seam_frame"),
                io.Int.Output("redraw_start", tooltip="重绘切片起点，接 ImageFromBatch.batch_index"),
                io.String.Output("report", tooltip="暂存的解析结果（JSON）"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images=None, report_upscale="", report_auto="", source_mode="auto",
                manual_cut_frame=69, manual_redraw_start=68,
                video_path="", video_start=0, video_frames=0):
        path = (video_path or "").strip()
        offset = 0
        if path:
            offset = max(0, int(video_start))
            out_images = load_video_frames(path, offset, int(video_frames))
            src_side = (f"frames loaded from {path} "
                        f"({int(out_images.shape[0])} frames, offset {offset})")
        elif images is not None:
            out_images = images
            src_side = "frames passed through from upstream (#16)"
        else:
            raise ValueError(
                "InfoBuffer: no frames coming in - connect the decoder or set "
                "video_path to an existing clip"
            )
        cut, src_name = None, None
        for name, rep in (("upscale", report_upscale), ("auto", report_auto)):
            cut, src_name = parse_boundary(rep)
            if cut is not None:
                src_name = f"{name}/{src_name}"
                break
        manual = source_mode == "manual"
        cut_final = int(manual_cut_frame)
        redraw = int(manual_redraw_start)
        note = None
        if manual:
            note = "manual mode: hand-filled values in effect"
        elif cut is not None:
            cut_final = cut
            redraw = cut - 1  # redraw span starts one frame before the seam
            note = f"auto: parsed from {src_name}"
        else:
            note = "auto found no boundary in either report - fell back to hand-filled values"
        if offset:
            # a partial load shifts the frame numbering: report the frames in
            # the NEW sequence's coordinates, and say so, so downstream nodes
            # (fuse start / blend seam / slice) line up with the loaded clip.
            cut_final = max(0, cut_final - offset)
            redraw = max(0, redraw - offset)
            note = f"{note}; frame numbers shifted by -{offset} for the partial load"
        report_out = json.dumps(
            {"cut_frame": cut_final, "redraw_start": redraw,
             "frames": int(out_images.shape[0]), "frames_source": src_side,
             "cut_source": note, "manual_mode": manual},
            ensure_ascii=False,
        )
        return io.NodeOutput(out_images, cut_final, cut_final - 1, redraw, report_out)


class MiniMaxH3SeamRepair(io.ComfyNode):
    """One node for the whole repair chain: slice -> fuse -> blend -> dissolve.

    Wiring is deliberately minimal:
      images  <- decoder frames (optional when video_path is set)
      bridge  <- the redraw branch's decoded frames (optional)
      report  <- the upscale report (optional; parsed for the cut frame)
      images  -> video save
      ref_slice -> the redraw branch's ref_videos (5-frame slice)
    Everything else (cut/seam/redraw frames, offsets, ramps) is computed here.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeamRepair",
            display_name="MiniMax H3 Seam Repair (One Node)",
            category="h3_hardcut/repair",
            inputs=[
                io.Image.Input("images", optional=True,
                               tooltip="主片帧（#16 frames）。留着快速验证：只 bypass #16 + 填 video_path"),
                io.Image.Input("bridge", optional=True,
                               tooltip="桥段重绘分支解码后的帧（#69）。桥段组 bypass 时缺参 = 自动跳过融合"),
                io.String.Input("report", optional=True, default="",
                                tooltip="#40 的 report（JSON）——auto 模式解析切点"),
                io.Combo.Input("source_mode", options=["auto", "manual"], default="auto",
                               tooltip="auto = 从 report 解析切点/重绘起点；manual = 用下面手填值"),
                io.Int.Input("manual_cut_frame", default=69, min=1,
                             tooltip="手动切点帧（manual 生效）——缝在它与前一帧之间"),
                io.Int.Input("manual_redraw_start", default=68, min=0,
                             tooltip="手动重绘/切片起点帧（manual 生效）"),
                io.Int.Input("redraw_frames", default=5, min=5, max=39,
                             tooltip="给桥段的参考切片长度（H3 锚定合法长度 5/22/39）"),
                io.Combo.Input("fuse_side", options=["after", "before"], default="after",
                               tooltip="桥段融合权重方向：after = 贴切点最强、沿帧递减"),
                io.Float.Input("fuse_min", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("fuse_max", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("blend_strength", default=0.0, min=0.0, max=0.5, step=0.01,
                               tooltip="缝磨平强度（0 = 关；0.3 ≈ 跳变降 40%）"),
                io.Combo.Input("blend_mode", options=["both", "prev", "next"], default="both"),
                io.Int.Input("dissolve_start", default=0, min=0,
                             tooltip="崩帧叠化区间起点（0 = 关）"),
                io.Int.Input("dissolve_end", default=0, min=0,
                             tooltip="崩帧叠化区间末帧"),
                io.String.Input("video_path", default="",
                                tooltip="★ 快速验证：填成片路径即从它加载帧（配合只 bypass #16）"),
                io.Int.Input("video_start", default=0, min=0),
                io.Int.Input("video_frames", default=0, min=0, tooltip="0 = 全部"),
            ],
            outputs=[
                io.Image.Output("images", tooltip="修复后的帧序列 → 视频保存节点"),
                io.Image.Output("ref_slice", tooltip="给桥段的参考切片 → ReferenceToVideo.ref_videos"),
                io.String.Output("report", tooltip="本次都做了什么（解析 + 各步状态）"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images=None, bridge=None, report="", source_mode="auto",
                manual_cut_frame=69, manual_redraw_start=68, redraw_frames=5,
                fuse_side="after", fuse_min=0.0, fuse_max=1.0,
                blend_strength=0.0, blend_mode="both",
                dissolve_start=0, dissolve_end=0,
                video_path="", video_start=0, video_frames=0):
        steps = []
        # 1. frame source
        path = (video_path or "").strip()
        offset = 0
        if path:
            offset = max(0, int(video_start))
            frames = load_video_frames(path, offset, int(video_frames))
            steps.append(f"frames from {path} ({int(frames.shape[0])}, offset {offset})")
        elif images is not None:
            frames = images
            steps.append(f"frames from upstream ({int(frames.shape[0])})")
        else:
            raise ValueError("SeamRepair: no frames - wire the decoder or set video_path")

        # 2. cut / redraw frames
        parsed, src_tag = parse_boundary(report)
        manual = source_mode == "manual"
        if manual or parsed is None:
            cut = int(manual_cut_frame)
            redraw = int(manual_redraw_start)
            steps.append("cut/redraw: " + ("manual" if manual else "manual (auto found nothing)"))
        else:
            cut = parsed
            redraw = parsed - 1
            steps.append(f"cut/redraw: auto from {src_tag} (cut {cut})")
        if offset:
            cut = max(0, cut - offset)
            redraw = max(0, redraw - offset)
            steps.append(f"shifted by -{offset} for the partial load")

        # 3. reference slice for the redraw branch
        n_ref = int(redraw_frames)
        s = max(0, min(int(redraw), max(0, int(frames.shape[0]) - 1)))
        ref_slice = frames[s:s + n_ref]
        if int(ref_slice.shape[0]) < n_ref:
            steps.append(f"WARN ref_slice short: {int(ref_slice.shape[0])}/{n_ref} frames")

        # 4. fuse the bridge in
        out = frames
        if bridge is not None and bridge.numel() > 0:
            n = min(int(bridge.shape[0]), int(frames.shape[0]) - int(cut))
            out = fuse_frames(frames, bridge, int(cut), fuse_side,
                              float(fuse_min), float(fuse_max))
            steps.append(f"fuse: {n} frame(s) from the bridge, side={fuse_side}")
        else:
            steps.append("fuse: skipped (no bridge - bridge group bypassed)")

        # 5. seam blend
        seam = max(0, int(cut) - 1)
        if float(blend_strength) > 0 and seam < out.shape[0] - 1:
            a, b = blend_pair(out[seam], out[seam + 1], float(blend_strength), blend_mode)
            out = out.clone()
            out[seam], out[seam + 1] = a, b
            steps.append(f"blend: seam {seam}|{seam + 1} at {float(blend_strength):.2f}")
        else:
            steps.append("blend: off")

        # 6. dissolve a corrupted span
        ds, de = int(dissolve_start), int(dissolve_end)
        if ds > 0 and de > ds and de < out.shape[0] - 1:
            out = dissolve_span(out, ds, de)
            steps.append(f"dissolve: frames {ds}..{de}")
        elif ds > 0:
            steps.append("dissolve: span invalid - skipped")

        report_out = json.dumps(
            {"cut_frame": int(cut), "seam_frame": seam, "redraw_start": s,
             "steps": steps}, ensure_ascii=False, indent=1)
        return io.NodeOutput(out, ref_slice, report_out)


SCHEDULERS = ["simple", "beta", "normal", "sgm_uniform", "karras",
              "exponential", "ddim_uniform"]
SAMPLERS = ["res_multistep", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
            "uni_pc", "heun", "ddim"]


class MiniMaxH3RedrawBridge(io.ComfyNode):
    """The redraw branch in one node: reference conditioning + sampling + decode.

    Four wires in (model / clip / vae / ref_slice) and one out (images) - the
    guider, scheduler, sampler, noise and decode are all created inside and
    are not meant to be tuned per run.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RedrawBridge",
            display_name="MiniMax H3 Redraw Bridge (One Node)",
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="LoRA 链尾的 MODEL（与主片同源）"),
                io.Clip.Input("clip", tooltip="Qwen3-VL 文本编码器"),
                io.Vae.Input("vae", tooltip="video VAE（参考与解码共用）"),
                io.Image.Input("ref_slice", optional=True,
                               tooltip="参考片段（来自 SeamRepair 的 ref_slice，5 帧）"),
                io.String.Input(
                    "prompt", multiline=True,
                    default="Redraw the content of the reference video clip: the same "
                            "subjects keep their action going, motion stays smooth and "
                            "continuous with no cuts, matching the clip frame by frame.",
                ),
                io.Int.Input("width", default=1664, min=64, max=4096, step=32),
                io.Int.Input("height", default=928, min=64, max=4096, step=32),
                io.Int.Input("length", default=5, min=5, max=39, step=1,
                             tooltip="输出帧数（合法值 5/22/39 由模型侧再吸附）"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Int.Input("steps", default=4, min=1, max=60, tooltip="= 主片步数（Turbo 则 4）"),
                io.Combo.Input("scheduler", options=SCHEDULERS, default="simple"),
                io.Combo.Input("sampler_name", options=SAMPLERS, default="res_multistep"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF,
                             tooltip="固定种子，便于对比；改它就换一次重绘"),
            ],
            outputs=[
                io.Image.Output("images", tooltip="重绘帧 → SeamRepair.bridge"),
                io.String.Output("report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, vae, ref_slice=None, prompt="", width=1664, height=928,
                length=5, ref_image_size="match", steps=4, scheduler="simple",
                sampler_name="res_multistep", denoise=1.0, seed=0):
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise,
            SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        from nodes import VAEDecode

        if ref_slice is None:
            raise ValueError(
                "RedrawBridge: ref_slice is empty - wire it from SeamRepair.ref_slice "
                "(or point SeamRepair.video_path at a clip)"
            )
        cond = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=int(width), height=int(height),
            length=int(length), ref_image_size=ref_image_size, vae=vae,
            audio_vae=None, ref_images={},
            ref_videos={"ref_video_0": ref_slice}, ref_video_audios={}, ref_audios={},
        )
        positive, latent = cond[0], cond[1]
        noise = RandomNoise.execute(int(seed))[0]
        guider = BasicGuider.execute(model, positive)[0]
        sigmas = BasicScheduler.execute(model, scheduler, int(steps), float(denoise))[0]
        sampler = KSamplerSelect.execute(sampler_name)[0]
        out_latent, _denoised = SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, latent)[0:2]
        images = VAEDecode().decode(vae, out_latent)[0]
        report = json.dumps(
            {"frames": int(images.shape[0]), "size": [int(width), int(height)],
             "steps": int(steps), "seed": int(seed), "sampler": sampler_name,
             "scheduler": scheduler}, ensure_ascii=False, indent=1)
        return io.NodeOutput(images, report)


class MiniMaxH3RepairExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SeamBlend,
            MiniMaxH3SeamFuse,
            MiniMaxH3SeamDissolve,
        ]


def comfy_entrypoint():
    return MiniMaxH3RepairExtension()
