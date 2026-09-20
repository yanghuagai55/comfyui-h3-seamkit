"""All-in-one seam repair node.

Enable it to repair (it runs the redraw model internally), bypass it for a
plain render (nothing runs).  Frame sizes are taken from the frames, the
slice length and the redraw length are the same number, and video_path
lets you test against an existing clip without generating one.
"""

import json

from comfy_api.latest import io, ComfyExtension

from .nodes_repair import (
    CATEGORY,
    blend_pair,
    dissolve_span,
    fuse_frames,
    load_video_frames,
    parse_boundary,
)


def _redraw(model, clip, vae, ref_slice, prompt, width, height, length, steps, seed):
    """Reference conditioning -> sample -> decode, using the official classes."""
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from nodes import VAEDecode

    cond = MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=prompt, width=int(width), height=int(height),
        length=int(length), ref_image_size="match", vae=vae, audio_vae=None,
        ref_images={}, ref_videos={"ref_video_0": ref_slice},
        ref_video_audios={}, ref_audios={},
    )
    positive, latent = cond[0], cond[1]
    noise = RandomNoise.execute(int(seed))[0]
    guider = BasicGuider.execute(model, positive)[0]
    sigmas = BasicScheduler.execute(model, "simple", int(steps), 1.0)[0]
    sampler = KSamplerSelect.execute("res_multistep")[0]
    out_latent = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
    return VAEDecode().decode(vae, out_latent)[0]


class MiniMaxH3SeamRepairAll(io.ComfyNode):
    """slice -> redraw -> fuse -> blend -> dissolve, in one node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SeamRepair",
            display_name="MiniMax H3 Seam Repair (All-in-One)",
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="LoRA 链尾的 MODEL（重绘与主片同源）"),
                io.Clip.Input("clip", tooltip="Qwen3-VL 文本编码器"),
                io.Vae.Input("vae", tooltip="video VAE（重绘参考/解码共用）"),
                io.Image.Input("images", optional=True,
                               tooltip="主片帧（#16 frames）。留空 + 填 video_path = 用本地视频测试"),
                io.String.Input("report", optional=True, default="",
                                tooltip="#40 的 report —— auto 模式据此取切点/切片起点"),
                io.String.Input(
                    "prompt", multiline=True,
                    default="Redraw the content of the reference video clip: the same "
                            "subjects keep their action going, motion stays smooth and "
                            "continuous with no cuts, matching the clip frame by frame.",
                    tooltip="重绘提示词（描述参考片段里正在发生的事）",
                ),
                io.Int.Input("redraw_frames", default=5, min=5, max=39,
                             tooltip="重绘帧数 —— 切片长度与生成长度共用这一个值"),
                io.Int.Input("steps", default=4, min=1, max=60,
                             tooltip="= 主片步数（Turbo 则 4）"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF,
                             tooltip="固定种子便于对比；改它换一次重绘"),
                io.Combo.Input("source_mode", options=["auto", "manual"], default="auto",
                               tooltip="auto = 用 report 解析切点；manual = 用手填值"),
                io.Int.Input("manual_cut_frame", default=69, min=1,
                             tooltip="手动切点帧（manual 生效）"),
                io.Int.Input("manual_redraw_start", default=68, min=0,
                             tooltip="手动切片/重绘起点帧（manual 生效）"),
                io.Combo.Input("fuse_side", options=["after", "before"], default="after",
                               tooltip="重绘片段融合权重方向：after = 贴切点最强、沿帧递减"),
                io.Int.Input("fuse_offset", default=0, min=-17, max=17, step=1,
                             tooltip="重绘片段的插入位置微调（帧）。默认 0 = 从切片起点\n"
                                     "（切点-1）开始 —— 与参考片段逐帧对齐；片段整体偏后时填负数\n"
                                     "（如 -1）往前挪"),
                io.Float.Input("fuse_min", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("fuse_max", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("blend_strength", default=0.0, min=0.0, max=0.5, step=0.01,
                               tooltip="缝磨平强度（0 = 关；0.3 ≈ 跳变降 40%）"),
                io.Combo.Input("blend_mode", options=["both", "prev", "next"], default="both"),
                io.Int.Input("dissolve_start", default=0, min=0,
                             tooltip="崩帧叠化区间起点（0 = 关）"),
                io.Int.Input("dissolve_end", default=0, min=0),
                io.String.Input("video_path", default="",
                                tooltip="★ 测试用：填一个成片路径即从它加载帧（images 可留空）"),
                io.Int.Input("video_start", default=0, min=0),
                io.Int.Input("video_frames", default=0, min=0, tooltip="0 = 全部"),
            ],
            outputs=[
                io.Image.Output("images", tooltip="修复后的帧序列 → 视频保存节点"),
                io.String.Output("report", tooltip="本次都做了什么"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model=None, clip=None, vae=None, images=None, report="",
                prompt="", redraw_frames=5, steps=4, seed=0, source_mode="auto",
                manual_cut_frame=69, manual_redraw_start=68,
                fuse_side="after", fuse_offset=0, fuse_min=0.0, fuse_max=1.0,
                blend_strength=0.0, blend_mode="both",
                dissolve_start=0, dissolve_end=0,
                video_path="", video_start=0, video_frames=0):
        log = []
        # 1. frame source
        path = (video_path or "").strip()
        offset = 0
        if path:
            offset = max(0, int(video_start))
            frames = load_video_frames(path, offset, int(video_frames))
            log.append(f"frames from {path} ({int(frames.shape[0])}, offset {offset})")
        elif images is not None:
            frames = images
            log.append(f"frames from upstream ({int(frames.shape[0])})")
        else:
            raise ValueError("SeamRepair: no frames - wire the decoder or set video_path")
        total = int(frames.shape[0])

        # 2. cut / slice start
        parsed, tag = parse_boundary(report)
        manual = source_mode == "manual"
        if manual or parsed is None:
            cut = int(manual_cut_frame)
            redraw = int(manual_redraw_start)
            log.append("cut/redraw: manual" if manual
                       else "cut/redraw: manual (auto found nothing)")
        else:
            cut = parsed
            redraw = parsed - 1
            log.append(f"cut/redraw: auto from {tag} (cut {cut})")
        if offset:
            cut = max(0, cut - offset)
            redraw = max(0, redraw - offset)
            log.append(f"shifted by -{offset} for the partial load")

        # 3. redraw
        n = int(redraw_frames)
        if model is None or clip is None or vae is None:
            raise ValueError("SeamRepair: wire model / clip / vae to enable the redraw")
        s = max(0, min(redraw, max(0, total - 1)))
        ref_slice = frames[s:s + n]
        if int(ref_slice.shape[0]) < 5:
            raise ValueError(
                f"SeamRepair: redraw slice has only {int(ref_slice.shape[0])} frame(s) "
                f"(start {s}) - H3 needs at least 5"
            )
        height, width = int(ref_slice.shape[1]), int(ref_slice.shape[2])
        log.append(f"slice: frames {s}..{s + int(ref_slice.shape[0]) - 1} "
                   f"@ {width}x{height} (auto size)")
        redrawed = _redraw(model, clip, vae, ref_slice, prompt, width, height,
                           int(ref_slice.shape[0]), int(steps), int(seed))
        log.append(f"redraw: {int(redrawed.shape[0])} frames, {int(steps)} steps, seed {int(seed)}")

        # 4. fuse: the redraw's first frame lines up with the slice's first
        #    frame, so the insert starts at the slice start (not at the cut);
        #    fuse_offset nudges it further (+/- frames) when needed.
        insert_at = max(0, s + int(fuse_offset))
        out = fuse_frames(frames, redrawed, int(insert_at), fuse_side,
                          float(fuse_min), float(fuse_max))
        log.append(f"fuse: insert at frame {insert_at} (slice start {s} + offset "
                   f"{int(fuse_offset)}), {min(int(redrawed.shape[0]), total - int(insert_at))} "
                   f"frame(s), side={fuse_side}")
        if int(out.shape[0]) != total:
            log.append(f"WARN frame count changed: {total} -> {int(out.shape[0])}")

        # 5. seam blend
        seam = max(0, int(cut) - 1)
        if float(blend_strength) > 0 and seam < out.shape[0] - 1:
            a, b = blend_pair(out[seam], out[seam + 1], float(blend_strength), blend_mode)
            out = out.clone()
            out[seam], out[seam + 1] = a, b
            log.append(f"blend: seam {seam}|{seam + 1} at {float(blend_strength):.2f}")
        else:
            log.append("blend: off")

        # 6. dissolve
        ds, de = int(dissolve_start), int(dissolve_end)
        if ds > 0 and de > ds and de < out.shape[0] - 1:
            out = dissolve_span(out, ds, de)
            log.append(f"dissolve: frames {ds}..{de}")
        elif ds > 0:
            log.append("dissolve: span invalid - skipped")

        report_out = json.dumps(
            {"frames_in": total, "frames_out": int(out.shape[0]),
             "bridge_frames": int(redrawed.shape[0]),
             "cut_frame": int(cut), "slice_start": s, "fuse_at": insert_at,
             "redraw_frames": n, "size": [width, height], "steps": log},
            ensure_ascii=False, indent=1)
        return io.NodeOutput(out, report_out)


class MiniMaxH3RepairAllExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SeamRepairAll]


def comfy_entrypoint():
    return MiniMaxH3RepairAllExtension()
