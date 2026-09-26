# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Hard-cut nodes for MiniMax H3 two-pass chunked upscale.

Node 1 (`MiniMaxH3HardCutPlan`) turns a duration plus up to four cut times into
a plan the upscale executor accepts verbatim, with `temporal_overlap_frames = 0`
so each window is sampled independently and appended without any blending.  A
cut slot left at `-1` means "no cut here", so an 8 s single cut is
`cut_1 = 4.25, cut_2 = cut_3 = cut_4 = -1`.

Node 2 (`MiniMaxH3HardCutValidate`) checks a hand-written R2V prompt against
that plan: does it describe the same number of shots, does it forbid cutting,
and does every `At MM:SS.mmm` timestamp land on a frame the executor can
actually split on?  It passes the prompt through, so it can sit inline between
the text node and the conditioning nodes and refuse to start a 30 minute render
that was never going to line up.

Node 3 (`MiniMaxH3HardCutShotPrompt`, optional) is the reverse — it *writes* the
`detailed_description` block from a shot list, for when you would rather not
hand-copy timestamps.
"""

from __future__ import annotations

import re

from comfy_api.latest import io

from .bridge import (
    AUDIO_POLICIES,
    PLAN_TYPE_STRING,
    PASS2_TYPE_STRING,
    PRECISIONS,
    RELEASE_POLICIES,
    build_hardcut_plan,
    upscaler_options,
)
from .hardcut_math import (
    CUT_SLOTS,
    FPS,
    FRAME_GRID,
    MAX_LISTED,
    MAX_SECONDS,
    auto_plan,
    auto_prompt,
    cuts_from_inputs,
    format_auto_report,
    format_report,
    geometry_from_plan,
    plan_hard_cut,
    resolution_for,
    shift_shot_times,
    timecode,
    validate_prompt,
)

# Aspect ratios are ComfyUI's own table, so a canvas picked here matches what a
# ResolutionSelector would have produced.  The fallback keeps the node usable
# when the core extras are not importable (standalone maths tests, etc.).
_FALLBACK_ASPECTS = {
    "1:1 (Square)": (1, 1),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Vertical)": (9, 16),
    "4:3 (Standard)": (4, 3),
    "3:4 (Portrait)": (3, 4),
    "3:2 (Classic)": (3, 2),
    "2:3 (Portrait Classic)": (2, 3),
    "21:9 (Ultrawide)": (21, 9),
}
_DEFAULT_ASPECT = "16:9 (Widescreen)"


def aspect_ratios() -> dict[str, tuple[float, float]]:
    try:
        from comfy_extras.nodes_resolution import ASPECT_RATIOS

        if ASPECT_RATIOS:
            return dict(ASPECT_RATIOS)
    except Exception:
        pass
    return dict(_FALLBACK_ASPECTS)


def aspect_options() -> list[str]:
    """Every ratio, default included — a Combo default must be one of its options."""
    return list(aspect_ratios())


def default_aspect() -> str:
    return _DEFAULT_ASPECT if _DEFAULT_ASPECT in aspect_ratios() else aspect_options()[0]

CATEGORY = "MiniMax H3/SeamKit"
PLAN_TYPE = io.Custom(PLAN_TYPE_STRING)
NO_CUT = -1.0

_SHOT_SPLIT = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)

_CUT_TOOLTIP = (
    "Cut time in seconds. -1 means 'no cut here' — a slot only produces a cut "
    "when it is a positive number, so a clip with one cut leaves the other three "
    "slots at -1. Reachable cut frames are multiples of 17 frames (0.708s), so "
    "the closest achievable time is reported instead of failing."
)

_CUT_N_TOOLTIP = (
    "切点：第 n 个 17 帧块 —— 实际切在 n x 17 帧处。\n"
    "例：6 -> 102f (4.25s)；12 -> 204f (8.5s)；18 -> 306f (12.75s)。\n"
    "四个槽填不同的 n 就是不等长分段（如 6 / 12 / 18）。\n"
    "**-1 = 不切**（该槽留空）。填 n 而不是秒：永远是 17 的倍数，天然合法。"
)


def _split_shots(text: str) -> list[str]:
    return [part.strip() for part in _SHOT_SPLIT.split(text or "") if part.strip()]


def _parse_cuts(text: str) -> list[float]:
    out: list[float] = []
    for token in (text or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return sorted(set(out))


class MiniMaxH3HardCutPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutPlan",
            display_name="MiniMax H3 Hard-Cut Plan",
            description=(
                "Plan a two-pass chunked upscale whose window boundary is a HARD CUT "
                "instead of a blend. Sets temporal_overlap_frames = 0, so every window "
                "is sampled independently and appended back-to-back: no cross-fade, no "
                "latent endpoint shift, no colour mix.\n\n"
                "Up to four cut times go into cut_1..cut_4 and a slot left at -1 means "
                "'no cut there'. Reachable cuts sit on token edges (multiples of "
                "17 frames are the safe picks), so the "
                "closest achievable time is reported together with a chunk ladder: "
                "chunk_step walks the window length in 17-frame steps from the value "
                "the cut slots imply, and the report lists every step's windows, cut "
                "times, cut frames and load. With all four slots at -1 the clip stays a "
                "single window (no cut at all).\n\n"
                "Outputs a plan that plugs straight into the upscale pack's Chunked "
                "Two-Pass Upscale node."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Float.Input(
                    "total_seconds",
                    default=8.0,
                    min=1.0,
                    max=MAX_SECONDS,
                    step=0.01,
                    tooltip="Clip length in seconds. Snapped to the 17n+5 frame grid.",
                ),
                io.Int.Input(
                    "cut_1",
                    default=-1,
                    min=-1,
                    max=256,
                    step=1,
                    tooltip=_CUT_N_TOOLTIP,
                ),
                io.Int.Input(
                    "cut_2",
                    default=-1,
                    min=-1,
                    max=256,
                    step=1,
                    tooltip=_CUT_N_TOOLTIP,
                ),
                io.Int.Input(
                    "cut_3",
                    default=-1,
                    min=-1,
                    max=256,
                    step=1,
                    tooltip=_CUT_N_TOOLTIP,
                ),
                io.Int.Input(
                    "cut_4",
                    default=-1,
                    min=-1,
                    max=256,
                    step=1,
                    tooltip=_CUT_N_TOOLTIP,
                ),
                io.Int.Input(
                    "chunk_step",
                    default=0,
                    min=-8,
                    max=8,
                    step=1,
                    tooltip=(
                        "RELATIVE window-length step, 17 frames each, from what the cut "
                        "slots imply: 0 keeps it, +1 lengthens the window by 17 frames, "
                        "-1 shortens it. Every step re-places the cuts, so the ladder in "
                        "the report is what to read before committing. NOT the Auto "
                        "node's chunk_step, which is an absolute per-segment cap."
                    ),
                    advanced=True,
                ),
                io.Float.Input(
                    "canvas_megapixels",
                    default=1.5,
                    min=0.2,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "Second-pass canvas size, only used for the load estimate. "
                        "Keep it in sync with the ResolutionSelector you actually feed "
                        "the high-resolution conditioning."
                    ),
                ),
                io.Combo.Input("model_name", options=upscaler_options()),
                io.Int.Input("target_width", default=1664, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=928, min=32, max=16384, step=32),
                io.Combo.Input("precision", options=list(PRECISIONS), default="bf16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_after",
                    advanced=True,
                ),
                io.Float.Input(
                    "anchor_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Kept for interface compatibility. With overlap 0 the executor "
                        "never takes the anchor branch, so this value has no effect."
                    ),
                    advanced=True,
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=list(AUDIO_POLICIES),
                    default="joint_av_preserve_input",
                    advanced=True,
                ),
                io.Int.Input(
                    "cut_offset_frames",
                    default=-10,
                    min=-68,
                    max=68,
                    step=1,
                    tooltip=(
                        "★ 切点偏移（帧）＝ **实测画面突变帧位 − 切点帧位**（手动路线专用）。\n"
                        "**负值 = 模型比切点提前起转**（实测常见 −10：8s 片 92−102、"
                        "10s 片 109−119）；正值 = 偏晚。\n"
                        "报告据此给出每个切点「模型实际转镜的帧」＝ 切点 + 本值，"
                        "写作时把上一镜的动作收在那一帧。\n"
                        "**只影响报告标注，不改画面**（分割点 ≠ 剪辑点）。\n"
                        "自动版不用这个 —— `#40` 的 `auto_seam_hunt` 会在 latent 上自己量，"
                        "并用 `#56` 的 `seam_tolerance_frames` 核对。"
                    ),
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=0,
                    min=0,
                    max=2048,
                    step=17,
                    tooltip=(
                        "★ 段间重叠（帧，17 的倍数）。0 = 硬切（现在的行为）。\n"
                        "> 0 时每段的**起点向前回看本值帧**，采样器把该段第一个 token "
                        "锚在上一段已生成的结果上（强度 = anchor_strength），接缝因此"
                        "不再是两条独立结果的硬拼。\n"
                        "**送入采样器的值会被自动压到 < chunk**（一整个窗口就没内容可生成了）。\n"
                        "负载按 (最长段 + 本值) 算：15s/1.5MP 下 **17 已经到 178.5，34 会爆**。"
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                PLAN_TYPE.Output("plan"),
                io.String.Output("cut_seconds"),
                io.String.Output("cut_report"),
            ],
        )

    # NOTE: the offset knob lives here (hand-written route), not on the auto
    # node: auto mode has #40 hunt the model's real change in the latent, so a
    # hand-filled number would just be a second, competing source of truth.

    @classmethod
    def execute(
        cls,
        total_seconds: float,
        cut_1: int,
        cut_2: int,
        cut_3: int,
        cut_4: int,
        chunk_step: int,
        canvas_megapixels: float,
        model_name: str,
        target_width: int,
        target_height: int,
        precision: str,
        release_policy: str,
        anchor_strength: float,
        second_pass_audio_policy: str,
        cut_offset_frames: int,
        overlap_frames: int = 0,
    ):
        # cut slots are 17-frame BLOCK COUNTS: n x 17 = the cut frame.
        # -1 (or anything < 1) means "no cut in this slot".  Different n
        # values give unequal segments for free, and every value is on the
        # 17-frame grid by construction so nothing needs snapping.
        slots = [cut_1, cut_2, cut_3, cut_4][:CUT_SLOTS]
        seg_frames = [float(int(n) * FRAME_GRID) for n in slots if int(n) > 0] or None
        cuts = []
        _ov = max(0, int(overlap_frames))
        info = plan_hard_cut(
            total_seconds, cuts, canvas_megapixels, chunk_step, seg_frames,
            overlap=_ov,
        )

        # The plan carries `temporal_chunk_frames` for the upstream builder's
        # own 17-multiple check; in unequal mode the real windowing comes from
        # `hardcut.segment_frames`, so snap the longest window up to a legal
        # multiple rather than feeding 226 straight in.
        plan_chunk = info["chunk"]
        if info["cut_mode"] == "frames" and plan_chunk % FRAME_GRID:
            plan_chunk = (plan_chunk // FRAME_GRID + 1) * FRAME_GRID

        plan, used_upstream = build_hardcut_plan(
            model_name=model_name,
            target_width=target_width,
            target_height=target_height,
            chunk_frames=plan_chunk,
            anchor_strength=anchor_strength,
            overlap_frames=max(0, int(overlap_frames)),
            precision=precision,
            release_policy=release_policy,
            second_pass_audio_policy=second_pass_audio_policy,
            geometry={
                "total_seconds": info["total_seconds"],
                "total_frames": info["total_frames"],
                "chunk_frames": plan_chunk,
                "overlap_frames": max(0, int(overlap_frames)),
                "cut_frames": list(info["cut_frames"]),
                "cut_seconds": list(info["actual_cuts"]),
                "segment_frames": list(info["segment_frames"]),
            },
        )

        notes = []
        if not used_upstream:
            notes.append(
                "WARNING: the upscale pack (comfyui-minimax-h3-audio) was not found in "
                "this process; a built-in fallback plan was emitted. Load that pack "
                "before running."
            )
        if info["chunk_step"]:
            notes.append(
                f"chunk_step {info['chunk_step']:+d} moved the window length from "
                f"{info['base_chunk']}f to {info['chunk']}f, so the cut times above are "
                "NOT the ones in the slots. Set chunk_step back to 0 to honour them."
            )
        if cuts and len(info["actual_cuts"]) != len(cuts):
            notes.append(
                f"the slots ask for {len(cuts)} cut(s) but a {info['chunk']}f window "
                f"produces {len(info['actual_cuts'])} — the model and the executor would "
                "cut a different number of times. Fix chunk_step or the slot values."
            )
        if not info["tail_ok"]:
            notes.append(
                f"The final window is only {info['tail_frames']}f "
                f"({info['tail_seconds']:.3f}s); anything under "
                f"{info['min_tail_frames']}f (2s) is not worth a cut. Take a step from "
                "the ladder, or move a cut slot."
            )
        if info["load"]["verdict"] != "SAFE":
            notes.append(
                f"Load verdict is {info['load']['verdict']} — lower canvas_megapixels to "
                f"<= {info['max_canvas_mp']:.3f} MP or take a smaller chunk step."
            )
        notes.append(
            "Reminder: feed the same second-pass canvas to the high-resolution "
            "conditioning, otherwise the shape contract will reject the run."
        )

        report = format_report(info, "\n".join(notes))

        # Where the model actually turns is not something this node can know - it
        # only sees the prompt.  The number below is the measured offset between
        # the planned cut and the picture change, so the action of the previous
        # shot can be written to resolve at the frame the model really turns on
        # (the auto route gets the same information from #40's auto_seam_hunt).
        offset = int(cut_offset_frames)
        if info["cut_frames"] and offset:
            report += "\n\n=== 模型实际转镜帧（切点 + 偏移）==="
            report += (
                f"\n  偏移 = {offset:+d} 帧（= 实测突变帧 − 切点帧；负 = 模型提前起转）"
            )
            for c in info["cut_frames"]:
                turn = max(0, min(int(info["total_frames"]), int(c) + offset))
                report += (
                    f"\n    cut f{int(c)} ({int(c) / FPS:.3f}s)"
                    f"  ->  model turns at f{turn} ({turn / FPS:.3f}s)"
                )
            report += "\n  （只影响本报告标注，不改画面；自动版请看 #40 的 auto_seam_hunt）"
        elif info["cut_frames"]:
            report += "\n\n=== 模型实际转镜帧 ===\n  （cut_offset_frames = 0，不标注）"

        cuts_csv = ",".join(f"{value:.3f}" for value in info["actual_cuts"])
        return io.NodeOutput(plan, cuts_csv, report)


class MiniMaxH3HardCutAuto(io.ComfyNode):
    """One-node automatic planner + prompt writer (replaces Plan + Validate)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutAuto",
            display_name="MiniMax H3 Hard-Cut Auto (Plan + Prompt)",
            description=(
                "One node for the whole hard-cut setup: it plans the split, checks "
                "your prompt against it, and doubles as both resolution selectors and "
                "the frame-count expression. Give the clip length and the chunk step; "
                "it picks the fewest, most evenly-spread cuts (up to 10 segments, "
                "variance minimised) and raises if none fit.\n\n"
                "Outputs: `plan` for the upscale executor, `prompt` (passed through "
                "and checked, or a six-section skeleton with every [Shot N] timestamp "
                "already filled in), `report` (split + check + errors, also printed on "
                "the node), `first_width/height` for the first-pass conditioning, "
                "`second_width/height` for the second-pass one, and `length` — the "
                "17n+5 frame count both conditioning nodes need."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    force_input=True,
                    tooltip=(
                        "Wire your R2V prompt in here (from the multiline text node). "
                        "It is checked against the split this node picked and passed "
                        "straight through to the conditioning nodes. Leave it unlinked "
                        "and the node emits a six-section skeleton with every [Shot N] "
                        "timestamp already filled in, for you to copy out and fill with "
                        "plot."
                    ),
                ),
                io.Float.Input(
                    "total_seconds",
                    default=8.0,
                    min=1.0,
                    max=MAX_SECONDS,
                    step=0.5,
                    tooltip="Clip length in seconds. Snapped to the 17n+5 frame grid.",
                ),
                io.Boolean.Input(
                    "loose_prompt",
                    default=True,
                    tooltip="自由提示词：开启时不再强制提示词的段落结构与 At 时间戳，边界由 hunt/自适应搜索决定。关掉则恢复严格校验（提示词必须与执行器切点一致）。",
                ),
                io.Float.Input(
                    "target_segment_seconds",
                    default=5.0,
                    min=0.71,
                    max=60.0,
                    step=0.05,
                    tooltip=(
                        "想要的最长分段（秒）—— 直接写时间，不用算 17 帧块。\n"
                        "内部会取最接近的 17n 帧档位：4.25s -> 102 帧（n=6）。\n"
                        "段越长、画布越大，二采负载越高；报告里的 `load estimate` 会给"
                        "SAFE / BORDERLINE / LIKELY-OOM，看那个决定要不要调小。"
                    ),
                ),
                io.Float.Input(
                    "first_megapixels",
                    default=0.4,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip=(
                        "First-pass canvas size (replaces the first "
                        "ResolutionSelector). 0.4 MP -> 864x480 at 16:9."
                    ),
                ),
                io.Float.Input(
                    "second_megapixels",
                    default=1.5,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip=(
                        "Second-pass canvas size (replaces the second "
                        "ResolutionSelector). 1.5 MP -> 1664x928 at 16:9. Also used "
                        "for the load estimate."
                    ),
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=aspect_options(),
                    default=default_aspect(),
                    tooltip="Aspect ratio for BOTH canvases, as in ResolutionSelector.",
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=8,
                    max=128,
                    step=4,
                    tooltip="Round both canvases to this multiple (H3 wants 32).",
                    advanced=True,
                ),
                io.Combo.Input("model_name", options=upscaler_options()),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="bf16",
                    advanced=True,
                ),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_after",
                    advanced=True,
                ),
                io.Float.Input(
                    "anchor_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="No effect at overlap 0; kept for interface compatibility.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=list(AUDIO_POLICIES),
                    default="joint_av_preserve_input",
                    advanced=True,
                ),
                io.Float.Input(
                    "second_pass_sigma0",
                    default=0.30,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "二采 denoise (σ₀) — 从本节点的 `sigma0` 输出口接到 BasicScheduler 的 "
                        "`denoise`。**接缝幅度 ∝ σ₀**：调小它，段边界那道缝会更淡（代价是"
                        "二采引入的细节变少）。本机实测 0.30 能用，可试 0.20~0.25。"
                    ),
                    advanced=True,
                ),
                io.Int.Input(
                    "seam_tolerance_frames",
                    default=4,
                    min=0,
                    max=8,
                    step=1,
                    tooltip=(
                        "★ 自动找缝的容差（帧）：交给 `#40` 的 `auto_seam_hunt` 用。\n"
                        "二采在 latent 上找模型真正的转镜帧，再把它**吸附到最近的独占帧**"
                        "（17 的倍数）作为窗口边界。本值管的是**吸附后的残差**：\n"
                        "    `|边界 − 测得转镜|` ≤ 本值 → 缝正压在转镜上 → **硬切**（overlap 0）\n"
                        "    `|边界 − 测得转镜|` > 本值 → 缝落在镜头内部，硬切会露台阶 → "
                        "该缝改走**锚定 overlap**\n"
                        "    （搜索没找到转镜 / 持续性门拒绝时，同样走锚定 overlap）\n"
                        "\n"
                        "**★ 量程只有 0~8**，因为边界只能落在 17 的整数倍上 —— "
                        "转镜帧到最近合法帧的距离最大就是 17/2 = 8。所以：\n"
                        "    0 = 只有转镜正好压在网格上才硬切（很罕见）\n"
                        "    4 = 接受中间一半（**推荐**）\n"
                        "    8 = 无条件硬切（等于关掉这道门）\n"
                        "    >8 与 8 完全等价，没有意义。\n"
                        "\n"
                        "注意：这里**不再**拿它和计划切点比 —— 边界该落在哪由**测得转镜**决定，"
                        "不是由计划决定。（旧版这里是规划漂移容差，量程 17 才有意义，已废弃。）"
                    ),
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=0,
                    min=0,
                    max=2048,
                    step=17,
                    tooltip=(
                        "★ 段间重叠（帧，17 的倍数）。0 = 硬切（现在的行为）。\n"
                        "> 0 时每段的**起点向前回看本值帧**，采样器把该段第一个 token "
                        "锚在上一段已生成的结果上（强度 = anchor_strength），接缝因此"
                        "不再是两条独立结果的硬拼。\n"
                        "**送入采样器的值会被自动压到 < chunk**（一整个窗口就没内容可生成了）。\n"
                        "负载按 (最长段 + 本值) 算：15s/1.5MP 下 **17 已经到 178.5，34 会爆**。"
                    ),
                    advanced=True,
                ),
                io.Boolean.Input(
                    "auto_calm_search",
                    default=False,
                    tooltip=(
                        "★ 自适应平缓搜索（需要 `#40` 的 `auto_seam_hunt` 一起开）。\n"
                        "hunt 判定某个切点**不可靠**时（没检测到转镜，或检测到的位置离计划点"
                        "超过 `seam_tolerance_frames`），不再硬切在计划点上，而是在 **± "
                        "`calm_search_window`** 范围内找 **latent 变化最小的独占帧**，"
                        "把边界挪过去，并给那条缝开 `calm_overlap_frames` 的重叠锚定。\n"
                        "目的：接缝既不落在内容剧变处，也不用两条独立结果硬拼。"
                    ),
                ),
                io.Int.Input(
                    "calm_search_window",
                    default=34,
                    min=0,
                    max=170,
                    step=17,
                    tooltip="平缓搜索半径（帧）：在 `切点 ± 本值` 内找最平缓的独占帧。",
                    advanced=True,
                ),
                io.Int.Input(
                    "calm_overlap_frames",
                    default=17,
                    min=0,
                    max=1632,
                    step=17,
                    tooltip=(
                        "平缓缝使用的重叠（帧）。17 = 一个 token 组。\n"
                        "负载按 (最长窗 + 本值) 算——15s/1.5MP 下 17 已到 178.5，34 会爆。"
                    ),
                    advanced=True,
                ),
                io.Combo.Input(
                    "calm_policy",
                    options=["calm_overlap", "jerk_hardcut"],
                    default="calm_overlap",
                    tooltip=(
                        "hunt 判定不可靠时怎么放这条缝。"
                        "calm_overlap：边界挪到最平缓的独占帧，并给这条缝加锚定重叠（靠内容连续+锚定把缝缝住）。"
                        "jerk_hardcut：反过来，把边界放在 jerk 最高（运动最剧烈/模型最容易糊）的独占帧上并硬切，"
                        "靠运动掩蔽藏缝，不用重叠。"
                    ),
                ),
                io.Boolean.Input(
                    "profile_camera_compensate",
                    default=False,
                    tooltip="先按整数位移把每帧对齐到前一帧，再做变化剖面。纯运镜(平移/摇镜)会被读成静止，只有相对相机的运动留下。提示词里有运镜时打开它。",
                    advanced=True,
                ),
                io.Combo.Input(
                    "profile_reduce",
                    options=["mean", "max", "top-decile"],
                    default="mean",
                    tooltip="空间聚合方式：mean=全网平均(默认)；max=取最热的一点；top-decile=最热10%的均值。后两者不会把局部热点平均掉。",
                    advanced=True,
                ),
                io.Float.Input(
                    "calm_abstain_below",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip="放弃门：jerk 的对比度(max/mean)低于本值就整片不做搜索、保持计划切点。0=关闭。",
                    advanced=True,
                ),
                io.Float.Input(
                    "calm_min_quality",
                    default=0.8,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="逐缝质量门：窗内最优点的分数仍高于它就退回计划点硬切。分数以全片中位为 1.0，0.8 表示「至少比平时安静两成」。0 = 关闭。",
                    advanced=True,
                ),
                io.Int.Input(
                    "hunt_search_window",
                    default=34,
                    min=0,
                    max=170,
                    step=17,
                    tooltip="hunt 的搜索半径（帧）。模型实际转镜的位置可能离计划点很远（实测 1~17 帧，甚至更多），窗口太小就什么都检测不到，残差判据也就无从触发。0 = 用 max(seam_tolerance_frames, 34)。",
                    advanced=True,
                ),
                io.Float.Input(
                    "hunt_min_persistence",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.02,
                    tooltip="hunt 采纳门：检测到的局部变化若持久性低于本值，判为假信号（闪烁/抖动/纹理划过）不予采纳，退回计划点。0 = 关闭。",
                    advanced=True,
                ),
                io.Float.Input(
                    "calm_min_gain",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="挪动收益门：候选点必须比计划点安静至少这个比例才值得挪。避免「压线抖动」（0.09 分之差决定两种成片）。0 = 关闭，退回旧的绝对门限。",
                    advanced=True,
                ),
                io.Float.Input(
                    "calm_too_quiet_below",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="平缓搜索的下限门：窗内最优点若过于静止（分数低于本值），4 帧量化的顿挫会显眼，此时不挪边界、直接硬切。0 = 关闭。",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "hunt_persistence",
                    default=True,
                    tooltip="hunt 排序时加入「持久性」判据：真转场 = 变化后停在新状态；闪烁/抖动/纹理划过 = 变化后回到原状态。关掉则退回旧的纯局部变化排序。",
                    advanced=True,
                ),
            ],
            outputs=[
                PLAN_TYPE.Output("plan"),
                io.String.Output("prompt"),
                io.String.Output("report"),
                io.Int.Output("first_width", tooltip="First-pass width (from first_megapixels)."),
                io.Int.Output("first_height", tooltip="First-pass height."),
                io.Int.Output("second_width", tooltip="Second-pass width (from second_megapixels)."),
                io.Int.Output("second_height", tooltip="Second-pass height."),
                io.Int.Output(
                    "length",
                    tooltip="Clip length in frames, on the 17n+5 grid — feed both conditioning nodes.",
                ),
                io.Float.Output(
                    "sigma0",
                    tooltip="Raw passthrough of second_pass_sigma0 → wire into BasicScheduler.denoise.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        total_seconds: float,
        target_segment_seconds: float,
        first_megapixels: float,
        second_megapixels: float,
        aspect_ratio: str,
        multiple: int,
        model_name: str,
        precision: str,
        release_policy: str,
        anchor_strength: float,
        second_pass_audio_policy: str,
        second_pass_sigma0: float,
        seam_tolerance_frames: int,
        overlap_frames: int = 0,
        loose_prompt: bool = True,
        auto_calm_search: bool = False,
        calm_search_window: int = 34,
        calm_overlap_frames: int = 17,
        calm_policy: str = "calm_overlap",
        profile_camera_compensate: bool = False,
        profile_reduce: str = "mean",
        calm_abstain_below: float = 0.0,
        hunt_persistence: bool = True,
        calm_min_quality: float = 0.8,
        hunt_min_persistence: float = 0.8,
        hunt_search_window: int = 34,
            calm_too_quiet_below: float = 0.05,
        calm_min_gain: float = 0.15,
    ):
        w_ratio, h_ratio = aspect_ratios().get(
            aspect_ratio, aspect_ratios()[default_aspect()]
        )
        first_w, first_h = resolution_for(first_megapixels, w_ratio, h_ratio, multiple)
        second_w, second_h = resolution_for(second_megapixels, w_ratio, h_ratio, multiple)
        canvas_mp = second_w * second_h / 1_000_000.0

        # The window sizing must reserve room for the overlap the executor will
        # actually use: calm search brings its own anchored overlap, so plan the
        # segments against THAT, or a calm seam pushes the longest window past
        # the load line (15s layout: 157.5 -> 183.7).
        _effective_overlap = (
            max(0, int(calm_overlap_frames)) if auto_calm_search
            else max(0, int(overlap_frames))
        )
        # Seconds -> the nearest legal 17-frame window cap.  The report's
        # `load estimate` is the number to watch, not this conversion.
        _chunk_step = max(1, int(round(float(target_segment_seconds) * FPS / FRAME_GRID)))
        info = auto_plan(total_seconds, _chunk_step, canvas_mp,
                         overlap=_effective_overlap)

        plan_chunk = info["longest"]
        if plan_chunk % FRAME_GRID:
            plan_chunk = (plan_chunk // FRAME_GRID + 1) * FRAME_GRID

        plan, used_upstream = build_hardcut_plan(
            model_name=model_name,
            target_width=second_w,
            target_height=second_h,
            chunk_frames=plan_chunk,
            anchor_strength=anchor_strength,
            overlap_frames=max(0, int(overlap_frames)),
            precision=precision,
            release_policy=release_policy,
            second_pass_audio_policy=second_pass_audio_policy,
            geometry={
                "total_seconds": info["total_seconds"],
                "total_frames": info["total_frames"],
                "chunk_frames": plan_chunk,
                "overlap_frames": max(0, int(overlap_frames)),
                "cut_frames": list(info["cut_frames"]),
                "cut_seconds": list(info["actual_cuts"]),
                "segment_frames": list(info["segment_frames"]),
                # Threshold the upscale node uses when it hunts the model's own
                # shot change in the latent: a detection farther than this from
                # one of the planned cuts is treated as a false positive.
                "seam_tolerance": max(0, int(seam_tolerance_frames)),
            },
        )

        # geometry is a whitelist in the bridge, so put the hunt tolerance on
        # the plan directly — the upscale node reads it back from there.
        plan.setdefault("hardcut", {})["seam_tolerance"] = max(0, int(seam_tolerance_frames))
        _hc = plan.setdefault("hardcut", {})
        _hc["auto_calm_search"] = bool(auto_calm_search)
        _hc["calm_search_window"] = max(0, int(calm_search_window))
        _hc["calm_overlap_frames"] = max(0, int(calm_overlap_frames))
        _hc["calm_policy"] = str(calm_policy)
        _hc["profile_camera_compensate"] = bool(profile_camera_compensate)
        _hc["profile_reduce"] = str(profile_reduce)
        _hc["calm_abstain_below"] = max(0.0, float(calm_abstain_below))
        _hc["hunt_persistence"] = bool(hunt_persistence)
        _hc["calm_min_quality"] = max(0.0, float(calm_min_quality))
        _hc["hunt_min_persistence"] = max(0.0, float(hunt_min_persistence))
        _hc["hunt_search_window"] = max(0, int(hunt_search_window))
        _hc["calm_too_quiet_below"] = max(0.0, float(calm_too_quiet_below))
        _hc["calm_min_gain"] = max(0.0, float(calm_min_gain))
        # 让下游的 MiniMaxH3HardCutValidate 也能看到 loose 口径 —— 它的入参只有
        # prompt + plan（没有 loose_prompt 控件），不放进 plan 就会用严格模式，
        # 于是同一个提示词在 #56 是 warning、在 Validate 却是 raise（实测：整图被拦）。
        _hc["loose_prompt"] = bool(loose_prompt)
        # Canvas megapixels, so #40 can re-check the load guard after the calm
        # search moves a boundary (auto_plan sized the PLANNED windows only).
        _hc["canvas_mp"] = float(canvas_mp)

        note = ""
        if not used_upstream:
            note = (
                "WARNING: the upscale pack was not found in this process; a built-in "
                "fallback plan was emitted."
            )
        report = format_auto_report(info, note)
        report += (
            f"\ncanvas        : first {first_w}x{first_h} ({first_megapixels:g} MP)  ->  "
            f"second {second_w}x{second_h} ({canvas_mp:.3f} MP)\n"
            f"clip length   : {info['total_frames']} frames (17n+5) -> `length` output\n"
            f"sigma0        : {second_pass_sigma0:.2f}  ->  wire the `sigma0` output into "
            "BasicScheduler.denoise (seam strength scales with it)"
        )

        # Where the model actually turns is not something this node can know - it
        # only sees the prompt.  The upscale node holds the first-pass latent and
        # hunts the change there (`auto_seam_hunt`); how far a detection may stray
        # from the cuts planned here before it counts as a false positive is
        # `seam_tolerance_frames`.
        tolerance = max(0, int(seam_tolerance_frames))
        report += (
            "\n\nauto seam hunt: turn on `auto_seam_hunt` on the upscale node (#40) - it finds "
            "where the model\n  really changes shots inside the first-pass latent (no VAE decode) "
            "and moves the window boundary\n  to just BEFORE it, so the seam lands on continuous "
            f"content. Only detections within {tolerance} frame(s)\n  of the cuts above are "
            "accepted (`seam_tolerance_frames`); anything farther is ignored as a false positive."
        )

        incoming = (prompt or "").strip()
        if incoming:
            # Check the prompt against the split we just picked, then pass it on.
            result = validate_prompt(
                incoming,
                total_seconds=info["total_seconds"],
                requested_cuts=list(info["actual_cuts"]),
                canvas_mp=canvas_mp,
                chunk_frames=plan_chunk,
                overlap_frames=max(0, int(overlap_frames)),
                segment_frames=list(info["cut_frames"]),
                overlap_anchored=bool(auto_calm_search) or int(overlap_frames) > 0,
                loose=bool(loose_prompt) or bool(auto_calm_search) or int(overlap_frames) > 0,
            )
            out_prompt = incoming
            errors = result.get("errors") or []
            report = report + "\n\n" + result["report"]
            if errors:
                raise ValueError(
                    f"Hard-cut auto check failed ({len(errors)} error(s)) — the prompt "
                    "and the split this node picked disagree, so the model would not "
                    "cut where the executor cuts.\n\n" + report
                )

            # The model turns a little off the time it is given (measured -10..+1
            # frames, content- and seed-dependent) while the executor can only
            # split on the token grid — asking for a slightly different time is
            # therefore the one lever finer than the grid.  It lives here, not in
            # the prompt, so whoever writes the prompt never has to be taught
            # plugin mechanics (and the plot keeps its weight in the text).
            # Prompt time-shifting is retired: the hunt/calm search owns boundary
            # placement now, so the prompt is passed through untouched.
            out_prompt, moved = incoming, False
            if moved:
                report += (
                    "\n\nprompt time shift — only the `prompt` OUTPUT moves:\n"
                    "  the executor still splits where the plan says; the model is asked for the "
                    "time it\n  actually turns at, so the shot change and the window boundary "
                    "line up:"
                )
                for row in moved:
                    report += (
                        f"\n    {row['old']} -> {row['new']}"
                        f"   (frame {row['old_frame']} -> {row['new_frame']})"
                    )
                report += (
                    "\n  only declared shot changes move; in-shot beats keep their values, and "
                    "every\n  quote of a moved stamp (summary / shot line / soundscape) is "
                    "rewritten together."
                )
                recheck = validate_prompt(
                    out_prompt,
                    total_seconds=info["total_seconds"],
                    requested_cuts=list(info["actual_cuts"]),
                    canvas_mp=canvas_mp,
                    chunk_frames=plan_chunk,
                    overlap_frames=max(0, int(overlap_frames)),
                    segment_frames=list(info["cut_frames"]),
                    overlap_anchored=bool(auto_calm_search) or int(overlap_frames) > 0,
                    loose=bool(loose_prompt) or bool(auto_calm_search) or int(overlap_frames) > 0,
                )
                late = recheck.get("errors") or []
                if late:
                    report += (
                        "\n  WARNING: the declared times do not sit within 1 "
                        "frame of the plan's cuts — write the prompt to land on the plan, "
                        "or let the hunt/calm search move the boundary instead:\n    - "
                        + "\n    - ".join(str(e) for e in late[:MAX_LISTED])
                    )
        else:
            out_prompt = auto_prompt(info, "", "")
            report = report + (
                "\n\n(note: no prompt wired in — the `prompt` output is a six-section "
                "skeleton with the timestamps already filled in; fill it with plot, "
                "paste it into your text node and wire it back here to get it checked.)"
            )

        return io.NodeOutput(
            plan,
            out_prompt,
            report,
            first_w,
            first_h,
            second_w,
            second_h,
            info["total_frames"],
            float(second_pass_sigma0),
            ui={"text": (report,)},
        )


class MiniMaxH3HardCutValidate(io.ComfyNode):
    """Prompt check with no settings: the plan already knows the geometry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutValidate",
            display_name="MiniMax H3 Hard-Cut Prompt Check",
            description=(
                "Check a hand-written R2V prompt against the plan it is about to run "
                "with, and print the verdict on this node. Wire `prompt` from your text "
                "node and `plan` from MiniMaxH3HardCutPlan. `prompt` is passed straight "
                "through, so this node sits inline in the prompt path and its verdict "
                "cannot be skipped. `plan` may be left unlinked: the prompt is then "
                "judged on its own, without the prompt-vs-plan timing checks.\n\n"
                "Flags: missing or renamed sections, '[Shot N]' numbering gaps, wording "
                "that forbids cutting ('no cuts anywhere', 'single continuous take', "
                "'unbroken', 'in one take'), a missing 'At MM:SS.mmm' on any shot after "
                "the first, a shot count that does not match the plan's window count, a "
                "timestamp that is not on the frame the executor really cuts on, and a "
                "second-pass load past the measured OOM anchor.\n\n"
                "On any error it raises, so a run that cannot line up stops in a second "
                "instead of after half an hour. There are no settings to get wrong: the "
                "window length, the overlap and the canvas all come from the plan."
            ),
            category=CATEGORY,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    force_input=True,
                    tooltip=(
                        "Wire from your prompt text node. The value is passed through "
                        "untouched; this node only reads it."
                    ),
                ),
                PLAN_TYPE.Input(
                    "plan",
                    optional=True,
                    tooltip=(
                        "Wire from MiniMaxH3HardCutPlan.plan. The plan is ground truth "
                        "- window length, overlap, cut frames and the second-pass canvas "
                        "are all read from it, so nothing needs configuring here. Leave "
                        "it unlinked and the prompt is judged on its own (section "
                        "skeleton, shot numbering, wording, timestamp ordering) without "
                        "the prompt-vs-plan timing checks."
                    ),
                ),
            ],
            outputs=[io.String.Output("prompt")],
        )

    @classmethod
    def execute(cls, prompt, plan=None):
        if plan is not None:
            geo = geometry_from_plan(plan)
            # ★ 与 #56 的口径对齐。修之前这里 `loose` 与 `overlap_anchored` 两个都没传，
            # 而 #56 传了 —— 同一个提示词在 #56 只是 warning，到这里却变成 raise，
            # 整个图在 0.02 秒被拦住（实测：一条"单镜头连续长镜"的提示词）。
            # 本节点入参只有 prompt + plan，所以从 plan 里读（#56 已写入 hardcut 字典）。
            _hc = (plan.get("hardcut") or {}) if isinstance(plan, dict) else {}
            _anchored = bool(geo.get("overlap") or 0) > 0 or bool(_hc.get("auto_calm_search"))
            _loose = _anchored or bool(_hc.get("loose_prompt"))
            result = validate_prompt(
                prompt,
                total_seconds=geo.get("total_seconds"),
                requested_cuts=geo.get("cut_seconds") or "",
                canvas_mp=geo.get("canvas_mp"),
                chunk_frames=geo.get("chunk"),
                overlap_frames=geo.get("overlap") or 0,
                overlap_anchored=_anchored,
                segment_frames=geo.get("segment_frames"),
                loose=_loose,
            )
        else:
            result = validate_prompt(prompt)

        if not result["ok"]:
            errors = result["errors"]
            listed = errors[:MAX_LISTED]
            extra = (
                f"\n  ... and {len(errors) - MAX_LISTED} more"
                if len(errors) > MAX_LISTED
                else ""
            )
            raise ValueError(
                f"Hard-cut prompt check failed ({len(errors)} error(s)) - the prompt and "
                "the plan disagree, so the model would not cut where the executor cuts.\n"
                + "\n".join(f"  - {item}" for item in listed)
                + extra
            )

        return io.NodeOutput(prompt, ui={"text": (result["report"],)})


class MiniMaxH3HardCutShotPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutShotPrompt",
            display_name="MiniMax H3 Hard-Cut Shot Prompt (R2V)",
            description=(
                "Assemble the detailed_description block of the official R2V prompt "
                "template from a style lead plus one description per shot. Shot 2 and "
                "later receive 'At MM:SS.mmm,' timestamps taken from the hard-cut plan, "
                "so the model places its own cut on the same frame the executor splits "
                "on. Separate shots with a line containing only '---'. The official "
                "template requires the timestamp on every shot after the first."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input(
                    "style_lead",
                    multiline=True,
                    default=(
                        "The target video is in a cinematic, softly lit style with a "
                        "slightly desaturated palette."
                    ),
                    tooltip=(
                        "One or two sentences that set style and grade. The official "
                        "template places this BEFORE [Shot 1]."
                    ),
                ),
                io.String.Input(
                    "shots",
                    multiline=True,
                    default=(
                        "<Shot 1: framing, subject position and action, lighting, camera "
                        "move, sound>\n---\n<Shot 2 at the cut: same six elements>"
                    ),
                    tooltip="Separate each shot with a line containing only '---'.",
                ),
                io.String.Input(
                    "cut_seconds",
                    default="4.250",
                    tooltip=(
                        "Wire this from MiniMaxH3HardCutPlan.cut_seconds, or type times "
                        "by hand (comma separated)."
                    ),
                ),
                io.Boolean.Input(
                    "timestamp_on_first_shot",
                    default=False,
                    tooltip=(
                        "Official template omits the timestamp on [Shot 1]; enable only "
                        "if you deliberately want one."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("detailed_description"),
                io.String.Output("shot_audit"),
            ],
        )

    @classmethod
    def execute(
        cls,
        style_lead: str,
        shots: str,
        cut_seconds: str,
        timestamp_on_first_shot: bool,
    ):
        cuts = _parse_cuts(cut_seconds)
        parts = _split_shots(shots)

        if not parts:
            return io.NodeOutput(
                "",
                "No shots found. Put one description per shot, separated by a line "
                "containing only '---'.",
            )

        lines = []
        if style_lead.strip():
            lines.append(style_lead.strip())

        for index, raw in enumerate(parts):
            # Accept both shapes: a bare shot description, or a line that already
            # carries its own "[Shot N]" label — the latter is what an LLM returns
            # when you hand it the template, so do not double it up.
            labelled = re.match(r"^\s*\[Shot\s+(\d+)\]\s*(.*)$", raw, re.DOTALL)
            if labelled:
                label = f"[Shot {labelled.group(1)}]"
                body = labelled.group(2).strip()
            else:
                label = f"[Shot {index + 1}]"
                body = raw

            # Same for the timestamp: keep an "At MM:SS.mmm," the caller already
            # wrote, otherwise insert the one that matches the executor's cut.
            stamped = re.match(r"^At\s+\d{2}:\d{2}\.\d{3}\s*,", body) is not None
            needs_stamp = not (index == 0 and not timestamp_on_first_shot)

            if needs_stamp and not stamped:
                if index == 0:
                    # The first shot starts at the clip start, never at cut 1.
                    stamp = "00:00.000"
                elif cuts:
                    stamp = timecode(
                        int(round(cuts[min(index - 1, len(cuts) - 1)] * FPS))
                    )
                else:
                    stamp = "00:00.000"
                body = f"At {stamp}, {body}"

            lines.append(f"{label} {body}")

        detailed = "\n".join(lines)

        audit = []
        audit.append(f"shots            : {len(parts)}")
        audit.append(
            "hard cuts needed : "
            + str(max(0, len(parts) - 1))
            + ("  (from cut_seconds)" if cuts else "  (cut_seconds empty)")
        )
        audit.append(
            "cut timestamps   : " + (", ".join(f"{c:.3f}s" for c in cuts) if cuts else "(none)")
        )
        problems = []
        if len(parts) - 1 != len(cuts):
            problems.append(
                f"shot count vs cut count mismatch: {len(parts)} shots need "
                f"{len(parts) - 1} cuts but got {len(cuts)}. The last shot will reuse "
                "the final timestamp."
            )
        if len(cuts) > 1 and any(b <= a for a, b in zip(cuts, cuts[1:])):
            problems.append("cut_seconds must increase.")
        if not style_lead.strip():
            problems.append(
                "style_lead is empty. The official template wants a style sentence "
                "before [Shot 1]."
            )
        audit.append("")
        audit.append("AUDIT: OK" if not problems else "AUDIT: CHECK")
        for item in problems:
            audit.append(f"  - {item}")
        audit.append("")
        audit.append("Pair these times with the same cut slots on the plan node so the")
        audit.append("model's cut lands on the exact frame the executor splits on.")

        return io.NodeOutput(detailed, "\n".join(audit))


# ═══════════════════════════════════════════════════════════════════════════
# 下游二采规划（2026-09-26 拆分）
# 只影响二采的参数集中到这里，接进 MiniMaxH3HardCutUpscale 的 pass2_plan。
# 动机：规划节点挂在一采上游链里（其输出改写条件中的提示词时间戳），它身上每个
# 控件的值都进一采缓存指纹 —— 在它上面动二采参数会击穿缓存、逼一采重采样。
# 搬到本节点（纯下游）后：随便改，不再触发重采样。
# ═══════════════════════════════════════════════════════════════════════════
PASS2_TYPE = io.Custom(PASS2_TYPE_STRING)


class MiniMaxH3HardCutPass2Plan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutPass2Plan",
            display_name="MiniMax H3 Pass-2 Plan (downstream)",
            description=(
                "把所有**只影响二采**的参数集中到这一个节点，`pass2_plan` 输出接到 "
                "`MiniMaxH3HardCutUpscale` 的 `pass2_plan` 输入；`sigma0` 输出接到 "
                "BasicScheduler 的 denoise（原先从规划节点接的那个）。\n\n"
                "为什么拆出来：规划节点在一采上游链里，它身上所有控件值都会进一采缓存指纹；"
                "二采参数搬到这里后，**随便改都不再触发一采重采样**。"
            ),
            category="MiniMax H3 Hard Cut",
            is_experimental=True,
            inputs=[
                io.Float.Input("second_pass_sigma0", default=0.30, min=0.0, max=1.0, step=0.01,
                               tooltip="二采 denoise (sigma0)。接 BasicScheduler.denoise。"),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001,
                               tooltip="锚定强度（接口兼容保留）。"),
                io.Combo.Input("second_pass_audio_policy", options=list(AUDIO_POLICIES),
                               default=AUDIO_POLICIES[0], tooltip="二采音频策略。"),
                io.Int.Input("seam_tolerance_frames", default=4, min=0, max=8, step=1,
                             tooltip="hunt 残差容差（帧）。"),
                io.Int.Input("calm_search_window", default=34, min=0, max=170, step=17,
                             tooltip="平缓搜索半径（帧）。"),
                io.Combo.Input("calm_policy", options=["calm_overlap", "jerk_hardcut"],
                               default="calm_overlap", tooltip="hunt 不可靠时怎么放这条缝。"),
                io.Boolean.Input("profile_camera_compensate", default=False,
                                 tooltip="latent profile 镜头补偿。"),
                io.Combo.Input("profile_reduce", options=["mean", "max", "top-decile"], default="mean",
                               tooltip="profile 空间聚合方式。"),
                io.Float.Input("calm_abstain_below", default=0.0, min=0.0, max=10.0, step=0.05,
                               tooltip="放弃门（0=关）。"),
                io.Float.Input("calm_min_quality", default=0.8, min=0.0, max=3.0, step=0.05,
                               tooltip="逐缝质量门（0=关）。"),
                io.Boolean.Input("hunt_persistence", default=True, tooltip="hunt profile 持久性。"),
                io.Float.Input("hunt_min_persistence", default=0.8, min=0.0, max=1.0, step=0.02,
                               tooltip="hunt 采纳门（0=关）。"),
                io.Int.Input("hunt_search_window", default=34, min=0, max=170, step=17,
                             tooltip="hunt 搜索半径（帧）。"),
                io.Float.Input("calm_too_quiet_below", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="平缓搜索下限门（0=关）。"),
                io.Float.Input("calm_min_gain", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="挪动收益门（0=关）。"),
                io.Boolean.Input("auto_seam_hunt", default=False,
                                 tooltip="自动找切镜挪边界；关=分析照跑、切点回退计划位置（锚定 overlap）。"),
                io.Int.Input("anchor_tokens", default=1, min=1, max=5, tooltip="锚定 keyframe token 数。"),
                io.Int.Input("upscale_pad_tokens", default=3, min=0, max=8,
                             tooltip="上采样分块时间重叠（熔化修复，默认 3；0=旧行为）。"),
                io.Boolean.Input("seam_blend", default=False,
                                 tooltip="锚定缝 latent 线性交叉淡化（静止镜头适用；运动镜头会重影）。"),
                io.Boolean.Input("seam_redenoise", default=False, tooltip="缝窗重去噪（E-3）。"),
                io.String.Input("seam_redenoise_frames", default="",
                                tooltip="只重去噪指定帧附近的缝（逗号分隔；空=自动）。"),
                io.Int.Input("seam_window_tokens", default=10, min=4, max=30, tooltip="缝窗 token 数。"),
                io.Int.Input("seam_lock_tokens", default=3, min=1, max=10, tooltip="缝窗两侧锁定 token 数。"),
                io.Combo.Input("seam_redenoise_gate", options=["off", "auto"], default="off",
                               tooltip="门控：auto=按 busy 度跳过闹缝。"),
                io.Boolean.Input("dump_latents", default=False, tooltip="诊断：中间 latent 落盘。"),
                io.String.Input("dump_dir", default="D:\\comfyui\\_hardcut_work\\latent_dump\\",
                                tooltip="dump 输出目录。"),
                io.Boolean.Input("show_memory_log", default=True, tooltip="每窗采样后打印显存。"),
            ],
            outputs=[
                PASS2_TYPE.Output("pass2_plan"),
                io.Float.Output("sigma0"),
            ],
        )

    @classmethod
    def execute(cls, second_pass_sigma0: float = 0.30, anchor_strength: float = 0.999,
                second_pass_audio_policy: str = None, seam_tolerance_frames: int = 4,
                calm_search_window: int = 34, calm_policy: str = "calm_overlap",
                profile_camera_compensate: bool = False, profile_reduce: str = "mean",
                calm_abstain_below: float = 0.0, calm_min_quality: float = 0.8,
                hunt_persistence: bool = True, hunt_min_persistence: float = 0.8,
                hunt_search_window: int = 34, calm_too_quiet_below: float = 0.05,
                calm_min_gain: float = 0.15, auto_seam_hunt: bool = False,
                anchor_tokens: int = 1, upscale_pad_tokens: int = 3, seam_blend: bool = False,
                seam_redenoise: bool = False, seam_redenoise_frames: str = "",
                seam_window_tokens: int = 10, seam_lock_tokens: int = 3,
                seam_redenoise_gate: str = "off", dump_latents: bool = False,
                dump_dir: str = "", show_memory_log: bool = True):
        plan = {
            "schema": PASS2_TYPE_STRING,
            "second_pass_sigma0": float(second_pass_sigma0),
            "anchor_strength": float(anchor_strength),
            "second_pass_audio_policy": str(second_pass_audio_policy or AUDIO_POLICIES[0]),
            "seam_tolerance_frames": int(seam_tolerance_frames),
            "calm_search_window": int(calm_search_window),
            "calm_policy": str(calm_policy),
            "profile_camera_compensate": bool(profile_camera_compensate),
            "profile_reduce": str(profile_reduce),
            "calm_abstain_below": float(calm_abstain_below),
            "calm_min_quality": float(calm_min_quality),
            "hunt_persistence": bool(hunt_persistence),
            "hunt_min_persistence": float(hunt_min_persistence),
            "hunt_search_window": int(hunt_search_window),
            "calm_too_quiet_below": float(calm_too_quiet_below),
            "calm_min_gain": float(calm_min_gain),
            "auto_seam_hunt": bool(auto_seam_hunt),
            "anchor_tokens": int(anchor_tokens),
            "upscale_pad_tokens": int(upscale_pad_tokens),
            "seam_blend": bool(seam_blend),
            "seam_redenoise": bool(seam_redenoise),
            "seam_redenoise_frames": str(seam_redenoise_frames or ""),
            "seam_window_tokens": int(seam_window_tokens),
            "seam_lock_tokens": int(seam_lock_tokens),
            "seam_redenoise_gate": str(seam_redenoise_gate),
            "dump_latents": bool(dump_latents),
            "dump_dir": str(dump_dir or ""),
            "show_memory_log": bool(show_memory_log),
        }
        print("[SeamKit] Pass-2 plan: sigma0=%.2f hunt=%s pad=%d blend=%s redenoise=%s(%s)"
              % (plan["second_pass_sigma0"], "on" if plan["auto_seam_hunt"] else "off",
                 plan["upscale_pad_tokens"], plan["seam_blend"],
                 "on" if plan["seam_redenoise"] else "off", plan["seam_redenoise_gate"]),
              flush=True)
        return io.NodeOutput(plan, float(second_pass_sigma0))
