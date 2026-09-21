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
                    step=0.5,
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
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=list(AUDIO_POLICIES),
                    default="joint_av_preserve_input",
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
        info = plan_hard_cut(
            total_seconds, cuts, canvas_megapixels, chunk_step, seg_frames
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
                io.Int.Input(
                    "chunk_step",
                    default=6,
                    min=1,
                    max=100,
                    step=1,
                    tooltip=(
                        "Chunk 档位 = 每段最多多少「17 帧块」（绝对上限）。每段最大帧数 = "
                        "chunk_step × 17，最大秒数 = chunk_step × 17 ÷ 24 = chunk_step × 0.708s。"
                        "例：6 → 102 帧 ≈ 4.25s；7 → 119 帧 ≈ 4.96s；8 → 136 帧 ≈ 5.67s；"
                        "11 → 187 帧 ≈ 7.79s。每一段（含末段）都不超过这个上限。"
                        "注意：与 Plan 节点的 chunk_step 同名不同义——那边是相对步进"
                        "（±17 帧），这里是绝对块数上限。"
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
                io.Combo.Input("precision", options=list(PRECISIONS), default="bf16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_after",
                ),
                io.Float.Input(
                    "anchor_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="No effect at overlap 0; kept for interface compatibility.",
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=list(AUDIO_POLICIES),
                    default="joint_av_preserve_input",
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
                ),
                io.Int.Input(
                    "seam_tolerance_frames",
                    default=17,
                    min=0,
                    max=68,
                    step=1,
                    tooltip=(
                        "★ 自动找缝的容差（帧）：交给 `#40` 的 `auto_seam_hunt` 用。\n"
                        "二采会在 latent 上找模型真正的转镜点，**只有它离本节点算出的切点 ≤ 本值时才采纳**"
                        "（超过＝疑似误检，忽略并在报告里说明）。\n"
                        "默认 17 = 一格网格：17 帧内算同一刀，超出就认为是别的东西在动。\n"
                        "调大 = 更信任检测（但误检风险↑）；调小 = 只认同一个网格点附近的变化。\n"
                        "**一值两用**（自适应搜索开启时）：它同时是**触发 overlap 的偏差阈值**——\n"
                        "模型实际转镜（`measured`）与计划切点的差超过本值 → 那条缝转用锚定重叠。\n"
                        "实测差 3~4 帧时缝就能看出来，想更敏感就调到 **3~4**。"
                    ),
                ),
                io.Int.Input(
                    "prompt_shift_frames",
                    default=1,
                    min=-17,
                    max=17,
                    step=1,
                    tooltip=(
                        "★ 改写 `prompt` 输出的时间戳（帧）：本节点算完切点后，把提示词里"
                        "**每个 `[Shot N] At MM:SS.mmm`** 整体平移这么多帧再输出（正 = 写晚一点）。\n"
                        "为什么需要：模型不会正好在被告知的时刻转镜（实测偏移 −10 ~ +1 帧，"
                        "随内容/种子变），而执行器的边界只能落在 token 网格上 —— "
                        "**改提示词的时间是唯一比网格更细的旋钮**。\n"
                        "它在这里做，是为了**不让写提示词的大模型知道这些机制**"
                        "（少喂非剧情内容 = 剧情权重不被稀释）。\n"
                        "只动镜头声明的秒数，镜内动作节拍不动；summary / soundscape 里引用的同一时间"
                        "会一起改，保持自洽。填 0 = 不改写。"
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
        chunk_step: int,
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
        prompt_shift_frames: int,
        overlap_frames: int = 0,
        auto_calm_search: bool = False,
        calm_search_window: int = 34,
        calm_overlap_frames: int = 17,
    ):
        w_ratio, h_ratio = aspect_ratios().get(
            aspect_ratio, aspect_ratios()[default_aspect()]
        )
        first_w, first_h = resolution_for(first_megapixels, w_ratio, h_ratio, multiple)
        second_w, second_h = resolution_for(second_megapixels, w_ratio, h_ratio, multiple)
        canvas_mp = second_w * second_h / 1_000_000.0

        info = auto_plan(total_seconds, chunk_step, canvas_mp)

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
                overlap_frames=0,
                segment_frames=list(info["cut_frames"]),
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
            out_prompt, moved = shift_shot_times(incoming, prompt_shift_frames)
            if moved:
                report += (
                    "\n\nprompt time shift ("
                    f"{int(prompt_shift_frames):+d} frame(s)) — only the `prompt` OUTPUT moves:\n"
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
                    overlap_frames=0,
                    segment_frames=list(info["cut_frames"]),
                )
                late = recheck.get("errors") or []
                if late:
                    report += (
                        "\n  WARNING: after the shift the declared times no longer sit within 1 "
                        "frame of the plan's cuts — lower `prompt_shift_frames`, or write the "
                        "prompt to land exactly on the plan:\n    - "
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
            result = validate_prompt(
                prompt,
                total_seconds=geo.get("total_seconds"),
                requested_cuts=geo.get("cut_seconds") or "",
                canvas_mp=geo.get("canvas_mp"),
                chunk_frames=geo.get("chunk"),
                overlap_frames=geo.get("overlap") or 0,
                segment_frames=geo.get("segment_frames"),
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
