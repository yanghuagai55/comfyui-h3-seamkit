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

CATEGORY = "MiniMax H3/HardCut"
PLAN_TYPE = io.Custom(PLAN_TYPE_STRING)
NO_CUT = -1.0

_SHOT_SPLIT = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)

_CUT_TOOLTIP = (
    "Cut time in seconds. -1 means 'no cut here' — a slot only produces a cut "
    "when it is a positive number, so a clip with one cut leaves the other three "
    "slots at -1. Reachable cut frames are multiples of 17 frames (0.708s), so "
    "the closest achievable time is reported instead of failing."
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
                "'no cut there'. Reachable cuts are multiples of 17 frames, so the "
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
                io.Float.Input(
                    "cut_1",
                    default=NO_CUT,
                    min=-1.0,
                    max=MAX_SECONDS,
                    step=0.01,
                    tooltip=_CUT_TOOLTIP,
                ),
                io.Float.Input(
                    "cut_2",
                    default=NO_CUT,
                    min=-1.0,
                    max=MAX_SECONDS,
                    step=0.01,
                    tooltip=_CUT_TOOLTIP,
                ),
                io.Float.Input(
                    "cut_3",
                    default=NO_CUT,
                    min=-1.0,
                    max=MAX_SECONDS,
                    step=0.01,
                    tooltip=_CUT_TOOLTIP,
                ),
                io.Float.Input(
                    "cut_4",
                    default=NO_CUT,
                    min=-1.0,
                    max=MAX_SECONDS,
                    step=0.01,
                    tooltip=_CUT_TOOLTIP,
                ),
                io.String.Input(
                    "cut_frames",
                    default="",
                    tooltip=(
                        "Explicit cut FRAMES, comma separated (e.g. '68' or '68,136'). "
                        "When non-empty it overrides cut_1..cut_4 and produces UNEQUAL "
                        "window lengths - the first shot can be 68 frames, the next 124. "
                        "Each frame snaps to the 17-frame grid (70 -> 68). Leave empty "
                        "for equal-length windows via cut_1..cut_4."
                    ),
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
            ],
            outputs=[
                PLAN_TYPE.Output("plan"),
                io.String.Output("cut_seconds"),
                io.String.Output("cut_report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        total_seconds: float,
        cut_1: float,
        cut_2: float,
        cut_3: float,
        cut_4: float,
        cut_frames: str,
        chunk_step: int,
        canvas_megapixels: float,
        model_name: str,
        target_width: int,
        target_height: int,
        precision: str,
        release_policy: str,
        anchor_strength: float,
        second_pass_audio_policy: str,
    ):
        slots = [cut_1, cut_2, cut_3, cut_4][:CUT_SLOTS]
        cuts = cuts_from_inputs(slots)
        seg_frames = (
            [float(t) for t in (cut_frames or "").replace(";", ",").split(",") if t.strip()]
            or None
        )
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
            precision=precision,
            release_policy=release_policy,
            second_pass_audio_policy=second_pass_audio_policy,
            geometry={
                "total_seconds": info["total_seconds"],
                "total_frames": info["total_frames"],
                "chunk_frames": plan_chunk,
                "overlap_frames": 0,
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
                    "cut_lead_frames",
                    default=10,
                    min=-68,
                    max=68,
                    step=1,
                    tooltip=(
                        "★ 切点提前量（帧）= **切点帧位 − 实测画面突变帧位**。\n"
                        "正值 = 模型比切点**提前**起转（实测两片都是 10：8s 片 102−92、"
                        "10s 片 119−109）；负值 = 模型偏晚。\n"
                        "用途：报告给出每个切点「模型实际转镜的帧」= 切点 − 本值，"
                        "把上一镜的动作收在那一帧，切点处就不会显得切早/切晚。\n"
                        "填 0 = 不补偿。**可填负数**（帧数单位，随内容/时长变，自己量了改）。"
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
        cut_lead_frames: int,
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
            precision=precision,
            release_policy=release_policy,
            second_pass_audio_policy=second_pass_audio_policy,
            geometry={
                "total_seconds": info["total_seconds"],
                "total_frames": info["total_frames"],
                "chunk_frames": plan_chunk,
                "overlap_frames": 0,
                "cut_frames": list(info["cut_frames"]),
                "cut_seconds": list(info["actual_cuts"]),
                "segment_frames": list(info["segment_frames"]),
            },
        )

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

        # The model does not turn exactly on the cut: it starts turning a few
        # frames early (measured lead = cut − observed change = 10 frames on both
        # an 8 s and a 10 s clip).  So the previous shot's action has to resolve at
        # cut − lead, not at the cut itself, or the edit reads as cutting late.
        report += "\n\n=== ACTION BEATS (write the action to these frames, not to the cut) ==="
        cuts = list(info["cut_frames"])
        lead = int(cut_lead_frames)
        if cuts and lead:
            report += (
                f"\n  cut lead = {lead:+d} frame(s)   "
                "(cut frame − observed change frame; positive = the model turns EARLY,"
                " negative = it turns late)"
                "\n  the model actually turns at these frames — resolve the previous shot's"
                " action by then:"
            )
            for c in cuts:
                turn = max(0, min(int(info["total_frames"]), int(c) - lead))
                report += (
                    f"\n    cut f{int(c)} ({int(c) / FPS:.3f}s)"
                    f"  ->  model turns at f{turn} ({turn / FPS:.3f}s)"
                )
        elif cuts:
            report += "\n  (cut_lead_frames = 0 — no compensation, actions may read as cutting late)"
        else:
            report += "\n  (single window, no cut)"
        report += "\n  lead is editable on this node (`cut_lead_frames`, frames, may be negative)"

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
