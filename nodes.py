# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""精简后的 MiniMax H3 双采上下游规划节点（已清理古早手动切点节点与 19 个冗余调参旋钮）。

两个核心节点：
  1. `MiniMaxH3HardCutFirstPassPlan` —— 上游规划（片长 / 分辨率 / 分段 / 一采缓存）
  2. `MiniMaxH3HardCutPass2Plan`     —— 下游二采规划（6 个核心控件 + 3 个高级控件，已适配 Route 3）

为什么拆成上下游两个节点：
  * 上游节点输出会进一采采样器（#93）的缓存指纹，只放真正影响一采的参数；
  * 下游节点在纯二采侧，随便改拼缝模式 / CFG / Route 3 开关都不会击穿一采缓存。
"""

from __future__ import annotations

from comfy_api.latest import io

from .bridge import (
    AUDIO_POLICIES,
    PASS2_TYPE_STRING,
    PLAN_TYPE_STRING,
    PRECISIONS,
    build_hardcut_plan,
    upscaler_options,
)
from .hardcut_math import (
    FPS,
    FRAME_GRID,
    MAX_SECONDS,
    auto_plan,
    auto_prompt,
    format_auto_report,
    resolution_for,
    validate_prompt,
)

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

CATEGORY = "MiniMax H3 Hard Cut"
PLAN_TYPE = io.Custom(PLAN_TYPE_STRING)
PASS2_TYPE = io.Custom(PASS2_TYPE_STRING)


def aspect_ratios() -> dict[str, tuple[float, float]]:
    try:
        from comfy_extras.nodes_resolution import ASPECT_RATIOS

        if ASPECT_RATIOS:
            return dict(ASPECT_RATIOS)
    except Exception:
        pass
    return dict(_FALLBACK_ASPECTS)


def aspect_options() -> list[str]:
    return list(aspect_ratios())


def default_aspect() -> str:
    return _DEFAULT_ASPECT if _DEFAULT_ASPECT in aspect_ratios() else aspect_options()[0]


class MiniMaxH3HardCutFirstPassPlan(io.ComfyNode):
    """上游规划节点：片长、一采/二采画布尺寸、单窗分段上限与一采缓存开关。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutFirstPassPlan",
            display_name="MiniMax H3 Plan (upstream / first-pass)",
            description=(
                "上游总控节点：一次定好片长（17n+5 帧网格）、一采/二采画布分辨率、单窗最大秒数，"
                "并统一管理一采缓存。\n"
                "提示：普通实景/动漫多镜片请保持 `green_screen_mode = False`（默认）；"
                "只有纯绿幕一镜到底才打开。"
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    force_input=True,
                    tooltip="接提示词节点；自由提示词默认直接放行，留空时输出标准分镜模板。",
                ),
                io.Float.Input(
                    "total_seconds",
                    default=5.0,
                    min=1.0,
                    max=MAX_SECONDS,
                    step=0.5,
                    tooltip="视频总时长（秒），自动吸附到 MiniMax H3 的 17n+5 帧网格。",
                ),
                io.Float.Input(
                    "target_segment_seconds",
                    default=6.4,
                    min=0.71,
                    max=60.0,
                    step=0.1,
                    tooltip=(
                        "单窗最大秒数（硬上限）。8GB 卡 1.3MP 推荐 5.0~6.4s；"
                        "A100-40GB 开 Route 3 跑 15s 可放宽到 9.0s（4 窗压成 2 窗）。"
                    ),
                ),
                io.Float.Input(
                    "first_megapixels",
                    default=0.4,
                    min=0.1,
                    max=4.0,
                    step=0.1,
                    tooltip="一采画布像素数（默认 0.4 MP ≈ 864×480）。",
                ),
                io.Float.Input(
                    "second_megapixels",
                    default=1.3,
                    min=0.1,
                    max=4.0,
                    step=0.05,
                    tooltip="二采最终画布像素数（1.3 MP ≈ 1568×864 为 8GB/A100 甜点；0.94 MP ≈ 1280×736 更稳更快）。",
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=aspect_options(),
                    default=default_aspect(),
                    tooltip="一采与二采共用的画幅比例。",
                ),
                io.Boolean.Input(
                    "green_screen_mode",
                    default=False,
                    tooltip=(
                        "仅纯绿幕一镜到底人物片才开！\n"
                        "• False（默认）：普通实景/动漫/多镜片正常按切镜硬切或自动判断；\n"
                        "• True：禁用硬切，全程 17 帧 overlap 平缓过渡并跳过分镜词检测。"
                    ),
                ),
                io.Boolean.Input(
                    "use_cache",
                    default=False,
                    advanced=True,
                    tooltip="一采缓存总开关（关 = 纯采样器；做二采 A/B 调参时再开）。",
                ),
                io.String.Input(
                    "cache_key",
                    default="r3_run",
                    advanced=True,
                    tooltip="一采缓存文件名 key（与上游指纹共同决定命中）。",
                ),
                io.String.Input(
                    "cache_path",
                    default="",
                    advanced=True,
                    tooltip="一采缓存目录（留空 = 输出目录下的 seamkit_latent_cache/）。",
                ),
                io.Int.Input(
                    "boundary_search_frames",
                    default=17,
                    min=0,
                    max=170,
                    step=17,
                    advanced=True,
                    tooltip="边界搜索带宽（17 帧的倍数，默认 17）。",
                ),
            ],
            outputs=[
                PLAN_TYPE.Output("plan"),
                io.String.Output("prompt"),
                io.String.Output("report"),
                io.Int.Output("first_width"),
                io.Int.Output("first_height"),
                io.Int.Output("second_width"),
                io.Int.Output("second_height"),
                io.Int.Output("length"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        total_seconds: float = 5.0,
        target_segment_seconds: float = 6.4,
        first_megapixels: float = 0.4,
        second_megapixels: float = 1.3,
        aspect_ratio: str = "16:9 (Widescreen)",
        green_screen_mode: bool = False,
        use_cache: bool = False,
        cache_key: str = "r3_run",
        cache_path: str = "",
        boundary_search_frames: int = 17,
        # 以下为兼容旧脚本/测试的内部固定参数（不再暴露到 UI 干扰操作）
        multiple: int = 32,
        loose_prompt: bool = True,
        require_sage_patch: bool = False,
        overlap_frames: int = 0,
        auto_calm_search: bool = False,
        calm_overlap_frames: int = 17,
        calm_policy: str = "auto",
        model_name: str = "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        precision: str = "bf16",
        release_policy: str = "clear_after",
        anchor_strength: float = 0.999,
        second_pass_audio_policy: str = "joint_av_preserve_input",
        seam_tolerance_frames: int = 4,
        **extra_legacy_kwargs,
    ):
        w_ratio, h_ratio = aspect_ratios().get(
            aspect_ratio, aspect_ratios()[default_aspect()]
        )
        first_w, first_h = resolution_for(first_megapixels, w_ratio, h_ratio, multiple)
        second_w, second_h = resolution_for(second_megapixels, w_ratio, h_ratio, multiple)
        canvas_mp = second_w * second_h / 1_000_000.0

        if green_screen_mode:
            _effective_overlap = max(17, int(calm_overlap_frames) if auto_calm_search else int(overlap_frames or 17))
        else:
            _effective_overlap = (
                max(0, int(calm_overlap_frames)) if auto_calm_search
                else max(0, int(overlap_frames))
            )

        _chunk_step = max(1, int(round(float(target_segment_seconds) * FPS / FRAME_GRID)))
        info = auto_plan(
            total_seconds,
            _chunk_step,
            canvas_mp,
            overlap=_effective_overlap,
            target_seconds=float(target_segment_seconds),
            reserve_frames=max(0, int(boundary_search_frames)),
        )

        plan_chunk = info["longest"]
        if plan_chunk % FRAME_GRID:
            plan_chunk = (plan_chunk // FRAME_GRID + 1) * FRAME_GRID

        plan, used_upstream = build_hardcut_plan(
            model_name=model_name,
            target_width=second_w,
            target_height=second_h,
            chunk_frames=plan_chunk,
            anchor_strength=anchor_strength,
            overlap_frames=_effective_overlap if green_screen_mode else max(0, int(overlap_frames)),
            precision=precision,
            release_policy=release_policy,
            second_pass_audio_policy=second_pass_audio_policy,
            geometry={
                "total_seconds": info["total_seconds"],
                "total_frames": info["total_frames"],
                "chunk_frames": plan_chunk,
                "overlap_frames": _effective_overlap if green_screen_mode else max(0, int(overlap_frames)),
                "cut_frames": list(info["cut_frames"]),
                "cut_seconds": list(info["actual_cuts"]),
                "segment_frames": list(info["segment_frames"]),
                "seam_tolerance": max(0, int(seam_tolerance_frames)),
            },
        )

        _hc = plan.setdefault("hardcut", {})
        _hc["seam_tolerance"] = max(0, int(seam_tolerance_frames))
        _hc["green_screen_mode"] = bool(green_screen_mode)
        plan["green_screen_mode"] = bool(green_screen_mode)
        if green_screen_mode:
            _hc["auto_calm_search"] = True
            _hc["calm_policy"] = "calm_overlap"
            _hc["calm_overlap_frames"] = int(_effective_overlap)
        else:
            _hc["auto_calm_search"] = bool(auto_calm_search)
            _hc["calm_policy"] = str(calm_policy)
            _hc["calm_overlap_frames"] = max(0, int(calm_overlap_frames))

        # 标定好的内部默认值（下游 Pass2Plan 会按需覆盖）
        _hc.update({
            "calm_search_window": 51,
            "profile_camera_compensate": True,
            "profile_reduce": "mean",
            "calm_abstain_below": 0.0,
            "calm_min_quality": 0.8,
            "hunt_persistence": True,
            "hunt_min_persistence": 0.8,
            "hunt_search_window": 34,
            "calm_too_quiet_below": 0.05,
            "calm_min_gain": 0.15,
            "loose_prompt": bool(loose_prompt),
            "use_cache": bool(use_cache),
            "cache_key": str(cache_key or "r3_run"),
            "require_sage_patch": bool(require_sage_patch),
            "cache_path": str(cache_path or ""),
            "canvas_mp": float(canvas_mp),
        })

        note = "" if used_upstream else "WARNING: upstream MiniMax H3 upscale pack not found; fallback plan emitted."
        report = format_auto_report(info, note)
        report += (
            f"\ncanvas        : first {first_w}x{first_h} ({first_megapixels:g} MP)  ->  "
            f"second {second_w}x{second_h} ({canvas_mp:.3f} MP)\n"
            f"clip length   : {info['total_frames']} frames (17n+5)"
        )

        _ov_f = max(0, int(_effective_overlap))
        _seglens = list(info.get("lengths") or [])
        _winmax = (max(_seglens) + _ov_f) if _seglens else int(info["total_frames"])
        _lat_t = max(1, (_winmax - 5) // 17 * 5 + 2) if _winmax > 5 else 1
        _tok_win = _lat_t * max(1, second_h // 16) * max(1, second_w // 16)
        _tok_r3 = int(round(_tok_win * 0.55))
        _proxy = _winmax * canvas_mp
        report += (
            "\n\n=== 显存 / 负载估算 ===\n"
            f"  稠密单窗 token : {_tok_win:,}  (latent_T {_lat_t} × {second_h // 16} × {second_w // 16})\n"
            f"  Route 3 预估   : ≈{_tok_r3:,} token (s_bg=3 约 50%~60%，依前景占比)\n"
            f"  稠密负载代理   : 最长窗 × MP = {_proxy:.1f}  (8GB 稠密安全线 ~184)"
        )

        incoming = (prompt or "").strip()
        if incoming:
            if green_screen_mode:
                out_prompt = incoming
                report += (
                    "\n\n=== 绿幕模式 (Green Screen Mode) ===\n"
                    f"  - 状态：开启（禁用硬切，共 {info['segments']} 段，段间 {int(_effective_overlap)} 帧 overlap 过渡）\n"
                    "  - 提示词：已跳过分镜检测直接放行"
                )
            else:
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
                        f"Hard-cut auto check failed ({len(errors)} error(s)).\n\n" + report
                    )
        else:
            out_prompt = auto_prompt(info, "", "")

        return io.NodeOutput(
            plan,
            out_prompt,
            report,
            first_w,
            first_h,
            second_w,
            second_h,
            info["total_frames"],
            ui={"text": (report,)},
        )


class MiniMaxH3HardCutPass2Plan(io.ComfyNode):
    """精简版下游二采规划节点（从 28 个控件瘦身为 6 个核心 + 3 个高级，已适配 Route 3）。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HardCutPass2Plan",
            display_name="MiniMax H3 Pass-2 Plan (downstream)",
            description=(
                "下游二采参数总控（修改这里**不会**触发一采重采样）。\n"
                "已将原本 19 个古早实验调参常数收归内部标定默认值，只保留真正有用的核心开关：\n"
                "• seam_mode = auto（默认：自动找切镜硬切，连续段平滑锚定）\n"
                "• seam_mode = hard_cut（多镜头/强运动纯硬切，零跨镜叠化）\n"
                "• seam_mode = smooth_overlap（一镜到底/绿幕平缓过渡）"
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=upscaler_options(),
                    tooltip="3D latent 上采样模型。",
                ),
                io.Combo.Input(
                    "seam_mode",
                    options=["auto", "hard_cut", "smooth_overlap"],
                    default="auto",
                    tooltip=(
                        "多窗拼缝策略（一键替代原先 14 个 hunt/calm/overlap 散乱参数）：\n"
                        "• auto：自动检测真切镜硬切，连续镜按画面平缓度自动选锚定或硬切；\n"
                        "• hard_cut：强制纯硬切（多镜头/大动态推荐，绝不跨镜叠化）；\n"
                        "• smooth_overlap：强制 17 帧锚定重叠（单镜头/绿幕慢动推荐）。"
                    ),
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=17,
                    min=0,
                    max=68,
                    step=17,
                    tooltip="auto / smooth_overlap 模式下的锚定重叠帧数（17 的倍数；hard_cut 模式下自动归 0）。",
                ),
                io.Boolean.Input(
                    "seam_blend",
                    default=False,
                    tooltip="重叠区线性交叉淡化（默认关：仅静机位/极慢镜头开；运动或多镜片开它会出现重影）。",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="二采 CFG（加速 Turbo LoRA 保持 1.0）。",
                ),
                io.Boolean.Input(
                    "route3_enabled",
                    default=True,
                    tooltip="配合 Route 3 节点使用：启用多窗 start_token 精确对齐与压缩负载口径汇报。",
                ),
                io.Int.Input(
                    "upscale_pad_tokens",
                    default=3,
                    min=0,
                    max=8,
                    advanced=True,
                    tooltip="3D 上采样分块时间外扩 padding（默认 3，消除窗边熔化）。",
                ),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="bf16",
                    advanced=True,
                    tooltip="3D 上采样模型精度（默认 bf16）。",
                ),
                io.Boolean.Input(
                    "dump_latents",
                    default=False,
                    advanced=True,
                    tooltip="开发诊断：把每窗中间 latent 落盘到 latent_dump/。",
                ),
            ],
            outputs=[
                PASS2_TYPE.Output("pass2_plan"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str = "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        seam_mode: str = "auto",
        overlap_frames: int = 17,
        seam_blend: bool = False,
        cfg: float = 1.0,
        route3_enabled: bool = True,
        upscale_pad_tokens: int = 3,
        precision: str = "bf16",
        dump_latents: bool = False,
        # 兼容旧 API JSON 传入的历史参数
        **legacy_kwargs,
    ):
        mode = str(seam_mode or "auto")
        if mode == "hard_cut":
            auto_seam_hunt = True
            auto_calm_search = False
            calm_policy = "jerk_hardcut"
            eff_overlap = 0
            eff_blend = False
        elif mode == "smooth_overlap":
            auto_seam_hunt = False
            auto_calm_search = True
            calm_policy = "calm_overlap"
            eff_overlap = max(17, int(overlap_frames))
            eff_blend = bool(seam_blend)
        else:  # auto
            auto_seam_hunt = True
            auto_calm_search = True
            calm_policy = "auto"
            eff_overlap = max(0, int(overlap_frames))
            eff_blend = bool(seam_blend)

        # 若旧脚本显式传了细粒度开关，仍予以尊重
        if "auto_seam_hunt" in legacy_kwargs:
            auto_seam_hunt = bool(legacy_kwargs["auto_seam_hunt"])
        if "auto_calm_search" in legacy_kwargs:
            auto_calm_search = bool(legacy_kwargs["auto_calm_search"])
        if "calm_policy" in legacy_kwargs:
            calm_policy = str(legacy_kwargs["calm_policy"])

        plan = {
            "schema": PASS2_TYPE_STRING,
            "model_name": str(model_name),
            "precision": str(precision),
            "release_policy": str(legacy_kwargs.get("release_policy", "clear_after")),
            "cfg": float(cfg),
            "anchor_strength": float(legacy_kwargs.get("anchor_strength", 0.999)),
            "second_pass_audio_policy": str(
                legacy_kwargs.get("second_pass_audio_policy", AUDIO_POLICIES[0])
            ),
            "seam_mode": mode,
            "route3_enabled": bool(route3_enabled),
            "seam_tolerance": int(legacy_kwargs.get("seam_tolerance_frames", 4)),
            "calm_search_window": int(legacy_kwargs.get("calm_search_window", 51)),
            "calm_policy": calm_policy,
            "overlap_frames": eff_overlap,
            "auto_calm_search": auto_calm_search,
            "calm_overlap_frames": int(legacy_kwargs.get("calm_overlap_frames", eff_overlap)),
            "profile_camera_compensate": bool(legacy_kwargs.get("profile_camera_compensate", True)),
            "profile_reduce": str(legacy_kwargs.get("profile_reduce", "mean")),
            "calm_abstain_below": float(legacy_kwargs.get("calm_abstain_below", 0.0)),
            "calm_min_quality": float(legacy_kwargs.get("calm_min_quality", 0.8)),
            "hunt_persistence": bool(legacy_kwargs.get("hunt_persistence", True)),
            "hunt_min_persistence": float(legacy_kwargs.get("hunt_min_persistence", 0.8)),
            "hunt_search_window": int(legacy_kwargs.get("hunt_search_window", 34)),
            "calm_too_quiet_below": float(legacy_kwargs.get("calm_too_quiet_below", 0.05)),
            "calm_min_gain": float(legacy_kwargs.get("calm_min_gain", 0.15)),
            "auto_seam_hunt": auto_seam_hunt,
            "anchor_tokens": int(legacy_kwargs.get("anchor_tokens", 1)),
            "upscale_pad_tokens": int(upscale_pad_tokens),
            "seam_blend": eff_blend,
            "dump_latents": bool(dump_latents),
            "dump_dir": str(legacy_kwargs.get("dump_dir", "latent_dump")),
            "show_memory_log": bool(legacy_kwargs.get("show_memory_log", True)),
        }
        print(
            "[SeamKit] Pass-2 plan: mode=%s overlap=%df blend=%s pad=%d route3=%s"
            % (mode, eff_overlap, eff_blend, plan["upscale_pad_tokens"], plan["route3_enabled"]),
            flush=True,
        )
        return io.NodeOutput(plan)
