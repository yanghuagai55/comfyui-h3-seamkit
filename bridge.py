# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""Bridge to the upstream MiniMax-H3 two-pass upscale node pack.

The generated plan must be byte-compatible with what the pack's own LowSigma
plan node emits, because its upscale executor reads the plan field by field.
Rather than duplicating that contract (and going stale whenever the pack ships
a fix), we call the pack's builder inside the running ComfyUI process.  A local
fallback keeps this node usable — with a clear warning — if the pack is absent
or renamed.

The strings below (`T8_H3_CHUNKED_TWO_PASS_PLAN`, `h3_t8`, the schema id) are the
upstream pack's own public identifiers, not ours: they are the wire contract and
must stay byte-for-byte identical.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

_BUILD_NAME = "build_chunked_two_pass_low_sigma_plan"
_MODULE_SUFFIX = "chunked_two_pass_upscale_advanced"

PLAN_SCHEMA_LOW_SIGMA_V3 = "t8.minimax_h3.chunked_two_pass.low_sigma.v3"
PLAN_TYPE_STRING = "T8_H3_CHUNKED_TWO_PASS_PLAN"

PRECISIONS = ("fp16", "bf16", "fp32")
RELEASE_POLICIES = ("keep_loaded", "offload_after", "clear_after")
AUDIO_POLICIES = ("joint_av_preserve_input", "locked_input_audio")


def find_upstream_module(suffix: str, *, require_substring: str | None = None):
    """Locate an upstream module object by its file-name suffix at runtime.

    The pack installs `h3_t8` as a directory on its package `__path__`, so the
    real module names carry the custom-node folder name (ComfyUI loads them via
    `spec_from_file_location`, which prepends the absolute path) and cannot be
    imported with a fixed statement.  Matching on the file-name suffix is stable
    regardless of how the folder is named or where it lives.
    """
    for name, module in list(sys.modules.items()):
        if module is None or not name.endswith(suffix):
            continue
        if require_substring and require_substring not in name:
            continue
        return module
    return None


def find_upstream_builder() -> Callable[..., tuple[dict, str]] | None:
    """Locate the pack's low-sigma plan builder in the running interpreter."""
    module = find_upstream_module(_MODULE_SUFFIX)
    if module is not None:
        builder = getattr(module, _BUILD_NAME, None)
        if callable(builder):
            return builder

    try:  # Not loaded yet (standalone tests, unusual load orders).
        import importlib

        module = importlib.import_module("h3_t8.chunked_two_pass_upscale_advanced")
        builder = getattr(module, _BUILD_NAME, None)
        if callable(builder):
            return builder
    except Exception:
        pass
    return None


def upstream_available() -> bool:
    return find_upstream_builder() is not None


def build_hardcut_plan(
    *,
    model_name: str,
    target_width: int,
    target_height: int,
    chunk_frames: int,
    anchor_strength: float,
    overlap_frames: int = 0,
    precision: str,
    release_policy: str,
    second_pass_audio_policy: str,
    geometry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return (plan, used_upstream_builder).

    `overlap` is pinned to 0 and `temporal_strategy` to `guarded_overlap_exp`:
    the guarded route is what enables windowing at all, and a zero overlap makes
    every window independent, so the executor appends them back to back with no
    cross-fade (see hardcut_math for the derivation).

    `geometry` is stashed under `hardcut.geometry` for our own validator to read.
    The executor reads the plan field by field with `plan.get(...)`, so carrying
    an extra sub-dict is inert for it.
    """
    builder = find_upstream_builder()
    kwargs = dict(
        model_name=model_name,
        target_width=int(target_width),
        target_height=int(target_height),
        temporal_chunk_frames=int(chunk_frames),
        temporal_overlap_frames=max(0, int(overlap_frames)),
        anchor_strength=float(anchor_strength),
        tile_width=int(target_width),
        tile_height=int(target_height),
        spatial_overlap=0,
        spatial_fade=0,
        minimum_tile_size=min(int(target_width), int(target_height)),
        overlap_blend="smoothstep",
        precision=precision,
        release_policy=release_policy,
        spatial_strategy="full_frame_safe",
        temporal_strategy="guarded_overlap_exp",
        second_pass_audio_policy=second_pass_audio_policy,
    )

    _ov = max(0, int(overlap_frames))
    hardcut = {
        "writer": "comfyui-h3-seamkit",
        "overlap_policy": (
            "zero_overlap_independent_windows_hard_cut" if not _ov
            else f"guarded_overlap_anchored_prefix_{_ov}f"
        ),
    }
    if geometry:
        hardcut["geometry"] = dict(geometry)
        if geometry.get("segment_frames"):
            # promoted to the top level of `hardcut` so our executor reads it
            # directly (unequal-window contract, not just a geometry note)
            hardcut["segment_frames"] = list(geometry["segment_frames"])

    if builder is not None:
        plan, _report = builder(**kwargs)
        plan["hardcut"] = {**hardcut, "upstream_builder": True}
        return plan, True

    fallback = _fallback_plan(**kwargs)
    fallback["hardcut"] = {**hardcut, "upstream_builder": False}
    return fallback, False


def _fallback_plan(**kw) -> dict[str, Any]:
    """Minimal contract-compatible plan if the upscale pack is unavailable."""
    width = int(kw["target_width"])
    height = int(kw["target_height"])
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA_LOW_SIGMA_V3,
        "model_name": kw["model_name"],
        "target_width": width,
        "target_height": height,
        "temporal_chunk_frames": int(kw["temporal_chunk_frames"]),
        "temporal_overlap_frames": max(0, int(kw.get("temporal_overlap_frames") or 0)),
        "anchor_strength": float(kw["anchor_strength"]),
        "tile_width": width,
        "tile_height": height,
        "spatial_overlap": 0,
        "spatial_fade": 0,
        "minimum_tile_size": min(width, height),
        "overlap_blend": "smoothstep",
        "precision": kw["precision"],
        "release_policy": kw["release_policy"],
        "spatial_strategy": "full_frame_safe",
        "spatial_quality_boundary": "full_frame_preserves_global_h3_spatial_context",
        "audio_policy": "exact_input_tensor_passthrough",
        "pixel_limit_policy": "no_project_pixel_area_limit",
        "noise_policy": "one_full_target_video_noise_then_exact_coordinate_slices",
        "global_noise_scope": "target_video_latent_only",
        "audio_noise_policy": (
            "one_global_audio_noise_reused_for_joint_model_context_then_discard_output"
            if kw["second_pass_audio_policy"] == "joint_av_preserve_input"
            else "zero_per_piece_with_zero_audio_noise_mask"
        ),
        "sampler_boundary": (
            "coordinates the external initial noise exactly; ancestral or SDE "
            "samplers may still add independent internal per-step noise"
        ),
        "temporal_strategy": "guarded_overlap_exp",
        "temporal_merge_policy": "previous_overlap_guarded_progressive_takeover",
        "temporal_overlap_policy": (
            "replace the next target overlap with the exact previous output; keep its "
            "first half read-only, progressively transfer the second half to the new "
            "trajectory, then append only newly owned tokens"
        ),
        "compatibility": (
            "append_only_v3; old v1/v2 plans, executor inputs, and workflows remain "
            "unchanged"
        ),
        "first_pass_contract": "complete_trajectory_to_zero_before_upscale",
        "recommended_refine": {
            "scheduler": "simple",
            "steps": 3,
            "denoise": 0.30,
            "upstream_readme_max_denoise": 0.40,
        },
        "second_pass_audio_policy": kw["second_pass_audio_policy"],
        "final_audio_policy": "return_exact_first_pass_audio_tensor",
    }
    return plan


def upscaler_options() -> list[str]:
    """Model files offered by the learned 3D latent upscaler loader."""
    try:
        import folder_paths  # provided by ComfyUI

        names = list(folder_paths.get_filename_list("latent_upscale_models"))
        if names:
            return sorted(names)
    except Exception:
        pass
    return ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]
