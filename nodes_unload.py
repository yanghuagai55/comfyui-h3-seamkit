# SPDX-License-Identifier: GPL-3.0-or-later
"""Unload only the text encoder(s), keeping the diffusion model resident.

Why this exists: on a 32 GB machine the H3 stack is DiT (19.5 GB int8) plus
the Qwen3-VL text encoder (14.6 GB).  ComfyUI keeps both pinned/parked, which
is what overflows aimdo's host buffer ("requested ... beyond reserved host
buffer").  The SECOND pass never touches the text encoder, so once the first
pass has produced its conditioning the TE is dead weight.

KJNodes' VRAM_Debug can only call model_management.unload_all_models(), which
would also evict the 19.5 GB DiT and force a full reload before the second
pass.  This node unloads CLIP/TE models only - the DiT stays put.
"""

from __future__ import annotations

import logging

import comfy.model_management as mm
from comfy_api.latest import io

CATEGORY = "model/conditioning/minimax"

# Class-name fragments that identify a text encoder rather than the DiT.
_TE_HINTS = ("clip", "temodel", "qwen", "t5", "llama", "bert")


def _looks_like_text_encoder(patched) -> bool:
    inner = getattr(patched, "model", None)
    if inner is None:
        return False
    # strongest signal first: the H3 text encoder is an SD1ClipModel subclass
    try:
        import comfy.sd1_clip as sd1

        if isinstance(inner, sd1.SD1ClipModel):
            return True
    except Exception:
        pass
    name = type(inner).__name__.lower()
    return any(h in name for h in _TE_HINTS)


class MiniMaxH3UnloadTextEncoder(io.ComfyNode):
    """Free the text encoder(s) while keeping every other model loaded.

    Wire it inline right after the first pass has been queued (its conditioning
    already consumed): pass anything through and the encoder is released.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UnloadTextEncoder",
            display_name="Unload Text Encoder (keep the rest)",
            category=CATEGORY,
            description=(
                "Unloads only CLIP/text-encoder models (frees their VRAM and the "
                "aimdo host buffer they occupy) while leaving the diffusion model "
                "loaded, so the second pass does not have to reload it. "
                "Pass-through: wire it inline wherever the encoder is no longer needed."
            ),
            inputs=[
                io.AnyType.Input("anything", optional=True),
                io.Boolean.Input(
                    "empty_cache_after",
                    default=True,
                    tooltip="卸载后顺手调 soft_empty_cache() 释放缓存块（不卸模型）。",
                ),
            ],
            outputs=[io.AnyType.Output("anything")],
        )

    @classmethod
    def execute(cls, anything=None, empty_cache_after: bool = True):
        freed = []
        try:
            loaded = list(mm.current_loaded_models)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[UnloadTE] cannot list loaded models: {exc}", flush=True)
            loaded = []
        for entry in list(loaded):
            if not _looks_like_text_encoder(entry):
                continue
            inner = getattr(entry, "model", None)
            name = type(inner).__name__ if inner is not None else "?"
            try:
                entry.model_unload()
                freed.append(name)
            except Exception as exc:
                print(f"[UnloadTE] {name} unload failed: {exc}", flush=True)
        if empty_cache_after:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
        if freed:
            print(f"[UnloadTE] released text encoder(s): {', '.join(freed)}", flush=True)
        else:
            print("[UnloadTE] no text encoder was loaded - nothing to free", flush=True)
        logging.debug("[UnloadTE] freed=%s", freed)
        return io.NodeOutput(anything)


NODE_CLASS_MAPPINGS = {"MiniMaxH3UnloadTextEncoder": MiniMaxH3UnloadTextEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3UnloadTextEncoder": "Unload Text Encoder (keep the rest)"
}
