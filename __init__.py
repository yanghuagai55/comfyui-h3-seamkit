"""comfyui-h3-hardcut

Hard-cut planning for MiniMax H3 chunked two-pass upscale.

Why this exists
---------------
The chunked second pass splits time into windows.  With a non-zero overlap it
blends the previous window's tail into the new window, which produces ghosting
and soft patches whenever the two windows disagree — and the official docs are
explicit that a longer overlap cannot promise to remove that drift.

A zero overlap removes the disagreement entirely: every window is sampled on
its own and the results are appended back-to-back, so the join is a cut rather
than a dissolve.  A cut is legitimate film grammar, so the seam stops being a
defect as soon as the prompt asks the model for a shot change on that exact
frame.  These nodes do the arithmetic and write the prompt.

Nodes
-----
* `MiniMaxH3HardCutPlan`      — duration + up to four cut times -> executor plan
* `MiniMaxH3HardCutValidate`  — prompt + plan -> pass-through, or raise if they disagree
* `MiniMaxH3HardCutShotPrompt` — shots + cut times -> R2V detailed_description (optional)

`hardcut_math.py` is dependency-free and runs standalone:

    python hardcut_math.py 8 4.25 --mp 1.5
    python hardcut_math.py 15 "4.25,8.5,12.75" --mp 1.5 --step 0
    python hardcut_math.py --check prompt.txt --total 8 --cuts 4.25 --mp 1.5
"""

from comfy_api.latest import ComfyExtension

from .nodes import (
    MiniMaxH3HardCutAuto,
    MiniMaxH3HardCutPlan,
    MiniMaxH3HardCutShotPrompt,
    MiniMaxH3HardCutValidate,
)
from .h3_upscale import MiniMaxH3HardCutUpscale
from .nodes_repair import (
    MiniMaxH3RepairExtension,
    MiniMaxH3SeamBlend,
    MiniMaxH3SeamDissolve,
    MiniMaxH3SeamFuse,
)

__all__ = ["comfy_entrypoint"]


class H3HardCutExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3HardCutPlan,
            MiniMaxH3HardCutAuto,
            MiniMaxH3HardCutValidate,
            MiniMaxH3HardCutShotPrompt,
            MiniMaxH3HardCutUpscale,
            MiniMaxH3SeamBlend,
            MiniMaxH3SeamFuse,
            MiniMaxH3SeamDissolve,
            MiniMaxH3InfoBuffer,
        ]


def comfy_entrypoint():
    return H3HardCutExtension()
