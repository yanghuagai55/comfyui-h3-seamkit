# SPDX-License-Identifier: GPL-3.0-or-later
"""教师提示词池（160 条，T2VA 78% + Ref2VA 22%）。

配比理由：Ref2VA 只有本地三张参考图（一个角色 + 两间同款客厅），占比过高会把
「这一个角色/屋子」焊进 LoRA；T2VA 负责通用画质与运动多样性，Ref2VA 只保留
身份/场景保持能力。默认 124 T2VA + 36 Ref2VA ≈ 78% / 22%。

来源优先级
----------
1. 公开数据集（远端可直接下）：
   * `BestiVictoryLab/VGA-Bench` 的 `prompts/200-aesthetic-prompts.csv`
   * `BestiVictoryLab/VGA-Bench` 的 `prompts/476-GQprompts.csv`
   * `meta-ai-for-media-research/movie_gen_video_bench_no_generations`
2. 内置 40 条场景描述（覆盖人像/植被/建筑/文字/夜景/运镜/微距/水/火/织物/动物/食物）。
3. 参考图：`--ref-images` 给什么用什么；默认用 ComfyUI input 里现成的三张。

输出 JSONL（`train/gen_teachers.py` 直接吃）：
  {"id","task","prompt","ref_image","frames","seed"}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

RAW_T2VA = [
    "a rain-soaked red maple leaf trembling on a mossy stone, droplets sliding off the edge",
    "sunlight filtering through dense bamboo leaves, a gentle breeze making the shadows ripple",
    "a macro shot of a jumping spider cleaning its front legs on a dried sunflower head",
    "a hummingbird hovering beside a scarlet hibiscus flower, wings blurred, proboscis probing",
    "a slow dolly along a narrow Kyoto alley after rain, wet stone lanterns and steam",
    "a modern glass skyscraper reflecting sunset clouds, slow upward tilt, lens flare",
    "a neon-lit Hong Kong street at night, rain, passing taxis, steam from a food stall",
    "a close-up of a vintage mechanical watch movement, gears turning, warm rim light",
    "a handwritten letter on textured paper, ink bleeding slightly, candlelight flicker",
    "a poster-covered concrete wall, peeling layers, graffiti, shallow depth of field",
    "a cup of black coffee on a wooden table, steam curling, morning window light",
    "a chef slicing a ripe tomato, juice and seeds visible, macro, kitchen light",
    "a spinning pottery wheel with wet clay, hands shaping a bowl, water droplets",
    "a glass sphere refracting a city skyline, slow rotation, caustics on a white table",
    "a field of lavender swaying, bees crossing frame, warm late-afternoon light",
    "a snow-covered pine branch, a chickadee landing, powder falling, soft overcast light",
    "a fast whip-pan across a desert highway, heat shimmer, distant mesas",
    "a low-angle tracking shot of a skateboarder's wheels on wet asphalt, sparks",
    "an underwater shot of a jellyfish pulsing, particles drifting, deep blue gradient",
    "a close-up of a cat's eye contracting in sunlight, fur detail, slow blink",
    "a candle flame bending in a draft, wax pooling, dark background, macro",
    "a silk scarf unfurling in slow motion, embroidered patterns catching light",
    "a rusty ship hull in a dry dock, rivets and peeling paint, wide-angle",
    "a rooftop clothesline in a dense city, laundry flapping, golden hour",
    "a bowl of ramen, steam, chopsticks lifting noodles, bokeh background",
    "a chessboard mid-game, a hand moving a knight, dramatic side light",
    "a typewriter carriage returning, keys striking, dust motes in a sunbeam",
    "a mountain stream rushing over mossy rocks, long exposure feel, spray",
    "a butterfly emerging from a chrysalis, wings unfolding, backlit, macro",
    "a violin bow crossing strings, rosin dust, close-up, warm stage light",
    "a thunderstorm over a wheat field, lightning illuminating clouds, time-lapse feel",
    "a fish market at dawn, ice, silver scales, vendors moving, handheld",
    "a child's paper boat floating down a rain gutter, ripples, low angle",
    "a glassblower shaping molten glass, orange glow, sparks, workshop",
    "a satellite view of clouds over an archipelago, slow drift, high contrast",
    "a portrait of an elderly fisherman mending a net, weathered hands, sea light",
    "a dancer's bare feet on a wooden floor, dust rising, slow motion",
    "a vintage motorcycle engine starting, exhaust smoke, chrome reflections",
    "a close-up of frost crystals forming on a window, dawn light, macro",
    "a crowded subway platform, train arriving, wind and motion blur, handheld",
]

REF_SUBJECTS = {
    "deepseek.png": "a petite young woman with very long deep-navy hair, sea-blue eyes, a navy maid dress with white lace, a white apron with a small whale emblem, a white frilled headband and a small whale tail",
    "室内 (1).jpg": "a cozy Japanese living room with warm sunlight, light wooden floors, a gray sofa with blue and yellow pillows, a wooden coffee table and large glass doors",
    "室内.jpg": "a sunlit Japanese living room with a gray sofa, wooden table, TV stand and sliding glass doors opening to a residential street",
}
GENERIC_SUBJECT = "the main subject shown in <Picture 1>"

REF_ACTIONS = [
    "turns her head slowly toward the lens and smiles",
    "walks a few steps through the room, then stops and looks off-camera",
    "reaches out and touches a nearby object, fingers brushing its surface",
    "sits down and settles into a relaxed pose, breathing calmly",
    "lifts one hand and adjusts her hair, eyes narrowing slightly",
    "stands still while the camera slowly pushes in",
    "looks down, then raises her eyes to the lens with a quiet expression",
    "turns away and begins to walk out of frame",
    "leans against a wall and crosses her arms, waiting",
    "picks something up and examines it in the light",
]

SCENE_ACTIONS = [
    "the camera slowly pushes in as sunlight shifts across the furniture",
    "a gentle breeze moves the curtains, dust motes drifting through the light",
    "the camera pans slowly left to right, revealing the depth of the room",
    "afternoon light changes from warm to soft as a cloud passes",
    "a person walks through the background, briefly crossing frame",
    "the camera tilts down toward the table, focus racking to the foreground",
]

T2VA_TEMPLATE = """subject_definitions:
N/A (text-to-video)

summary:
[text generation] A short single-take clip: {raw}.

retention_analysis:
N/A

detailed_description:
{raw}. High micro-detail, natural lighting, stable camera, no cuts.

overall_soundscape:
Natural ambient sound matching the scene, close and detailed.

non_diegetic_music:
N/A
"""

REF_TEMPLATE = """subject_definitions:
<Subject 1> is {subject} in <Picture 1>, preserving face, hairstyle, clothing, colors and body proportions exactly.

summary:
[reference generation] A short single-take clip of <Subject 1>: {action}.

retention_analysis:
<Subject 1>: fully_preserved - identity, hair, outfit and proportions remain consistent throughout.

detailed_description:
Photorealistic, cinematic soft light, clean composition. <Subject 1> (<S1>) {action}. The camera makes a slow, steady push-in; fabric and hair move naturally.

overall_soundscape:
Quiet room tone, soft fabric rustle, subtle footsteps.

non_diegetic_music:
N/A
"""

SCENE_TEMPLATE = """subject_definitions:
<Subject 1> is {subject} in <Picture 1>, preserving layout, furniture, colors and lighting exactly.

summary:
[reference generation] A short single-take clip inside <Subject 1>: {action}.

retention_analysis:
<Subject 1>: fully_preserved - architecture, furniture placement, materials and light direction remain consistent.

detailed_description:
Photorealistic, natural interior light, fine material detail. Inside <Subject 1> (<S1>), {action}. No people in the foreground; the space stays identical to the reference.

overall_soundscape:
Quiet room tone, faint outdoor ambience through the glass.

non_diegetic_music:
N/A
"""

SCENE_IMAGES = {"室内 (1).jpg", "室内.jpg"}


def _try_vga() -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []
    out = []
    for name in ("prompts/200-aesthetic-prompts.csv", "prompts/476-GQprompts.csv"):
        try:
            path = hf_hub_download(
                repo_id="BestiVictoryLab/VGA-Bench", repo_type="dataset", filename=name
            )
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            rows = list(csv.reader(io.StringIO(text)))
            for row in rows:
                if not row:
                    continue
                cell = max(row, key=len).strip()
                if len(cell) > 20 and not cell.lower().startswith(("prompt", "id")):
                    out.append(cell.replace("\n", " "))
        except Exception as exc:
            print(f"[prompts] VGA-Bench {name} skipped: {exc}")
    return out


def _try_meta() -> list[str]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        return []
    repo = "meta-ai-for-media-research/movie_gen_video_bench_no_generations"
    try:
        files = list_repo_files(repo, repo_type="dataset")
    except Exception as exc:
        print(f"[prompts] Meta bench listing skipped: {exc}")
        return []
    out = []
    for name in files:
        low = name.lower()
        if "test" not in low or not low.endswith((".jsonl", ".json", ".csv")):
            continue
        try:
            path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name)
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            if low.endswith(".jsonl"):
                for line in text.splitlines():
                    obj = json.loads(line)
                    for key in ("prompt", "caption", "text", "description"):
                        if isinstance(obj.get(key), str) and len(obj[key]) > 20:
                            out.append(obj[key].replace("\n", " "))
            elif low.endswith(".json"):
                data = json.loads(text)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            for key in ("prompt", "caption", "text", "description"):
                                if isinstance(obj.get(key), str) and len(obj[key]) > 20:
                                    out.append(obj[key].replace("\n", " "))
        except Exception as exc:
            print(f"[prompts] Meta {name} skipped: {exc}")
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = " ".join(item.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/prompts_teacher.jsonl")
    ap.add_argument("--n-ref", type=int, default=36)
    ap.add_argument("--n-t2va", type=int, default=124)
    ap.add_argument("--ref-images", nargs="*", default=["deepseek.png", "室内 (1).jpg", "室内.jpg"])
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--no-datasets", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = []
    if not args.no_datasets:
        raw = _dedupe(_try_vga() + _try_meta())
        print(f"[prompts] datasets -> {len(raw)} unique")
    if len(raw) < args.n_t2va:
        raw = _dedupe(raw + RAW_T2VA)
    raw = raw[: max(args.n_t2va * 3, args.n_t2va)]

    samples = []
    # 多长度桶：22f 40% / 39f 35% / 73f 25%
    def frames_for(i: int) -> int:
        slot = i % 20
        if slot < 8:
            return 22
        if slot < 15:
            return 39
        return 73

    for i in range(args.n_t2va):
        raw_prompt = raw[i % len(raw)]
        samples.append(
            {
                "id": f"t2va_{i:04d}",
                "task": "t2va",
                "prompt": T2VA_TEMPLATE.format(raw=raw_prompt),
                "ref_image": None,
                "frames": frames_for(i),
                "seed": rng.randrange(1, 2**31 - 1),
            }
        )

    ref_pool = [p for p in args.ref_images if p]
    if not ref_pool:
        ref_pool = ["deepseek.png"]
    for i in range(args.n_ref):
        image = ref_pool[i % len(ref_pool)]
        subject = REF_SUBJECTS.get(image, GENERIC_SUBJECT)
        is_scene = image in SCENE_IMAGES
        action = (SCENE_ACTIONS if is_scene else REF_ACTIONS)[i % len(SCENE_ACTIONS if is_scene else REF_ACTIONS)]
        template = SCENE_TEMPLATE if is_scene else REF_TEMPLATE
        samples.append(
            {
                "id": f"ref2va_{i:04d}",
                "task": "ref2va",
                "prompt": template.format(subject=subject, action=action),
                "ref_image": image,
                "frames": frames_for(i),
                "seed": rng.randrange(1, 2**31 - 1),
            }
        )

    rng.shuffle(samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[prompts] wrote {len(samples)} -> {out}")


if __name__ == "__main__":
    main()
