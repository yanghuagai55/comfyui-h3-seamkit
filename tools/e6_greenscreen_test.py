# -*- coding: utf-8 -*-
"""E-6 单元测试：绿幕模式 (Green Screen Mode)。

测试目标：
1. 上游节点 MiniMaxH3HardCutFirstPassPlan 在 green_screen_mode=True 时：
   - 不再检测提示词的分镜点（无需 [Shot N]、无需 At 时间戳、包含一镜到底/禁切词亦不报错）；
   - 全程使用 overlap（默认至少 17 帧，无需改动 overlap 参数），均等分割视频并预留 overlap 预算；
   - plan 与 hardcut 字典中均注入 green_screen_mode=True、auto_calm_search=True、calm_policy="calm_overlap"。
2. 校验器 MiniMaxH3HardCutValidate 接收绿幕 plan 时：
   - 提示词免检分镜点，直接放行通过。
3. 下游 find_calm_boundaries 在 green_screen_mode=True 时：
   - 即使转镜残差为 0 亦禁止硬切（overlap 绝不为 0）；
   - 强制使用 calm_overlap 策略在搜索范围内寻找最平缓点，所有接缝赋予 overlap 帧数。
"""

import importlib
import sys
import types

sys.path.insert(0, r"d:\comfyui\ComfyUI")
sys.path.insert(0, r"d:\comfyui\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8")

import comfy.nested_tensor  # noqa: F401
importlib.import_module("h3_t8.chunked_two_pass_upscale_advanced")

pkg = types.ModuleType("seamkit")
pkg.__path__ = [r"d:\comfyui\ComfyUI\custom_nodes\comfyui-h3-seamkit"]
sys.modules["seamkit"] = pkg

nodes = importlib.import_module("seamkit.nodes")
h3u = importlib.import_module("seamkit.h3_upscale")

fails = []

def check(name, condition, extra=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" - {extra}" if extra and not condition else ""))
    if not condition:
        fails.append(name)


print("\n=== E-6.1 上游节点绿幕模式测试 ===")

# 测试一：一镜到底提示词（无 [Shot N]、无 At 时间戳、包含禁切词）
prompt_free = (
    "A continuous single take of a woman dancing gracefully in front of a green screen. "
    "Unbroken, no cuts anywhere, camera slowly tracking forward."
)

# 1. 普通模式下预期报错（因为禁切词和缺少 Shot 分镜点）
try:
    nodes.MiniMaxH3HardCutFirstPassPlan.execute(
        prompt=prompt_free,
        total_seconds=8.0,
        target_segment_seconds=4.0,
        first_megapixels=0.4,
        second_megapixels=1.5,
        aspect_ratio="16:9 (Widescreen)",
        multiple=32,
        green_screen_mode=False,
    )
    check("普通模式拦截一镜到底提示词", False, "预期抛出 ValueError 但没有抛出")
except ValueError:
    check("普通模式拦截一镜到底提示词", True)

# 2. 绿幕模式下预期直接通过
try:
    res = nodes.MiniMaxH3HardCutFirstPassPlan.execute(
        prompt=prompt_free,
        total_seconds=8.0,
        target_segment_seconds=4.0,
        first_megapixels=0.4,
        second_megapixels=1.5,
        aspect_ratio="16:9 (Widescreen)",
        multiple=32,
        green_screen_mode=True,
    )
    plan, out_prompt, report = res[0], res[1], res[2]
    check("绿幕模式直接放行提示词", out_prompt == prompt_free)
    check("plan 中包含 green_screen_mode=True", plan.get("green_screen_mode") is True)
    hc = plan.get("hardcut", {})
    check("hardcut 字典包含 green_screen_mode=True", hc.get("green_screen_mode") is True)
    check("hardcut 自动开启 auto_calm_search", hc.get("auto_calm_search") is True)
    check("hardcut 策略强制为 calm_overlap", hc.get("calm_policy") == "calm_overlap")
    check("段间 overlap 默认生效 (>=17f)", plan.get("temporal_overlap_frames", 0) >= 17)
    check("报告中包含绿幕模式说明", "绿幕模式" in report)
except Exception as e:
    check("绿幕模式直接放行提示词", False, f"异常: {e}")


print("\n=== E-6.2 精简版 Pass2Plan seam_mode 适配测试 ===")

try:
    p2_auto = nodes.MiniMaxH3HardCutPass2Plan.execute(seam_mode="auto")[0]
    p2_hc = nodes.MiniMaxH3HardCutPass2Plan.execute(seam_mode="hard_cut")[0]
    p2_ov = nodes.MiniMaxH3HardCutPass2Plan.execute(seam_mode="smooth_overlap")[0]
    check("Pass2Plan seam_mode=auto 默认关闭 seam_blend 且开启 route3_enabled",
          p2_auto["calm_policy"] == "auto" and p2_auto["seam_blend"] is False and p2_auto["route3_enabled"] is True)
    check("Pass2Plan seam_mode=hard_cut 强制 overlap=0 且 jerk_hardcut",
          p2_hc["overlap_frames"] == 0 and p2_hc["calm_policy"] == "jerk_hardcut")
    check("Pass2Plan seam_mode=smooth_overlap 强制 overlap>=17 且 calm_overlap",
          p2_ov["overlap_frames"] >= 17 and p2_ov["calm_policy"] == "calm_overlap")
except Exception as e:
    check("Pass2Plan seam_mode 映射测试", False, f"异常: {e}")


print("\n=== E-6.3 find_calm_boundaries 绿幕模式测试 ===")

PROFILE = [
    (15, 0.90, 0.90, 1.30, 0.90),
    (20, 1.10, 1.65, 1.39, 0.82),
    (25, 1.00, 1.00, 1.10, 0.80),   # 规划帧 85
    (30, 0.40, 0.40, 0.50, 0.90),   # 最平缓 -> 帧 102
    (35, 0.80, 0.80, 0.90, 0.85),
]

aligned_hardcut = [{
    "planned_cut": 85,
    "moved": True,
    "boundary_token": 20,
    "boundary_frame": 68,
    "measured_turn_frame": 68,     # residual 0f -> 原本必硬切
    "ratio": 2.5,
}]

KW_BASE = dict(
    window=51,
    overlap_frames=17,
    seam_tolerance=4,
    abstain_below=0.0,
    calm_min_quality=0.8,
    too_quiet_below=0.05,
    calm_min_gain=0.15,
)

# 非绿幕模式下：残差 0 应该硬切 (overlap 0)
b_normal, ov_normal, _ = h3u.find_calm_boundaries(
    PROFILE, [85], aligned_hardcut, 362,
    policy="calm_overlap",
    green_screen_mode=False,
    **KW_BASE
)
check("非绿幕模式下残差0硬切", ov_normal == [0] and b_normal == [68])

# 绿幕模式下：即便残差 0，亦严禁硬切，走平缓搜索且必须 overlap 17
b_green, ov_green, notes_green = h3u.find_calm_boundaries(
    PROFILE, [85], aligned_hardcut, 362,
    policy="calm_overlap",
    green_screen_mode=True,
    **KW_BASE
)
check("绿幕模式严禁硬切并走平缓 overlap", ov_green == [17] and b_green == [102])
check("绿幕日志包含 green screen mode", any("green screen mode" in n for n in notes_green))

# 绿幕模式下 + policy=jerk_hardcut：依然严禁硬切，强制 calm_overlap
b_jerk_green, ov_jerk_green, _ = h3u.find_calm_boundaries(
    PROFILE, [85], aligned_hardcut, 362,
    policy="jerk_hardcut",
    green_screen_mode=True,
    **KW_BASE
)
check("绿幕模式强制覆盖 jerk_hardcut 为 overlap", ov_jerk_green == [17] and b_jerk_green == [102])


print("\n" + "=" * 40)
if fails:
    print(f"FAILED ({len(fails)} failures): {fails}")
    sys.exit(1)
else:
    print("ALL GREEN SCREEN TESTS PASSED!")
