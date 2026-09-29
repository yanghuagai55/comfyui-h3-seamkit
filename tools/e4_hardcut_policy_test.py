# -*- coding: utf-8 -*-
"""E-4 单元测试：硬切 / 锚定 overlap 的判定必须只看「边界 vs 测得转镜帧」。

背景（2026-09-23 实测抓到的 bug）：
  `_align_to_profile` 原来只在 snap 移动过帧号时才写 `measured_turn_frame`；
  峰值本来就落在 17 的整数倍上时这个键缺失，`find_calm_boundaries` 便静默
  fallback 到 `|boundary - planned|`（规划漂移）—— 与门限 4 完全不同的量级。
  后果：同一个情形（规划 85、转镜 68 与转镜 66）会分别得到「锚定 overlap」和
  「硬切」，差别只在于 argmax 恰好落在哪。

修复后的语义：
  边界只能落在独占帧（17k）上，所以残差 = 转镜帧的网格量化误差（0~8）。
  残差 <= seam_tolerance -> 硬切（缝正压在模型自己的转镜上）。
  拿不到测得转镜 -> 不硬切，交给 calm 搜索做锚定 overlap。

验证：
  A  转镜 68（在网格上）           -> 硬切 @68
  B  转镜 66（snap 66->68）        -> 硬切 @68   ← 修复前这里是硬切、A 是 overlap
  C  搜索弃权（无 boundary_frame） -> calm 搜索，overlap
  D  转镜 60，边界 68（残差 8>4）  -> calm 搜索，overlap
  E  有边界但无转镜键（回归防线）  -> 不许硬切（修复前会硬切 @85）
  F  真实 00067 的三条缝           -> 全部硬切
"""
import importlib
import sys
import types

sys.path.insert(0, r"d:\comfyui\ComfyUI")
sys.path.insert(0, r"d:\comfyui\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8")

# _align_to_profile / find_calm_boundaries go through _core(), which resolves the
# upstream H3 pack by module name - so it has to be importable in this process.
import comfy.nested_tensor  # noqa: F401
importlib.import_module("h3_t8.chunked_two_pass_upscale_advanced")

pkg = types.ModuleType("seamkit")
pkg.__path__ = [r"d:\comfyui\ComfyUI\custom_nodes\comfyui-h3-seamkit"]
sys.modules["seamkit"] = pkg
h3u = importlib.import_module("seamkit.h3_upscale")

# 与 `硬切自动版-锚定.json` 一致的判定参数
KW = dict(
    window=51,
    overlap_frames=17,
    seam_tolerance=4,
    policy="calm_overlap",
    abstain_below=0.0,
    calm_min_quality=0.8,
    too_quiet_below=0.05,
    calm_min_gain=0.15,
)

# profile 行 = (idx, local, glob, jerk, pers)；score = max(glob, jerk)
#   帧 51 / 68 / 85 / 102 / 119  <->  token 15 / 20 / 25 / 30 / 35
PROFILE = [
    (15, 0.90, 0.90, 1.30, 0.90),
    (20, 1.10, 1.65, 1.39, 0.82),
    (25, 1.00, 1.00, 1.10, 0.80),   # 规划帧 85，score 1.1
    (30, 0.40, 0.40, 0.50, 0.90),   # 最安静 -> 帧 102，score 0.5
    (35, 0.80, 0.80, 0.90, 0.85),
]


def run(aligned, planned=(85,), frame_count=362):
    return h3u.find_calm_boundaries(
        PROFILE, list(planned), aligned, frame_count, **KW
    )[:2]


def entry(boundary_frame, measured=None, cut=85):
    e = {"planned_cut": cut, "moved": boundary_frame != cut,
         "boundary_token": boundary_frame // 17 * 5,
         "boundary_frame": boundary_frame, "ratio": 1.65,
         "local_ratio": 1.18, "top_candidates": []}
    if measured is not None:
        e["measured_turn_frame"] = measured
    return e


fails = []


def check(name, got, want):
    ok = tuple(got[0]) == tuple(want[0]) and tuple(got[1]) == tuple(want[1])
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: boundaries={got[0]} overlaps={got[1]}"
          + ("" if ok else f"   期望 {want}"))
    if not ok:
        fails.append(name)


print("\n=== E-4 硬切判定 ===")

check("A 转镜 68（在网格上）", run([entry(68, 68)]), ((68,), (0,)))
check("B 转镜 66（snap 到 68）", run([entry(68, 66)]), ((68,), (0,)))
check("C 搜索弃权 -> calm 搜索", run([{"planned_cut": 85, "moved": False,
                                     "note": "abstained", "top_candidates": []}]),
      ((102,), (17,)))
check("D 转镜 60（残差 8 > 4）", run([entry(68, 60)]), ((102,), (17,)))
check("E 有边界但无转镜键 -> 不许硬切", run([entry(85)]), ((102,), (17,)))

# 真实 00067：三条缝的峰值本身就是边界（68 / 187 / 255），修复后全部硬切
REAL = [
    entry(68, 68, cut=85),
    entry(187, 187, cut=187),
    entry(255, 255, cut=272),
]
check("F 真实 00067 三条缝全硬切",
      run(REAL, planned=(85, 187, 272), frame_count=362),
      ((68, 187, 255), (0, 0, 0)))

def check_line(name, line, needles, absent=()):
    missing = [n for n in needles if n not in line]
    present = [n for n in absent if n in line]
    ok = not missing and not present
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {line}")
    if not ok:
        print(f"         缺: {missing} / 不该有: {present}")
        fails.append(name)


print("\n=== E-4 空窗口不再崩（原为 NameError: notes）===")
try:
    got = h3u._align_to_profile([(15, 0.90, 0.90, 1.30, 0.90)], [340], 4, 107,
                                search_window=34)
    if got == ([], []):
        print(f"  [PASS] 空窗口返回 {got}，不抛异常")
    else:
        print(f"  [FAIL] 空窗口返回 {got}，期望 ([], [])")
        fails.append("empty window")
except NameError as exc:
    print(f"  [FAIL] 仍然 NameError: {exc}")
    fails.append("empty window NameError")

print("\n=== E-4 日志行格式（唯一诊断窗口）===")
L = h3u._hunt_log_line
check_line(
    "L1 接受 + 吸附说明",
    L({"planned_cut": 85, "moved": True, "boundary_frame": 68,
       "measured_turn_frame": 73, "ratio": 1.78,
       "note": "snapped the measured turn 73 -> 68 (nearest exclusive frame)"}),
    ["cut 85: bound 68", "planΔ -17", "turn 73 residual 5f", "ratio 1.78"],
)
check_line(
    "L2 持续性门拒绝（本次 187 就是这个）",
    L({"planned_cut": 187, "moved": False, "ratio": None,
       "note": "local change is real but not persistent (persistence 0.62 < 0.80)"
               " -> treated as a false positive, cut not accepted"}),
    ["cut 187: REJECTED", "persistence 0.62"],
)
check_line(
    "L3 平坦门拒绝",
    L({"planned_cut": 187, "moved": False, "ratio": None,
       "note": "latent is flat near the cut (peak ratio 1.42 < 1.6)"}),
    ["cut 187: REJECTED", "latent is flat"],
)
check_line(
    "L4 无 note 也不崩、不出现空 why",
    L({"planned_cut": 85, "moved": False, "ratio": 1.20, "boundary_frame": 85,
       "measured_turn_frame": 85}),
    ["cut 85: bound 85", "planΔ +0", "turn 85 residual 0f", "ratio 1.2"],
    absent=("why:",),
)

print("\n=== E-4 转镜置信门（RATIO，exp_4v10a_00086 @51 事故复现）===")
try:
    got = h3u._align_to_profile(
        [(15, 1.50, 1.69, 1.00, 0.95)],  # ratio 1.69 ∈ [FLAT 1.6, 置信门 2.0)，persistence 0.95 过
        [51], 4, 107, min_persistence=0.80)
    entry = got[0][0] if got[0] else {}
    ok = (got[1] == [] and entry.get("moved") is False
          and entry.get("boundary_frame") is None
          and "below turn confidence" in (entry.get("note") or ""))
    if ok:
        print(f"  [PASS] ratio 1.69 拒绝硬切、回退锚定: {entry.get('note')}")
    else:
        print(f"  [FAIL] {got}")
        fails.append("ratio gate")
except Exception as exc:
    print(f"  [FAIL] 异常: {exc}")
    fails.append("ratio gate exception")

def check_ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


print("\n=== E-4 calm 策略树（auto：静缝挪 / 闹缝硬切）===")
_kw_auto = dict(KW)
_kw_auto["policy"] = "auto"
# 缝 85 的 aligned 条目：有边界帧、**没有** measured_turn_frame
#   → 不许硬切（同测试 E），落到 calm 搜索；K2 里则由闹度门改走 jerk 硬切。
# 注：不调 `entry()` 构造器 —— 上面 `for entry in ...` 循环会把那个名字
#     覆盖成 dict（TypeError: 'dict' object is not callable）。这里写字面量。
_e85 = {"planned_cut": 85, "moved": False, "boundary_token": 85 // 17 * 5,
        "boundary_frame": 85, "ratio": 1.65, "local_ratio": 1.18,
        "top_candidates": []}
# 静景 profile：缝 85（token 25）邻域闹度 = 1.10/0.90 = 1.22 < 1.5 → calm 路径
_got = h3u.find_calm_boundaries(PROFILE, [85], [dict(_e85)], 362, **_kw_auto)[:2]
check_ok("K1 auto·静缝 → calm 挪最平缓(102) + overlap17",
         _got == ([102], [17]), f"got={_got}")

# 剧烈 profile：缝 85 邻域 jerk 全在 2.0+，全片中位 0.45 → busy≈5.6 ≥1.5 → jerk 路径
PROFILE_LOUD = [
    (10, 0.20, 0.25, 0.30, 0.85),
    (15, 0.30, 0.35, 0.40, 0.85),
    (20, 0.50, 0.55, 2.00, 0.88),
    (25, 0.60, 0.60, 2.50, 0.90),   # 缝 token：邻域最炸
    (30, 0.55, 0.50, 2.10, 0.87),
    (35, 0.30, 0.35, 0.50, 0.86),
    (40, 0.20, 0.25, 0.30, 0.85),
    (45, 0.20, 0.22, 0.28, 0.85),
]
_got = h3u.find_calm_boundaries(
    PROFILE_LOUD, [85], [dict(_e85)], 362, **_kw_auto
)[:2]
check_ok("K2 auto·闹缝 → jerk 峰硬切(85) + overlap0",
         _got == ([85], [0]), f"got={_got}")

print()
if fails:
    print(f"E-4 有 {len(fails)} 项失败：" + ", ".join(fails))
    sys.exit(1)
print("E-4 全过：硬切/锚定判定的唯一依据是「边界 vs 测得转镜帧」，"
      "不再受规划漂移或键是否写入影响。")
