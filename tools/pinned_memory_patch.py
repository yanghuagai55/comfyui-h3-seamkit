# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""ComfyUI pinned-memory 上限补丁（多档位，交互式小终端）。

改 **两个** 独立的数值，都可以自由切档、随时复原：

    A. 主机侧 pinned 上限   comfy/model_management.py:1593  MAX_PINNED_MEMORY = ram * ratio
    B. 设备侧 cast 预留     comfy/model_management.py:1378  DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE

================================================================================
A. 主机侧 pinned 上限
================================================================================
aimdo 给模型权重用的 **页锁定主机内存** 上限是这么算的：

    comfy/model_management.py   MAX_PINNED_MEMORY = ram * 0.40             ← Windows 分支（原版）
                                pinned_hostbuf_size = min(size, MAX) * 2   ← 注意这个 x2
    comfy/model_patcher.py      HostBuffer(0, 64MB, pinned_hostbuf_size(model.model_size()))

即 **上限 ≈ 0.80 x 物理内存**。MiniMax H3 二采实测需要 ~25.6 GiB，
32GB 机器上原版上限只有 ~25.3 GiB，于是分段采样中途被拒：

    aimdo: src/hostbuf.c:46:ERROR:hostbuf_grow: requested ... beyond reserved host buffer ...

净差只有 3.5~235 MiB，所以表现很迷惑：**前面几段能跑完，后面某一段撞线**
（上限建模型时定死、请求大小随内容浮动 → 每段都是掷骰子、与段长无关）。

【A 的档位表】（ram = 本机物理内存；本机实测 31.63 GiB）
    ram*0.40  原版      上限 ≈ 0.80 x ram ≈ 25.30 GiB  ComfyUI 默认；复原档
    ram*0.45  标准档    上限 ≈ 0.90 x ram ≈ 28.47 GiB  日常跑片
    ram*0.50  缓存一采档 上限 ≈ 1.00 x ram ≈ 31.63 GiB  ★ 见下

【★ 缓存一采档（ram*0.50）为什么存在】
`MiniMaxH3FirstPassSampler` 走 **一采缓存 HIT** 时，一采采样被整体跳过，
但**二采仍要加载 32.4 GB 主模型、并第一次建立 host buffer**。
本机 2026-09-25 14:38 实测，HIT 轮二采的请求峰值：

    requested 30635134976 bytes beyond reserved host buffer 30631395328   ← 只差 3.57 MiB
    requested 30741573632 bytes beyond reserved host buffer 30631395328   ← 差 105 MiB

- 需求峰值 = 30741573632 B = **28.63 GiB**
- 0.45 档 reserve = 30631395328 B = **28.53 GiB** → 差 105 MiB，被拒
- 0.50 档 reserve ≈ **31.63 GiB** → 余量 ~3.0 GiB，覆盖这类场景

**代价说清楚**：reserve 只是"预留地址空间 + 按需 extend"（HostBuffer 初始仅 8 MB，
prewarm 64 MB），所以 **抬 cap 的实际代价 ≈ 超出量** —— 本场景只会比 0.45 多占 ~105 MiB，
不是一次性吃满 31.63 GiB。**但** reserve ≈ 1.00 x 物理内存，万一某轮真吃满会挤压系统，
所以本档定位为"HIT / 缓存链路专用"，日常跑片建议切回标准档。

为什么不干脆用启动参数 `--high-ram`：它把上限**整个去掉**
（`pinned_hostbuf_size` 直接 `return size * 2`，32.4 GB 模型 → **64.8 GiB**），
还会连带 `cache_classic=True`。pinned 内存不能换页，护栏拆掉后失败模式会从
"干净报错"变成"掉页 / 整机卡死"。**护栏留着** —— 本档只是把护栏抬到物理内存量级。

================================================================================
B. ★ 设备侧 cast 预留（`host_buffer.py:109 read_file_slice failed` 的真凶）
================================================================================
**这不是 pinned 问题**，别再抬 A 来治它。链条：

    comfy/model_management.py:1378  DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3
    comfy/model_management.py:1416  get_aimdo_cast_buffer() -> VRAMBuffer(DEFAULT_..., device.index)
    comfy_aimdo/vram_buffer.py:32   VRAMBuffer.__init__ -> lib.vrambuf_create(device, 16 GiB)
    comfy/ops.py:160-161            cast_buffer.get(buffer_size, offset) -> aimdo_to_tensor(...)
    comfy/ops.py:466                cast_bias_weight  ← 32B TE 前向第一个权重去量化时就崩

即 aimdo 为**权重去量化/类型转换**申请了一块 **16 GiB 的设备显存 VRAMBuffer**，
本机只有 8187 MB → `vrambuf_grow` 拿不到 → 往那块 VRAM 拷权重时
`HostBuffer.read_file_slice failed` → Sticky error → `CUDA error: out of memory`。

**崩点固定、确定性可复现**（本机 2026-09-25 三次实测，栈逐行相同，与 HIT/缓存无关）：
    nodes.py:340 → conditioning.py:543 → sd.py:410 → text_encoders/minimax.py:100
      → qwen3vl.py:100 → llama.py:906/892/685/651 → ops.py:466 cast_bias_weight
      → ops.py:377/234/232/223 → model_management.py:1525 cast_to_gathered
      → memory_management.py:70 → comfy_aimdo/host_buffer.py:109

【B 的档位表】（本机显存 8187 MB）
    16 GiB   原版      ComfyUI 默认；复原档（8 GB 卡上必然失败）
    4 GiB    保守档    留足余量，覆盖绝大多数单权重转换
    2 GiB    极小档    只在保守档仍失败时用

**语义要点（决定风险）：**
`vrambuf_create(max_size)` 里的 max_size 是**预留上限**，不是立刻分配 ——
真正占显存发生在 `VRAMBuffer.get()` 首次 `vrambuf_grow`，且按 16 MB chunk 向上取整。
所以压小 max_size **不会立即省显存**，但会**给 grow 设一个天花板**：
一旦某个权重转换请求超过它，就会从"OOM"变成 `VRAM grow failed: N bytes`。
**B 的档位不要设得比 B 档位表更小**；若出现 `VRAM grow failed`，往上调一档。

> ⚠️ 该机制原作者标注 *"this is temporary and will be removed in a future comfy.
> Not supported for custom node use."* —— 所以本工具只改**数值**，不动逻辑；
> ComfyUI 升级后状态会变 unknown，届时按 §升级后 处理。

================================================================================
安全设计（两个数值共用同一套）
================================================================================
  - 打补丁前完整备份；`.orig` 只在"当前确实是原版"时写入一次，**绝不覆盖**
  - 正则精确匹配目标行做单次替换，命中数 != 1 就拒绝
    （A 锚在 `if WINDOWS:` 上，**不会误改 Linux 分支那行同样的 0.40**）
  - 状态机 orig / profile / unknown；unknown（例如 ComfyUI 升级过）**一律拒绝，不硬来**
  - 切档时若缺 `.orig` 且当前不是原版 → 拒绝（保证任何状态都退得回原版）
  - 复原走逐字节拷贝，不靠"反向替换"
  - 写盘后做"新文本 + 落盘"双重校验，失败自动回滚

用法：
    <ComfyUI>/comfyenv/python.exe <本仓库>/tools/pinned_memory_patch.py
    ...python pinned_memory_patch.py [model_management.py 的路径]

    # 非交互（脚本 / 自动化）
    ...python pinned_memory_patch.py --status
    ...python pinned_memory_patch.py --to 0.50 -y            # A 切到缓存一采档
    ...python pinned_memory_patch.py --to 0.45 -y            # A 切回标准档
    ...python pinned_memory_patch.py --cast-to 4g -y         # B 切到保守档（4 GiB）
    ...python pinned_memory_patch.py --cast-to 2g -y         # B 切到极小档
    ...python pinned_memory_patch.py --restore -y            # 两个都复原为原版
    ...python pinned_memory_patch.py --restore-a -y          # 只复原 A
    ...python pinned_memory_patch.py --restore-b -y          # 只复原 B
"""
import hashlib
import re
import shutil
import socket
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

# ---- 路径从文件位置推导，不硬编码 -------------------------------------------------
# <ComfyUI>/custom_nodes/comfyui-h3-seamkit/tools/pinned_memory_patch.py
SCRIPT = Path(__file__).resolve()
PLUGIN_DIR = SCRIPT.parents[1]          # comfyui-h3-seamkit
COMFY_ROOT = SCRIPT.parents[3]          # ComfyUI
HOME_DIR = COMFY_ROOT.parent            # D:\comfyui 这类

TARGET = COMFY_ROOT / "comfy" / "model_management.py"

# 备份位置：优先用"已经有 .orig 的那个"，否则用第一个（新位置）。
# 新增候选时**往后面加**，别调换顺序 —— 换顺序会让已有的备份被"看不见"。
BACKUP_DIRS = [
    COMFY_ROOT / "user" / "pinned_memory_patch",          # 默认
]
HOST_PORT = 8188

# ---- A. 主机侧 pinned 档位表 ------------------------------------------------------
RATIO_ORIG = 0.40           # ComfyUI 原版（Windows 分支）
RATIO_STD = 0.45            # 标准档
RATIO_CACHE = 0.50          # ★ 缓存一采档（一采缓存 HIT 链路专用）

PROFILES = {
    RATIO_STD: ("标准档", "日常跑片；上限 ≈ 0.90 x 物理内存"),
    RATIO_CACHE: ("缓存一采档 ★", "一采缓存 / HIT 链路专用；上限 ≈ 1.00 x 物理内存"),
}

# --to 允许的范围：低于原版没意义，高于 0.60 就是拆护栏了（那种情况用 --high-ram）
RATIO_MIN, RATIO_MAX = 0.40, 0.60

# 只认这一行：`if WINDOWS:` 之后（允许中间夹注释行）的 MAX_PINNED_MEMORY 赋值
PAT_ANY = re.compile(
    r"if WINDOWS:\n(?:[ \t]*#.*\n)*[ \t]*MAX_PINNED_MEMORY = ram \* ([0-9]+(?:\.[0-9]+)?)\b")
PAT_SUB = re.compile(
    r"(if WINDOWS:\n(?:[ \t]*#.*\n)*[ \t]*)MAX_PINNED_MEMORY = ram \* [0-9]+(?:\.[0-9]+)?\b")

# ---- B. 设备侧 cast 预留档位表（单位 GiB）------------------------------------------
CAST_ORIG = 16              # ComfyUI 原版
CAST_PROFILES = {
    4: ("保守档", "8 GB 卡推荐；覆盖绝大多数单权重转换"),
    2: ("极小档", "只在保守档仍失败时用"),
}

# --cast-to 允许的范围：太小会让权重转换直接 VRAM grow failed，太大等于没改
CAST_MIN, CAST_MAX = 1, 64

# 目标行：DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = <int> * 1024 ** 3
PAT_CAST_ANY = re.compile(
    r"^DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = ([0-9]+) \* 1024 \*\* 3", re.M)
PAT_CAST_SUB = re.compile(
    r"^DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = [0-9]+ \* 1024 \*\* 3", re.M)

# ---- 启动脚本（bat）参数管理 -------------------------------------------------------
# 只改 `set COMMANDLINE_ARGS=` 行里"基础部分之后的附加参数"，基础部分必须原样保留。
# bat 是 GBK 编码（含 `D:\共享`），**全程字节级处理，绝不按 UTF-8 读写**。
BAT = COMFY_ROOT.parent / "启动ComfyUI.bat"          # D:\comfyui\启动ComfyUI.bat
BAT_KEY = b"set COMMANDLINE_ARGS="
BAT_BASE = '--use-sage-attention --output-directory "D:\\共享"'

# ---- 配置预设：一键把 A / B / bat 附加参数切到某个完整组合 ---------------------------
# (key, 显示名, A 比例, B GiB, bat 附加参数, 备注)
PRESETS = [
    ("official", "官方默认", 0.40, 16, "",
     "完全回到 ComfyUI 出厂状态"),
    ("std045", "0.45 默认", 0.45, 16, "",
     "本机旧基线（13:40 那次跑通用的就是它）"),
    ("stable", "新设置（稳定跑通）★", 0.45, 16, "--disable-async-offload",
     "2026-09-25 实测二采 4/4 全过、25分33秒出片"),
]
PRESET_BY_KEY = {k: (k, n, a, b, e, note) for k, n, a, b, e, note in PRESETS}

GIB = 1024 ** 3
MIB = 1024 ** 2


# ---- 基础工具 --------------------------------------------------------------------
BACKUP_DIRS_PINNED = False      # --backup-dir 显式指定时置 True：禁用"回退找 .orig"


def backup_dir() -> Path:
    """决定用哪个备份目录。

    正常情况：优先用"已经有 .orig 的那个"（保证既有备份不被"看不见"）。
    但若用户用 --backup-dir 显式指定了目录，就**只用那个** —— 否则隔离失效，
    试跑会意外读到真实备份并往里写 bak（2026-09-25 实测踩过）。
    """
    if BACKUP_DIRS_PINNED:
        return BACKUP_DIRS[0]
    for d in BACKUP_DIRS:
        if (d / "model_management.py.orig").exists():
            return d
    return BACKUP_DIRS[0]


def sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def total_ram_bytes() -> int:
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return int(ms.ullTotalPhys)


def comfyui_running() -> bool:
    s = socket.socket()
    s.settimeout(0.6)
    try:
        s.connect(("127.0.0.1", HOST_PORT))
        return True
    except Exception:
        return False
    finally:
        s.close()


def read_target() -> str:
    return TARGET.read_text(encoding="utf-8")


# ---- 状态识别 --------------------------------------------------------------------
def read_ratio(text: str):
    """读出 Windows 分支当前的比例。命中数 != 1（0 处或多处）-> None。"""
    ms = PAT_ANY.findall(text)
    if len(ms) != 1:
        return None
    try:
        return float(ms[0])
    except (TypeError, ValueError):
        return None


def profile_key(ratio):
    """把读到的浮点对齐到档位表的键；不对齐任何档 -> None。"""
    for k in sorted(PROFILES):
        if abs(ratio - k) < 1e-6:
            return k
    return None


def state_of(text: str):
    """返回 (kind, ratio)。kind ∈ {'orig', 'profile', 'custom', 'unknown'}。

    'custom' = 落在 [RATIO_MIN, RATIO_MAX] 内但不在预设档位表里的值（例如用 --to 0.47 切过去的）。
    这类值必须**可切回**，否则切出去就回不来了 —— 2026-09-25 实测踩过这个坑。
    """
    r = read_ratio(text)
    if r is None:
        return "unknown", None
    if abs(r - RATIO_ORIG) < 1e-6:
        return "orig", r
    if profile_key(r) is not None:
        return "profile", r
    if RATIO_MIN - 1e-9 <= r <= RATIO_MAX + 1e-9:
        return "custom", r
    return "unknown", r


def label_of(kind: str, ratio) -> str:
    if kind == "orig":
        return f"原版 (ram * {RATIO_ORIG:.2f})"
    if kind == "profile":
        return f"{PROFILES[profile_key(ratio)][0]} (ram * {ratio:.2f})"
    if kind == "custom":
        return f"自定义档 (ram * {ratio:.2f})"
    if ratio is None:
        return "!! 无法识别（没找到那一行 —— 可能被改过，或 ComfyUI 升级换了写法）"
    return f"!! 无法识别（读到 ram * {ratio:.2f}，不在允许范围 [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}] 内）"


# ---- B. 设备侧 cast 预留：读写与状态 -------------------------------------------------
def read_cast_gib(text: str):
    """读出 DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE 的 GiB 整数。命中数 != 1 -> None。"""
    ms = PAT_CAST_ANY.findall(text)
    if len(ms) != 1:
        return None
    try:
        return int(ms[0])
    except (TypeError, ValueError):
        return None


def cast_profile_key(gib):
    """对齐到档位表的键；16 视为原版；不对齐任何档 -> None。"""
    if gib is None:
        return None
    if gib == CAST_ORIG:
        return CAST_ORIG
    for k in CAST_PROFILES:
        if gib == k:
            return k
    return None


def cast_state_of(text: str):
    """返回 (kind, gib)。kind ∈ {'orig', 'profile', 'custom', 'unknown'}。"""
    g = read_cast_gib(text)
    if g is None:
        return "unknown", None
    if g == CAST_ORIG:
        return "orig", g
    if g in CAST_PROFILES:
        return "profile", g
    if CAST_MIN <= g <= CAST_MAX:
        return "custom", g
    return "unknown", g


def cast_label_of(kind: str, gib) -> str:
    if kind == "orig":
        return f"原版 ({CAST_ORIG} GiB)"
    if kind == "profile":
        return f"{CAST_PROFILES[gib][0]} ({gib} GiB)"
    if kind == "custom":
        return f"自定义档 ({gib} GiB)"
    if gib is None:
        return "!! 无法识别（没找到那一行 —— 可能被改过，或 ComfyUI 升级换了写法）"
    return f"!! 无法识别（读到 {gib} GiB，不在允许范围 [{CAST_MIN}, {CAST_MAX}] 内）"


def projected(ratio: float) -> tuple:
    """返回 (MAX_PINNED_MEMORY, 上限 = x2)。"""
    ram = total_ram_bytes()
    mp = ram * ratio
    return mp, mp * 2


def vram_total_bytes():
    """本机显存总量（字节）。取不到返回 None。"""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


# ---- 界面 ------------------------------------------------------------------------
def show_status() -> str:
    """打印 A/B 两组状态，返回 A 的 kind ∈ {'orig','profile','unknown','missing'}。"""
    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    print("=" * 74)
    print(" ComfyUI 显存/内存上限补丁（A: 主机 pinned  ·  B: 设备侧 cast 预留）")
    print("=" * 74)
    print(f"  目标文件 : {TARGET}")
    if not TARGET.exists():
        print("  !! 文件不存在。若 ComfyUI 不在默认位置，把路径当参数传进来：")
        print("     python pinned_memory_patch.py <路径>/comfy/model_management.py")
        return "missing"

    text = read_target()
    kind, ratio = state_of(text)
    ckind, cgib = cast_state_of(text)
    ram = total_ram_bytes()
    vtot = vram_total_bytes()

    print(f"  A 主机 pinned : {label_of(kind, ratio)}")
    print(f"  B 设备 cast   : {cast_label_of(ckind, cgib)}")
    print(f"  文件指纹 : sha256:{sha(TARGET)}")
    print(f"  备份目录 : {bdir}")
    if orig.exists():
        print(f"  原始备份 : {orig.name}  sha256:{sha(orig)}")
    else:
        print("  原始备份 : （还没有，切档时会自动建立 —— 但仅当当前是原版时）")
    mem = f"  物理内存 : {ram / GIB:.2f} GiB"
    if vtot:
        mem += f"      显存 : {vtot / GIB:.2f} GiB ({vtot / MIB:.0f} MB)"
    print(mem)

    print()
    print("  ── A. 主机 pinned 上限（≈ 2 x ram x ratio）──────────────────────────")
    print("     档位                    MAX_PINNED     上限(x2)     说明")
    for r, (name, note) in [(RATIO_ORIG, ("原版", "ComfyUI 默认；复原档"))] + \
                           sorted(PROFILES.items()):
        mp, rv = ram * r, ram * r * 2
        mark = " <== 当前" if (ratio is not None and abs(r - ratio) < 1e-6) else ""
        print(f"     ram*{r:.2f}  {name:<12} {mp / GIB:6.2f} GiB  {rv / GIB:7.2f} GiB   {note}{mark}")
    print("     作用：修 `hostbuf_grow: requested N beyond reserved M`（hostbuf.c:46）")

    print()
    print("  ── B. 设备侧 cast 预留（VRAMBuffer max_size）────────────────────────")
    print("     档位               预留上限      说明")
    for g, (name, note) in [(CAST_ORIG, ("原版", "ComfyUI 默认；8 GB 卡上必然失败"))] + \
                           sorted(CAST_PROFILES.items()):
        mark = " <== 当前" if (cgib is not None and g == cgib) else ""
        flag = ""
        if vtot and g * GIB > vtot:
            flag = "  ⚠ 超过显存!"
        print(f"     {g:>2} GiB  {name:<10} {g:>6} GiB   {note}{mark}{flag}")
    print("     作用：修 `host_buffer.py:109 read_file_slice failed` / Sticky error")
    print("     （真凶是 16 GiB 设备侧预留 vs 8 GB 显存，抬 A 无效）")
    print()
    print("  ── 配置预设（A + B + bat 附加参数 一键切）────────────────────────")
    cur_extra = bat_current_extra()
    cur_extra_n = (cur_extra or "").strip()
    for k, n, a, b, e, note in PRESETS:
        hit = ""
        if (ratio is not None and abs(a - ratio) < 1e-6 and cgib == b
                and cur_extra_n == e.strip()):
            hit = "   <== 当前"
        print(f"     [{k:<9}] {n:<18} A={a:.2f} B={b:>2}GiB  bat=`{e or '（无）'}`{hit}")
    print(f"     bat 实际附加参数 : `{cur_extra if cur_extra is not None else '(读不到)'}`")
    print(f"     （bat = {BAT}）")
    print()
    print("  ComfyUI 正在运行（8188 有响应）：改完必须重启才生效"
          if comfyui_running() else
          "  ComfyUI 未在运行（改动会在下次启动时生效）")
    print("=" * 74)
    return kind


def cast_looks_broken() -> bool:
    """B 档当前是否与显存明显不匹配（用于给出建议）。"""
    text = read_target()
    _, cgib = cast_state_of(text)
    vtot = vram_total_bytes()
    if cgib is None or not vtot:
        return False
    return cgib * GIB > vtot


def ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "1", "是", "确定")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ---- 切档 / 复原 ------------------------------------------------------------------
# A/B 两个数值共用同一套"备份 → 替换 → 双重校验 → 失败回滚"流程。
def _prepare_backup(cur_kind_orig: bool, cur_label: str, *, what: str):
    """建立/校验备份。返回 (bdir, orig, stamp) 或 None。"""
    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    bdir.mkdir(parents=True, exist_ok=True)
    if not orig.exists():
        if cur_kind_orig:
            shutil.copy2(TARGET, orig)
            print(f"\n  已建立原始备份 : {orig}")
        else:
            print(f"\n  !! 缺原始备份（{orig}），且当前不是原版（{cur_label}）。")
            print(f"     拒绝改 {what} —— 否则一旦出问题就退不回原版。")
            print("     补救：先把文件手工改回原版值，再跑本工具。")
            return None
    stamp = bdir / f"model_management.py.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(TARGET, stamp)
    print(f"  已建立本次备份 : {stamp.name}")
    return bdir, orig, stamp


def _verify_or_rollback(new_text: str, checker, expect, stamp) -> bool:
    """双重校验：文本 + 落盘。失败则回滚。"""
    if checker(new_text) != expect:
        print("  !! 文本校验失败，正在回滚 ...")
        shutil.copy2(stamp, TARGET)
        print(f"  已回滚，文件指纹 sha256:{sha(TARGET)}")
        return False
    if checker(read_target()) != expect:
        print("  !! 落盘校验失败，正在回滚 ...")
        shutil.copy2(stamp, TARGET)
        print(f"  已回滚，文件指纹 sha256:{sha(TARGET)}")
        return False
    return True


def switch_to(target_ratio: float, *, assume_yes: bool = False) -> bool:
    """把 A（Windows 分支的 pinned 比例）切到 target_ratio。"""
    if not TARGET.exists():
        print("\n  !! 目标文件不存在。")
        return False

    if not (RATIO_MIN - 1e-9 <= target_ratio <= RATIO_MAX + 1e-9):
        print(f"\n  !! 目标比例 {target_ratio} 超出允许范围 "
              f"[{RATIO_MIN:.2f}, {RATIO_MAX:.2f}]，拒绝。")
        print("     想完全去掉上限请用 ComfyUI 的 --high-ram 启动参数（风险自负）。")
        return False

    text = read_target()
    kind, cur = state_of(text)
    if cur is None or kind == "unknown":
        print("\n  !! A 的状态无法识别，拒绝动手。")
        if cur is not None:
            known = " / ".join(f"{r:.2f}" for r in
                              sorted([RATIO_ORIG] + list(PROFILES)))
            print(f"     读到的是 ram * {cur:.2f}，不在已知档位 [{known}] 里。")
        print("     可能 ComfyUI 升级过、或这行被人改过。请手工核对：")
        print(f"     {TARGET}")
        return False

    if abs(cur - target_ratio) < 1e-6:
        print(f"\n  已经是 {label_of(kind, cur)}，不需要切。")
        return False

    bk = _prepare_backup(abs(cur - RATIO_ORIG) < 1e-6, label_of(kind, cur), what="A")
    if bk is None:
        return False
    _, _, stamp = bk

    new_line = f"MAX_PINNED_MEMORY = ram * {target_ratio:.2f}"
    new_text, n = PAT_SUB.subn(lambda m: m.group(1) + new_line, text)
    if n != 1:
        print(f"\n  !! 替换命中 {n} 处（应为 1），拒绝动手。")
        return False

    tgt_name = PROFILES.get(profile_key(target_ratio), ("自定义档", ""))[0] \
        if profile_key(target_ratio) else ("原版" if abs(target_ratio - RATIO_ORIG) < 1e-6 else "自定义档")
    _, rv = projected(target_ratio)
    print(f"\n  将把 A `ram * {cur:.2f}` 切到 `ram * {target_ratio:.2f}`（{tgt_name}）")
    print(f"    新上限 ≈ {rv / GIB:.2f} GiB")
    if not assume_yes and not ask("  继续？[y/N] "):
        print("  已取消。")
        return False

    TARGET.write_text(new_text, encoding="utf-8")
    if not _verify_or_rollback(new_text, read_ratio, round(target_ratio, 2), stamp):
        return False

    mp, rv = projected(target_ratio)
    print("\n  ✓ 已切档（A 主机 pinned）。")
    print(f"    新文件指纹 : sha256:{sha(TARGET)}")
    print(f"    MAX_PINNED : {mp / GIB:.2f} GiB  ->  上限 {rv / GIB:.2f} GiB")
    print()
    print("  生效方式：重启 ComfyUI。启动日志里应该看到")
    print(f"    Enabled pinned memory {int(mp // MIB)}")
    print(f"  （= 物理内存 x {target_ratio:.2f} 取整到 MiB；原版是 "
          f"{int(total_ram_bytes() * RATIO_ORIG // MIB)}）")
    return True


def switch_cast_to(target_gib: int, *, assume_yes: bool = False) -> bool:
    """把 B（设备侧 cast 预留）切到 target_gib。"""
    if not TARGET.exists():
        print("\n  !! 目标文件不存在。")
        return False

    if not (CAST_MIN <= target_gib <= CAST_MAX):
        print(f"\n  !! 目标预留 {target_gib} GiB 超出允许范围 "
              f"[{CAST_MIN}, {CAST_MAX}] GiB，拒绝。")
        return False

    text = read_target()
    ckind, cur = cast_state_of(text)
    if cur is None or ckind == "unknown":
        print("\n  !! B 的状态无法识别，拒绝动手。")
        if cur is not None:
            known = " / ".join(str(g) for g in
                              sorted([CAST_ORIG] + list(CAST_PROFILES)))
            print(f"     读到的是 {cur} GiB，不在已知档位 [{known}] 里。")
        print("     可能 ComfyUI 升级过、或这行被人改过。请手工核对：")
        print(f"     {TARGET}")
        return False

    if cur == target_gib:
        print(f"\n  已经是 {cast_label_of(ckind, cur)}，不需要切。")
        return False

    # 缺 .orig 时，只有"A 和 B 都处于原版"才允许建立原始备份
    akind, _ = state_of(text)
    both_orig = (cur == CAST_ORIG) and (akind == "orig")
    bk = _prepare_backup(both_orig, cast_label_of(ckind, cur), what="B")
    if bk is None:
        print("     补救：把 B 手工改回 `16 * 1024 ** 3`，再跑本工具。")
        return False
    _, _, stamp = bk

    new_line = f"DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = {target_gib} * 1024 ** 3"
    new_text, n = PAT_CAST_SUB.subn(new_line, text)
    if n != 1:
        print(f"\n  !! 替换命中 {n} 处（应为 1），拒绝动手。")
        return False

    tgt_name = CAST_PROFILES.get(target_gib, ("自定义档", ""))[0] if target_gib in CAST_PROFILES \
        else ("原版" if target_gib == CAST_ORIG else "自定义档")
    vtot = vram_total_bytes()
    print(f"\n  将把 B 的 `{cur} * 1024 ** 3` 切到 `{target_gib} * 1024 ** 3`（{tgt_name}）")
    if vtot:
        print(f"    新预留 {target_gib} GiB  vs  本机显存 {vtot / GIB:.2f} GiB")
    if not assume_yes and not ask("  继续？[y/N] "):
        print("  已取消。")
        return False

    TARGET.write_text(new_text, encoding="utf-8")
    if not _verify_or_rollback(new_text, read_cast_gib, target_gib, stamp):
        return False

    print("\n  ✓ 已切档（B 设备侧 cast 预留）。")
    print(f"    新文件指纹 : sha256:{sha(TARGET)}")
    print(f"    VRAMBuffer max_size : {target_gib} GiB")
    print()
    print("  生效方式：重启 ComfyUI（该值在首次权重转换时才 grow，与启动无关，")
    print("  但模块级常量要重启才重新求值）。若日志出现 `VRAM grow failed: N bytes`")
    print(f"  说明档位压得太小 —— 用 --cast-to {target_gib + 2} -y 往上调一档。")
    return True


def do_restore(*, assume_yes: bool = False, which: str = "both") -> bool:
    """从原始备份复原。which ∈ {'both','A','B'}。

    - 'both'：整文件逐字节还原（最彻底，A+B 一起回原版）
    - 'A' / 'B'：只把对应那一个数值切回原版，另一个保持不动
      （这时用"改回原版值"的方式，而不是整文件覆盖 —— 否则会把另一个档位一起冲掉）
    """
    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    if not orig.exists():
        print(f"\n  !! 找不到原始备份 {orig}")
        print("     没打过补丁、或备份被删了 —— 无法复原。")
        return False

    text = read_target()

    # ---- 只复原 A 或只复原 B ----
    if which in ("A", "B"):
        if which == "A":
            kind, cur = state_of(text)
            if kind == "orig":
                print("\n  A 已经是原版（ram * 0.40），不需要复原。")
                return False
            if kind not in ("profile", "custom"):
                print("\n  !! A 状态无法识别，拒绝动手。请手工核对。")
                return False
            print(f"\n  将把 A 从 {label_of(kind, cur)} 改回原版 "
                  f"`ram * {RATIO_ORIG:.2f}`")
            new_text, n = PAT_SUB.subn(
                lambda m: m.group(1) + f"MAX_PINNED_MEMORY = ram * {RATIO_ORIG:.2f}", text)
            expect_checker, expect_val = read_ratio, RATIO_ORIG
            tgt_name = "A（主机 pinned）"
        else:
            ckind, ccur = cast_state_of(text)
            if ckind == "orig":
                print("\n  B 已经是原版（16 GiB），不需要复原。")
                return False
            if ckind not in ("profile", "custom"):
                print("\n  !! B 状态无法识别，拒绝动手。请手工核对。")
                return False
            print(f"\n  将把 B 从 {cast_label_of(ckind, ccur)} 改回原版 `{CAST_ORIG} GiB`")
            new_text, n = PAT_CAST_SUB.subn(
                f"DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = {CAST_ORIG} * 1024 ** 3", text)
            expect_checker, expect_val = read_cast_gib, CAST_ORIG
            tgt_name = "B（设备侧 cast 预留）"

        if n != 1:
            print(f"\n  !! 替换命中 {n} 处（应为 1），拒绝动手。")
            return False
        if not assume_yes and not ask(f"  确认复原 {tgt_name}？[y/N] "):
            print("  已取消。")
            return False

        stamp = bdir / f"model_management.py.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(TARGET, stamp)
        TARGET.write_text(new_text, encoding="utf-8")
        if not _verify_or_rollback(new_text, expect_checker, expect_val, stamp):
            return False
        print(f"\n  ✓ 已复原 {tgt_name}，文件指纹 sha256:{sha(TARGET)}")
        print("    重启 ComfyUI 后生效。")
        return True

    # ---- 两个一起：整文件逐字节还原 ----
    akind, acur = state_of(text)
    ckind, ccur = cast_state_of(text)
    if akind == "orig" and ckind == "orig":
        print("\n  当前 A 和 B 都已经是原版，不需要复原。")
        return False
    if acur is None or ccur is None:
        print("\n  !! 状态无法识别，拒绝直接覆盖。")
        print("     当前文件有值既不是原版也不是已知档位（可能升级过）。")
        print(f"     原始备份在 : {orig}")
        print("     如确认要用备份覆盖，请手工执行拷贝。")
        return False

    print("\n  将用原始备份**整文件覆盖**当前文件：")
    print(f"    当前 A : {label_of(akind, acur)}")
    print(f"    当前 B : {cast_label_of(ckind, ccur)}")
    print(f"    备份   : sha256:{sha(orig)}")
    print(f"    当前   : sha256:{sha(TARGET)}")
    if not assume_yes and not ask("  确认复原（A+B 都回原版）？[y/N] "):
        print("  已取消。")
        return False

    shutil.copy2(orig, TARGET)
    if state_of(read_target())[0] != "orig" or sha(TARGET) != sha(orig):
        print("  !! 复原后校验失败，请手工检查。")
        return False
    print(f"\n  ✓ 已复原为原版，文件指纹 sha256:{sha(TARGET)}")
    print(f"    重启 ComfyUI 后，启动日志应恢复 "
          f"Enabled pinned memory {int(total_ram_bytes() * RATIO_ORIG // MIB)}")
    return True


# ---- bat 读写（GBK 字节级）---------------------------------------------------------
def bat_find_line(raw: bytes):
    for ln in raw.split(b"\r\n"):
        if ln.startswith(BAT_KEY):
            return ln
    return None


def bat_split(raw: bytes):
    """拆出 (基础部分bytes, 附加参数字符串)。基础部分不是预期内容 -> (None, None)。"""
    ln = bat_find_line(raw)
    if ln is None:
        return None, None
    body = ln[len(BAT_KEY):]
    base_b = BAT_BASE.encode("gbk")
    if not body.startswith(base_b):
        return None, None
    return base_b, body[len(base_b):].decode("gbk", "replace").strip()


def bat_current_extra():
    """读出当前 bat 的附加参数；读不到返回 None。"""
    if not BAT.exists():
        return None
    _, extra = bat_split(BAT.read_bytes())
    return extra


def bat_set_extra(extra: str, *, assume_yes: bool = False) -> bool:
    """把 bat 的附加参数设为 extra（空串 = 无附加）。"""
    if not BAT.exists():
        print(f"\n  !! 找不到启动脚本：{BAT}")
        return False

    raw = BAT.read_bytes()
    base_b, cur = bat_split(raw)
    if base_b is None:
        print(f"\n  !! {BAT.name} 的 `{BAT_KEY.decode('gbk')}` 行不是预期结构，拒绝自动改。")
        print("     （可能基础参数被手改过）请手工核对这一行。")
        return False

    extra = extra.strip()
    if cur == extra:
        print(f"\n  bat 附加参数已经是 `{extra or '（无）'}`，无需改。")
        return True

    # 备份：.orig 只在不存在时建立（绝不覆盖）
    bak_dir = BACKUP_DIRS[-1] if BACKUP_DIRS else BAT.parent
    bak_dir.mkdir(parents=True, exist_ok=True)
    orig = bak_dir / "启动ComfyUI.bat.orig"
    if not orig.exists():
        shutil.copy2(BAT, orig)
        print(f"  已建立原始备份 : {orig}")

    new_body = base_b + ((" " + extra) if extra else "").encode("gbk")
    parts = []
    for ln in raw.split(b"\r\n"):
        if ln.startswith(BAT_KEY):
            ln = BAT_KEY + new_body
        parts.append(ln)
    new_raw = b"\r\n".join(parts)

    # 字节级安全校验
    try:
        new_raw.decode("gbk")
    except UnicodeDecodeError:
        print("  !! 新内容 GBK 解码失败，拒绝写入。")
        return False
    if new_raw.count(b"\r\n") != raw.count(b"\r\n"):
        print("  !! 行数变了，拒绝写入。")
        return False
    if new_raw.count(b"\xb9\xb2\xcf\xed") < raw.count(b"\xb9\xb2\xcf\xed"):   # "共享"
        print("  !! 中文字节受损，拒绝写入。")
        return False

    print(f"\n  将把 bat 附加参数 `{cur or '（无）'}` 改为 `{extra or '（无）'}`")
    if not assume_yes and not ask("  继续？[y/N] "):
        print("  已取消。")
        return False

    BAT.write_bytes(new_raw)
    _, after = bat_split(BAT.read_bytes())
    if after != extra:
        print("  !! 写入后校验失败。")
        return False
    print(f"  ✓ bat 已更新（{len(new_raw)} 字节，GBK 完好）：")
    print(f"      {bat_find_line(new_raw).decode('gbk')}")
    return True


# ---- 预设 ------------------------------------------------------------------------
def apply_preset(key: str, *, assume_yes: bool = False) -> bool:
    """把 A / B / bat 附加参数一次切到同名预设。"""
    p = PRESET_BY_KEY.get(key)
    if p is None:
        print(f"\n  !! 未知预设 {key!r}。可用：{' / '.join(PRESET_BY_KEY)}")
        return False
    _, name, a, b, extra, note = p

    print(f"\n  ── 应用预设「{name}」──")
    print(f"     A = ram * {a:.2f}   B = {b} GiB   bat 附加 = `{extra or '（无）'}`")
    if not assume_yes and not ask("  继续？[y/N] "):
        print("  已取消。")
        return False

    print("\n  [1/3] A 主机 pinned …")
    switch_to(a, assume_yes=True)
    print("\n  [2/3] B 设备 cast 预留 …")
    if b == CAST_ORIG:
        do_restore(assume_yes=True, which="B")
    else:
        switch_cast_to(b, assume_yes=True)
    print("\n  [3/3] 启动脚本附加参数 …")
    bat_set_extra(extra, assume_yes=True)

    print(f"\n  ✓ 预设「{name}」已应用。重启 ComfyUI 后生效。")
    if extra:
        print(f"     （注意：改动 model_management.py 与 bat 都需要重启）")
    return True


# ---- 入口 ------------------------------------------------------------------------
def parse_gib(s: str):
    """把 '4' / '4g' / '4G' / '4gi' 解析成整数 GiB。失败返回 None。"""
    if s is None:
        return None
    t = str(s).strip().lower().rstrip("ib").rstrip("g").strip()
    try:
        return int(float(t))
    except ValueError:
        return None


def parse_args(argv):
    opts = {"target_file": None, "backup_dir": None, "to": None, "cast_to": None,
            "preset": None, "restore": False, "restore_a": False, "restore_b": False,
            "status": False, "yes": False, "bad": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-y", "--yes"):
            opts["yes"] = True
        elif a == "--status":
            opts["status"] = True
        elif a == "--restore":
            opts["restore"] = True
        elif a == "--restore-a":
            opts["restore_a"] = True
        elif a == "--restore-b":
            opts["restore_b"] = True
        elif a.startswith("--preset="):
            opts["preset"] = a.split("=", 1)[1]
        elif a == "--preset":
            i += 1
            opts["preset"] = argv[i] if i < len(argv) else None
        elif a.startswith("--cast-to="):
            opts["cast_to"] = a.split("=", 1)[1]
        elif a == "--cast-to":
            i += 1
            opts["cast_to"] = argv[i] if i < len(argv) else None
        elif a.startswith("--to="):
            opts["to"] = a.split("=", 1)[1]
        elif a == "--to":
            i += 1
            opts["to"] = argv[i] if i < len(argv) else None
        elif a.startswith("--backup-dir="):
            opts["backup_dir"] = a.split("=", 1)[1]
        elif a == "--backup-dir":
            i += 1
            opts["backup_dir"] = argv[i] if i < len(argv) else None
        elif a.startswith("-"):
            print(f"  !! 未知参数 {a}")
            opts["bad"] = True
        else:
            opts["target_file"] = a       # 兼容旧用法：位置参数 = 目标路径
        i += 1
    return opts


def apply_opts(opts) -> None:
    global TARGET, BACKUP_DIRS_PINNED
    if opts.get("target_file"):
        TARGET = Path(opts["target_file"]).resolve()
    if opts.get("backup_dir"):
        d = Path(opts["backup_dir"]).resolve()
        d.mkdir(parents=True, exist_ok=True)
        BACKUP_DIRS.insert(0, d)
        # 显式指定了备份目录 = 用户要隔离，就**不要**再回退去找别的目录的 .orig，
        # 否则测试/试跑会意外命中真实备份，污染真实环境（2026-09-25 实测踩过）。
        BACKUP_DIRS_PINNED = True


def main() -> int:
    opts = parse_args(sys.argv[1:])
    apply_opts(opts)

    # ---- 非交互模式 ----
    if opts["status"]:
        return 0 if show_status() != "missing" else 1

    if opts["restore"]:
        show_status()
        return 0 if do_restore(assume_yes=opts["yes"], which="both") else 1
    if opts["restore_a"]:
        show_status()
        return 0 if do_restore(assume_yes=opts["yes"], which="A") else 1
    if opts["restore_b"]:
        show_status()
        return 0 if do_restore(assume_yes=opts["yes"], which="B") else 1

    if opts["to"] is not None:
        try:
            ratio = round(float(opts["to"]), 2)
        except ValueError:
            print(f"  !! --to 参数不是数字：{opts['to']!r}")
            return 1
        show_status()
        return 0 if switch_to(ratio, assume_yes=opts["yes"]) else 1

    if opts["cast_to"] is not None:
        gib = parse_gib(opts["cast_to"])
        if gib is None:
            print(f"  !! --cast-to 参数无法解析成整数 GiB：{opts['cast_to']!r}")
            print("     例：--cast-to 4  /  --cast-to 4g")
            return 1
        show_status()
        return 0 if switch_cast_to(gib, assume_yes=opts["yes"]) else 1

    if opts["preset"] is not None:
        show_status()
        return 0 if apply_preset(opts["preset"], assume_yes=opts["yes"]) else 1

    # ---- 交互菜单 ----
    while True:
        kind = show_status()
        if kind == "missing":
            return 1
        print("  ── 配置预设（一键切 A + B + 启动参数，推荐）─────────────────")
        print("    1) 官方默认            A=0.40  B=16GiB  无附加参数")
        print("    2) 0.45 默认           A=0.45  B=16GiB  无附加参数")
        print("    3) 新设置（稳定跑通）★ A=0.45  B=16GiB  --disable-async-offload")
        print("  ── 单项微调 ───────────────────────────────────────────────")
        print("    4) A 主机 pinned 单独切（0.45 标准 / 0.50）")
        print("    5) B 设备 cast 预留单独切（4 GiB 保守 / 2 GiB 极小）")
        print("  ── 单独复原 ───────────────────────────────────────────────")
        print("    6) 只复原 A -> ram*0.40        7) 只复原 B -> 16 GiB")
        print("    0) 退出")
        try:
            choice = input("\n  选择> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            apply_preset("official")
        elif choice == "2":
            apply_preset("std045")
        elif choice == "3":
            apply_preset("stable")
        elif choice == "4":
            print()
            print("      a) 标准档        ram*0.45")
            print("      b) 缓存一采档 ★  ram*0.50   （注意：实测会让 TE 阶段崩，非必要不用）")
            c = input("      选择> ").strip().lower()
            if c == "a":
                switch_to(RATIO_STD)
            elif c == "b":
                switch_to(RATIO_CACHE)
            else:
                print("\n  未改动。")
        elif choice == "5":
            print()
            print("      a) 保守档  4 GiB")
            print("      b) 极小档  2 GiB")
            c = input("      选择> ").strip().lower()
            if c == "a":
                switch_cast_to(4)
            elif c == "b":
                switch_cast_to(2)
            else:
                print("\n  未改动。")
        elif choice == "6":
            do_restore(which="A")
        elif choice == "7":
            do_restore(which="B")
        elif choice in ("0", "q", "quit", "exit"):
            return 0
        else:
            print("\n  请输入 0~7")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
