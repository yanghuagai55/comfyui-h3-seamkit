# -*- coding: utf-8 -*-
"""ComfyUI pinned-memory 上限补丁（交互式小终端）。

【解决什么问题】
aimdo 给模型权重用的 **页锁定主机内存** 上限是这么算的：

    comfy/model_management.py   MAX_PINNED_MEMORY = ram * 0.40             ← Windows 分支
                                pinned_hostbuf_size = min(size, MAX) * 2   ← 注意这个 x2
    comfy/model_patcher.py      HostBuffer(0, 64MB, pinned_hostbuf_size(model.model_size()))

即 **上限 ≈ 0.80 x 物理内存**。MiniMax H3 二采实测需要 ~25.6 GiB，
32GB 机器上上限只有 ~25.3 GiB，于是分段采样中途被拒：

    aimdo: src/hostbuf.c:46:ERROR:hostbuf_grow: requested ... beyond reserved host buffer ...

净差只有 15~235 MiB，所以表现很迷惑：**前面几段能跑完，后面某一段撞线**
（上限建模型时定死、请求大小随内容浮动 → 每段都是掷骰子，与段长无关）。

【本工具做什么】
只把 Windows 分支那一行的 `ram * 0.40` 改成 `ram * 0.45`：
上限提到 ~28.5 GiB（留 10% 物理内存给系统），需求 25.6 GiB 有 ~2.9 GiB 余量。

相比启动参数 `--high-ram`（它把上限**整个去掉**，还会连带 `cache_classic=True`）：
pinned 内存不能换页，32GB 机器跑 32.4GB 模型时去掉护栏会让失败模式
从"干净报错"变成"掉页 / 整机卡死"。**护栏留着。**

【安全设计】
  - 打补丁前完整备份；`.orig` 只在未打补丁状态下写入一次，**绝不覆盖**
  - 正则锚在 `if WINDOWS:` 上做单次替换，命中数 != 1 就拒绝（**不会误改 Linux 分支那行同样的 0.40**）
  - 状态机 unpatched / patched / unknown；unknown（例如 ComfyUI 升级过）**一律拒绝，不硬来**
  - 复原走逐字节拷贝，不靠"反向替换"

用法：
    <ComfyUI>/comfyenv/python.exe <本仓库>/tools/pinned_memory_patch.py
    ...python pinned_memory_patch.py [model_management.py 的路径]
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

TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
    else COMFY_ROOT / "comfy" / "model_management.py"

# 备份位置：优先用"已经有 .orig 的那个"，否则用第一个（新位置）。
# 新增候选时**往后面加**，别调换顺序 —— 换顺序会让已有的备份被"看不见"。
BACKUP_DIRS = [
    COMFY_ROOT / "user" / "pinned_memory_patch",          # 默认
    HOME_DIR / "_hardcut_work" / "patches",               # 兼容早期手工位置的备份
]
HOST_PORT = 8188

PAT_UNPATCHED = re.compile(
    r"(if WINDOWS:\n(?:[ \t]*#.*\n)*[ \t]*)MAX_PINNED_MEMORY = ram \* 0\.40\b")
PAT_PATCHED = re.compile(
    r"(if WINDOWS:\n(?:[ \t]*#.*\n)*[ \t]*)MAX_PINNED_MEMORY = ram \* 0\.45\b")
NEW_LINE = "MAX_PINNED_MEMORY = ram * 0.45"

RAM_RATIO_OLD = 0.40
RAM_RATIO_NEW = 0.45
GIB = 1024 ** 3


def backup_dir() -> Path:
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


def state_of(text: str) -> str:
    """'unpatched' | 'patched' | 'unknown'"""
    n_old = len(PAT_UNPATCHED.findall(text))
    n_new = len(PAT_PATCHED.findall(text))
    if n_old == 1 and n_new == 0:
        return "unpatched"
    if n_new == 1 and n_old == 0:
        return "patched"
    return "unknown"


def read_target() -> str:
    return TARGET.read_text(encoding="utf-8")


def projected(ratio: float) -> tuple:
    ram = total_ram_bytes()
    mp = ram * ratio
    return mp, mp * 2


def show_status() -> str:
    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    print("=" * 68)
    print(" ComfyUI pinned-memory 上限（aimdo host buffer）补丁")
    print("=" * 68)
    print(f"  目标文件 : {TARGET}")
    if not TARGET.exists():
        print("  !! 文件不存在。若 ComfyUI 不在默认位置，把路径当参数传进来：")
        print("     python pinned_memory_patch.py <路径>/comfy/model_management.py")
        return "missing"
    st = state_of(read_target())
    label = {"unpatched": "原始 (ram * 0.40)",
             "patched": "已打补丁 (ram * 0.45)",
             "unknown": "!! 无法识别（既不是 0.40 版也不是 0.45 版）"}[st]
    mp_o, rv_o = projected(RAM_RATIO_OLD)
    mp_n, rv_n = projected(RAM_RATIO_NEW)
    print(f"  当前状态 : {label}")
    print(f"  文件指纹 : sha256:{sha(TARGET)}")
    print(f"  备份目录 : {bdir}")
    if orig.exists():
        print(f"  原始备份 : {orig.name}  sha256:{sha(orig)}")
    else:
        print("  原始备份 : （还没有，打补丁时会自动建立）")
    print(f"  物理内存 : {total_ram_bytes() / GIB:.2f} GiB")
    print()
    print(f"    现在 : MAX_PINNED {mp_o / GIB:6.2f} GiB  ->  上限 {rv_o / GIB:6.2f} GiB")
    print(f"    改后 : MAX_PINNED {mp_n / GIB:6.2f} GiB  ->  上限 {rv_n / GIB:6.2f} GiB")
    print("    （H3 二采实测需求约 25.6 GiB；上限 = 2 x MAX_PINNED）")
    print()
    print("  ComfyUI 正在运行（8188 有响应）：改完必须重启才生效"
          if comfyui_running() else
          "  ComfyUI 未在运行（改动会在下次启动时生效）")
    print("=" * 68)
    return st


def ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "1", "是", "确定")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def do_patch() -> bool:
    if not TARGET.exists():
        print("\n  !! 目标文件不存在。")
        return False
    st = state_of(read_target())
    if st == "patched":
        print("\n  已经是补丁版（ram * 0.45），不需要重复打。")
        return False
    if st != "unpatched":
        print("\n  !! 状态无法识别，拒绝动手。")
        print("     可能 ComfyUI 升级过、或这行被人改过。请手工核对：")
        print(f"     {TARGET}")
        return False

    new_text, n = PAT_UNPATCHED.subn(lambda m: m.group(1) + NEW_LINE, read_target())
    if n != 1:
        print(f"\n  !! 替换命中 {n} 处（应为 1），拒绝动手。")
        return False

    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    bdir.mkdir(parents=True, exist_ok=True)
    if not orig.exists():
        shutil.copy2(TARGET, orig)
        print(f"\n  已建立原始备份 : {orig}")
    stamp = bdir / f"model_management.py.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(TARGET, stamp)
    print(f"  已建立本次备份 : {stamp.name}")

    if not ask("  将把 `ram * 0.40` 改为 `ram * 0.45`，继续？[y/N] "):
        print("  已取消。")
        return False

    TARGET.write_text(new_text, encoding="utf-8")
    if state_of(read_target()) != "patched":
        print("  !! 写入后状态校验失败，正在回滚 ...")
        shutil.copy2(stamp, TARGET)
        print(f"  已回滚，文件指纹 sha256:{sha(TARGET)}")
        return False

    _, rv = projected(RAM_RATIO_NEW)
    print("\n  ✓ 补丁已写入。")
    print(f"    新文件指纹 : sha256:{sha(TARGET)}")
    print(f"    新上限约   : {rv / GIB:.2f} GiB")
    print()
    print("  生效方式：重启 ComfyUI。启动日志里应该看到")
    print("    Enabled pinned memory 14573.0      <- 原来是 12954.0")
    print("  （数字随物理内存变化：= 物理内存 x 0.45 取整到 MiB）")
    print("  然后重跑即可；本工具选 2 可随时复原。")
    return True


def do_restore() -> bool:
    bdir = backup_dir()
    orig = bdir / "model_management.py.orig"
    if not orig.exists():
        print(f"\n  !! 找不到原始备份 {orig}")
        print("     没打过补丁、或备份被删了 —— 无法复原。")
        return False

    st = state_of(read_target())
    if st == "unpatched":
        print("\n  当前已经是原始版（ram * 0.40），不需要复原。")
        return False
    if st != "patched":
        print("\n  !! 状态无法识别，拒绝直接覆盖。")
        print("     当前文件既不是原始版也不是补丁版（可能升级过）。")
        print(f"     原始备份在 : {orig}")
        print("     如确认要用备份覆盖，请手工执行拷贝。")
        return False

    print("\n  将用原始备份覆盖当前文件：")
    print(f"    备份 : sha256:{sha(orig)}")
    print(f"    当前 : sha256:{sha(TARGET)}")
    if not ask("  确认复原？[y/N] "):
        print("  已取消。")
        return False

    shutil.copy2(orig, TARGET)
    if state_of(read_target()) != "unpatched" or sha(TARGET) != sha(orig):
        print("  !! 复原后校验失败，请手工检查。")
        return False
    print(f"\n  ✓ 已复原为原始版，文件指纹 sha256:{sha(TARGET)}")
    print("    重启 ComfyUI 后，启动日志应恢复 Enabled pinned memory 12954.0")
    return True


def main() -> int:
    while True:
        if show_status() == "missing":
            return 1
        print("  1) 打补丁（ram*0.40 -> 0.45，需重启 ComfyUI）")
        print("  2) 复原（从原始备份逐字节还原）")
        print("  0) 退出")
        try:
            choice = input("\n  选择> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            do_patch()
        elif choice == "2":
            do_restore()
        elif choice in ("0", "q", "quit", "exit"):
            return 0
        else:
            print("\n  请输入 1 / 2 / 0")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
