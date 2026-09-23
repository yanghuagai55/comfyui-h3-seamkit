# -*- coding: utf-8 -*-
"""ComfyUI 注意力后端补丁：强制 SDPA 走 MATH（A/B 可复现用）。

【解决什么问题】
ComfyUI 默认按 FLASH → CUDNN → EFFICIENT → MATH 的优先级挑 SDPA 后端
（`comfy/ops.py:68-73`）。前三者都不保证逐位复现，于是「同 seed 两次跑」结果不同，
A/B 实验的方差盖过要测的效应（本机实测：二采本底差 2.8~4.5 灰阶）。

只改 `SDPA_BACKEND_PRIORITY` 那张表**不够** —— `ops.py:76-77` 有个捷径：
小输入（< 128K 元素）直接调 `F.scaled_dot_product_attention`，**绕过整个 sdpa_kernel 上下文**。
所以本工具改为**全局关掉三个非 MATH 后端**，一个插桩点覆盖两条路径。

【配套（本工具会检查并提示）】
  ① 启动脚本必须去掉 `--use-sage-attention` —— 这是**前提**：
     SageAttention 是自定义 CUDA 扩展（`comfy/ldm/modules/attention.py:679` 直调 `sageattn`），
     不经过 PyTorch 算子注册，`--deterministic` 对它**完全无效、连警告都没有**。
  ② 启动脚本加 `--deterministic`：它做 `use_deterministic_algorithms(True, warn_only=True)`
     并**自动设** `CUBLAS_WORKSPACE_CONFIG=:4096:8`（不必手设）。

【代价】
  MATH 后端比 Sage 慢数倍 —— **只在实验会话用**，日常出片请选 2 复原。

用法：
    <ComfyUI>/comfyenv/python.exe <本仓库>/tools/attention_backend_patch.py
"""
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

SCRIPT = Path(__file__).resolve()
PLUGIN_DIR = SCRIPT.parents[1]
COMFY_ROOT = SCRIPT.parents[3]
HOME_DIR = COMFY_ROOT.parent

TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else COMFY_ROOT / "comfy" / "ops.py"
LAUNCHER = HOME_DIR / "启动ComfyUI.bat"
EXPERIMENT_LAUNCHER = HOME_DIR / "启动ComfyUI-确定性.bat"

BACKUP_DIRS = [
    COMFY_ROOT / "user" / "seamkit_patches",
    HOME_DIR / "_hardcut_work" / "patches",
]
ORIG_NAME = "ops.py.orig"

# 插桩点：在 `from torch.nn.attention import ...` 之后、priority 表之前
ANCHOR = "        from torch.nn.attention import SDPBackend, sdpa_kernel\n"
MARKER = "[seamkit patch] 强制 SDPA 走 MATH"
BLOCK = (
    "\n"
    "        # ─── " + MARKER + "（A/B 可复现用，见 tools/attention_backend_patch.py）───\n"
    "        # 关掉三个非 MATH 后端后：既覆盖下面那张优先级表，\n"
    "        # 也覆盖「小输入直接调 F.scaled_dot_product_attention」的捷径。\n"
    "        # 代价是慢数倍 —— 日常出片请用该工具选 2 复原。\n"
    "        torch.backends.cuda.enable_flash_sdp(False)\n"
    "        torch.backends.cuda.enable_mem_efficient_sdp(False)\n"
    "        if hasattr(torch.backends.cuda, \"enable_cudnn_sdp\"):\n"
    "            torch.backends.cuda.enable_cudnn_sdp(False)\n"
    "        # ─────────────────────────────────────────────────────────────\n"
)


def backup_dir() -> Path:
    for d in BACKUP_DIRS:
        if (d / ORIG_NAME).exists():
            return d
    return BACKUP_DIRS[0]


def sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def state_of(text: str) -> str:
    return "patched" if MARKER in text else "unpatched"


def read_target() -> str:
    return TARGET.read_text(encoding="utf-8")


def ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "1", "是", "确定")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def launcher_report():
    """（状态, 有没有 sage, 有没有 deterministic）"""
    if not LAUNCHER.exists():
        return "missing", None, None
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    return "ok", ("--use-sage-attention" in text), ("--deterministic" in text)


def show_status() -> str:
    bdir = backup_dir()
    orig = bdir / ORIG_NAME
    print("=" * 68)
    print(" ComfyUI 注意力后端补丁（A/B 逐位复现用）")
    print("=" * 68)
    print(f"  目标文件 : {TARGET}")
    if not TARGET.exists():
        print("  !! 文件不存在。若 ComfyUI 不在默认位置，把路径当参数传进来。")
        return "missing"
    st = state_of(read_target())
    print(f"  当前状态 : " + ("已打补丁（强制 MATH）" if st == "patched" else "原始（按 FLASH→CUDNN→EFF→MATH 优先级）"))
    print(f"  文件指纹 : sha256:{sha(TARGET)}")
    print(f"  备份目录 : {bdir}")
    if orig.exists():
        print(f"  原始备份 : {ORIG_NAME}  sha256:{sha(orig)}")
    else:
        print("  原始备份 : （还没有，打补丁时会自动建立）")
    print()
    print("  ── 阶段 0 检查表（启动脚本）──")
    tag, sage, det = launcher_report()
    if tag == "missing":
        print(f"    !! 找不到 {LAUNCHER}")
    else:
        print(f"    {'✗ 还开着' if sage else '✓ 已关闭'}  --use-sage-attention"
              "     （必须关：--deterministic 看不见自定义 CUDA 扩展）")
        print(f"    {'✓ 已加' if det else '✗ 未加'}  --deterministic"
              "          （会自动设 CUBLAS_WORKSPACE_CONFIG，不必手设）")
        if sage or not det:
            print(f"    → 选 3 可生成实验用启动脚本（不改你原来那个）: {EXPERIMENT_LAUNCHER.name}")
    print("=" * 68)
    return st


def do_patch() -> bool:
    if not TARGET.exists():
        print("\n  !! 目标文件不存在。")
        return False
    text = read_target()
    if state_of(text) == "patched":
        print("\n  已经打过补丁，不需要重复打。")
        return False
    if text.count(ANCHOR) != 1:
        print(f"\n  !! 锚点命中 {text.count(ANCHOR)} 处（应为 1），拒绝动手。")
        print("     可能 ComfyUI 版本不同 —— 请人工核对 ops.py 里")
        print("     `from torch.nn.attention import SDPBackend, sdpa_kernel` 这一行。")
        return False

    bdir = backup_dir()
    orig = bdir / ORIG_NAME
    bdir.mkdir(parents=True, exist_ok=True)
    if not orig.exists():
        shutil.copy2(TARGET, orig)
        print(f"\n  已建立原始备份 : {orig}")
    stamp = bdir / f"ops.py.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(TARGET, stamp)
    print(f"  已建立本次备份 : {stamp.name}")

    if not ask("  将插入「关闭 flash / mem-efficient / cudnn SDP」的代码块，继续？[y/N] "):
        print("  已取消。")
        return False

    new_text = text.replace(ANCHOR, ANCHOR + BLOCK, 1)
    TARGET.write_text(new_text, encoding="utf-8")

    if state_of(read_target()) != "patched":
        print("  !! 写入后校验失败，正在回滚 ...")
        shutil.copy2(stamp, TARGET)
        print(f"  已回滚，文件指纹 sha256:{sha(TARGET)}")
        return False

    print("\n  ✓ 补丁已写入。")
    print(f"    新文件指纹 : sha256:{sha(TARGET)}")
    print()
    print("  生效方式：用【实验启动脚本】重启 ComfyUI（去掉 --use-sage-attention、加 --deterministic）。")
    print("  验证：同 seed 跑两次（**中间重启**，否则会被节点缓存掩盖），")
    print("        比 `cut planned=… -> boundary=… measured=…` 那几行是否逐字相同。")
    print("  提示：用短片验即可（total_seconds=5 / target_segment_seconds≈1.2），省时间也省 I/O。")
    return True


def do_restore() -> bool:
    bdir = backup_dir()
    orig = bdir / ORIG_NAME
    if not orig.exists():
        print(f"\n  !! 找不到原始备份 {orig}")
        print("     没打过补丁、或备份被删了 —— 无法复原。")
        return False
    if state_of(read_target()) == "unpatched":
        print("\n  当前已经是原始版，不需要复原。")
        return False

    print("\n  将用原始备份覆盖当前文件：")
    print(f"    备份 : sha256:{sha(orig)}")
    print(f"    当前 : sha256:{sha(TARGET)}")
    if not ask("  确认复原？（复原后速度恢复，但 A/B 会重新带方差）[y/N] "):
        print("  已取消。")
        return False

    shutil.copy2(orig, TARGET)
    if state_of(read_target()) != "unpatched" or sha(TARGET) != sha(orig):
        print("  !! 复原后校验失败，请手工检查。")
        return False
    print(f"\n  ✓ 已复原为原始版，文件指纹 sha256:{sha(TARGET)}")
    print("    重启 ComfyUI 后即恢复默认后端优先级。")
    return True


def do_write_launcher() -> bool:
    """生成实验用启动脚本（永不改动原脚本）。

    ★ 全程走**字节级**操作，绝不解码改写整个文件 —— 中文 Windows 上的 .bat 通常是
    GBK/cp936 编码，若按 UTF-8 读+写会把 `--output-directory "D:\\共享"` 这类
    非 ASCII 参数毁成 `????`（本工具第一版就踩了这个坑）。只替换纯 ASCII 的
    `set COMMANDLINE_ARGS=` 行与 title 行，其余字节原样保留。
    """
    if not LAUNCHER.exists():
        print(f"\n  !! 找不到 {LAUNCHER}，无法据此生成。")
        return False
    raw = LAUNCHER.read_bytes()
    # 用 [^\r\n]* 而不是 .* —— 否则会把行尾的 \r 也吃进捕获组，
    # 替换后那一行就变成 LF 结尾，在 CRLF 的 .bat 里混行尾。
    m = re.search(rb"^set COMMANDLINE_ARGS=([^\r\n]*)", raw, re.M)
    if not m:
        print("\n  !! 原脚本里找不到 `set COMMANDLINE_ARGS=...`，拒绝改动。")
        return False

    def _show(b: bytes) -> str:
        for enc in ("gbk", "utf-8"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", "replace")

    old_args = m.group(1)
    new_args = old_args.replace(b"--use-sage-attention", b"").replace(b"  ", b" ").strip()
    if b"--deterministic" not in new_args:
        new_args = new_args + b" --deterministic"
    # ★ --cache-none：每次队列都重算所有节点 → 一采必然重跑，
    #   于是「同 seed 跑两次」不再依赖重启，也顺带不再囤缓存张量（省内存）。
    #   代价是每跑一次都要重算一采（约 10 分钟），但 A/B 本来就要跑两次。
    #   注意：它与 --cache-ram/--cache-classic/--cache-lru 互斥（同一 mutually_exclusive_group）。
    if b"--cache-none" not in new_args:
        new_args = new_args + b" --cache-none"

    out = raw[: m.start(1)] + new_args + raw[m.end(1) :]
    out = out.replace(
        b"title ComfyUI GPU + SageAttention (cu130)",
        b"title ComfyUI deterministic session (A/B only, no Sage)",
    )
    header = (
        b"rem === generated by seamkit tools/attention_backend_patch.py ===\r\n"
        b"rem A/B experiment sessions only: SageAttention off + --deterministic.\r\n"
        b"rem (--deterministic also sets CUBLAS_WORKSPACE_CONFIG; do not set it by hand)\r\n"
        b"rem Daily renders: use the original launcher -- Sage is several times faster.\r\n"
    )
    out = header + out

    print(f"\n  原 COMMANDLINE_ARGS : {_show(old_args)}")
    print(f"  新 COMMANDLINE_ARGS : {_show(new_args)}")
    print(f"  将写出 : {EXPERIMENT_LAUNCHER}")
    print("  （不覆盖、不改动你原来的启动脚本；除 COMMANDLINE_ARGS 与 title 外字节原样保留）")
    if not ask("  继续？[y/N] "):
        print("  已取消。")
        return False

    EXPERIMENT_LAUNCHER.write_bytes(out)
    # 自检：非 ASCII 参数必须没被破坏
    check = EXPERIMENT_LAUNCHER.read_bytes()
    if b"?" in re.search(rb"^set COMMANDLINE_ARGS=([^\r\n]*)", check, re.M).group(1) and b"?" not in old_args:
        print("  !! 自检失败：参数里出现了 '?'，可能编码被破坏。正在删除生成的文件 ...")
        EXPERIMENT_LAUNCHER.unlink(missing_ok=True)
        return False
    print(f"\n  ✓ 已写出 {EXPERIMENT_LAUNCHER.name}（{len(out)} 字节）")
    return True


def main() -> int:
    while True:
        if show_status() == "missing":
            return 1
        print("  1) 打补丁（ops.py 强制 SDPA 走 MATH，需重启 ComfyUI）")
        print("  2) 复原（从原始备份逐字节还原）")
        print("  3) 生成实验用启动脚本（不改动你原来的那个）")
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
        elif choice == "3":
            do_write_launcher()
        elif choice in ("0", "q", "quit", "exit"):
            return 0
        else:
            print("\n  请输入 1 / 2 / 3 / 0")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
