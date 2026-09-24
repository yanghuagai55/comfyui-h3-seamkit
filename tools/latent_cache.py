# -*- coding: utf-8 -*-
"""一采 latent 缓存的管理工具（ComfyUI 之外用）。

ComfyUI 里的 `MiniMax H3 AV Latent Cache (Save/Load)` 负责存读；
这个脚本负责**看和清**（对应 ComfyUI_MiniMaxH3_Director 里的
`inspect_first_pass_cache()` 与 `clear_segment_cache()` 两个能力的职能，
只是我们这边没有面板 UI，就用命令行）。

用法：
    python tools/latent_cache.py                       # 列出全部
    python tools/latent_cache.py show <key>            # 看一个的详情
    python tools/latent_cache.py verify <key>          # 读一遍，确认没坏
    python tools/latent_cache.py purge <key>           # 删一个
    python tools/latent_cache.py purge --all           # 全删（会先列出来要确认）
    python tools/latent_cache.py dir                   # 只打印缓存目录路径

所有子命令都接受 `--path <目录>`：节点里 `cache_path` 填了自定义目录时，
这里也用 `--path` 指到同一个目录，否则工具会看着默认目录说"没有缓存"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

SCRIPT = Path(__file__).resolve()
COMFY_ROOT = SCRIPT.parents[3]          # <repo>/tools/x.py -> tools -> <repo> -> custom_nodes -> <ComfyUI>
CACHE_DIRNAME = "seamkit_latent_cache"
_SAFE_KEY = re.compile(r"[^0-9A-Za-z._-]+")


def _launcher_output_dir() -> Path | None:
    """从启动脚本里读 --output-directory。

    本机把输出目录改成了 D:\\共享（启动脚本里的参数），此时 ComfyUI 的
    folder_paths 会指向那里而不是 <ComfyUI>/output —— 所以命令行工具必须
    也读这个参数，否则会看着一个空目录说"没有缓存"。

    .bat 是 GBK，所以全程按字节处理，不要解码改写整个文件。
    """
    for name in ("启动ComfyUI.bat", "启动ComfyUI-确定性.bat"):
        bat = COMFY_ROOT.parent / name
        if not bat.is_file():
            continue
        try:
            raw = bat.read_bytes()
        except OSError:
            continue
        m = re.search(rb"--output-directory\s+\"?([^\"\r\n]+)\"?", raw)
        if not m:
            continue
        for enc in ("gbk", "utf-8"):
            try:
                return Path(m.group(1).decode(enc).strip())
            except Exception:
                continue
    return None


def candidate_dirs() -> list:
    """节点可能写到哪几个地方（顺序 = 概率）。"""
    cands = []
    out = _launcher_output_dir()
    if out is not None:
        cands.append(out / CACHE_DIRNAME)
    cands.append(COMFY_ROOT / "output" / CACHE_DIRNAME)
    cands.append(COMFY_ROOT.parent / "共享" / CACHE_DIRNAME)
    return cands


def cache_dir() -> Path:
    """与节点一致：<输出目录>/seamkit_latent_cache。

    顺序：启动脚本里 --output-directory 指定的 -> <ComfyUI>/output -> D:\\共享。
    返回**第一个真实存在且有文件的**；都没有就返回启动脚本指定的那个
    （新机器也能报对路径）。用 `dir` 子命令可以看到全部候选。
    """
    cands = candidate_dirs()
    for c in cands:
        if c.is_dir() and any(c.glob("*.pt")):
            return c
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


def stem_of(key: str) -> str:
    safe = _SAFE_KEY.sub("_", (key or "").strip())[:48] or "unnamed"
    return f"{safe}.{hashlib.sha256((key or '').encode('utf-8')).hexdigest()[:12]}"


def read_meta(root: Path, stem: str) -> dict:
    p = root / f"{stem}.json"
    try:
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def entries(root: Path) -> list:
    if not root.is_dir():
        return []
    out = []
    for pt in sorted(root.glob("*.pt")):
        stem = pt.stem
        meta = read_meta(root, stem)
        out.append(
            {
                "stem": stem,
                "key": meta.get("key", "(无 meta —— 可能是旧格式或被截断)"),
                "fp": meta.get("fingerprint") or "<无>",
                "saved_at": meta.get("saved_at", "?"),
                "mb": pt.stat().st_size / 1e6,
                "shapes": meta.get("shapes"),
                "summary": meta.get("summary") or {},
                "pt": pt,
            }
        )
    return out


def cmd_list(root: Path, _args) -> int:
    rows = entries(root)
    print(f"缓存目录: {root}")
    if not rows:
        print("  （空 —— 还没存过，或目录不同）")
        return 0
    print(f"  {'key':<28} {'fp':<14} {'大小':>8}  {'存于':<20} 种子 / 节点数")
    for r in rows:
        s = r["summary"]
        print(
            f"  {str(r['key'])[:28]:<28} {str(r['fp'])[:12]:<14} "
            f"{r['mb']:7.1f}M  {str(r['saved_at']):<20} "
            f"{s.get('seeds')} / {s.get('nodes')}"
        )
    return 0


def cmd_show(root: Path, args) -> int:
    stem = stem_of(args.key)
    meta = read_meta(root, stem)
    pt = root / f"{stem}.pt"
    if not meta and not pt.is_file():
        print(f"找不到 key={args.key!r}（按文件名规则找 {stem}.pt）")
        return 1
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n文件: {pt}   存在={pt.is_file()}")
    return 0


def cmd_verify(root: Path, args) -> int:
    """读一遍并尝试还原 NestedTensor —— 确认文件没坏、形状自洽。"""
    stem = stem_of(args.key)
    pt = root / f"{stem}.pt"
    if not pt.is_file():
        print(f"找不到 {pt}")
        return 1
    try:
        import torch
    except Exception as exc:
        print(f"需要 torch 才能 verify（{exc}）—— 直接看 show 也行")
        return 1
    # verify 要真正还原 NestedTensor，所以得能 import comfy
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    payload = torch.load(pt, map_location="cpu")
    if not isinstance(payload, dict) or "tensors" not in payload:
        print("✗ 格式不对（缺 'tensors'）")
        return 1
    tensors = list(payload["tensors"])
    meta = payload.get("meta") or {}
    want = meta.get("shapes")
    ok = True
    for i, t in enumerate(tensors):
        got = list(t.shape)
        if isinstance(want, list) and i < len(want) and got != list(want[i]):
            print(f"✗ 第 {i} 个张量形状不符：缓存写 {want[i]}，读出 {got}")
            ok = False
    try:
        import comfy.nested_tensor  # noqa: F401

        if len(tensors) == 2:
            comfy.nested_tensor.NestedTensor(tuple(tensors))
            nested = "✓ NestedTensor 可还原"
        else:
            nested = f"（{len(tensors)} 个张量，非 AV 双流）"
    except Exception as exc:
        nested = f"✗ 无法还原 NestedTensor: {exc}"
        ok = False
    print(f"{'✓' if ok else '✗'} {pt.name}  {pt.stat().st_size/1e6:.1f} MB  "
          f"tensors={len(tensors)}  {nested}")
    print(f"  fp={str(meta.get('fingerprint'))[:12]}  saved_at={meta.get('saved_at')}")
    return 0 if ok else 1


def cmd_purge(root: Path, args) -> int:
    if args.all:
        rows = entries(root)
        if not rows:
            print("（本来就没有）")
            return 0
        print("将删除以下全部缓存：")
        for r in rows:
            print(f"  {r['key']}  ({r['mb']:.1f}M)  {r['pt'].name}")
        if not args.yes:
            ans = input("确认删除？[y/N] ").strip().lower()
            if ans not in ("y", "yes", "是"):
                print("已取消。")
                return 0
        for r in rows:
            r["pt"].unlink(missing_ok=True)
            (root / f"{r['stem']}.json").unlink(missing_ok=True)
        print(f"已删除 {len(rows)} 项。")
        return 0

    stem = stem_of(args.key)
    pt = root / f"{stem}.pt"
    if not pt.is_file():
        print(f"找不到 key={args.key!r}")
        return 1
    print(f"将删除 {pt.name}（{pt.stat().st_size/1e6:.1f} MB）")
    if not args.yes:
        ans = input("确认？[y/N] ").strip().lower()
        if ans not in ("y", "yes", "是"):
            print("已取消。")
            return 0
    pt.unlink(missing_ok=True)
    (root / f"{stem}.json").unlink(missing_ok=True)
    print("已删除。")
    return 0


def cmd_dir(root: Path, _args) -> int:
    out = _launcher_output_dir()
    print(f"用这个     : {root}")
    print(f"启动脚本指向: {out if out else '（没读到 --output-directory）'}")
    for c in candidate_dirs():
        n = len(list(c.glob('*.pt'))) if c.is_dir() else -1
        tag = "" if c == root else "   <- 另一个候选"
        print(f"  {'存在' if n >= 0 else '不存在'} {c}  ({max(n,0)} 个缓存){tag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="一采 latent 缓存管理")
    ap.add_argument(
        "--path",
        default=None,
        help="缓存目录覆盖（与节点里 cache_path 填的一致）。留空 = 自动探测默认目录。",
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="列出全部（默认）")
    p_show = sub.add_parser("show", help="看一个的详情")
    p_show.add_argument("key")
    p_ver = sub.add_parser("verify", help="读一遍确认没坏")
    p_ver.add_argument("key")
    p_purge = sub.add_parser("purge", help="删除")
    p_purge.add_argument("key", nargs="?")
    p_purge.add_argument("--all", action="store_true")
    p_purge.add_argument("--yes", action="store_true", help="跳过确认")
    sub.add_parser("dir", help="打印缓存目录")

    args = ap.parse_args()
    root = Path(args.path).expanduser() if args.path else cache_dir()
    if args.path:
        print(f"（--path 覆盖：{root}）")
    handlers = {
        None: cmd_list,
        "list": cmd_list,
        "show": cmd_show,
        "verify": cmd_verify,
        "purge": cmd_purge,
        "dir": cmd_dir,
    }
    if args.cmd == "purge" and not args.all and not args.key:
        print("purge 需要 <key> 或 --all")
        return 2
    return handlers.get(args.cmd, cmd_list)(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
