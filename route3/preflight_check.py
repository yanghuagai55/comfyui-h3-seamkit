# -*- coding: utf-8 -*-
"""Route 3 · 开机预检（云端跑，60 秒内出结论）

设计原则
--------
* **只读**：不改文件、不 pip install、不下载大文件。
* **不烧卡**：不加载模型权重、不跑前向，最多做一个 1024² 的 flex LSE 自检。
* **宁可误报**：任何 fatal 直接退出，让人在 10 分钟内决定停机，而不是跑 3 小时后才发现。

用法（云端）：
    python3 route3/preflight_check.py --work-root /content/h3_route3
    python3 route3/preflight_check.py --work-root /content/h3_route3 --check-net

退出码：0 = 全部通过（可能有 warn）；2 = 有 fatal，**立刻停机**。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

RESULT = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": [], "fatal": [], "warn": []}


def add(name, ok, detail="", fatal=True, extra=None):
    lvl = "PASS" if ok else ("FAIL" if fatal else "WARN")
    RESULT["checks"].append({"name": name, "status": lvl, "detail": detail,
                             "extra": extra or {}})
    print("  [%s] %s%s" % (lvl, name, ("  " + detail) if detail else ""))
    if not ok:
        (RESULT["fatal"] if fatal else RESULT["warn"]).append(name + (": " + detail if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
def check_env():
    import torch
    print("=== 1. 运行时 ===")
    add("torch", torch.__version__.startswith("2."), torch.__version__ + " cuda=" + str(torch.version.cuda))
    add("cuda 可用", torch.cuda.is_available(), "")
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    add("GPU 是 A100", "A100" in name, "%s / %.1f GB" % (name, total), fatal=False)
    add("显存 >= 75GB", total >= 75, "%.1f GB" % total)
    add("主机内存 >= 64GB", os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 >= 64,
        "%.0f GB" % (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9), fatal=False)


def check_flex():
    """LSE 通路 —— 整个 Route 3 的地基，不通就不要开机。"""
    import torch
    print("=== 2. LSE 通路 ===")
    try:
        from torch.nn.attention.flex_attention import flex_attention
        add("flex_attention 可导入", True)
    except Exception as e:
        add("flex_attention 可导入", False, repr(e))
        return
    torch.manual_seed(0)
    q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    try:
        fn = torch.compile(flex_attention, dynamic=False)
        o, lse = fn(q, q, q, return_lse=True)
        torch.cuda.synchronize()
    except Exception as e:
        add("flex + torch.compile + return_lse", False, repr(e)[:200])
        return
    import math
    s = (q.float() @ q.float().transpose(-1, -2)) / math.sqrt(128)
    err = (lse.float() - torch.logsumexp(s, -1)).abs().max().item()
    add("flex + torch.compile + return_lse", err < 1e-2,
        "首编 %.1fs / lse_err=%.2e" % (time.time() - t0, err),
        extra={"compile_s": round(time.time() - t0, 1), "lse_err": err})
    try:
        import flash_attn  # noqa: F401
        add("flash-attn（可选加速）", True, getattr(flash_attn, "__version__", "?"), fatal=False)
    except Exception:
        add("flash-attn（可选加速）", False, "未安装 —— 不影响，用 flex", fatal=False)


def check_ck():
    print("=== 3. comfy_kitchen ===")
    try:
        import comfy_kitchen as ck
    except Exception as e:
        add("comfy_kitchen 可导入", False, repr(e))
        return
    try:
        be = ck.list_backends()
        add("comfy_kitchen 后端", True, json.dumps(be)[:160])
    except Exception as e:
        add("comfy_kitchen 后端", False, repr(e), fatal=False)
    try:
        add("sol_attn 可用", bool(ck.sol_attn_is_available()),
            "启用 [33][34] 才需要", fatal=False)
    except Exception as e:
        add("sol_attn 可用", False, repr(e)[:80], fatal=False)
    # 关键：训练路径不能用融合 RoPE（无 autograd），这里只做提示
    add("训练需绕开融合 RoPE", True,
        "route3_block.rms_rope_split_half_ref 已提供纯 torch 版（实测差 9.5e-7）",
        fatal=False)


# --------------------------------------------------------------------------- #
def _model_roots():
    roots = []
    for r in ("/content/ComfyUI/models", "/content/local_models"):
        if os.path.isdir(r):
            roots.append(r)
    return roots


def _find(patterns):
    hits = []
    for root in _model_roots():
        for p in patterns:
            hits += glob.glob(os.path.join(root, "**", p), recursive=True)
    # 去软链重复
    seen, out = set(), []
    for h in hits:
        rp = os.path.realpath(h)
        if rp in seen or not os.path.isfile(h):
            continue
        seen.add(rp)
        out.append(h)
    return out


def _header(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
    return keys, meta


ROLES = {
    "扩散底模": (["**/dasiwa*.safetensors", "**/*minimax_h3*ref2va*.safetensors",
                 "**/*minimax_h3*fl2va*.safetensors"], 15e9),
    "文本编码器": (["**/qwen3vl*safetensors"], 10e9),
    "视频 VAE": (["**/minimax_h3_video_vae*.safetensors"], 1e9),
    "音频 VAE": (["**/minimax_h3_audio_vae*.safetensors"], 0.3e9),
    "latent 上采样": (["**/h3_upscaler*.safetensors"], 0.5e9),
}


def check_models(allow_mismatch=False):
    print("=== 4. 模型清单 ===")
    found = {}
    for role, (pats, min_bytes) in ROLES.items():
        hits = _find(pats)
        if not hits:
            add(role, False, "未找到（patterns=%s）" % pats[0])
            continue
        hits.sort(key=lambda p: os.path.getsize(p), reverse=True)
        path = hits[0]
        size = os.path.getsize(path)
        add(role, size >= min_bytes, "%s  %.2f GB" % (os.path.basename(path), size / 1e9),
            extra={"path": path, "size": size})
        found[role] = path

    dm = found.get("扩散底模")
    if not dm:
        return found
    try:
        keys, meta = _header(dm)
    except Exception as e:
        add("底模 header 可读", False, repr(e)[:160])
        return found

    ids = sorted({int(m.group(1)) for k in keys
                  if (m := re.search(r"(?:^|\.)blocks\.(\d+)\.", k))})
    info = {"num_layers": len(ids), "max_block_idx": ids[-1] if ids else -1}
    for k in keys:
        if k.endswith("attn.out_proj.weight"):
            info["inner"], info["hidden"] = None, None
            try:
                from safetensors import safe_open
                with safe_open(dm, framework="pt", device="cpu") as f:
                    sh = f.get_slice(k).get_shape()
                info["inner"], info["hidden"] = int(sh[0]), int(sh[1])
            except Exception:
                pass
            break
    for k in keys:
        if k.endswith("q_norm.weight"):
            from safetensors import safe_open
            with safe_open(dm, framework="pt", device="cpu") as f:
                info["head_dim"] = int(f.get_slice(k).get_shape()[0])
            break
    if info.get("inner") and info.get("head_dim"):
        info["heads"] = info["inner"] // info["head_dim"]

    # 任务书假设：50 层 / hidden 5376 / 56 头 x 128
    ok_layers = info["num_layers"] == 50
    ok_hidden = info.get("hidden") == 5376
    ok_heads = info.get("heads") == 56 and info.get("head_dim") == 128
    msg = "layers=%s hidden=%s heads=%s head_dim=%s" % (
        info["num_layers"], info.get("hidden"), info.get("heads"), info.get("head_dim"))
    add("底模结构符合任务书（50L/5376/56x128）", ok_layers and ok_hidden and ok_heads,
        msg, fatal=not allow_mismatch, extra=info)
    if not (ok_layers and ok_hidden and ok_heads):
        print("      !! 任务书里的「50 层 / 56 头」结论全部作废，Stage 0 探针必须按真实层数改")

    # 内嵌注意力配置
    attn_cfg = {}
    for k in keys:
        if k.endswith(".config") or k.endswith("attention.config"):
            try:
                from safetensors import safe_open
                with safe_open(dm, framework="pt", device="cpu") as f:
                    t = f.get_tensor(k)
                attn_cfg[k] = bytes(t.numpy().tobytes()).decode("utf-8", "replace")[:200]
            except Exception:
                pass
    RESULT["checks"][-1]["extra"]["attention_config"] = attn_cfg
    int8 = any("comfy_kitchen_int8" in v for v in attn_cfg.values())
    add("Oracle 与自研分支同一注意力口径", not int8,
        ("内嵌 comfy_kitchen_int8 —— 启动必须加 --use-pytorch-cross-attention" if int8
         else "未发现内嵌 int8 注意力配置"), fatal=False,
        extra={"attention_config": attn_cfg})
    return found


# --------------------------------------------------------------------------- #
def check_disk(work_root):
    print("=== 5. 磁盘与工作目录 ===")
    os.makedirs(work_root, exist_ok=True)
    probe = os.path.join(work_root, ".preflight_write")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        add("工作目录可写", True, work_root)
    except Exception as e:
        add("工作目录可写", False, repr(e))
    for p in ("/content", work_root):
        if os.path.isdir(p):
            free = shutil.disk_usage(p).free / 1e9
            add("磁盘余量 %s" % p, free >= 80, "%.0f GB 可用" % free)
    drv = "/content/drive/MyDrive/h3_assets"
    add("Drive 已挂载", os.path.isdir(drv), drv, fatal=False)
    if os.path.isdir(drv):
        add("Drive 可写", os.access(drv, os.W_OK), "", fatal=False)


def check_comfy():
    print("=== 6. ComfyUI ===")
    comfy = "/content/ComfyUI"
    add("ComfyUI 目录", os.path.isdir(os.path.join(comfy, "main.py")), comfy)
    try:
        out = subprocess.run("git -C %s log -1 --format=%%h" % comfy, shell=True,
                             capture_output=True, text=True, timeout=20).stdout.strip()
        add("ComfyUI 版本", bool(out), out, fatal=False)
    except Exception:
        pass
    try:
        out = subprocess.run("curl -s -o /dev/null -w '%{http_code}' --max-time 8 "
                             "http://127.0.0.1:8188/", shell=True,
                             capture_output=True, text=True, timeout=15).stdout.strip()
        add("8188 已就绪", out == "200", "HTTP " + out, fatal=False)
    except Exception as e:
        add("8188 已就绪", False, repr(e)[:80], fatal=False)
    # 启动参数里不能有 --highvram
    try:
        ps = subprocess.run("ps -eo args | grep '[m]ain.py'", shell=True,
                            capture_output=True, text=True, timeout=10).stdout
        add("启动参数无 --highvram", "--highvram" not in ps,
            ps.strip()[:160] or "未检测到进程", fatal=False)
    except Exception:
        pass


NET_ITEMS = {
    "RMBG-1.4（抠像权重 ~176MB）":
        "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth",
}


def check_net():
    """大文件只能云端自取 —— 先确认直链活着，别到 Stage 里才发现 404。"""
    print("=== 7. 云端自取直链 ===")
    for name, url in NET_ITEMS.items():
        try:
            out = subprocess.run(
                "curl -sIL --max-time 20 -o /dev/null -w '%{http_code} %{size_download} %{url_effective}' '%s'"
                % url, shell=True, capture_output=True, text=True, timeout=30).stdout.split()
            code = out[0] if out else "000"
            add(name, code in ("200", "302"), "HTTP %s" % code, fatal=False,
                extra={"url": url})
        except Exception as e:
            add(name, False, repr(e)[:80], fatal=False, extra={"url": url})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default="/content/h3_route3")
    ap.add_argument("--allow-layer-mismatch", action="store_true")
    ap.add_argument("--check-net", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print("Route3 preflight  →  %s\n" % a.work_root)
    try:
        if not a.skip_gpu:
            check_env()
            check_flex()
        check_ck()
        check_models(a.allow_layer_mismatch)
        check_disk(a.work_root)
        check_comfy()
        if a.check_net:
            check_net()
    except Exception as e:
        import traceback
        traceback.print_exc()
        RESULT["fatal"].append("preflight 自身异常: " + repr(e))

    RESULT["elapsed_s"] = round(time.time() - t0, 1)
    RESULT["ok"] = not RESULT["fatal"]
    out = os.path.join(a.work_root, "preflight_ok.json") if RESULT["ok"] else \
        os.path.join(a.work_root, "preflight_FAILED.json")
    os.makedirs(a.work_root, exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=2)
    os.replace(out + ".tmp", out)

    print("\n%s  用时 %.1fs  →  %s" % ("全部通过" if RESULT["ok"] else "有 FATAL，立刻停机",
                                      RESULT["elapsed_s"], out))
    if RESULT["warn"]:
        print("warn: " + " | ".join(RESULT["warn"]))
    if RESULT["fatal"]:
        print("fatal: " + " | ".join(RESULT["fatal"]))
    return 0 if RESULT["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
