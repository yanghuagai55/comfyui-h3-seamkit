#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""离线重放双拼装：同一批 window latent → VER-blend / VER-frozen（零采样）。

为什么这是一次**严格** A/B
--------------------------
拼装分岔发生在**每窗采样之后**，而唯一带 overlap 的缝在本片是 272 —— 它属于**末窗 W3**，
下游没有任何窗会读 W3 的拼装结果（W3 采样时读的锚来自 W2 自己的输出，与 W2 的拼装无关）。
⇒ blend 还是 frozen **只改变成片拼装**，两个版本共享同一批 window latent。
⇒ 同种子 / 同一采缓存 / 同计划 / **同一批 window latent** ——
   运行间非确定性（实测 W0 级 16.4%）被完全排除。

复刻来源（1:1，源码位置）
------------------------
`custom_nodes/comfyui-minimax-h3-audio-T8/h3_t8/chunked_two_pass_upscale_advanced.py`
  · `_crossfade`                       :196
  · `_append_video`                    :607   ← crossfade（blend 版）
  · `_append_video_guarded_overlap`    :656   ← frozen（锁住前窗像素）
seamkit 侧的调用与单位换算：`h3_upscale.py:1857-1884`（`_locked_tokens = 帧 * 5 // FRAME_GRID`）

用法
----
    python tools/replay_assemble.py <dump_dir>                 # 重放 + 自检 + 存 latent cache
    python tools/replay_assemble.py <dump_dir> --no-save       # 只重放 + 自检（不写盘）
    python tools/replay_assemble.py --check B1_20260928        # 拿历史 dump 验证本脚本

自检（强制）
------------
若 dump_dir 里有线上 `accumulated.pt`，脚本会分别与两版对比，报告
「哪一版能复现线上结果」+ 最大绝对差 —— 复现不了说明本脚本有 bug，**先修再往下用**。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

COMFY = r"D:\comfyui\ComfyUI"
SEAMKIT = os.path.join(COMFY, "custom_nodes", "comfyui-h3-seamkit")
DUMP_BASE = r"D:\comfyui\_hardcut_work\latent_dump"
FRAME_GRID = 17          # 一个 17 帧块
FRAMES_PER_TOKEN = 5     # 17 像素帧 = 5 latent token（1+4+4+4+4）


# --------------------------------------------------------------------------
# 1:1 复刻 upstream（纯张量，无运行时依赖）
# --------------------------------------------------------------------------
def _crossfade(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    count = left.shape[dim]
    weights = torch.linspace(0, 1, count, device=left.device, dtype=left.dtype)
    shape = [1] * left.ndim
    shape[dim] = count
    return left + (right - left) * weights.view(shape)


def _append_video(accumulated, chunk, start_token: int):
    """crossfade 拼接（blend 版线上路径）。"""
    if accumulated is None:
        return chunk
    total = max(accumulated.shape[2], start_token + chunk.shape[2])
    output = torch.zeros(
        (1, accumulated.shape[1], total, accumulated.shape[3], accumulated.shape[4]),
        device=accumulated.device, dtype=accumulated.dtype)
    output[:, :, : accumulated.shape[2]] = accumulated
    overlap = max(0, accumulated.shape[2] - start_token)
    overlap = min(overlap, chunk.shape[2])
    if overlap:
        output[:, :, start_token:start_token + overlap] = _crossfade(
            output[:, :, start_token:start_token + overlap].clone(),
            chunk[:, :, :overlap], 2)
    if chunk.shape[2] > overlap:
        output[:, :, start_token + overlap:start_token + chunk.shape[2]] = chunk[:, :, overlap:]
    return output


def _append_video_guarded_overlap(accumulated, chunk, start_token: int,
                                  locked_overlap_tokens: int):
    """frozen 拼接：前 `locked` 个 token 保持前窗原样，其余 transition 换成新采样。"""
    if accumulated is None:
        return chunk, 0, 0
    overlap = max(0, int(accumulated.shape[2]) - int(start_token))
    overlap = min(overlap, int(chunk.shape[2]))
    if start_token > accumulated.shape[2]:
        raise ValueError("temporal segment starts after the published video")
    locked = max(0, min(int(locked_overlap_tokens), overlap))
    transition = overlap - locked
    published = accumulated
    if transition:
        published = accumulated.clone()
        published[:, :, start_token + locked:start_token + overlap] = chunk[:, :, locked:overlap]
    if overlap >= chunk.shape[2]:
        return published, overlap, transition
    return torch.cat((published, chunk[:, :, overlap:]), dim=2), overlap, transition


# --------------------------------------------------------------------------
def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read_windows(dump_dir):
    """读 window_W{n}.pt（本窗采样原始输出 + meta）。"""
    wins = []
    for w in range(8):
        p = os.path.join(dump_dir, f"window_W{w}.pt")
        if not os.path.isfile(p):
            break
        payload = load(p)
        wins.append(payload)
    return wins


def overlap_frames(prev_meta, meta):
    """上一窗 end_frame 与本窗 start_frame 的差（像素帧）；硬切=0。"""
    if prev_meta is None:
        return 0
    return max(0, int(prev_meta["frames"][1]) - int(meta["frames"][0]))


def replay(wins, *, locked_frames: int, frozen: bool):
    """按窗顺序重放，返回 (accumulated, meta_list)。"""
    acc = None
    metas = []
    prev_meta = None
    for w, payload in enumerate(wins):
        chunk = payload["video"]
        meta = payload.get("meta", {}) or {}
        start_token = int(meta["tokens"][0])
        ov = overlap_frames(prev_meta, meta)
        rec = {"window": w, "start_token": start_token,
               "frames": meta.get("frames"), "overlap_frames": ov,
               "path": None, "locked_tokens": 0, "transition_tokens": 0}
        if acc is None:
            acc = chunk
            rec["path"] = "first-window"
        elif frozen and ov > 0:
            locked_tokens = max(0, min(int(locked_frames), ov)) * FRAMES_PER_TOKEN // FRAME_GRID
            acc, _ov, _tr = _append_video_guarded_overlap(acc, chunk, start_token, locked_tokens)
            rec.update(path="frozen", locked_tokens=locked_tokens, transition_tokens=_tr)
        else:
            acc = _append_video(acc, chunk, start_token)
            rec["path"] = "crossfade" if ov > 0 else "hard-cut(append)"
        metas.append(rec)
        prev_meta = meta
    return acc, metas


def compare(a, b):
    """返回 (max_abs_diff, common_frames)。"""
    t = min(a.shape[2], b.shape[2])
    if t == 0:
        return float("nan"), 0
    d = (a[:, :, :t].float() - b[:, :, :t].float()).abs().max().item()
    return d, t


# --------------------------------------------------------------------------
def load_save_helper():
    """装载 seamkit 的 save_av_latent/load_av_latent（脱离 ComfyUI 运行时也能用）。"""
    sys.path.insert(0, COMFY)
    spec = importlib.util.spec_from_file_location(
        "seamkit_lc", os.path.join(SEAMKIT, "nodes_latent_cache.py"),
        submodule_search_locations=[SEAMKIT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seamkit_lc"] = mod
    spec.loader.exec_module(mod)
    return mod


def nested(samples_video, samples_audio):
    from comfy.nested_tensor import NestedTensor
    return NestedTensor((samples_video, samples_audio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", help="含 window_W{n}.pt 的目录")
    ap.add_argument("--check", help="用 latent_dump/<名字> 验证本脚本")
    ap.add_argument("--locked-frames", type=int, default=17,
                    help="线上 locked_overlap（帧）；默认 17（=plan 的 ov_tokens）")
    ap.add_argument("--no-save", action="store_true", help="只重放，不写 latent cache")
    ap.add_argument("--prefix", default="REPLAY", help="latent cache 的 key 前缀")
    args = ap.parse_args()

    dump_dir = args.dump_dir
    if args.check:
        dump_dir = os.path.join(DUMP_BASE, args.check)
    if not dump_dir or not os.path.isdir(dump_dir):
        print("!! 需要有效的 dump 目录"); return 2

    print("=" * 78)
    print(f"dump_dir = {dump_dir}")
    wins = read_windows(dump_dir)
    if len(wins) < 2:
        print(f"!! 只找到 {len(wins)} 个 window_W*.pt，无法重放"); return 2
    for w, p in enumerate(wins):
        m = p.get("meta", {}) or {}
        v = p["video"]
        a = p.get("audio")
        print(f"  W{w}: frames {m.get('frames')} tokens {m.get('tokens')} "
              f"video {tuple(v.shape)} {v.dtype}"
              + (f" audio {tuple(a.shape)}" if a is not None else ""))

    acc_blend, meta_blend = replay(wins, locked_frames=args.locked_frames, frozen=False)
    acc_frozen, meta_frozen = replay(wins, locked_frames=args.locked_frames, frozen=True)

    print("-" * 78)
    print("逐窗拼装路径：")
    for a, b in zip(meta_blend, meta_frozen):
        print(f"  W{a['window']}: blend={a['path']:<18} frozen={b['path']:<18}"
              f" overlap {a['overlap_frames']}f"
              + (f"  locked {b['locked_tokens']}tok / transition {b['transition_tokens']}tok"
                 if b['path'] == 'frozen' else ""))

    online_pt = os.path.join(dump_dir, "accumulated.pt")
    if os.path.isfile(online_pt):
        online = load(online_pt)["video"]
        db, tb = compare(acc_blend, online)
        df, tf = compare(acc_frozen, online)
        print("-" * 78)
        print("★ 自检（对线上 accumulated.pt）：")
        print(f"   VER-blend   max|Δ| = {db:.6f}  （比 {tb} 帧）")
        print(f"   VER-frozen  max|Δ| = {df:.6f}  （比 {tf} 帧）")
        verdict = ("blend（线上是 crossfade）" if db < df else "frozen（线上是 frozen/locked）")
        print(f"   ⇒ 线上实际用的是：{verdict}"
              f"   [复现差 {'OK' if min(db, df) < 5e-2 else '!! 偏大，先查复刻逻辑'}]")
    else:
        print("-" * 78)
        print("（无线上 accumulated.pt，跳过自检）")

    if not args.no_save:
        print("-" * 78)
        try:
            lc = load_save_helper()
            for tag, acc, m in (("blend", acc_blend, meta_blend),
                                ("frozen", acc_frozen, meta_frozen)):
                key = f"{args.prefix}_{tag}"
                audio = wins[-1].get("audio")
                samples = nested(acc.to(torch.float16).contiguous(),
                                 (audio if audio is not None else
                                  torch.zeros((1, 32, 2, 1), dtype=torch.float16)).contiguous())
                out = lc.save_av_latent(samples, key, True, f"replay/{tag}",
                                        {"source": "replay_assemble", "variant": tag,
                                         "locked_frames": args.locked_frames,
                                         "dump_dir": dump_dir},
                                        path="")
                print(f"  {out}")
            print("  ⇒ T2 用 `MiniMaxH3AVLatentLoad` 填上面的 key 即可解码（零新增节点）")
        except Exception as exc:
            print(f"  !! 存 latent cache 失败：{exc}")
            print("     （重放本身已完成，可加 --no-save 只看数字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
