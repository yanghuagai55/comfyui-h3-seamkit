# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""把一采的联合 AV latent 落盘 / 读回，用来冻结二采的输入。

为什么需要这两个节点
--------------------
分块二采的接缝好坏，必须在**同一份二采输入**上比较两次跑（A/B）。但本机实测
（2026-09-23）：一采**不是逐位可复现的** —— 同 seed 同 prompt，跨会话跑出来的
测得的转镜帧能差 9 帧。而"让采样可复现"这条路在 8GB 上走不通：

* 关 `--use-sage-attention` → 一采直接 CUDA OOM（Sage 是 int8/fp8 量化注意力，
  省的是显存，不是可选项）；
* 钉 MATH 后端 → 注意力矩阵 O(n²) 内存，更不可能。

所以唯一干净的 A/B 是**冻结一采**：把它落盘、让二采读盘。这样二采输入逐位相同，
而生成速度一点不损失（一采 latent 只有约 17 MB，107 token / 0.4MP 实测）。

怎么接
------
* 只**存档**（不改行为）：把 `AV Latent Save` 串在一采与二采之间即可 ——
  它是直通的（LATENT 进、LATENT 出），二采照样从一采拿数据。
* 要**跳过一采**做 A/B：把二采执行器的 `latent` 输入从「一采输出」改接
  `AV Latent Load`。必须换接线 —— ComfyUI 会先算完所有输入才调用下游，
  光加一个存节点是不会让一采停跑的。

⚠️ ComfyUI 核心的 `SaveLatent`/`LoadLatent` 做不了这件事：它存的是
`latent_tensor` 单张量、走 safetensors（`nodes.py:553/576`），读回来还是单张量，
不是 NestedTensor。本文件的存法是把嵌套张量拆成普通张量再存 —— 实测
`torch.load` 的默认严格模式（`weights_only=True`）也能读，且逐位一致。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch

import folder_paths
from comfy_api.latest import io


CACHE_DIRNAME = "seamkit_latent_cache"
_SAFE_KEY = re.compile(r"[^0-9A-Za-z._-]+")


def _cache_dir(create: bool) -> Path | None:
    try:
        root = Path(folder_paths.get_output_directory()) / CACHE_DIRNAME
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError:
        return None


def _file_stem(key: str) -> str:
    """key -> 安全的文件名。

    文件名 = 「安全化的可读前缀 + key 的哈希」，所以：
    * 不同 key 撞车的概率可忽略；
    * key 里带斜杠 / 中文 / 空格也不会写坏路径；
    * 哈希只取 key 本身 —— 换 key 就是换缓存，不会误用旧数据。
    """
    safe = _SAFE_KEY.sub("_", (key or "").strip())[:48] or "unnamed"
    digest = hashlib.sha256((key or "").encode("utf-8")).hexdigest()[:12]
    return f"{safe}.{digest}"


def _av_to_plain(samples) -> list:
    """NestedTensor -> 普通 CPU 张量列表（可被默认严格模式读回）。"""
    if hasattr(samples, "unbind"):
        return [p.detach().cpu().contiguous() for p in samples.unbind()]
    return [samples.detach().cpu().contiguous()]


def _plain_to_av(tensors: list):
    import comfy.nested_tensor

    if len(tensors) == 1:
        return comfy.nested_tensor.NestedTensor((tensors[0],))
    return comfy.nested_tensor.NestedTensor(tuple(tensors))


class MiniMaxH3AVLatentSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVLatentSave",
            display_name="MiniMax H3 AV Latent Cache (Save)",
            description=(
                "把联合 AV latent 落盘，用于冻结二采的输入（A/B 实验）。\n"
                "**直通节点**：LATENT 进、LATENT 出，串在一采与二采之间即可，不用改接线。\n"
                "同一 key 已存在时默认跳过（不重写 17 MB）。\n"
                "要真正跳过一采，把二采执行器的 latent 输入改接 AV Latent Cache (Load)。"
            ),
            category="MiniMax H3 Hard Cut",
            is_experimental=True,
            inputs=[
                io.Latent.Input("latent"),
                io.String.Input(
                    "key",
                    default="run1",
                    tooltip=(
                        "缓存标识。换 key = 换缓存；同一个 key 再跑就是复用同一份 latent。\n"
                        "建议写清来源，例如 `seed786_s5_t2.8_promptA`。\n"
                        "文件落在 <输出目录>/seamkit_latent_cache/。"
                    ),
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    tooltip=(
                        "关（默认）：指纹一致就跳过写入，省一次 17 MB —— 指纹不一致时**自动覆盖**并提示。\n"
                        "开：无条件重写。\n"
                        "指纹 = 上游整条一采链（提示词/种子/参考图/底模/LoRA/采样器）的哈希，\n"
                        "二采参数不进指纹 —— 所以改二采不会让一采缓存失效，改一采则必然刷新。"
                    ),
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, latent, key: str, overwrite: bool):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            raise ValueError("MiniMaxH3AVLatentSave: expected a LATENT dict with 'samples'")

        # cls.hidden 由 ComfyUI 运行时注入；用 getattr 链兜底，脱离运行时（单测）也不会炸
        _hidden = getattr(cls, "hidden", None)
        fp, summary = fingerprint_of(
            getattr(_hidden, "prompt", None), getattr(_hidden, "unique_id", None)
        )
        # 真正的存盘逻辑在 save_av_latent()：二采执行器也调它（它本来就在一采下游，
        # 手上就有那份 latent），所以两处必须共用一份实现，避免行为漂移。
        print(save_av_latent(samples, key, overwrite, fp, summary), flush=True)
        return io.NodeOutput(latent)


def _read_meta(meta_path) -> dict:
    try:
        if meta_path.is_file():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


class MiniMaxH3AVLatentLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVLatentLoad",
            display_name="MiniMax H3 AV Latent Cache (Load)",
            description=(
                "读回冻结的联合 AV latent，用来**跳过一采**做干净的 A/B。\n"
                "接到二采执行器的 latent 输入（替换原来那条一采连线）。\n"
                "找不到缓存直接报错 —— 不会静默回退到别的 latent（那会让 A/B 悄悄失效）。"
            ),
            category="MiniMax H3 Hard Cut",
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "key",
                    default="run1",
                    tooltip="与 Save 节点填的 key 一致。文件在 <输出目录>/seamkit_latent_cache/。",
                ),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, key: str):
        samples, meta, name = load_av_latent(key)
        fp = meta.get("fingerprint") or "<无>"
        summary = meta.get("summary") or {}
        print(
            f"[SeamKit] latent cache LOADED: {name}  "
            f"tensors={meta.get('tensor_count')}  "
            f"saved_at={meta.get('saved_at', '?')}  fp={str(fp)[:12]}  "
            f"seed={summary.get('seeds')}  nodes={summary.get('nodes')}",
            flush=True,
        )
        print(
            "           注意：本节点图上没有上游可走，**无法自动核对指纹** —— "
            "请把这里的 fp 与 Save 那次日志里的 fp 对一眼（前 12 位相同即可）。",
            flush=True,
        )
        return io.NodeOutput({"samples": samples})


def _check_shapes(tensors: list, meta: dict, name: str) -> None:
    want = meta.get("shapes")
    if not isinstance(want, list) or len(want) != len(tensors):
        return
    for i, (t, w) in enumerate(zip(tensors, want)):
        if list(t.shape) != list(w):
            raise ValueError(
                f"MiniMaxH3AVLatentLoad: {name} 第 {i} 个张量形状不符 —— "
                f"缓存是 {w}，读出来是 {list(t.shape)}。文件可能被截断或替换过。"
            )


def save_av_latent(samples, key: str, overwrite: bool, fp: str, summary: dict, *, tag: str = "") -> str:
    """把 AV latent 落盘。返回一行状态（供调用方打印）。

    抽成模块级函数，因为有两个调用方：
      * `MiniMaxH3AVLatentSave` 节点（独立存）
      * **二采执行器 `MiniMaxH3HardCutUpscale`** —— 它本来就在一采下游，
        手上就有那份 latent，所以由它存最省事：不用加节点、不用动接线。
    永不抛异常（缓存是尽力而为，不能影响生成）。
    """
    root = _cache_dir(create=True)
    if root is None:
        return "[SeamKit] latent cache dir unavailable; skip"
    stem = _file_stem(key)
    pt_path = root / f"{stem}.pt"
    meta_path = root / f"{stem}.json"
    prefix = f"[SeamKit]{(' ' + tag) if tag else ''}"

    old = _read_meta(meta_path)
    old_fp = old.get("fingerprint")
    if pt_path.is_file() and not overwrite:
        if fp and old_fp == fp:
            return f"{prefix} latent cache HIT (skip write): {pt_path.name}  fp={fp[:12]}"
        if not fp:
            return (
                f"{prefix} latent cache HIT by key (skip write): {pt_path.name}"
                "  ⚠ 未取到上游图，无法核对指纹；要刷新请开 overwrite"
            )
        diff = fingerprint_diff(old.get("summary") or {}, summary)
        hit = (
            f"{prefix} latent cache 指纹不一致 -> 覆盖 {pt_path.name}  "
            f"({old_fp[:12] if old_fp else '无'} -> {fp[:12]});  变化: " + "; ".join(diff)
        )
    else:
        hit = None

    tensors = _av_to_plain(samples)
    meta = {
        "key": key,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": fp,
        "summary": summary,
        "tensor_count": len(tensors),
        "shapes": [list(t.shape) for t in tensors],
        "dtypes": [str(t.dtype) for t in tensors],
        "nested": bool(getattr(samples, "is_nested", False)),
    }
    try:
        tmp = root / f".{stem}.pt.tmp"
        torch.save({"tensors": tensors, "meta": meta}, tmp)
        tmp.replace(pt_path)
        tmp_json = root / f".{stem}.json.tmp"
        tmp_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_json.replace(meta_path)
    except Exception as exc:  # 缓存失败绝不能影响生成
        return f"{prefix} latent cache write skipped ({exc})"

    size_mb = pt_path.stat().st_size / 1e6
    line = (
        f"{prefix} latent cache SAVED: {pt_path.name}  {size_mb:.1f} MB  fp={fp[:12]}  "
        f"shapes={meta['shapes']}  seed={summary.get('seeds')}  nodes={summary.get('nodes')}"
    )
    return f"{hit}\n{line}" if hit else line


def load_av_latent(key: str):
    """按 key 读回 AV latent（`{"samples": NestedTensor}`）。读不到就抛。"""
    root = _cache_dir(create=False)
    stem = _file_stem(key)
    pt_path = (root / f"{stem}.pt") if root else None
    if pt_path is None or not pt_path.is_file():
        names = []
        if root and root.is_dir():
            names = sorted(p.name for p in root.glob("*.pt"))[:8]
        raise ValueError(
            f"no cached AV latent for key={key!r} (looked for {pt_path}).\n"
            f"  现有缓存: {names or '（空）'}"
        )
    payload = torch.load(pt_path, map_location="cpu")  # 默认严格模式即可（普通张量）
    if not isinstance(payload, dict) or "tensors" not in payload:
        raise ValueError(f"{pt_path.name} 格式不对（缺 'tensors'）")
    tensors = list(payload["tensors"])
    meta = payload.get("meta") or {}
    _check_shapes(tensors, meta, pt_path.name)
    return _plain_to_av(tensors), meta, pt_path.name


NODES = [MiniMaxH3AVLatentSave, MiniMaxH3AVLatentLoad]


# --------------------------------------------------------------------------
# 指纹：照搬 ComfyUI_MiniMaxH3_Director 的 first_pass_cache_fingerprint 思路 ——
# 「身份 = 只影响一采的那些东西」，二采参数不进指纹，改二采不会让一采缓存失效。
#
# 区别在于它从自己的 plan 取字段（它自己就是执行器），而我们从**图**上取：
# Save 节点的上游恰好就是整条一采链（提示词/种子/参考图/底模/LoRA/采样器），
# 二采在它下游 —— 所以「沿 latent 链接反向走完上游子图再哈希」正好等于我们要的身份。
# 这样用户填错 key 也不会静默复用旧 latent。
# --------------------------------------------------------------------------

_SEED_KEYS = ("noise_seed", "seed", "seed_num")
_TEXT_KEYS = ("value", "text", "prompt", "string")


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))


def upstream_subgraph(prompt, node_unique_id) -> dict:
    """从本节点的 `latent` 输入出发，反向收集完整上游子图。"""
    if not isinstance(prompt, dict):
        return {}
    me = prompt.get(str(node_unique_id))
    if not isinstance(me, dict):
        return {}
    link = (me.get("inputs") or {}).get("latent")
    if not _is_link(link):
        return {}
    seen: dict = {}
    stack = [str(link[0])]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        node = prompt.get(nid)
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") or {}
        literals = {k: v for k, v in ins.items() if not _is_link(v)}
        links = {k: str(v[0]) for k, v in ins.items() if _is_link(v)}
        seen[nid] = {"t": node.get("class_type"), "lit": literals, "lnk": links}
        stack.extend(links.values())
    return seen


def _summarize(graph: dict) -> dict:
    seeds, texts, classes = [], [], []
    for node in graph.values():
        classes.append(str(node.get("t") or "?"))
        for k, v in (node.get("lit") or {}).items():
            if k in _SEED_KEYS and isinstance(v, (int, float, str)):
                seeds.append(f"{v}")
            if k in _TEXT_KEYS and isinstance(v, str) and v.strip():
                texts.append(v.strip().replace("\n", " ")[:70])
    return {
        "nodes": len(graph),
        "seeds": sorted(set(seeds))[:3],
        "texts": texts[:2],
        "has": sorted(set(classes))[:12],
    }


def fingerprint_of(prompt, node_unique_id) -> tuple[str, dict]:
    """沿本节点 `latent` 输入反向上游（给 AV Latent Cache (Save) 用）。"""
    graph = upstream_subgraph(prompt, node_unique_id)
    if not graph:
        return "", {"note": "no upstream graph available"}
    blob = json.dumps(graph, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), _summarize(graph)


def upstream_subgraph_from_inputs(prompt, node_unique_id) -> dict:
    """从本节点**所有**链接输入反向收集上游子图。

    一采节点（噪声/guider/sampler/sigmas/latent 全是输入）的身份 = 全部上游，
    所以一采缓存用这个；AV Latent Save 只需要 latent 链，用 upstream_subgraph。
    """
    if not isinstance(prompt, dict):
        return {}
    me = prompt.get(str(node_unique_id))
    if not isinstance(me, dict):
        return {}
    links = [v for v in (me.get("inputs") or {}).values() if _is_link(v)]
    seen: dict = {}
    stack = [str(v[0]) for v in links]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        node = prompt.get(nid)
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") or {}
        literals = {k: v for k, v in ins.items() if not _is_link(v)}
        lnks = {k: str(v[0]) for k, v in ins.items() if _is_link(v)}
        seen[nid] = {"t": node.get("class_type"), "lit": literals, "lnk": lnks}
        stack.extend(lnks.values())
    return seen


def fingerprint_from_node_inputs(prompt, node_unique_id) -> tuple[str, dict]:
    """一采节点的指纹 = 全部输入的上游子图（噪声/提示词/guider/sampler/sigmas…）。"""
    graph = upstream_subgraph_from_inputs(prompt, node_unique_id)
    if not graph:
        return "", {"note": "no upstream graph available"}
    blob = json.dumps(graph, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), _summarize(graph)


def fingerprint_diff(stored: dict, expected: dict) -> list:
    """粗粒度对比摘要，给出"哪里变了"的线索（不追求完备，够定位就行）。"""
    out = []
    if not isinstance(stored, dict) or not isinstance(expected, dict):
        return ["<no-summary>"]
    for key, label in (("seeds", "种子"), ("texts", "文本"), ("nodes", "节点数")):
        if stored.get(key) != expected.get(key):
            out.append(f"{label}: {stored.get(key)} -> {expected.get(key)}")
    if len(out) == 0 and stored.get("has") != expected.get("has"):
        out.append("上游节点集合变了")
    return out or ["（摘要看不出差异，可能只是数值细节）"]

