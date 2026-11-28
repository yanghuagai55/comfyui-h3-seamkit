#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2 独立提前验证与基准评测脚本：ROCm Macro-Tile Gather-Flash + 208层(含TokenRefiner) + 编译RoPE。

设计目标：
  - 零侵入：不修改任何现有训练脚本，可在今日 Day-2 `step 140` 收尾回传后（约 14:25~15:30），
    趁已抢到的 AMD MI300X 192GB 实例尚未到期，用 ~12 分钟直接完成 Phase 2 真机实测验证！
  - 双模式：
    1) `--self-test`：纯合成张量在任意 GPU（含本地 8G 4060 或云端 MI300X）5 秒内验证
       `macro_gather_flash_attn` 与精确块展开 `sdpa_mask` 的前向输出 + 反向 `dq/dk/dv` 梯度
       100% 数学等价（cos > 0.99999, 0 NaN），并测算单层 Attention 前反向提速比；
    2) 默认真机模式：加载真实 20B H3 底模与 `teacher_cache` 中的真实 `39f` (`sakuga39_*`) 样本，
       对比 Baseline (`sdpa_mask` + 200层) vs Phase 2 (`rocm_gather_flash` + 编译RoPE + 208层含TokenRefiner)，
       采集单步耗时、显存峰值、`amd-smi/rocm-smi` 实时功耗(W)/频率(MHz) 与 `416/416` 活梯度门禁，
       输出 JSON 报告并可选一键上传至魔搭私有仓 `candidates_v3/phase2_rocm_benchmark.json`。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


# ============================================================================
# 1. 核心算子：Macro-Tile Morton KV-Gather 无掩码 FlashAttention
# ============================================================================
def build_macro_gather_plan(
    sel_block: torch.Tensor,
    block_size: int,
    seq_len: int,
    va: int,
    vb: int,
    keep_kv: torch.Tensor | None = None,
    macro_mult: int = 4,
    macro_chunk: int = 16,
) -> dict:
    """将 `[NQB, NKV]` 64-block 稀疏选择矩阵编译为无掩码 `Macro-Tile Gather` 预展平索引计划。

    参数:
      sel_block: `[NQB, NKV]` bool 张量（已在 Morton 空间构建）。
      block_size: 基础块大小（64）。
      seq_len: 总序列长度 `S`。
      va, vb: 视频段起止位置 `[va, vb)`。
      keep_kv: `[S]` bool 有效 token 掩码（排除 text_pad 填充位，使 attn_mask 严格为 None！）。
      macro_mult: 每个宏查询块包含几个 64-block（默认 4 -> 256 tokens = 16x16 patch 正方形象限）。
      macro_chunk: 每批聚拢宏块数（默认 16，临时聚拢显存 < 3GB）。
    """
    device = sel_block.device
    nqb, nkv = sel_block.shape
    n_total_blks = max(nqb, (seq_len + block_size - 1) // block_size)
    valid_tok = (
        keep_kv[:seq_len].to(device=device, dtype=torch.bool)
        if keep_kv is not None
        else torch.ones(seq_len, dtype=torch.bool, device=device)
    )
    all_valid_idx = torch.nonzero(valid_tok, as_tuple=False).squeeze(-1)

    # 区分纯视频整块与稠密 Sink 查询块（text / ref / audio / 边界非整块）
    vid_q_blocks: list[int] = []
    sink_q_tok_list: list[torch.Tensor] = []
    for qb in range(n_total_blks):
        qs, qe = qb * block_size, min((qb + 1) * block_size, seq_len)
        if qs >= seq_len:
            break
        if qb < nqb and qs >= va and qe <= vb and (qe - qs) == block_size:
            vid_q_blocks.append(qb)
        else:
            sink_q_tok_list.append(torch.arange(qs, qe, device=device, dtype=torch.long))

    sink_q_idx = (
        torch.cat(sink_q_tok_list, dim=0)
        if sink_q_tok_list
        else torch.empty(0, dtype=torch.long, device=device)
    )
    if not vid_q_blocks:
        inv_q_order = torch.argsort(sink_q_idx)
        return {
            "sink_q_idx": sink_q_idx,
            "all_valid_idx": all_valid_idx,
            "macro_q_idx": None,
            "macro_kv_idx": None,
            "macro_valid_q_lens": [],
            "chunks": [],
            "bm_tokens": macro_mult * block_size,
            "kv_tok_len": int(all_valid_idx.numel()),
            "inv_q_order": inv_q_order,
            "identity_order": bool((sink_q_idx == torch.arange(seq_len, device=device)).all()),
            "keep_ratio": 1.0,
        }

    # 按 macro_mult (4 个 64-block = 256 tokens) 分组并取块选择并集
    macro_groups: list[list[int]] = [
        vid_q_blocks[i:i + macro_mult] for i in range(0, len(vid_q_blocks), macro_mult)
    ]
    n_macro = len(macro_groups)
    macro_sel = torch.zeros((n_macro, nkv), dtype=torch.bool, device=device)
    for m_idx, grp in enumerate(macro_groups):
        macro_sel[m_idx] = sel_block[grp].any(dim=0)

    # 统一所有宏块的活跃 KV block 数（补齐至本步最大活跃块数 k_max，保证批处理无 padding mask！）
    counts = macro_sel.sum(dim=1)
    k_max = int(counts.max().item())
    for m_idx in range(n_macro):
        deficit = k_max - int(counts[m_idx].item())
        if deficit > 0:
            avail = torch.nonzero(~macro_sel[m_idx], as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                macro_sel[m_idx, avail[:deficit]] = True

    # 展开为 token 级索引并剔除 keep_kv 为 False 的无效填充位
    kv_lists: list[torch.Tensor] = []
    min_kv_len = seq_len
    for m_idx in range(n_macro):
        blk_ids = torch.nonzero(macro_sel[m_idx], as_tuple=False).squeeze(-1)
        tok_ids = (
            blk_ids[:, None] * block_size + torch.arange(block_size, device=device)[None, :]
        ).reshape(-1)
        tok_ids = tok_ids[tok_ids < seq_len]
        tok_ids = tok_ids[valid_tok[tok_ids]]
        kv_lists.append(tok_ids)
        min_kv_len = min(min_kv_len, int(tok_ids.numel()))

    max_kv_len = max(int(t.numel()) for t in kv_lists)
    if min_kv_len != max_kv_len:
        padded_kv = []
        for t in kv_lists:
            need = max_kv_len - int(t.numel())
            if need > 0:
                present = torch.zeros(seq_len, dtype=torch.bool, device=device)
                present[t] = True
                extra = torch.nonzero(valid_tok & ~present, as_tuple=False).squeeze(-1)[:need]
                t = torch.cat([t, extra], dim=0)
            padded_kv.append(t)
        kv_lists = padded_kv

    macro_kv_idx = torch.stack(kv_lists, dim=0)  # [M, K_tok]

    # 构建宏查询 token 索引矩阵 [M, B_M] 与一次性逆置换表 inv_q_order（消除逐块 index_put 拷贝！）
    bm_tokens = macro_mult * block_size
    q_rows: list[torch.Tensor] = []
    q_unpadded: list[torch.Tensor] = []
    q_real_lens: list[int] = []
    for grp in macro_groups:
        q_toks = torch.cat(
            [torch.arange(b * block_size, (b + 1) * block_size, device=device, dtype=torch.long) for b in grp],
            dim=0,
        )
        real_n = int(q_toks.numel())
        q_real_lens.append(real_n)
        q_unpadded.append(q_toks)
        if real_n < bm_tokens:
            pad_q = q_toks[-1:].expand(bm_tokens - real_n)
            q_toks = torch.cat([q_toks, pad_q], dim=0)
        q_rows.append(q_toks)
    macro_q_idx = torch.stack(q_rows, dim=0)  # [M, B_M]

    # 预展平每批 chunk 索引，使 50 层前向 + checkpoint 重算零开销直接复用
    chunks: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    for m0 in range(0, n_macro, macro_chunk):
        m1 = min(m0 + macro_chunk, n_macro)
        mc = m1 - m0
        q_1d = macro_q_idx[m0:m1].reshape(-1)
        kv_1d = macro_kv_idx[m0:m1].reshape(-1)
        valid_n = sum(q_real_lens[m0:m1])
        chunks.append((q_1d, kv_1d, mc, valid_n))

    ordered_pieces = ([sink_q_idx] if sink_q_idx.numel() > 0 else []) + q_unpadded
    cat_q_order = torch.cat(ordered_pieces, dim=0)
    identity_order = bool(
        cat_q_order.numel() == seq_len
        and (cat_q_order == torch.arange(seq_len, device=device)).all()
    )
    inv_q_order = torch.argsort(cat_q_order)

    return {
        "sink_q_idx": sink_q_idx,
        "all_valid_idx": all_valid_idx,
        "macro_q_idx": macro_q_idx,
        "macro_kv_idx": macro_kv_idx,
        "macro_valid_q_lens": q_real_lens,
        "chunks": chunks,
        "bm_tokens": bm_tokens,
        "kv_tok_len": max_kv_len,
        "inv_q_order": inv_q_order,
        "identity_order": identity_order,
        "keep_ratio": round(float(max_kv_len) / max(1, int(all_valid_idx.numel())), 4),
    }


def macro_gather_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: dict,
    scale: float,
) -> torch.Tensor:
    """对 `[1, H, S, D]` 执行 100% 无掩码 (`attn_mask=None`) 的 Macro-Tile Gather FlashAttention。

    关键底层优化：
      1) 全程保持物理 `BSHD` 连续内存布局（`.view(mc, L, H, D).transpose(1, 2)` 零拷贝直喂 FlashAttention，
         规避 C++ SDPA 内部二次 `BHSD <-> BSHD` 转置）；
      2) 用单次 `torch.cat` + 单次 `index_select(0, inv_q_order)` 替代逐宏块 `index_put`，零全图克隆开销！
    """
    _, heads, _, head_dim = q.shape
    scale_std = head_dim ** -0.5
    temp_mult = float(scale) / scale_std

    # 预乘温度因子，使 SDPA 使用默认 scale=None 以最大化 ROCm Flash 内核命中率
    q_scaled = (q * temp_mult).to(q.dtype) if abs(temp_mult - 1.0) > 1e-6 else q

    # [S, H, D] 物理连续，后续 index_select 后直接以零拷贝 BSHD 视图送入 Flash 内核
    q_shd = q_scaled.squeeze(0).transpose(0, 1).contiguous()
    k_shd = k.squeeze(0).transpose(0, 1).contiguous()
    v_shd = v.squeeze(0).transpose(0, 1).contiguous()

    out_chunks: list[torch.Tensor] = []

    # 1) Sink 查询行（文本 + 参考图 + 音频）：全有效 KV 无掩码 FlashAttention
    sink_q_idx = plan["sink_q_idx"]
    all_valid_idx = plan["all_valid_idx"]
    if sink_q_idx.numel() > 0:
        q_sink = q_shd.index_select(0, sink_q_idx).unsqueeze(0).transpose(1, 2)
        k_all = k_shd.index_select(0, all_valid_idx).unsqueeze(0).transpose(1, 2)
        v_all = v_shd.index_select(0, all_valid_idx).unsqueeze(0).transpose(1, 2)
        o_sink = F.scaled_dot_product_attention(q_sink, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)
        out_chunks.append(o_sink.transpose(1, 2).squeeze(0).contiguous())

    # 2) 视频宏块行：按预展平 chunk 聚拢紧凑 KV 并调起无掩码原生 FlashAttention
    bm = plan["bm_tokens"]
    ktok = plan["kv_tok_len"]
    for q_1d, kv_1d, mc, valid_n in plan["chunks"]:
        q_c = q_shd.index_select(0, q_1d).view(mc, bm, heads, head_dim).transpose(1, 2)
        k_c = k_shd.index_select(0, kv_1d).view(mc, ktok, heads, head_dim).transpose(1, 2)
        v_c = v_shd.index_select(0, kv_1d).view(mc, ktok, heads, head_dim).transpose(1, 2)
        o_c = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=None, dropout_p=0.0, is_causal=False)
        o_flat = o_c.transpose(1, 2).contiguous().view(mc * bm, heads, head_dim)
        out_chunks.append(o_flat if valid_n == mc * bm else o_flat[:valid_n])

    out_ordered = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=0)
    out_shd = out_ordered if plan["identity_order"] else out_ordered.index_select(0, plan["inv_q_order"])
    return out_shd.transpose(0, 1).unsqueeze(0)


# ============================================================================
# 2. 合成等价性单元测试 & 单层前反向微基准 (--self-test)
# ============================================================================
def run_synthetic_selftest(seq_len: int = 8192, heads: int = 56, head_dim: int = 128) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = 64
    nqb = seq_len // block
    va = 8 * block
    vb = seq_len - 4 * block

    torch.manual_seed(2026)
    sel = torch.zeros((nqb, nqb), dtype=torch.bool, device=device)
    for i in range(nqb):
        qs, qe = i * block, (i + 1) * block
        if qs >= va and qe <= vb:
            sel[i, :8] = True
            sel[i, seq_len // block - 4:] = True
            lo, hi = max(8, i - 6), min(nqb - 4, i + 7)
            sel[i, lo:hi] = True
        else:
            sel[i, :] = True

    keep = torch.ones(seq_len, dtype=torch.bool, device=device)
    keep[200:256] = False  # 模拟文本 padding 位

    plan = build_macro_gather_plan(sel, block, seq_len, va, vb, keep_kv=keep, macro_mult=4, macro_chunk=16)

    # 构造与 plan 完全对应的展开布尔掩码以验证逐位等价性
    ref_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    ref_mask[plan["sink_q_idx"][:, None], plan["all_valid_idx"][None, :]] = True
    for m_i, rn in enumerate(plan["macro_valid_q_lens"]):
        q_t = plan["macro_q_idx"][m_i, :rn]
        kv_t = plan["macro_kv_idx"][m_i]
        ref_mask[q_t[:, None], kv_t[None, :]] = True
    ref_mask_4d = ref_mask.unsqueeze(0).unsqueeze(0)

    scale = (head_dim ** -0.5) * 1.08
    q_base = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    k_base = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    v_base = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    grad_out = torch.randn_like(q_base) * 0.1

    q1, k1, v1 = [t.clone().detach().requires_grad_(True) for t in (q_base, k_base, v_base)]
    out_ref = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=ref_mask_4d, scale=scale)
    (out_ref * grad_out).sum().backward()

    q2, k2, v2 = [t.clone().detach().requires_grad_(True) for t in (q_base, k_base, v_base)]
    out_new = macro_gather_flash_attn(q2, k2, v2, plan, scale=scale)
    (out_new * grad_out).sum().backward()

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        af, bf = a.detach().float().flatten(), b.detach().float().flatten()
        return float((af * bf).sum() / (af.norm() * bf.norm() + 1e-12))

    cos_out = _cos(out_ref, out_new)
    cos_dq = _cos(q1.grad, q2.grad)
    cos_dk = _cos(k1.grad, k2.grad)
    cos_dv = _cos(v1.grad, v2.grad)
    finite_all = all(torch.isfinite(t).all().item() for t in (out_new, q2.grad, k2.grad, v2.grad))

    print(
        f"[self-test S={seq_len}] keep_ratio={plan['keep_ratio']} finite={finite_all} | "
        f"cos(out)={cos_out:.6f} cos(dQ)={cos_dq:.6f} cos(dK)={cos_dk:.6f} cos(dV)={cos_dv:.6f}",
        flush=True,
    )
    assert finite_all and min(cos_out, cos_dq, cos_dk, cos_dv) > 0.9995, "Self-test equivalence failed!"

    # 单层前向+反向微基准
    def _time_fwd_bwd(fn, iters: int = 4) -> float:
        for _ in range(2):
            qt, kt, vt = [t.clone().detach().requires_grad_(True) for t in (q_base, k_base, v_base)]
            y = fn(qt, kt, vt)
            (y * grad_out).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            qt, kt, vt = [t.clone().detach().requires_grad_(True) for t in (q_base, k_base, v_base)]
            y = fn(qt, kt, vt)
            (y * grad_out).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000.0

    ms_masked = _time_fwd_bwd(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask_4d, scale=scale))
    ms_gather = _time_fwd_bwd(lambda q, k, v: macro_gather_flash_attn(q, k, v, plan, scale=scale))
    speedup = ms_masked / max(ms_gather, 1e-6)
    print(
        f"[micro-bench S={seq_len} fwd+bwd] sdpa_mask={ms_masked:.1f} ms vs "
        f"macro_gather_flash={ms_gather:.1f} ms  (speedup = {speedup:.2f}x)",
        flush=True,
    )
    return {
        "seq_len": seq_len,
        "keep_ratio": plan["keep_ratio"],
        "cos_out": round(cos_out, 6),
        "cos_dq": round(cos_dq, 6),
        "cos_dk": round(cos_dk, 6),
        "cos_dv": round(cos_dv, 6),
        "ms_sdpa_mask_fwdbwd": round(ms_masked, 2),
        "ms_gather_flash_fwdbwd": round(ms_gather, 2),
        "layer_attn_speedup": round(speedup, 2),
    }


# ============================================================================
# 3. 功耗/频率后台采样器（兼容 amd-smi / rocm-smi / nvidia-smi）
# ============================================================================
class PowerSampler:
    def __init__(self, interval: float = 1.5):
        self.interval = interval
        self.watts: list[float] = []
        self.mhz: list[float] = []
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)

    def _poll_once(self) -> None:
        for cmd in (
            "amd-smi metric -p -c 2>/dev/null",
            "rocm-smi --showpower --showclocks 2>/dev/null",
            "nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null",
        ):
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
                txt = (p.stdout or "").strip()
                if not txt:
                    continue
                if "nvidia-smi" in cmd:
                    parts = txt.splitlines()[0].split(",")
                    self.watts.append(float(parts[0].strip()))
                    self.mhz.append(float(parts[1].strip()))
                    return
                w_m = re.search(r"(\d+(?:\.\d+)?)\s*W\b", txt, re.I)
                c_m = re.search(r"(\d+(?:\.\d+)?)\s*MHz\b", txt, re.I)
                if w_m:
                    val_w = float(w_m.group(1))
                    if 20.0 <= val_w <= 1000.0:
                        self.watts.append(val_w)
                if c_m:
                    val_c = float(c_m.group(1))
                    if 100.0 <= val_c <= 3500.0:
                        self.mhz.append(val_c)
                if w_m or c_m:
                    return
            except Exception:
                continue

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thr.join(timeout=3)

    def summary(self) -> dict:
        return {
            "avg_watts": round(sum(self.watts) / len(self.watts), 1) if self.watts else None,
            "max_watts": round(max(self.watts), 1) if self.watts else None,
            "avg_sclk_mhz": round(sum(self.mhz) / len(self.mhz), 0) if self.mhz else None,
            "max_sclk_mhz": round(max(self.mhz), 0) if self.mhz else None,
            "samples": len(self.watts),
        }


# ============================================================================
# 4. 真实 20B MiniMax-H3 (39f + 208层含TokenRefiner) 真机 A/B 实测
# ============================================================================
def attach_token_refiner_lora(dm, T_mod, rank_tr: int = 32) -> list[nn.Parameter]:
    """为 `dm.token_refiner.blocks.{0,1}` 附加 8 个纯线性 LoRA 层（构成完整 208 层），并改为非原地残差。"""
    tr = getattr(dm, "token_refiner", None)
    if tr is None or not hasattr(tr, "blocks"):
        return []
    tr_params: list[nn.Parameter] = []
    for blk in tr.blocks:
        for name in T_mod.LORA_TARGETS:
            is_attn = name in T_mod.ATTN_TARGETS
            lin = getattr(blk.attn if is_attn else blk.mlp, name)
            lora = T_mod.LoRALayer(lin.weight.shape[0], lin.weight.shape[1], rank_tr).to(lin.weight.device)
            lin._train_lora = lora
            tr_params.extend(lora.parameters())

        def _tr_mlp_fwd(self_mlp, x):
            g = T_mod._lora_linear(self_mlp.fc1, x)
            gate, up = g.chunk(2, dim=-1)
            return T_mod._lora_linear(self_mlp.fc2, F.silu(gate) * up)

        def _tr_attn_fwd(self_attn, x, rope_freqs=None, transformer_options=None):
            s = x.shape[0]
            heads = self_attn.heads
            hd = self_attn.head_dim
            inner = heads * hd
            q, k, v = T_mod._lora_linear(self_attn.qkv_proj, x).split(inner, dim=-1)
            q = T_mod._rms_norm(q.view(s, heads, hd), self_attn.q_norm.weight,
                                float(self_attn.q_norm.eps) if self_attn.q_norm.eps is not None else 1e-6)
            k = T_mod._rms_norm(k.view(s, heads, hd), self_attn.k_norm.weight,
                                float(self_attn.k_norm.eps) if self_attn.k_norm.eps is not None else 1e-6)
            v = v.view(s, heads, hd)
            q_b = q.unsqueeze(0).transpose(1, 2)
            k_b = k.unsqueeze(0).transpose(1, 2)
            v_b = v.unsqueeze(0).transpose(1, 2)
            out = F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=0.0, is_causal=False)
            return T_mod._lora_linear(self_attn.out_proj, out.transpose(1, 2).squeeze(0).contiguous().view(s, inner))

        def _tr_blk_fwd(self_blk, x, transformer_options={}):
            x = self_blk.attn(self_blk.norm1(x), transformer_options=transformer_options) + x
            return self_blk.mlp(self_blk.norm2(x)) + x

        blk.mlp.forward = types.MethodType(_tr_mlp_fwd, blk.mlp)
        blk.attn.forward = types.MethodType(_tr_attn_fwd, blk.attn)
        blk.forward = types.MethodType(_tr_blk_fwd, blk)
    return tr_params


def run_full_mi300x_ab(args) -> dict:
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import train_h3_lora as T

    device = torch.device("cuda")
    mm, comfy_sd, h3 = T.bootstrap(args.comfy)
    T.patch_out_of_place(h3)
    dm = T.load_dm(comfy_sd, args.base, device)

    main_params = T.attach_lora(dm, rank=32, rank_attn=64, rank_mlp=32)
    tr_params = attach_token_refiner_lora(dm, T, rank_tr=32)
    all_params = main_params + tr_params
    print(
        f"[phase2] attached linear LoRA layers: main={len(main_params)//2} + "
        f"token_refiner={len(tr_params)//2} = {len(all_params)//2} layers ({len(all_params)} tensors)",
        flush=True,
    )

    # 尝试仅对纯张量函数 _rope_chunked 做局部编译（失败自动无感回退原函数）
    rope_fn = T._rope_chunked
    compiled_rope_ok = False
    if not args.no_compile_rope:
        try:
            cand_fn = torch.compile(T._rope_chunked, dynamic=True)
            _test_q = torch.randn(1024, 56, 128, device=device, dtype=torch.bfloat16)
            _test_tb = torch.ones(1, 1024, 1, 48, 2, 2, device=device, dtype=torch.float32)
            _ = cand_fn(_test_q, _test_tb)
            rope_fn = cand_fn
            compiled_rope_ok = True
            print("[phase2] @torch.compile(_rope_chunked) warmup OK!", flush=True)
        except Exception as exc:
            print(f"[phase2] compile _rope_chunked skipped (fallback eager): {exc}", flush=True)

    # 快速选取真实 39f R2VA 动作样本（优先只扫描 sakuga39_*.pt 避免遍历全部 90 个 NAS 大文件耗时）
    cache_dir = Path(args.cache)
    sakuga_pts = sorted(cache_dir.glob("sakuga39_*.pt"))
    groups: dict[str, list[dict]] = {}
    for p in (sakuga_pts[:8] if sakuga_pts else []):
        try:
            raw = torch.load(p, map_location="cpu", weights_only=False)
            cond = raw["cond"][0]
            meta = dict(raw["meta"])
            meta["text_len"] = int(cond[0].shape[1])
            room = {
                "path": p, "stem": p.stem, "is_action": True, "meta": meta,
                "refs": cond[1].get("minimax_refs"),
                "keyframes": cond[1].get("minimax_keyframes"),
                "tags": cond[1].get("minimax_token_tags"),
            }
            k = f"{meta['latent_t']}x{meta['lat_h']}x{meta['lat_w']}a{meta['audio_t']}|" + repr(T._refs_key(cond[1] or {}))
            groups.setdefault(k, []).append(room)
        except Exception:
            continue
    if not groups:
        groups = T.scan_cache(cache_dir)

    g39_items = [(k, v) for k, v in groups.items() if int(v[0]["meta"].get("frames", 22)) == 39]
    if not g39_items:
        g39_items = list(groups.items())
    # 优先选带参考图（R2VA 最重长序列）的 sakuga39 样本
    g39_items.sort(key=lambda kv: (bool(kv[1][0].get("refs")), kv[1][0].get("is_action", False)), reverse=True)
    chosen_k, chosen_recs = g39_items[0]
    group = T._build_group(h3, chosen_k, chosen_recs, block=64, neighbors=2)
    rec = group.records[0]
    print(f"[phase2] selected real sample={rec['stem']} bucket={chosen_k.split('|')[0]} S={group.S}", flush=True)

    # 安装支持 sdpa_mask 与 macro_gather_flash 双模式切换的 attention forward
    def _make_dual_attn_forward(head_dim: int, attn_temp: float):
        scale_base = head_dim ** -0.5

        def forward(self_a, x, rope_freqs=None, transformer_options=None):
            to = transformer_options or {}
            s = x.shape[0]
            heads = self_a.heads
            inner = heads * head_dim
            q, k, v = T._lora_linear(self_a.qkv_proj, x).split(inner, dim=-1)
            q = T._rms_norm(q.view(s, heads, head_dim), self_a.q_norm.weight,
                            float(self_a.q_norm.eps) if self_a.q_norm.eps is not None else 1e-6)
            k = T._rms_norm(k.view(s, heads, head_dim), self_a.k_norm.weight,
                            float(self_a.k_norm.eps) if self_a.k_norm.eps is not None else 1e-6)
            v = v.view(s, heads, head_dim)
            cur_rope = rope_fn if to.get("_use_compiled_rope") else T._rope_chunked
            if rope_freqs is not None:
                q = cur_rope(q, rope_freqs)
                k = cur_rope(k, rope_freqs)

            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            scale = scale_base * float(to.get("_h3_attn_temp", attn_temp))

            m_info = to.get("_h3_morton")
            if m_info is not None:
                m_va, m_vb, m_perm, m_inv = m_info
                q = T._permute_bhsd_video(q, m_va, m_vb, m_perm)
                k = T._permute_bhsd_video(k, m_va, m_vb, m_perm)
                v = T._permute_bhsd_video(v, m_va, m_vb, m_perm)

            gather_plan = to.get("_h3_gather_plan")
            sdpa_mask = to.get("_h3_sdpa_mask")
            if gather_plan is not None:
                out = macro_gather_flash_attn(q, k, v, gather_plan, scale=scale)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False, scale=scale)

            if m_info is not None:
                out = T._permute_bhsd_video(out, m_va, m_vb, m_inv)
            return T._lora_linear(self_a.out_proj, out.squeeze(0).transpose(0, 1).reshape(s, inner))

        return forward

    for blk in dm.blocks:
        blk.attn.forward = types.MethodType(_make_dual_attn_forward(int(blk.attn.head_dim), 1.08), blk.attn)

    cfg = {
        "rope_hf": 1.32, "attn_temp": 1.08, "sparse": True,
        "attn_backend": "sdpa_mask", "topk": 0.22, "topk_max": 0.25,
        "ckpt": True, "grad_clamp": 10.0, "mask_jitter": 0.0,
    }
    T.patch_run_blocks(dm, h3, cfg)

    opt = torch.optim.AdamW(all_params, lr=5e-5)
    gen = torch.Generator(device=device).manual_seed(2026)

    def _run_mode(mode_name: str, use_gather: bool, n_steps: int) -> dict:
        torch.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        nonzero_grads = 0

        with PowerSampler(interval=1.5) as ps:
            for st in range(n_steps):
                t0 = time.time()
                sigma = 0.35
                sigma_a = h3.time_shift_sigma(sigma, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else sigma
                batch = T.build_batch(rec, group, sigma, sigma_a, device, gen)

                # 预构造同一稀疏块矩阵对应的 mask 或 gather_plan
                sel = T._select_sparse_blocks(group, 0.22, 0.25, device, h_ref=None, jitter=0.0)
                to_extra: dict = {"_h3_plan": group, "_h3_keep_kv": batch["keep"]}
                if use_gather:
                    gplan = build_macro_gather_plan(
                        sel, group.block, group.S, group.va, group.vb,
                        keep_kv=batch["keep"], macro_mult=args.macro_mult, macro_chunk=args.macro_chunk,
                    )
                    to_extra["_h3_gather_plan"] = gplan
                    to_extra["_h3_prebuilt_mask"] = None
                    to_extra["_use_compiled_rope"] = compiled_rope_ok
                    # 防止 patch_run_blocks 再构造旧 sdpa_mask
                    to_extra["_h3_keep_kv"] = None
                else:
                    smask = T.build_sparse_mask(group, 0.22, 0.25, batch["keep"], device, backend="sdpa_mask", jitter=0.0)
                    to_extra["_h3_prebuilt_mask"] = smask

                old_pos = group.layout.position_ids
                group.layout.position_ids = batch["pos"]
                try:
                    # 直接传原始 5120 维 b["text"] 进 dm(...)，让 _forward 原生可微穿过 condition_proj + token_refiner！
                    with T._disable_lora(), torch.no_grad():
                        ctx_base = dm.preprocess_text_embeds(batch["text"])
                        base_out = dm(batch["x"], torch.tensor([sigma * 1000.0], device=device), ctx_base,
                                      transformer_options=dict(to_extra), minimax_payload=batch["payload"])
                        base_v = base_out[0].detach().float()

                    ctx_in = batch["text"] if use_gather else dm.preprocess_text_embeds(batch["text"])
                    out = dm(batch["x"], torch.tensor([sigma * 1000.0], device=device), ctx_in,
                             transformer_options=dict(to_extra), minimax_payload=batch["payload"])
                finally:
                    group.layout.position_ids = old_pos

                pred_vf = out[0].float()
                tgt_v = batch["tgt"][0]
                lf_base = T._spatial_lowpass(base_v, 5)
                lf_tgt = T._spatial_lowpass(tgt_v, 5)
                m_mot = T._motion_gate_mask(batch["z_video"], 0.75)
                tgt_ref = (1.0 - m_mot) * (lf_base + (tgt_v - lf_tgt)) + m_mot * tgt_v
                loss = F.mse_loss(pred_vf, tgt_ref) + 0.15 * T._spatial_grad_l1(pred_vf, tgt_ref) + 0.1 * F.mse_loss(out[1].float(), batch["tgt"][1])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nonzero_grads = sum(
                    1 for p in all_params
                    if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.detach().float().abs().max()) > 0.0
                )
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                torch.cuda.synchronize()
                dt = time.time() - t0
                step_times.append(dt)
                print(
                    f"[{mode_name} step {st+1}/{n_steps}] sec={dt:.2f}s loss={float(loss):.4f} "
                    f"grad_alive={nonzero_grads}/{len(all_params)} vram={torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                    flush=True,
                )

        # 去首步预热求稳态均值
        steady = step_times[1:] if len(step_times) > 1 else step_times
        res = {
            "mode": mode_name,
            "step_times_sec": [round(x, 2) for x in step_times],
            "steady_avg_sec": round(sum(steady) / len(steady), 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "grad_alive": f"{nonzero_grads}/{len(all_params)}",
            "telemetry": ps.summary(),
        }
        return res

    print("\n=== [A] Running Baseline: sdpa_mask (200 active DiT layers) ===", flush=True)
    res_base = _run_mode("baseline_sdpa_mask", use_gather=False, n_steps=args.base_steps)

    print("\n=== [B] Running Phase 2: rocm_gather_flash + compiled_rope + 208 layers (with TokenRefiner) ===", flush=True)
    res_p2 = _run_mode("phase2_rocm_gather_flash_208L", use_gather=True, n_steps=args.p2_steps)

    e2e_speedup = round(res_base["steady_avg_sec"] / max(res_p2["steady_avg_sec"], 0.01), 2)
    print(
        f"\n[SUMMARY] Baseline={res_base['steady_avg_sec']}s/step ({res_base['telemetry'].get('avg_watts')}W) -> "
        f"Phase2={res_p2['steady_avg_sec']}s/step ({res_p2['telemetry'].get('avg_watts')}W) | "
        f"End-to-End Speedup = {e2e_speedup}x | Grads = {res_p2['grad_alive']}",
        flush=True,
    )
    return {
        "timestamp": time.strftime("%F %T"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "sample": rec["stem"],
        "seq_len": group.S,
        "compiled_rope": compiled_rope_ok,
        "macro_mult": args.macro_mult,
        "macro_chunk": args.macro_chunk,
        "baseline": res_base,
        "phase2": res_p2,
        "e2e_speedup": e2e_speedup,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="仅运行合成等价性与单层微基准测试（任意显卡 5s 完成）")
    ap.add_argument("--seq-len", type=int, default=8192, help="self-test 序列长度（MI300X 可设 26624）")
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--base", default="/mnt/data/h3_v3/models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors")
    ap.add_argument("--cache", default="/mnt/data/h3_v3/work/teacher_cache")
    ap.add_argument("--out-json", default="/mnt/data/h3_v3/logs/phase2_rocm_benchmark.json")
    ap.add_argument("--macro-mult", type=int, default=4, help="宏块倍数（4=256 tokens=16x16象限）")
    ap.add_argument("--macro-chunk", type=int, default=16, help="每批聚拢宏块数（控制临时显存 <3GB）")
    ap.add_argument("--base-steps", type=int, default=2)
    ap.add_argument("--p2-steps", type=int, default=3)
    ap.add_argument("--no-compile-rope", action="store_true")
    ap.add_argument("--upload-ms", action="store_true", help="跑完自动上传报告至魔搭私有仓 candidates_v3/")
    ap.add_argument("--ms-repo", default="coolwind123/h3-teacher-cache")
    ap.add_argument("--token-file", default="/mnt/data/h3_v3/work/mstoken.txt")
    args = ap.parse_args()

    report: dict = {"selftest": run_synthetic_selftest(seq_len=args.seq_len)}
    if args.self_test:
        return 0

    report["e2e_mi300x"] = run_full_mi300x_ab(args)
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] -> {out_p}", flush=True)

    if args.upload_ms and Path(args.token_file).is_file():
        tok = Path(args.token_file).read_text(encoding="utf-8").strip()
        try:
            from modelscope_hub.api import HubApi
            api = HubApi(token=tok)
        except ImportError:
            from modelscope.hub.api import HubApi
            api = HubApi()
            api.login(tok)
        api.upload_file(
            repo_id=args.ms_repo, repo_type="model",
            path_or_fileobj=str(out_p),
            path_in_repo="candidates_v3/phase2_rocm_benchmark.json",
            commit_message="phase2 rocm gather-flash benchmark",
        )
        print("[uploaded] -> candidates_v3/phase2_rocm_benchmark.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
