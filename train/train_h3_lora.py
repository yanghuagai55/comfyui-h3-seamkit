# SPDX-License-Identifier: GPL-3.0-or-later
"""MiniMax-H3 蒸馏 LoRA 训练（A100-80GB / RTX PRO 6000-96GB，纯 PyTorch）。

任务
----
教师：官方 BF16 满血、24 步 CFG5、Split-RoPE 1.32 + attn_temp 1.08 生成的
1080p 原生真值 `z_native`；学生：同一份 BF16 底模 + 200 个 DiT 线性层 LoRA，
输入 `x_σ = (1-σ)·z_up + σ·ε`（z_up = 480p VAE 往返 + 2× 上采样糊底），
回归 `v* = (x_σ - z_native)/σ`。σ∈[0.12, 0.85]，Phase B 用固定 2 点直线化。

关键实现选择
------------
* **不做文本长度动态**：同一 shape 组把文本 padding 到组内最大长度（并对齐到
  attention block 的整数倍），保证每个 bucket 的 packed 序列长度恒定 →
  flex_attention 以 `dynamic=False` 只编译 3~6 次。padding 行的 RoPE 时间坐标
  用 `text_pad`，其余段整体回移 Δ = text_pad - real_len，与教师（真实文本长度）
  的位置编码逐位一致；padding 文本只作为被 mask 掉的 key。
* **稀疏注意力**：video 行 = 前缀 sink(text/refs/audio) + 自身块 + 邻帧块 +
  随机 top-k 视频块；非 video 行全稠密。`BlockMask.from_kv_blocks` +
  `flex_attention`，`mask_mod` 用设备端 bool 张量屏蔽 padding 文本。
* **LoRA 走低秩旁路**（F.linear 两次），不把 ΔW 叠进 40GB 基座权重：
  qkv/out/fc1/fc2 共 200 个线性层，B 初始化为 0（step0 与稠密基座等价）。
* 每块 `torch.utils.checkpoint`，73f 桶也能在 80GB 上跑。

用法
----
  python train/train_h3_lora.py --phase A --steps 1200 \
      --cache /content/work/teacher_cache --base /content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
      --out /content/work/lora_run --hf-repo <user>/<repo>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LORA_TARGETS = ("qkv_proj", "out_proj", "fc1", "fc2")
DEFAULT_TOKEN_FILE = "/content/work/hftoken.txt"


# ----------------------------------------------------------------------------
# comfy 引导
# ----------------------------------------------------------------------------
def bootstrap(comfy_root: str):
    root = str(Path(comfy_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.model_management as mm
    import comfy.sd
    from comfy.ldm.minimax import model as h3

    mm.in_training = True
    return mm, comfy.sd, h3


def load_dm(comfy_sd, base_path: str, device: torch.device):
    patcher = comfy_sd.load_diffusion_model(str(base_path), model_options={"dtype": torch.bfloat16})
    try:
        patcher.to(device)
    except Exception:
        patcher.model.to(device)
    dm = patcher.model.diffusion_model
    dm.eval()
    dm.requires_grad_(False)
    return dm


# ----------------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, out_f: int, in_f: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = float(rank if alpha is None else alpha) / float(rank)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


_LORA_ENABLED = True


@contextlib.contextmanager
def _disable_lora():
    """临时关闭 200 层 LoRA 旁路以获取冻结基座参考速度 v_base（异常安全）。"""
    global _LORA_ENABLED
    prev = _LORA_ENABLED
    _LORA_ENABLED = False
    try:
        yield
    finally:
        _LORA_ENABLED = prev


def _lora_linear(lin: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, lin.weight, getattr(lin, "bias", None))
    if _LORA_ENABLED:
        lora = getattr(lin, "_train_lora", None)
        if lora is not None:
            y = y + lora.delta(x).to(y.dtype)
    return y


def _spatial_lowpass(x: torch.Tensor, k: int) -> torch.Tensor:
    """视频潜空间 [B, C, T, H, W] 逐帧 2D 空间低通（replicate 边界，零边缘暗角）。"""
    if x.ndim != 5:
        raise ValueError(f"_spatial_lowpass 期望 5D [B,C,T,H,W]，实得 {tuple(x.shape)}")
    if k <= 1:
        return x.float()
    b, c, t, h, w = x.shape
    pad = k // 2
    y = x.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    y = F.pad(y, (pad, pad, pad, pad), mode="replicate")
    y = F.avg_pool2d(y, kernel_size=k, stride=1)
    return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _parse_blocks(spec: str, n: int) -> list[int]:
    if not spec.strip():
        return list(range(n))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


def attach_lora(dm, rank: int, blocks_spec: str = "") -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for i in _parse_blocks(blocks_spec, len(dm.blocks)):
        blk = dm.blocks[i]
        for name in LORA_TARGETS:
            lin = getattr(getattr(blk, "attn", None) if name in ("qkv_proj", "out_proj") else blk.mlp, name)
            lora = LoRALayer(lin.weight.shape[0], lin.weight.shape[1], rank).to(lin.weight.device)
            lin._train_lora = lora  # nn.Module 子模块注册，随 dm 一起搬设备
            params.extend(lora.parameters())

    def _mlp_forward(self, x):
        g = _lora_linear(self.fc1, x)
        gate, up = g.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return _lora_linear(self.fc2, h)

    for blk in dm.blocks:
        blk.mlp.forward = types.MethodType(_mlp_forward, blk.mlp)
    return params


class _GradClamp(torch.autograd.Function):
    """恒等前向 + 裁剪反向：放在两个 block 之间截断指数级放大的回传。"""

    @staticmethod
    def forward(ctx, x, limit):
        ctx.limit = limit
        return x

    @staticmethod
    def backward(ctx, grad):
        # 保方向的 L2 缩放（逐元素 clamp 会改变梯度方向）
        norm = grad.float().reshape(-1).norm()
        if float(norm) > float(ctx.limit):
            grad = grad * (float(ctx.limit) / (float(norm) + 1e-12))
        return grad, None


def lora_state(dm) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for i, blk in enumerate(dm.blocks):
        for name in LORA_TARGETS:
            owner = blk.attn if name in ("qkv_proj", "out_proj") else blk.mlp
            lin = getattr(owner, name)
            lora = getattr(lin, "_train_lora", None)
            if lora is not None:
                out[f"blocks.{i}.{'attn' if name in ('qkv_proj', 'out_proj') else 'mlp'}.{name}"] = {
                    "A": lora.A.detach().cpu(),
                    "B": lora.B.detach().cpu(),
                    "scale": lora.scale,
                }
    return out


def load_lora_state(dm, state: dict) -> None:
    for key, val in state.items():
        parts = key.split(".")
        blk = dm.blocks[int(parts[1])]
        owner = blk.attn if parts[2] == "attn" else blk.mlp
        lora = getattr(getattr(owner, parts[3]), "_train_lora", None)
        if lora is None:
            continue
        lora.A.data.copy_(val["A"].to(lora.A.device, lora.A.dtype))
        lora.B.data.copy_(val["B"].to(lora.B.device, lora.B.dtype))


# ----------------------------------------------------------------------------
# 注意力（稠密 SDPA / 稀疏 flex）
# ----------------------------------------------------------------------------
_FLEX = {"fn": None, "ok": None}
_KEEP_GPU = None


def _keep_mod(b, h, q_idx, kv_idx):
    """模块级 mask_mod：只读固定地址的全局 buffer，避免 dynamo 每步重编译。"""
    return _KEEP_GPU[kv_idx]


def _flex_fn():
    """编译版 flex_attention。禁止回退稠密：eager 会物化 S² 分数矩阵（56GB）直接 OOM。"""
    if _FLEX["ok"] is None:
        from torch.nn.attention.flex_attention import flex_attention

        # checkpoint 重算必须命中同一张静态图，否则报
        # CheckpointError: A different number of tensors was saved ...
        try:
            torch._dynamo.config.automatic_dynamic_shapes = False
            torch._dynamo.config.cache_size_limit = 64
        except Exception:
            pass
        try:
            torch._C._dynamo.eval_frame._set_lru_cache(False)
        except Exception:
            pass
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
        _FLEX["ok"] = True
        print("[attn] flex_attention compiled (dynamic=False, static shapes)", flush=True)
    return _FLEX["fn"]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _rope_chunked(q: torch.Tensor, table: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """FP32 旋转、BF16 存储；按 token 分块避免 73f 时 2×S×H×D fp32 瞬时峰值。"""
    half = table.shape[-3]
    rot = half * 2
    if rot <= 0 or rot > q.shape[-1]:
        return q
    s = q.shape[0]
    c_all = table[..., 0, 0][0]  # [S, 1, half]
    s_all = table[..., 1, 0][0]
    outs = []
    for i in range(0, s, chunk):
        j = min(i + chunk, s)
        qc = q[i:j].float()
        q0, q1 = qc[..., :half], qc[..., half:rot]
        c, sn = c_all[i:j].float(), s_all[i:j].float()
        o0 = c * q0 - sn * q1
        o1 = sn * q0 + c * q1
        outs.append(torch.cat([o0, o1, qc[..., rot:]], dim=-1).to(q.dtype))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _make_attn_forward(head_dim: int, attn_temp: float):
    scale_base = head_dim ** -0.5

    def forward(self, x, rope_freqs=None, transformer_options=None):
        to = transformer_options or {}
        s = x.shape[0]
        heads = self.heads
        inner = heads * head_dim
        q, k, v = _lora_linear(self.qkv_proj, x).split(inner, dim=-1)
        q = q.view(s, heads, head_dim)
        k = k.view(s, heads, head_dim)
        v = v.view(s, heads, head_dim)
        q = _rms_norm(q, self.q_norm.weight, float(self.q_norm.eps))
        k = _rms_norm(k, self.k_norm.weight, float(self.k_norm.eps))
        if rope_freqs is not None:
            q = _rope_chunked(q, rope_freqs)
            k = _rope_chunked(k, rope_freqs)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        scale = scale_base * float(to.get("_h3_attn_temp", attn_temp))

        bm = to.get("_h3_block_mask")
        if bm is not None:
            # 禁止静默回退：稠密路径会物化 S² 分数矩阵（56GB）直接 OOM
            try:
                out = _flex_fn()(q, k, v, block_mask=bm, scale=scale)
            except Exception as exc:
                raise RuntimeError(f"flex_attention 执行失败（禁止回退稠密）: {exc}") from exc
        else:
            # 稠密路径必须带 padding mask，否则重复的 padding token 会污染注意力
            keep_kv = to.get("_h3_keep_kv")
            attn_mask = keep_kv.view(1, 1, 1, -1) if keep_kv is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 dropout_p=0.0, is_causal=False, scale=scale)
        out = out.squeeze(0).transpose(0, 1).reshape(s, inner)
        return _lora_linear(self.out_proj, out)

    return forward


# ----------------------------------------------------------------------------
# 分桶 / 块稀疏 mask
# ----------------------------------------------------------------------------
@dataclass
class Group:
    key: str
    layout: object
    block: int
    S: int
    va: int
    vb: int
    NQB: int
    NKV: int
    base: torch.Tensor      # [NQB, NKV] bool (CPU)
    dense: torch.Tensor     # [NQB] bool
    video: torch.Tensor     # [NKV] bool
    tpf: int
    text_pad: int
    records: list = field(default_factory=list)


def _refs_key(meta: dict) -> tuple:
    parts = []
    for kf in meta.get("minimax_keyframes") or []:
        lat = kf.get("latent")
        parts.append(("kf", kf.get("latent_h"), kf.get("latent_w"), int(lat.shape[2]) if torch.is_tensor(lat) else 0))
    for r in meta.get("minimax_refs") or []:
        lat = r.get("latent")
        parts.append((r.get("kind"), r.get("latent_h"), r.get("latent_w"),
                      int(r.get("latent_t") or (lat.shape[2] if torch.is_tensor(lat) else 0)),
                      int(r.get("ref_audio_t") or 0)))
    return tuple(parts)


def _build_group(h3, key: str, records: list[dict], block: int, neighbors: int) -> Group:
    first = records[0]
    meta = first["meta"]
    sample_refs = first["refs"]
    sample_kf = first["keyframes"]
    text_pad = max(int(r["meta"]["text_len"]) for r in records)
    layout = h3.PackedLayout(text_pad, meta["latent_t"], meta["lat_h"], meta["lat_w"], meta["audio_t"],
                             keyframes=sample_kf, refs=sample_refs)
    pad = (-layout.seq_len) % block
    if pad:
        text_pad += pad
        layout = h3.PackedLayout(text_pad, meta["latent_t"], meta["lat_h"], meta["lat_w"], meta["audio_t"],
                                 keyframes=sample_kf, refs=sample_refs)
    S = layout.seq_len
    va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
    NQB = S // block
    NKV = NQB
    tpf = (meta["lat_h"] // 2) * (meta["lat_w"] // 2)

    base = torch.zeros(NQB, NKV, dtype=torch.bool)
    dense = torch.zeros(NQB, dtype=torch.bool)
    video = torch.zeros(NKV, dtype=torch.bool)
    q_first = va // block
    q_last = (vb - 1) // block
    video[q_first:q_last + 1] = True
    prefix = (va + block - 1) // block
    for b in range(NQB):
        qs, qe = b * block, min((b + 1) * block, S)
        if qs >= va and qe <= vb:
            base[b, :prefix] = True
            # 必须保留同帧上下相邻空间块（±2 块 ≈ ±136px 连续带），否则光栅条带间断链产生横纹撕裂
            for s_off in (-2, -1, 0, 1, 2):
                sb = b + s_off
                if q_first <= sb <= q_last:
                    base[b, sb] = True
            for d in range(1, neighbors + 1):
                for sign in (-1, 1):
                    center = b + int(round(sign * d * tpf / block))
                    for s_off in (-1, 0, 1):
                        nb = center + s_off
                        if q_first <= nb <= q_last:
                            base[b, nb] = True
        else:
            dense[b] = True
            base[b, :] = True
    g = Group(key=key, layout=layout, block=block, S=S, va=va, vb=vb, NQB=NQB, NKV=NKV,
              base=base, dense=dense, video=video, tpf=tpf, text_pad=text_pad, records=records)
    return g


def build_block_mask(g: Group, topk_ratio: float, topk_max_ratio: float, heads: int,
                     keep: torch.Tensor, device: torch.device,
                     h_ref: torch.Tensor | None = None, jitter: float = 0.05):
    from torch.nn.attention.flex_attention import BlockMask

    sel = g.base.to(device).clone()
    n_video = int(g.video.sum())
    k = 0 if topk_ratio <= 0 else max(1, int(round(topk_ratio * n_video)))
    k = min(k, max(1, int(round(topk_max_ratio * n_video))), max(1, g.NKV - 1))
    if k > 0:
        pool = (g.video.to(device)[None, :] & ~sel)
        if h_ref is not None and h_ref.shape[0] >= g.NQB * g.block:
            # 内容感知块质心余弦路由（模拟 4060 Sol-Attn 真实语义检索，彻底消除纯随机块造成的竖纹鬼影）
            with torch.no_grad():
                hb = h_ref[:g.NQB * g.block].detach().view(g.NQB, g.block, -1).float().mean(dim=1)
                hb = F.normalize(hb, dim=-1)
                scores = hb @ hb.transpose(0, 1)
                if jitter > 0.0:
                    scores = scores + float(jitter) * torch.rand_like(scores)
            scores = scores.masked_fill(~pool, -1e4)
        else:
            scores = torch.rand(g.NQB, g.NKV, device=device).masked_fill_(~pool, -1.0)
        top = torch.topk(scores, k, dim=1).indices
        sel.scatter_(1, top, True)
    sel |= g.dense.to(device)[:, None]
    kv_num = sel.sum(1).to(torch.int32)
    order = torch.argsort(sel.to(torch.int8), dim=1, descending=True, stable=True).to(torch.int32)

    # mask_mod 必须是模块级函数 + 固定地址 buffer：
    # 每步新建闭包会让 dynamo 每步重编译（撞 cache_size_limit 后退回 eager → 56GB OOM）
    global _KEEP_GPU
    if _KEEP_GPU is None or _KEEP_GPU.numel() < g.S or _KEEP_GPU.device != device:
        _KEEP_GPU = torch.ones(max(g.S, 32768), dtype=torch.bool, device=device)
    _KEEP_GPU[:g.S].copy_(keep)

    kwargs = dict(BLOCK_SIZE=g.block, mask_mod=_keep_mod)
    try:
        return BlockMask.from_kv_blocks(
            kv_num[None, None],
            order[None, None],
            seq_lengths=(g.S, g.S),
            **kwargs,
        )
    except TypeError:
        return BlockMask.from_kv_blocks(
            kv_num[None, None],
            order[None, None],
            **kwargs,
        )


# ----------------------------------------------------------------------------
# _run_blocks：Split-Band RoPE + 分块 checkpoint + mask 注入
# ----------------------------------------------------------------------------
def patch_run_blocks(dm, h3, cfg: dict):
    def run_blocks(self, h, t_emb, mod_segments, layout, transformer_options):
        device = h.device
        angles = self.rope_freqs(layout.position_ids.to(device), device)
        if cfg["rope_hf"] != 1.0 and layout.segments:
            va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
            nf = 16
            half = nf * 3
            g = torch.ones(nf, dtype=torch.float32, device=angles.device)
            s = float(cfg["rope_hf"])
            g[:min(6, nf)] = s
            if nf > 6:
                g[6] = 1.0 + (s - 1.0) * 0.75
            if nf > 7:
                g[7] = 1.0 + (s - 1.0) * 0.25
            angles = angles.clone()
            angles[va:vb, nf:2 * nf].mul_(g)
            angles[va:vb, 2 * nf:3 * nf].mul_(g)
            angles[va:vb, half + nf:half + 2 * nf].mul_(g)
            angles[va:vb, half + 2 * nf:half + 3 * nf].mul_(g)
        rope_freqs = h3.rope_rotation_table(angles, torch.float32)

        plan = transformer_options.get("_h3_plan")
        keep = transformer_options.get("_h3_keep_kv")
        prebuilt_bm = transformer_options.get("_h3_prebuilt_bm")
        if prebuilt_bm is not None:
            transformer_options["_h3_block_mask"] = prebuilt_bm
        elif plan is not None and keep is not None and cfg["sparse"]:
            bm = build_block_mask(
                plan, cfg["topk"], cfg["topk_max"], self.blocks[0].attn.heads, keep, device,
                h_ref=h, jitter=float(cfg.get("mask_jitter", 0.05)))
            transformer_options["_h3_block_mask"] = bm
            if transformer_options.get("_h3_save_bm"):
                transformer_options["_h3_saved_bm"] = bm
        transformer_options["_h3_attn_temp"] = cfg["attn_temp"]
        grad_on = torch.is_grad_enabled()
        use_clamp = cfg.get("grad_clamp", 0) > 0 and grad_on
        use_ckpt = cfg["ckpt"] and grad_on
        try:
            for i, block in enumerate(self.blocks):
                if use_clamp:
                    # 这个 checkpoint 的 adaln gate 极大（blk0 输出 9e4、blk49 1.7e7），
                    # 反向每层放大约 100×，纯 bf16 下 25 层后必 inf/NaN（原生模型也一样）。
                    # 在每个 block 边界把 incoming grad 夹到 ±limit，截断指数爆炸。
                    h = _GradClamp.apply(h, float(cfg["grad_clamp"]))
                if use_ckpt:
                    # reentrant 版不做「保存张量数量」校验，可绕开 compiled flex 的
                    # CheckpointError（85 vs 82）；代价是输入必须 requires_grad。
                    if not h.requires_grad:
                        h = h.detach().requires_grad_(True)

                    def _call(x, _i=i, _blk=block):
                        transformer_options["block_index"] = _i
                        return _blk(x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
                    h = torch.utils.checkpoint.checkpoint(_call, h, use_reentrant=True)
                else:
                    transformer_options["block_index"] = i
                    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        finally:
            transformer_options.pop("_h3_block_mask", None)
        return h

    dm._run_blocks = types.MethodType(run_blocks, dm)


def patch_out_of_place(h3) -> None:
    """把 `_mod_gate` 换成非 in-place 版本。

    原实现 `x[a:b].addcmul_(...)` 会就地改 block 输入；配 gradient checkpoint 时
    该输入是被保存的张量，反向重算会报 "modified by an inplace operation"。
    数学完全等价，只是多一次 clone（S×hidden 的瞬时拷贝）。
    """
    current = getattr(h3, "_mod_gate", None)
    if current is None or getattr(current, "_train_oop", False):
        return

    def _mod_gate_oop(x, gate, other, segments):
        out = x.clone()
        for a, b, row in segments:
            out[a:b].addcmul_(other[a:b], h3._mod_row(gate, row, x.dtype))
        return out

    _mod_gate_oop._train_oop = True
    h3._mod_gate = _mod_gate_oop


def patch_attention(dm, cfg: dict):
    for blk in dm.blocks:
        blk.attn.forward = types.MethodType(_make_attn_forward(int(blk.attn.head_dim), cfg["attn_temp"]), blk.attn)


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def scan_cache(cache_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for path in sorted(cache_dir.glob("*.pt")):
        try:
            rec = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[data] skip {path.name}: {exc}", flush=True)
            continue
        cond = rec["cond"][0]
        meta = dict(rec["meta"])
        meta["text_len"] = int(cond[0].shape[1])
        try:
            room = {"path": path, "meta": meta,
                    "refs": cond[1].get("minimax_refs"),
                    "keyframes": cond[1].get("minimax_keyframes"),
                    "tags": cond[1].get("minimax_token_tags")}
        except Exception:
            continue
        key = f"{meta['latent_t']}x{meta['lat_h']}x{meta['lat_w']}a{meta['audio_t']}|" + repr(_refs_key(cond[1] or {}))
        groups.setdefault(key, []).append(room)
    return groups


def build_batch(rec: dict, group: Group, sigma: float, sigma_a: float, device, gen):
    data = torch.load(rec["path"], map_location="cpu", weights_only=False)
    z_video = data["video"].to(device=device, dtype=torch.bfloat16)
    z_audio = data["audio"].to(device=device, dtype=torch.bfloat16)
    z_up = data["z_up"].to(device=device, dtype=torch.bfloat16)
    cond_text, cond_meta = data["cond"][0]

    real_len = int(cond_text.shape[1])
    delta = group.text_pad - real_len
    # padding 行复制真实末 token（全 0 行会污染 RMSNorm/注意力统计），dtype 一步到 bf16
    text = torch.empty(1, group.text_pad, cond_text.shape[-1], dtype=torch.bfloat16)
    text[0, :real_len] = cond_text[0].to(torch.bfloat16)
    if group.text_pad > real_len:
        text[0, real_len:] = cond_text[0, -1].to(torch.bfloat16)
    tags = torch.ones(group.text_pad, dtype=torch.long)
    if rec["tags"] is not None:
        tags[:real_len] = rec["tags"][:real_len].to(torch.long)
    keep = torch.ones(group.S, dtype=torch.bool)
    keep[real_len:group.text_pad] = False

    pos = group.layout.position_ids.clone()
    if delta:
        pos[group.text_pad:, 0] -= float(delta)

    payload = {}
    if rec["tags"] is not None:
        payload["text_token_tags"] = tags.to(device)
    refs = rec["refs"]
    kf = rec["keyframes"]
    if kf:
        payload["keyframes"] = kf
    if refs:
        payload["refs"] = refs
    payload["cond_video_latents"] = [k["latent"] for k in (kf or []) if k.get("latent") is not None] + \
                                    [r["latent"] for r in (refs or []) if r.get("latent") is not None]
    aud = [r["audio_latent"] for r in (refs or []) if r.get("audio_latent") is not None]
    if aud:
        payload["cond_audio_latents"] = aud
    payload["seed"] = int(rec["meta"].get("seed") or 0)
    payload["audio_scale"] = 1.0
    payload["layout"] = group.layout

    eps_v = torch.randn(z_up.shape, generator=gen, device=device, dtype=torch.float32)
    eps_a = torch.randn(z_audio.shape, generator=gen, device=device, dtype=torch.float32)
    x_v = ((1.0 - sigma) * z_up.float() + sigma * eps_v).to(torch.bfloat16)
    x_a = ((1.0 - sigma_a) * z_audio.float() + sigma_a * eps_a).to(torch.bfloat16)
    tgt_v = ((x_v.float() - z_video.float()) / sigma)
    tgt_a = ((x_a.float() - z_audio.float()) / max(sigma_a, 1e-6))
    return {
        "x": [x_v, x_a],
        "tgt": [tgt_v, tgt_a],
        "text": text.to(device),
        "payload": payload,
        "keep": keep.to(device),
        "pos": pos,
    }


def sanity_symbol(dm, h3, groups, block: int, neighbors: int, device, gen) -> None:
    """LoRA 全零时底模输出应与教师目标同向。符号错了立刻中止，别烧 GPU。"""
    key = next(iter(groups))
    group = groups[key]
    rec = group.records[0]
    sigma = 0.5
    sigma_a = h3.time_shift_sigma(sigma, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else sigma
    b = build_batch(rec, group, sigma, sigma_a, device, gen)
    layout = group.layout
    old_pos = layout.position_ids
    layout.position_ids = b["pos"]
    try:
        ctx = dm.preprocess_text_embeds(b["text"])
        to = {"_h3_plan": group, "_h3_keep_kv": b["keep"]}
        with torch.no_grad():
            out = dm(b["x"], torch.tensor([sigma * 1000.0], device=device), ctx,
                     transformer_options=to, minimax_payload=b["payload"])
    finally:
        layout.position_ids = old_pos
    pv = out[0].float().flatten()
    tgt = b["tgt"][0].float().flatten()
    cos = float((pv * tgt).sum() / (pv.norm() * tgt.norm() + 1e-12))
    mse = float(((pv - tgt) ** 2).mean())
    print(f"[sanity] sigma=0.5 cos={cos:+.4f} mse={mse:.4f} "
          f"std(out)={float(pv.std()):.3f} std(tgt)={float(tgt.std()):.3f}", flush=True)
    if not (cos > 0.7 and mse < 2.0):
        raise SystemExit(f"[sanity] 失败：cos={cos:.4f} mse={mse:.4f}，符号或目标构造有误")


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sigma-min", type=float, default=0.12)
    ap.add_argument("--sigma-max", type=float, default=0.85)
    ap.add_argument("--sigmas", default="0.85,0.54,0.20")
    ap.add_argument("--anchor-prob", type=float, default=0.0,
                    help="Phase A 收尾锚点混采概率（推荐 0.30：30%% 采 --sigmas 真实 Pass-2 锚点 + 70%% 连续区间）")
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    ap.add_argument("--topk-min", type=float, default=0.15)
    ap.add_argument("--topk-max", type=float, default=0.30)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--neighbors", type=int, default=2)
    ap.add_argument("--audio-weight", type=float, default=0.1)
    ap.add_argument("--lf-base-anchor", action="store_true",
                    help="高低频解耦去油：低频 L_k(v_base) 100%% 锚定冻结基座、高频对齐教师（零额外峰值显存）")
    ap.add_argument("--lf-kernel", type=int, default=5,
                    help="高低频解耦 2D 空间低通核大小（奇数，默认 5 对应约 80×80 px 光影包络）")
    ap.add_argument("--dense", action="store_true", help="关闭稀疏（稠密 SDPA 对照）")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--optim", choices=["auto", "adamw", "adamw8bit"], default="auto")
    ap.add_argument("--blocks", default="", help="只训这些 block（如 25-49 / 0,3,5）；留空=全部")
    ap.add_argument("--grad-clamp", type=float, default=0.0,
                    help="每层 block 边界梯度裁剪阈值；0=关闭（这个模型 bf16 下需要）")
    ap.add_argument("--auto-resume", action="store_true",
                    help="本地无 resume/latest_state.pt 时自动从 HF 拉取")
    ap.add_argument("--pull-shards", action="store_true", help="先拉云端教师分片到 --cache")
    ap.add_argument("--shard-prefix", default="teacher_cache_v1")
    ap.add_argument("--shard-work", default="/content/work/shard_dl")
    ap.add_argument("--frames", default="", help="只训这些帧数桶，如 22,39")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-prefix", default="lora_run")
    ap.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-lora", default="",
                    help="热启动 LoRA 权重文件（.pt 续训包或 .safetensors）；仅载入权重、step 从 0 重新开始（用于 Phase A -> Phase B）")
    ap.add_argument("--init-lora-hf", default="",
                    help="云端热启动路径（如 lora_run/resume/latest_state.pt）；仅在当前目录尚无 resume 时生效")
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    mm, comfy_sd, h3 = bootstrap(args.comfy)
    patch_out_of_place(h3)
    dm = load_dm(comfy_sd, args.base, device)
    params = attach_lora(dm, args.rank, args.blocks)
    patch_attention(dm, {"attn_temp": args.attn_temp})
    cfg = {
        "rope_hf": args.rope_hf,
        "attn_temp": args.attn_temp,
        "sparse": not args.dense,
        "topk": args.topk_min,
        "topk_max": args.topk_max,
        "ckpt": not args.no_ckpt,
        "grad_clamp": args.grad_clamp,
    }
    patch_run_blocks(dm, h3, cfg)
    if args.lf_kernel > 1 and args.lf_kernel % 2 == 0:
        args.lf_kernel += 1
    print(f"[init] lora params {sum(p.numel() for p in params) / 1e6:.1f}M rank={args.rank}", flush=True)
    if args.lf_base_anchor:
        print(f"[init] lf-base-anchor ENABLED (kernel={args.lf_kernel}, clean target = L_{args.lf_kernel}(v_base) + HP_{args.lf_kernel}(v_teacher))", flush=True)

    if args.pull_shards and args.hf_repo:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hf_shards import ShardStore

        ShardStore(args.hf_repo, args.shard_prefix, args.token_file).pull_all(args.cache, args.shard_work)

    groups = scan_cache(Path(args.cache))
    if args.frames:
        allowed = {int(x) for x in args.frames.split(",") if x.strip()}
        groups = {k: v for k, v in groups.items() if int(v[0]["meta"]["frames"]) in allowed}
    if not groups:
        raise SystemExit("teacher cache 为空或帧数过滤后无样本")
    plan_groups = {}
    for key, recs in groups.items():
        plan_groups[key] = _build_group(h3, key, recs, args.block_size, args.neighbors)
    for key, g in plan_groups.items():
        print(f"[bucket] {key.split('|')[0]} S={g.S} text_pad={g.text_pad} video=[{g.va},{g.vb}) "
              f"n={len(g.records)}", flush=True)

    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "resume").mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "logs" / "metrics.jsonl"
    resume_path = out_dir / "resume" / "latest_state.pt"

    # ---------------- 优化器（AdamW8bit 可选） ----------------
    opt = None
    if args.optim in ("auto", "adamw8bit"):
        try:
            import bitsandbytes as bnb
        except Exception:
            if args.optim == "adamw8bit":
                print("[init] bitsandbytes 缺失，尝试 pip install ...", flush=True)
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
                    import bitsandbytes as bnb  # noqa: F811
                except Exception as exc:
                    print(f"[init] bnb 安装失败，回退 torch AdamW: {exc}", flush=True)
                    bnb = None
            else:
                bnb = None
        if bnb is not None:
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
            print("[init] optimizer = AdamW8bit", flush=True)
    if opt is None:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        print("[init] optimizer = torch AdamW", flush=True)

    def _lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_scale)

    start_step = 0
    resume_file = Path(args.resume) if args.resume else (resume_path if resume_path.is_file() else None)
    if args.auto_resume and resume_file is None and args.hf_repo:
        try:
            import shutil

            from huggingface_hub import hf_hub_download

            if str(Path(__file__).resolve().parent) not in sys.path:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
            from hf_shards import read_token

            got = hf_hub_download(args.hf_repo, f"{args.hf_prefix}/resume/latest_state.pt",
                                  token=read_token(args.token_file))
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, resume_path)
            resume_file = resume_path
            print(f"[init] 云端续训包已拉取 -> {resume_path}", flush=True)
        except Exception as exc:
            print(f"[init] 云端无续训包（从头训练）: {exc}", flush=True)

    if resume_file is not None and Path(resume_file).is_file():
        ck = torch.load(resume_file, map_location=device, weights_only=False)
        load_lora_state(dm, ck["lora"])
        if ck.get("optim") is not None:
            try:
                opt.load_state_dict(ck["optim"])
            except Exception as exc:
                print(f"[init] optimizer state 载入失败（继续用新状态）: {exc}", flush=True)
        if ck.get("sched") is not None:
            try:
                sched.load_state_dict(ck["sched"])
            except Exception as exc:
                print(f"[init] scheduler state 载入失败: {exc}", flush=True)
        start_step = int(ck.get("step", 0))
        rng = ck.get("rng") or {}
        try:
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu())
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
            if "python" in rng:
                random.setstate(rng["python"])
            if "gen" in rng:
                gen.set_state(rng["gen"].cpu())
        except Exception as exc:
            print(f"[init] rng 恢复失败: {exc}", flush=True)
        print(f"[init] resumed step {start_step} from {resume_file}", flush=True)
    elif args.init_lora or (args.init_lora_hf and args.hf_repo):
        init_path = Path(args.init_lora) if args.init_lora else None
        if (init_path is None or not init_path.is_file()) and args.init_lora_hf and args.hf_repo:
            from huggingface_hub import hf_hub_download

            if str(Path(__file__).resolve().parent) not in sys.path:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
            from hf_shards import read_token

            init_path = Path(hf_hub_download(args.hf_repo, args.init_lora_hf, token=read_token(args.token_file)))
            print(f"[init] 云端热启动权重已拉取 -> {args.init_lora_hf}", flush=True)
        if init_path is None or not init_path.is_file():
            raise SystemExit(f"[init] 热启动权重不存在: {args.init_lora or args.init_lora_hf}")
        if init_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(str(init_path), device="cpu")
            lora_dict = {}
            for k, v in sd.items():
                if k.startswith("diffusion_model.") and k.endswith(".lora_down.weight"):
                    base_k = k[len("diffusion_model."):-len(".lora_down.weight")]
                    up_k = f"diffusion_model.{base_k}.lora_up.weight"
                    if up_k in sd:
                        lora_dict[base_k] = {"A": v, "B": sd[up_k]}
            load_lora_state(dm, lora_dict)
            print(f"[init] warm-start ({len(lora_dict)} layers) from safetensors {init_path.name} (step=0)", flush=True)
        else:
            ck = torch.load(init_path, map_location=device, weights_only=False)
            load_lora_state(dm, ck["lora"])
            print(f"[init] warm-start from {init_path.name} (src_step={ck.get('step')}, new step=0)", flush=True)

    uploader = None
    if args.hf_repo:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hf_sync import AsyncUploader

        uploader = AsyncUploader(args.hf_repo, token_file=args.token_file,
                                 log_path=str(out_dir / "logs" / "hf_sync.log"))

    def _atomic_save(obj, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def _export_safetensors(lora: dict, path: Path) -> None:
        from safetensors.torch import save_file

        sd = {}
        for key, val in lora.items():
            down = val["A"].to(torch.bfloat16).contiguous()
            up = val["B"].to(torch.bfloat16).contiguous()
            alpha = float(val.get("scale", 1.0)) * down.shape[0]
            prefix = f"diffusion_model.{key}"
            sd[f"{prefix}.lora_down.weight"] = down
            sd[f"{prefix}.lora_up.weight"] = up
            sd[f"{prefix}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(sd, str(tmp), metadata={"format": "pt"})
        tmp.replace(path)

    def save_everything(step: int) -> None:
        lora = lora_state(dm)
        state = {
            "step": step,
            "lora": lora,
            "optim": opt.state_dict(),
            "sched": sched.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "gen": gen.get_state(),
            },
            "metrics": metrics_path.read_text(encoding="utf-8") if metrics_path.is_file() else "",
            "config": {k: v for k, v in vars(args).items()},
        }
        _atomic_save(state, resume_path)
        step_safe = out_dir / "steps" / f"step_{step:06d}.safetensors"
        step_safe.parent.mkdir(parents=True, exist_ok=True)
        _export_safetensors(lora, step_safe)
        print(f"[ckpt] step {step} -> {resume_path.name} + {step_safe.name}", flush=True)
        if uploader is not None:
            uploader.submit(step_safe, f"{args.hf_prefix}/steps/{step_safe.name}")
            uploader.submit(resume_path, f"{args.hf_prefix}/resume/latest_state.pt")
            uploader.submit(metrics_path, f"{args.hf_prefix}/logs/metrics.jsonl")

    sanity_symbol(dm, h3, plan_groups, args.block_size, args.neighbors, device, gen)

    keys = list(plan_groups)
    key_weights = [max(1, len(plan_groups[k].records)) for k in keys]
    sigmas_fixed = [float(x) for x in args.sigmas.split(",") if x.strip()]
    nan_steps = 0
    t_start = time.time()

    for step in range(start_step, args.steps):
        # 前 len(keys) 步轮询每个桶一次确保所有静态图编译/显存探顶，之后按桶内样本数加权抽样避免小桶过拟合
        if step - start_step < len(keys):
            cur_key = keys[(step - start_step) % len(keys)]
        else:
            cur_key = random.choices(keys, weights=key_weights, k=1)[0]
        group = plan_groups[cur_key]
        rec = random.choice(group.records)
        if args.phase == "B" and sigmas_fixed:
            sigma = random.choice(sigmas_fixed)
        elif args.anchor_prob > 0.0 and sigmas_fixed and random.random() < args.anchor_prob:
            sigma = random.choice(sigmas_fixed)
        else:
            sigma = random.uniform(args.sigma_min, args.sigma_max)
        sigma_a = h3.time_shift_sigma(sigma, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else sigma
        batch = build_batch(rec, group, sigma, sigma_a, device, gen)

        # padding 文本按样本屏蔽；pos 覆盖写回 layout（用完恢复）
        layout = group.layout
        old_pos = layout.position_ids
        layout.position_ids = batch["pos"]
        base_v = None
        try:
            with torch.no_grad():
                context = dm.preprocess_text_embeds(batch["text"])
            cfg["topk"] = random.uniform(args.topk_min, args.topk_max)
            t_vec = torch.tensor([sigma * 1000.0], device=device)
            saved_bm = None
            if args.lf_base_anchor:
                with _disable_lora(), torch.no_grad():
                    to_base = {"_h3_plan": group, "_h3_keep_kv": batch["keep"], "_h3_save_bm": True}
                    out_base = dm(batch["x"], t_vec, context,
                                  transformer_options=to_base, minimax_payload=batch["payload"])
                    base_v = out_base[0].detach().float()
                    saved_bm = to_base.get("_h3_saved_bm")
                    del out_base
            to = {"_h3_plan": group, "_h3_keep_kv": batch["keep"]}
            if saved_bm is not None:
                to["_h3_prebuilt_bm"] = saved_bm
            out = dm(batch["x"], t_vec, context,
                     transformer_options=to, minimax_payload=batch["payload"])
        finally:
            layout.position_ids = old_pos

        # dm() 返回的就是 +v（已实测：cos(out,tgt)>0.7），不要再取负
        pred_v, pred_a = out[0], out[1]
        pred_vf = pred_v.float()
        tgt_v = batch["tgt"][0]
        loss_lf_val = None
        loss_hf_val = None
        if args.lf_base_anchor and base_v is not None:
            lf_base = _spatial_lowpass(base_v, args.lf_kernel)
            lf_tgt = _spatial_lowpass(tgt_v, args.lf_kernel)
            hf_tgt = tgt_v - lf_tgt
            tgt_v_clean = lf_base + hf_tgt
            loss_v = F.mse_loss(pred_vf, tgt_v_clean)
            with torch.no_grad():
                lf_pred = _spatial_lowpass(pred_vf.detach(), args.lf_kernel)
                loss_lf_val = float(F.mse_loss(lf_pred, lf_base))
                loss_hf_val = float(F.mse_loss(pred_vf.detach() - lf_pred, hf_tgt))
            del base_v, lf_base, lf_tgt, hf_tgt, tgt_v_clean
        else:
            loss_v = F.mse_loss(pred_vf, tgt_v)
        loss_a = F.mse_loss(pred_a.float(), batch["tgt"][1])
        loss = loss_v + args.audio_weight * loss_a

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step in (start_step, start_step + 1):
            n_nz = sum(
                1 for p in params
                if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.detach().float().abs().max()) > 0.0
            )
            min_need = (len(params) // 2) if step == start_step else len(params)
            print(f"[grad-check] step {step + 1} nonzero finite grad params: {n_nz}/{len(params)} (need >= {min_need})", flush=True)
            if n_nz < min_need:
                raise SystemExit(
                    f"[grad-check] 失败：step {step + 1} 仅 {n_nz}/{len(params)} 参数拿到非零有限梯度！"
                    "（严禁空烧 GPU，检查 checkpoint 闭包绑定或梯度截断）"
                )
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            nan_steps += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            if nan_steps <= 5 or nan_steps % 20 == 0:
                print(f"[warn] step {step + 1} 梯度非有限（累计 {nan_steps}），已跳过更新", flush=True)
            continue
        opt.step()
        sched.step()

        if (step + 1) % args.log_every == 0:
            rec_line = {
                "step": step + 1,
                "loss": round(float(loss), 5),
                "loss_v": round(float(loss_v), 5),
                "loss_a": round(float(loss_a), 5),
                "sigma": round(sigma, 4),
                "sigma_a": round(sigma_a, 4),
                "topk": round(float(cfg["topk"]), 3),
                "bucket": cur_key.split("|")[0],
                "lr": f"{sched.get_last_lr()[0]:.2e}",
                "gn": round(float(gn), 4),
                "sec": round((time.time() - t_start) / max(1, step + 1 - start_step), 2),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }
            if loss_lf_val is not None:
                rec_line["loss_lf"] = round(loss_lf_val, 5)
                rec_line["loss_hf"] = round(loss_hf_val, 5)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
            print(json.dumps(rec_line, ensure_ascii=False), flush=True)
            if uploader is not None:
                uploader.submit(metrics_path, f"{args.hf_prefix}/logs/metrics.jsonl")

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save_everything(step + 1)

    if uploader is not None:
        uploader.submit(metrics_path, f"{args.hf_prefix}/logs/metrics.jsonl")
        uploader.close(wait=True)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
