# SPDX-License-Identifier: GPL-3.0-or-later
"""MiniMax-H3 自蒸馏 LoRA 训练器（v3 Action-Sharp：兼容 AMD MI300X ROCm & NVIDIA A100/G4）。

v3 核心升级
-----------
1. **异构秩与无损扩秩热启动**（`--rank-attn 64 --rank-mlp 32`）：
   从 v2 的 `r=32` 权重热启动时，前 32 秩保持 `step 0` 逐位等价，新增 32 个正交 Attention 秩
   （`B_extra=0, A_extra=Kaiming`）在第 1 步立即激活，翻倍跨帧 `<Picture 1/2>` 五官检索容量。
2. **运动门控高低频解耦（Motion-Gated Scheme 2A，`--motion-gate-tau`）+ 空间梯度损失（`--grad-xy-weight`）**：
   静止背景区 100% 锁死冻结基座低频 `L_k(v_base)` 保持零油光；高速大动作区（`M_mot -> 1`）
   自动放开中低频向满血教师 `v_teacher` 对齐以重建五官/肢体轮廓，配合 `∇_xy` L1 损失根治色块涂抹。
3. **推理同构 `64`-token `2d_frame` Morton 块路由 + 双后端（`--attn-backend auto|flex|sdpa_mask`）**：
   视频段重排为与 4060 `chunked_sage.py` 完全同构的 `8×8` 二维 Morton 块（含帧内 2D 邻域 + 邻帧运动锥 +
   块质心余弦路由）；AMD ROCm 自动切 `sdpa_mask`（单张层间共享布尔掩码走 AOTriton/CK Flash SDPA，
   彻底规避 `gfx942` 64KB LDS 溢出与重编译风暴），并修复 checkpoint 反向重算掩码丢失问题。
4. **无缝多日接力续训**：`--auto-resume` 自动同步新 `--steps` 余弦终点与 `--lr`，支持多分片前缀
   `--shard-prefix a,b` 与动作样本加权抽样 `--action-boost`。
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

LORA_TARGETS = ("qkv_proj", "out_proj", "fc1", "fc2")
ATTN_TARGETS = ("qkv_proj", "out_proj")
DEFAULT_TOKEN_FILE = "/content/work/hftoken.txt"


def _is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


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
# LoRA（支持 attn / mlp 异构 rank 与无损扩秩热启动）
# ----------------------------------------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, out_f: int, in_f: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.rank = int(rank)
        self.A = nn.Parameter(torch.empty(self.rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = float(self.rank if alpha is None else alpha) / float(self.rank)

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


def _motion_gate_mask(z_vid: torch.Tensor, tau: float) -> torch.Tensor:
    """计算时空自适应运动强度掩码 M_mot ∈ [0, 1]，形状 [B, 1, T, H, W]。

    采用双向一阶差分均值 0.5*(|z_{t+1}-z_t| + |z_t-z_{t-1}|)，确保折返动作顶点帧不漏检。
    """
    if tau <= 0.0 or z_vid.ndim != 5 or z_vid.shape[2] < 2:
        return torch.zeros(
            (z_vid.shape[0], 1, z_vid.shape[2], z_vid.shape[3], z_vid.shape[4]),
            device=z_vid.device,
            dtype=torch.float32,
        )
    zf = z_vid.detach().float()
    t = int(zf.shape[2])
    idx_prev = [max(0, i - 1) for i in range(t)]
    idx_next = [min(t - 1, i + 1) for i in range(t)]
    diff = 0.5 * ((zf[:, :, idx_next] - zf).abs() + (zf - zf[:, :, idx_prev]).abs())
    mot = diff.mean(dim=1, keepdim=True)  # [B, 1, T, H, W]
    b, _, _, h, w = mot.shape
    mot2d = F.pad(mot.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w), (1, 1, 1, 1), mode="replicate")
    mot2d = F.avg_pool2d(mot2d, kernel_size=3, stride=1)
    mot = mot2d.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
    return (mot / float(tau)).clamp_(0.0, 1.0)


def _spatial_grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """潜空间水平/垂直一阶差分 L1 损失，惩罚大动作轮廓软化。"""
    dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    return 0.5 * (F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t))


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


def attach_lora(
    dm,
    rank: int,
    blocks_spec: str = "",
    rank_attn: int = 0,
    rank_mlp: int = 0,
) -> list[nn.Parameter]:
    r_attn = int(rank_attn) if int(rank_attn) > 0 else int(rank)
    r_mlp = int(rank_mlp) if int(rank_mlp) > 0 else int(rank)
    params: list[nn.Parameter] = []
    for i in _parse_blocks(blocks_spec, len(dm.blocks)):
        blk = dm.blocks[i]
        for name in LORA_TARGETS:
            is_attn = name in ATTN_TARGETS
            lin = getattr(blk.attn if is_attn else blk.mlp, name)
            cur_r = r_attn if is_attn else r_mlp
            lora = LoRALayer(lin.weight.shape[0], lin.weight.shape[1], cur_r).to(lin.weight.device)
            lin._train_lora = lora
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
    """恒等前向 + 保方向 L2 裁剪反向：放在两个 block 之间截断指数级放大的回传。"""

    @staticmethod
    def forward(ctx, x, limit):
        ctx.limit = limit
        return x

    @staticmethod
    def backward(ctx, grad):
        norm = grad.float().reshape(-1).norm()
        if float(norm) > float(ctx.limit):
            grad = grad * (float(ctx.limit) / (float(norm) + 1e-12))
        return grad, None


def lora_state(dm) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for i, blk in enumerate(dm.blocks):
        for name in LORA_TARGETS:
            owner = blk.attn if name in ATTN_TARGETS else blk.mlp
            lin = getattr(owner, name)
            lora = getattr(lin, "_train_lora", None)
            if lora is not None:
                out[f"blocks.{i}.{'attn' if name in ATTN_TARGETS else 'mlp'}.{name}"] = {
                    "A": lora.A.detach().cpu(),
                    "B": lora.B.detach().cpu(),
                    "scale": float(lora.scale),
                }
    return out


def load_lora_state(dm, state: dict) -> tuple[int, int]:
    """加载 LoRA 权重；当目标层 rank > 源权重 rank 时执行逐位等价无损扩秩热启动。"""
    exact_cnt = 0
    expanded_cnt = 0
    for key, val in state.items():
        parts = key.split(".")
        blk = dm.blocks[int(parts[1])]
        owner = blk.attn if parts[2] == "attn" else blk.mlp
        lora = getattr(getattr(owner, parts[3]), "_train_lora", None)
        if lora is None:
            continue
        a_src = val["A"].to(device=lora.A.device, dtype=torch.float32)
        b_src = val["B"].to(device=lora.B.device, dtype=torch.float32)
        src_scale = float(val.get("scale", 1.0))
        r_old = int(a_src.shape[0])
        r_new = int(lora.A.shape[0])
        if r_old == r_new:
            lora.A.data.copy_(a_src.to(lora.A.dtype))
            lora.B.data.copy_(b_src.to(lora.B.dtype))
            lora.scale = src_scale
            exact_cnt += 1
        elif r_old < r_new:
            # 无损扩秩：前 r_old 秩吸收 src_scale 后与原模型 step=0 100% 等价；
            # 后 (r_new - r_old) 秩保持 A=Kaiming、B=0，step=1 反向立即获得非零梯度！
            lora.A.data[:r_old].copy_((a_src * src_scale).to(lora.A.dtype))
            lora.B.data.zero_()
            lora.B.data[:, :r_old].copy_(b_src.to(lora.B.dtype))
            lora.scale = 1.0
            expanded_cnt += 1
        else:
            # 截断降秩兜底（SVD 取前 r_new 主奇异方向）
            u, s_vals, vh = torch.linalg.svd((b_src @ a_src) * src_scale, full_matrices=False)
            s_root = torch.sqrt(s_vals[:r_new])
            lora.B.data.copy_((u[:, :r_new] * s_root[None, :]).to(lora.B.dtype))
            lora.A.data.copy_((s_root[:, None] * vh[:r_new, :]).to(lora.A.dtype))
            lora.scale = 1.0
            exact_cnt += 1
    return exact_cnt, expanded_cnt


# ----------------------------------------------------------------------------
# 2d_frame Morton 置换（与 4060 推理端 chunked_sage.py 完全对齐）
# ----------------------------------------------------------------------------
def _part1by1(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.int64) & 0x0000FFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def _morton_2d_frame_perm(lat_t: int, hp: int, wp: int, va: int, block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 `(perm_cpu, inv_cpu, coords_zyx)`，其中 coords_zyx 按重排后顺序记录每个视频 token 的 (t, y, x)。"""
    n_vid = lat_t * hp * wp
    linear = torch.arange(n_vid, dtype=torch.int64)
    area = hp * wp
    z = linear // area
    rem = linear - z * area
    y = rem // wp
    x = rem - y * wp
    code = (z << 42) | _part1by1(x) | (_part1by1(y) << 1)
    perm = linear[torch.argsort(code)]
    pad = (-int(va)) % int(block)
    if pad and n_vid > 0:
        perm = torch.roll(perm, pad)
    inv = torch.argsort(perm)
    coords = torch.stack([z[perm], y[perm], x[perm]], dim=-1).float()
    return perm, inv, coords


def _permute_bhsd_video(t: torch.Tensor, va: int, vb: int, idx: torch.Tensor) -> torch.Tensor:
    """对 [B, H, S, D] 的视频区间 [va:vb] 执行可微序列重排（out-of-place，兼容 autograd 与 checkpoint）。"""
    parts = []
    if va > 0:
        parts.append(t[:, :, :va, :])
    parts.append(t[:, :, va:vb, :].index_select(2, idx))
    if vb < t.shape[2]:
        parts.append(t[:, :, vb:, :])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)


# ----------------------------------------------------------------------------
# 注意力后端（flex_attention / ROCm sdpa_mask / 稠密 SDPA）
# ----------------------------------------------------------------------------
_FLEX = {"fn": None, "ok": None}
_KEEP_GPU = None


def _keep_mod(b, h, q_idx, kv_idx):
    """模块级 mask_mod：只读固定地址的全局 buffer，避免 dynamo 每步重编译。"""
    return _KEEP_GPU[kv_idx]


def _flex_fn():
    if _FLEX["ok"] is None:
        from torch.nn.attention.flex_attention import flex_attention

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


def resolve_attn_backend(spec: str) -> str:
    s = (spec or "auto").strip().lower()
    if s in ("flex", "sdpa_mask"):
        return s
    return "sdpa_mask" if _is_rocm() else "flex"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _rope_chunked(q: torch.Tensor, table: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """FP32 旋转、BF16 存储；按 token 分块避免 73f 时 2×S×H×D fp32 瞬时峰值。"""
    half = table.shape[-3]
    rot = half * 2
    if rot <= 0 or rot > q.shape[-1]:
        return q
    s = q.shape[0]
    c_all = table[..., 0, 0][0]
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
        q = _rms_norm(q, self.q_norm.weight, float(self.q_norm.eps) if self.q_norm.eps is not None else 1e-6)
        k = _rms_norm(k, self.k_norm.weight, float(self.k_norm.eps) if self.k_norm.eps is not None else 1e-6)
        if rope_freqs is not None:
            q = _rope_chunked(q, rope_freqs)
            k = _rope_chunked(k, rope_freqs)

        q = q.transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        scale = scale_base * float(to.get("_h3_attn_temp", attn_temp))

        m_info = to.get("_h3_morton")
        if m_info is not None:
            m_va, m_vb, m_perm, m_inv = m_info
            q = _permute_bhsd_video(q, m_va, m_vb, m_perm)
            k = _permute_bhsd_video(k, m_va, m_vb, m_perm)
            v = _permute_bhsd_video(v, m_va, m_vb, m_perm)

        bm = to.get("_h3_block_mask")
        sdpa_mask = to.get("_h3_sdpa_mask")
        if bm is not None:
            try:
                out = _flex_fn()(q, k, v, block_mask=bm, scale=scale)
            except Exception as exc:
                raise RuntimeError(f"flex_attention 执行失败（可加 --attn-backend sdpa_mask 切换）: {exc}") from exc
        elif sdpa_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False, scale=scale
            )
        else:
            keep_kv = to.get("_h3_keep_kv")
            attn_mask = keep_kv.view(1, 1, 1, -1) if keep_kv is not None else None
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=scale
            )

        if m_info is not None:
            out = _permute_bhsd_video(out, m_va, m_vb, m_inv)

        out = out.squeeze(0).transpose(0, 1).reshape(s, inner)
        return _lora_linear(self.out_proj, out)

    return forward


# ----------------------------------------------------------------------------
# 分桶 / 2d_frame Morton 块稀疏路由
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
    base: torch.Tensor         # [NQB, NKV] bool (CPU)
    dense: torch.Tensor        # [NQB] bool
    video: torch.Tensor        # [NKV] bool
    tpf: int
    text_pad: int
    morton_perm: torch.Tensor  # [vb - va] int64 (CPU)
    morton_inv: torch.Tensor   # [vb - va] int64 (CPU)
    records: list = field(default_factory=list)
    _dev_cache: dict = field(default_factory=dict)

    def morton_on(self, device: torch.device) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        k = str(device)
        hit = self._dev_cache.get(k)
        if hit is None:
            hit = (self.va, self.vb, self.morton_perm.to(device), self.morton_inv.to(device))
            self._dev_cache[k] = hit
        return hit


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
    hp, wp = int(meta["lat_h"]) // 2, int(meta["lat_w"]) // 2
    lat_t = int(meta["latent_t"])
    tpf = hp * wp

    morton_perm, morton_inv, coords = _morton_2d_frame_perm(lat_t, hp, wp, va, block)

    base = torch.zeros(NQB, NKV, dtype=torch.bool)
    dense = torch.zeros(NQB, dtype=torch.bool)
    video = torch.zeros(NKV, dtype=torch.bool)
    prefix = (va + block - 1) // block

    pure_vid_blocks: list[int] = []
    vid_centers: list[torch.Tensor] = []
    for b in range(NQB):
        qs, qe = b * block, min((b + 1) * block, S)
        if qs >= va and qe <= vb:
            video[b] = True
            base[b, :prefix] = True
            base[b, b] = True
            pure_vid_blocks.append(b)
            vid_centers.append(coords[qs - va:qe - va].mean(dim=0))
        else:
            dense[b] = True
            base[b, :] = True

    if pure_vid_blocks:
        vb_idx = torch.tensor(pure_vid_blocks, dtype=torch.long)
        ctrs = torch.stack(vid_centers, dim=0)  # [M, 3] = (t, y, x)
        dt = (ctrs[:, None, 0] - ctrs[None, :, 0]).abs()
        dy = (ctrs[:, None, 1] - ctrs[None, :, 1]).abs()
        dx = (ctrs[:, None, 2] - ctrs[None, :, 2]).abs()
        d_xy = torch.maximum(dy, dx)
        # 同帧 2D 邻域（±12 patch ≈ ±384px）+ 邻帧运动锥（|dt| <= neighbors 且 ±10 patch ≈ ±320px）
        same_frame = (dt <= 0.6) & (d_xy <= 12.5)
        adj_frame = (dt > 0.6) & (dt <= float(neighbors) + 0.55) & (d_xy <= 10.5)
        local_keep = same_frame | adj_frame
        # 保底保留紧邻块索引 ±2
        bi = torch.arange(len(pure_vid_blocks))
        local_keep |= (bi[:, None] - bi[None, :]).abs() <= 2
        base[vb_idx[:, None], vb_idx[None, :]] |= local_keep

    return Group(
        key=key, layout=layout, block=block, S=S, va=va, vb=vb, NQB=NQB, NKV=NKV,
        base=base, dense=dense, video=video, tpf=tpf, text_pad=text_pad,
        morton_perm=morton_perm, morton_inv=morton_inv, records=records,
    )


def _select_sparse_blocks(
    g: Group,
    topk_ratio: float,
    topk_max_ratio: float,
    device: torch.device,
    h_ref: torch.Tensor | None = None,
    jitter: float = 0.05,
) -> torch.Tensor:
    """在 Morton 置换后的块空间内计算 [NQB, NKV] bool 块选择矩阵。"""
    sel = g.base.to(device).clone()
    n_video = int(g.video.sum())
    k = 0 if topk_ratio <= 0 else max(1, int(round(topk_ratio * n_video)))
    k = min(k, max(1, int(round(topk_max_ratio * n_video))), max(1, g.NKV - 1))
    if k > 0:
        pool = (g.video.to(device)[None, :] & ~sel)
        if h_ref is not None and h_ref.shape[0] >= g.NQB * g.block:
            with torch.no_grad():
                _, _, m_perm, _ = g.morton_on(device)
                h_m = h_ref.detach()
                if g.vb <= h_m.shape[0]:
                    parts = []
                    if g.va > 0:
                        parts.append(h_m[:g.va])
                    parts.append(h_m[g.va:g.vb].index_select(0, m_perm))
                    if g.vb < h_m.shape[0]:
                        parts.append(h_m[g.vb:])
                    h_m = torch.cat(parts, dim=0)
                hb = h_m[:g.NQB * g.block].view(g.NQB, g.block, -1).float().mean(dim=1)
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
    return sel


def build_sparse_mask(
    g: Group,
    topk_ratio: float,
    topk_max_ratio: float,
    keep: torch.Tensor,
    device: torch.device,
    backend: str = "flex",
    h_ref: torch.Tensor | None = None,
    jitter: float = 0.05,
):
    """按 backend 生成 `BlockMask`（CUDA flex）或 `[1, 1, S, S]` bool 掩码（ROCm/通用 sdpa_mask）。"""
    sel = _select_sparse_blocks(g, topk_ratio, topk_max_ratio, device, h_ref=h_ref, jitter=jitter)

    # 注意：keep 的 padding 文本位全部落在 `< va`，不受视频区间 Morton 重排影响
    if backend == "sdpa_mask":
        with torch.no_grad():
            tok_mask = sel.repeat_interleave(g.block, dim=0).repeat_interleave(g.block, dim=1)
            tok_mask = tok_mask[:g.S, :g.S] & keep[None, :g.S]
            tok_mask.diagonal().fill_(True)
        return tok_mask.unsqueeze(0).unsqueeze(0)

    from torch.nn.attention.flex_attention import BlockMask

    kv_num = sel.sum(1).to(torch.int32)
    order = torch.argsort(sel.to(torch.int8), dim=1, descending=True, stable=True).to(torch.int32)

    global _KEEP_GPU
    if _KEEP_GPU is None or _KEEP_GPU.numel() < g.S or _KEEP_GPU.device != device:
        _KEEP_GPU = torch.ones(max(g.S, 65536), dtype=torch.bool, device=device)
    _KEEP_GPU[:g.S].copy_(keep[:g.S])

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
# _run_blocks：Split-Band RoPE + 分块 checkpoint（修复反向重算掩码保持）
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

        plan: Group | None = transformer_options.get("_h3_plan")
        keep = transformer_options.get("_h3_keep_kv")
        backend = resolve_attn_backend(cfg.get("attn_backend", "auto"))

        # 使用独立的局部 options 副本，确保 checkpoint 反向重算时掩码 100% 在位（绝不退回无掩码稠密！）
        step_to = dict(transformer_options)
        step_to["_h3_attn_temp"] = cfg["attn_temp"]

        prebuilt = transformer_options.get("_h3_prebuilt_mask")
        if plan is not None and cfg["sparse"]:
            step_to["_h3_morton"] = plan.morton_on(device)
            if prebuilt is not None:
                mask_obj = prebuilt
            elif keep is not None:
                mask_obj = build_sparse_mask(
                    plan, cfg["topk"], cfg["topk_max"], keep, device,
                    backend=backend, h_ref=h, jitter=float(cfg.get("mask_jitter", 0.05)),
                )
                if transformer_options.get("_h3_save_bm"):
                    transformer_options["_h3_saved_bm"] = mask_obj
            else:
                mask_obj = None

            if mask_obj is not None:
                if backend == "sdpa_mask":
                    step_to["_h3_sdpa_mask"] = mask_obj
                else:
                    step_to["_h3_block_mask"] = mask_obj

        grad_on = torch.is_grad_enabled()
        use_clamp = cfg.get("grad_clamp", 0) > 0 and grad_on
        use_ckpt = cfg["ckpt"] and grad_on

        for i, block in enumerate(self.blocks):
            if use_clamp:
                h = _GradClamp.apply(h, float(cfg["grad_clamp"]))
            if use_ckpt:
                if not h.requires_grad:
                    h = h.detach().requires_grad_(True)

                def _call(x, _i=i, _blk=block, _to=step_to):
                    local_to = dict(_to)
                    local_to["block_index"] = _i
                    return _blk(x, t_emb, mod_segments, rope_freqs, transformer_options=local_to)

                h = torch.utils.checkpoint.checkpoint(_call, h, use_reentrant=True)
            else:
                step_to["block_index"] = i
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=step_to)
        return h

    dm._run_blocks = types.MethodType(run_blocks, dm)


def patch_out_of_place(h3) -> None:
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
# 数据扫描与动作样本加权
# ----------------------------------------------------------------------------
def _is_action_sample(stem: str) -> bool:
    s = stem.lower()
    return s.startswith(("sakuga", "act_", "combat", "action"))


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
            room = {
                "path": path,
                "stem": path.stem,
                "is_action": _is_action_sample(path.stem),
                "meta": meta,
                "refs": cond[1].get("minimax_refs"),
                "keyframes": cond[1].get("minimax_keyframes"),
                "tags": cond[1].get("minimax_token_tags"),
            }
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
    cond_text, _ = data["cond"][0]

    real_len = int(cond_text.shape[1])
    delta = group.text_pad - real_len
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
        "z_video": z_video,
        "text": text.to(device),
        "payload": payload,
        "keep": keep.to(device),
        "pos": pos,
    }


def sanity_symbol(dm, h3, groups, device, gen) -> None:
    """训练前自检：验证基座输出与教师目标同向（防止符号/RoPE/Morton 错位白烧 GPU）。"""
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
        with _disable_lora(), torch.no_grad():
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
    if not (cos > 0.7 and mse < 2.5):
        raise SystemExit(f"[sanity] 失败：cos={cos:.4f} mse={mse:.4f}，符号或目标构造有误")


# ----------------------------------------------------------------------------
# 主训练入口
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="/content/ComfyUI")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--rank-attn", type=int, default=0, help="Attention 层独立 rank（0=沿用 --rank；v3 推荐 64）")
    ap.add_argument("--rank-mlp", type=int, default=0, help="MLP 层独立 rank（0=沿用 --rank；v3 推荐 32）")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sigma-min", type=float, default=0.14)
    ap.add_argument("--sigma-max", type=float, default=0.75)
    ap.add_argument("--sigmas", default="0.75,0.54,0.20")
    ap.add_argument("--anchor-prob", type=float, default=0.35,
                    help="锚点混采概率（推荐 0.35：35%% 采 --sigmas 关键锚点 + 65%% 连续区间）")
    ap.add_argument("--rope-hf", type=float, default=1.32)
    ap.add_argument("--attn-temp", type=float, default=1.08)
    ap.add_argument("--topk-min", type=float, default=0.15)
    ap.add_argument("--topk-max", type=float, default=0.30)
    ap.add_argument("--block-size", type=int, default=64, help="稀疏块大小（默认 64，对齐 4060 Sol-Attn 8x8 Morton 块）")
    ap.add_argument("--neighbors", type=int, default=2)
    ap.add_argument("--attn-backend", choices=["auto", "flex", "sdpa_mask"], default="auto",
                    help="稀疏注意力训练后端（auto 在 AMD ROCm 下自动选 sdpa_mask，CUDA 下选 flex）")
    ap.add_argument("--audio-weight", type=float, default=0.1)
    ap.add_argument("--lf-base-anchor", action="store_true",
                    help="开启高低频解耦去油目标（配合 --motion-gate-tau 实现静处去油、动处重塑轮廓）")
    ap.add_argument("--lf-kernel", type=int, default=5)
    ap.add_argument("--motion-gate-tau", type=float, default=0.35,
                    help="运动门控阈值（>0 时在高速运动区放开低频向满血教师对齐；0=退回 v2 全图锁低频）")
    ap.add_argument("--grad-xy-weight", type=float, default=0.15,
                    help="潜空间一阶空间梯度 L1 损失权重（抑制大动作边缘发糊；0=关闭）")
    ap.add_argument("--action-boost", type=float, default=2.5,
                    help="混训时对 sakuga*/act_* 高动态样本的抽样倍率加权（默认 2.5）")
    ap.add_argument("--dense", action="store_true", help="关闭稀疏（稠密 SDPA 对照）")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--optim", choices=["auto", "adamw", "adamw8bit"], default="auto")
    ap.add_argument("--blocks", default="", help="只训这些 block（留空=全部 50 层）")
    ap.add_argument("--grad-clamp", type=float, default=10.0,
                    help="每层 block 边界保方向 L2 梯度裁剪阈值（H3 bf16 推荐 10.0）")
    ap.add_argument("--auto-resume", action="store_true",
                    help="本地无 resume/latest_state.pt 时自动从 HF 拉取并支持跨日接力扩步数")
    ap.add_argument("--pull-shards", action="store_true", help="先拉云端教师分片到 --cache")
    ap.add_argument("--shard-prefix", default="teacher_cache_v1",
                    help="支持逗号分隔多个分片前缀，如 teacher_cache_v1,teacher_cache_v3_action")
    ap.add_argument("--shard-work", default="/tmp/h3_shard_dl")
    ap.add_argument("--frames", default="", help="只训这些帧数桶，如 22,39,73")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-prefix", default="lora_run_v3")
    ap.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-lora", default="",
                    help="热启动 LoRA 权重（支持从 r=32 无损扩秩到 r=64）；仅当无续训包时生效")
    ap.add_argument("--init-lora-hf", default="",
                    help="云端热启动路径（如 final/minimax_h3_sol_sharp_v2_final.safetensors）")
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    eff_backend = resolve_attn_backend(args.attn_backend)
    r_attn = args.rank_attn if args.rank_attn > 0 else args.rank
    r_mlp = args.rank_mlp if args.rank_mlp > 0 else args.rank

    mm, comfy_sd, h3 = bootstrap(args.comfy)
    patch_out_of_place(h3)
    dm = load_dm(comfy_sd, args.base, device)
    params = attach_lora(dm, args.rank, args.blocks, rank_attn=r_attn, rank_mlp=r_mlp)
    patch_attention(dm, {"attn_temp": args.attn_temp})
    cfg = {
        "rope_hf": args.rope_hf,
        "attn_temp": args.attn_temp,
        "sparse": not args.dense,
        "attn_backend": eff_backend,
        "topk": args.topk_min,
        "topk_max": args.topk_max,
        "ckpt": not args.no_ckpt,
        "grad_clamp": args.grad_clamp,
    }
    patch_run_blocks(dm, h3, cfg)
    if args.lf_kernel > 1 and args.lf_kernel % 2 == 0:
        args.lf_kernel += 1
    print(
        f"[init] backend={eff_backend} rocm={_is_rocm()} block_size={args.block_size} "
        f"lora_params={sum(p.numel() for p in params) / 1e6:.1f}M rank_attn={r_attn} rank_mlp={r_mlp}",
        flush=True,
    )
    if args.lf_base_anchor:
        print(
            f"[init] lf-base-anchor ENABLED (kernel={args.lf_kernel}, motion_gate_tau={args.motion_gate_tau}, "
            f"grad_xy_weight={args.grad_xy_weight})",
            flush=True,
        )

    if args.pull_shards and args.hf_repo:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hf_shards import ShardStore

        for sp in [p.strip() for p in args.shard_prefix.split(",") if p.strip()]:
            print(f"[shards] pulling prefix={sp} -> {args.cache}", flush=True)
            ShardStore(args.hf_repo, sp, args.token_file).pull_all(args.cache, f"{args.shard_work}_{sp}")

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
        n_act = sum(1 for r in g.records if r.get("is_action"))
        print(f"[bucket] {key.split('|')[0]} S={g.S} text_pad={g.text_pad} video=[{g.va},{g.vb}) "
              f"n={len(g.records)} (action={n_act})", flush=True)

    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "resume").mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "logs" / "metrics.jsonl"
    resume_path = out_dir / "resume" / "latest_state.pt"

    # ---------------- 优化器（ROCm 下自动选原生 AdamW 避开 bitsandbytes） ----------------
    opt = None
    want_bnb = args.optim == "adamw8bit" or (args.optim == "auto" and not _is_rocm())
    if want_bnb:
        try:
            import bitsandbytes as bnb
        except Exception:
            if args.optim == "adamw8bit":
                print("[init] bitsandbytes 缺失，尝试安装 ...", flush=True)
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
                    import bitsandbytes as bnb  # noqa: F811
                except Exception as exc:
                    print(f"[init] bnb 不可用，回退原生 torch AdamW: {exc}", flush=True)
                    bnb = None
            else:
                bnb = None
        if bnb is not None:
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
            print("[init] optimizer = AdamW8bit", flush=True)
    if opt is None:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        print("[init] optimizer = torch AdamW (FP32 state)", flush=True)

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
            print(f"[init] 云端尚无续训包（将走热启动或从头训练）: {exc}", flush=True)

    if resume_file is not None and Path(resume_file).is_file():
        ck = torch.load(resume_file, map_location=device, weights_only=False)
        ex_c, exp_c = load_lora_state(dm, ck["lora"])
        if ck.get("optim") is not None and exp_c == 0:
            try:
                opt.load_state_dict(ck["optim"])
            except Exception as exc:
                print(f"[init] optimizer state 载入跳过（使用新动量）: {exc}", flush=True)
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
            print(f"[init] rng 恢复跳过: {exc}", flush=True)
        print(f"[init] resumed step {start_step} -> target {args.steps} (exact={ex_c}, expanded={exp_c})", flush=True)
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
                    alpha_k = f"diffusion_model.{base_k}.alpha"
                    if up_k in sd:
                        r_src = int(v.shape[0])
                        a_val = float(sd[alpha_k].item()) if alpha_k in sd else float(r_src)
                        lora_dict[base_k] = {"A": v, "B": sd[up_k], "scale": a_val / float(r_src)}
            ex_c, exp_c = load_lora_state(dm, lora_dict)
            print(f"[init] warm-start from {init_path.name}: exact={ex_c} rank_expanded={exp_c} (step=0)", flush=True)
        else:
            ck = torch.load(init_path, map_location=device, weights_only=False)
            ex_c, exp_c = load_lora_state(dm, ck["lora"])
            print(f"[init] warm-start from {init_path.name}: exact={ex_c} rank_expanded={exp_c} (step=0)", flush=True)

    # 强制同步最新 --lr 与最新总终点 --steps（彻底根治跨日接力续训时旧 scheduler 覆写新 lr/终点问题！）
    for pg in opt.param_groups:
        pg["lr"] = float(args.lr)
        pg["initial_lr"] = float(args.lr)

    def _lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_scale, last_epoch=start_step - 1)

    if start_step >= args.steps:
        print(f"[done] start_step={start_step} 已达到或超过 --steps={args.steps}（若继续接力请调大 --steps）", flush=True)
        return 0

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

    def _export_safetensors(lora: dict, path: Path, cur_step: int) -> None:
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
        save_file(sd, str(tmp), metadata={
            "format": "pt",
            "step": str(cur_step),
            "rank_attn": str(r_attn),
            "rank_mlp": str(r_mlp),
        })
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
            "config": {k: v for k, v in vars(args).items()},
        }
        _atomic_save(state, resume_path)
        step_safe = out_dir / "steps" / f"step_{step:06d}.safetensors"
        step_safe.parent.mkdir(parents=True, exist_ok=True)
        _export_safetensors(lora, step_safe, step)
        print(f"[ckpt] step {step} -> {resume_path.name} + {step_safe.name}", flush=True)
        if uploader is not None:
            uploader.submit(step_safe, f"{args.hf_prefix}/steps/{step_safe.name}")
            uploader.submit(resume_path, f"{args.hf_prefix}/resume/latest_state.pt")
            uploader.submit(metrics_path, f"{args.hf_prefix}/logs/metrics.jsonl")

    sanity_symbol(dm, h3, plan_groups, device, gen)

    # 按样本数 × 动作加权系数构建桶权重与桶内样本权重
    keys = list(plan_groups)
    rec_weights_by_key: dict[str, list[float]] = {}
    key_weights: list[float] = []
    for k in keys:
        rw = [float(args.action_boost) if r.get("is_action") else 1.0 for r in plan_groups[k].records]
        rec_weights_by_key[k] = rw
        key_weights.append(max(1e-6, sum(rw)))

    sigmas_fixed = [float(x) for x in args.sigmas.split(",") if x.strip()]
    nan_steps = 0
    t_start = time.time()

    for step in range(start_step, args.steps):
        if step - start_step < len(keys):
            cur_key = keys[(step - start_step) % len(keys)]
        else:
            cur_key = random.choices(keys, weights=key_weights, k=1)[0]
        group = plan_groups[cur_key]
        rec = random.choices(group.records, weights=rec_weights_by_key[cur_key], k=1)[0]

        if args.phase == "B" and sigmas_fixed:
            sigma = random.choice(sigmas_fixed)
        elif args.anchor_prob > 0.0 and sigmas_fixed and random.random() < args.anchor_prob:
            sigma = random.choice(sigmas_fixed)
        else:
            sigma = random.uniform(args.sigma_min, args.sigma_max)
        sigma_a = h3.time_shift_sigma(sigma, 12.0, 3.0) if hasattr(h3, "time_shift_sigma") else sigma
        batch = build_batch(rec, group, sigma, sigma_a, device, gen)

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
                to["_h3_prebuilt_mask"] = saved_bm
            out = dm(batch["x"], t_vec, context,
                     transformer_options=to, minimax_payload=batch["payload"])
        finally:
            layout.position_ids = old_pos

        pred_v, pred_a = out[0], out[1]
        pred_vf = pred_v.float()
        tgt_v = batch["tgt"][0]
        loss_lf_val = None
        loss_hf_val = None
        loss_grad_val = None
        mot_mean_val = None

        if args.lf_base_anchor and base_v is not None:
            lf_base = _spatial_lowpass(base_v, args.lf_kernel)
            lf_tgt = _spatial_lowpass(tgt_v, args.lf_kernel)
            hf_tgt = tgt_v - lf_tgt
            tgt_static_clean = lf_base + hf_tgt
            if args.motion_gate_tau > 0.0:
                m_mot = _motion_gate_mask(batch["z_video"], args.motion_gate_tau)
                mot_mean_val = float(m_mot.mean().item())
                tgt_ref = (1.0 - m_mot) * tgt_static_clean + m_mot * tgt_v
            else:
                tgt_ref = tgt_static_clean
            loss_mse = F.mse_loss(pred_vf, tgt_ref)
            with torch.no_grad():
                lf_pred = _spatial_lowpass(pred_vf.detach(), args.lf_kernel)
                loss_lf_val = float(F.mse_loss(lf_pred, lf_base))
                loss_hf_val = float(F.mse_loss(pred_vf.detach() - lf_pred, hf_tgt))
            del base_v, lf_base, lf_tgt, hf_tgt, tgt_static_clean
        else:
            tgt_ref = tgt_v
            loss_mse = F.mse_loss(pred_vf, tgt_ref)

        if args.grad_xy_weight > 0.0:
            loss_grad = _spatial_grad_l1(pred_vf, tgt_ref)
            loss_grad_val = float(loss_grad.detach().item())
            loss_v = loss_mse + float(args.grad_xy_weight) * loss_grad
        else:
            loss_v = loss_mse

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
                "topk": round(float(cfg["topk"]), 3),
                "bucket": cur_key.split("|")[0],
                "sample": rec["stem"],
                "lr": f"{sched.get_last_lr()[0]:.2e}",
                "gn": round(float(gn), 4),
                "sec": round((time.time() - t_start) / max(1, step + 1 - start_step), 2),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }
            if loss_lf_val is not None:
                rec_line["loss_lf"] = round(loss_lf_val, 5)
                rec_line["loss_hf"] = round(loss_hf_val, 5)
            if loss_grad_val is not None:
                rec_line["loss_grad_xy"] = round(loss_grad_val, 5)
            if mot_mean_val is not None:
                rec_line["mot_gate"] = round(mot_mean_val, 3)
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
    print("[done] 训练完成！若在魔搭免费实例运行，验收上传完成后请务必点击控制台「停止」防空扣时长！", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
