# -*- coding: utf-8 -*-
"""E-1: denoise_mask 时间维可行性 — CPU 级验证（不加载模型权重、不占 GPU 成片）。

问题（SAMPLER_RESEARCH_20260923.md §五）：构造 (1,1,T,H,W) 时间维掩码
（锁两端、放中间），沿真实 comfy 代码链走一遍，锁区是否真的不动？

验证链（comfy 代码全部真实，仅模型本体用可解析 mock 代替）：

  T1  comfy.utils.reshape_mask: (1,1,T,H,W) -> (1,24,T,H,W)
      时间维零插值（T 相等时），锁区 0/1 精确保持。
  T2  guider.sample 同款掩码预备（samplers.py:1297-1314 逐行复刻）：
      prepare_mask -> pack_latents，形状与 packed AV latent 完全一致。
  T3  MiniMaxH3._token_grid_masks（真实方法绑定 fake 实例）：
      2x2 空间 patch 池化（amax）后时间维完整保留，不坍缩。
  T4  KSamplerX0Inpaint + 真实 sample_euler + 真实
      MiniMaxH3.scale_latent_inpaint：
      a) 模型每步看到的锁区输入 == 0.999*anchor + 0.001*noise（注入锚）
      b) Euler 轨迹上锁区 == anchor + sigma*noise（keyframe 同款轨迹）
      c) 最终输出锁区 == anchor 精确相等（fp32 epsilon 内）
      d) 中段自由演化（既不等于 anchor 也不等于起步 x0）
  T5  audio_scale != 1 的 AV 分支不炸（真实 time_shift_sigma 参与）。

推导（T4c）：Euler 更新 x' = x + (x-a)/sigma * (sigma'-sigma)，
末步 sigma'=0 时 x' = x - (x-a) = a —— 锁区精确收敛到锚，
与 mock 模型说什么无关。
"""
import sys
import types

import torch

sys.path.insert(0, r"d:\comfyui\ComfyUI")

import comfy.utils
import comfy.samplers
import comfy.sampler_helpers
import comfy.k_diffusion.sampling as kds
from comfy.model_base import MiniMaxH3
from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP

torch.manual_seed(1234)

T, H, W = 20, 32, 64
A = 13
VIDEO_SHAPE = (1, 24, T, H, W)
AUDIO_SHAPE = (1, 32, 2, A)
LOCK_HEAD, LOCK_TAIL = 4, 4
SIGMAS = torch.tensor([8.0, 5.0, 3.0, 1.5, 0.6, 0.0])

EPS = 1e-4


def build_temporal_mask(lock_head=LOCK_HEAD, lock_tail=LOCK_TAIL):
    m = torch.ones(1, 1, T, H, W)
    m[:, :, :lock_head] = 0.0
    m[:, :, T - lock_tail:] = 0.0
    return m


class FakeBase:
    """MiniMaxH3 的可解析替身：latent_shapes / patch_size / model_sampling
    可控，方法本体全部绑定真实 MiniMaxH3 实现。"""

    def __init__(self, audio_scale=1.0):
        self.latent_shapes = [VIDEO_SHAPE, AUDIO_SHAPE]
        self.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))
        self.model_sampling = types.SimpleNamespace(
            shift=1.0, audio_shift=3.0, audio_scale=audio_scale
        )

    _pool_masks_to_token_grid = MiniMaxH3._pool_masks_to_token_grid
    _token_grid_masks = MiniMaxH3._token_grid_masks
    scale_latent_inpaint = MiniMaxH3.scale_latent_inpaint

    def audio_scale(self):
        if self.latent_shapes is None or len(self.latent_shapes) < 2:
            return 1.0
        return self.model_sampling.audio_scale


class FakeWrap:
    """KSamplerX0Inpaint 期望的 model_wrap：本身可调用（相当于 Guider 的
    __call__），且 .inner_model 持有 scale_latent_inpaint（相当于 base model）。"""

    def __init__(self, base, fn):
        self.inner_model = base
        self._fn = fn

    def __call__(self, x, sigma, model_options={}, seed=None):
        return self._fn(x, sigma)


def t1():
    m = build_temporal_mask()
    out = comfy.utils.reshape_mask(m, VIDEO_SHAPE)
    assert tuple(out.shape) == VIDEO_SHAPE, f"shape {tuple(out.shape)}"
    assert torch.equal(out[:, :, :LOCK_HEAD], torch.zeros_like(out[:, :, :LOCK_HEAD]))
    assert torch.equal(out[:, :, -LOCK_TAIL:], torch.zeros_like(out[:, :, -LOCK_TAIL:]))
    mid = out[:, :, LOCK_HEAD:T - LOCK_TAIL]
    assert torch.equal(mid, torch.ones_like(mid))
    print("[T1] reshape_mask (1,1,T,H,W)->(1,24,T,H,W) 时间维零插值、锁区精确保持")


def t2():
    vm = build_temporal_mask()
    am = torch.ones(1, 1, 2, A)
    vmp = comfy.sampler_helpers.prepare_mask(vm, VIDEO_SHAPE, "cpu")
    amp = comfy.sampler_helpers.prepare_mask(am, AUDIO_SHAPE, "cpu")
    packed_mask, _ = comfy.utils.pack_latents([vmp, amp])

    video = torch.randn(*VIDEO_SHAPE)
    audio = torch.randn(*AUDIO_SHAPE)
    packed_latent, shapes = comfy.utils.pack_latents([video, audio])
    assert shapes == [VIDEO_SHAPE, AUDIO_SHAPE]
    assert tuple(packed_mask.shape) == tuple(packed_latent.shape), (
        f"mask {tuple(packed_mask.shape)} vs latent {tuple(packed_latent.shape)}"
    )
    # pack 是 reshape(B,1,-1)+cat，元素级对齐由相同 shape 保证
    back = comfy.utils.unpack_latents(packed_mask, shapes)
    assert torch.equal(back[0], vmp) and torch.equal(back[1], amp)
    print("[T2] guider.sample 掩码预备路径：prepare_mask+pack 后与 packed latent 逐元素一致")


def t3():
    base = FakeBase()
    vm = build_temporal_mask()
    vmp = comfy.utils.reshape_mask(vm, VIDEO_SHAPE)
    amp = torch.ones(*AUDIO_SHAPE)
    packed, _ = comfy.utils.pack_latents([vmp, amp])
    grids = base._token_grid_masks(packed, base.latent_shapes)
    gv = grids[0]
    assert tuple(gv.shape) == VIDEO_SHAPE, f"token grid shape {tuple(gv.shape)}"
    assert torch.equal(gv[:, :, :LOCK_HEAD], torch.zeros_like(gv[:, :, :LOCK_HEAD]))
    assert torch.equal(gv[:, :, -LOCK_TAIL:], torch.zeros_like(gv[:, :, -LOCK_TAIL:]))
    assert torch.equal(gv[:, :, LOCK_HEAD:T - LOCK_TAIL], torch.ones_like(gv[:, :, LOCK_HEAD:T - LOCK_TAIL]))
    print("[T3] MiniMaxH3._token_grid_masks：2x2 patch 池化后时间维完整保留（池化只做空间）")


def run_euler(audio_scale=1.0):
    video = torch.randn(*VIDEO_SHAPE)
    audio = torch.randn(*AUDIO_SHAPE)
    nvideo = torch.randn(*VIDEO_SHAPE)
    naudio = torch.randn(*AUDIO_SHAPE)

    vm = build_temporal_mask()
    am = torch.ones(*AUDIO_SHAPE)
    vmp = comfy.sampler_helpers.prepare_mask(vm, VIDEO_SHAPE, "cpu")
    packed_mask, _ = comfy.utils.pack_latents([vmp, am])
    packed_latent, shapes = comfy.utils.pack_latents([video, audio])
    packed_noise, _ = comfy.utils.pack_latents([nvideo, naudio])

    base = FakeBase(audio_scale=audio_scale)
    seen = []       # (x, sigma) 模型实际收到的
    traj = []       # euler 轨迹上的 x（callback 记录）

    def model_fn(x, sigma):
        seen.append((x.detach().clone(), float(sigma.reshape(-1)[0])))
        return torch.full_like(x, 0.0)  # “模型”坚持把一切去噪到 0

    def cb(state):
        traj.append((state["x"].detach().clone(), float(state["sigma"])))

    model_k = comfy.samplers.KSamplerX0Inpaint(FakeWrap(base, model_fn), SIGMAS)
    model_k.latent_image = packed_latent
    model_k.noise = packed_noise

    x0 = packed_latent + packed_noise * SIGMAS[0]
    out = kds.sample_euler(
        model_k, x0, SIGMAS,
        extra_args={"denoise_mask": packed_mask},
        callback=cb, disable=True,
    )
    return video, nvideo, seen, traj, out, shapes


def t4():
    video, nvideo, seen, traj, out, shapes = run_euler(audio_scale=1.0)

    # a) 模型每步看到的锁区输入 == 注入锚（0.999a + 0.001n）
    for x, sigma in seen:
        xv = comfy.utils.unpack_latents(x, shapes)[0]
        injected = VISUAL_COND_TIMESTEP * video + (1 - VISUAL_COND_TIMESTEP) * nvideo
        err = (xv[:, :, :LOCK_HEAD] - injected[:, :, :LOCK_HEAD]).abs().max().item()
        assert err < EPS, f"注入锚检查失败: max err {err:.3e}"
        err = (xv[:, :, -LOCK_TAIL:] - injected[:, :, -LOCK_TAIL:]).abs().max().item()
        assert err < EPS, f"注入锚检查失败(tail): max err {err:.3e}"
    print("[T4a] 模型每步收到的锁区 == 0.999*anchor + 0.001*noise（H3 键帧注入语义）")

    # b) Euler 轨迹锁区 == anchor + sigma*noise
    for x, sigma in traj:
        xv = comfy.utils.unpack_latents(x, shapes)[0]
        expect = video + nvideo * sigma
        err = (xv[:, :, :LOCK_HEAD] - expect[:, :, :LOCK_HEAD]).abs().max().item()
        assert err < EPS, f"轨迹检查失败: max err {err:.3e}"
    print("[T4b] Euler 轨迹锁区 == anchor + sigma*noise（键帧同款噪声轨迹）")

    # c) 最终输出锁区 == anchor 精确
    ov = comfy.utils.unpack_latents(out, shapes)[0]
    err_head = (ov[:, :, :LOCK_HEAD] - video[:, :, :LOCK_HEAD]).abs().max().item()
    err_tail = (ov[:, :, -LOCK_TAIL:] - video[:, :, -LOCK_TAIL:]).abs().max().item()
    assert err_head < EPS and err_tail < EPS, f"锁区未收敛: head {err_head:.3e} tail {err_tail:.3e}"
    print(f"[T4c] 最终输出锁区 == anchor（head err {err_head:.2e}, tail err {err_tail:.2e}）")

    # d) 中段自由演化：不等于 anchor，也不等于起步点
    mid = ov[:, :, LOCK_HEAD:T - LOCK_TAIL]
    mid_anchor = video[:, :, LOCK_HEAD:T - LOCK_TAIL]
    assert not torch.allclose(mid, mid_anchor), "中段被意外锁死"
    x0_mid = (video + nvideo * SIGMAS[0])[:, :, LOCK_HEAD:T - LOCK_TAIL]
    assert not torch.allclose(mid, x0_mid), "中段没有演化（模型输出未生效）"
    d_anchor = (mid - mid_anchor).abs().max().item()
    print(f"[T4d] 中段自由演化（与锚最大差 {d_anchor:.3f}，由 mock 模型动力学决定）")


def t5():
    video, nvideo, seen, traj, out, shapes = run_euler(audio_scale=2.0)
    ov = comfy.utils.unpack_latents(out, shapes)[0]
    err = (ov[:, :, :LOCK_HEAD] - video[:, :, :LOCK_HEAD]).abs().max().item()
    assert err < EPS, f"audio_scale=2 下锁区漂移: {err:.3e}"
    print(f"[T5] audio_scale=2.0（AV 双时钟分支）下锁区仍精确收敛（err {err:.2e}）")


if __name__ == "__main__":
    t1()
    t2()
    t3()
    t4()
    t5()
    print()
    print("E-1 结论：时间维 (1,1,T,H,W) 掩码在真实 comfy 链路上完全可用——")
    print("  锁区逐步注入 0.999 锚 + Euler 轨迹即键帧噪声轨迹 + 最终精确收敛到锚，")
    print("  即 RePaint 式逐步锚定，无需任何上游改动。")
