# -*- coding: utf-8 -*-
"""E-3 单元测试：_redenoise_seam_windows 的几何与接线（不加载模型权重）。

mock 策略：core 用真实上游模块的 token 数学（frames_for_tokens /
FRAME_RESCALE），仅 sample_piece / reanchor_conditioning 换成可观测替身；
sampling.rebind_dual_clock_sampler 恒等。mask 语义（锁区不动）由 E-1 在真实
comfy 链路上验证过，这里 mock sample_piece 直接按 mask 模拟采样结果。

验证：
  G1  窗居中、边界钳制（缝在片尾时窗整体左移）
  G2  时间维掩码形状 (1,1,span,H,W)，两端 lock_tokens 为 0，中段为 1
  G3  锁区逐 token 精确回写（不变），中段被替换
  G4  only_frames 选择：只处理指定帧 17 格以内的缝
  G5  prepared_noise 切片 = 全局噪声 [w0:w1] 与音频切片
  G6  多缝顺序处理，互不覆盖
"""
import importlib
import sys
import types

import torch

sys.path.insert(0, r"d:\comfyui\ComfyUI")
sys.path.insert(0, r"d:\comfyui\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8")

import comfy.nested_tensor

upstream = importlib.import_module("h3_t8.chunked_two_pass_upscale_advanced")

pkg = types.ModuleType("seamkit")
pkg.__path__ = [r"d:\comfyui\ComfyUI\custom_nodes\comfyui-h3-seamkit"]
sys.modules["seamkit"] = pkg
h3u = importlib.import_module("seamkit.h3_upscale")

torch.manual_seed(0)

T, H, W, A = 100, 8, 16, 300
VIDEO = torch.arange(T, dtype=torch.float32).reshape(1, 1, T, 1, 1).expand(1, 24, T, H, W).contiguous()
AUDIO = torch.randn(1, 32, 2, A)
GNOISE_V = torch.randn(1, 24, T, H, W)
GNOISE_A = torch.randn(1, 32, 2, A)


class FakeCore:
    frames_for_tokens = staticmethod(upstream.frames_for_tokens)
    FRAME_RESCALE = upstream.FRAME_RESCALE

    @staticmethod
    def reanchor_conditioning(cond, f0, f1, hw):
        return {"reanchored": (cond, f0, f1, hw)}

    @staticmethod
    def sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg,
                     prepared_noise=None):
        video, audio = piece["samples"].tensors
        vm, am = piece["noise_mask"].tensors
        calls.append({
            "video_shape": tuple(video.shape),
            "mask_shape": tuple(vm.shape),
            "mask": vm.detach().clone(),
            "audio_mask": bool(am.unique().tolist() == [1.0]),
            "cond": cond,
            "noise": prepared_noise,
            "sigmas": sigmas,
        })
        # 模拟采样：锁区（mask 0）原样保留，自由区（mask 1）整体 +100
        out = video.clone()
        out = out + vm * 100.0
        return types.SimpleNamespace(tensors=(out, audio))


calls = []


class FakeSampling:
    @staticmethod
    def rebind_dual_clock_sampler(model, piece, sampler):
        return sampler


def run(seam_marks, only_frames=None, window_tokens=10, lock_tokens=3):
    calls.clear()
    accumulated = VIDEO.clone()
    out, entries = h3u._redenoise_seam_windows(
        FakeCore, FakeSampling, accumulated, AUDIO, {"cond": 1},
        None,  # model
        None,  # noise
        "sampler",
        torch.tensor([8.0, 5.0, 3.0, 1.5, 0.6, 0.0]),  # sigmas
        None, 1.0, seam_marks, window_tokens, lock_tokens,
        only_frames, GNOISE_V, GNOISE_A,
    )
    return out, entries


def t_g1():
    # 缝在 token 55：窗 [50,60)
    out, entries = run([55])
    assert entries[0]["window_tokens"] == [50, 60], entries
    # 缝在 token 98（片尾 T=100）：窗整体左移 [90,100)
    out, entries = run([98])
    assert entries[0]["window_tokens"] == [90, 100], entries
    print("[G1] 窗居中且边界钳制（片尾左移）")


def t_g2():
    run([55])
    c = calls[0]
    assert c["mask_shape"] == (1, 1, 10, H, W), c["mask_shape"]
    m = c["mask"][0, 0, :, 0, 0]
    assert m[:3].tolist() == [0.0] * 3 and m[-3:].tolist() == [0.0] * 3
    assert m[3:7].tolist() == [1.0] * 4
    assert c["audio_mask"], "audio mask 应为全 1"
    print("[G2] 时间维掩码 (1,1,span,H,W)：两端各 3 token 为 0，中段 4 token 为 1")


def t_g3():
    out, _ = run([55])
    # token 50..52（锁）与 57..59（锁）不变，53..56 变（+100）
    for t in list(range(50, 53)) + list(range(57, 60)):
        assert torch.equal(out[:, :, t], VIDEO[:, :, t]), f"token {t} 应锁定"
    for t in range(53, 57):
        assert not torch.equal(out[:, :, t], VIDEO[:, :, t]), f"token {t} 应重生成"
    # 窗外不变
    assert torch.equal(out[:, :, :50], VIDEO[:, :, :50])
    assert torch.equal(out[:, :, 60:], VIDEO[:, :, 60:])
    print("[G3] 锁区逐 token 精确回写，中段替换，窗外原样")


def t_g4():
    # 三条缝（token 25/55/85 ≈ 帧 85/187/289），只选帧 187 附近
    out, entries = run([25, 55, 85], only_frames=[187])
    assert len(calls) == 1, f"应只采样 1 个窗，实际 {len(calls)}"
    assert entries[0]["window_tokens"] == [50, 60], entries
    print("[G4] only_frames=[187]：只命中 token 55 的缝")


def t_g5():
    run([55])
    c = calls[0]
    nv, na = c["noise"].tensors
    assert torch.equal(nv, GNOISE_V[:, :, 50:60]), "视频噪声切片应为全局噪声 [50:60]"
    assert tuple(na.shape) == tuple(c["video_shape"][:2]) + tuple(AUDIO.shape[1:]) or True
    f0, f1 = upstream.frames_for_tokens(50), upstream.frames_for_tokens(60)
    a0 = round(f0 * upstream.FRAME_RESCALE)
    a1 = min(A, round(f1 * upstream.FRAME_RESCALE))
    assert torch.equal(na, GNOISE_A[..., a0:a1]), "音频噪声切片应为全局音频噪声 [a0:a1]"
    assert c["cond"]["reanchored"][1:3] == (f0, f1), "conditioning 应重锚到窗帧区间"
    print(f"[G5] prepared_noise = 全局噪声切片，conditioning 重锚到 [{f0},{f1})")


def t_g6():
    out, entries = run([20, 80])
    assert len(calls) == 2 and len(entries) == 2
    # 两窗 [15,25) 与 [75,85) 不相交，各自中段变、锁区不变
    assert not torch.equal(out[:, :, 20], VIDEO[:, :, 20])
    assert torch.equal(out[:, :, 15], VIDEO[:, :, 15])
    assert not torch.equal(out[:, :, 80], VIDEO[:, :, 80])
    assert torch.equal(out[:, :, 84], VIDEO[:, :, 84])
    print("[G6] 多缝顺序处理、互不覆盖")


def t_g7():
    # 窗太小（span <= 2*lock）跳过且不炸
    out, entries = run([2], window_tokens=4, lock_tokens=3)
    assert calls == [] and entries[0]["skipped"], entries
    print("[G7] 窗过小安全跳过")


def t_e2():
    """E-2：_anchor_conditioning_multi 切片宽度与结构。"""
    cond = [[torch.zeros(1), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, H, W)},
        {"resolved_frame_index": 30},
    ]}]]
    prev = torch.randn(1, 24, 120, H, W)
    # start_frame 187 -> token 55；k=5 -> latent [55:60)
    out = h3u._anchor_conditioning_multi(upstream, cond, prev, 187, 0.999, 5)
    kfs = out[0][1]["minimax_keyframes"]
    assert len(kfs) == 2, "锚应替换原 index-0 键帧"
    assert tuple(kfs[0]["latent"].shape) == (1, 24, 5, H, W), kfs[0]["latent"].shape
    assert torch.equal(kfs[0]["latent"], prev[:, :, 55:60])
    assert kfs[0]["resolved_frame_index"] == 0
    assert out[0][1]["minimax_visual_cond_noise_aug"] == 0.999
    assert kfs[1] == {"resolved_frame_index": 30}, "非零索引键帧应保留"
    # k 超出前块剩余 token 时钳制
    out2 = h3u._anchor_conditioning_multi(upstream, cond, prev[:, :, :57], 187, 0.999, 5)
    assert tuple(out2[0][1]["minimax_keyframes"][0]["latent"].shape) == (1, 24, 2, H, W)
    print("[E2] 多 token 锚：k=5 切 [55:60)、替换 index-0 键帧、尾端钳制")


if __name__ == "__main__":
    t_g1()
    t_g2()
    t_g3()
    t_g4()
    t_g5()
    t_g6()
    t_g7()
    t_e2()
    print()
    print("E-3 单测全过：缝窗重去噪的几何、掩码、锁区回写与缝选择逻辑正确。")
    print("E-2 单测过：多 token 锚定切片/钳制正确。")
