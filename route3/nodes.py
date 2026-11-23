# -*- coding: utf-8 -*-
"""Route 3 的三个 ComfyUI 节点

* `MiniMaxH3Route3Apply`    —— 把引擎挂到 model 上（只压你接的那条链路）
* `MiniMaxH3Route3Probe`    —— 第一次前向时顺手量输出替换误差 + 双口径 Δt 分布，写 json
* `MiniMaxH3Route3Assemble` —— 法 A / 法 B / 静机位：用二采关键帧残差还原中间帧背景，
                               完整支持 MiniMax H3 的 NestedTensor(video, audio) 与前景保真

接法（API 格式里就是改一个 inputs 引用）：
    UNET/LoRA 链 → Route3Apply → Guider.model / Upscale.model → 二采采样器
    latent_p1_up（3D 上采样/Reconcile 输出）+ latent_p2（二采输出）→ Route3Assemble → AVDecode
"""

from __future__ import annotations

import logging
import os

import torch

from comfy_api.latest import io

from .assembly import cut_pin_tokens, detect_latent_cuts, motion_pin_tokens, reconstruct_moving, reconstruct_static
from .mask_provider import load_mask, save_mask
from .masks import active_mask, auto_fg_mask
from .patch import Route3State, install
from .splitstream_attn import bg_anchor_indices

CATEGORY = "MiniMaxH3/route3"


def _unpack_av(samples):
    """兼容 NestedTensor((video, audio)) 与纯 5D 视频张量。返回 (video_5d, audio_or_none, is_nested)。"""
    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())
        return parts[0], (parts[1] if len(parts) > 1 else None), True
    return samples, None, False


def _repack_av(video_5d, audio, is_nested):
    if is_nested and audio is not None:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor((video_5d, audio))
    return video_5d


class MiniMaxH3Route3Apply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Apply",
            display_name="Route3 · Split-Stream (apply to model)",
            category=CATEGORY,
            description=(
                "把「前景全速率 + 背景关键帧」的双流引擎挂到这条模型链上。"
                "s_bg=0 表示静机位单帧 BG (T_bg=1)；s_bg=1 表示不压缩对照；s_bg=5 表示每 17 帧 1 个 BG 锚点。"
                "beta_path 留空时自动启用理论初值 beta0 = log(T / n_keep)。"
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input("mask_path", default="", tooltip="mask .npz 路径；留空 = 直通（等价性 smoke）"),
                io.Int.Input("s_bg", default=3, min=0, max=64, step=1,
                             tooltip="0 = 静机位单帧 BG (T_bg=1)；1 = 0% 压缩对照；3 = 推荐甜点(A100 3/3全过)；≥4易出锚点色斑"),
                io.String.Input("beta_path", default="", tooltip="splitstream_gates.pt；留空自动用 beta0=log(T/n_keep)"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("beta_scale", default=1.0, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="对有效 beta 乘以缩放系数（默认 1.0；可压低过大 beta0，如 0.75）"),
                io.Float.Input("beta_max", default=0.0, min=0.0, max=5.0, step=0.05, optional=True,
                               tooltip="有效 beta 上限钳位；0.0 = 不钳位；>0 时按 min(beta, beta_max) 截断（例 1.05）"),
                io.Boolean.Input("last_step_dense", default=True, optional=True,
                                 tooltip="末步全稠密收尾（默认开启）：前 N-1 步 Route 3 压缩提速，最后 1 步自动回切全稠密（+SageAttention）联合细化全帧背景，彻底消除复杂实景背景呼吸跳变"),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, mask_path="", s_bg=3, beta_path="", enabled=True,
                beta_scale=1.0, beta_max=0.0, last_step_dense=True):
        if not enabled:
            print("[route3] disabled - pass through", flush=True)
            return io.NodeOutput(model)

        provider = load_mask(mask_path) if mask_path and os.path.exists(mask_path) else None
        if mask_path and provider is None:
            logging.warning("[route3] mask not found: %s -> 直通（不压缩）", mask_path)

        beta = None
        if beta_path and os.path.exists(beta_path):
            blob = torch.load(beta_path, map_location="cpu", weights_only=True)
            beta = blob.get("beta", blob) if isinstance(blob, dict) else blob
            print("[route3] beta loaded %s" % (tuple(beta.shape) if torch.is_tensor(beta) else beta,), flush=True)

        fn = (lambda lt, lh, lw, st=0: provider.for_grid(lt, lh, lw, start_token=st)) if provider else None
        st = Route3State()
        m = install(model, s_bg=s_bg, beta=beta, fg_mask_fn=fn, state=st,
                    strict_malloc=False, use_beta=True,
                    beta_scale=float(beta_scale), beta_max=float(beta_max),
                    last_step_dense=bool(last_step_dense))
        print("[route3] attached: s_bg=%d mask=%s beta=%s scale=%.2f max=%.2f last_step_dense=%s"
              % (s_bg, "yes" if provider else "no",
                 "custom" if beta is not None else "auto-beta0",
                 float(beta_scale), float(beta_max), bool(last_step_dense)),
              flush=True)
        return io.NodeOutput(m)


class MiniMaxH3Route3Probe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Probe",
            display_name="Route3 · Output-error & RoPE probe",
            category=CATEGORY,
            description="第一次前向时逐层量输出替换误差 (rel_err / bg_cos / bg_mass_ratio) 与双口径 Δt 分布，写 json 后直通。",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("out_path", default="/content/h3_route3/artifacts/probe_attn_stats.json"),
                io.Int.Input("n_query", default=8, min=1, max=64),
                io.Int.Input("layers", default=0, tooltip="0 = 全部层；>0 = 只测前 N 层"),
                io.Int.Input("s_bg", default=5, min=0, max=64, tooltip="探针模拟的背景步长（默认 5）"),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, out_path, n_query=8, layers=0, s_bg=5):
        import types
        from .probe_attn import probe_attention, summarize, dump

        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        orig = dm.forward
        state = {"busy": False, "done": False}

        def fwd(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kw):
            if state["busy"] or state["done"] or denoise_mask is not None:
                return orig(x, timestep, context, transformer_options,
                            minimax_payload=minimax_payload,
                            denoise_mask=denoise_mask,
                            audio_denoise_mask=audio_denoise_mask, **kw)
            state["busy"] = True
            try:
                lay = list(range(len(self.blocks)))[:layers] if layers else None
                p = probe_attention(self, x, timestep, context, transformer_options,
                                    minimax_payload, s_bg=s_bg, n_query=n_query, layers=lay)
                s = summarize(p)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                dump(p, s, out_path)
                print("[route3] probe -> %s  verdict=%s rel_err=%.4f bg_cos=%.4f T=%.3f"
                      % (out_path, s["verdict"], s["rel_err_median"], s["bg_cos_median"], s["overall_median"]),
                      flush=True)
                state["done"] = True
            except Exception as e:
                logging.warning("[route3] probe failed: %r", e)
                state["done"] = True
            finally:
                state["busy"] = False
            return orig(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload,
                        denoise_mask=denoise_mask,
                        audio_denoise_mask=audio_denoise_mask, **kw)

        m.add_object_patch("diffusion_model.forward", types.MethodType(fwd, dm))
        return io.NodeOutput(m)


class MiniMaxH3Route3Assemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3Assemble",
            display_name="Route3 · Background reassembly (法A / 法B)",
            category=CATEGORY,
            description=(
                "衔接「一采 3D 上采样 latent (latent_p1_up)」与「二采压缩采样输出 (latent_p2)」。"
                "利用二采在关键帧上真正重绘出的高清残差 δ = z_p2 - z_p1_up 还原中间帧背景：\n"
                "  lerp = 固定坐标线性插值（默认主推：A100 实测 <1ms 且 bg_hf 三场景全面胜过 flow）\n"
                "  flow = latent 空间分块位移场双向 warp（可选对照，耗时较长）\n"
                "  none = 不重装，直接透传 latent_p2（用于对照引擎内置 fill_dropped）\n"
                "当 only_bg=True 且提供 mask_path 时，前景区 100% 保留二采全速率结果，只替换背景区。"
            ),
            inputs=[
                io.Latent.Input("latent_p1_up", tooltip="一采 3D 上采样（或 Reconcile）后的全帧 AV latent"),
                io.Latent.Input("latent_p2", tooltip="二采采样器输出的 AV latent"),
                io.Int.Input("s_bg", default=3, min=0, max=64,
                             tooltip="0 = 静机位单帧复用 z_p2[0]；3 = 推荐甜点"),
                io.Combo.Input("method", options=["lerp", "flow", "none"], default="lerp"),
                io.Boolean.Input("only_bg", default=True,
                                 tooltip="True = 前景区保留 latent_p2 全速率动作，只对背景区做残差还原"),
                io.String.Input("mask_path", default="",
                                tooltip="mask .npz 路径（only_bg=True 时使用）"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent_p1_up, latent_p2, s_bg=3, method="lerp", only_bg=True, mask_path=""):
        if method == "none" or latent_p2.get("route3_last_step_dense"):
            if latent_p2.get("route3_last_step_dense"):
                print("[route3] Assemble: detected last_step_dense=True -> direct passthrough (full-frame refined)", flush=True)
            return io.NodeOutput(latent_p2)

        vid_a, _aud_a, is_nested_a = _unpack_av(latent_p1_up["samples"])
        vid_b, aud_b, is_nested_b = _unpack_av(latent_p2["samples"])
        if vid_a.shape != vid_b.shape:
            raise ValueError("两路视频 latent 形状不一致: %s vs %s" % (tuple(vid_a.shape), tuple(vid_b.shape)))
        if vid_a.ndim != 5 or vid_a.shape[0] != 1:
            raise ValueError("期望视频 latent 形状 [1, C, T, H, W]，收到 %s" % (tuple(vid_a.shape),))

        # [1, C, T, H, W] -> [T, C, H, W]（reconstruct_moving/static 口径）
        dev = vid_b.device
        dtype = vid_b.dtype
        a_tchw = vid_a[0].to(device=dev, dtype=torch.float32).permute(1, 0, 2, 3).contiguous()
        b_tchw = vid_b[0].to(device=dev, dtype=torch.float32).permute(1, 0, 2, 3).contiguous()
        T, C, H, W = a_tchw.shape

        # 构建 latent 网格 [T, H, W] 的前景掩码（若 only_bg 且掩码存在）
        fg_lat = None
        if only_bg and mask_path and os.path.exists(mask_path):
            prov = load_mask(mask_path)
            # patch 网格是 (H//2, W//2)，直接放大回 latent (H, W) 对齐每个 2x2 DiT patch
            ph, pw = max(1, H // 2), max(1, W // 2)
            m_patch = prov.for_grid(T, ph * 2, pw * 2).view(T, ph, pw).to(dev)
            fg_lat = m_patch.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
            if fg_lat.shape[1] != H or fg_lat.shape[2] != W:
                fg_lat = torch.nn.functional.interpolate(
                    fg_lat.float().unsqueeze(1), size=(H, W), mode="nearest"
                ).squeeze(1) > 0.5

        if s_bg <= 0 or s_bg >= T:
            # 静机位档：把二采画好的第 0 帧高清背景铺满所有背景区，前景留二采全速率
            rec_tchw = reconstruct_static(b_tchw, b_tchw[0], fg_mask=fg_lat)
        else:
            cuts = detect_latent_cuts(a_tchw)
            pins = sorted(set(cut_pin_tokens(cuts, T, radius=1)) | set(motion_pin_tokens(a_tchw, s_bg=s_bg, cuts=cuts, fg_mask=fg_lat)))
            anchors = bg_anchor_indices(T, s_bg, pin_tokens=pins)
            delta = {t: (b_tchw[t] - a_tchw[t]) for t in anchors}
            bg_tchw = reconstruct_moving(a_tchw, delta, anchors, method=method, fg_mask=fg_lat, cuts=cuts)
            if fg_lat is not None:
                # 前景核心 100% 保留 b_tchw（fg_err == 0），外缘 1-2 latent 格做空间软羽化，
                # 彻底消除 2×2 DiT patch（32×32 像素）硬二值切边导致的方框马赛克！
                fg_f = fg_lat.float().unsqueeze(1)
                fg_dil = torch.nn.functional.max_pool2d(fg_f, kernel_size=3, stride=1, padding=1)
                fg_soft = torch.nn.functional.avg_pool2d(fg_dil, kernel_size=3, stride=1, padding=1)
                alpha = torch.maximum(fg_f, fg_soft).clamp(0.0, 1.0)
                rec_tchw = alpha * b_tchw + (1.0 - alpha) * bg_tchw
            else:
                rec_tchw = bg_tchw

        out_5d = rec_tchw.permute(1, 0, 2, 3).unsqueeze(0).to(dtype=dtype).contiguous()
        out_samples = _repack_av(out_5d, aud_b, is_nested_b or is_nested_a)
        res = dict(latent_p2)
        res["samples"] = out_samples
        return io.NodeOutput(res)


class MiniMaxH3Route3AutoMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Route3AutoMask",
            display_name="Route3 · Auto Foreground Mask (一采直出通用掩码)",
            category=CATEGORY,
            description=(
                "适用于**任意普通实景视频与绿幕视频**：直接吃一采输出 latent（如 #93 denoised_output），"
                "自动提取「语义前景主体 ∪ 全局运镜补偿后的局部高频背景 ∪ 外扩阴影环」，"
                "写 .npz 并输出 mask_path 直连 Route3Apply 与 Route3Assemble。\n"
                "• 接 video_vae：解码低清一采帧跑通用语义抠图（bria_rmbg / torchvision / rembg / 绿幕色键）+ 运动补偿；\n"
                "• 不接 video_vae：0.3 秒直接在 latent 空间做全局运动补偿差分（零下载、零 VAE 显存）。"
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="一采输出 AV latent（#93 denoised_output）"),
                io.Int.Input("ring", default=1, min=0, max=4, tooltip="前景外扩阴影/边缘环（patch 格数）"),
                io.Combo.Input(
                    "backend",
                    options=["auto", "latent_motion", "bria_rmbg", "torchvision", "rembg", "chroma"],
                    default="auto",
                    tooltip="auto: 绿幕走色键，普通视频自动选语义前景分割 + 运动补偿；latent_motion: 纯 latent 零下载",
                ),
                io.String.Input("out_dir", default="", tooltip="掩码落盘目录（留空写到 ComfyUI/temp/route3_masks，不占 C 盘；可填任意自定义路径）"),
                io.Vae.Input("video_vae", optional=True, tooltip="可选：接 Video VAE 可解码一采低清帧跑语义前景抠图"),
            ],
            outputs=[
                io.String.Output(display_name="mask_path"),
                io.Mask.Output(display_name="preview_mask"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def _default_mask_dir(cls):
        try:
            import folder_paths
            d = os.path.join(folder_paths.get_temp_directory(), "route3_masks")
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            pass
        # 回落到插件目录内，绝不写入 C 盘系统 Temp
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "route3_masks")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def execute(cls, latent, ring=1, backend="auto", out_dir="", video_vae=None):
        import hashlib

        vid_5d, _aud, _ = _unpack_av(latent["samples"])
        if vid_5d.ndim != 5 or vid_5d.shape[0] != 1:
            raise ValueError("期望视频 latent [1, C, T, H, W]，收到 %s" % (tuple(vid_5d.shape),))

        # 稳定内容指纹，方便缓存与复用
        h_hex = hashlib.sha1(vid_5d[:, :, :1].detach().float().cpu().numpy().tobytes()).hexdigest()[:12]
        base_dir = out_dir.strip() if out_dir and out_dir.strip() else cls._default_mask_dir()
        os.makedirs(base_dir, exist_ok=True)
        mpath = os.path.join(base_dir, "r3_mask_%s_r%d_%s.npz" % (h_hex, int(ring), backend))

        used_backend = "latent_motion"
        if video_vae is not None and backend != "latent_motion":
            # 解码低清 0.4MP 一采视频帧 → [T_px, H_px, W_px, 3] float [0..1]
            with torch.no_grad():
                decoded = video_vae.decode(vid_5d)
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            if decoded.ndim == 5:
                decoded = decoded[0]
            # 统一转 [T, 3, H, W] 并降到短边 ~180 提速
            fr_nchw = decoded.detach().float().clamp(0, 1).permute(0, 3, 1, 2).contiguous()
            _, _, hp, wp = fr_nchw.shape
            if min(hp, wp) > 192:
                sc = 180.0 / float(min(hp, wp))
                fr_nchw = torch.nn.functional.interpolate(
                    fr_nchw, size=(max(32, int(round(hp * sc))), max(32, int(round(wp * sc)))),
                    mode="area",
                )
            fg_sem, used_backend = auto_fg_mask(fr_nchw.cpu(), backend=backend, allow_download=True)
            m_grid, info = active_mask(
                fr_nchw.cpu(), fg_mask=fg_sem, ring=int(ring),
                patch_grid=(max(8, fr_nchw.shape[-2] // 2), max(8, fr_nchw.shape[-1] // 2)),
            )
            prev_mask = m_grid.float()
        else:
            # 纯 latent 空间全局运动补偿活跃区（0 额外权重、0 VAE 解码）
            lat_tchw = vid_5d[0].detach().float().cpu().permute(1, 0, 2, 3).contiguous()
            _, _, lh, lw = lat_tchw.shape
            m_grid, info = active_mask(
                lat_tchw, fg_mask=None, ring=int(ring),
                patch_grid=(max(4, lh // 2), max(4, lw // 2)),
            )
            prev_mask = m_grid.float()

        cov = float(m_grid.float().mean())
        save_mask(mpath, m_grid.numpy(), {
            "backend": used_backend, "ring": int(ring),
            "cov": round(cov, 4), "thr": info["thr"],
        })
        rep = "backend=%s fg=%.1f%% shape=%s thr=%.4f -> %s" % (
            used_backend, 100.0 * cov, tuple(m_grid.shape), info["thr"], mpath
        )
        print("[route3] AutoMask: " + rep, flush=True)
        return io.NodeOutput(mpath, prev_mask, rep)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Route3Apply": MiniMaxH3Route3Apply,
    "MiniMaxH3Route3Probe": MiniMaxH3Route3Probe,
    "MiniMaxH3Route3Assemble": MiniMaxH3Route3Assemble,
    "MiniMaxH3Route3AutoMask": MiniMaxH3Route3AutoMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Route3Apply": "Route3 · Split-Stream (apply to model)",
    "MiniMaxH3Route3Probe": "Route3 · Output-error & RoPE probe",
    "MiniMaxH3Route3Assemble": "Route3 · Background reassembly (法A/法B)",
    "MiniMaxH3Route3AutoMask": "Route3 · Auto Foreground Mask (通用实景/绿幕)",
}
