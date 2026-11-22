# -*- coding: utf-8 -*-
"""Route 3 端到端集成单测：覆盖 ComfyUI 插件加载、NestedTensor Assemble、新探针、s_bg=0、API 手术 dry-run"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, r"D:/comfyui/ComfyUI")

FAIL = []


def check(name, ok, detail=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    # 1. ComfyUI exact spec_from_file_location import test (no ROOT on sys.path for route3)
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "route3" or k.startswith("route3.")}
    try:
        for k in list(saved_mods):
            sys.modules.pop(k, None)
        sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(ROOT)]
        spec = importlib.util.spec_from_file_location(
            "comfyui_h3_seamkit_loadtest",
            os.path.join(ROOT, "__init__.py"),
            submodule_search_locations=[ROOT],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        check("ComfyUI 真实插件加载路径（相对导入无 ModuleNotFoundError）",
              hasattr(mod, "comfy_entrypoint"))
    except Exception as e:
        check("ComfyUI 真实插件加载路径（相对导入无 ModuleNotFoundError）", False, repr(e))
    finally:
        sys.path = saved_path
        sys.modules.update(saved_mods)

    import comfy.nested_tensor
    import comfy.ops
    import comfy.model_management as mm
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from route3.mask_provider import save_mask
    from route3.nodes import MiniMaxH3Route3Assemble, MiniMaxH3Route3AutoMask
    from route3.probe_attn import probe_attention, summarize
    from route3.splitstream_attn import route3_plan
    from route3.engine import splitstream_forward

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. s_bg=0 (静机位单帧 T_bg=1) vs s_bg=1 (0% 压缩)
    fm = torch.zeros(10, 4, 4, dtype=torch.bool)
    fm[:, 1:3, 1:3] = True
    p0 = route3_plan(20, 10, 8, 8, s_bg=0, fg_mask=fm.reshape(10, -1))
    p1 = route3_plan(20, 10, 8, 8, s_bg=1, fg_mask=fm.reshape(10, -1))
    check("s_bg=0 真正只留第 0 帧背景 (12 BG rows)", int(p0.bg_local.numel()) == 12 and p0.beta0 > 2.0,
          "bg=%d beta0=%.3f" % (p0.bg_local.numel(), p0.beta0))
    check("s_bg=1 保留全部 10 帧背景 (120 BG rows, beta0=0)", int(p1.bg_local.numel()) == 120 and p1.beta0 == 0.0,
          "bg=%d beta0=%.3f" % (p1.bg_local.numel(), p1.beta0))

    # 3. MiniMaxH3Route3Assemble on real NestedTensor((video, audio)) + only_bg FG preservation
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.npz")
        m_px = torch.zeros(39, 16, 16, dtype=torch.bool)
        m_px[:, 4:12, 4:12] = True
        save_mask(mp, m_px.numpy())

        vid_a = torch.randn(1, 24, 7, 16, 16, device=DEV)
        vid_b = vid_a + 0.25
        vid_b[:, :, :, 4:12, 4:12] = 9.0   # 鲜明 FG 标记，必须 100% 保留
        aud_a = torch.randn(1, 32, 2, 40, device=DEV)
        aud_b = torch.randn(1, 32, 2, 40, device=DEV)
        lat_a = {"samples": comfy.nested_tensor.NestedTensor((vid_a, aud_a))}
        lat_b = {"samples": comfy.nested_tensor.NestedTensor((vid_b, aud_b))}

        for method, s_bg in (("flow", 5), ("lerp", 5), ("none", 5), ("flow", 0)):
            out = MiniMaxH3Route3Assemble.execute(
                lat_a, lat_b, s_bg=s_bg, method=method, only_bg=True, mask_path=mp
            )[0]["samples"]
            ov, oa = out.unbind()
            fg_err = (ov[:, :, :, 4:12, 4:12] - 9.0).abs().max().item()
            aud_err = (oa - aud_b).abs().max().item()
            check("Assemble(NestedTensor, method=%s, s_bg=%d) 保 FG 且保音频" % (method, s_bg),
                  ov.shape == vid_b.shape and fg_err < 1e-5 and aud_err < 1e-6,
                  "fg_err=%.2e aud_err=%.2e" % (fg_err, aud_err))

        # 3b. MiniMaxH3Route3AutoMask (通用无绿幕：直接吃一采 NestedTensor latent 产出 .npz)
        am_path, am_prev, am_rep = MiniMaxH3Route3AutoMask.execute(
            lat_a, ring=1, backend="latent_motion", out_dir=td
        )
        check("AutoMask 直接吃一采 NestedTensor 产出 .npz 掩码与预览张量",
              os.path.exists(am_path) and am_prev.shape[0] == vid_a.shape[2],
              am_rep)

        # 4. Probe v2: 恢复 hook + 视频区 sel_global + 输出替换误差指标
        torch.manual_seed(0)
        m = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1,
                           num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                           text_dim=64, timestep_input_dim=256, time_embed_hidden_size=256,
                           time_embed_dim=128, dtype=torch.float32, device=DEV,
                           operations=comfy.ops.disable_weight_init).to(DEV).eval()
        with torch.no_grad():
            for name, p in m.named_parameters():
                if p.ndim == 1 and ("norm" in name or "scale" in name):
                    p.fill_(1.0)
                    p.add_(torch.randn_like(p) * 0.05)
                else:
                    p.normal_(0.0, 0.05)
        m.requires_grad_(False)
        mm.in_training = False
        orig_fwds = [blk.attn.forward for blk in m.blocks]
        x = torch.randn(1, 24, 6, 8, 8, device=DEV)
        a = torch.randn(1, 32, 2, 20, device=DEV)
        ctx = torch.randn(1, 7, 64, device=DEV)
        ts = torch.tensor([500.0], device=DEV)
        pr = probe_attention(m, [x, a], ts, ctx, {}, {}, s_bg=5, n_query=4)
        sm = summarize(pr)
        restored = all(blk.attn.forward == f for blk, f in zip(m.blocks, orig_fwds))
        check("Probe 结束后所有 block.attn.forward 已完整恢复（无泄漏）", restored)
        errs = [L.get("error") for L in pr["layers"] if L.get("error")]
        check("Probe 产出直接输出替换误差 (rel_err / bg_cos / bg_mass_ratio)",
              not errs and sm["rel_err_median"] > 0 and sm["bg_cos_median"] != 0 and "rel_err" in sm["verdict"],
              "rel_err=%.4f bg_cos=%.4f verdict=%s errs=%s"
              % (sm["rel_err_median"], sm["bg_cos_median"], sm["verdict"], errs[:2]))

        # 5. beta=None 自动触发 beta0 (s_bg=3 时不再静默跳过)
        fm6 = torch.zeros(6, 4, 4, dtype=torch.bool, device=DEV)
        fm6[:, 1:3, 1:3] = True
        with torch.no_grad():
            g_auto = splitstream_forward(m, [x, a], ts, ctx, {}, {}, s_bg=3, beta=None,
                                         fg_mask=fm6.reshape(6, -1), use_beta=True)
            g_nobeta = splitstream_forward(m, [x, a], ts, ctx, {}, {}, s_bg=3, beta=None,
                                           fg_mask=fm6.reshape(6, -1), use_beta=False)
        diff_b = (g_auto[0] - g_nobeta[0]).abs().max().item()
        check("beta=None + use_beta=True 自动注入 beta0（与无偏置存在明显差异）",
              diff_b > 1e-4, "max_diff=%.4e" % diff_b)

        # 6. stage0 & stage1 dry-run against real Colab WenWu_H3_A100_FINAL_api.json
        wenwu = r"D:/dev/colab/workflows/WenWu_H3_A100_FINAL_api.json"
        if os.path.exists(wenwu):
            s0_out = os.path.join(td, "stage0")
            s1_out = os.path.join(td, "stage1")
            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage0_oracle.py"),
                "--api", wenwu, "--out", s0_out, "--probe", "--dry-run",
            ])
            c1_api = json.load(open(os.path.join(s0_out, "case1_static.api.json"), encoding="utf-8"))
            check("Stage0 dry-run: 正确改写 CR Prompt Text.prompt",
                  "hanfu" in c1_api["28"]["inputs"]["prompt"])
            check("Stage0 dry-run: 正确改写 duration (节点 27=5.0) 且关闭 tiling + 旁路 SolAttn",
                  c1_api["27"]["inputs"]["value"] == 5.0
                  and c1_api["59"]["inputs"]["enable_tiling"] is False
                  and "34" not in c1_api)
            probe_nodes = [k for k, v in c1_api.items() if v["class_type"] == "MiniMaxH3Route3Probe"]
            check("Stage0 dry-run: Probe 精确插到二采 HIGH guider (#17)",
                  len(probe_nodes) == 1 and c1_api["17"]["inputs"]["model"] == [probe_nodes[0], 0])

            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage1_ablate.py"),
                "--api", wenwu, "--stage0", s0_out, "--out", s1_out,
                "--combos", "5:1:flow,5:1:lerp,0:1:flow", "--dry-run",
            ])
            s1_api = json.load(open(os.path.join(s1_out, "case2_dolly_s5_r1_flow.api.json"), encoding="utf-8"))
            asm_nodes = [k for k, v in s1_api.items() if v["class_type"] == "MiniMaxH3Route3Assemble"]
            check("Stage1 dry-run: 保留 case2 自己的 prompt + 接入 3D 上采样 latent_p1_up (#15) -> Assemble -> AVDecode (#20)",
                  "dolly-in" in s1_api["28"]["inputs"]["prompt"]
                  and len(asm_nodes) == 1
                  and s1_api[asm_nodes[0]]["inputs"]["latent_p1_up"] == ["15", 0]
                  and s1_api[asm_nodes[0]]["inputs"]["latent_p2"] == ["59", 0]
                  and s1_api["20"]["inputs"]["av_latent"] == [asm_nodes[0], 0])

        # 7. T8 双时钟模板 dry-run（StageSamplerEXPT8 -> StageSave -> Audit -> AVDecode）
        t8_tpl = r"D:/dev/colab/workflows/T8_双时钟_二次采样_普通提示词版_api.json"
        if os.path.exists(t8_tpl):
            t8_s0 = os.path.join(td, "t8_stage0")
            t8_s1 = os.path.join(td, "t8_stage1")
            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage0_oracle.py"),
                "--api", t8_tpl, "--out", t8_s0, "--only", "case1_static", "--probe", "--dry-run",
            ])
            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage1_ablate.py"),
                "--api", t8_tpl, "--stage0", t8_s0, "--out", t8_s1,
                "--combos", "5:1:flow", "--dry-run",
            ])
            t8_api = json.load(open(os.path.join(t8_s1, "case1_static_s5_r1_flow.api.json"), encoding="utf-8"))
            t8_asm = [k for k, v in t8_api.items() if v["class_type"] == "MiniMaxH3Route3Assemble"]
            check("T8 模板 dry-run: 正确改写双份 PlainPromptRecipe + 17n+5 帧长(124) + Assemble 接在 AVDecode(#10) 前",
                  "hanfu" in t8_api["21"]["inputs"]["prompt"]
                  and "hanfu" in t8_api["22"]["inputs"]["prompt"]
                  and t8_api["33"]["inputs"]["value"] == 124
                  and len(t8_asm) == 1
                  and t8_api[t8_asm[0]]["inputs"]["latent_p1_up"] == ["16", 3]
                  and t8_api[t8_asm[0]]["inputs"]["latent_p2"] == ["28", 0]
                  and t8_api["10"]["inputs"]["av_latent"] == [t8_asm[0], 0])

        # 8. 本地一采双时钟模板 dry-run（local_dualclock_api.json，含 --strip-lowvram、自动 T2VA、一采缓存、HardCutUpscale 3口->Assemble）
        local_tpl = os.path.join(ROOT, "route3", "workflows", "local_dualclock_api.json")
        if os.path.exists(local_tpl):
            loc_s0 = os.path.join(td, "loc_stage0")
            loc_s1 = os.path.join(td, "loc_stage1")
            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage0_oracle.py"),
                "--api", local_tpl, "--out", loc_s0, "--only", "case2_dolly",
                "--megapixels", "0.94", "--strip-lowvram", "--probe", "--dry-run",
            ])
            loc0 = json.load(open(os.path.join(loc_s0, "case2_dolly.api.json"), encoding="utf-8"))
            check("本地双时钟 Stage0 dry-run: 自动切 T2VA + 关绿幕模式 + 旁路 SolAttn/LowVRAM + 开启一采缓存 + Probe 接 #40",
                  loc0["8"]["inputs"]["task_type"] == "T2VA"
                  and "ref_images.ref_image_0" not in loc0["8"]["inputs"]
                  and loc0["102"]["inputs"]["green_screen_mode"] is False
                  and "106" not in loc0 and "34" not in loc0 and "35" not in loc0 and "17" not in loc0
                  and loc0["102"]["inputs"]["use_cache"] is True
                  and loc0["102"]["inputs"]["second_megapixels"] == 0.94
                  and "dolly-in" in loc0["9"]["inputs"]["value"])

            subprocess.check_call([
                sys.executable, os.path.join(ROOT, "route3", "stage1_ablate.py"),
                "--api", local_tpl, "--stage0", loc_s0, "--out", loc_s1,
                "--combos", "5:1:flow", "--strip-lowvram", "--dry-run",
            ])
            loc1 = json.load(open(os.path.join(loc_s1, "case2_dolly_s5_r1_flow.api.json"), encoding="utf-8"))
            loc_asm = [k for k, v in loc1.items() if v["class_type"] == "MiniMaxH3Route3Assemble"]
            loc_load = [k for k, v in loc1.items() if v["class_type"] == "MiniMaxH3AVLatentLoad"]
            check("本地双时钟 Stage1 dry-run: MiniMaxH3AVLatentLoad 剪枝整条一采链保音频 + #40.upscaled_latent([40,2]) -> Assemble -> AVDecode(#16)",
                  len(loc_load) == 1
                  and loc1[loc_load[0]]["inputs"]["key"] == loc0["102"]["inputs"]["cache_key"]
                  and loc1["40"]["inputs"]["latent"] == [loc_load[0], 0]
                  and len(loc_asm) == 1
                  and loc1[loc_asm[0]]["inputs"]["latent_p1_up"] == ["40", 2]
                  and loc1[loc_asm[0]]["inputs"]["latent_p2"] == ["40", 0]
                  and loc1["16"]["inputs"]["av_latent"] == [loc_asm[0], 0])

    print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
