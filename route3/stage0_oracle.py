# -*- coding: utf-8 -*-
"""Stage 0：3 条叙事 Case 的稠密二采 Oracle + 活跃掩码 + 输出替换误差探针

    python3 route3/stage0_oracle.py --api /content/h3_route3/v3_api.json \
        --out /content/h3_route3/artifacts/stage0 --probe --resume

只依赖 ComfyUI HTTP 接口。默认关闭空间 tiling 与 SolAttn 稀疏近似（可用 --keep-tiling / --keep-sparse 保留），
确保产出纯净的「全稠密二采 Oracle」作为 Stage 1 所有画质/锐度指标的分母基准。

产物：
    <out>/<case>.mp4            稠密 Oracle（一采 + 二采）
    <out>/<case>.mask.npz       像素帧级活跃掩码（供 Stage 1 注入）
    <out>/<case>.meta.json      该 Case 实际使用的 prompt / 时长 / 分辨率 / 覆盖率
    <out>/<case>.api.json       本次实际提交的工作流（可复现）
    <out>/probe_attn_stats.json （--probe 时，输出替换误差 + 双口径 Δt 分布）
    <out>/run_state.json        看门狗心跳状态
    <out>/stage0_DONE.ok
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import route3.wf as W
from route3.mask_provider import save_mask
from route3.masks import active_mask, auto_fg_mask
from route3.metrics import video_frames

CASES = [
    {"name": "case1_static",
     "prompt": "Fixed camera, medium close-up of a woman in a dark red hanfu standing still, "
               "slow natural breathing, subtle blink, hands making a continuous small gesture, "
               "warm indoor tungsten light, shallow depth of field, ambient room tone",
     "note": "静机位表演：标定 T_bg=1 上限与发丝/衣纹细节"},
    {"name": "case2_dolly",
     "prompt": "Slow dolly-in following a man walking from left to right across a narrow "
               "alley, clear foreground-background parallax, wet brick wall, footsteps and "
               "light ambient hiss, 35mm film look",
     "note": "动态运镜 + 人物大位移：拷问 S_bg=5 残差传播与光流 warp"},
    {"name": "case3_neon",
     "prompt": "Neon-lit night street, strong side backlight, a woman turns from medium shot "
               "into a close-up, her shadow thrown on the wet ground, reflections on the wall, "
               "continuous street ambience",
     "note": "强光影 + 一次景别切换：拷问跨镜连贯与脚底阴影"},
]


def _write_state(out_dir, stage, status, extra=None):
    state_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "run_state.json")
    blob = {
        "stage": stage,
        "status": status,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
    }
    W.atomic_json(state_path, blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--out", default="/content/h3_route3/artifacts/stage0")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--megapixels", type=float, default=0.0,
                    help="兼容旧参数；等价于 --second-mp")
    ap.add_argument("--first-mp", type=float, default=0.0, help="一采 MP；0 = 保持工作流原值")
    ap.add_argument("--second-mp", type=float, default=1.3,
                    help="二采 MP（最终画质）。默认 1.3 ≈ 1536x864；别超 1.4")
    ap.add_argument("--cases", default="", help="cases.json；留空用内置 3 条")
    ap.add_argument("--only", default="", help="只跑某个 case 名")
    ap.add_argument("--probe", action="store_true", help="在第一个 Case 二采上挂输出替换误差探针")
    ap.add_argument("--probe-s-bg", type=int, default=5)
    ap.add_argument("--keep-tiling", action="store_true", help="保留 Goohai 空间 tiling")
    ap.add_argument("--keep-sparse", action="store_true", help="保留 SolAttn 稀疏注意力")
    ap.add_argument("--strip-lowvram", action="store_true", help="A100 80GB 上旁路本地 8GB 低显存分头/分块补丁")
    ap.add_argument("--mask-backend", default="auto",
                    choices=["auto", "bria_rmbg", "torchvision", "rembg", "chroma", "motion"],
                    help="前景主体分割后端（auto: 绿幕走色键，普通实景视频自动走通用语义抠图 + 运动补偿）")
    ap.add_argument("--cache-dir", default="", help="一采缓存目录（默认 <out>/p1_cache，供 Stage 1 0秒复用一采）")
    ap.add_argument("--resume", action="store_true", help="跳过已完成（mp4+mask+meta 均存在）的 Case")
    ap.add_argument("--dry-run", action="store_true", help="只改图落盘，不提交 ComfyUI")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cases = json.load(open(a.cases, encoding="utf-8")) if a.cases else CASES
    if a.only:
        cases = [c for c in cases if c["name"] == a.only]

    base = W.load(a.api)
    cli = W.Comfy(a.server)
    if not a.dry_run and a.probe and not cli.has_node("MiniMaxH3Route3Probe"):
        raise RuntimeError(
            "ComfyUI 未注册 MiniMaxH3Route3Probe 节点 —— 请重启 ComfyUI 让最新 route3 生效！"
        )

    ok = []
    t_all = time.time()
    probe_path = os.path.join(a.out, "probe_attn_stats.json")
    probe_done = os.path.exists(probe_path)

    for idx, c in enumerate(cases):
        cname = c["name"]
        dest = os.path.join(a.out, cname + ".mp4")
        mpath = os.path.join(a.out, cname + ".mask.npz")
        meta_path = os.path.join(a.out, cname + ".meta.json")

        if a.resume and os.path.exists(dest) and os.path.exists(mpath) and os.path.exists(meta_path):
            print("[%s] resume skip (already complete)" % cname, flush=True)
            ok.append(cname)
            continue

        wf = W.clone(base)
        notes = W.disable_tiling_and_sparse(
            wf, disable_tiling=not a.keep_tiling, disable_sol_attn=not a.keep_sparse,
            strip_lowvram=a.strip_lowvram,
        )
        if "<Picture" not in c["prompt"]:
            t2va_ids = W.set_t2va_mode(wf)
            if t2va_ids:
                notes.append("t2va:%s" % ",".join(t2va_ids))
        # 普通实景视频自动关闭 green_screen_mode（仅当提示词明确要求绿幕时保留）
        is_gs_prompt = ("green screen" in c["prompt"].lower()) or ("chromakey" in c["prompt"].lower())
        for pid_plan in W.find(wf, "MiniMaxH3HardCutFirstPassPlan"):
            wf[pid_plan]["inputs"]["green_screen_mode"] = bool(is_gs_prompt)
        cache_dir = a.cache_dir or os.path.join(os.path.abspath(a.out), "p1_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = "r3_%s_s%d" % (cname, int(a.seed))
        W.enable_pass1_cache(wf, cache_key, cache_dir)
        p_nodes = W.patch_prompt(wf, c["prompt"])
        d_node = W.patch_duration(wf, a.duration)
        second = a.megapixels or a.second_mp
        mp_nodes = W.set_pass_megapixels(wf, a.first_mp or None, second)
        W.patch_seed(wf, a.seed)
        p2 = W.find_pass2_sampler(wf)
        if not p2:
            raise RuntimeError("工作流中未定位到二采采样器/执行器节点")

        attach_probe = bool(a.probe and not probe_done)
        if attach_probe:
            W.splice_pass2_model(
                wf, p2, "MiniMaxH3Route3Probe",
                {"out_path": probe_path, "n_query": 8, "layers": 0, "s_bg": int(a.probe_s_bg)},
                "Route3 Probe",
            )

        print(
            "[%s] prompt<-%s dur<-%s mp<-%s pass2=%s(%s) probe=%s clean=%s"
            % (cname, p_nodes, d_node, mp_nodes, p2, wf[p2]["class_type"], attach_probe, notes),
            flush=True,
        )
        W.save(wf, os.path.join(a.out, cname + ".api.json"))
        meta = {
            "name": cname,
            "prompt": c["prompt"],
            "note": c.get("note", ""),
            "duration": float(a.duration),
            "megapixels": float(second),
            "first_mp": float(a.first_mp),
            "seed": int(a.seed),
            "cache_key": cache_key,
            "cache_dir": cache_dir,
            "pass2_node": p2,
            "pass2_class": wf[p2]["class_type"],
        }

        if a.dry_run:
            W.atomic_json(meta_path, meta)
            ok.append(cname)
            continue

        _write_state(a.out, "stage0", "running", {"current_case": cname, "done": ok})
        t0 = time.time()
        pid = cli.submit(wf)
        hist = cli.wait(pid)
        files = cli.files(hist)
        if not files:
            raise RuntimeError("[%s] ComfyUI 执行完成但没有输出视频文件" % cname)
        cli.download(files[-1][1], dest)
        dt_min = (time.time() - t0) / 60.0
        print("  %s  %.2f MB  %.1f min" % (dest, os.path.getsize(dest) / 1e6, dt_min), flush=True)

        if attach_probe and os.path.exists(probe_path):
            probe_done = True

        # 解码 320x180 预览帧，转 NCHW float32 [0..1] 后算通用前景主体 + 全局运动补偿活跃掩码
        fr = video_frames(dest, size=(320, 180))  # [T, H, W, 3] uint8
        fr_nchw = torch.from_numpy(fr.copy()).permute(0, 3, 1, 2).float() / 255.0
        sem_fg, fg_backend = auto_fg_mask(fr_nchw, backend=a.mask_backend, allow_download=True)
        m, info = active_mask(fr_nchw, fg_mask=sem_fg, ring=1, patch_grid=(90, 160))
        cov = float(m.float().mean())
        save_mask(mpath, m.numpy(), {"thr": info["thr"], "size": (320, 180), "src": dest,
                                     "cov": cov, "fg_backend": fg_backend})
        print("  mask 覆盖 %.1f%% (thr=%.4f, backend=%s)"
              % (100.0 * cov, info["thr"], fg_backend), flush=True)

        meta.update({"elapsed_min": round(dt_min, 2), "mask_coverage": round(cov, 4),
                     "mask_thr": info["thr"], "mask_backend": fg_backend})
        W.atomic_json(meta_path, meta)
        ok.append(cname)

    summary = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": ok,
               "elapsed_min": round((time.time() - t_all) / 60.0, 2)}
    if os.path.exists(probe_path):
        pr = json.load(open(probe_path, encoding="utf-8"))
        summary["probe_verdict"] = pr.get("summary", {}).get("verdict")
        summary["rel_err_median"] = pr.get("summary", {}).get("rel_err_median")
        summary["bg_cos_median"] = pr.get("summary", {}).get("bg_cos_median")
        print("probe verdict: %s" % summary["probe_verdict"], flush=True)

    W.atomic_json(os.path.join(a.out, "stage0_DONE.ok"), summary)
    _write_state(a.out, "stage0", "done", summary)
    print("stage0 done: %s" % ok, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
