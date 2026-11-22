# -*- coding: utf-8 -*-
"""Stage 0：3 条 Case 的稠密 Oracle + 活跃掩码（可选：顺便跑 RoPE 衰减探针）

    python3 route3/stage0_oracle.py --api /content/h3_route3/v3_api.json \
        --out /content/h3_route3/artifacts/stage0 --probe

只依赖 ComfyUI 的 HTTP 接口。**不加任何 Route 3 节点**（除非 --probe），
所以它跑出来的是纯净基准。

产物：
    <out>/<case>.mp4          稠密 Oracle（一采+二采，原生分辨率）
    <out>/<case>.mask.npz     像素帧级活跃掩码（供 Stage 1 注入）
    <out>/<case>.api.json     本次实际提交的工作流（可复现）
    <out>/probe_attn_stats.json （--probe 时）
    <out>/stage0_DONE.ok
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import route3.wf as W
from route3.mask_provider import save_mask

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
     "note": "动态运镜 + 人物大位移：拷问 S_bg=5 残差传播"},
    {"name": "case3_neon",
     "prompt": "Neon-lit night street, strong side backlight, a woman turns from medium shot "
               "into a close-up, her shadow thrown on the wet ground, reflections on the wall, "
               "continuous street ambience",
     "note": "强光影 + 一次景别切换：拷问跨镜连贯与脚底阴影"},
]


def find_pass2_sampler(wf):
    rec = W.find(wf, "MiniMaxH3TwoPassLatentReconcileT8Advanced")
    samplers = W.find(wf, "SamplerCustomAdvanced")
    for sid in samplers:
        up = W.upstream(wf, wf[sid]["inputs"].get("latent_image"))
        if up and rec and up[0] in rec:
            return sid
    return samplers[-1] if samplers else None


def patch_duration(wf, seconds):
    for nid in W.find(wf, "MiniMaxH3AudioConditioningT8"):
        up = W.upstream(wf, wf[nid]["inputs"].get("length"))
        if not up:
            continue
        mid = up[0]
        if wf[mid]["class_type"] != "ComfyMathExpression":
            continue
        a = wf[mid]["inputs"].get("a")
        src = W.upstream(wf, a)
        if src and "value" in wf[src[0]]["inputs"]:
            wf[src[0]]["inputs"]["value"] = float(seconds)
            return src[0]
        if isinstance(a, (int, float)):
            wf[mid]["inputs"]["a"] = float(seconds)
            return mid
    for nid in W.find(wf, "PrimitiveFloat"):
        v = wf[nid]["inputs"].get("value")
        if isinstance(v, (int, float)) and 1 <= v <= 60:
            wf[nid]["inputs"]["value"] = float(seconds)
            return nid
    return None


def patch_megapixels(wf, mp):
    ids = W.find(wf, "ResolutionSelector")
    for nid in ids:
        wf[nid]["inputs"]["megapixels"] = float(mp)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--out", default="/content/h3_route3/artifacts/stage0")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--megapixels", type=float, default=0.94)
    ap.add_argument("--cases", default="", help="cases.json；留空用内置 3 条")
    ap.add_argument("--only", default="", help="只跑某个 case 名")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只改图不提交")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cases = json.load(open(a.cases, encoding="utf-8")) if a.cases else CASES
    if a.only:
        cases = [c for c in cases if c["name"] == a.only]

    base = W.load(a.api)
    cli = W.Comfy(a.server)
    ok = []
    for c in cases:
        wf = W.clone(base)
        W.patch_prompt(wf, c["prompt"])
        d = patch_duration(wf, a.duration)
        mp = patch_megapixels(wf, a.megapixels)
        W.patch_seed(wf, a.seed)
        p2 = find_pass2_sampler(wf)
        if a.probe and p2:
            W.splice_guider_model(wf, p2, "MiniMaxH3Route3Probe",
                                  {"out_path": os.path.join(a.out, "probe_attn_stats.json"),
                                   "n_query": 8, "layers": 0}, "route3 probe")
        print("[%s] duration<-%s mp<-%s pass2=%s probe=%s"
              % (c["name"], d, mp, p2, bool(a.probe)), flush=True)

        W.save(wf, os.path.join(a.out, c["name"] + ".api.json"))
        if a.dry_run:
            ok.append(c["name"])
            continue

        t0 = time.time()
        pid = cli.submit(wf)
        hist = cli.wait(pid)
        files = cli.files(hist)
        if not files:
            print("  !! 没有输出文件", flush=True)
            continue
        dest = os.path.join(a.out, c["name"] + ".mp4")
        cli.download(files[-1][1], dest)
        print("  %s  %.2f GB  %.1f min" % (dest, os.path.getsize(dest) / 1e9,
                                           (time.time() - t0) / 60), flush=True)

        try:
            from route3.mask_provider import MaskProvider
            from route3.masks import active_mask
            from route3.metrics import video_frames
            fr = video_frames(dest, size=(320, 180))
            m, info = active_mask(fr, ring=1)
            save_mask(os.path.join(a.out, c["name"] + ".mask.npz"), m.numpy(),
                      {"thr": info["thr"], "size": (320, 180), "src": dest})
            print("  mask 覆盖 %.1f%% (thr=%.4f)" % (100 * float(m.float().mean()), info["thr"]),
                  flush=True)
        except Exception as e:
            print("  !! 掩码失败（不影响 Oracle）: %r" % e, flush=True)
        ok.append(c["name"])

    with open(os.path.join(a.out, "stage0_DONE.ok"), "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": ok}, f, ensure_ascii=False)
    print("done: %s" % ok, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
