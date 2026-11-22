# -*- coding: utf-8 -*-
"""Stage 1：消融（S_bg × ring × method[flow/lerp/none] × 3 Case），自动接二采 3D 上采样→Assemble，出 csv + 拼板

    python3 route3/stage1_ablate.py --api /content/h3_route3/v3_api.json \
        --stage0 /content/h3_route3/artifacts/stage0 \
        --out /content/h3_route3/artifacts/stage1 \
        --combos 5:1:flow,5:1:lerp,5:0:flow,3:1:flow,0:1:flow --resume

combos 语法：`s_bg:ring[:method]`
  * `s_bg = 0` 表示静机位单帧 BG (T_bg=1)；`5` 表示每 17 帧 1 个 BG 锚点；`3` 表示每 3 latent 1 锚点
  * `method` 可选 `flow`（法 B，默认） / `lerp`（法 A 对照组） / `none`（仅用引擎内置 fill_dropped）
  * 只压二采（一采保持原生），并自动将「一采 3D 上采样 latent_p1_up + 二采输出 latent_p2」接入 MiniMaxH3Route3Assemble。

产物：
    <out>/<case>_s{S}_r{R}_{M}.mp4
    <out>/stage1_ablation.csv
    <out>/stage1_contact_sheet.png
    <out>/stage1_DONE.ok
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import route3.wf as W
from route3.mask_provider import load_mask, save_mask
from route3.masks import dilate
from route3.metrics import evaluate, video_frames
from route3.stage0_oracle import CASES, _write_state


def parse_combos(spec: str):
    out = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        s_bg = int(parts[0])
        ring = int(parts[1]) if len(parts) > 1 else 1
        method = parts[2].strip() if len(parts) > 2 else "flow"
        out.append((s_bg, ring, method))
    return out


def ring_variant(stage0, case, ring, cache_dir):
    """从 Stage 0 基准掩码生成外扩 ring 格的版本（基准掩码自带 ring=1，ring=0 原样使用，>0 额外膨胀）。"""
    src = os.path.join(stage0, case + ".mask.npz")
    dst = os.path.join(cache_dir, "%s_r%d.npz" % (case, ring))
    if os.path.exists(dst):
        return dst
    prov = load_mask(src)
    m = prov.mask > 0.5
    if ring > 0:
        m = torch.stack([dilate(m[t], ring) for t in range(m.shape[0])])
    save_mask(dst, m.numpy(), {"ring": ring, "src": src})
    return dst


def _case_meta(stage0, case):
    mp = os.path.join(stage0, case + ".meta.json")
    if os.path.exists(mp):
        return json.load(open(mp, encoding="utf-8"))
    for c in CASES:
        if c["name"] == case:
            return c
    return {"name": case}


def build_contact_sheet(stage0, out_dir, cases, combos, dest_png):
    """把每个 Case 的 Oracle 与各消融档的中间帧拼成一张图，方便 5 秒肉眼审图。"""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    thumb_w, thumb_h = 320, 180
    cols = 1 + len(combos)
    rows = len(cases)
    if rows == 0 or cols == 0:
        return None
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 22)), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)

    def _mid_frame(mp4_path):
        if not os.path.exists(mp4_path):
            return None
        try:
            fr = video_frames(mp4_path, size=(thumb_w, thumb_h))
            return Image.fromarray(fr[len(fr) // 2])
        except Exception:
            return None

    for r, case in enumerate(cases):
        y0 = r * (thumb_h + 22)
        items = [("ORACLE " + case, os.path.join(stage0, case + ".mp4"))]
        for s_bg, ring, method in combos:
            tag = "%s_s%d_r%d_%s" % (case, s_bg, ring, method)
            items.append(("s%d r%d %s" % (s_bg, ring, method), os.path.join(out_dir, tag + ".mp4")))
        for c_idx, (label, path) in enumerate(items):
            x0 = c_idx * thumb_w
            draw.text((x0 + 6, y0 + 4), label, fill=(235, 235, 235))
            im = _mid_frame(path)
            if im is not None:
                canvas.paste(im, (x0, y0 + 22))
    tmp = dest_png + ".tmp.png"
    canvas.save(tmp)
    os.replace(tmp, dest_png)
    return dest_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--out", default="/content/h3_route3/artifacts/stage1")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--combos", default="5:1:flow,5:1:lerp,5:0:flow,3:1:flow",
                    help="格式 s_bg:ring:method（通用叙事档；s_bg=0 仅静机位档请单独在静机位 case 启用）")
    ap.add_argument("--only", default="", help="只跑指定 case（逗号分隔，用于 Stage 1a 单条金丝雀门禁）")
    ap.add_argument("--beta-path", default="", help="可选：Stage 2 校准后的 splitstream_gates.pt")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--keep-tiling", action="store_true")
    ap.add_argument("--keep-sparse", action="store_true")
    ap.add_argument("--strip-lowvram", action="store_true", help="A100 80GB 上旁路本地 8GB 低显存补丁")
    ap.add_argument("--no-av-load", action="store_true", help="禁用 MiniMaxH3AVLatentLoad 剪枝一采子图（改走普通全图）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    masks_dir = os.path.join(a.out, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # dry-run 时若 stage0 还没有 mp4，回退读取 meta.json / 内置 CASES
    cases = sorted(f[:-4] for f in os.listdir(a.stage0) if f.endswith(".mp4"))
    if not cases:
        cases = sorted(f[:-10] for f in os.listdir(a.stage0) if f.endswith(".meta.json"))
    if not cases and a.dry_run:
        cases = [c["name"] for c in CASES]
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        cases = [c for c in cases if c in want]

    combos = parse_combos(a.combos)
    print("cases=%s / combos=%s" % (cases, combos), flush=True)

    base = W.load(a.api)
    cli = W.Comfy(a.server)
    if not a.dry_run:
        for req in ("MiniMaxH3Route3Apply", "MiniMaxH3Route3Assemble"):
            if not cli.has_node(req):
                raise RuntimeError("ComfyUI 未注册 %s 节点 —— 请重启 ComfyUI！" % req)

    rows = []
    t_all = time.time()

    for case in cases:
        oracle = os.path.join(a.stage0, case + ".mp4")
        meta = _case_meta(a.stage0, case)
        for s_bg, ring, method in combos:
            tag = "%s_s%d_r%d_%s" % (case, s_bg, ring, method)
            dest = os.path.join(a.out, tag + ".mp4")
            mpath = (ring_variant(a.stage0, case, ring, masks_dir)
                     if os.path.exists(os.path.join(a.stage0, case + ".mask.npz"))
                     else os.path.join(masks_dir, "%s_r%d.npz" % (case, ring)))

            wf = W.clone(base)
            W.disable_tiling_and_sparse(
                wf, disable_tiling=not a.keep_tiling, disable_sol_attn=not a.keep_sparse,
                strip_lowvram=a.strip_lowvram,
            )
            prompt_txt = meta.get("prompt", "")
            if prompt_txt:
                if "<Picture" not in prompt_txt:
                    W.set_t2va_mode(wf)
                is_gs_prompt = ("green screen" in prompt_txt.lower()) or ("chromakey" in prompt_txt.lower())
                for pid_plan in W.find(wf, "MiniMaxH3HardCutFirstPassPlan"):
                    wf[pid_plan]["inputs"]["green_screen_mode"] = bool(is_gs_prompt)
                W.patch_prompt(wf, prompt_txt)
            if meta.get("cache_key"):
                W.enable_pass1_cache(wf, meta["cache_key"], meta.get("cache_dir", ""))
            if meta.get("duration") is not None:
                W.patch_duration(wf, meta["duration"])
            if meta.get("megapixels") is not None:
                W.set_pass_megapixels(wf, meta.get("first_mp") or None,
                                      meta["megapixels"])
            W.patch_seed(wf, int(meta.get("seed", a.seed)))

            p2 = W.find_pass2_sampler(wf)
            if not p2:
                raise RuntimeError("未定位到二采采样器/执行器节点")

            load_id = None
            if (not a.no_av_load) and meta.get("cache_key"):
                load_id = W.splice_pass1_cache_load(wf, p2, meta["cache_key"], meta.get("cache_dir", ""))

            apply_id = W.splice_pass2_model(
                wf, p2, "MiniMaxH3Route3Apply",
                {"mask_path": mpath, "s_bg": s_bg, "beta_path": a.beta_path, "enabled": True},
                "Route3 Apply (s_bg=%d)" % s_bg,
            )
            asm_id = W.splice_pass2_assemble(
                wf, p2, mask_path=mpath, s_bg=s_bg, method=method, only_bg=True
            )
            print("[%s] pass2=%s av_load=%s apply=%s assemble=%s mask=%s"
                  % (tag, p2, load_id, apply_id, asm_id, os.path.basename(mpath)), flush=True)

            W.save(wf, os.path.join(a.out, tag + ".api.json"))
            if a.dry_run:
                continue

            if not (a.resume and os.path.exists(dest)):
                _write_state(a.out, "stage1", "running", {"current": tag, "done_rows": len(rows)})
                t0 = time.time()
                pid = cli.submit(wf)
                hist = cli.wait(pid)
                files = cli.files(hist)
                if not files:
                    print("  !! 无输出: %s" % tag, flush=True)
                    continue
                cli.download(files[-1][1], dest)
                elapsed_s = round(time.time() - t0, 1)
            else:
                print("  resume hit: %s" % dest, flush=True)
                elapsed_s = 0.0

            try:
                m = load_mask(mpath).mask
                met = evaluate(dest, oracle, (m.numpy() > 0.5))
            except Exception as e:
                met = {"error": repr(e)[:160]}
            met.update({
                "case": case, "s_bg": s_bg, "ring": ring, "method": method,
                "elapsed_s": elapsed_s,
                "ok": bool(
                    met.get("fg_psnr", 0) >= 24.0
                    and met.get("seam_grad_ratio", 9.0) <= 1.8
                    and met.get("bg_hf_sharpness", 0) >= 0.85
                ),
            })
            rows.append(met)
            print("  %s" % json.dumps(met, ensure_ascii=False), flush=True)

    if rows:
        keys = ["case", "s_bg", "ring", "method", "fg_psnr", "bg_hf_sharpness",
                "seam_grad_ratio", "frames", "elapsed_s", "ok", "error"]
        csv_path = os.path.join(a.out, "stage1_ablation.csv")
        tmp_csv = csv_path + ".tmp"
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp_csv, csv_path)
        print("csv -> %s" % csv_path, flush=True)

    sheet = None
    if not a.dry_run and rows:
        sheet = build_contact_sheet(
            a.stage0, a.out, cases, combos, os.path.join(a.out, "stage1_contact_sheet.png")
        )
        if sheet:
            print("contact sheet -> %s" % sheet, flush=True)

    done = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": len(rows),
        "elapsed_min": round((time.time() - t_all) / 60.0, 2),
        "contact_sheet": sheet,
    }
    W.atomic_json(os.path.join(a.out, "stage1_DONE.ok"), done)
    _write_state(a.out, "stage1", "done", done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
