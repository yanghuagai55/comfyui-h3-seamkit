# -*- coding: utf-8 -*-
"""Stage 1：消融（S_bg × ring × 3 Case），出 csv + 拼板

    python3 route3/stage1_ablate.py --api /content/h3_route3/v3_api.json \
        --stage0 /content/h3_route3/artifacts/stage0 \
        --out /content/h3_route3/artifacts/stage1 \
        --combos 5:0,5:1,3:1,1:1

只压**二采**（一采保持原生）：少一个变量，而二采的 token 数是一采的 2.3 倍，
收益的大头在这里。`1:1` 是静机位档（s_bg=1 会被 plan 解释成「整片 1 份 BG」）。

产物：
    <out>/<case>_s{S}_r{R}.mp4
    <out>/stage1_ablation.csv
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
from route3.metrics import evaluate
from route3.stage0_oracle import find_pass2_sampler


def ring_variant(stage0, case, ring, cache_dir):
    """从基准掩码生成外扩 ring 格的版本（一次生成，多次复用）。"""
    src = os.path.join(stage0, case + ".mask.npz")
    prov = load_mask(src)
    m = prov.mask > 0.5
    if ring > 0:
        m = torch.stack([dilate(m[t], ring) for t in range(m.shape[0])])
    dst = os.path.join(cache_dir, "%s_r%d.npz" % (case, ring))
    if not os.path.exists(dst):
        save_mask(dst, m.numpy(), {"ring": ring})
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--out", default="/content/h3_route3/artifacts/stage1")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--combos", default="5:0,5:1,3:1,1:1")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    masks_dir = os.path.join(a.out, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    cases = sorted(f[:-4] for f in os.listdir(a.stage0) if f.endswith(".mp4"))
    combos = [tuple(int(x) for x in c.split(":")) for c in a.combos.split(",") if c.strip()]
    print("cases %s / combos %s" % (cases, combos), flush=True)

    base = W.load(a.api)
    cli = W.Comfy(a.server)
    rows = []

    for case in cases:
        oracle = os.path.join(a.stage0, case + ".mp4")
        for s_bg, ring in combos:
            tag = "%s_s%d_r%d" % (case, s_bg, ring)
            mpath = ring_variant(a.stage0, case, ring, masks_dir)
            wf = W.clone(base)
            W.patch_seed(wf, a.seed)
            p2 = find_pass2_sampler(wf)
            W.splice_guider_model(wf, p2, "MiniMaxH3Route3Apply",
                                  {"mask_path": mpath, "s_bg": s_bg, "beta_path": "",
                                   "enabled": True}, "route3 apply")
            print("[%s] pass2=%s mask=%s" % (tag, p2, os.path.basename(mpath)), flush=True)
            if a.dry_run:
                W.save(wf, os.path.join(a.out, tag + ".api.json"))
                continue

            pid = cli.submit(wf)
            hist = cli.wait(pid)
            files = cli.files(hist)
            if not files:
                print("  !! 无输出", flush=True)
                continue
            dest = os.path.join(a.out, tag + ".mp4")
            cli.download(files[-1][1], dest)
            try:
                m = load_mask(mpath).mask
                met = evaluate(dest, oracle, (m.numpy() > 0.5))
            except Exception as e:
                met = {"error": repr(e)[:120]}
            met.update({"case": case, "s_bg": s_bg, "ring": ring,
                        "ok": bool(met.get("fg_psnr", 0) >= 24
                                   and met.get("seam_grad_ratio", 9) <= 1.8
                                   and met.get("bg_hf_sharpness", 0) >= 0.90)})
            rows.append(met)
            print("  %s" % json.dumps(met, ensure_ascii=False), flush=True)

    if rows:
        keys = ["case", "s_bg", "ring", "fg_psnr", "bg_hf_sharpness",
                "seam_grad_ratio", "frames", "ok"]
        csv_path = os.path.join(a.out, "stage1_ablation.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("csv -> %s" % csv_path, flush=True)

    with open(os.path.join(a.out, "stage1_DONE.ok"), "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": len(rows)}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
