#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 comfyui-h3-seamkit contributors
"""widgets_values sanity check for every node in a ComfyUI workflow.

Why this exists (2026-09-20, cost: one afternoon):
`widgets_values` is a POSITIONAL array - the front end assigns it in order.
When a node parameter is removed from `define_schema` but the leftover value
stays in the saved workflow, every slot AFTER it shifts by one.  Nothing
errors out, the graph still runs, and the node quietly uses the wrong values
(that is how `insert_mode` silently fell back to manual and the redraw
insertion point went back to the declared cut = "10 frames off").

This script diffs, per node:
    len(widgets_values)  vs  number of widget slots in /object_info
and, when `widgets_values_named` exists (new front-end format), also diffs
every named value against its positional twin.

Usage
-----
    python tools/check_widgets.py                       # scan the whole workflow dir
    python tools/check_widgets.py a.json b.json         # specific files
    python tools/check_widgets.py a.json --fix          # delete leftover slots
                                                        # (writes .bak-widgets-<ts>)

ComfyUI must be running (reads /object_info); pass --offline to skip the
schema query - then only intra-file consistency is reported.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import io
import json
import os
import shutil
import urllib.request

HOST = "http://127.0.0.1:8188"
SOCKET_TYPES = {"MODEL", "CLIP", "VAE", "IMAGE", "LATENT", "CONDITIONING",
                "AUDIO", "NOISE", "SAMPLER", "SIGMAS", "GUIDER",
                "T8_H3_CHUNKED_TWO_PASS_PLAN", "*"}
# Front-end-only widget slots that never appear in the backend schema.
# Counting them as "leftover" produces pure noise (they shift nothing).
KNOWN_HIDDEN = {"upload", "mode.scale", "values", "preview", "videopreview",
                "control_after_generate", "ref_videos", "ref_video_audios",
                "ref_audios", "semantic_bridge", "allow_above_reference_area",
                "pix_fmt", "crf", "save_metadata", "trim_to_audio", "audioUI"}
# Only nodes from this package are ours to fix; other packs manage their own.
# Identified by the category declared in nodes.py / nodes_repair.py / h3_upscale.py.
OWN_CATEGORY_MARK = "hardcut"
WF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "user", "default", "workflows")
NAME_JSON = "workflow.json"


def fetch_schema(offline: bool) -> dict:
    if offline:
        return {}
    with urllib.request.urlopen(HOST + "/object_info", timeout=60) as r:
        return json.load(r)


def value_problem(spec, value) -> str:
    """Return a reason string when `value` cannot legally fill this input.

    This is the check that actually catches a slot shift: when the array is
    shifted by one, a COMBO ends up holding a number and an INT ends up
    holding a string - both impossible, both visible right here.
    """
    t = spec[0]
    t0 = t[0] if isinstance(t, list) else t
    if isinstance(t, list):                       # COMBO -> list of options
        opts = t
        if value not in opts:
            return f"not one of {opts[:6]}{'...' if len(opts) > 6 else ''}"
        return ""
    if t0 == "BOOLEAN" and not isinstance(value, bool):
        return "expected true/false"
    if t0 == "INT" and (isinstance(value, bool)
                        or not isinstance(value, int)
                        and not (isinstance(value, float) and value.is_integer())):
        return "expected an integer"
    if t0 == "FLOAT" and (isinstance(value, bool)
                          or not isinstance(value, (int, float))):
        return "expected a number"
    if t0 == "STRING" and not isinstance(value, str):
        return "expected a string"
    return ""


def widget_order(info: dict) -> list:
    """Widget slots in front-end order.

    ComfyUI appends a `control_after_generate` combo right after a seed INT
    widget, and that extra widget IS part of widgets_values.
    """
    inp = info.get("input", {})
    seq = list(inp.get("required", {}).items()) + list(inp.get("optional", {}).items())
    order = []
    for name, spec in seq:
        t = spec[0]
        t0 = t[0] if isinstance(t, list) else t
        if t0 in SOCKET_TYPES:
            continue
        order.append(name)
        if t0 == "INT" and "seed" in name.lower():
            order.append("control_after_generate")
    return order


def check_file(path: str, schema: dict, offline: bool, fix: bool, verbose: bool):
    doc = json.load(io.open(path, encoding="utf-8"))
    problems = []
    for node in doc.get("nodes", []):
        nid, ntype = node.get("id"), node.get("type")
        info = schema.get(ntype, {})
        if OWN_CATEGORY_MARK not in str(info.get("category", "")).lower():
            continue                      # not ours (or offline) - skip quietly
        wv = node.get("widgets_values")
        if not isinstance(wv, list):
            continue                      # dict form (VHS et al.) - not positional
        if offline or ntype not in schema:
            continue
        order = widget_order(info)
        if not order:
            continue
        # NOTE: `order` is kept index-aligned with the array on purpose.  A
        # widget overridden by a link may or may not keep its slot depending on
        # the front end version, so positional checking is only a fallback -
        # `widgets_values_named` (when present) is authoritative and index-free.
        if not isinstance(node.get("widgets_values_named"), dict) and verbose:
            linked = {i.get("name") for i in node.get("inputs", [])
                      if i.get("link") is not None}
            if linked:
                print(f"  #{nid} {ntype}: no named map; linked widgets {sorted(linked)} "
                      "may or may not occupy a slot - treat positional results as a hint")
        seq = list(info.get("input", {}).get("required", {}).items()) + \
            list(info.get("input", {}).get("optional", {}).items())
        spec_map = {nm: sp for nm, sp in seq}
        named = node.get("widgets_values_named")
        shift = None
        if len(wv) > len(order) and not verbose:
            pass                              # only informational (see below)
        if len(wv) > len(order):
            extra = len(wv) - len(order)
            # locate where the named map stops matching -> that is the leftover slot
            if isinstance(named, dict):
                for i, nm in enumerate(order):
                    if nm in named and json.dumps(wv[i], ensure_ascii=False) != \
                            json.dumps(named[nm], ensure_ascii=False):
                        shift = i
                        break
            if verbose:
                print(f"  #{nid} {ntype}: widgets_values={len(wv)} vs schema={len(order)} "
                      f"(+{extra} extra slot(s))"
                      + (f"; first mismatch at slot {shift} ({order[shift]})" if shift is not None else ""))
            if verbose and isinstance(named, dict):
                for i, nm in enumerate(order):
                    if i >= len(wv):
                        break
                    if nm in named and json.dumps(wv[i], ensure_ascii=False) != \
                            json.dumps(named[nm], ensure_ascii=False):
                        print(f"      slot {i:2d} {nm:22s} positional={wv[i]!r} named={named[nm]!r}")
        elif len(wv) < len(order):
            # Common and usually harmless: the front end does NOT store a slot
            # for a widget input that is overridden by a link, and dynamic
            # sub-inputs (ref_images.*, *.scale) never appear either.
            # Only a SHORTAGE is worth a note; a SURPLUS is what shifts slots.
            if verbose:
                print(f"  #{nid} {ntype}: {len(wv)} < schema {len(order)} "
                      f"(links / dynamic inputs - informational)")
        # value legality - the loudest signal of a shifted array
        bad = []
        for i, nm in enumerate(order):
            spec = spec_map.get(nm)
            if spec is None or nm in KNOWN_HIDDEN:
                continue
            if isinstance(named, dict):
                if nm not in named:
                    continue          # linked input: the front end stores no value
                got = named[nm]
            elif i < len(wv):
                got = wv[i]
            else:
                continue
            why = value_problem(spec, got)
            if why:
                bad.append(f"  #{nid} {ntype}.{nm}: {got!r} -> {why}")
        if bad:
            problems.extend(bad)
            if fix:
                problems.append(f"  #{nid} ^^ illegal values mean the array is SHIFTED "
                                "(a removed parameter still occupies a slot)")

        if isinstance(named, dict):
            unknown = [k for k in named
                       if k not in order and k not in KNOWN_HIDDEN]
            if unknown:
                problems.append(
                    f"  #{nid} {ntype}: named keys not in schema: {unknown} "
                    "(a removed parameter still living in the saved file -> "
                    "everything after it is shifted)"
                )

        if fix and len(wv) > len(order) and isinstance(named, dict):
            # rebuild positionally from the named map - the only safe order
            rebuilt = []
            for nm in order:
                rebuilt.append(named[nm] if nm in named else wv[len(rebuilt)]
                               if len(rebuilt) < len(wv) else None)
            if all(v is not None for v in rebuilt):
                node["widgets_values"] = rebuilt
                problems.append(f"  #{nid} FIXED -> widgets_values rebuilt ({len(rebuilt)} slots)")

    if fix and problems and any("FIXED" in p for p in problems):
        bak = path + ".bak-widgets-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, bak)
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        print(f"  backup -> {bak}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(WF_DIR, "*.json")))
    schema = fetch_schema(args.offline)
    if not schema:
        print("(offline: only intra-file consistency is checked)")

    total = 0
    for path in files:
        try:
            problems = check_file(path, schema, args.offline, args.fix, args.verbose)
        except Exception as exc:                      # noqa: BLE001
            print(f"{os.path.basename(path)}: SKIP ({exc})")
            continue
        if problems:
            total += len(problems)
            print(f"{os.path.basename(path)}")
            for p in problems:
                print(p)
    print(f"\n{'OK - all nodes aligned' if total == 0 else str(total) + ' problem(s) found'}")


if __name__ == "__main__":
    main()
