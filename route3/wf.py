# -*- coding: utf-8 -*-
"""ComfyUI API 格式工作流的读写、手术与投递（无第三方依赖，只用 urllib）

设计：**只操作 API 格式 JSON**（`{id: {class_type, inputs}}`），不碰 UI 图。
兼容：
  * 文武/Goohai 双采（GoohaiMinimaxH3TiledSampler + GoohaiMinimaxH3LatentUpscalerGH + Reconcile）
  * T8 双采（SamplerCustomAdvanced / MiniMaxH3StageSamplerEXPT8 + Reconcile / RFHandoff）
  * SeamKit 本机双采（MiniMaxH3HardCutUpscale + MiniMaxH3HardCutFirstPassPlan）
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

__all__ = [
    "load", "save", "atomic_json", "clone", "find", "set_input", "upstream",
    "new_node", "splice_before", "splice_guider_model", "splice_pass2_model",
    "splice_pass2_assemble", "splice_pass1_cache_load", "find_pass2_sampler",
    "disable_tiling_and_sparse", "set_t2va_mode", "enable_pass1_cache",
    "patch_prompt", "patch_seed", "patch_duration", "patch_megapixels", "Comfy",
]

_UPSCALER_CLASSES = {
    "MiniMaxH3TwoPassLatentReconcileT8Advanced",
    "GoohaiMinimaxH3LatentUpscalerGH",
    "MiniMaxH3LearnedLatentUpscaleT8",
    "MiniMaxH3RFRestartJointClockSetupEXPT8",
    "MiniMaxH3RFHandoffEXPT8",
}

_PASS2_SAMPLER_CLASSES = (
    "MiniMaxH3HardCutUpscale",
    "GoohaiMinimaxH3TiledSampler",
    "MiniMaxH3StageSamplerEXPT8",
    "SamplerCustomAdvanced",
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def save(wf, path):
    return atomic_json(path, wf)


def clone(wf):
    return copy.deepcopy(wf)


def find(wf, class_type):
    return [nid for nid, v in wf.items() if v.get("class_type") == class_type]


def upstream(wf, link):
    """['12', 0] → ('12', 0)；不是链接就返回 None"""
    if isinstance(link, list) and link and isinstance(link[0], str):
        return link[0], (link[1] if len(link) > 1 else 0)
    return None


def set_input(wf, nid, key, value):
    if nid not in wf:
        raise KeyError("node %s not in workflow" % nid)
    wf[nid]["inputs"][key] = value


def new_node(wf, class_type, inputs, title=None):
    nums = [int(k) for k in wf if str(k).isdigit()]
    nid = str((max(nums) if nums else 0) + 1)
    wf[nid] = {"class_type": class_type, "inputs": inputs}
    if title:
        wf[nid].setdefault("_meta", {})["title"] = title
    return nid


def splice_before(wf, target_id, input_key, class_type, inputs, title=None):
    """把新节点插到 target 的某个输入之前（默认透传同名口）。返回新节点 id。"""
    old = wf[target_id]["inputs"].get(input_key)
    if old is None:
        raise KeyError("node %s 没有输入 %r" % (target_id, input_key))
    nid = new_node(wf, class_type, dict(inputs), title)
    wf[nid]["inputs"][input_key] = old
    wf[target_id]["inputs"][input_key] = [nid, 0]
    return nid


def splice_guider_model(wf, sampler_id, class_type, inputs, title=None):
    """sampler 的 model 藏在 guider 里：sampler.guider → guider.model ← 这里插。幂等。"""
    g = upstream(wf, wf[sampler_id]["inputs"].get("guider"))
    if not g:
        raise KeyError("sampler %s 没有 guider 输入" % sampler_id)
    gid = g[0]
    cur = upstream(wf, wf[gid]["inputs"].get("model"))
    if cur and wf[cur[0]]["class_type"] == class_type:
        wf[cur[0]]["inputs"].update(inputs)
        return cur[0]
    return splice_before(wf, gid, "model", class_type, inputs, title)


def splice_pass2_model(wf, sampler_id, class_type, inputs, title=None):
    """同时支持 guider 型采样器（SamplerCustomAdvanced / GoohaiTiled / StageSampler）
    与直连 model 型执行器（MiniMaxH3HardCutUpscale）。
    """
    node = wf[sampler_id]
    if "guider" in node.get("inputs", {}) and upstream(wf, node["inputs"].get("guider")):
        return splice_guider_model(wf, sampler_id, class_type, inputs, title)
    if "model" in node.get("inputs", {}) and upstream(wf, node["inputs"].get("model")):
        cur = upstream(wf, node["inputs"].get("model"))
        if cur and wf[cur[0]]["class_type"] == class_type:
            wf[cur[0]]["inputs"].update(inputs)
            return cur[0]
        return splice_before(wf, sampler_id, "model", class_type, inputs, title)
    raise KeyError("pass2 sampler %s 既没有 guider 也没有 model 输入" % sampler_id)


def _traces_to_upscaler(wf, link, depth=0):
    if depth > 6:
        return False
    up = upstream(wf, link)
    if not up:
        return False
    node = wf.get(up[0])
    if not node:
        return False
    if node.get("class_type") in _UPSCALER_CLASSES:
        return True
    for key in ("av_latent", "latent", "latent_image", "learned_latent", "rf_handoff", "samples"):
        if key in node.get("inputs", {}):
            if _traces_to_upscaler(wf, node["inputs"][key], depth + 1):
                return True
    return False


def find_pass2_sampler(wf):
    """定位二采采样器/执行器节点 id（兼容 Goohai / T8 / SeamKit / 原生）。"""
    for cls in ("MiniMaxH3HardCutUpscale", "GoohaiMinimaxH3TiledSampler"):
        hits = find(wf, cls)
        if hits:
            return hits[-1]
    cands = []
    for cls in ("MiniMaxH3StageSamplerEXPT8", "SamplerCustomAdvanced"):
        cands.extend(find(wf, cls))
    for sid in cands:
        if _traces_to_upscaler(wf, wf[sid]["inputs"].get("latent_image")):
            return sid
    return cands[-1] if cands else None


def _traces_to_node(wf, link, target_id, depth=0):
    if depth > 6:
        return False
    up = upstream(wf, link)
    if not up:
        return False
    if up[0] == target_id:
        return True
    node = wf.get(up[0])
    if not node:
        return False
    for v in (node.get("inputs") or {}).values():
        if _traces_to_node(wf, v, target_id, depth + 1):
            return True
    return False


def splice_pass2_assemble(wf, sampler_id, mask_path="", s_bg=5, method="flow", only_bg=True):
    """把「一采 3D 上采样输出 (latent_p1_up)」和「二采输出 (latent_p2)」接入 MiniMaxH3Route3Assemble，
    并把所有下游对二采输出 [sampler_id, 0]（或下游 AVDecode.av_latent）的引用改指向 Assemble 节点。

    兼容三类二采拓扑：
      1. MiniMaxH3HardCutUpscale（本地一采双时钟工作流）：latent_p1_up=[sampler_id, 2] (upscaled_latent)，latent_p2=[sampler_id, 0]
      2. WenWu/Goohai（GoohaiMinimaxH3TiledSampler）：latent_p1_up=inputs.latent_image，latent_p2=[sampler_id, 0]
      3. T8 双时钟（MiniMaxH3StageSamplerEXPT8）：latent_p1_up=inputs.latent_image，latent_p2=AVDecode.av_latent
    """
    if method == "none":
        return None
    node = wf.get(sampler_id)
    if not node:
        return None
    if node.get("class_type") == "MiniMaxH3HardCutUpscale":
        p1_up = [sampler_id, 2]
    else:
        p1_up = node.get("inputs", {}).get("latent_image")
        if not upstream(wf, p1_up):
            return None

    # 寻找消费 [sampler_id, 0] 的下游端口（如 HardCutUpscale / WenWu 直接连 AVDecode）
    consumers = []
    for nid, n in wf.items():
        if nid == sampler_id:
            continue
        for k, v in (n.get("inputs") or {}).items():
            u = upstream(wf, v)
            if u and u[0] == sampler_id and u[1] == 0:
                consumers.append((nid, k))
    p2_link = [sampler_id, 0]
    if not consumers:
        # 兼容 T8 EXP：StageSampler 输出 stage_result[2] -> StageSave -> Audit -> AVDecode.av_latent
        for did in find(wf, "MiniMaxH3AVDecodeT8"):
            av_in = wf[did].get("inputs", {}).get("av_latent")
            if upstream(wf, av_in) and _traces_to_node(wf, av_in, sampler_id):
                p2_link = copy.deepcopy(av_in)
                consumers.append((did, "av_latent"))
                break
    if not consumers:
        return None

    aid = new_node(
        wf,
        "MiniMaxH3Route3Assemble",
        {
            "latent_p1_up": copy.deepcopy(p1_up),
            "latent_p2": p2_link,
            "s_bg": int(s_bg),
            "method": str(method),
            "only_bg": bool(only_bg),
            "mask_path": str(mask_path or ""),
        },
        title="Route3 Assemble (%s, s_bg=%d)" % (method, int(s_bg)),
    )
    for nid, k in consumers:
        wf[nid]["inputs"][k] = [aid, 0]
    return aid


def _bypass_model_node(wf, sid):
    src = wf[sid].get("inputs", {}).get("model")
    if src is None:
        return False
    for nid, n in wf.items():
        if nid == sid:
            continue
        for k, v in (n.get("inputs") or {}).items():
            u = upstream(wf, v)
            if u and u[0] == sid:
                wf[nid]["inputs"][k] = copy.deepcopy(src)
    del wf[sid]
    return True


def disable_tiling_and_sparse(wf, *, disable_tiling=True, disable_sol_attn=True, strip_lowvram=False):
    """在 A100 Oracle/消融时按需关闭空间 tiling、SolAttn 稀疏近似，以及（可选）本地 8GB 低显存补丁。"""
    notes = []
    if disable_tiling:
        for nid in find(wf, "GoohaiMinimaxH3TiledSampler"):
            if wf[nid]["inputs"].get("enable_tiling"):
                wf[nid]["inputs"]["enable_tiling"] = False
                notes.append("tiling_off:%s" % nid)
    if disable_sol_attn:
        for sid in list(find(wf, "SolAttnMiniMax")):
            if _bypass_model_node(wf, sid):
                notes.append("sol_attn_bypassed:%s" % sid)
    if strip_lowvram:
        for cls in ("MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward",
                    "MiniMaxH3MemoryEfficientSageAttentionPatch"):
            for sid in list(find(wf, cls)):
                if _bypass_model_node(wf, sid):
                    notes.append("lowvram_bypassed:%s(%s)" % (sid, cls))
    return notes


def _write_prompt_node(wf, nid, text, depth=0):
    if depth > 4 or nid not in wf:
        return None
    inp = wf[nid].get("inputs", {})
    # 若该节点的 prompt/text/value 仍连着上游字符串节点，继续往上追
    for key in ("prompt", "value", "text"):
        if key in inp:
            up = upstream(wf, inp[key])
            if up:
                res = _write_prompt_node(wf, up[0], text, depth + 1)
                if res:
                    return res
            inp[key] = text
            return "%s.%s" % (nid, key)
    return None


def patch_prompt(wf, text, nodes=None):
    """把 prompt 写进条件节点上游的源文本节点（兼容 CR Prompt Text / PlainPromptRecipe / PrimitiveString）。"""
    from_nodes = nodes or find(wf, "MiniMaxH3AudioConditioningT8")
    done = []
    for nid in from_nodes:
        res = _write_prompt_node(wf, nid, text)
        if res:
            done.append(res)
    return sorted(set(done))


def patch_seed(wf, seed):
    ids = []
    for nid in find(wf, "RandomNoise"):
        wf[nid]["inputs"]["noise_seed"] = int(seed)
        ids.append(nid)
    for nid, v in wf.items():
        if "noise_seed" in v.get("inputs", {}) and nid not in ids:
            v["inputs"]["noise_seed"] = int(seed)
            ids.append(nid)
    return ids


def patch_duration(wf, seconds):
    """支持 SeamKit Plan / ComfyMathExpression (a 或 values.a) / PrimitiveFloat / 17n+5 帧数 PrimitiveInt。"""
    for nid in find(wf, "MiniMaxH3HardCutFirstPassPlan"):
        wf[nid]["inputs"]["total_seconds"] = float(seconds)
        return "%s.total_seconds" % nid
    for nid in find(wf, "MiniMaxH3AudioConditioningT8"):
        up = upstream(wf, wf[nid]["inputs"].get("length"))
        if not up:
            continue
        mid = up[0]
        mnode = wf.get(mid, {})
        if mnode.get("class_type") == "ComfyMathExpression":
            for key in ("values.a", "a"):
                if key not in mnode.get("inputs", {}):
                    continue
                val = mnode["inputs"][key]
                src = upstream(wf, val)
                if src and "value" in wf[src[0]].get("inputs", {}):
                    wf[src[0]]["inputs"]["value"] = float(seconds)
                    return "%s.value" % src[0]
                if isinstance(val, (int, float)):
                    mnode["inputs"][key] = float(seconds)
                    return "%s.%s" % (mid, key)
        if mnode.get("class_type") in ("MiniMaxH3PlainPromptRecipeEXPT8",):
            up2 = upstream(wf, mnode["inputs"].get("length"))
            if up2 and "value" in wf[up2[0]].get("inputs", {}):
                frames = max(5, int(round(float(seconds) * 24)))
                frames += (5 - (frames % 17)) % 17
                wf[up2[0]]["inputs"]["value"] = frames
                return "%s.value(%df)" % (up2[0], frames)
    for nid in find(wf, "PrimitiveFloat"):
        v = wf[nid]["inputs"].get("value")
        if isinstance(v, (int, float)) and 1 <= v <= 60:
            wf[nid]["inputs"]["value"] = float(seconds)
            return "%s.value" % nid
    return None


def set_t2va_mode(wf):
    """把 Ref2VA 工作流切到纯 T2VA（移除 ref_images.* 引用，避免纯文生视频提示词撞 <Picture 1> 校验）。"""
    changed = []
    for nid in find(wf, "MiniMaxH3AudioConditioningT8"):
        inp = wf[nid].get("inputs", {})
        if str(inp.get("task_type", "")).startswith("Ref2VA"):
            inp["task_type"] = "T2VA"
            for k in [k for k in list(inp) if k.startswith("ref_images")]:
                del inp[k]
            changed.append(nid)
    return changed


def enable_pass1_cache(wf, cache_key, cache_path=""):
    """若工作流带 MiniMaxH3HardCutFirstPassPlan，开启一采落盘缓存（Stage 0 存档用）。"""
    ids = find(wf, "MiniMaxH3HardCutFirstPassPlan")
    for nid in ids:
        wf[nid]["inputs"]["use_cache"] = True
        wf[nid]["inputs"]["cache_key"] = str(cache_key)
        if cache_path:
            wf[nid]["inputs"]["cache_path"] = str(cache_path)
    return ids


def splice_pass1_cache_load(wf, sampler_id, cache_key, cache_path=""):
    """Stage 1 零秒跳过整条一采子图（含 LOW Conditioning）且**保留原声轨**：
    用 `MiniMaxH3AVLatentLoad`（非内置 LoadLatent，完整还原 NestedTensor(video, audio)）
    替换二采执行器 `#40.inputs.latent`（或上游 3D Upscaler 的 latent 输入）。
    ComfyUI 执行器会自动剪枝无下游的一采整条链，且不清零音频。
    """
    node = wf.get(sampler_id)
    if not node or not cache_key:
        return None
    target_id, target_key = None, None
    if node.get("class_type") == "MiniMaxH3HardCutUpscale" and "latent" in node.get("inputs", {}):
        target_id, target_key = sampler_id, "latent"
    else:
        for cls in ("GoohaiMinimaxH3LatentUpscalerGH", "MiniMaxH3LearnedLatentUpscaleT8"):
            hits = find(wf, cls)
            if hits:
                uid = hits[-1]
                for k in ("av_latent", "latent", "samples"):
                    if k in wf[uid].get("inputs", {}):
                        target_id, target_key = uid, k
                        break
                if target_id:
                    break
    if not target_id:
        return None
    lid = new_node(
        wf,
        "MiniMaxH3AVLatentLoad",
        {"key": str(cache_key), "cache_path": str(cache_path or "")},
        title="Load Frozen Pass-1 AV Latent (%s)" % cache_key,
    )
    wf[target_id]["inputs"][target_key] = [lid, 0]
    return lid


def set_pass_megapixels(wf, first=None, second=None):
    """分别设一采/二采的 MP。second 是最终画质上限（1.3 ≈ 1536x864）。

    二采不是越高越好：8GB 目标机跑不到 1080p（2.1MP），而且 1.3MP 是这条链的甜点。
    """
    touched = []
    for nid in find(wf, "MiniMaxH3HardCutFirstPassPlan"):
        if first:
            wf[nid]["inputs"]["first_megapixels"] = float(first)
            touched.append("%s.first=%.2f" % (nid, first))
        if second:
            wf[nid]["inputs"]["second_megapixels"] = float(second)
            touched.append("%s.second=%.2f" % (nid, second))
    if second:
        for nid in find(wf, "ResolutionSelector"):
            wf[nid]["inputs"]["megapixels"] = float(second)
            touched.append("%s.rs=%.2f" % (nid, second))
    return touched


def patch_megapixels(wf, mp):
    ids = find(wf, "ResolutionSelector")
    for nid in ids:
        wf[nid]["inputs"]["megapixels"] = float(mp)
    for nid in find(wf, "MiniMaxH3HardCutFirstPassPlan"):
        wf[nid]["inputs"]["second_megapixels"] = float(mp)
        ids.append(nid)
    return ids


class Comfy:
    def __init__(self, server="http://127.0.0.1:8188", timeout=1800):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.server + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def _get(self, path, timeout=30):
        with urllib.request.urlopen(self.server + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def has_node(self, class_type):
        try:
            info = self._get("/object_info/%s" % urllib.parse.quote(class_type), timeout=10)
            return class_type in info
        except Exception:
            return False

    def submit(self, wf):
        res = self._post("/prompt", {"prompt": wf, "client_id": self.client_id})
        if res.get("node_errors"):
            raise RuntimeError("节点校验失败: %s" % json.dumps(res["node_errors"], ensure_ascii=False)[:1000])
        if "prompt_id" not in res:
            raise RuntimeError("提交被拒: %s" % json.dumps(res, ensure_ascii=False)[:800])
        return res["prompt_id"]

    def wait(self, pid, poll=3.0):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            h = self._get("/history/%s" % pid)
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error" or st.get("completed") is False:
                    msgs = json.dumps(st.get("messages", []), ensure_ascii=False)[:600]
                    raise RuntimeError("执行失败: " + msgs)
                return h[pid]
            time.sleep(poll)
        raise TimeoutError("等待 %s 超过 %ds" % (pid, self.timeout))

    def files(self, hist):
        out = []
        for nid, o in (hist.get("outputs") or {}).items():
            for key in ("gifs", "videos", "images", "audio"):
                for f in o.get(key, []) or []:
                    if isinstance(f, dict) and f.get("filename"):
                        out.append((nid, f))
        return out

    def download(self, f, dest):
        q = urllib.parse.urlencode({"filename": f["filename"],
                                    "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        tmp = dest + ".tmp"
        with urllib.request.urlopen(self.server + "/view?" + q, timeout=600) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        os.replace(tmp, dest)
        return dest
