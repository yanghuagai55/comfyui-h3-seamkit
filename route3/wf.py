# -*- coding: utf-8 -*-
"""ComfyUI API 格式工作流的读写与投递（无第三方依赖，只用 urllib）

设计：**只操作 API 格式 JSON**（`{id: {class_type, inputs}}`），不碰 UI 图。
UI 图那种 nodes+links 的结构改起来极易错位；API 格式插一个节点就是加一个 dict + 改一个引用。
在 ComfyUI 里菜单 → Workflow → Export (API) 就能拿到。
"""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

__all__ = ["load", "save", "clone", "find", "set_input", "upstream", "new_node",
           "splice_before", "splice_guider_model", "patch_prompt", "patch_seed", "Comfy"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(wf, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=1)
    return path


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
    nid = str(max(int(k) for k in wf if str(k).isdigit()) + 1)
    wf[nid] = {"class_type": class_type, "inputs": inputs}
    if title:
        wf[nid].setdefault("_meta", {})["title"] = title
    return nid


def splice_before(wf, target_id, input_key, class_type, inputs, title=None):
    """把新节点插到 target 的某个模型输入之前。返回新节点 id。"""
    old = wf[target_id]["inputs"].get(input_key)
    if old is None:
        raise KeyError("node %s 没有输入 %r" % (target_id, input_key))
    nid = new_node(wf, class_type, dict(inputs), title)
    wf[nid]["inputs"][input_key] = old
    wf[target_id]["inputs"][input_key] = [nid, 0]
    return nid


def splice_guider_model(wf, sampler_id, class_type, inputs, title=None):
    """sampler 的 model 藏在 guider 里：sampler.guider → guider.model ← 这里插。

    返回新节点 id。若 guider 的 model 已经就是这个类，直接返回，避免重复插。
    """
    g = upstream(wf, wf[sampler_id]["inputs"].get("guider"))
    if not g:
        raise KeyError("sampler %s 没有 guider 输入" % sampler_id)
    gid = g[0]
    cur = upstream(wf, wf[gid]["inputs"].get("model"))
    if cur and wf[cur[0]]["class_type"] == class_type:
        return cur[0]                      # 已经插过，幂等
    return splice_before(wf, gid, "model", class_type, inputs, title)


def _walk_up(wf, nid, key="model", depth=0):
    """从某个节点沿 model 输入往上找，拿到真正的 model 来源"""
    if depth > 24:
        return None
    n = wf.get(nid)
    if n is None:
        return None
    link = n["inputs"].get(key)
    up = upstream(wf, link)
    return up[0] if up else None


def patch_prompt(wf, text, nodes=None):
    """把 prompt 写进所有条件节点上游的那个字符串节点。"""
    from_nodes = nodes or find(wf, "MiniMaxH3AudioConditioningT8")
    done = []
    for nid in from_nodes:
        link = wf[nid]["inputs"].get("prompt")
        up = upstream(wf, link)
        if not up:
            wf[nid]["inputs"]["prompt"] = text
            done.append(nid + ":direct")
            continue
        src = up[0]
        if "value" in wf[src]["inputs"]:
            wf[src]["inputs"]["value"] = text
            done.append(src)
        else:
            wf[src]["inputs"]["text"] = text
            done.append(src)
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

    def submit(self, wf):
        res = self._post("/prompt", {"prompt": wf, "client_id": self.client_id})
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
        with urllib.request.urlopen(self.server + "/view?" + q, timeout=600) as r, open(dest, "wb") as fh:
            fh.write(r.read())
        return dest

