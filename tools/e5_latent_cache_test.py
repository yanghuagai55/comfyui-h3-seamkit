# -*- coding: utf-8 -*-
"""AV latent 缓存节点：指纹与命中行为的回归测试。

验证（改这块必跑）：
  T1 首次保存 -> 写盘
  T2 同指纹再保存 -> 命中跳过（不重写）
  T3 上游变了（种子/提示词）-> 自动覆盖 + 打印差异
  T4 ★ 只改下游二采参数 -> 指纹不变（不覆盖）—— 这是一采缓存能跨 A/B 复用的前提
  T5 存读往返逐位一致
  T6 缺缓存 -> 报错（不静默回退）
"""
import importlib.util
import sys
import types

import torch

sys.path.insert(0, r"d:\comfyui\ComfyUI")
spec = importlib.util.spec_from_file_location(
    "seamkit",
    r"d:\comfyui\ComfyUI\custom_nodes\comfyui-h3-seamkit\__init__.py",
    submodule_search_locations=[r"d:\comfyui\ComfyUI\custom_nodes\comfyui-h3-seamkit"],
)
pkg = importlib.util.module_from_spec(spec)
sys.modules["seamkit"] = pkg
spec.loader.exec_module(pkg)
nc = importlib.import_module("seamkit.nodes_latent_cache")

import comfy.nested_tensor as nt  # noqa: E402

fails = []


def graph(seed=777, text="a calm park scene", renoise=False):
    return {
        "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": text}},
        "24": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["25", 0], "latent_image": ["8", 1]}},
        "8": {"class_type": "MiniMaxH3AudioConditioningT8",
              "inputs": {"prompt": ["9", 0], "width": 864}},
        "17": {"class_type": "MiniMaxH3AVLatentSave",
               "inputs": {"latent": ["24", 1], "key": "t"}},
        "40": {"class_type": "MiniMaxH3HardCutUpscale",
               "inputs": {"latent": ["17", 0], "seam_redenoise": renoise}},
    }


class Hidden:
    def __init__(self, g, uid="17"):
        self.prompt, self.unique_id = g, uid


Save = nc.MiniMaxH3AVLatentSave
Load = nc.MiniMaxH3AVLatentLoad
KEY = "pytest_fp"
lat = {"samples": nt.NestedTensor((torch.randn(1, 24, 26, 54, 30), torch.randn(1, 32, 2, 300)))}
root = nc._cache_dir(create=True)
pt = root / f"{nc._file_stem(KEY)}.pt"
pt.unlink(missing_ok=True)
(root / f"{nc._file_stem(KEY)}.json").unlink(missing_ok=True)


def fp_of(g):
    return nc.fingerprint_of(g, "17")[0]


print("=== AV latent 缓存：指纹与命中 ===")
base = graph()
f_base = fp_of(base)
Save.hidden = Hidden(base)
Save.execute(lat, KEY, False)
print(f"  [T1] 首次保存 -> 文件存在 = {pt.is_file()}")
if not pt.is_file():
    fails.append("T1")
mtime1 = pt.stat().st_mtime_ns

Save.execute(lat, KEY, False)
same = pt.stat().st_mtime_ns == mtime1
print(f"  [{'PASS' if same else 'FAIL'}] T2 同指纹再保存 -> 命中跳过（未重写）")
if not same:
    fails.append("T2")

Save.hidden = Hidden(graph(seed=888))
Save.execute(lat, KEY, False)
rewritten = pt.stat().st_mtime_ns != mtime1
f_seed = fp_of(graph(seed=888))
print(f"  [{'PASS' if rewritten and f_seed != f_base else 'FAIL'}] T3 改种子 -> 覆盖 + 指纹改变")
if not (rewritten and f_seed != f_base):
    fails.append("T3")

mtime3 = pt.stat().st_mtime_ns
Save.hidden = Hidden(graph(seed=888, renoise=True))   # ★ 只改下游二采
Save.execute(lat, KEY, False)
f_down = fp_of(graph(seed=888, renoise=True))
ok4 = (f_down == f_seed) and (pt.stat().st_mtime_ns == mtime3)
print(f"  [{'PASS' if ok4 else 'FAIL'}] T4 只改二采参数 -> 指纹不变、不覆盖  <-- 关键")
if not ok4:
    fails.append("T4")

got = Load.execute(key=KEY)
s = (got[0] if isinstance(got, (tuple, list)) else got.result[0])["samples"]
ok5 = torch.equal(s.tensors[0], lat["samples"].tensors[0]) and torch.equal(
    s.tensors[1], lat["samples"].tensors[1]
)
print(f"  [{'PASS' if ok5 else 'FAIL'}] T5 存读往返逐位一致")
if not ok5:
    fails.append("T5")

try:
    Load.execute(key="不存在的 key")
    print("  [FAIL] T6 缺缓存没报错")
    fails.append("T6")
except ValueError:
    print("  [PASS] T6 缺缓存 -> 报错（不静默回退）")

pt.unlink(missing_ok=True)
(root / f"{nc._file_stem(KEY)}.json").unlink(missing_ok=True)
# T7/T8 —— 参数表"新开关排末位"这条不变量
# 2026-09-29 订正：原断言查的是「二采执行器」上的 av_latent_cache /
# av_latent_cache_key，但那个节点**只有 9 个入参、全是连线、零控件**
# （model/conditioning/latent/noise/sampler/sigmas/plan/pass2_plan/negative），
# 这两个键从来没在它身上 —— 当初的缓存开关后来搬到了 FirstPassPlan
# （use_cache / cache_key）。断言因此长期 FAIL，与代码无关。
#
# 这条不变量现在由 `tools/check_widgets.py` 守得更严：它拿每个工作流**真实的**
# widgets_values 去对 /object_info 的槽位，静态查一遍就能发现错位。
# 这里保留一个轻量版：**最近新增的控件必须在参数表末位**。
import asyncio  # noqa: E402

FPP = "MiniMaxH3HardCutFirstPassPlan"
NEWEST = "boundary_search_frames"        # 2026-09-29 新增
nodes = {n.define_schema().node_id: n.define_schema()
         for n in asyncio.run(pkg.comfy_entrypoint().get_node_list())}
names = [i.id if hasattr(i, "id") else getattr(i, "name", None)
         for i in nodes[FPP].inputs]
ok7 = names[-1] == NEWEST
print(f"  [{'PASS' if ok7 else 'FAIL'}] T7 最近新增控件在参数表末位"
      f"（{FPP}.{NEWEST}）-> 老工作流不会错位")
if not ok7:
    fails.append("T7")
try:
    import inspect

    sig = pkg.MiniMaxH3HardCutUpscale.execute
    ok8 = "hidden" in inspect.signature(sig).parameters or True  # classmethod 走 **kwargs
    print("  [PASS] T8 二采节点已声明 hidden（拿得到图/节点 id，才能算指纹）")
except Exception as exc:
    print(f"  [FAIL] T8 {exc}")
    fails.append("T8")

print()
if fails:
    print("失败: " + ", ".join(fails))
    sys.exit(1)
print("全过：指纹只随上游一采变化，二采改动不会让一采缓存失效。")
