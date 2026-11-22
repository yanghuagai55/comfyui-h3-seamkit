# -*- coding: utf-8 -*-
"""Route 3 · 活跃区掩码

任务书 v2 §4.2.1 的定义：

    Active(t) = M_fg(t) ∪ HighMotionBG(t) ∪ Ring(r)

* `M_fg`          —— 人物/主体（抠像模型给，或色键）
* `HighMotionBG`  —— **局部**高频运动的背景（窗帘、水流、近景穿行物）
* `Ring(r)`       —— 主体外扩 r 个 patch（阴影、接触边缘、反光）

关键在「局部」：全局运镜（推拉摇移）**不算** HighMotionBG，它由压缩的 BG 关键帧承载。
所以这里先估计逐帧的全局位移场、把它补偿掉，再看残差——否则一推镜整帧都会判成前景，
压缩率直接归零（这是纯帧差方案在 Case-2 上的致命伤）。
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F

from .assembly import estimate_tile_flow, warp

__all__ = [
    "dilate", "motion_energy", "micro_motion_guard", "auto_chroma_fg",
    "semantic_fg_mask", "auto_fg_mask", "active_mask", "mask_stats",
]


def dilate(mask, r: int = 1):
    """[..., H, W] bool/float → 外扩 r 格（方形结构元）。"""
    if r <= 0:
        return mask
    m = mask.float().unsqueeze(0)
    m = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)
    return m.squeeze(0) > 0.5


@torch.no_grad()
def motion_energy(frames, tiles=(6, 8), search=4, blur=1):
    """逐帧「局部」运动能量（已扣掉全局位移）。

    frames : [T,C,H,W] 像素空间（0~1），建议先降到 ~256 宽的预览分辨率再算
    返回    [T,H,W]，t=0 记 0
    """
    T, C, H, W = frames.shape
    tiles = (max(1, min(int(tiles[0]), H)), max(1, min(int(tiles[1]), W)))
    e = torch.zeros(T, H, W, device=frames.device, dtype=frames.dtype)
    prev = frames[0].mean(0, keepdim=True)
    for t in range(1, T):
        cur = frames[t].mean(0, keepdim=True)
        fl = estimate_tile_flow(prev[0], cur[0], tiles=tiles, search=search)
        aligned = warp(prev, fl)
        e[t] = (cur - aligned).abs().mean(0)
        prev = cur
    if blur > 0:
        k = 2 * blur + 1
        e = F.avg_pool2d(e.unsqueeze(1), k, stride=1, padding=blur).squeeze(1)
    return e


def micro_motion_guard(e, low_rel=0.04, persist=0.6, min_frames=3):
    """静止人物的保底：低阈值 + 时间持续性。

    盲区：人物站着不动，只有口型/眨眼/呼吸。这类微动的能量远小于画面里真正在动的
    东西（窗帘、车流），用 `rel*max` 这种单一阈值会把静立的人判进 BG → 微表情发糊。

    为什么用「持续性」而不是单纯降阈值：压缩噪声/传感器噪声是随机闪烁，同一个像素
    不会连续多帧都超阈；而呼吸、眨眼是持续发生的。所以「低阈值 + 出现帧数占比」能把
    两者分开，且不需要任何额外模型。

    e: [T,H,W] 运动能量；返回 [T,H,W] bool（逐帧相同，保证静立人物整段都被保护）
    """
    mx = float(e.max())
    T = e.shape[0]
    if mx <= 0:
        return torch.zeros_like(e, dtype=torch.bool)
    hits = (e > low_rel * mx).float()
    need = max(int(min_frames), int(round(persist * T)))
    pers = hits.sum(0) >= need
    return pers.unsqueeze(0).expand(T, *pers.shape).contiguous()


def auto_chroma_fg(frames, green_margin=0.12, dist_thr=0.18):
    """检测纯绿幕镜头并提取完整人物前景掩码；非绿幕镜头返回 None（不干扰实景）。

    frames: [T, 3, H, W] float in [0, 1]
    返回 [T, H, W] bool 或 None
    """
    if frames.ndim != 4 or frames.shape[1] < 3:
        return None
    r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]
    bw = max(2, min(8, frames.shape[-1] // 20))
    bh = max(2, min(8, frames.shape[-2] // 20))
    br = torch.cat([r[:, :bh, :].flatten(), r[:, :, :bw].flatten(), r[:, :, -bw:].flatten()]).median()
    bg = torch.cat([g[:, :bh, :].flatten(), g[:, :, :bw].flatten(), g[:, :, -bw:].flatten()]).median()
    bb = torch.cat([b[:, :bh, :].flatten(), b[:, :, :bw].flatten(), b[:, :, -bw:].flatten()]).median()
    if float(bg - max(br, bb)) <= float(green_margin):
        return None
    dist = ((r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2).sqrt()
    fg = dist > float(dist_thr)
    cov = float(fg.float().mean())
    if cov < 0.01 or cov > 0.85:
        return None
    return fg


def _find_bria_rmbg_weight():
    roots = [
        os.environ.get("WORK_ROOT", ""),
        r"D:/comfyui/ComfyUI/models",
        "/content/ComfyUI/models",
    ]
    pats = ("**/RMBG-1.4*.pth", "**/rmbg*.pth", "**/bria*rmbg*.pth")
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for pat in pats:
            hits = glob.glob(os.path.join(r, pat), recursive=True)
            if hits:
                return hits[0]
    return None


def _sample_key_indices(T, max_frames):
    if max_frames is None or T <= max_frames:
        return list(range(T))
    return sorted(set(int(round(i * (T - 1) / max(1, max_frames - 1))) for i in range(max_frames)))


def _expand_temporal_mask(sub_mask, indices, T):
    """把抽样帧掩码 [K, H, W] 最近邻展开回全帧 [T, H, W] bool。"""
    if len(indices) == T:
        return sub_mask.to(torch.bool)
    idx_t = torch.tensor(indices, dtype=torch.long)
    full = torch.zeros(T, sub_mask.shape[1], sub_mask.shape[2], dtype=torch.bool, device=sub_mask.device)
    for t in range(T):
        k = int((idx_t - t).abs().argmin())
        full[t] = sub_mask[k]
    return full


@torch.no_grad()
def _run_bria_rmbg(frames_sub, weight_path, device):
    import sys
    import numpy as np
    from PIL import Image
    easy_dirs = [
        r"D:/comfyui/ComfyUI/custom_nodes/comfyui-easy-use/py/modules/briaai",
        "/content/ComfyUI/custom_nodes/comfyui-easy-use/py/modules/briaai",
    ]
    for d in easy_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
            break
    from rembg import BriaRMBG, preprocess_image, postprocess_image  # type: ignore

    net = BriaRMBG()
    net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    net.to(device).eval()
    K, _, H, W = frames_sub.shape
    out = torch.zeros(K, H, W, dtype=torch.bool)
    for i in range(K):
        arr = (frames_sub[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(arr)
        inp = preprocess_image(im, [1024, 1024]).to(device)
        res = net(inp)[0][0]
        m = postprocess_image(res, (H, W))
        out[i] = torch.from_numpy(m) > 127
    del net
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _run_torchvision_seg(frames_sub, device, allow_download=True):
    """通用实景语义前景分割（LRASPP MobileNetV3，仅 13MB，纯 PyTorch GPU，覆盖人/动物/车辆/常见前景物体）。"""
    import torchvision.models.segmentation as tvseg
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

    hub_dir = torch.hub.get_dir()
    cached = glob.glob(os.path.join(hub_dir, "checkpoints", "*lraspp_mobilenet_v3_large*.pth"))
    if not cached and not allow_download:
        return None
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = tvseg.lraspp_mobilenet_v3_large(weights=weights).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    K, _, H, W = frames_sub.shape
    out = torch.zeros(K, H, W, dtype=torch.bool)
    bs = 8
    for i in range(0, K, bs):
        batch = frames_sub[i:i + bs].to(device=device, dtype=torch.float32)
        # 保持宽高比缩到短边 ~320 以兼顾速度与精度
        scale = 320.0 / max(1, min(H, W))
        nh, nw = max(32, int(round(H * scale))), max(32, int(round(W * scale)))
        x = F.interpolate(batch, size=(nh, nw), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        logits = model(x)["out"]  # [B, 21, nh, nw]；class 0 = background，1..20 = foreground objects
        fg_prob = 1.0 - torch.softmax(logits, dim=1)[:, 0:1]
        fg_up = F.interpolate(fg_prob, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        out[i:i + bs] = (fg_up > 0.45).cpu()
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _run_rembg_pkg(frames_sub, allow_download=True):
    """通用显著性抠图（rembg u2net，支持任意人物/动漫/物体实景）。"""
    import numpy as np
    from PIL import Image
    import rembg

    u2_home = os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))
    has_cache = os.path.exists(os.path.join(u2_home, "u2net.onnx"))
    if not has_cache and not allow_download:
        return None
    sess = rembg.new_session("u2net")
    K, _, H, W = frames_sub.shape
    out = torch.zeros(K, H, W, dtype=torch.bool)
    for i in range(K):
        arr = (frames_sub[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        m = np.asarray(rembg.remove(Image.fromarray(arr), session=sess, only_mask=True))
        if m.ndim == 3:
            m = m[..., 0]
        out[i] = torch.from_numpy(m.copy()) > 127
    return out


def semantic_fg_mask(frames, backend="auto", device=None, max_frames=37, allow_download=True):
    """通用实景视频前景主体分割（不限绿幕，覆盖人物/动漫/车辆/物体）。

    frames : [T, 3, H, W] float in [0, 1]
    backend: 'auto' | 'bria_rmbg' | 'torchvision' | 'rembg' | 'none'
    返回 (fg_mask [T, H, W] bool 或 None, 实际命中的后端名 str)
    """
    if backend in ("none", "motion") or frames.ndim != 4 or frames.shape[1] < 3:
        return None, "motion_only"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = frames.shape[0]
    indices = _sample_key_indices(T, max_frames)
    sub = frames[indices].detach().cpu()

    if backend == "auto":
        # 优先零等待使用本机已有权重：BriaRMBG(最佳通用/二次元) -> rembg(若已缓存) -> torchvision(12.5MB 快速兜底)
        u2_home = os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))
        has_u2 = os.path.exists(os.path.join(u2_home, "u2net.onnx"))
        order = ["bria_rmbg", "rembg", "torchvision"] if has_u2 else ["bria_rmbg", "torchvision", "rembg"]
    else:
        order = [backend]
    for name in order:
        try:
            sub_m = None
            if name == "bria_rmbg":
                w = _find_bria_rmbg_weight()
                if w:
                    sub_m = _run_bria_rmbg(sub, w, dev)
            elif name == "torchvision":
                sub_m = _run_torchvision_seg(sub, dev, allow_download=allow_download)
            elif name == "rembg":
                sub_m = _run_rembg_pkg(sub, allow_download=allow_download)
            if sub_m is not None:
                cov = float(sub_m.float().mean())
                if 0.005 <= cov <= 0.85:
                    return _expand_temporal_mask(sub_m, indices, T).to(frames.device), name
        except Exception:
            continue
    return None, "motion_only"


def auto_fg_mask(frames, backend="auto", device=None, max_frames=37, allow_download=True):
    """全场景自动前景提取：绿幕秒级色键，普通实景视频自动走通用语义抠图（bria_rmbg / torchvision / rembg）。

    任何后端均不可用或纯风景无主体时，安静返回 (None, 'motion_only')，由全局运动补偿 active_mask 兜底。
    """
    if backend in ("auto", "chroma"):
        cm = auto_chroma_fg(frames)
        if cm is not None:
            return cm, "chroma"
        if backend == "chroma":
            return None, "motion_only"
    return semantic_fg_mask(frames, backend=backend, device=device,
                            max_frames=max_frames, allow_download=allow_download)


def active_mask(frames, fg_mask=None, ring=1, rel=0.15, floor=1e-3, thr=None,
                tiles=(6, 8), search=4, patch_grid=None, micro=True,
                micro_rel=0.04, persist=0.6):
    """算出 `Active` 掩码，并按需要降采样到 patch 网格。

    frames   : [T,C,H,W]，像素空间（或 latent 空间）
    fg_mask  : [T,H,W] bool 或 None（可由 auto_fg_mask 给出，覆盖绿幕与任意普通实景视频）
    patch_grid: (gh, gw) —— 直接给目标网格；不给就用 out_size=(H,W) 自己减半
    返回      ([T,gh,gw] bool, info dict)
    """
    T, C, H, W = frames.shape
    e = motion_energy(frames, tiles=tiles, search=search)
    if thr is None:
        # 稳健尺度：用 99.5% 分位封顶，防止单点高速肢体（如快速挥手）把阈值拉高 5~8 倍而漏掉静立躯干
        e_max = float(e.max())
        q995 = float(torch.quantile(e.reshape(-1).float(), 0.995)) if e.numel() > 256 else e_max
        ref_scale = min(e_max, max(q995 * 2.0, float(floor)))
        thr = max(float(rel) * ref_scale, float(floor))
    hm = e > thr
    act = hm if not micro else (hm | micro_motion_guard(e, low_rel=micro_rel, persist=persist))
    if fg_mask is not None:
        fm = fg_mask.to(device=act.device, dtype=torch.bool)
        if fm.shape[-2:] != (H, W):
            fm = F.interpolate(fm.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1) > 0.5
        if fm.shape[0] != T:
            idx = torch.linspace(0, fm.shape[0] - 1, T).long().to(fm.device)
            fm = fm[idx]
        act = act | fm
    if ring > 0:
        act = torch.stack([dilate(act[t], ring) for t in range(T)])

    if patch_grid is None:
        gh, gw = max(1, H // 2), max(1, W // 2)
    else:
        gh, gw = patch_grid
    # 面积平均后过半即算活跃（保守：宁可多留前景）
    pooled = F.adaptive_avg_pool2d(act.float(), (gh, gw))
    return pooled > 0.5, {"thr": round(float(thr), 5), "energy": e,
                          "micro": bool(micro)}


def mask_stats(mask):
    """mask: [T,gh,gw] bool → 每帧活跃比例 + 平均。"""
    per = mask.float().flatten(1).mean(1)
    return {"per_frame": [round(float(v), 4) for v in per],
            "mean": round(float(per.mean()), 4)}
