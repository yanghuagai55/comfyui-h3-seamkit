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


def micro_motion_guard(e, low_rel=0.04, persist=0.6, min_frames=3, floor=0.004):
    """静止人物的保底：低阈值 + 时间持续性。

    盲区：人物站着不动，只有口型/眨眼/呼吸。这类微动的能量远小于画面里真正在动的
    东西（窗帘、车流），用 `rel*max` 这种单一阈值会把静立的人判进 BG → 微表情发糊。

    为什么用「持续性」而不是单纯降阈值：压缩噪声/传感器噪声是随机闪烁，同一个像素
    不会连续多帧都超阈；而呼吸、眨眼是持续发生的。同时必须带物理底噪下限 `floor`
    （默认 0.004 ≈ 1/255 灰度级），否则纯全局运镜完全对齐后 `e.max()≈0.002` 时
    会把 1e-4 级浮点残差误判成大片活跃区。

    e: [T,H,W] 运动能量；返回 [T,H,W] bool（逐帧相同，保证静立人物整段都被保护）
    """
    mx = float(e.max())
    T = e.shape[0]
    if mx <= float(floor):
        return torch.zeros_like(e, dtype=torch.bool)
    thr_low = max(low_rel * mx, float(floor))
    hits = (e > thr_low).float()
    need = max(int(min_frames), int(round(persist * T)))
    pers = hits.sum(0) >= need
    return pers.unsqueeze(0).expand(T, *pers.shape).contiguous()


def auto_chroma_fg(frames, green_margin=0.12, dist_thr=0.14):
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


def _detect_frame_cuts(frames_ref, cut_ratio=3.5, cut_floor=0.12):
    """检测像素/latent 序列中的硬切镜帧下标（返回新镜头起始帧列表）。"""
    T = int(frames_ref.shape[0])
    if T < 3:
        return []
    f = frames_ref.detach().float()
    d = (f[1:] - f[:-1]).abs().mean(dim=tuple(range(1, f.ndim)))
    med = float(d.median().item()) if d.numel() else 0.0
    thr = max(float(cut_ratio) * max(med, 1e-6), float(cut_floor))
    return [int(t + 1) for t in range(T - 1) if float(d[t].item()) > thr]


def _rembg_model_dir():
    """rembg 权重目录：严格放在 ComfyUI/models/background_removal（D 盘），绝不写 C 盘用户目录！"""
    env_dir = os.environ.get("U2NET_HOME", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "background_removal")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        pass
    for cand in (r"D:/comfyui/ComfyUI/models/background_removal", "/content/ComfyUI/models/background_removal"):
        if os.path.isdir(cand):
            return cand
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "models", "background_removal")
    os.makedirs(d, exist_ok=True)
    return d


def _has_rembg_weight(model_dir):
    for rel in (
        os.path.join("models", "isnet-general-use", "isnet-general-use.onnx"),
        "isnet-general-use.onnx",
        os.path.join("models", "u2net", "u2net.onnx"),
        "u2net.onnx",
    ):
        if os.path.isfile(os.path.join(model_dir, rel)):
            return True
    return False


def _sample_key_indices(T, max_frames, cuts=None):
    if max_frames is None or T <= max_frames:
        return list(range(T))
    idx = set(int(round(i * (T - 1) / max(1, max_frames - 1))) for i in range(max_frames))
    # 确保每个硬切镜两侧紧邻帧都被精确抽样，绝不跨切镜最近邻错配！
    for c in (cuts or ()):
        if 0 < c < T:
            idx.add(c - 1)
            idx.add(c)
    return sorted(idx)


def _expand_temporal_mask(sub_mask, indices, T, cuts=None):
    """把抽样帧掩码 [K, H, W] 在同镜头段内最近邻展开回全帧 [T, H, W] bool（绝不跨切镜串帧）。"""
    if len(indices) == T:
        return sub_mask.to(torch.bool)
    cut_list = sorted(set(int(c) for c in (cuts or ()) if 0 < int(c) < T))
    bounds = [0] + cut_list + [T]
    full = torch.zeros(T, sub_mask.shape[1], sub_mask.shape[2], dtype=torch.bool, device=sub_mask.device)
    for si in range(len(bounds) - 1):
        lo, hi = bounds[si], bounds[si + 1]
        seg_pairs = [(k, t_idx) for k, t_idx in enumerate(indices) if lo <= t_idx < hi]
        if not seg_pairs:
            # 兜底：若极短镜头无抽样帧则取全局最近
            idx_t = torch.tensor(indices, dtype=torch.long)
            for t in range(lo, hi):
                full[t] = sub_mask[int((idx_t - t).abs().argmin())]
            continue
        ks = [p[0] for p in seg_pairs]
        ts = torch.tensor([p[1] for p in seg_pairs], dtype=torch.long)
        for t in range(lo, hi):
            j = int((ts - t).abs().argmin())
            full[t] = sub_mask[ks[j]]
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


# PASCAL VOC / COCO-21 动态前景类（人/动物/交通工具），刻意剔除静止家具
# （chair=9, diningtable=11, pottedplant=16, sofa=18, tvmonitor=20, bottle=5），
# 防止室内静止沙发/电视白白吃掉 15%~20% 前景 token 预算。
_VOC_DYNAMIC_CLASSES = (1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19)


@torch.no_grad()
def _run_torchvision_seg(frames_sub, device, allow_download=True):
    """通用实景语义前景分割（LRASPP MobileNetV3，仅 13MB，过滤静止家具，只保留人/动物/交通工具）。"""
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
    dyn_idx = torch.tensor(_VOC_DYNAMIC_CLASSES, dtype=torch.long, device=device)
    K, _, H, W = frames_sub.shape
    out = torch.zeros(K, H, W, dtype=torch.bool)
    bs = 8
    for i in range(0, K, bs):
        batch = frames_sub[i:i + bs].to(device=device, dtype=torch.float32)
        scale = 320.0 / max(1, min(H, W))
        nh, nw = max(32, int(round(H * scale))), max(32, int(round(W * scale)))
        x = F.interpolate(batch, size=(nh, nw), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        logits = model(x)["out"]  # [B, 21, nh, nw]
        probs = torch.softmax(logits, dim=1)
        fg_prob = probs.index_select(1, dyn_idx).sum(dim=1, keepdim=True)
        fg_up = F.interpolate(fg_prob, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        out[i:i + bs] = (fg_up > 0.35).cpu()
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _run_rembg_pkg(frames_sub, allow_download=True):
    """通用/二次元高精度显著性抠图（优先 isnet-general-use，权重固定落盘 ComfyUI/models/background_removal 不占 C 盘）。"""
    import numpy as np
    from PIL import Image

    model_dir = _rembg_model_dir()
    os.environ["U2NET_HOME"] = model_dir
    has_cache = _has_rembg_weight(model_dir)
    if not has_cache and not allow_download:
        return None
    import rembg
    # 优先 isnet-general-use（对二次元与实景人物发丝/肢体精度远超 torchvision COCO 与 u2net）
    model_name = "isnet-general-use"
    if (not os.path.isfile(os.path.join(model_dir, "models", "isnet-general-use", "isnet-general-use.onnx"))
            and not os.path.isfile(os.path.join(model_dir, "isnet-general-use.onnx"))):
        if os.path.isfile(os.path.join(model_dir, "u2net.onnx")) or os.path.isfile(os.path.join(model_dir, "models", "u2net", "u2net.onnx")):
            model_name = "u2net"
    sess = rembg.new_session(model_name, providers=["CPUExecutionProvider"])
    K, _, H, W = frames_sub.shape
    out = torch.zeros(K, H, W, dtype=torch.bool)
    for i in range(K):
        arr = (frames_sub[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        m = np.asarray(rembg.remove(Image.fromarray(arr), session=sess, only_mask=True))
        if m.ndim == 3:
            m = m[..., 0]
        out[i] = torch.from_numpy(m.copy()) > 100
    return out


def _temporal_bridge_same_shot(mask_thw, frames_ref, cut_ratio=3.5, cut_floor=0.12):
    """在同镜头内做 3 帧形态学闭运算 `m[t] |= (m[t-1] & m[t+1])`：
    - 填平动漫 12fps（一拍二）导致的隔帧 0/1 闪动；
    - 桥接单帧语义漏检（如动漫近景单帧掉检人脸）；
    - 自动跳过镜头硬切边界，绝不把上一镜掩码串进下一镜。
    """
    T = int(mask_thw.shape[0])
    if T < 3:
        return mask_thw
    f = frames_ref.detach().float()
    d = (f[1:] - f[:-1]).abs().mean(dim=tuple(range(1, f.ndim)))
    med = float(d.median().item()) if d.numel() else 0.0
    thr = max(float(cut_ratio) * max(med, 1e-6), float(cut_floor))
    is_cut = [bool(float(d[i].item()) > thr) for i in range(T - 1)]
    out = mask_thw.clone()
    for t in range(1, T - 1):
        if (not is_cut[t - 1]) and (not is_cut[t]):
            out[t] = out[t] | (mask_thw[t - 1] & mask_thw[t + 1])
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
    cuts = _detect_frame_cuts(frames)
    eff_max = min(T, max(int(max_frames or 37), T // 4))
    indices = _sample_key_indices(T, eff_max, cuts=cuts)
    sub = frames[indices].detach().cpu()

    if backend == "auto":
        # 优先零等待使用本机已有权重：BriaRMBG -> rembg(isnet-general-use/u2net, 存 D 盘) -> torchvision 兜底
        has_rembg = _has_rembg_weight(_rembg_model_dir())
        order = ["bria_rmbg", "rembg", "torchvision"] if has_rembg else ["bria_rmbg", "torchvision", "rembg"]
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
                sub_m = _temporal_bridge_same_shot(sub_m, sub)
                cov = float(sub_m.float().mean())
                if 0.005 <= cov <= 0.85:
                    return _expand_temporal_mask(sub_m, indices, T, cuts=cuts).to(frames.device), name
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


def active_mask(frames, fg_mask=None, ring=1, rel=0.15, floor=0.008, thr=None,
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
    # 屏蔽硬切镜当帧的全局运动能量尖峰（防止 00007 f058 硬切镜把整帧 95% 误标成前景！）
    for c_idx in _detect_frame_cuts(frames):
        e[c_idx].zero_()
    if thr is None:
        # 稳健尺度：用 99.5% 分位封顶，防止单点高速肢体（如快速挥手）把阈值拉高 5~8 倍而漏掉静立躯干
        e_max = float(e.max())
        q995 = float(torch.quantile(e.reshape(-1).float(), 0.995)) if e.numel() > 256 else e_max
        ref_scale = min(e_max, max(q995 * 2.0, float(floor)))
        thr = max(float(rel) * ref_scale, float(floor))
    if fg_mask is not None:
        # 已有高质量语义/绿幕前景掩码时：不再开启 micro_motion_guard（防窗户/窗帘微闪误入 FG），
        # 仅并入真正的强局部运动（阈值提高到 max(2.2*thr, 0.05)），保持背景掩码边界全程稳定不乱跳！
        fm = fg_mask.to(device=e.device, dtype=torch.bool)
        if fm.shape[-2:] != (H, W):
            fm = F.interpolate(fm.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1) > 0.5
        if fm.shape[0] != T:
            idx = torch.linspace(0, fm.shape[0] - 1, T).long().to(fm.device)
            fm = fm[idx]
        strong_hm = _temporal_bridge_same_shot(e > max(float(thr) * 2.2, 0.05), frames)
        act = fm | strong_hm
    else:
        hm = _temporal_bridge_same_shot(e > thr, frames)
        act = hm if not micro else (hm | micro_motion_guard(e, low_rel=micro_rel, persist=persist))
    if patch_grid is None:
        gh, gw = max(1, H // 2), max(1, W // 2)
    else:
        gh, gw = patch_grid
    # 保守空间降采样（宁留不漏）：只要格内包含运动指尖/发梢/细肢体就保留为前景，
    # 彻底杜绝 avg_pool > 0.5 面积阈值啃掉边缘缺口！
    act_f = act.float().unsqueeze(1)
    if H >= gh and W >= gw:
        pooled = F.adaptive_max_pool2d(act_f, (gh, gw)).squeeze(1) > 0.5
    else:
        pooled = F.interpolate(act_f, size=(gh, gw), mode="nearest").squeeze(1) > 0.5
    # ring 在目标 patch 网格上外扩（1 格 = 真实 1 patch 护城河，不再在 180p 上扩完又被降采样吃掉）
    if ring > 0:
        pooled = torch.stack([dilate(pooled[t], ring) for t in range(T)])
    return pooled, {"thr": round(float(thr), 5), "energy": e,
                          "micro": bool(micro)}


def mask_stats(mask):
    """mask: [T,gh,gw] bool → 每帧活跃比例 + 平均。"""
    per = mask.float().flatten(1).mean(1)
    return {"per_frame": [round(float(v), 4) for v in per],
            "mean": round(float(per.mean()), 4)}
