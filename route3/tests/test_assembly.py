# -*- coding: utf-8 -*-
"""法 A / 法 B 的 A/B：已知平移下，法 B 必须明显优于线性插值"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, r"D:/comfyui/ComfyUI")
from route3.assembly import (estimate_tile_flow, warp, reconstruct_moving,
                             reconstruct_static, lerp_residual, flow_residual)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []
def check(n, ok, d=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + n + ("  " + d if d else ""))
    if not ok: FAIL.append(n)

def shift_img(x, dy, dx):
    C, H, W = x.shape
    g = torch.zeros(1, H, W, 2, device=x.device, dtype=x.dtype)
    ys = torch.arange(H, device=x.device, dtype=x.dtype) - dy
    xs = torch.arange(W, device=x.device, dtype=x.dtype) - dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g[0, ..., 0] = (gx + 0.5) / W * 2 - 1
    g[0, ..., 1] = (gy + 0.5) / H * 2 - 1
    return torch.nn.functional.grid_sample(x.unsqueeze(0), g, align_corners=False,
                                           padding_mode="border").squeeze(0)

torch.manual_seed(0)
H, W, C = 48, 80, 4
tex = torch.randn(C, H, W, device=DEV) + torch.sin(
    torch.arange(W, device=DEV).float() * 0.9)[None, None, :] * 3
D = torch.randn(C, H, W, device=DEV) * 0.3        # 细节残差
shift = (2, 4)

f0, f1, f2 = tex, shift_img(tex, *[s // 2 for s in shift]), shift_img(tex, *shift)
d0, d2 = D, shift_img(D, *shift)
z = torch.stack([f0, f1, f2])

gt1 = f1 + shift_img(D, *[s // 2 for s in shift])

fl = estimate_tile_flow(z[0], z[2], tiles=(6, 8), search=5)
med = fl.reshape(-1, 2).median(0).values
check("分块位移估计 ≈ (2,4)", abs(float(med[0]) - 2) < 1.5 and abs(float(med[1]) - 4) < 1.5,
      "median=(%.1f, %.1f)" % (med[0], med[1]))

delta_keys = {0: d0, 2: d2}
out_flow = reconstruct_moving(z, delta_keys, [0, 2], method="flow", tiles=(6, 8), search=5)
out_lerp = reconstruct_moving(z, delta_keys, [0, 2], method="lerp")
e_flow = (out_flow[1] - gt1).abs().mean().item()
e_lerp = (out_lerp[1] - gt1).abs().mean().item()
check("法 B 误差 < 法 A", e_flow < e_lerp * 0.6,
      "flow=%.4f lerp=%.4f (%.0f%% 更低)" % (e_flow, e_lerp, 100 * (1 - e_flow / max(e_lerp, 1e-9))))

st = reconstruct_static(torch.stack([f0, f1, f2]), f0)
check("静机位档：整片铺同一张高清底", bool((st[2] - f0).abs().max().item() < 1e-6))

m = torch.zeros(3, H, W, dtype=torch.bool, device=DEV); m[:, 10:30, 20:50] = True
st2 = reconstruct_static(torch.stack([f0, f1, f2]), f0, fg_mask=m)
check("静机位档：前景保留原帧",
      bool((st2[2][:, 10:30, 20:50] - f2[:, 10:30, 20:50]).abs().max().item() < 1e-6))
print("\n%d 项失败%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
sys.exit(1 if FAIL else 0)
