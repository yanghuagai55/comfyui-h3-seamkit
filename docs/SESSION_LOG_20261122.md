# 工作日志与明日交接单 · 2026-11-22

> 今日核心产出：A100 云端 4 步消融收口 + 本机 8GB `s_bg=4` 证伪定案 + 10s 双镜动漫实片 4 大 Bug 定位与全量修复 + 项目古早/像素修缝节点大清理 + 两大参数节点瘦身并适配 Route 3。

---

## 一、今日已完成事项

### 1. A100-40GB 云端实测与全帧核验
- **`s_bg=3 + method="lerp" + auto-beta0` 确立为默认主推**：
  - 5s / `1.3MP (1568×864)` 三场景 **3/3 全绿**：`case1_static` `28.45 dB / SSIM 0.9299 / bg_hf 0.941`，`case2_dolly` `24.82 / 0.8346 / 0.941`（超过跨 seed 稠密 Oracle `24.13 / 0.7998`），`case3_neon` `25.39 / 0.8933 / 0.987`。
  - 淘汰光流 `flow` 作为默认项：`lerp` 耗时 `<1ms`（省去 `50–150s` Python 循环）且 `bg_hf` 三场景全面胜出。
  - 15s（362 帧）：采样器单步 **`3.04×`**（`22.4 → 7.36 s/it`），端到端 `288s → 220s`（`1.31×`，受一采/解码/换模稀释）；全帧时序分析证实 2 窗 `s3_lerp` 在切缝 `f187` 处零跳变，保留稠密 **`95.3%`** 高频锐度。
  - 8GB 补丁（`MiniMaxLowVRAMAttention(6)` + `MiniMaxChunkFeedForward(4)`）零画质回退。

### 2. 本机 8GB RTX 4060 真机验证 + `s_bg=4` 机制证伪实验
在 `case2_dolly`（5s `1312×736`, `T=37`, 32kHz 立体声）冻结同一份一采，实测：

| 变体 | `s_bg` | `β` | 二采耗时 | 峰值显存 | `180p fg_psnr` | `fg_ssim` | `bg_hf` | `全清 bg_psnr` | 肉眼伪影 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s3_lerp` | 3 | `1.046` *(auto)* | `130.9s` | `7801 MiB` | `22.66` | `0.8443` | `0.969` | **`24.89`** | **零色斑** |
| `s4_default` | 4 | `1.308` *(auto)* | `114.8s` | `7583 MiB` | `21.69` | `0.8131` | `1.011` ↑ | `22.95` | 严重黄褐矩形贴块（复现 A100） |
| `s4_beta105` | 4 | `1.046` *(同 `s3`)* | `114.7s` | `7583 MiB` | `21.94` | `0.8243` | `1.009` ↑ | `23.12` | 色斑依旧严重 |
| **`s4_beta050`** | **4** | **`0.500`** *(`0.38×`)* | **`114.7s`** | `7583 MiB` | **`22.78`** | **`0.8472`** | **`0.987`** | **`23.91`** | **~95% 色斑消失，前景反超 `s3`** |

- **机制定案**：既非纯「2-hop 不可救」，也非「`β=1.05` 跨步长通用安全值」，而是**跨步长相关性衰减 + 未衰减 `β0=ln(s)` 过冲的耦合失稳**。已在 `route3/engine.py` / `route3/nodes.py` 新增 `beta_scale` 与 `beta_max` 控件。

### 3. 10s 双镜动漫实片（`exp_4v10a_00002`）逐帧体检抓出的 4 个真问题与修复
用户反馈「场景一直在跳、有方框颗粒、背景没用新东西变出来」，逐帧拆图 + 掩码叠加定位到全部根因并已完成代码修复：

| # | 现象 | 根因 | 今日已完成修复 |
|---|---|---|---|
| **1** | **Shot 2 特写脸每隔 3 个 token 变黑变糊再弹回**（`f199..203` 整脸糊成褐色双嘴，`f204` 弹回） | **多窗掩码错位 Bug**：10s 分 2 窗，`h3_upscale.py` 跑第 2 窗时没把 `start_token=35` 传给 `Route3Apply`，第 2 窗全程错用了 Shot 1（左小人）掩码，把 Shot 2 正中央大特写人脸当背景丢掉 2/3 token | `h3_upscale.py` 在每窗采样前把 `route3_start_token` 注入 `transformer_options`；`route3/patch.py` + `mask_provider.py` 按 `start_token` 精确切片对应窗口帧（新增单测通过） |
| **2** | **`f118..f121` 巨型 `32×32` 蓝白方框马赛克 + 切镜叠影** | ① `Route3Assemble` 跨镜头硬切盲目线性插值，把 Shot 2 蓝天残差糊到 Shot 1 尾帧；② `torch.where(fg_lat, b, bg)` 是硬二值 `2×2` patch（`32×32px`）阶跃；③ `#102` 遗留 `green_screen_mode=True` 禁用硬切强行叠化 | ① `route3/assembly.py` 新增 `detect_latent_cuts()` 自动按镜头硬切断开插值，绝不跨镜借残差；② `route3/nodes.py` 改为前景核心 100% 保真 + 外缘 3×3 空间软羽化融合；③ `#102` 默认改回 `green_screen_mode=False`（新增单测通过） |
| **3** | **二次元人脸单帧漏检（`f123/124`）、静止沙发占 15% FG、12fps 掩码隔帧狂闪** | `torchvision` COCO 21 类把 `sofa/chair/tv/table/plant` 当成前景且对动漫脸弱；12fps 动漫一拍二导致逐帧运动差分隔帧 0/1 闪 | `route3/masks.py` 过滤 COCO 静止家具类 + 随片长自适应提高抽帧密度 + 新增同镜头 3 帧形态学闭运算 `_temporal_bridge_same_shot()`（填平 12fps 隔帧与单帧漏检，且绝不跨切镜串镜） |
| **4** | **「背景没用新东西变出来」** | Route 3 固有物理边界：被标成 BG 的非锚点帧由 `z_p1_up + lerp(δ)` 还原，本质是把锚点重绘细节贴回一采底图；若背景自身在变/新露出，非锚点帧不会凭空重画 | 明确记录为明日 A100 进阶探索项（见下方 P2） |

### 4. 项目大清理与参数节点瘦身
- **清理删除 11 个无用/失败节点（菜单 `21 → 10` 个）**：
  - 删除像素域修缝失败品（7 个节点）：`nodes_repair.py`（`SeamBlend / SeamFuse / SeamDissolve / InfoBuffer / SeamRepair / RedrawBridge`）、`nodes_repair_all.py`（`SeamRepairAll`）、`tools/repair_seam.py`、`tools/_obsolete_e3_redenoise_unit_test.py`。
  - 删除古早独立节点（5 个节点）：`MiniMaxH3HardCutPlan`、`MiniMaxH3HardCutValidate`、`MiniMaxH3HardCutShotPrompt`、`MiniMaxH3VRamGuard`（`_attn_forward_name` 已内联进 `nodes_firstpass.py`）、`MiniMaxH3UnloadTextEncoder`（`_looks_like_text_encoder` 已内联进 `h3_upscale.py`）。
- **两大参数节点瘦身 + 适配 Route 3**：
  - **`MiniMaxH3HardCutPass2Plan`**：**`28 个控件 → 6 个核心 + 3 个高级（砍掉 19 个古早实验常数）**！新增一键 `seam_mode`（`auto / hard_cut / smooth_overlap`）替代原先 14 个 hunt/calm/overlap 散乱开关，默认 `seam_blend=False`，新增 `route3_enabled=True`。
  - **`MiniMaxH3HardCutFirstPassPlan`**：**`14 个控件 → 7 个核心 + 4 个高级缓存控件`**！移除 `loose_prompt / multiple / require_sage_patch` 杂项，默认 `green_screen_mode=False`、`second_megapixels=1.3`，负载报告同步给出 Route 3 压缩 token 估算。
  - **`h3_upscale.py`**：补齐每窗 `torch.cuda.reset_peak_memory_stats()` 真实峰值显存统计与 Route 3 负载口径提示。
- **同步更新工作流**：已按新精简节点重新生成并校验 [泛化验证-pad3-优化-v2-一采双时钟-Route3.json](file:///D:/comfyui/ComfyUI/user/default/workflows/泛化验证-pad3-优化-v2-一采双时钟-Route3.json) 与 [local_dualclock_api.json](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/workflows/local_dualclock_api.json)。

---

## 二、明日行动清单（按优先级）

1. **【P0 · 本机 5 分钟验收】重跑今晚 10s 双镜动漫片**
   - 重启 ComfyUI → 打开更新后的 `泛化验证-pad3-优化-v2-一采双时钟-Route3.json`（确认 `#102 green_screen_mode=False`、`#107 seam_mode=auto, seam_blend=False`）。
   - 验证今晚 3 大修复全部见效：① Shot 2 特写脸全程稳定不糊不跳；② `f118..f121` 切镜处零方框颗粒、零跨镜叠影；③ 掩码不再圈静止沙发、不再隔帧狂闪。
2. **【P1 · 本机/云端二次元抠图升级】放一份 `RMBG-1.4` 权重**
   - 把 `RMBG-1.4`（`model.pth` / `RMBG-1.4.pth`，约 176 MB）放进 `D:\comfyui\ComfyUI\models\background_removal\`（`route3/masks.py` 已自动扫描 `models/` 下的 `*rmbg*.pth`），替代 `torchvision` COCO 对二次元人物的弱点。
3. **【P2 · 明日 A100 剩余积分高 ROI 验证】**
   - **实验 A（`s_bg=4 + beta_max=0.50` 全量验收）**：在 A100 `1.3MP` 上把 `s_bg=4, beta_max=0.50`（或 `beta_scale=0.38`）跑一遍 `case1/case2/case3` + `15s`，确认能否零色斑通过并比 `s_bg=3` 再省 ~12%–15% 二采时间。
   - **实验 B（多镜头 10s–15s 含硬切压测）**：验证今日修复的「多窗 `start_token` 对齐 + 跨镜自动断链 + 软羽化」在 A100 真实多镜长片上的表现。
   - **实验 C（攻克「背景没用新东西变出来」痛点）**：在 A100 测两组低成本改进——
     1. **末步全稠密（3 步二采里前 2 步 Route 3 压缩、最后第 3 步回切稠密）**：前 2 步省 67% 算力定大结构，最后 1 步全帧一起刷背景高频细节，看能否彻底让动态/新露出背景也长出新细节且无锚点脉冲！
     2. 对比 `only_bg=True` 下 `Assemble` 设 `method="none"`（直接用引擎内置 `x0_hat` 速度插值结果，配合第 3 步稠密）vs `method="lerp"` 的背景观感差异。
4. **【P3 · 本机 8GB 提速微调】**
   - 实测 `#34 head_chunks = 3` + `#35 chunks = 2`（Route 3 序列已减半，放宽分块拉高 GPU 利用率）；
   - 按需评估去掉 `--disable-async-offload` 后的 8GB 内存稳定性与功耗提升。

---

## 三、2026-11-23 晨间重大突破：绿幕边缘光晕精确定位与 I/P 相位隔离修复

- **实测样本**：`D:/共享/MiniMaxH3/exp_4v10a_00003-audio.mp4`（5s 1.3MP 绿幕，`s_bg=3 + lerp + chroma mask 22.4%`）。人物本身完全不跳变，但人物边缘绿幕呈现周期性闪光晕。
- **7/7 逐帧实锤（100% 精准对应 MiniMax H3 的 5-token = 17 帧 I/P 周期）**：
  - 所有**被丢弃的 `p=0` 独立 I-token**（`t=5/10/20/25/35` → `f17/34/68/85/119`）：整圈人物边缘爆出亮绿白光晕（`p95` 偏色从 `24` 飙升到 **`37.7~45.3`**），因为邻近的 4 帧 `P-token` 被跨类型插值塞进了 1 帧 `I-token`（`P → I` 分布错配）。
  - 唯二恰好落在锚点上的 `p=0` I-token（`t=15/30` → `f51/102`）：**零光晕**（`23.0~25.7`），但紧邻它们的 P-token（`f50/52`、`f101/103`）出现了反向的暗绿边污影（`I → P` 分布错配）。
- **已落地根治修复**：
  1. [route3/splitstream_attn.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/splitstream_attn.py) 新增唯一真相源 `bg_anchor_indices(T, s_bg, phase_aware=True)`：`s_bg=3` **100% 保留全部 `p=0` I-token** + 每 5-token 组保留 1 个 `p=2` P-token（`T=37` 为 `16/37` 锚，`beta0=0.838` 更稳）；`s_bg>=4` 则 I/P 两池各自独立抽样。
  2. [route3/engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py)（`fill_dropped`）与 [route3/assembly.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/assembly.py)（`reconstruct_moving`）与 [route3/nodes.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/nodes.py) 全面切换为 **I↔I / P↔P 严格分池插值**，杜绝一切 `I ↔ P` 跨类型污染。

---

## 四、2026-11-23 上午收口：三代绿幕边缘伪影彻底归零（`00003 → 00004 → 00005` 1:1 实测验收）

在 `exp_4v10a_00004-audio.mp4` 验证 I/P 光晕 100% 消除后，逐帧查出剩余两类边界伪影并落地 3 项根治修复，随后重启 ComfyUI 以完全相同种子（`342114565407140`）与参数 1:1 重跑生成 `exp_4v10a_00005-audio.mp4`：

### 1. 根因定位与代码修复
- **Bug A（`f001..f004` / `t=1` 整圈洋红/紫边环）**：P 池首帧 `t=1` 被删时 `prev = -1`，旧 `_fill_dropped_linear` 写成 `pv = prev.clamp(min=0) = 0` 误读了全 0 空行（H3 VAE 解码成洋红紫色），再被 `Route3Assemble` 外缘软羽化 `alpha` 混进 32px 边缘环。
  - **修复**（[route3/engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py)）：`prev < 0` 时改为 `pv = nxt.clamp(max=K-1)`（首帧外推取右侧 P 锚 `t=2`），绝不读全 0 空行。
- **Bug B1 & B2（`f045 / f053..f055 / f079..f088` 运动手臂/手腕绿幕方框残影）**：人物手臂快速摆动时，在相邻锚点（如 `t=12 / t=17`）仍被前景占据的格位，在中间被删帧（`t=16`）新露出绿幕；旧代码把前景手臂的 `x0_hat`（`fill_dropped`）与前景残差 `δ`（`reconstruct_moving`）直接贴到了新露出的背景上。
  - **修复 B1**（[route3/engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py)）：`fill_dropped` 新增 `source_mask_2d`，`splitstream_forward` 严格传入纯 BG 锚点 `plan.bg_local`，被删 BG 格只从纯 BG 锚点插值，绝不混入运动前景 `x0_hat`。
  - **修复 B2**（[route3/assembly.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/assembly.py) & [route3/nodes.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/nodes.py)）：`reconstruct_moving` 新增 `fg_mask=fg_lat` 屏蔽锚点前景（外扩 1 latent 格）残差 `δ`（双净插值、单净 100% 取净侧、双占归 0）。
- **回归测试**：已在 [route3/tests/test_engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/tests/test_engine.py) 与 [route3/tests/test_assembly.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/tests/test_assembly.py) 补齐三项精确单测，全仓 9 套测试全部 `0 项失败`。

### 2. 三代同种子 1:1 实测对比（`1568×864`, `s_bg=3`, `lerp`, `ring=1`, `N=124`）

| 指标（人物外缘 12–84px 近景绿幕环） | `00003`（原版） | `00004`（仅修 I/P） | **`00005`（全修复实测）** | 变化 |
|---|---:|---:|---:|---|
| I-token 边缘偏色 `dev_p95`（`f17/34/68/85/119`） | `36.6 ~ 51.6` | `22.0 ~ 26.6` | **`22.0 ~ 26.8`** | 亮绿光晕 **100% 消失** |
| `f001..f004`（`t=1`）边缘偏色 `dev_p99` | `28.9 ~ 30.9` | `37.5 ~ 47.7`（紫边环） | **`29.6 ~ 39.4`**（6× 放大零紫边） | 紫边环 **100% 消失** |
| `f053..f055`（`t=16` 手臂摆动）`dev_p99` | `30.9 ~ 43.0` | `51.7 ~ 62.1`（手臂鬼影） | **`28.6 ~ 30.6`**（与锚点 `26.0` 持平） | 运动残影 **100% 消失** |
| 全片边缘帧间抖动 `flicker_p95`（均值 / 峰值） | `4.42 / 19.00` | `4.15 / 18.00` | **`2.66 / 6.33`** | 峰值抖动 **下降 65%（3×）** |
| 全片 `near-BG dev_p99`（均值 / 中位） | `34.87 / 30.67` | `33.92 / 29.94` | **`27.94 / 26.80`** | 8× 时序差分 `|dt|` 边缘全黑 |

- **未来显存/速度升级方向**（已写入 [docs/ROUTE3_A100_TASK_SPEC.md](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/docs/ROUTE3_A100_TASK_SPEC.md) §九）：① Route 3 接入本地已安装的 `sageattention(..., return_lse=True)` 替代 fp16 cuDNN LSE 合并；② 绿幕双锚点（`I+P`）与二采前景 ROI 裁画布方案。

---

## 五、2026-11-23 中午终极收口：根除 6Hz 明度锯齿频闪与运动边缘缺口（`00005 → 00006` 实测验收）

用户复核 `00005` 指出两个隐蔽真问题：「纯绿幕背景明度一直在跳变」与「人物一动边缘出现缺口」。通过示波器 + 掩码叠加体检全部精确定位（**证明无需改动 VAE，也无需训 VAE LoRA**）并完成代码根治与 1:1 实测验收（[exp_4v10a_00006-audio.mp4](file:///D:/共享/MiniMaxH3/exp_4v10a_00006-audio.mp4)）：

1. **6Hz（每 4 帧）明度锯齿 + 17 帧色相跳变根治**（[route3/splitstream_attn.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/splitstream_attn.py)、[route3/engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py)）：
   - **根因**：`00004/00005` 把 100% 保留未压缩的 `p=0` I-token 错放进 `branch2` 强乘 `exp(+0.838)=2.31×`，导致注意力 I:P 质量比从原生 `20%:80%` 畸变到 `50%:50%`，50 层注意力把 I-token 首帧脉冲灌进每个 4 帧 P-token 锚点；且 P 锚死磕 `p=2` 间距拉大到 5。
   - **修复**：未压缩 `p=0` I-BG 移入 `branch1`（`β_I = 0`），仅真正抽帧压缩的 P-BG 留 `branch2` 乘 `β_P = ln(29/11) = 0.969`（精确守恒原生 `20%:80%`）；P 池改为 `p_all[::3]` 跨全部 4 个 P 相位 `{1,4,3,2}` 均匀抽样（含首尾 `t=1,36`，最大 P 间距 3~4）。
2. **运动指尖/发梢边缘缺口根治**（[route3/masks.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/masks.py)、[route3/mask_provider.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/mask_provider.py)）：
   - **根因**：`active_mask` 先在 180p 上 `ring=1`（仅 ~4.8px）再做 `avg_pool > 0.5`，`MaskProvider.for_grid` 又做第二遍 `avg_pool > 0.5`，两轮 >50% 面积腐蚀直接把高速指尖与青蓝发梢啃成背景（单帧漏检高达 `2,985 px`）。
   - **修复**：`active_mask` 与 `MaskProvider.for_grid` 统一改为保守空间 `adaptive_max_pool2d`（宁留不漏），`ring` 移至 patch 网格上真实外扩，`auto_chroma_fg` `dist_thr` 从 `0.18 → 0.14`。

| 实测指标（同种子 `342114565407140` / 全 124 帧） | `00003` | `00005` | **`00006`（终版实测）** | 提升（vs `00005`） |
|---|---:|---:|---:|---|
| 远景 `P→I`（17 帧跳变 `signed / max`） | `+1.22 / 2.14` | `+2.15 / 2.80` | **`+0.12 / 0.55`** | **↓ 17.5×（反超 `00003` 10×！）** |
| 远景 `P→P`（4 帧锯齿跳变 `signed / abs`） | `+0.07 / 0.43` | `+1.40 / 1.40` | **`+0.19 / 0.29`** | **↓ 7.3×** |
| 远景 `intra_P`（4 帧内斜率 `signed / abs`） | `+0.014 / 0.147` | `-0.483 / 0.486` | **`+0.012 / 0.132`** | **↓ 40×（彻底拉平）** |
| 远景全片平均帧间明度变化 `mean \|dY\|` | `0.283` | `0.735` | **`0.187`** | **↓ 4.0×** |
| 远景 I vs P 色差 `G-R` 跳变 | `+0.13` | `+4.68` | **`+0.64`** | **↓ 7.3×** |
| 近景边缘帧间抖动 `flicker_p95`（均值 / 峰值） | `4.42 / 19.00` | `2.66 / 6.33` | **`2.28 / 5.00`** | **全史最低** |
| 运动指尖 / 发梢 1:1 边缘缺口 | 严重光晕 | 指尖浮斑+发梢咬块 | **零缺口、零浮斑** | **121/124 帧 0 漏检** |


---

### 2026-11-23 下午 · 实景多镜头背景跳变彻底根治（Iteration 1 → Iteration 2，逐帧追平 Dense Oracle）

**测试基准**：用户原版实景双镜头工作流 `exp_4v10a_00007.png`（Shot 1 强推镜客厅 `f00..56` + 硬切 `f57→58` + Shot 2 近景喝茶 `f58..123`），同种子 1:1 对比真稠密基准 `exp_4v10a_dense_00001-audio.mp4`（Route 3 关闭）。

**迭代根因与修复**：
1. **Iteration 1（`exp_4v10a_lsd_00001-audio.mp4`，`last_step_dense=True`）**：
   - `MiniMaxH3Route3Apply` 新增 `last_step_dense`，前 N-1 步压缩采样，末步自动切回原生全稠密（+SageAttention）全帧联合精修，`Route3Assemble` 自动检测直通。
   - **效果**：Shot 2（`f067..123`）所有 `drp ↔ ANC` 背景跳变立即归零，与 Dense Oracle 持平（`2.0~2.6`），`bg_psnr` `+1.01 dB`。
   - **暴露出的最后两处硬伤**：① Shot 1 推镜仍出现 `f046 dx=+5.27px` 错位与 `f049 sharp=0.788` 双曝光模糊（因 `engine.py` 前两步对原始 `x0_hat` 做固定坐标线性插值，丢掉了 `z_p1_up` 的 24fps 运镜几何）；② 硬切处 `f058..066` 叠出镜头 1 客厅重影（`f060=30.67`）。
2. **Iteration 2（`exp_4v10a_lsd2_00001-audio.mp4`，Pass-1 引导亚像素光流残差轨迹 + 相位感知硬切钉住）**：
   - `h3_upscale.py` + `route3/engine.py`：前 N-1 步透传 `route3_p1_up` 与光流缓存，被删 BG 帧由 `z_p1_up[j] + 双向亚像素光流对齐后的 δ(x0_hat - z_p1_up)` 引导（并用 `comfy.model_prefetch.pause_malloc_graph()` 保护避免撑爆 `comfy_aimdo` C 析构栈）。
   - `route3/assembly.py` + `route3/splitstream_attn.py`：解除 `_match_tiles` 对 `<0.5` latent 格（`<8px`）亚像素推镜位移的锁死；升级 `detect_latent_cuts` 为纯 P-pool 1/2 步相位感知硬切检测（`cut_ratio=2.1` + 跨 I 步距 `/1.25` 归一化）并支持硬切邻域 `pin_tokens` 归入 `branch1`（`β=0` 精确 P-质量守恒）。
   - `route3/nodes.py` + `route3/patch.py`：`last_step_dense` 默认值切换为 `True`。

**最终四方量化结果（124 帧全片，剔除硬切帧 `f57..59`）**：

| 指标 | `00008` (旧全压缩) | `LSD_iter1` | **`LSD_iter2` (最终版)** | **`DENSE ORACLE` (真稠密)** |
|---|---:|---:|---:|---:|
| 全片 MC 背景抖动 `mean_p95` | 5.12 | 4.22 | **2.91** | **3.08** |
| 全片 MC 背景抖动 `max_p95` | 17.67 (`f051`) | 30.67 (`f060`) | **5.00** | **5.00** |
| Shot 1 强推镜 MC `mean / max` | 6.38 / 17.67 | 5.34 / 12.67 | **3.98 / 5.00** | **4.34 / 5.00** |
| Shot 2 近景 MC `mean / max` | 4.02 / 13.33 | 3.25 / 30.67 | **1.97 / 4.33** | **1.98 / 3.33** |
| vs DENSE `bg_psnr / fg_psnr` | 28.28 / 23.27 dB | 29.29 / 23.44 dB | **30.73 / 23.80 dB** | ∞ |
| vs DENSE 背景高频保持 `bg_hf` | 0.990 | 0.957 | **0.983** | 1.000 |
| Shot 1 `f046` 位移 / `f049` 锐度 | — | `+5.27px` / `0.788` | **`+0.55px` / `0.929`** | `0.00px` / `1.000` |
| 切镜 `f058..064` 最大亮度偏离 | 1.898 | 8.317 | **0.444** | 0.000 |

---

### 3. 晚间接力审计与 Iteration 4 -> Iteration 5 稳定化（Hydrogen Stabilizer）

在 `LSD3` 已实测可用的基础上，对照 `DENSE ORACLE`（`exp_4v10a_dense_00001-audio.mp4`）对 124 帧做全片+ROI 逐 token 审计，定位出 `LSD3` 仍存留的 5 个根因：

1. **误切断桥与未保护的强推镜间隔**：`LSD3` 跑在修复前的旧切镜检测上（`cuts=[16, 17, 18, 19, 21]`，`motion_pins=[2, 6, 9, 24, 28]`），误判 `cuts=16` 打断 `(12->16)` 使 `t13, t14` 崩坏（`BG_p95=15.0/17.0`，沙发扶手 `Arm_dW=-3.41/-3.26 px`），同时遗漏 `t03, t07, t11`（`t03 Arm_dW=+1.69 px`，`t11 Tow_dW=-0.91 px`）。
2. **局部 Voronoi 注意力质量不守恒（`route3_plan`）**：`extra_p_pins` 放入 `branch1`（权重 `1.0x`）而同镜头内的常规 `base_p_set` 仍留在 `branch2`（`exp(β_P) ≈ 1.73x..1.82x`），形成相邻 token `1.8x / 1.0x` 棋盘格权重震荡与跨镜头总质量偏移。
3. **周期纹理光流异常跳变**：`estimate_tile_flow(..., tiles=(10, 14))` 在百叶窗等周期纹理上出现 `(±4, 0)` 跳变并穿透 `fb_check`（`9–14/140` tiles）。
4. **未隔离前景的运动锚点误判**：`motion_pin_tokens` 未屏蔽前景区域，Shot 2 人物面部/身体近景运动将 `med_mag` 抬高到 `0.929..1.006`（纯背景仅 `0.374..0.440`），导致 `LSD3` 在静态 Shot 2 误锁 `t24, t28`（连同误切 `t21` 将全片/Shot 2 保留率推至 `79.8% / 79.7%`）。
5. **硬块置信度与运动背景前景边缘渗透**：`flow_residual` tile 最近邻硬切换及 `reconstruct_moving` 在未做 `clean_bg` 邻域填充时，移动机位下插值会从 `~clean_bg` 区域渗入 FG delta。

**核心工程改动（Iteration 4 -> Iteration 5）**：
- `route3/splitstream_attn.py` (`route3_plan`)：实现局部 Voronoi 注意力质量精确守恒——若某保留 P-token 的相邻 P 邻居均已被保留（`branch1` 或 pinned），该 P-token 自身也移入 `branch1`（`β=0`）；仅邻接至少 1 个 dropped P-token 的保留 P-token 留在 `branch2`，令 `beta0 = math.log((len(p_all) - len(b1_p_set)) / len(b2_p_set))`。
- `route3/assembly.py`：
  - `estimate_tile_flow`：增加 coarse-consensus（`min(th, tw) >= 8`）+ `3x3` 邻域空间一致性门控，剔除周期纹理 `(±4, 0)` 跳变异常块；
  - `flow_residual`：将 tile 置信度由最近邻硬切改为双线性空间平滑 `good_soft`（同时保持 `conf == 0` 低纹理内区严格为 `0.0`）；
  - `reconstruct_moving`：追加 1-ring `clean_bg` 邻域填充 `_prepare_clean_delta` 与 `_warped_clean` 源块校验，消除移动机位前景边缘 delta 渗透；
  - `motion_pin_tokens`：新增 `fg_mask` 参数按纯背景 tile 中位数（`fg_tile < 0.35`，不足 6 块时回退全局）评估背景真实推拉运动；结合 `LSD4`（`high_thr=1.25` 仅锁 `[2,6,9,13]` 遗漏 `[3,7,11,14]`）对比，在 `LSD5` 标定 `high_thr = float(flow_thr) * 1.15`（`0.8625`），既全锁定高视差 Shot 1 `[2, 3, 6, 7, 9, 11, 13, 14]`，又保持冷静 Shot 2 零误锁。
- `route3/patch.py`, `h3_upscale.py`, `route3/nodes.py`：透传 `route3_fg_mask_fn` -> `fg_mask`，并在 `st.calls == 1` 输出 `[route3] plan` 压缩统计。

**六方量化对照（124 帧全片 + Shot 1 沙发角/格子毛巾 ROI，相对 `DENSE ORACLE`）**：

| 指标 | `LSD1` | `LSD2` | `LSD3` | `LSD4` | **`LSD5` (最终版)** | **`DENSE ORACLE`** |
|---|---:|---:|---:|---:|---:|---:|
| 全片 MC 背景抖动 `ALL mean / max` | 4.37 / 35.33 | 2.95 / 6.67 | 2.82 / 6.00 | 2.72 / 5.67 | **2.54 / 5.67 (`f056`)** | **2.55 / 5.67 (`f056`)** |
| Shot 1 MC 背景抖动 `mean / max` | 5.39 / 14.00 | 3.87 / 6.67 | 3.62 / 6.00 | 3.40 / 5.67 | **3.03 / 5.67 (`f056`)** | **3.03 / 5.67 (`f056`)** |
| Shot 2 MC 背景抖动 `mean / max` | 3.47 / 35.33 | 2.14 / 5.67 | 2.11 / 5.00 | 2.11 / 5.33 | **2.11 / 5.33 (`f060`)** | **2.14 / 5.00 (`f060`)** |
| vs DENSE `bg_psnr / fg_psnr` (dB) | 27.80 / 22.98 | 30.51 / 23.27 | 33.66 / 24.97 | 33.67 / 24.70 | **35.11 / 25.24** | ∞ |
| 背景高频保持 `bg_hf` | 0.964 | 0.981 | 0.995 | 0.992 | **0.995** | 1.000 |
| 切镜 `cut \|dY\|` | 8.577 | 0.444 | 0.189 | 0.304 | **0.514** | 0.000 |
| Shot 1 ROI `MAE mean / max` | — | 3.90 / 4.99 | 2.93 / 4.27 | 2.74 / 3.80 | **1.67 / 1.86** | 0 |
| Shot 1 ROI `p95 mean / max` | — | 14.98 / 21.00 | 10.06 / 17.00 | 9.25 / 15.33 | **4.36 / 5.00** | 0 |
| 左沙发扶手 `Armrest dW` `std / max\|e\| / p2p` (px) | — | 1.94 / 4.47 / 7.57 | 1.28 / 3.51 / 6.12 | 0.69 / 2.45 / 3.84 | **0.07 / 0.16 / 0.29** | 0 |
| 格子毛巾 `Towel dW` `std / max\|e\| / p2p` (px) | — | 0.64 / 2.00 / 2.63 | 0.51 / 2.01 / 2.15 | 0.16 / 0.89 / 0.78 | **0.05 / 0.20 / 0.27** | 0 |
| 格子毛巾 `Towel dx` `std / max\|e\| / p2p` (px) | — | 0.79 / 2.48 / 3.81 | 0.42 / 1.13 / 1.73 | 0.41 / 1.10 / 1.63 | **0.13 / 0.34 / 0.43** | 0 |

**`LSD5` 运行配置与压缩诚实度**：
- `AutoMask`: `backend=rembg, fg=21.2%, shape=(124, 90, 162)`；`cuts=[17, 18]`, `cut_pins=[16, 17, 18, 19]`, `motion_pins=[2, 3, 6, 7, 9, 11, 13, 14]`, `pin_tokens=[2, 3, 6, 7, 9, 11, 13, 14, 16, 17, 18, 19]`。
- `[route3] plan`: `kept=42111/48951 (86.0%)`, `anchors=29`, `n_full=43463`, `beta0=0.9555`（实测分镜头保留率：强推镜 Shot 1 `t0..16` `99.1%` / BG `98.9%`；冷静 Shot 2 `t18..36` `77.1%` / BG `73.8%` 且零 motion pin，无 `pin_tokens` 冷静片段基线为 `64.6%` / BG `62.2%`）。
- 耗时：Pass-2 `395.0s` (`88.60s/it`)，端到端 `00:10:55`。
- 单元测试：`run_all_tests.sh` 全 5 个测试套件（`test_assembly.py`, `test_masks.py`, `test_splitstream_math.py`, `test_engine.py`, `test_route3_integration.py`）在最终修改后全部通过，0 失败。
