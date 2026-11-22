# 任务书（v3 · 本地双时钟同源 + 真机冒烟校准版）：MiniMax H3 路线 3「多速率前/背景双流引擎」A100 验证规程

> **算力预算**：云端 **Colab A100 80GB**，总计 **≤ 3.0 小时（硬停机红线 180 分钟，软停 165 分钟）** —— 相比 v2（9.25h）砍掉高风险 LoRA 训练、启用一采落盘缓存，实测预计 **1.5–2.2 小时**完成全量验证。
> **基准工作流**：全面切换为用户**本地一采双时钟工作流**（[local_dualclock_api.json](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/workflows/local_dualclock_api.json)，由 `泛化验证-pad3-优化-v2-一采双时钟.json` 导出，**不再使用 `WenWu_H3_A100_FINAL_api.json`**）。
> **同源权重**：全程直测现网 `dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors`（与本机 **RTX 4060 Laptop 8GB** 完全同源，结论零偏差回迁）。
> **定位红线**：MiniMax H3 是完整的 **50 层视听叙事大模型**，不是静图模型。必须同时覆盖 **动态运镜叙事（`S_bg=5 / 3` + 关键帧 `m=1` 完整二采重绘 + 光流残差 warp）** 与 **静机位极限档（`s_bg=0 / T_bg=1`）**，全链路保留原生双声道音频。

---

## 🔴 一、执行红线（DeepSeek 监控代理必读，违反即停机）

1. **🔴 必须后台看门狗托管**：在 `tmux` 或独立后台进程同时拉起 [watchdog.sh](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/watchdog.sh) 与 [run_pipeline.sh](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/run_pipeline.sh)；**不依赖 Agent 在线**，断连自动续跑/超时自动止损。
2. **🔴 严禁在 A100 上编译 `flash-attn` 或折腾环境 > 10 分钟**：开机先跑 2 分钟 [preflight_check.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/preflight_check.py)；LSE 通路走**零编译原生 ATen `cudnn → memeff → flash(Linux) → flex → manual`**，绝不源码编译任何算子。
3. **🔴 严禁重新引入 Stage 3（43M LoRA 训练）**：本机已实证三座死墙（原 `DiTBlock` 原地算子反传崩、`comfy_kitchen` 融合 RoPE 无 autograd、32–64 样本训 43M 过拟合且与 INT8 `convrot` 不同源）。本轮**只做免训练消融 +（可选）14,000 参数相位门控 $\beta[50, 56, 5]$ 校准（`< 80 KB`）**。
4. **🔴 A100 基线纯净 vs 本机 8GB 回归分离**：
   - 云端 Stage 0–1 通过 `--strip-lowvram` 自动旁路本地 8GB 补丁（`#34 MiniMaxLowVRAMAttention`、`#35 MiniMaxChunkFeedForward`、`#17 MiniMaxH3MemoryEfficientSageAttentionPatch`）以及 `#106 SolAttnMiniMax`，确保拿到纯净无稀疏干扰的稠密二采 Oracle；
   - 阶段验证通过后，再关掉 `--strip-lowvram` 回归测试本地 8GB 全链路。
5. **🔴 严禁放宽验收口径 / 严禁无限重试**：
   - `T1/T2` 数学单测口径：`fp32 < 1e-5` / `bf16 < 5e-3`（bf16 8 位尾数天然差 ~`1.9e-3`，不再用会误报的 `1e-3`）；
   - Oracle **必须跑完整二采**（否则 `bg_hf_sharpness ≥ 0.90× Oracle` 的分母只是一采 0.4MP 糊底）；
   - 同一 Stage 连续失败 **2 次** 立即中止并写 `ABORT_REPORT.md`，严禁死循环烧卡。
6. **🔴 Colab 持久化防丢盘**：高速读写放 `WORK_ROOT=/content/h3_route3`，每完成一个 Stage 或指标文件，由看门狗原子同步（先 `.tmp` 再 `cp` + 校验大小）到 Google Drive `DRIVE_BACKUP`；严禁把唯一产物留在 `/tmp`。

---

## 🔬 二、本机 RTX 4060 8GB 真机冒烟实证（已提前踩平 4 个大坑）

上传 Colab 前，已在用户本机 ComfyUI（`torch 2.14.0+cu130` / RTX 4060 8GB / 真实 `dasiwa` INT8 50 层底模）对着 `local_dualclock_api.json`（3.0s = 73 帧 = `T22`，`0.4MP → 0.85MP`）跑通了全套真机冒烟，获得以下**已验证事实**：

1. **真实 50 层注意力分布（`probe_attn_stats.json`，50/50 层 `err=0`）**：
   - **质量守恒精确成立**：$\beta_0 = \log(T/n_{\text{keep}}) = 1.4816$ 下，全 50 层 `bg_mass_ratio_median = 0.9999`（范围 `0.984–1.025`）。
   - **长程时间依赖极强**：全 50 层中位 `T = mass(Δt ≥ 8.33) / mass(Δt ≈ 0) = 5.03`，`L00–L40` 同帧占比仅 `16%–28%`，证明 MiniMax H3 确实跨帧聚合背景语义。
   - **发现 U 型层敏感度（云端消融重要依据）**：
     - `L00`：`bg_cos = 0.9933`，`rel_err = 0.121`；
     - `L15–L35`（中段 20+ 层）：`bg_cos = 0.900–0.925`；
     - `L01–L03` 与 `L45–L49`（首末渲染层）：同帧占比飙升至 `39%`，`bg_cos` 降至 `0.78–0.83`。
     - **推论**：若云端全 50 层 `s_bg=5` 出现细微背景噪点或边缘软化，**零成本首选调优是“末 5 层（L45–L49）保持稠密、其余 45 层压缩”**。
2. **真机排雷 1 — Flow Matching 被删帧速度插值必出彩色噪点（已修）**：
   - 冒烟实测发现：直接对速度 $v = \varepsilon - x_0$ 跨帧插值会因各帧输入噪声 $\varepsilon(t)$ 独立，在非锚点帧残余 $\sqrt{2}\sigma$ 彩色高斯块噪（锚点 `f=0` 与 FG 区域清晰，其余全是噪点）。
   - **已修正**（[engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py)）：先在保留行还原干净预测 $\hat{x}_0 = x_t + \sigma v_{\text{kept}}$（匹配 ComfyUI `CONST` 符号），对干净 $\hat{x}_0$ 跨帧线性插值，再反算被删帧速度 $v_{\text{drop}} = (\hat{x}_{0,\text{interp}} - x_t)/\sigma$，精确消去被删帧自身的独立噪声 $\varepsilon(j)$（单测 `err_x0 < 1e-5`）。
3. **真机排雷 2 — 普通实景视频与绿幕通用的前景提取（已修）**：
   - 冒烟中单点高速挥手把 `e.max()` 拉高到 `0.83`（99 分位仅 `0.11`），让纯运动差分漏掉静立躯干（仅 `0.9%`）。
   - **已修正**（[masks.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/masks.py) / [nodes.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/nodes.py)）：
     - ① `active_mask` 阈值改用 `q995` 稳健封顶，避免局部快速肢体遮蔽躯干；
     - ② 新增全场景 `auto_fg_mask`：绿幕自动走色键（`0.02s`），**任意普通实景视频自动走语义主体分割链（`BriaRMBG` → `torchvision LRASPP-MobileNetV3 (13MB)` → `rembg u2net` → 运动补偿兜底）**，并与全局运镜补偿后的 `HighMotionBG` 取并集；
     - ③ 新增 ComfyUI 一键节点 **`MiniMaxH3Route3AutoMask`**，直接接在一采输出 `#93` 后为任意普通视频自动产出 `mask_path` 喂给 `Route3Apply` 和 `Route3Assemble`；
     - ④ `stage0_oracle.py` / `stage1_ablate.py` 在提示词不含绿幕关键词时自动关闭 `#102` 的 `green_screen_mode`，不干扰普通多镜视频的硬切策略。
4. **真机排雷 3 — 探针必须严格模拟 Route 3 双流而非“删光所有非锚点”（已修）**：
   - 初版探针删掉了非锚点 query 自己所在的同帧自注意力，造成假性 `rel_err = 0.45` 误报；[probe_attn.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/probe_attn.py) 已升级为精确模拟 `[Prefix | FG_full] + [BG_anchors + β0]`，且仅在 Route 3 真实保留的 `FG_full ∪ BG_anchors` 上评估。
5. **真机排雷 4 — `MiniMaxH3HardCutUpscale` 3 口直出 `upscaled_latent` + 一采缓存 0 秒复用（已接通）**：
   - [h3_upscale.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/h3_upscale.py) 在 CPU 上零显存开销累积 3D 上采样底图，新增第 3 输出口 `[40, 2] upscaled_latent`（向后兼容旧工作流），由 [wf.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/wf.py) 自动将 `[40, 2]` + `[40, 0]` 接入 `MiniMaxH3Route3Assemble` → `#16 AVDecode`；
   - 实测本机 Run 2 命中一采缓存后，3.0s 完整二采仅 **75.2s**（对比含一采+探针的 Run 1 `246.8s`），真实 `NestedTensor` Assemble `flow` 仅需 **3.19s**、音轨误差 **`0.00e+00`**。

---

## 🛡 三、Colab 断连与无人值守容错架构

### 3.1 一键启动（上传 `comfyui-h3-seamkit` 到 Colab 后执行）
```bash
export REPO_ROOT="/content/ComfyUI/custom_nodes/comfyui-h3-seamkit"
export WORK_ROOT="/content/h3_route3"
export DRIVE_BACKUP="/content/drive/MyDrive/h3_route3_backup"
export API_JSON="$REPO_ROOT/route3/workflows/local_dualclock_api.json"
export DURATION=5.0
export MEGAPIXELS=0.94        # 1280x736 = 920 tok/帧；通过后再升 1.3MP
export STRIP_LOWVRAM=1        # A100 上自动旁路 #34/#35/#17 本地 8GB 补丁
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/artifacts" "$DRIVE_BACKUP"

# 1) 独立后台看门狗（每 60s 巡检 + Drive 原子同步 + 165m 软停 / 180m 硬杀）
nohup bash "$REPO_ROOT/route3/watchdog.sh" >> "$WORK_ROOT/logs/watchdog.log" 2>&1 &

# 2) 主流水线（断点续跑 --resume + Stage 1a 1.5 分钟金丝雀真实成片止损门禁）
nohup bash "$REPO_ROOT/route3/run_pipeline.sh" >> "$WORK_ROOT/logs/pipeline.log" 2>&1 &
```

### 3.2 四重熔断规则（[watchdog.sh](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/watchdog.sh)）
| 熔断规则 | 触发条件 | 自动处置 |
|---|---|---|
| **① 总时长软/硬停机** | `≥ 165 min` 软停；`≥ 180 min (3.0h)` 硬杀 | 写 `STOP_REQUESTED` 让主进程完成当前条目落盘并同步 Drive；超时硬杀防止烧卡 |
| **② GPU 空转 / 僵死熔断** | 连续 **15 分钟** GPU 利用率 `< 5%` 且日志字节无增长 | 写 `ABORT_REPORT.md` + 抓最后 200 行日志 → 同步 Drive → 强杀任务 |
| **③ 单阶段超时锁** | Preflight `3m` / Stage 0 `50m` / Stage 1 `90m` | `timeout --signal=TERM --kill-after=60` 强制封顶 |
| **④ 磁盘保护 & Drive 同步** | 可用空间 `< 10 GB` 写 `LOW_DISK_FLAG`；每 60s 同步小文件 | 优先保住 `probe_attn_stats.json`、`stage1_ablation.csv`、`stage1_contact_sheet.png` 与对比 mp4 |

---

## 🧬 四、核心算法规格与三项关键工程定调（已落代码、本机全绿）

1. **序列打包与运动自适应活跃区**：
   - $\text{Active}(t) = M_{\text{fg,语义}}(t) \cup \text{HighMotionBG}_{\text{去运镜}}(t) \cup \text{Ring}(r)$，全局运镜经分块位移补偿剔除（纯推镜活跃区仅 `0.9%`，局部物体精确标出），同时通吃绿幕与普通实景视频。
   - 打包为紧凑单序列 `H = [Prefix | FG_full | BG_strided]`，50 层锁步前向，零跨层 KV 常驻显存。
   - **二采背景重绘余地**：`BG_strided` 关键帧默认 `m = 1` 参与完整二采重绘（`denoise = 0.30`），把背景重绘自由度收束到每 17 帧（`s_bg=5`）画准一次。
2. **精确质量守恒合并（Online-Softmax LSE Merge，零编译）**：
   - 设 Branch 1 键为 `[Prefix | FG_full]`，Branch 2 键为 `BG_strided`，引入相位感知偏置 $\beta \in \mathbb{R}^{50 \times 56 \times 5}$（初值 $\beta_0 = \log(T / n_{\text{keep}})$）：
     $$L = \operatorname{logaddexp}(L_1,\; L_2 + \beta), \qquad O = e^{L_1 - L} O_1 + e^{L_2 + \beta - L} O_2$$
   - 后端优先级：`cudnn`（无 padding、误差 `9.5e-7`）→ `memeff`（切片 `lse[..., :S]` 防 8 对齐越界）→ `flash` → `flex` → `manual`。
3. **动态背景中间帧高清还原（法 A vs 法 B 必做对照）**：
   - 取 `MiniMaxH3HardCutUpscale` 输出的 3D 上采样底图 `z_p1_up`（`[40, 2]`）与二采结果 `z_p2`（`[40, 0]`），在锚点帧提高清增量 $\delta(5k) = z_{\text{p2}}(5k) - z_{\text{p1\_up}}(5k)$：
     - **法 B（`flow`，主推）**：latent 分块位移场双向 warp 残差 + 尾帧（如 `T=37` 的 `t=36`）顺流外推；
     - **法 A（`lerp`，对照组）**：固定坐标线性插值，用来量化 `flow` 在运镜下的净收益。
4. **🔴 三项关键校准定调（杜绝误判与踩坑）**：
   - **定调 1（探针降格为诊断排序器，真止损改用 1.5 分钟金丝雀成片）**：单层 `rel_err` 经 50 层残差门控与 `denoise=0.30` 积分是非线性的，无法从 `0.08/0.20` 解析反推最终画质。**取消 `rel_err > 0.20` 自动硬杀**；探针保留为「U 型层敏感度诊断 + 方案排序器」（看 `early/mid/late` 哪几层炸、看 `bg_cos` 判方向是否坏透），真正止损门禁改为 **Stage 1a 金丝雀（仅跑 `case2_dolly × 5:1:flow`，A100 约 1.5 分钟）**，直接用真实成片 `fg_psnr / bg_hf_sharpness / seam_grad_ratio` + 肉眼截帧定去留。
   - **定调 2（`s_bg=0` 严格降格为「仅静机位专享档、非动态背景重绘」）**：5s 下 `β0 = log(37) = 3.61`（37× 权重放大）、最远跨 `Δt = 61.7`（15s 达 `107×` / `Δt = 181`），数学上**只在背景像素与光照真正不动时成立**，中间帧本质是「把二采刚画好的第 0 帧 1.3MP 高清背景静态锁屏贴满全片」。报告中明确标为「仅静机位 / 绿幕专享，非通用动态档」，**Stage 1b 只在 `case1_static` 上单跑，绝不拿 `case2_dolly` / `case3_neon` 考核它**。
   - **定调 3（冻结一采必须用 `MiniMaxH3AVLatentLoad` 剪枝整条一采子图，严禁使用 ComfyUI 内置 `SaveLatent/LoadLatent`）**：
     - ComfyUI 内置 `SaveLatent/LoadLatent`（`nodes.py:553/576`）只能存单张量 safetensors，**直接破坏 `NestedTensor((video, audio))` 且丢音频**；
     - `#93 MiniMaxH3FirstPassSampler` 的 HIT 缓存虽能跳过采样，但仍会跑完上游 LOW conditioning 且按设计把音频清零为静音（`nodes_firstpass.py:197`）；
     - **标准做法（已集成进 [wf.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/wf.py) `splice_pass1_cache_load`）**：Stage 0 跑完由 `#93`（或 `MiniMaxH3AVLatentSave`）把完整未清零的 `NestedTensor((video, audio))` 拆成 2 个普通 CPU 张量写 `.pt`（~6–17 MB，`weights_only=True` 逐位无损读回）；Stage 1 由脚本自动把 `#40.inputs.latent`（或上游 3D Upscaler 输入）改指向 **`MiniMaxH3AVLatentLoad(key, cache_dir)`**——**零修改 UI 图、ComfyUI 自动剪枝整条一采子图（连 LOW conditioning 都 0 秒跳过）、音频原封不动穿过二采进 `AVDecode`**（用带 `case+seed` 的唯一 `key` 防串档）。

---

## 📋 五、3 小时 A100 分阶段日程与硬止损门禁

| 阶段 | 预算 | 执行脚本 | 必须产物 | 🟡 Go / No-Go 硬门禁 |
|---|---|---|---|---|
| **0. Pre-flight** | **5 min** | `route3/preflight_check.py` | `preflight_ok.json` | 校验 GPU/显存/磁盘/权重 header（50L/5376/56×128）+ 零编译 LSE 实测；任何失败立即停机 |
| **1. Stage 0** | **20 min** | `route3/stage0_oracle.py --strip-lowvram --probe --resume` | 3 条完整二采 Oracle mp4 + 冻结一采 AV latent `.pt` + `mask.npz` + `probe_attn_stats.json` | 仅作**诊断与 U 型层画像**（不盲目 auto-abort）；确认 3 条 Oracle 成片、掩码与冻结 `.pt` 全部落盘即进 Stage 1a |
| **2a. Stage 1a *(金丝雀门禁)*** | **3 min** | `route3/stage1_ablate.py --only case2_dolly --combos 5:1:flow`（`MiniMaxH3AVLatentLoad` 0秒跳一采） | `stage1_canary/case2_dolly_s5_r1_flow.mp4` + 指标 CSV | **真正物理止损门禁**：最难动态运镜 `case2_dolly` 满足 **`fg_psnr ≥ 22 dB`**、**`bg_hf_sharpness ≥ 0.80× Oracle`**、**`seam_grad_ratio ≤ 2.2×`** → 放行全量 Stage 1b；不达标则对照 `probe_attn_stats.json` U 型画像决定「保留末 5 层稠密 / 降 `S_bg=3`」或直接止损转**路线 A** |
| **2b. Stage 1b *(全量消融)*** | **55 min** | `route3/stage1_ablate.py --combos 5:1:flow,5:1:lerp,5:0:flow,3:1:flow` + 静机位专享 `case1_static × 0:1:flow` | `stage1_ablation.csv` + `stage1_contact_sheet.png` + 全量有声对比 mp4 | 至少 1 组动态档满足：**`fg_psnr ≥ 24 dB`**、**`bg_hf_sharpness ≥ 0.85× Oracle`**、**`seam_grad_ratio ≤ 1.8×`**，且 `flow` 优于 `lerp` |
| **3. Stage 2 *(可选)*** | **30 min** | 仅当 Stage 1b `seam_grad_ratio ∈ (1.2, 1.8]` 时启动 | `splitstream_gates.pt`（`< 80 KB`，$\beta[50,56,5]$） | 校准后 `seam_grad_ratio ≤ 1.20×`、`bg_hf_sharpness ≥ 0.90× Oracle` 即收工 |
| **4. 15s 压测 & 备份** | **30 min** | `DURATION=15.0`（`T=107`, 362f）单/双窗对比 + Drive 同步 | `FINAL_REPORT.md` + 15s 对比 mp4 | 15s 长片相对原 4 窗分块背景跳变显著下降、音画同步无崩坏，全部产物同步至 Drive 后关机 |

> **总计**：约 **2.0–2.6 小时**（留 20–30 分钟停机缓冲），硬上线 **3.0 小时**。

---

## 🎯 六、核心测试用例规格（5s / 124 帧 / `T=37` / `1280×736 = 0.94MP`，同 seed）

1. **`case1_static`（静机位表演叙事）**：固定机位汉服人物慢速呼吸、眨眼、连贯手势 + 暖光室内 + 房间底噪（标定 `s_bg=0` 极限压缩上限与发丝/衣纹保真；也可追加用户自带 `deepseek.png` 绿幕 Ref2VA 用例）。
2. **`case2_dolly`（动态推镜 + 大位移 — 重点拷问）**：缓慢推进镜头跟拍行人横穿湿润砖墙巷道 + 明显前景-背景视差 + 脚步声（检验 `s_bg=5` 下 `flow` vs `lerp` 残差传播与全局补偿掩码）。
3. **`case3_neon`（强光影 + 景别切换 — 重点拷问）**：夜景霓虹强侧逆光室内/街景、中景转特写 + 地面/墙面投影交互 + 连续环境音（检验跨镜连贯与阴影环 `ring=1` vs `ring=0` 效果）。

> **说明**：[stage0_oracle.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/stage0_oracle.py) 与 [stage1_ablate.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/stage1_ablate.py) 已内置自动判断：当 prompt 不含 `<Picture` 标记时自动把 `local_dualclock_api.json` 从 `Ref2VA` 切为 `T2VA` 并剥离参考图；当 prompt 包含 `<Picture 1>`（如用户自带的 `deepseek.png` 绿幕用例）时自动保留 `Ref2VA` 链路。

---

## 📦 七、最终必须回传本地的交付物清单（同步至 Google Drive）

1. `preflight_ok.json` 与 `run_state.json`
2. `probe_attn_stats.json`（50 层 × 56 头直接输出替换误差 + 双口径 $\Delta t$ 分布 + 止损结论）
3. `stage1_ablation.csv` + `stage1_contact_sheet.png`（可直接 5 秒肉眼对比 Oracle / `flow` / `lerp` / `s_bg=3` / `s_bg=0`）
4. `splitstream_gates.pt`（若触发 Stage 2，`< 80 KB`，零开销加载到本机 INT8 模型）
5. `FINAL_REPORT.md` + 3 条 5s 对比成片 +（若通过）1 条 15s 长片对比成片
