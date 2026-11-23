# 任务书（v3 · 本地双时钟同源 + 真机冒烟校准版）：MiniMax H3 路线 3「多速率前/背景双流引擎」A100 验证规程

> **算力与目标定位**：云端 **Colab A100-40GB**（剩余约 **3 小时**），**一切实验服务于本机 RTX 4060 Laptop 8GB 回迁**——不再浪费算力做超高分辨率压测，全程锁定 8GB 可落地的 `0.94–1.3MP`（`1280×736` / `1568×864`）与 `5s / 15s` 长片降窗验证。
> **已达成里程碑（A100-40GB 1.3MP 9 条实测全过）**：`s_bg=3` 在 3 条片子上**全绿过关**（`fg_psnr 24.88–28.36 dB`、`bg_hf 0.901–0.981`、`seam 0.727–0.828`），全面取代 `s_bg=5` 成为默认主推甜点！
> **基准工作流**：全面切换为用户**本地一采双时钟工作流**（[local_dualclock_api.json](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/workflows/local_dualclock_api.json)，**不再使用 `WenWu_H3_A100_FINAL_api.json`**）。
> **同源权重**：全程直测现网 `dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors`（与本机 8GB 完全同源）。

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

## 📋 五、剩余 3 小时 A100-40GB 冲刺日程（面向本机 8GB 回迁，不做无意义超高分辨率测试）

> **已完成（首轮 A100-40GB 1.3MP 9 条消融）**：
> - `s_bg=3` **3/3 全绿**：`case1` 28.36 dB / `case2` 24.88 dB / `case3` 25.63 dB，`seam 0.727–0.828`（Oracle 204s → 54–93s）；
> - `s_bg=5` 因与 H3 5-token VAE 周期 `(1,4,4,4,4)` 共振（100% 锚点落单帧 I-token `p=0`、80% P-token 零同型键），全线低 `1.6–2.0 dB` → **淘汰出主推，默认甜点锁死 `s_bg=3`（可加测互质的 `s_bg=4`）**。

| 阶段 | 预算 | 核心目标（直奔本机 8GB 痛点） | 必须产物 | 🟡 验收口径 |
|---|---|---|---|---|
| **A. `flow v2` 复测** | **15 min** | 拉取升级版 [assembly.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/assembly.py)（全 24 通道 + 纹理门控 + 前后向一致性 + 全局中位兜底 + `bicubic`）与 `auto_fg_mask`，复用已有冻结一采缓存，只重跑 `case2`/`case3` × `s_bg=3` (`flow v2` vs `lerp`) | `stage1_flow_v2_compare.png` + 新 CSV（含 `fg_ssim`） | 确认湿地面红砖污块彻底消失、移动车辆进入 `FG_full` 消除青绿重影、`bg_hf ∈ [0.88, 1.03]` |
| **B. 随机基线 & 纯净测速** | **15 min** | 在 `case2` 同份冻结一采上换二采 `seed+1` 跑 1 条稠密 Oracle 量 **`PSNR_stoch / SSIM_stoch`**；同时剥离冷启动/缓存抖动，在预热后测纯净二采耗时与峰值显存（Oracle vs `s_bg=3` vs `s_bg=4`） | `stochastic_baseline.json` + 干净耗时/显存账本 | 用相对门槛 `PSNR ≥ PSNR_stoch - 1.5 dB` + `fg_ssim ≥ 0.85` 替代一刀切 24 dB；拿到无抖动的真实加速比 |
| **C. 15s 长片降窗实战** *(高 ROI)* | **50 min** | 在本机 8GB 真实常用画布（`0.94MP 1280×736` 与 `1.0–1.2MP`）跑 **15s（362f / `T=107`）**：对比「原 4 窗 SeamKit 分块」vs「Route 3 `s_bg=3`（与 `s_bg=4`）1–2 窗直出」 | `15s_compare_seamkit_vs_r3.mp4` + 跨窗跳变对比 | 验证 15s 长片能不能靠 Route 3 **把 4 窗压成 1–2 窗**、从根源消灭跨窗背景跳变，且音画同步不崩 |
| **D. 8GB 补丁同源回归 + 可选末层/β 微调** | **40 min** | 开回本机 8GB 补丁（`#34 MiniMaxLowVRAMAttention(6)` + `#35 MiniMaxChunkFeedForward(4)`）跑 1 条 `s_bg=3` 验收零回退；若 `case3` 夜景光影仍有轻微偏移，按需跑 20–30 min `β[50,56,5]` 标量校准（`< 80 KB`）或测“末 5 层 L45–L49 保稠密” | `8gb_parity_ok.json` + `FINAL_REPORT.md` +（可选）`splitstream_gates.pt` | 确认 8GB 补丁链下数值与画质零退化，全部核心产物同步至 Google Drive 后停机 |

> **剩余 3h 合计**：约 **2.0 小时**完成全部高价值验证，留足 **1 小时安全余量**，绝不花时间去压测 8GB 卡根本跑不动的超高分辨率。

---

## 🎯 六、核心测试用例规格（5s / 124 帧 / `T=37` / `1280×736 = 0.94MP`，同 seed）

1. **`case1_static`（静机位表演叙事）**：固定机位汉服人物慢速呼吸、眨眼、连贯手势 + 暖光室内 + 房间底噪（标定 `s_bg=0` 极限压缩上限与发丝/衣纹保真；也可追加用户自带 `deepseek.png` 绿幕 Ref2VA 用例）。
2. **`case2_dolly`（动态推镜 + 大位移 — 重点拷问）**：缓慢推进镜头跟拍行人横穿湿润砖墙巷道 + 明显前景-背景视差 + 脚步声（检验 `s_bg=3` 下 `flow v2` vs `lerp` 残差传播与全局补偿掩码）。
3. **`case3_neon`（强光影 + 景别切换 — 重点拷问）**：夜景霓虹强侧逆光室内/街景、中景转特写 + 地面/墙面投影交互 + 连续环境音（检验跨镜连贯与阴影环 `ring=1` vs `ring=0` 效果）。

> **说明**：[stage0_oracle.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/stage0_oracle.py) 与 [stage1_ablate.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/stage1_ablate.py) 已内置自动判断：当 prompt 不含 `<Picture` 标记时自动把 `local_dualclock_api.json` 从 `Ref2VA` 切为 `T2VA` 并剥离参考图；当 prompt 包含 `<Picture 1>`（如用户自带的 `deepseek.png` 绿幕用例）时自动保留 `Ref2VA` 链路。

---

## 📦 七、最终必须回传本地的交付物清单（同步至 Google Drive）

1. `preflight_ok.json` 与 `run_state.json`
2. `probe_attn_stats.json`（50 层 × 56 头直接输出替换误差 + 双口径 $\Delta t$ 分布 + 止损结论）
3. `stage1_ablation.csv` + `stage1_contact_sheet.png`（可直接 5 秒肉眼对比 Oracle / `flow` / `lerp` / `s_bg=3` / `s_bg=0`）
4. `splitstream_gates.pt`（若触发 Stage 2，`< 80 KB`，零开销加载到本机 INT8 模型）
5. `FINAL_REPORT.md` + 3 条 5s 对比成片 +（若通过）1 条 15s 长片对比成片

---

## ✅ 八、云端 A100 + 本机 8GB 真机实测收口与 `s_bg=4` 证伪定案（2026-11-22）

### 8.1 云端 A100-40GB 最终结论（1.3MP `1568×864` & 15s 362f）
- **默认推荐配置锁死：`s_bg=3 + method="lerp" + auto-beta0`**（[nodes.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/nodes.py) 与 [wf.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/wf.py) 默认值已切换）：
  - 5s 三场景 **3/3 全绿**：`case1_static` `28.45 dB / SSIM 0.9299 / bg_hf 0.941`，`case2_dolly` `24.82 / 0.8346 / 0.941`（**超越跨 seed 稠密 Oracle 随机基线 `24.13 / 0.7998`**），`case3_neon` `25.39 / 0.8933 / 0.987`。
  - `lerp` 相对 `flow v2`：`fg_psnr` 差 `≤ 0.05 dB`、`bg_hf` 三场景全面胜出（`0.941 / 0.941 / 0.987` vs `0.940 / 0.920 / 0.943`），且耗时 **`< 1 ms`**（省去 Python 光流循环 `50–150s`）。
  - 15s（362 帧）：采样器单步 `22.4 → 7.36 s/it`（**`3.04×`**），全流程受一采/362f 解码/模型换入稀释为 **`288s → 220s`（`1.31×`）**；全帧时序核验显示 2-win `s3_lerp` 与 4-win 稠密在自然运镜峰值帧完全一致（`f171/195/219/243`），接缝 `f187` 处零跳变，整体高频锐度保住稠密的 **`95.3%`**。
  - 8GB 补丁（`MiniMaxLowVRAMAttention(6)` + `MiniMaxChunkFeedForward(4)`）在 A100 回归验证零画质回退（`24.83 / 0.8342 / 0.943` vs `24.82 / 0.8346 / 0.941`）。

### 8.2 本机 8GB RTX 4060 Laptop 真机验证 + `s_bg=4` 机制证伪实验（`case2_dolly`, 5s `1312×736`, `T=37`, 32kHz 立体声）
固定同一份冻结一采（`MiniMaxH3AVLatentLoad` 0 秒复用，保留原声轨），对比 `s_bg=3` 与 `s_bg=4` 在不同 `β` 下的表现（以同机 `oracle_dense` 为分母，并附加同核 `cuDNN BF16` 下相对 `s3_lerp` 的背景 99.9% 极值误差）：

| 变体 | `s_bg` | `β` | 二采耗时 | 峰值显存 | `180p fg_psnr` | `fg_ssim` | `bg_hf` | `full-res bg_psnr` | 相对 `s3` 背景 `p99.9` 误差 | 肉眼伪影核验 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `oracle_dense` | 稠密 | — | `282.8s` *(含TE+一采)* | `7365 MiB` | — | — | `1.000` | — | — | 基准（含 SageAttention） |
| **`s3_lerp`** | **3** | `1.046` *(auto)* | `130.9s` | `7801 MiB` | `22.66` | `0.8443` | `0.969` | **`24.89`** | `0.0` *(基线)* | **零色斑**，与 Oracle 肉眼不可分 |
| `s4_default` | 4 | `1.308` *(auto)* | `114.8s` | `7583 MiB` | `21.69` | `0.8131` | `1.011` ↑ | `22.95` | `107.3` | **严重黄褐/红褐矩形贴块**（完美复现 A100） |
| `s4_beta105` | 4 | `1.046` *(同 s3)* | `114.7s` | `7583 MiB` | `21.94` | `0.8243` | `1.009` ↑ | `23.12` | `105.3` | **色斑依旧明显**（地面/左墙/远楼仍有大块色斑） |
| **`s4_beta050`** | **4** | **`0.500`** *(0.38×)* | **`114.7s`** | `7583 MiB` | **`22.78`** | **`0.8472`** | **`0.987`** | **`23.91`** | **`96.0`** | **~95% 色斑消失**，前景反超 `s3`，仅墙角极弱残影 |

- **证伪定案（推翻两种单一归因，确立耦合失稳机制）**：
  1. **排除「纯 2-hop 几何断崖不可救」**：保持完全相同的 `s_bg=4`（2-hop）几何不变，只把 `β` 从 `1.308` 压到 `0.500`，黄褐矩形贴块即消除 ~95%，`fg_psnr` 回升 `+1.09 dB`（反超 `s_bg=3`）、`bg_hf` 从异常 `1.011` 回落正常 `0.987`。
  2. **排除「`β=1.05` 是跨步长通用安全常数」**：在**完全相同的 `β = 1.046`** 下，`s_bg=3` 零色斑，而 `s_bg=4` 依然长出大片色斑——说明步长拉大导致锚点间真实注意力相关性衰减更陡，`β_true_equiv(s)` 远小于理论上限 `ln(s)`，未衰减的 `β0` 与更陡的 2-hop 相关性衰减叠加，在 50 层 DiT 内触发局部背景锚点自激塌缩。
  3. **工程落地**：生产默认稳选 **`s_bg=3 + lerp + auto-beta0`**（本机 8GB `bg_psnr 24.89` 全程零色斑）；代码已新增 `beta_scale` / `beta_max` 参数（[engine.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/engine.py) / [nodes.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/nodes.py)），支持进阶用户用 `s_bg=4 + beta_scale=0.38`（或 `beta_max=0.50`）再省 ~12% 二采耗时；同时已修复 `h3_upscale.py` 逐窗 `reset_peak_memory_stats()` 真峰值日志与 `nodes.py` 张量 `beta_path` 加载格式化 bug。

---

## 🗺️ 九、远期优化规划（二采极致省显存 + SageAttention 兼容，待核心画质彻底稳定后推进）

1. **接入本机 `sageattention(..., return_lse=True)` 作为 Route 3 首选 LSE 后端**：
   - 现状：Route 3 二采走 `cuDNN` BF16 FlashAttention，覆盖了 `#17 SageAttention INT8`（不报错，但少吃一层 INT8 加速）。
   - 已核验本机 `comfyenv` 的 `sageattention`（`sageattn` / `sageattn_qk_int8_pv_fp16_cuda` / `fp8`）原生支持 `return_lse=True`；后续在 [route3/splitstream_attn.py](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/route3/splitstream_attn.py) 校准 `ln` vs `log2` 换算后接入，实现「Route 3 减 token + Sage INT8」双加速叠加、零冲突。
2. **「一采不改、二采只画人物」绿幕极致省显存两条路线（人物仅占 ~22% 画布）**：
   - **路线 A · 绿幕双锚 `I+P`（只留 `t=0, 1` 两帧纯绿锚 + 可选纯绿区空间抽稀）**：保留完整 16:9 原生 RoPE（零变形风险），二采 token 直接砍掉 **73%~77%**，`qkv` 激活显存从 `2.1 GB → 0.56 GB`，8GB 卡释放出 ~2–3 GB 显存多常驻 6–8 层 DiT 权重，反过来缓解 PCIe 搬运瓶颈。
   - **路线 B · 二采 ROI 人物外接框裁剪**：按一采绿幕掩码裁出含 64px 绿边的紧矩形框（对齐 32 倍数，如 `1568×864 → 768×864`）再送 3D 上采样与二采，出图后贴回纯绿画布——**连 3D 上采样与 VAE 解码显存一起砍半**，且原生直通 `#17 SageAttention` 零冲突（需实测 2D RoPE 子坐标切片防比例微偏移）。
3. **已排除的不可行方向（物理边界记录，避免后续走弯路）**：
   - **不对人物前景内部做时间冻结/丢弃 token**：H3 视频 P-token 已是 `4×16×16`（1024×）高密度潜变量，人物内部任何呼吸/微动/光影变化一旦被冻结或按 `32×32px` 丢弃，空间相邻 patch 轨迹撕裂，直接产生硬边方框切块或隔帧糊脸。
   - **不对 H3 50 层 DiT 做无训练的通用 token 合并（ToMe）**：50 层注意力强依赖精确 `(t,y,x)` 3D RoPE 相位，无训练 merge 会打碎空间/时间相位对齐，画质劣于抽帧。
   - **复杂实景有运镜场景不能靠纯 0 步插值（`lerp/flow`）完全追平稠密**：两个相隔 ~0.67s 的锚点高频残差加权平均必有方差衰减，导致 `drp↔ANC` 清晰度呼吸跳变；实景稳画质需结合「末步全稠密收尾（`last_step_dense`）」等步骤级混合策略。
