# 使用说明书 · comfyui-h3-seamkit

> 面向使用者的完整说明。README 讲"是什么/怎么装"，本文件讲**为什么这样跑**与**每个参数怎么用**。
> 版本：2026-09-26（上下游节点拆分版）

---

# 第一章 · 工作原理

## 1.1 总流程（一张图）

![总流程：输入 → 上游规划 → 一采(缓存) → 二采执行器 → 解码合成](img/flow.png)

<sub>总流程：输入 → 上游规划 → 一采(缓存) → 二采执行器 → 解码合成</sub>

**核心思想**：提示词里写"哪一秒换镜头"，执行器按 **17 帧网格**把整片切成几个窗口，
**逐窗重新生成**再拼起来。窗口边界落在哪里、拼缝怎么处理，全部由两个规划节点控制。

![时间轴与窗口划分：15s → 4 个窗口，边界只能落 17 的倍数](img/timeline.png)

<sub>时间轴与窗口划分：15s → 4 个窗口，边界只能落 17 的倍数</sub>

## 1.2 各部分怎么实现的

### ① 一采（#93）与缓存

```text
   上游有任何变化 ──► 指纹变 ──► 缓存 MISS ──► 重新采样并写缓存
   上游完全一致   ──► 指纹同 ──► 缓存 HIT  ──► 直接读回一采 latent（跳过采样）
```

**指纹怎么算**（`nodes_latent_cache.py` → `upstream_subgraph_from_inputs`）：

1. 从 #93 的**每一个链接输入**出发（noise / guider / sampler / sigmas / latent_image / plan）
2. 沿连线反向遍历整张上游子图，每个节点记录三样：**类名** + **全部字面量输入**（控件值 / 文本 / 文件名 / 种子）+ **连线的目标节点 id**
3. 按 key 排序后序列化成 JSON → **sha256**（取前 12 位做文件名）

**关键推论**：上游规划节点（#102）的输出会改写条件节点里的**提示词时间戳** → 它确实在 #93 的上游 →
**它身上 16 个控件的值全部进指纹**。这正是"二采参数必须搬去下游"的原因：
下游节点不在 #93 的上游子图里，改它看不见。

缓存文件 = `<key>.<指纹前12位>.pt` + 同名 `.json`（记录 seed / shapes / 节点清单），三个分支：

| 情况 | 行为 |
|---|---|
| key 存在 + 指纹一致 | **HIT**：读出 latent，整段跳过采样（约省 9 分钟） |
| key 不存在 | **MISS**：正常采样，结束后写缓存 |
| key 存在 + 指纹不一致 | **不使用**：正常采样；保存时覆盖旧档，并打印"哪里变了"（文本前 70 字 / 种子 / 节点类清单） |

### ② 二采执行器（#40）逐窗做什么

![二采执行器逐窗六步：切片 → 上采样 → 锚定 → 采样 → 装配 → 可选去噪](img/pass2.png)

<sub>二采执行器逐窗六步：切片 → 上采样 → 锚定 → 采样 → 装配 → 可选去噪</sub>

窗口 = `[start_frame, end_frame)`，帧号（起点必须是 17 的倍数）。token ↔ 帧：**1 token = 17 帧 = 5 个 latent 槽**。

| 步 | token 级的具体动作 |
|---|---|
| **1 切片** | 取 `video[:, :, start_token:end_token]` |
| **2 上采样** | 输入**向外扩**：`pt_lo = max(0, start_token − pad)`、`pt_hi = min(T, end_token + pad)`；升完裁回本窗：`chunk[:, :, _trim : _trim + 本窗 token 数]`（`_trim = start_token − pt_lo`）。音频按 `frames_for_tokens` 换算区间直通 |
| **3 锚定** | 本窗开头 `anchor_tokens` 个 token 以 keyframe 条件钉在上一窗输出上（`noise_aug 0.999`）。只在**有重叠的缝**上生效；硬切的窗口不带锚定 |
| **4 采样** | 整窗重新生成（DualClock 采样器）；显存 ≈ 段帧数 × 二采 MP，有负载护栏 |
| **5 装配** | 见 ③ |
| **6 去噪** | 见 ⑥（默认关） |

### ③ 拼缝：hard cut / anchor overlap / blend

![拼缝三态：硬切 / 冻结锚定 / 交叉淡化](img/seam.png)

<sub>拼缝三态：硬切 / 冻结锚定 / 交叉淡化</sub>

装配函数 `_append_video_guarded_overlap(accumulated, chunk, start_token, locked_overlap_tokens)`：

```text
   overlap    = 已发布 token 数 − start_token      ← 与上一窗重叠多少（默认 17 帧 = 5 token）
   locked     = min(locked_overlap_tokens, overlap)
   transition = overlap − locked

   发布 = 上一窗 的 [start+locked, start+overlap)  ← 用本窗 chunk[locked:overlap] **硬替换**（不是混合）
          本窗 chunk[overlap:] 追加到末尾
```

- **默认 `locked = overlap`（整段 17 帧）→ transition = 0**：重叠区**一个 token 都不取本窗** →
  一次硬接。好处是零重影；代价是两窗渲染有差异时，接缝处表现为**跳变**
- `transition > 0` 时，重叠区尾部若干 token 用本窗值**硬替换**——注意仍然是替换而非混合，
  所以"可见交界"落在 `start_token + locked` 这个 token 上（写进报告作 `seam_marks`）
- `seam_blend=true` 改走 `_append_video`：同一区间做 **latent 域线性溶解**
  `left + (right − left) × linspace(0,1)`，再统一解码 → 静止内容平滑，运动内容可能重影

### ④ hunt：找"模型真正换镜头的那一帧"

![hunt 与 calm：一条缝怎么定形（流程 + 门禁）](img/decide.png)

<sub>hunt 与 calm：一条缝怎么定形（流程 + 门禁）</sub>

**变化剖面**（`_latent_change_profile`）——对**每个 token 边界**算 5 个数，全程无阈值：

```text
   d[i]      = |latent[i+1] − latent[i]| 在 H×W 按 profile_reduce 聚合（mean / max / top-decile）
   local[i]  = d[i] ÷ median(d[i−2 … i+2])            ← 相对自己的邻域
   global[i] = d[i] ÷ median(d)                        ← 相对全片（排序用它）
   jerk[i]   = |latent[i+3] − 3·latent[i+2] + 3·latent[i+1] − latent[i]| ÷ 同式中位   ← 三阶差分
   pers[i]   = |after − before| ÷ ½(|after| + |before|)      ← 变化前后各 10 token 的平均状态之差，再按全片峰值归一化
```

为什么是这 5 个数：

- **排序用 `global` 不用 `local`**：软化的转镜会抬高自己邻域的中位数，反而把自己的 local 压低（实测 2026-09-20）
- **`jerk` 比一阶差更接近"平静"**：值域一阶差被运动能量污染（纹理划过就会脉冲），三阶差量的是"运动变化得多突然"
- **`pers` 区分"真切断"与"闪烁/抖动/遮挡"**：真切断留在**另一个稳态**，闪烁会回落到原趋势
- 候选排序分 = `global × (0.25 + 0.75 × pers)`（pers 不可用时退化为纯 global）

**取峰与门禁**：在 `计划切点 ± hunt_search_window` 内取排序分最大者 → 依次过
`pers ≥ hunt_min_persistence(0.8)` → `global ≥ FLAT_RATIO(1.6)` → `global ≥ HUNT_MIN_CUT_RATIO(2.0)` →
吸附到最近 17k 帧 → 残差 ≤ `seam_tolerance_frames` 才判"压在转镜上" → 硬切。

### ⑤ calm：hunt 不可靠时"挪到最平缓的地方"

**取分**：与 hunt 同源的 `score[i] = max(global[i], jerk[i])`（两个视角都安静才算真安静）。
**扫描**：在 `计划切点 ± calm_search_window`（默认 34，你的配置 51）内逐 17k 独占帧评估，取分数最低者为候选。

四道门（任一不过就退回计划点）：

| 门 | 默认 | 语义 |
|---|---|---|
| `calm_min_gain` | 0.15 | 候选分数必须比计划点低 ≥15%（防"压线抖动"：0.09 分之差就会翻结论） |
| `calm_too_quiet_below` | 0.05 | 候选过于静止 → 不挪（4 帧量化的顿挫会显眼） |
| `calm_min_quality` | 0.8 | 候选分数仍 > 0.8 × 全片中位 → 不值得搜 → 退回计划点**硬切** |
| `calm_abstain_below` | 0（关） | 全片 jerk 对比度 max/mean 低于本值 → 整片不搜、全保计划 |

另有**负载护栏**：挪动若使某段 `帧数 × canvas_mp ≥ 负载线` → 撤销挪动、退回计划点。

### ⑥ 缝窗重去噪（`seam_redenoise`，默认关）

![缝窗重去噪：取哪一段、锁哪两端](img/redenoise.png)

<sub>缝窗重去噪：取哪一段、锁哪两端</sub>

**取窗**（对 `seam_marks` 里的每条缝各执行一次）：

```text
   half = seam_window_tokens // 2               ← 默认 10 // 2 = 5
   w0   = max(0, seam_token − half)             ← 窗口起点（token）
   w1   = min(总 token 数, w0 + seam_window_tokens)   ← 默认 10 个 token ≈ 34 帧
```

**遮罩**：`video_mask[0 : lock_tokens] = 0`、`video_mask[span−lock_tokens : span] = 0`
（默认各 3 个 token）——两端**锁回已发布 latent**；中间 `window − 2×lock`（默认 4 个 token ≈ 14 帧）自由生成。
**窗口内音频全自由**（mask 全 1）。

**锁端怎么保持**：不是只在开头注入一次，而是**采样每一步都把锁端重注入一次**
（`KSamplerX0Inpaint`，时间步 0.999）→ 模型始终在"两侧都是干净关键帧"的条件下解中间区域
（RePaint 思路；ComfyUI 核心已实现，本包未搬运其代码）。
噪声用全局噪声切片 `[w0:w1]`（与首轮同相，不引入新随机性）。实测锁端收敛到已发布 latent，误差 ~2e-7。

**前置门**（`seam_redenoise_gate=auto`）：每条缝算"闹度" =
缝邻域 5 个 token 的 `max(global, jerk)` 均值 ÷ 全片中位；**≥ 1.5 直接跳过**（闹处重画容易生伪纹理）。
`seam_redenoise_frames` 可指定只做某几条缝（逗号分隔帧号，±17 帧内匹配）。

**事后否决**：`stroke_check.py A.mp4 B.mp4` → 裂纹笔画比 B/A > 1.2 即判注入伪纹理，回退。

### ⑦ 上采样分块时间重叠（`upscale_pad_tokens`，默认 3）

```text
   pad = 0（旧行为）               [ 窗口 token 区间 ]              → 分块尾单侧感受野 → 尾 ~8 帧轻度软化
   pad = 3（默认）      [借3][ 窗口 token 区间 ][借3]              → 上采样后裁回窗口区间 → 尾部锐利
                        ↑pt_lo=start−3      pt_hi=end+3            ↑ 裁掉 [_trim : _trim + 本窗 token 数]
```

一采 latent 是整片**连续生成**的（无接缝），所以"借"是免费的：只是多算几个 token 的上采样。
音频直通不经 upscaler，采样用的音频仍取窗口自身区间。

---

## 1.3 测试工具（诊断用，可单独运行）

所有工具都用 `D:\comfyui\comfyenv\python.exe` 执行。

| 工具 | 看什么 | 命令例 |
|---|---|---|
| `window_edge_check.py` | 边界两侧的锐度比值（判"熔化/软化"） | `python window_edge_check.py A.mp4 --boundaries 187,272 --video-end` |
| `seam_report.py` | 逐缝台阶 / 闪烁 / 局部中位倍数比 | `python seam_report.py A.mp4 B.mp4 --seams 85,187,272 --segments 85,187,272` |
| `stroke_check.py` | 细深色"裂纹"笔画 B/A 比（>1.2 否决） | `python stroke_check.py A.mp4 B.mp4 --from 176 --to 192` |
| `zoom_check.py` | 单帧缩放翻转/打嗝（\|dev\|>6） | `python zoom_check.py A.mp4 --boundaries 85,187,272` |
| `view_grid.py` | 抽帧网格（目视） | `python view_grid.py A.mp4 --from 80 --to 92 --cols 4 --out g.png` |
| `ab_view.py` | A/B 逐像素互差定位改动区 | `python ab_view.py A.mp4 B.mp4 --out diff` |
| `latent_decode_lab.py` | **离线**重解码实验（build/decode 两子命令） | `python latent_decode_lab.py decode --variant all` |
| `hardcut_policy_test`（`_hardcut_work/seamfix/e4_*.py`） | 策略单测（改 hunt/calm 必跑） | `python e4_hardcut_policy_test.py` |
| `hardcut_math.py` | 纯计算器：片长+秒数→切点/窗口 | `python hardcut_math.py 15 "4.96" --mp 1.5` |

---

# 第二章 · 节点输入怎么用

## 2.1 接线总览

```text
   上游 Plan ──plan──┬─► #93.plan          （一采开关全在 plan 里）
                     ├─► #40.plan          （切点/画布）
                     └─► 条件节点 #8/#41   （prompt / 宽度 / length）

   下游 Pass-2 Plan ──pass2_plan──► #40.pass2_plan   （所有二采参数）

   #93 其余端口: noise / guider / sampler / sigmas / latent_image
   #40 其余端口: model / conditioning / latent / noise / sampler / sigmas / negative
```

**两个执行节点零控件**；所有可调参数集中在两个规划节点上。

## 2.2 上游 · `MiniMax H3 Plan (upstream / first-pass)`（16 控件）

> 改这里的任何值 → **一采会重采**（因为进指纹）。开工前先把值定好。

### 画布与模型

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `total_seconds` | 片长（秒） | 8 | 每条片都按需 | **15**（你的 15s 片） |
| `first_megapixels` | 一采画布（MP） | 0.4 | 一般不碰 | 0.4 |
| `second_megapixels` | 二采画布（MP） | 1.5 | 要更高清就加，但注意负载线 | **1.3** |
| `aspect_ratio` | 画幅 | 16:9 | 竖屏改 9:16 | 16:9 (Widescreen) |
| `multiple` | 分辨率对齐粒度 | 32 | 一般不碰 | 32 |
| `model_name` | 上采样模型 | — | 换模型时 | minimax_h3_latent_upscaler_3d_fp16 |
| `precision` | 模型精度 | bf16 | 8GB 卡别升 | bf16 |
| `release_policy` | 模型释放策略 | clear_after | 显存紧保持 | clear_after |

### 切点规划

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `target_segment_seconds` | **每段最长秒数**（切点由它算） | 5.0 | 想让分块更粗/更细 | **4.96** |
| `overlap_frames` | 缝的重叠帧数 | 0 | **0=全硬切；17=锚定重叠** | **17**（推荐） |
| `auto_calm_search` | 开 calm 挪界搜索 | False | **建议开** | **true** |
| `calm_overlap_frames` | calm 缝用的重叠帧数 | 17 | 一般不动 | 17 |

### 一采与提示词

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `use_cache` | 一采缓存总开关 | True | 想强制重采时关 | true |
| `cache_key` | 缓存标识（换 key = 换缓存） | s15_cam | 换剧情/换片时改名 | **s15_cam** |
| `cache_path` | 缓存目录（空=输出目录下） | 空 | 想放别的盘 | 空 |
| `require_sage_patch` | 要求显存补丁在位 | True | 8GB 卡保持开 | true |
| `loose_prompt` | 提示词校验宽松模式 | True | 严格模式改 False | true |

## 2.3 下游 · `MiniMax H3 Pass-2 Plan (downstream)`（27 控件）

> 改这里**不触发一采重采**，可以随便试。

### 采样

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `cfg` | 二采 CFG | 1.0 | 保持 | 1.0 |
| `anchor_strength` | 锚定强度（接口保留） | 0.999 | 不动 | 0.999 |
| `second_pass_audio_policy` | 二采音频策略 | joint_av_preserve_input | 保持 | joint_av_preserve_input |
| `upscale_pad_tokens` | 上采样分块时间重叠 | **3** | 想回到旧行为填 0 | 3 |

### hunt / calm 门禁（详见 1.2 ④⑤）

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `auto_seam_hunt` | 开"找真转镜" | False | **有真转镜的片子建议开** | **true** |
| `seam_tolerance_frames` | 残差容差（0-8） | 4 | 硬切更严填 2 | 4 |
| `hunt_search_window` | hunt 搜索半径 | 34 | 转镜离计划点远 → 加大 | 34 |
| `hunt_min_persistence` | 采纳门（防闪烁/抖动） | 0.8 | 误切多 → 提高 | 0.8 |
| `calm_search_window` | calm 搜索半径 | 34 | **你的片子用 51** | **51** |
| `calm_min_gain` | 挪动收益门 | 0.15 | 挪太频繁 → 提高 | 0.15 |
| `calm_min_quality` | 逐缝质量门 | 0.8 | 保持 | 0.80 |
| `calm_too_quiet_below` | 过静保护 | 0.05 | 保持 | 0.05 |
| `calm_abstain_below` | 整片平淡时放弃搜索 | 0 | 动作片可设 2.0 | 0.00 |
| `calm_policy` | 拒绝后策略 | calm_overlap | 保持 | calm_overlap |
| `profile_camera_compensate` | 剖面做镜头补偿 | False | **提示词有运镜时开** | **true** |
| `profile_reduce` | 剖面聚合方式 | mean | 保持 | mean |

### 缝处理

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `seam_blend` | 冻结式 ↔ latent 交叉淡化 | False | **静止/对话戏开；动作戏关** | 对话戏 true / 动作 false |
| `anchor_tokens` | 锚定 keyframe 的 token 数 | 1 | 想更强锚定 2-5（实验） | 1 |
| `seam_redenoise` | 缝窗重去噪 | False | 缝有真台阶时才开 | false |
| `seam_redenoise_frames` | 只重去噪指定缝 | 空 | 定点修复 | "187" |
| `seam_window_tokens` | 缝窗大小 | 10 | 保持 | 10 |
| `seam_lock_tokens` | 缝窗两端锁定 | 3 | 保持 | 3 |
| `seam_redenoise_gate` | 闹缝跳过门 | off | 开则自动跳过 busy 缝 | off |

### 诊断

| 参数 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `dump_latents` | 中间 latent 落盘（离线解剖用） | False | **只在调查问题时开** | false |
| `dump_dir` | dump 输出目录 | latent_dump\ | 调查时按运行改名 | `...\latent_dump\A5_20260926\` |
| `show_memory_log` | 每窗打印显存 | True | 保持 | true |

## 2.4 "什么情况下启用最好"速查

| 场景 | 建议 |
|---|---|
| **动作 / 运镜剧烈的片** | `overlap_frames=17` + `auto_seam_hunt=true` + `auto_calm_search=true` + `profile_camera_compensate=true` + `seam_blend=false` |
| **对话 / 静态镜头片** | 同上，但 `seam_blend=true`（交叉淡化在静止内容上零重影，能抹掉背景跳变） |
| **只想快速出片（不追缝）** | `overlap_frames=0`（全硬切）+ hunt/calm 关 —— 要求提示词里的切点非常准 |
| **缝有台阶、想定点修** | `seam_redenoise=true` + `seam_redenoise_frames="187"` 单缝试，之后用 `stroke_check` 检查裂纹 |
| **换剧情/换片** | 改 `cache_key`（避免读到上一条片的缓存） |
| **怀疑某段软化/裂纹** | `dump_latents=true` + `dump_dir` 指新目录 → 跑完后用 `latent_decode_lab.py` 离线解剖 |
