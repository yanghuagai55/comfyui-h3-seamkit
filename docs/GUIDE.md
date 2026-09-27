# 使用说明书 · comfyui-h3-seamkit

> 面向使用者的完整说明。README 讲"是什么/怎么装"，本文件讲**为什么这样跑**与**每个参数属于哪个家族、怎么用**。
> 版本：2026-09-27（上下游节点拆分版）

---

# 第一章 · 工作原理

## 1.0 数据模型：帧、token、秒（一切换算的根）

H3 的时间轴不是均匀帧，而是**按 token 组组织**的：

```text
   1 组 = 5 个 token = 17 帧 = 0.708 秒（24fps）
   组内分配：FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
            第 1 个 token 只带 1 帧，后 4 个各带 4 帧
```

| 换算 | 公式 / 值 |
|---|---|
| 组 → 帧 | 17 帧（**边界网格：切点只能是 17 的倍数**，`FRAME_GRID=17`） |
| 组 → 秒 | 17 ÷ 24 = **0.708s** |
| token → 帧 | 组内累加（非均匀：第 1 个 1 帧、其余各 4 帧），`frames_for_tokens` 逐项求和 |
| 一采 latent 形状 | video `1×24×T×H×W` + audio `1×32×2×N`（T = token 数） |

例：窗口 0 = 帧 `[0, 85)` = 25 个 token（5 组）。**为什么边界只能落 17 的倍数**：
边界必须落在"独占帧"上（模型在这里有干净的转镜位），17 的倍数就是每组的组界。

## 1.1 总流程

![总流程：输入 → 上游规划 → 一采(缓存) → 二采执行器 → 解码合成](img/flow.png)

<sub>总流程：输入 → 上游规划 → 一采(缓存) → 二采执行器 → 解码合成</sub>

**核心思想**：提示词写"哪一秒换镜头"，执行器按 17 帧网格切窗、逐窗重生成、再拼接。
窗口边界落在哪里、拼缝怎么处理，由两个规划节点控制：

- **上游规划**（一采上游）：画布 + 切点规划 + 一采开关 → 它的 plan 会改写条件节点里的提示词时间戳
- **下游规划**（二采上游）：全部二采参数 → 它不在一采的上游链里，**随便改不触发重采**

## 1.2 规划：切点与窗口尺寸怎么定

输入只有两个数：`total_seconds`（片长）与 `target_segment_seconds`（每段上限秒数）。

```text
   ① 总帧数   F = round(秒 × 24) 对齐到 17n+5 网格
   ② 每段上限 C = round(target_segment_seconds × 24) 吸附到 17 的倍数
   ③ 分段     贪心切分，使每段 ≤ C 且尾段 ≥ 17 帧
   ④ 负载校验 最长窗 × canvas_mp ≤ LOAD_PASS(180)   ← 191 = 实测 OOM 线（带质量 LoRA）
              超线 → 自动增加段数摊薄
```

三个"缝"参数**参与窗口尺寸计算**，所以必须在上游节点上（改它们 = 改切点 = 该重采）：

- `overlap_frames`（缝的重叠帧数，0 = 全硬切）
- `auto_calm_search`（开 calm 搜索）
- `calm_overlap_frames`（calm 缝的重叠帧数，17 = 一个 token 组）

**负载线**：`LOAD_PASS=180`（带质量 LoRA 的实测安全值）、`LOAD_FAIL=191`（实测 OOM）。
15s / 1.3MP 的典型布局：切点 85 / 187 / 272，四窗 `[85, 102, 85, 90]` 帧。

![时间轴与窗口划分：15s → 4 个窗口，边界只能落 17 的倍数](img/timeline.png)

<sub>时间轴与窗口划分：15s → 4 个窗口，边界只能落 17 的倍数</sub>

## 1.3 一采（#93）与缓存

```text
   上游有任何变化 ──► 指纹变 ──► 缓存 MISS ──► 重新采样并写缓存
   上游完全一致   ──► 指纹同 ──► 缓存 HIT  ──► 直接读回一采 latent（跳过采样）
```

**指纹怎么算**（`nodes_latent_cache.py` → `upstream_subgraph_from_inputs`）：

1. 从 #93 的**每一个链接输入**出发（noise / guider / sampler / sigmas / latent_image / plan）
2. 沿连线反向遍历整张上游子图，每个节点记录三样：**类名** + **全部字面量输入**（控件值 / 文本 / 文件名 / 种子）+ **连线的目标节点 id**
3. 按 key 排序后序列化成 JSON → **sha256**（取前 12 位做文件名）

**关键推论**：上游规划节点的输出会改写条件节点里的**提示词时间戳** → 它确实在 #93 的上游 →
**它身上所有控件的值全部进指纹**。这正是"二采参数必须搬去下游"的原因：
下游节点不在 #93 的上游子图里，改它看不见。

缓存文件 = `<key>.<指纹前12位>.pt` + 同名 `.json`（记录 seed / shapes / 节点清单），三个分支：

| 情况 | 行为 |
|---|---|
| key 存在 + 指纹一致 | **HIT**：读出 latent，整段跳过采样（约省 9 分钟） |
| key 不存在 | **MISS**：正常采样，结束后写缓存 |
| key 存在 + 指纹不一致 | **不使用**：正常采样；保存时覆盖旧档，并打印"哪里变了"（文本前 70 字 / 种子 / 节点类清单） |

## 1.4 二采执行器（#40）逐窗做什么

![二采执行器逐窗六步：切片 → 上采样 → 锚定 → 采样 → 装配 → 可选去噪](img/pass2.png)

<sub>二采执行器逐窗六步：切片 → 上采样 → 锚定 → 采样 → 装配 → 可选去噪</sub>

窗口 = `[start_frame, end_frame)`，帧号（起点必须是 17 的倍数）。

| 步 | token 级的具体动作 |
|---|---|
| **1 切片** | 取 `video[:, :, start_token:end_token]` |
| **2 上采样** | 输入**向外扩**：`pt_lo = max(0, start_token − pad)`、`pt_hi = min(T, end_token + pad)`；升完裁回本窗：`chunk[:, :, _trim : _trim + 本窗 token 数]`（`_trim = start_token − pt_lo`）。音频按 `frames_for_tokens` 换算区间直通 |
| **3 锚定** | 本窗开头 `anchor_tokens` 个 token 以 keyframe 条件钉在上一窗输出上（`noise_aug 0.999`）。只在**有重叠的缝**上生效；硬切的窗口不带锚定 |
| **4 采样** | 整窗重新生成（DualClock 采样器）；显存 ≈ 段帧数 × 二采 MP，有负载护栏 |
| **5 装配** | 见 1.5 |
| **6 去噪** | 见 1.8（默认关） |

## 1.5 拼缝：hard cut / anchor overlap / blend

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
- `transition > 0` 时，重叠区尾部若干 token 用本窗值**硬替换**——仍是替换而非混合，
  "可见交界"落在 `start_token + locked` 这个 token 上（写进报告作 `seam_marks`）
- `seam_blend=true` 改走 `_append_video`：同一区间做 **latent 域线性溶解**
  `left + (right − left) × linspace(0,1)`，再统一解码 → 静止内容平滑，运动内容可能重影

## 1.6 hunt：找"模型真正换镜头的那一帧"

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

## 1.7 calm：hunt 不可靠时怎么放这条缝

⚠ 这里有**两条方向相反的策略**（`calm_policy`），不是只有"找最平缓"。

**候选**：`计划切点 ± calm_search_window`（默认 34，你的配置 51）内的全部 17k 独占帧。

### 策略 A · `calm_overlap`（默认，静态/对话片）——取分数**最低**者

四道门（任一不过退回计划点）：

| 门 | 默认 | 语义 |
|---|---|---|
| `calm_min_gain` | 0.15 | 候选分数必须比计划点低 ≥15%（防"压线抖动"：0.09 分之差就会翻结论） |
| `calm_too_quiet_below` | 0.05 | 候选过于静止 → 不挪（4 帧量化的顿挫会显眼） |
| `calm_min_quality` | 0.8 | 候选分数仍 > 0.8 × 全片中位 → 不值得搜 → 退回计划点**硬切** |
| `calm_abstain_below` | 0（关） | 全片 jerk 对比度 max/mean 低于本值 → 整片不搜、全保计划 |

挪动成功 → 该缝给 `calm_overlap_frames`（17f）的锚定 overlap。

### 策略 B · `jerk_hardcut`（动作 / 运镜剧烈片）——**反过来**取 jerk **最猛**者

```text
   burst(f) = jerk[t−5] + jerk[t] + jerk[t+5]        ← 3-token 窗（同相），防单点尖峰胜出
   peak = max(候选, key=burst)  →  硬切，overlap = 0（**不锚定**）
   burst 全为 0（全片无 jerk）→ 退回计划点，硬切
```

设计理由（出生提交 `c573602` 原话：*continuity is not expected here, concealment is*）：

**为何不 overlap**：

1. 锚定/overlap 的意义是"续"，这里不需要续；**在最猛的运动处冻结 17 帧 = 最显眼的停顿**，比跳切糟得多
2. 两窗在缝附近本来就都糊（高 jerk 处模型"放弃"），硬切的锐度台阶最小
3. 当时做过**镜像实证**：同一 profile 上 calm 选 170+overlap、jerk 选 204+硬切
   （外部审计 `CODE_REVIEW_20260922.md` D3 亦确认"那是本意"）

**overlap 与输入的关系（重要）**：

| 分支 | overlap | 受 `calm_overlap_frames` 影响？ |
|---|---|---|
| jerk_hardcut 主路径（找到 jerk 峰 / 无 jerk） | **写死 0** | **无关** |
| jerk_hardcut 无法执行时的**回退**（abstain / 无候选 / 无分数） | `calm_overlap_frames`（17f） | **有关**——刻意回退到锚定语义（搜不了 → 锚定而非硬切，审计 D3 哲学） |

即：**策略本身写死 0；只有"策略跑不了"的回退才用你输入的重叠值**。
（`calm_overlap_frames` 在上游节点上——它参与窗口尺寸计算；下游只放策略与门禁。）

**负载护栏**（两条策略共用）：挪动若使某段 `帧数 × canvas_mp ≥ 负载线` → 撤销挪动、退回计划点。

## 1.8 缝窗重去噪（`seam_redenoise`，默认关）

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
**窗口内音频全自由**（mask 全 1）——但见下方"音频"：生成音频**不会**被发布。

**锁端怎么保持**：不是只在开头注入一次，而是**采样每一步都把锁端重注入一次**
（`KSamplerX0Inpaint`，时间步 0.999）→ 模型始终在"两侧都是干净关键帧"的条件下解中间区域
（RePaint 思路；ComfyUI 核心已实现，本包未搬运其代码）。
噪声用全局噪声切片 `[w0:w1]`（与首轮同相，不引入新随机性）。实测锁端收敛到已发布 latent，误差 ~2e-7。

**作用范围**：只作用于**锚定 overlap 缝**（`seam_marks` = `start_token + locked`，硬切永不进入）——
所以它与 `jerk_hardcut` 互斥（那边是硬切、不进名单），与 `calm_overlap` 是天然搭档：
锚定 overlap 先铺结构，redenoise 重画"A 冻结前缀 | B 新鲜渲染"的交界（窗口正好骑在交界上，
外侧锁 A 前缀尾 3 token、内侧锁 B 开头 3 token，中间重画接缝本身）。

**前置门**（`seam_redenoise_gate=auto`）：每条缝算"闹度" =
缝邻域 5 个 token 的 `max(global, jerk)` 均值 ÷ 全片中位；**≥ 1.5 直接跳过**（闹处重画容易生伪纹理）。
`seam_redenoise_frames` 可指定只做某几条缝（逗号分隔帧号，±17 帧内匹配）。

**事后否决**：`stroke_check.py A.mp4 B.mp4` → 裂纹笔画比 B/A > 1.2 即判注入伪纹理，回退。

**音频**：联合生成时音频参与（帮视频与声音对齐），但写回**只有视频**
（`accumulated[:, :, w0:w1] = new_video`）——成片音频仍是参考音频透传，**对白/音效不会被破坏**。

## 1.9 音频与解码

- **音频透传**：参考音频整片透传，不随视频切（条件节点的音频槽全片同源）
- redenoise 窗口内生成的音频被丢弃（见 1.8）
- 解码：`AVDecodeT8` 一次解码整条拼接后的 latent → VHS_VideoCombine 写 mp4

---

# 第二章 · 节点输入：按家族

## 2.0 家族图谱（哪组按钮管哪个功能、在哪个节点上）

| 家族 | 管什么 | 所在节点 | 改了会重采一采吗 |
|---|---|---|---|
| **画布家族** | 分辨率 / 模型 / 精度 | 上游 Plan | **会**（尺寸进切点计算） |
| **切点家族** | 每段多长 / 缝的重叠 / calm 开关 | 上游 Plan | **会**（就是切点本身） |
| **缓存家族** | 一采缓存开关 / key / 目录 | 上游 Plan | 不会（但改 key = 换缓存） |
| **提示词家族** | 提示词本体 / 校验宽严 | 上游 Plan + 校验节点 | **会**（提示词进指纹） |
| **采样家族** | 步数 / shift / 采样器名 | 两个 T8 采样器节点（#30/#38） | 一采那只**会**；二采那只不会 |
| **hunt 家族** | 找真转镜 | 下游 Pass-2 Plan | 不会 |
| **calm 家族** | 挪界搜索 / 落缝策略 | 下游 Pass-2 Plan | 不会 |
| **锚定家族** | keyframe 锚定强度与宽度 | 下游 Pass-2 Plan | 不会 |
| **拼缝家族** | 三态怎么选 / 上采样时间重叠 | 下游 Pass-2 Plan | 不会 |
| **重去噪家族** | 缝窗重去噪 | 下游 Pass-2 Plan | 不会 |
| **诊断家族** | dump / 显存日志 | 下游 Pass-2 Plan | 不会 |

## 2.1 接线总览

```text
   上游 Plan ──plan──┬─► #93.plan          （缓存家族的开关全在 plan 里）
                     ├─► #40.plan          （切点/画布）
                     └─► 条件节点 #8/#41   （prompt / 宽度 / length）

   下游 Pass-2 Plan ──pass2_plan──► #40.pass2_plan   （除缓存外的全部二采参数）

   #93 其余端口: noise / guider / sampler / sigmas / latent_image
   #40 其余端口: model / conditioning / latent / noise / sampler / sigmas / negative
```

**两个执行节点零控件**；所有可调参数集中在两个规划节点 + 两个 T8 采样器上。

## 2.2 画布家族（上游 Plan · 8 控件）

> 管分辨率与模型。**改任何一项 = 重采一采**（尺寸进切点计算）。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `total_seconds` | 片长（秒） | 8.0 | 每条片按需 | **15** |
| `first_megapixels` | 一采画布 MP | 0.4 | 一般不碰 | 0.4 |
| `second_megapixels` | 二采画布 MP | 1.5 | 要更高清就加，盯负载线 | **1.3** |
| `aspect_ratio` | 画幅 | 16:9 | 竖屏改 9:16 | 16:9 (Widescreen) |
| `multiple` | 分辨率对齐粒度 | 32 | 一般不碰 | 32 |
| `model_name` | 上采样模型 | — | 换模型时 | minimax_h3_latent_upscaler_3d_fp16 |
| `precision` | 模型精度 | bf16 | 8GB 卡别升 | bf16 |
| `release_policy` | 模型释放策略 | clear_after | 显存紧保持 | clear_after |

**联动**：`second_megapixels` × `aspect_ratio` → `canvas_mp` → 负载校验（1.2）与二采分辨率。

## 2.3 切点家族（上游 Plan · 4 控件）

> 管窗口怎么切。**改任何一项 = 重采**（切点进提示词时间戳）。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `target_segment_seconds` | **每段最长秒数** | 5.0 | 分块更粗/更细 | **4.96** |
| `overlap_frames` | 缝的重叠帧数 | 0 | **0=全硬切；17=锚定重叠**（推荐） | **17** |
| `auto_calm_search` | 开 calm 挪界搜索 | False | **建议开**（关=切点不挪） | **true** |
| `calm_overlap_frames` | calm 缝的重叠帧数 | 17 | 一般不动（17=一组） | 17 |

**联动**：`auto_calm_search=true` 时窗口尺寸按 `calm_overlap_frames` 预留（否则按 `overlap_frames`）——
所以这两个值在上游（尺寸是它们的功能）。**`calm_policy` 不在这里**——它是下游的落缝策略。

## 2.4 缓存家族（上游 Plan · 4 控件）

> 管一采缓存。这些值**随 plan 下发**给 #93（#93 零控件，只收 plan 口）。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `use_cache` | 一采缓存总开关 | True | 想强制重采时关 | true |
| `cache_key` | 缓存标识（换 key = 换缓存） | s15_cam | 换剧情/换片时改名 | **s15_cam** |
| `cache_path` | 缓存目录（空=输出目录下） | 空 | 想放别的盘 | 空 |
| `require_sage_patch` | 要求显存补丁在位 | True | 8GB 卡保持开 | true |

**联动**：HIT = key 相同 **且** 指纹一致（1.3）。改 `cache_key` = 主动弃用旧缓存。

## 2.5 提示词家族（上游 Plan + 校验节点）

| 控件 | 位置 | 作用 | 什么时候改 |
|---|---|---|---|
| `prompt` | 上游 Plan | 六段提示词本体 | 每条片（照 `templates/prompt-template.md` 写） |
| `loose_prompt` | 上游 Plan | 校验宽松模式 | 结构拿不准时开 true |

**识别语义（2026-09-27 放宽后）**：

- 镜头声明 = **行首的 `[Shot N]`**；`[Shot 2]` 起跟 `At + 时间点`——时间点**任意浮点都行**
  （`At 00:03.75` 与 `At 00:03.750` 等价），不强求 0.001 精度
- 时间戳与执行器边界对不齐 → **警告**（列出执行器边界作参考），不再拦停；
  overlap 模式下由锚定 overlap 吸收
- 同一行/正文里**其余位置**的 `[Shot N]`（如 "closer than in [Shot 1]."）会被误当声明
  （实测制造过幽灵第 5 镜）→ 校验器发现会**警告**；引用前面的镜头请用文字
  （`the opening wide shot`）
- ~~声音跨切点连续~~ 不再要求：硬切本来就是突变几帧删掉，音频整片透传自然连续
- 模板见 `templates/prompt-template.md`（官方六段骨架 + 4 条注意事项）

## 2.6 采样家族（两个 T8 采样器节点）

> 一采那只（#30 MultiRateSamplerEXPT8）的参数进指纹（它在 #93 上游）→ 改 = 重采；
> 二采那只（#38 DualClockSamplerT8）在 #40 上游 → 不进一采指纹。

| 控件 | 哪只 | 作用 | 你的值 |
|---|---|---|---|
| `video_steps` | 一采 | 视频宏步数 | **6** |
| `audio_steps` | 一采 | 音频微步数（=真 DiT 调用数，≥ video_steps） | **8** |
| `shift_video` / `shift_audio` | 两只都有 | 视频/音频 shift | 12 / 3 |
| `steps` | 二采 | 二采步数 | **3** |
| `sampler_name` | 二采 | dual_clock_euler（原生双钟路径） | dual_clock_euler |
| `scheduler` | 二采 | native_flow（官方流调度） | native_flow |
| `cfg` | 下游 Pass-2 | 二采 CFG | 1.0 |

## 2.7 hunt 家族（下游 Pass-2 · 5 控件）

> 管"找真转镜"。原理见 1.6。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `auto_seam_hunt` | 总开关：找真转镜并挪边界 | False | **有真转镜的片子开** | **true** |
| `seam_tolerance_frames` | 残差容差（0-8） | 4 | 硬切更严填 2 | 4 |
| `hunt_search_window` | 搜索半径（帧） | 34 | 转镜离计划点远 → 加大 | 34 |
| `hunt_min_persistence` | 采纳门（防闪烁/抖动） | 0.8 | 误切多 → 提高 | 0.80 |
| `hunt_persistence` | 持续性计算开关 | True | 保持 | true |

**联动**：`hunt_persistence=false` 时排序退化（0.25 地板），`hunt_min_persistence` 失效。

## 2.8 calm 家族（下游 Pass-2 · 8 控件）

> 管"hunt 不可靠时怎么放缝"。原理与双策略见 1.7。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `calm_search_window` | 搜索半径（帧） | 34 | 你的配置 | **51** |
| `calm_policy` | **落缝策略**（两条方向相反） | calm_overlap | 动作片 → jerk_hardcut | calm_overlap / jerk_hardcut |
| `calm_min_gain` | 挪动收益门（仅 A 策略） | 0.15 | 挪太频繁 → 提高 | 0.15 |
| `calm_min_quality` | 逐缝质量门（仅 A 策略） | 0.8 | 保持 | 0.80 |
| `calm_too_quiet_below` | 过静保护（仅 A 策略） | 0.05 | 保持 | 0.05 |
| `calm_abstain_below` | 整片平淡放弃搜索 | 0 | 动作片可设 2.0 | 0.00 |
| `profile_camera_compensate` | 剖面做镜头补偿 | False | **提示词有运镜时开** | **true** |
| `profile_reduce` | 剖面聚合方式 | mean | 保持 | mean |

**联动**：A 策略的 `calm_min_gain/quality/too_quiet` 三门只在 `calm_policy=calm_overlap` 时生效；
`jerk_hardcut` 只看 `burst`。`profile_camera_compensate` 影响剖面 → 也影响 hunt（共用一份 profile）。

## 2.9 锚定家族（下游 Pass-2 · 2 控件）

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `anchor_tokens` | 锚定 keyframe 的 token 数 | 1 | 想更强锚定 2-5（E-2 实验） | 1 |
| `anchor_strength` | 锚定强度（接口保留） | 0.999 | 不动 | 0.999 |

## 2.10 拼缝家族（下游 Pass-2 · 2 控件）

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `seam_blend` | 冻结式 ↔ latent 交叉淡化 | False | **静止/对话戏开；动作戏关** | 对话 true / 动作 false |
| `upscale_pad_tokens` | 上采样分块时间重叠 | **3** | 回旧行为填 0 | 3 |

## 2.11 重去噪家族（下游 Pass-2 · 5 控件）

> 原理见 1.8。**只作用于锚定 overlap 缝**——与 `jerk_hardcut` 互斥。

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `seam_redenoise` | 总开关 | False | 缝有真台阶时开 | false |
| `seam_redenoise_frames` | 只做指定缝 | 空 | 定点修复（单缝实验） | "85" |
| `seam_window_tokens` | 缝窗大小 | 10 | 想要更大自由区 → 12-14 | 10 |
| `seam_lock_tokens` | 两端锁定数 | 3 | 保持（≥ 窗/2 会整窗锁死） | 3 |
| `seam_redenoise_gate` | 闹度自动门 | off | 全开自动时设 auto | off |

## 2.12 诊断家族（下游 Pass-2 · 3 控件）

| 控件 | 作用 | 默认 | 什么时候改 | 举例 |
|---|---|---|---|---|
| `dump_latents` | 中间 latent 落盘（离线解剖） | False | **只在调查问题时开** | false |
| `dump_dir` | dump 输出目录 | latent_dump\ | 按运行改名防覆盖 | `...\latent_dump\A5\` |
| `show_memory_log` | 每窗打印显存 | True | 保持 | true |

## 2.13 场景速查

| 场景 | 配置 |
|---|---|
| **动作 / 运镜剧烈的片** | `overlap_frames=17` + `auto_seam_hunt=true` + `auto_calm_search=true` + `profile_camera_compensate=true` + **`calm_policy=jerk_hardcut`（缝藏进最猛的运动里）** + `seam_blend=false` |
| **对话 / 静态镜头片** | 同上，但 `calm_policy=calm_overlap`（缝挪到最平缓处）+ `seam_blend=true`（交叉淡化在静止内容上零重影） |
| **剧烈连续动作的缝还想更平滑** | 上述动作配置 + `seam_redenoise=true` + `gate=off` + `seam_redenoise_frames="<缝>"`（单缝实验，1.8） |
| **只想快速出片（不追缝）** | `overlap_frames=0`（全硬切）+ hunt/calm 关 —— 要求提示词切点非常准 |
| **换剧情 / 换片** | 改 `cache_key`（避免读到上一条片的缓存） |
| **怀疑某段软化 / 裂纹** | `dump_latents=true` + `dump_dir` 指新目录 → `latent_decode_lab.py` 离线解剖 |

## 2.14 指纹规则速记（哪些改动重采）

```text
   重采 ✗   上游 Plan 的 16 控件（画布/切点/缓存/提示词）＋ 一采采样器（#30）＋ 条件节点
   不采 ✓   下游 Pass-2 的 27 控件（hunt/calm/锚定/拼缝/重去噪/诊断/cfg）＋ 二采采样器（#38）
```

---

## 附：文档地图

| 文件 | 内容 |
|---|---|
| `README.md` | 概览 / 安装 / 节点一览 |
| `docs/GUIDE.md` | **本文件**——原理 + 参数 |
| `templates/prompt-template.md` | 提示词模板（官方六段骨架 + 硬切注意事项） |
| `MANUAL.md` 已删除 | 旧作业手册（被本文件取代） |
| `_hardcut_work/` | 调查工作区（测试工具 / dump / 策略单测） |
