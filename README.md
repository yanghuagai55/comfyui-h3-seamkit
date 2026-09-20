# comfyui-h3-seamkit

**把 MiniMax H3 分块二采的「接缝」，变成一次有意识的剪辑。**

> **名字**：`seam` = 接缝，`kit` = 工具箱。硬切（把缝变成剪辑点）只是其中一种策略 ——
> 缝还看得见时，这里还有**磨平 / 桥接 / 叠化 / 重绘**一整套处置办法。
> 节点 ID 保留 `MiniMaxH3HardCut*` / `MiniMaxH3Seam*` 前缀不变，**已有工作流无需改动**。

分块二采在段与段之间用 `overlap` 做混合，而两段是各自独立采样的 —— 内容必然分歧，
观感就是**重影 / 突然发糊**。本插件把 overlap 设为 **0**：每个窗口独立采样、首尾直接相接，
于是接合处是一次**硬切**而不是溶解；再让提示词**在同一帧要求模型换镜头**，
硬切就成了合法的剪辑语法 —— 缝不再是缺陷。

副作用是好的：每段更短，**峰值显存更低**。RTX 4060 Laptop 8GB 上，1.5MP 画布可以跑 15 秒
（8 秒单窗在 1.0MP 就已经到负载上限）。

> **依赖上游**：[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
> （GPL-3.0-or-later）。本插件在运行时调用它的放大器 / 采样 / 条件重锚，plan 契约与它保持逐字段兼容。
> **许可**：**GPL-3.0-or-later**（与上游一致，理由见 [许可](#许可)）。
> **安装**：除 ComfyUI 与 torch 外无额外 pip 依赖。

| 文档 | 用途 |
|---|---|
| **`MANUAL.md`** | ★ **作业手册** —— 跨模型协作全流程（定切点 → 喂 LLM → 抄回 ComfyUI）+ 可复制的 LLM 约束块 |
| **`AUTO.md`** | 自动版节点 `MiniMaxH3HardCutAuto` 的使用与排查 |
| `templates/hardcut-prompt-template.md` | 提示词模板（六段骨架 + 硬切改写实例 + 常见错误） |
| `templates/auto-prompt-template.md` | 纯 LLM 用的模板（只要一句约束就能写镜头） |

## 目录

- [安装](#安装)
- [节点一览](#节点一览)
- [最短上手路径](#最短上手路径)
- [新手快速入门](#新手快速入门先看这里)
- [切点语义与参数](#参数一图流秒-vs-帧这是你问的重点)
- [节点详解](#节点)
- [接缝修复工具链](#接缝修复工具链)
- [命令行工具](#命令行工具)
- [实测数据与已知限制](#实测负载锚点rtx-4060-laptop-8gb)
- [许可](#许可)

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<你的账号>/comfyui-h3-seamkit.git
```

前置条件（缺一不可）：

| 依赖 | 说明 |
|---|---|
| ComfyUI | 用 `comfy_api.latest`（V3 节点注册），需要较新的 ComfyUI |
| `comfyui-minimax-h3-audio-T8` | **必须**。本插件在运行时按模块名后缀定位它的放大器 / 采样 / plan 构建器 |
| MiniMax H3 模型 | 底模 + Qwen3-VL 文本编码器 + video VAE + 3D latent upscaler，按上游说明安装 |

装完**重启 ComfyUI**。若上游没装，`MiniMaxH3HardCutUpscale` 会直接报
`upstream H3 upscale pack not loaded in this process`；plan 构建器找不到时会自动回退到
内置 plan，并在 `cut_report` 里打一行 WARNING（此时仍可跑，但契约同步不再有保证）。

## 节点一览

| 节点 | 分类 | 干什么 |
|---|---|---|
| `MiniMaxH3HardCutAuto` ★ | `MiniMax H3/SeamKit` | 填时长 + 档位 + 两块 MP，自动挑切点、生成时间戳、算画布、校验提示词 |
| `MiniMaxH3HardCutPlan` | 同上 | 手填切点（秒或帧）→ 执行器用的 plan + 报告 |
| `MiniMaxH3HardCutValidate` ★ | 同上 | 提示词体检：与 plan 不一致就抛错拦停 |
| `MiniMaxH3HardCutShotPrompt` | 同上 | 镜头描述 → 官方 R2V 模板的 `detailed_description` |
| `MiniMaxH3HardCutUpscale` ★ | 同上 | **二采执行器**，支持不等长分段 + 自动找真实转镜帧 |
| `MiniMaxH3InfoBuffer` | `h3_seamkit/repair` | 暂存主片帧 + 解析 report（重绘/修复的 coordinate 换算） |
| `MiniMaxH3SeamRepairAll` ★ | `h3_seamkit/repair` | 全包：切片 → 重绘 → 融合 → 磨平 → 叠化，一个节点做完 |
| `MiniMaxH3RedrawBridge` | `h3_seamkit/repair` | 只做"重绘桥段"这一支 |
| `MiniMaxH3SeamFuse` | `h3_seamkit/repair` | 把桥段按权重渐变贴回主片 |
| `MiniMaxH3SeamBlend` | `h3_seamkit/repair` | 缝两侧两帧互相靠拢（磨平跳变） |
| `MiniMaxH3SeamDissolve` | `h3_seamkit/repair` | 一段崩坏帧用两端做锚点叠化替换 |

> 规划类与执行类是可独立使用的**主链路**；`repair` 类是**救急工具**，
> 主链路跑得好的时候它们全部 bypass 即可，不影响出片。

## 最短上手路径

1. 装好上游 T8 包并确认它能跑通分块二采。
2. 把工作流里的 `MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced` 换成 **`MiniMaxH3HardCutAuto`**
   （或手填党用 `MiniMaxH3HardCutPlan`）。
3. 把 `MiniMaxH3ChunkedTwoPassUpscaleT8Advanced` 换成 **`MiniMaxH3HardCutUpscale`**
   （同名同型输入，连线自动保留）。
4. 提示词按切点写好 `[Shot N] At MM:SS.mmm, …`，跑之前看校验器报告 `status: OK`。

想先不碰 ComfyUI 就验证参数？跳到 [命令行工具](#命令行工具)。

---

## 新手快速入门（先看这里）

这台机器如果不太熟 AI 视频，别慌，这个插件只干一件事：**让长视频分成好几段来生成，
段与段交接的地方，从"糊在一起"变成"干净地切镜头"**。

### 三个最常用的名词，先混个脸熟

| 词 | 大白话 |
|---|---|
| **帧（frame）** | 视频是很多张画快速连起来组成的，一张画就是一帧。这个插件用的视频是 **1 秒 = 24 帧**。 |
| **二采（two-pass）** | 生成视频时通常先做一张小图（省显存），再放大重画一遍（变清晰）。"分块二采"就是把这个放大步骤按时间段切成几块分着做。 |
| **硬切（hard cut）** | 两段交接处，直接"啪"一刀切过去，换成另一个镜头，**不做渐变、不叠影**。 |

### ⚠️「切点」= **分割点**，不是"画面真的换镜头的那一刻"

这两个东西**经常被当成一回事，但它们不是** —— 这是本文档里最容易读错的地方：

| | 是什么 | 由谁决定 | 报告里长什么样 |
|---|---|---|---|
| **切点 / cut** | **执行器的分割点**：把 latent 切成两块分别放大的地方 | **插件**（`cut_1~4` 的 n×17 / `chunk_step`，只落在 17 帧网格上）| `cut points : 4.958s (frame 119)` |
| **画面切换** | 镜头**真的换了**（机位/景别变了）| **模型自己**（它总是**提前**起转）| 实测帧差曲线的尖峰 |

**为什么会有差别**：模型不会正好在分割点上转镜，它**提前几帧就起转了** ——
实测 8 秒片和 10 秒片**都提前 10 帧**（8s：切点 102 → 画面在 92 换；10s：切点 119 → 画面在 109 换）。

所以看到 `cut points : 4.958s` 时要知道：**那是分割点，画面在 4.542s 就已经切了**。
两者的差值就记在 `#56` 的 **`cut_offset_frames`**（默认 `-10`，即"实测突变帧 − 切点帧"，可自己量了改）。

> **一句话**：文档里的「切点」一律指**分割点**；要指"画面切换"的地方，会写成「实际转镜帧」。

### 它到底帮你省了什么

视频太长，一块儿放大容易爆显存，所以要**切开分块**。原来的做法是让两块接头处有
一段"重叠区"慢慢融合，结果两块各自生成的内容在重叠区对不上，画面就**变糊/出现重影**。

这个插件改成：**重叠区设为 0，两段之间不融合、直接接到一起**。
再配合提示词让 AI"就在这里切镜头"，接缝就变成了一次**有意识的剪辑**，看着流畅又干净。

---

## 参数一图流：秒 vs 帧（这是你问的重点）

切点只有一个写法：**填 `n` —— 切在第 `n × 17` 帧处**。

| 参数 | 写法 | 含义 |
|---|---|---|
| `cut_1` ~ `cut_4` | `cut_1 = 6` | 在**第 6 × 17 = 102 帧**（4.25 秒）处切开 |
| | `cut_1 = -1` | **不切**（该槽留空） |

**为什么填 n 而不是秒？** 切点只能是 **17 的倍数**（底层模型的硬性限制：17 帧 = 0.708 秒）。
填 n 就永远不会填出非法值 —— `6 → 102`、`12 → 204`、`18 → 306`，全是网格上的点。

### 换算表（24 fps）

| 你填 n | 切在 | 秒 |
|---|---|---|
| 4 | 68 帧 | 2.833 |
| 6 | 102 帧 | 4.250 |
| 7 | 119 帧 | 4.958 |
| 8 | 136 帧 | 5.667 |
| 12 | 204 帧 | 8.500 |
| 18 | 306 帧 | 12.750 |

### 不等长分块 = 填不同的 n

四个槽填不同的 n，段长自然就不一样：

```
cut_1 = 4                    →  [68 帧] + [剩下的]
cut_1 = 4, cut_2 = 10        →  [68] + [102] + [剩下的]
```

比如 8 秒的视频（192 帧）填 `cut_1 = 4`：第 1 段 68 帧（≈2.8 秒）、第 2 段 124 帧（≈5.2 秒）——
**两段长短不一样**，这就是"**不等长分块**"。

> 想要等长 → 看报告的 `chunk ladder`（它列出每个档位的段数与负载），或用**自动版**（`#56` 自动算）。

---

## 它在解决什么

分块二采（`guarded_overlap_exp`）在段间用 `overlap` 做混合：前一半锁定为前段输出，
后一半 smoothstep 渐变接管。两段独立采样必然有分歧，于是观感就是**突然变糊 / 重影**。
上游在 `docs/DUAL_MODEL_SEAM_FIX_20260913.md` 里明确说这类 post-sampling 混合**会产生重影**，
且「不能靠增大重叠承诺自动消除内容漂移」。

**本插件的做法：不混了。**

`temporal_overlap_frames = 0` 时，执行器里 `locked_overlap_tokens` 恒为 0
（`chunked_two_pass_upscale_advanced.py:1423` 是「已发布长度 − 本段起点」，不是 overlap），
于是硬拷回、mask、渐变整块跳过，段与段**纯追加**。

而 H3 的官方 R2V 提示词模板本来就支持在指定时间点切镜头：

```text
[Shot 2] At 00:04.250, the shot cuts to a low angle behind the hedge...
```

**两边对齐到同一帧 —— 缝就变成了剪辑。** 顺带每段变短，峰值显存也降了。

---

## 节点

### `MiniMaxH3HardCutPlan`

时长 + 最多四个切点 → **执行器可直接使用的 plan**。

| 输入 | 说明 |
|---|---|
| `total_seconds` | 1–15 秒。自动 snap 到 17n+5 帧网格。 |
| `cut_1` ~ `cut_4` | 切点 = **n × 17 帧**（`6` → 102 帧 = 4.25s）。**`-1` = 这里不切**。
填不同的 n 就是不等长分段（如 `4 / 10` → 68f / 170f）。只有正数才算一个切点。 |
| `chunk_step` | 窗口长度步进，**每 ±1 = ±17 帧**。`0` = 保持切点槽推出的长度；改它会移动切点，先看报告里的阶梯表。（不等长模式下无效） |
| `canvas_megapixels` | 只用于负载估算 —— 填你实际用的二采画布。 |
| `model_name` | 学习放大器模型文件。 |
| `target_width` / `target_height` | 二采画布分辨率，**必须与 HIGH conditioning 同源**。 |
| `precision` / `release_policy` | 建议 `bf16` / `clear_after`。 |
| `anchor_strength` | **硬切下无效**（永不进 anchor 分支），保留只为接口兼容。 |
| `second_pass_audio_policy` | 默认 `joint_av_preserve_input`（音频透传一采）。 |

输出：

| 输出 | 用途 |
|---|---|
| `plan` | → `MiniMaxH3ChunkedTwoPassUpscaleT8Advanced.plan`（和校验器 `.plan`） |
| `cut_seconds` | 实际切点（逗号分隔），可接给 ShotPrompt 节点。 |
| `cut_report` | 预览：切点 / 帧号 / 时间码、`chunk` 阶梯表、负载判定、最大画布、手动填参对照表 |

**`cut_report` 会直接告诉你「分几块、每秒切在哪、切在哪一帧」**：

```
cut inputs    : 3 of 4 slots used -> 4.250s, 8.500s, 12.750s   (-1 = no cut)
chunk/overlap : 102f = 4.250s  /  0   <- window length (6 x 17 frames) / HARD CUT
windows       : 4 window(s) of 102f (4.250s) each; the last is the tail
  #0  frames [   0 ->  102)   102f    0.000s ->   4.250s
  #1  frames [ 102 ->  204)   102f    4.250s ->   8.500s  <- CUT #1
  #2  frames [ 204 ->  306)   102f    8.500s ->  12.750s  <- CUT #2
  #3  frames [ 306 ->  362)    56f   12.750s ->  15.083s  <- CUT #3
cut points    : 3 hard cut(s)
  #1  frame  102  @   4.250s  00:04.250
  #2  frame  204  @   8.500s  00:08.500
  #3  frame  306  @  12.750s  00:12.750
```

**★ `chunk` 阶梯表** —— `chunk_step` 每档对应一行，直接看到「这档会切几刀、切在哪、负载多少、尾段合不合法」：

```
--- chunk ladder (base 102f = 6 x 17; chunk_step +0 -> 102f; one step = 17 frames) ---
  step  chunk  wins  cut times (sec @ frame)                     longest   load  note
   -2     68     6  2.833@68, 5.667@136, ...                        68  102.0  tail 22f < 48f
   +0    102     4  4.250@102, 8.500@204, 12.750@306               102  153.0  SAFE  <= selected
   +2    136     3  5.667@136, 11.333@272                           136  204.0  SAFE
```

切点可落在任意 **token 边界**（1-4 帧粒度），推荐 **17 帧的整数倍**（= 0.708 秒的倍数，独占帧，模型能放最干净的转镜），所以"想切在整数秒"常常差几帧。**先看菜单再定时间戳。**

### `MiniMaxH3HardCutValidate`  ★ 建议串在链路里

**提示词体检 —— 对不上就直接抛错拦停，不让你白跑半小时。报告直接印在节点上。**

输入 `prompt`，输出 `prompt`（**原样透传**）。所以它可以插在提示词节点和条件节点之间，
一个字节都不改你的提示词，只负责判断。

**它没有任何设置项** —— 窗口长度、切帧、二采画布全部从 plan 里读：

| 输入 | 说明 |
|---|---|
| `prompt` | **从你的提示词节点接线过来**（`PrimitiveStringMultiline` 之类）。 |
| `plan` | **从 `MiniMaxH3HardCutPlan.plan` 接线过来**。plan 是唯一真值：窗口长度 `temporal_chunk_frames`、`temporal_overlap_frames`、`target_width/height`（算画布 MP）都从它读。 |

| 输出 | 说明 |
|---|---|
| `prompt` | 原样透传，接到两个条件节点的 `prompt`。 |
| （界面文本） | 体检报告，直接显示在节点上 —— **不需要 `PreviewAny`**。 |

**判据**：节点上印的报告第一行 `status` 必须是 **OK**。不是 OK 它就抛错，跑都不会开始。

**降级行为**：`plan` 没接线时，它退化成"只按提示词自己判"（会提示你 `no plan was supplied`）。

**查 8 类问题：**

| # | 查什么 | 级别 |
|---|---|---|
| 1 | 六段 section 齐全 / 无重名 | 错 / 警 |
| 2 | `[Shot N]` 编号从 1 连续（**只在 `detailed_description` 里数**，不会把 `retention_analysis` 的引用误算成镜头） | 错 |
| 3 | **"不切"措辞** —— `no cuts` / `without a cut` / `never cuts` / `continuous take` / `unbroken` / `in one take` / `no edit`，报错还告诉你行号 | 错 |
| 4 | `[Shot 2]` 起是否都以 `At MM:SS.mmm,` 开头 | 错 |
| 5 | **执行器每一刀是否都在提示词时间戳里**（段边界是硬断点）。**提示词时间戳多于刀数 = 段内模型自己切镜，合法** | 错（只拦"刀不在提示词里"） |
| 6 | **时间戳是否落在执行器真会切的那一帧** | 错 |
| 7 | 提示词写的时长 vs `total_seconds` | 警 |
| 8 | 段长 × 画布MP 是否越过 OOM 锚点 | 错 / 警 |

**报错的样子**：

```
Hard-cut prompt check failed (1 error(s)) — the prompt and the plan disagree,
so the model would not cut where the executor cuts.
  - [Shot 2] timestamp is frame 108 (00:04.500) but the plan cuts at frame 102
    (00:04.250) — off by +6 frame(s)
```

**★ 自动匹配可分割的时间段** —— 报告末尾永远附一张可达切点菜单，照着换就行：

```
--- reachable windings for this clip (cuts are multiples of 17 frames) ---
  chunk 102  2 windows, cuts @ 4.250s
  chunk 119  2 windows, cuts @ 4.958s
  chunk 136  2 windows, cuts @ 5.667s
```

### `MiniMaxH3HardCutShotPrompt`（可选）

定调句 + 每镜描述 → 官方 R2V 模板的 `detailed_description` 段。
**有了校验器之后这个不是必需品** —— 它只是帮你少抄几次时间戳。

| 输入 | 说明 |
|---|---|
| `style_lead` | 一两句定调（风格/色调）。官方要求**放在 `[Shot 1]` 之前**。 |
| `shots` | 每镜一段，用**只含 `---` 的一行**分隔。裸描述或 LLM 的完整行都吃，不会重复加前缀/时间戳。 |
| `cut_seconds` | **从 Plan 节点的 `cut_seconds` 接线过来** —— 保证时间戳与分块边界一致。 |
| `timestamp_on_first_shot` | 官方模板 `[Shot 1]` 不带时间戳，默认关。 |

输出 `detailed_description`（抄进提示词第 4 段）+ `shot_audit`。

### `MiniMaxH3HardCutAuto`  ★ 一个节点顶替 Plan + Validate + 两个画布选择器

全自动版：只填**时长** + **`chunk_step`（17 帧档位）** + **两块的 MP**，它自己挑切点、
自己生成时间戳、自己算两块画布和帧数，顺手校验提示词，把报告印在节点上。

| 输入 | 说明 |
|---|---|
| **`prompt`** | **输入口** —— 从提示词节点接线进来。接了就校验并**原样透传**给条件节点；不接就输出六段骨架（时间戳已填好）供复制 |
| `total_seconds` | 片长（秒）→ 17n+5 帧 |
| `chunk_step` | **每段最多几个「17 帧块」**。每段最大帧数 `= chunk_step × 17`，最大秒数 `= chunk_step × 0.708` |
| `first_megapixels` / `second_megapixels` | 一采 / 二采画布 MP（取代两个 `ResolutionSelector`；二采的还用于负载估算） |
| `aspect_ratio` / `multiple` | 两块画布共用（16:9 / 32） |

| 输出 | 说明 |
|---|---|
| `plan` | 接二采执行器 |
| `prompt` | 透传（已校验）或骨架 |
| `report` | 分块报告 + 校验报告 + 报错，同时直印在节点上 |
| `first_width` / `first_height` | 一采画布 → 接一采条件节点 |
| `second_width` / `second_height` | 二采画布 → 接二采条件节点 |
| `length` | 17n+5 帧数 → 接两个条件节点的 `length`（取代 `ComfyMathExpression`） |

切分由节点内部算（筛选顺序：段数最少 → 段长方差最小 → 每段 ≤ `chunk_step×17` 帧、**尾段 ≥ 17 帧**、
负载 SAFE，最多 10 段），切不出来直接报错。**尾段下限和执行器一致**，所以自动版算出的方案一定能喂进
`MiniMaxH3HardCutUpscale`。
⚠️ **这是节点挑方案的筛选规则，不是 LLM 写镜头的规则** —— 喂给 LLM 只需一句「每段不超过
`chunk_step × 0.708` 秒」。

**给 LLM 的模板**：`templatesuto-prompt-template.md`（六段骨架 + chunk_step 换算）。
节点使用与排查：详见 `AUTO.md`。

### `MiniMaxH3HardCutUpscale`  ★ 二采执行器（可不等长分段）

**fork 自上游执行器，但分段逻辑完全自主**：复用上游的放大器/采样/条件重锚/拼接（运行时定位），
自写分块层。当 plan 带 `hardcut.segment_frames` 时按**显式切帧**切不等长窗口（如 `68 + 124`），
否则回退等长。删掉了上游的 spatial tile / mask / 多 schema / dual-clock 研究路径。

| 输入 | 说明 |
|---|---|
| `model` / `conditioning` / `latent` / `noise` / `sampler` / `sigmas` | 与上游执行器同名同型，**直接换节点、连线自动保留** |
| `plan` | 从 `MiniMaxH3HardCutPlan.plan` 接。plan 的 `hardcut.segment_frames` 控制不等长 |
| `negative` / `cfg` | 可选 |

输出 `latent`（接 AVDecode）+ `report`。

**用法**：把工作流里原来的 `MiniMaxH3ChunkedTwoPassUpscaleT8Advanced` 换成它，其余不动。
要不等长就填不同的 `n`（如 `4 / 10` → 68f / 170f）。

**切槽的四道校验**（填错立刻报错，不会静默切出跑不了的窗口）：

| # | 拦什么 | 报错样子 |
|---|---|---|
| 1 | 切点超出片长 | `cut frame 200 is not inside the clip (0, 192)` |
| 2 | 切点太靠起点、被网格吸回 token 0 | `cut frame 8 snaps back onto the clip start (token 0) …` |
| 3 | 相邻两个切点吸到同一网格点（夹出空段） | `cuts 68 and 70 snap onto the same 17-frame grid point (token 20) …` |
| 4 | **最后一个切点贴片尾，尾段不足 17 帧** | `… leaving only a 5-frame tail (0.208s) — less than one 17-frame block …` |

> 第 4 条是实测出来的：`cut=180 / 187 / 190` 都会吸到帧 187，静默切出**只有 5 帧（2 token）的尾段**，
> 采样器建不出窗口。安全的尾段约 **2 秒**（48 帧）。
> **窗口边界仍固定 17 帧网格**（可用点 17 / 34 / 51 / …），这条改不了。

---

## 接缝修复工具链

主链路跑得好时这一节可以跳过（修复节点全部 bypass 即直通）。
**只有当缝还能看出来** —— 跳变、几帧糊掉、人物边缘融化 —— 才需要它。

### 先分清两种"缝"

| 现象 | 判据 | 治法 |
|---|---|---|
| **位置错位**：缝两侧不是同一镜 | 像素域逐帧 diff 出现尖峰，且**不在**你声明的切点上 | 让边界对齐真实转镜帧（`auto_seam_hunt`） |
| **强度跳变**：两侧同镜，但清晰度/色调不一致 | 该帧 `lap`（拉普拉斯能量）显著高于邻域 | `blend_strength=0.3`，或降 `second_pass_sigma0`（0.30 → 0.22） |

⚠️ **量 `lap` 时必须排除 17 的倍数的帧**。H3 的时间压缩是 **1+4+4+4+4 token**：
块首帧独占一个 token（最锐），其后 4 帧共享一个（被平均）。实测 14 条成片**全部**有这个
17 帧周期（块首 1.8–2.1× 其余帧），**未硬切的对照片也有**（1.25×）。
这是架构固有特征，不是接缝，也不是 SageAttention / int8 的锅（量化误差是随机的，产生不了周期）。

### 让边界自动对齐真实转镜帧

执行器 `MiniMaxH3HardCutUpscale` 自带：

| 参数 | 说明 |
|---|---|
| `auto_seam_hunt` | 在 latent 域找真实转镜帧：**全窗口取最强峰 → snap 到最近的独占帧（17k）** |
| `seam_side` | `before` / `after`：峰落在转镜两侧时的取舍（见下方"固有两难"） |
| `auto_seam_sensitivity` | **已停用**，调它无效 |

report 关键字段：`measured_turn_frame`（测得的转镜帧）/ `boundary_frame`（最终采用的边界）。

**为什么是"全窗口取峰再吸附"，而不是"只在 17 倍数帧里取峰"**：转镜是信号，
独占帧只是**窗口起点**的约束。旧实现只在 17 倍数里 argmax，实测把边界从 68 判到了 85（差 17 帧）。

**两个硬约束（改不了）**：

1. **窗口起点必须是独占帧 17k**。起点落在共享组首（如 69）会让整段生成崩坏 ——
   同 seed 对比实测：68 干净 / 69 崩。曾经"只差 1 帧"的那版成片就是这么来的，画面是废的。
2. **边界精度下限约 2 帧**。latent 时间粒度是 4 帧/token，帧 66 附近的合法边界只有 64 和 68，
   而 64 属共享组首（不能用作起点）→ 只能取 68，距 66 恰好 2 帧。**这是架构精度，不是 bug。**

**固有两难**：当转镜恰好落在"独占帧的下一帧"（如 69 = 68+1）时 ——
取前肩 68 差 1 帧（缝在旧镜内，轻度可见），取后肩 85 差 16 帧（缝落在更静的段，重度可见）。
此时 `seam_side` 无法保证"缝落在转镜面上"，只能二选一。

### 修复节点怎么串

```
主片帧 (#16) ──► MiniMaxH3InfoBuffer ──► MiniMaxH3SeamFuse ──► MiniMaxH3SeamBlend ──► 保存
      #40 report ──► InfoBuffer (auto 解析切点)      ▲
                         桥段（可选） ────────────────┘
```

- **`MiniMaxH3InfoBuffer`**：主片帧直通 + 从 `#40` report 解析出 cut / seam / redraw 帧号；
  填 `video_path` 可以**只 bypass 解码节点**、几秒钟验证修复效果，生成链一条都不跑。
- **`MiniMaxH3SeamFuse`**：把桥段按权重渐变贴回主片。`bridge` 是 optional —— 不接（桥段组 bypass）就直通主片。
- **`MiniMaxH3SeamBlend`**：`strength=0` 直通；`0.3` ≈ 峰值跳变降 40%（实测）。
- **`MiniMaxH3SeamDissolve`**：崩坏区间用两端做锚点叠化替换（帧数不变）。
- **`MiniMaxH3SeamRepairAll`** ★：上面几步 + 重绘打包进一个节点，推荐直接用它。

### `MiniMaxH3SeamRepairAll`（全包）

| 输入 | 说明 |
|---|---|
| `model` / `clip` / `vae` | 重绘用，与主片同源（LoRA 链尾的 MODEL） |
| `images` | 主片帧；**留空 + 填 `video_path`** 可直接拿本地成片试 |
| `report` | `#40` 的 report —— `auto` 模式据此取切点 |
| `prompt` | 重绘提示词（描述参考片段里正在发生的事） |
| `redraw_frames` | 重绘帧数 5–39（切片长度与生成长度共用）。**H3 的 R2V 至少 5 帧** |
| `steps` / `seed` | 步数取主片同值（Turbo 则 4）；固定种子便于对比 |
| `source_mode` | `auto`（用 report）/ `manual`（手填 `manual_cut_frame`） |
| `insert_mode` | `manual`（切点 + 偏移）/ `edge`（像素域自动检测真实转镜帧，±17 帧搜索） |
| `fuse_offset` | 插入位置微调（帧），**两种模式都生效**；`−1` = 整体往前放一帧 |
| `fuse_side` / `fuse_min` / `fuse_max` | 融合权重方向（after = 贴切点最强递减）与范围 |
| `blend_strength` / `blend_mode` | 缝磨平 |
| `dissolve_start` / `dissolve_end` | 叠化区间，`[start, end)` 左闭右开，0 = 关 |

输出 `images`（修好的整片）/ `redraw`（本次重绘的 N 帧，供对比）/ `report`。

**帧语义（全插件统一）**：帧号 **0 基**，区间一律**左闭右开**。
「边界帧 68」= 该帧开头 = 缝落在 `67|68` 之间。内容切片跟随**插入位置**走，
不是围绕声明的切点 —— 早期版本两者脱节，实测整段偏移 10 帧。

---

## 命令行工具

`tools/` 下的脚本全部**不需要 ComfyUI 在跑**，装了 `av` / `numpy` / `cv2` 就能用
（ComfyUI 环境自带）。本机示例用 `D:\comfyui\comfyenv\python.exe`，换成你自己的 python 即可。

| 脚本 | 干什么 |
|---|---|
| `hardcut_math.py` | **零依赖**纯计算：帧网格、切点求解、chunk 阶梯、负载估算、提示词校验 |
| `tools/analyze_cut.py` | **成片体检**：量出画面实际在第几帧换镜，与理论切点差几帧 |
| `tools/repair_seam.py` | **像素域修复**：`--locate` / `--replace` / `--blend` / `--dissolve` / `--fuse` |
| `tools/check_widgets.py` | **体检工作流**：查本包节点的 `widgets_values` 是否与 schema 对得上 |

```bash
# 量实际转镜帧（--sheet 导出逐帧接触表方便肉眼复核）
python tools/analyze_cut.py "成片.mp4" --cuts 4.958,8.500

# 像素域找缝 / 修缝
python tools/repair_seam.py "成片.mp4" --locate
python tools/repair_seam.py "成片.mp4" --seams 67 --strength 0.3 --out 修好的.mp4

# 体检工作流（扫 user/default/workflows 下全部）
python tools/check_widgets.py
```

### ★ `check_widgets.py` —— 槽位错位是沉默杀手

ComfyUI 的 `widgets_values` 是**按位置**赋值的。改了节点 `define_schema`（删/加一个参数）之后，
工作流里**旧的那个值还留着** → 从那一槽起**整体错位一位**，节点照跑、不报错，值全歪。

判据：`widgets_values_named` 里出现**类型不可能**的值 —— COMBO 拿到数字、INT 拿到字符串。

```bash
python tools/check_widgets.py            # 扫全部工作流
python tools/check_widgets.py 某.json --verbose
python tools/check_widgets.py 某.json --fix    # 自动删残留槽位（会先 .bak）
python tools/check_widgets.py --offline        # 不连 /object_info，用内置 schema
```

⚠️ 只对**本包**节点报警：第三方包的 `upload`、`ref_images.*`、被连线覆盖的 widget
**全是假阳性**，别去修。**"多出的槽位"才危险，"少于 schema"普遍正常。**

---

## 接入现有工作流（6 步）

1. 界面里删掉旧的 `MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced`（原 `#37`）。
2. 拖入 `MiniMaxH3HardCutPlan`。**从 `#36 ResolutionSelector` 拉两条线到 `target_width` / `target_height`**
   （输入名与上游原名一致，若你直接换节点，这两条连线会自动保留）。
3. `plan` → `MiniMaxH3ChunkedTwoPassUpscaleT8Advanced.plan`；`cut_report` → 一个 `PreviewAny`。
4. **拖入 `MiniMaxH3HardCutValidate`，插进提示词链路**：
   - 断开 提示词节点 → 条件节点的线
   - `提示词节点` → `校验器.prompt`
   - `校验器.prompt` → 低清条件的 `prompt`（和 高清条件的 `prompt`）
   - `Plan.plan` → `校验器.plan`
   - **无需任何 PreviewAny** —— 报告印在校验器节点自己身上
5. 跑之前看校验器节点上的报告：`status` 必须是 `OK`。
6. （可选）需要自动拼时间戳时再拖入 `MiniMaxH3HardCutShotPrompt`。

**输出名与上游的差异**：`plan` 同名可直连；上游的 `report_json` 对应本插件的 `cut_report`（纯文本报告，不是 JSON）。

---

## 独立计算脚本

`hardcut_math.py` **零依赖**（不需要 torch / ComfyUI），可以单独跑：

```bash
D:\comfyui\comfyenv\python.exe hardcut_math.py 8 4.25 --mp 1.5
D:\comfyui\comfyenv\python.exe hardcut_math.py 15 "4.25,8.5,12.75" --mp 1.5   # 15 秒 4 段
D:\comfyui\comfyenv\python.exe hardcut_math.py 15 --mp 1.5 --step 0           # 全 -1 = 单窗不切
D:\comfyui\comfyenv\python.exe hardcut_math.py 8 4.25 --mp 1.5 --step -2      # chunk 挡位 -2
```

输出含**切点 / 帧号 / 时间码 + chunk 阶梯表 + 可达方案菜单**，用来挑时间戳。

**还能不开 ComfyUI 就体检提示词**（跑的是 `#49` 同一个校验器，退出码 0=通过 / 1=有问题）：

```bash
# 等长（切点用秒）
D:\comfyui\comfyenv\python.exe hardcut_math.py --check prompt.txt --total 8 --cuts 4.25 --mp 1.544

# 不等长（切点用帧，可多个）—— 窗口长度自动按模型里的算法推导，与真实运行一致
D:\comfyui\comfyenv\python.exe hardcut_math.py --check prompt.txt --total 8 --frames 68 --mp 1.544
D:\comfyui\comfyenv\python.exe hardcut_math.py --check prompt.txt --total 8 --frames 68,136 --mp 1.544

# 想手动指定窗口长度（覆盖自动推导）
... --check prompt.txt --total 8 --frames 68 --chunk 136
```

| 参数 | 说明 |
|---|---|
| `--total` | 片长（秒） |
| `--cuts` | **等长**切点，秒，逗号分隔 |
| `--frames` | **不等长**切点，帧，逗号分隔（`68` / `68,136`）；与 `--frames` 同用时以它为准 |
| `--chunk` | 窗口长度（帧），只在不自动推导时用 |
| `--mp` | 二采画布 MP（**建议填真值 1.544**，即 1664×928） |
| `--step` | overlap 帧数，硬切填 0 |

定调句写完先在这里过一遍，比在 ComfyUI 里点运行快得多。

---

## 成片体检：量出「实际转镜帧」（`tools/analyze_cut.py`）

跑完片子想知道**画面到底在第几帧换的**、和切点差几帧，用它：

```bash
D:\comfyui\comfyenv\python.exe toolsnalyze_cut.py "<成片.mp4>" --cuts 4.958,8.500
```

输出：fps / 帧数 / **中位帧差** / 突变点排名 / 每个理论切点对应的实测突变帧与偏移。

```
--- 画面突变点（> 2.5× 中位）---
  帧 109  时间 4.542s  差值 92.93  = 3.6× 中位
--- 理论切点 vs 实测突变 ---
  理论 4.958s (帧 119)  →  最近突变 帧 109 (4.542s)  偏移 -10 帧  幅度 3.6×
  理论 8.500s (帧 204): 附近 30 帧内【没有突变】 — 模型没在这里切镜
```

- **偏移 = 突变帧 − 切点帧** —— 直接填进 `#56` 的 `cut_offset_frames`
- 行"【没有突变】"表示模型**没在那里切**（段内自切常见）
- 加 `--sheet 100,130` 会导出一张逐帧接触表，方便肉眼复核

> 注意：`cut` 是**分割点**，见上文术语表 —— 量出来的突变帧才是**画面真的换镜头**的地方。

---

## 实测负载锚点（RTX 4060 Laptop 8GB）

负载 = **最长段帧数 × 二采画布 MP**。通过线 **180**，爆线 **191**
（2026-09-20 加了质量 LoRA 后重测；裸模型时代是 210 / 236.2）。

| 时长 | 段数 | chunk | 切点 | 最长段 | 1.5MP 下负载 |
|---|---|---|---|---|---|
| 8s | 2 | 102 | 4.250s | 102 | **153.0 ✅** |
| 8s | 2 | 119 | 4.958s | 119 | 178.5 ✅ |
| 8s | 3 | 68 | 2.833 / 5.667s | 68 | 102.0 ✅ |
| 12s | 3 | 102 | 4.250 / 8.500s | 102 | 153.0 ✅ |
| **15s** | **3** | **136** | **5.667 / 11.333s** | 136 | **204.0 ✅** |
| 15s | 4 | 102 | 4.250 / 8.500 / 12.750s | 102 | 153.0 ✅ |

对照：8 秒**单窗**（`full_clip_safe`）在 1.0MP 下就已经是 202.9。
**硬切让你在 1.5MP 下跑 15 秒，比单窗 8 秒还省。**

---

## 已知限制

1. **边界精度 = 单个 token（1~4 帧）**，不是整块 17 帧 —— `_snap_boundary` 可落在任意 token 边
   （合法帧号 `0,1,5,9,13,17,18,22,…`；每 5 个 token 才是一个 17 帧块，那只是**粗档**）。
   但**模型不受网格约束**：它实际转镜常比被告知的时刻**早/晚 1~10 帧**（随内容/种子变）——
   这个偏差只能靠两条路收：**让模型去够边界**（`#56` 的 `prompt_shift_frames`，自动改写提示词时间戳）
   或**把边界挪到转镜所在的 token 上**（`#40` 的 `auto_seam_hunt`）—— 缝与切重合，被切镜盖住。
2. **不等长 = 填不同的 `n`** —— 四个槽的 n 不同，段长自然不等（如 `4 / 10` → 68f / 170f）。
3. **末段 ≥ 2 秒**（`MIN_TAIL_SECONDS`，可在 `hardcut_math.py` 改）。
4. **依赖上游的构建器**：运行时按模块名后缀 `chunked_two_pass_upscale_advanced`
   在 `sys.modules` 里找 `build_chunked_two_pass_low_sigma_plan`。
   找到就用它（保证 plan 契约永远与上游同步）；找不到会回退到内置 plan
   并在 `cut_report` 里打 WARNING。
5. **音频不随视频硬切** —— 本路线 `final_audio_policy = return_exact_first_pass_audio_tensor`，
   最终音频是一采整片透传。想让声音也切，**只能在提示词里写**（见提示词工程文档 §4）。

---

## 文件

| 文件 | 作用 |
|---|---|
| `hardcut_math.py` | 纯计算：帧网格、切点求解、chunk 阶梯、可达方案枚举、负载估算、**提示词校验**。可独立运行（`--check` / `--step`）。 |
| `bridge.py` | 定位并调用上游的 plan 构建器；含 fallback plan。 |
| `h3_upscale.py` | `MiniMaxH3HardCutUpscale` —— 自写分块层，运行时复用上游的放大器 / 采样 / 条件重锚。 |
| `nodes.py` | 规划类节点（Plan / Validate / ShotPrompt / Auto）。 |
| `nodes_repair.py` | 修复类节点（InfoBuffer / SeamBlend / SeamFuse / SeamDissolve / RedrawBridge）。 |
| `nodes_repair_all.py` | `MiniMaxH3SeamRepairAll` —— 全包修复节点。 |
| `tools/` | 命令行工具：成片体检、像素域修复、工作流槽位体检。 |
| `__init__.py` | `comfy_entrypoint`（V3 注册，与上游同机制）。 |
| `MANUAL.md` | **跨模型作业手册**：定切点 → 喂 LLM → 校验 → 跑。 |
| `AUTO.md` | 自动版节点的使用与排查。 |
| `templates/` | 提示词骨架 + 实例 + 常见错误表。 |

---

## 许可

**GPL-3.0-or-later**（`LICENSE` 内为 GPLv3 全文）。

理由不是"随手选的"：本插件在**同一个 Python 进程内** import 并调用上游
[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) 的模块
（`chunked_two_pass_upscale_advanced`、`learned_latent_upscale_advanced`、`sampling`），
并且 plan 的字段契约与它逐字段对齐。上游是 **GPL-3.0-or-later**，这种程度的耦合
按 FSF 对 Python 模块 import 的立场构成衍生作品，因此本仓库沿用同一许可。

配套说明：

- **所有源文件都带 SPDX 头**（`# SPDX-License-Identifier: GPL-3.0-or-later`），
  与根目录 `LICENSE` 全文配套。往仓库提 PR 即视为以同一许可授权。
- 本仓库**不包含**任何上游源码副本。所有上游能力都是运行时按模块名解析调用的；
  `bridge.py` 里的 `T8_H3_CHUNKED_TWO_PASS_PLAN`、`t8.minimax_h3.chunked_two_pass.low_sigma.v3`
  等字符串是**上游的公开契约标识**，必须逐字节保持一致，不属于本项目的原创内容。
- 上游缺失时，plan 构建会回退到 `bridge.py` 里的内置 fallback（会打 WARNING），
  执行器则直接报错 —— **执行器不回退**，因为它必须调用上游的放大器与采样。
- 模型权重不在本仓库内。MiniMax H3 及其衍生模型遵循 **MiniMax H3 Community License
  Agreement**，请自行阅读你所用模型的完整协议与 Acceptable Use Policy。
- 若你希望以更宽松的许可（MIT / Apache-2.0）发布，唯一干净的做法是**断开 import**：
  只通过 ComfyUI 连线传递数据、不调用上游 Python 函数。代价是失去不等长分段执行器
  与放大器复用，需要自己实现整套 H3 二采链路 —— 通常不划算。

## 致谢

- **T8mars / comfyui-minimax-h3-audio-T8** —— 分块二采、3D latent 放大器、音频条件与
  整套 H3 节点生态都来自这个包。本插件只是站在它上面改了"分段"这一层。
- **MiniMax / Comfy-Org** —— H3 模型与 ComfyUI 集成。
- 所有实测数字都来自本机 RTX 4060 Laptop 8GB + torch 2.14.0+cu130 的真实跑片，
  换卡请重新量锚点。
