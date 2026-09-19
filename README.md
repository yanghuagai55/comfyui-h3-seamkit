# comfyui-h3-hardcut

**把分块二采的"接缝"变成剪辑点。**

| 文档 | 用途 |
|---|---|
| **`MANUAL.md`** | ★ **作业手册** —— 跨模型协作全流程（定切点 → 喂 LLM → 抄回 ComfyUI）+ 可复制的 LLM 约束块 |
| `templates/hardcut-prompt-template.md` | 提示词模板（六段骨架 + 硬切改写实例 + 常见错误） |
| `refs\H3-硬切分镜提示词工程.md` | 原理与一手依据（源码层面为什么 overlap=0 就是硬切） |

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
| **切点 / cut** | **执行器的分割点**：把 latent 切成两块分别放大的地方 | **插件**（`cut_frames` / `chunk_step`，只落在 17 帧网格上）| `cut points : 4.958s (frame 119)` |
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

这个插件给你**两种指定"切在哪"的方式**，一种用秒，一种用帧：

| 参数 | 写法 | 含义 | 什么时候用 |
|---|---|---|---|
| `cut_1` ~ `cut_4` | `cut_1 = 4.25` | 在**第 4.25 秒**切开 | 你心里有"几秒几秒"的概念 |
| `cut_frames` | `"68"` | 在**第 68 帧**切开 | 你想要**不等长**的分块（后面细讲） |

### 那 `cut_frames="68"` 到底是什么？

**它告诉你：在第 68 帧的那个点切一刀。**

结合"1 秒 = 24 帧"来换算：`68 ÷ 24 ≈ 2.83 秒`，也就是大约在视频的 **2.8 秒**处切开。

举个实在的例子 —— 一个 **8 秒**的视频（8 秒 = 192 帧）：

```
cut_frames = "68"
  ↓
整条视频  ┌─────────────┬──────────────────────┐
              [0——68 帧]     [68——192 帧]
              第 1 段         第 2 段
              ≈2.8 秒          ≈5.2 秒
```

- 第 1 段：从开头到第 68 帧（大约 2.8 秒）
- 第 2 段：从第 68 帧到最后（剩下约 5.2 秒）
- **两段长短不一样**（68 帧和 124 帧）——这就是插件标题里说的"**不等长分块**"

> ✅ 如果你填 `cut_frames = "68,136"`，就是切两刀，切成三段：68 + 68 + 56 帧。
> 想切几刀就写几个数字，用英文逗号隔开。

### 一个小提醒：帧数会自动"取整"

上面说了切点必须是 **17 的倍数**（这是底层模型的硬性限制，17 帧 = 0.708 秒）。
你填 `70`，插件会悄悄帮你改成最近的合法值 `68`。
所以报告里的实际切点，可能和你手填的数差几帧——**以报告（cut_report / 校验器）为准就行**。

### 怎么选用哪一个？

- 想要**每段一样长**（省事、好安排）→ 用 `cut_1`~`cut_4` 的**秒**，或干脆留空用默认切法。
- 想要**两段长短故意不一样**（比如第一段慢镜头 3 秒、第二段动作戏 5 秒）
  → 用 `cut_frames` 填**帧**。

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
| `cut_1` ~ `cut_4` | 切点秒数。**`-1` = 这里不切**（8 秒单切点就是 `cut_1=4.25`、其余 `-1`）。只有正数才算一个切点。 |
| `cut_frames` | **显式切帧（帧，逗号分隔，如 `68` 或 `68,136`）**。非空时覆盖 `cut_1`~`cut_4`，产出**不等长窗口**——第一段 68 帧、下一段 124 帧都行。每帧自动 snap 到 17 帧网格（70→68）。留空 = 等长窗口。 |
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

切点只能落在 **17 帧的整数倍**（= 0.708 秒的倍数），所以"想切在整数秒"常常差几帧。**先看菜单再定时间戳。**

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
要不等长就在 `#48` 的 `cut_frames` 填帧（如 `68`）。

**`cut_frames` 的四道校验**（填错立刻报错，不会静默切出跑不了的窗口）：

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

负载 = **最长段帧数 × 二采画布 MP**。通过线 210，爆线 236.2（均为本机实测）。

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
2. **不等长必须用 `cut_frames`** —— 靠 `cut_1`~`cut_4` 的秒数是**等长**分块（末段可短）；
   想要每段长短故意不同（如 3s / 9s / 15s），就在 `cut_frames` 里填**帧**号（如 `"68"`）。
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
| `nodes.py` | 三个 ComfyUI 节点（Plan / Validate / ShotPrompt）。 |
| `__init__.py` | `comfy_entrypoint`（V3 注册，与上游同机制）。 |
| `MANUAL.md` | **跨模型作业手册**：定切点 → 喂 LLM → 校验 → 跑。 |
| `templates/hardcut-prompt-template.md` | 提示词骨架 + 实例 + 常见错误表。 |
