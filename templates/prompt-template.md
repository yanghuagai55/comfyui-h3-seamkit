# H3 硬切提示词模板（合并版 · v1.3.0）

> 合并自 `hardcut-prompt-template.md` + `auto-prompt-template.md` + `anchor-frame-template.md`（2026-09-26，内容无删减只去重）。
> 母本：`refs\H3-R2V提示词模板-官方.md`（MiniMax 官方 R2V 指南）· 原理：`refs\H3-硬切分镜提示词工程.md`
>
> ★ **本版新增铁律：blocking 显式化**（站位/朝向必须写死并声明全片统一）——
> 实测教训：站位语义模糊（"facing slightly toward each other"）会让两个采样窗口
> 各自解读（一个侧身相对、一个正面朝镜），锚定接缝处出现用户可见的姿势跳变
> （exp_4v10a_00087 @85/187）。详见第二部分「硬规则清单」第 6 条。

---

## 0. 三条路线选路表（先选路，再往下读对应章节）

| 路线 | 节点 | 切点谁说了算 | 何时用 |
|---|---|---|---|
| **A. 手动硬切** | `MiniMaxH3HardCutPlan` + `ShotPrompt`（`cut_1~4` 手填） | 你（逐刀填帧） | 切点你说了算；不等长分段 |
| **B. 自动硬切** | `MiniMaxH3HardCutAuto`（只填 `chunk_step` 档位） | 执行器（按档位切） | 让执行器按档位切，省心 |
| **C. 锚定帧** | `MiniMaxH3AddGuide`（ComfyUI 0.34 原生） | **帧级硬约束** | 切点必须帧级精确，模型必须照画 |

> A/B 生成**六段提示词**（本文第一、二部分共用）；C 生成**锚定帧画面描述**（第五部分），
> C 可与 A/B **叠加**（六段定内容 + AddGuide 把切点钉死）。

---

# 第一部分 · 通用六段骨架（A/B 共用）

```text
subject_definitions:
<Subject 1> is <角色/物体> in <Picture 1>, <外形特征>.
<Subject 2> is <场景> in <Picture 2>, <特征>.

summary:
[reference generation] The target video is a <N>-second <画幅> clip with native stereo
sound, executed as a <M>-shot sequence with a hard cut at <MM:SS.mmm> [, and a second hard
cut at <MM:SS.mmm>], fusing <A> into <B>: <一句话动作/情节总纲>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_copy - <要保持不变的具体特征>.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_copy - <环境/光线/构图>.

detailed_description:
<一两句定调：风格/画质/色调 —— 必须放在 [Shot 1] 之前>.
[Shot 1] <构图 + 主体位置与动作 + 环境光线 + 镜头运动 + 声音>, building to the cut at
<MM:SS.mmm>.
[Shot 2] At <MM:SS.mmm>, <新机位/新景别>, <同上六要素>.

overall_soundscape:
Throughout the video, <环境音 + 物理音效>. The sound bed carries across the cut without
interruption — the video cuts, the ambience does not.

non_diegetic_music:
<配乐描述，或 N/A>
```

---

# 第二部分 · 硬规则清单（A/B 共用，逐条过）

## 1. 硬切版必须改的 5 处

| # | 位置 | 怎么改 |
|---|---|---|
| 1 | `summary` | **删掉任何"一镜到底/无剪辑"的说法**，改成 `a N-shot sequence with a hard cut at MM:SS.mmm`。**这是最容易漏的一处** —— 留着它，模型会抵抗你。 |
| 2 | `retention_analysis` | 每项的 `(appears in ...)` **列出全部 `[Shot N]`**。跨切点必须写全，否则模型不知道切完还得保持长相。 |
| 3 | `detailed_description` | `[Shot 1]` 不带时间戳；`[Shot 2]` 起写 `At MM:SS.mmm, `，**时间戳 = 切点秒数**。 |
| 4 | 切点的镜头语言 | `[Shot 2]` **必须明确换机位/景别/角度**，写清是"切"（`the shot cuts to ...`）。 |
| 5 | 音效 | `overall_soundscape` 加一句"跨切点连续"（本路线音频整片透传，不随视频切）。 |

## 2. 三条时间线规则

**① 切点记法**：`00:04.250`（分:秒.毫秒三位），**秒数必须等于节点的 `cut_seconds`**。

**② 切点推荐 17 帧的整数倍**（`0.708s × k` = 独占帧，模型能放最干净的转镜）。更细的 **token 边界**（1-4 帧粒度，如 115 帧 = 00:04.792）也合法——"提前切"就靠它。17 倍数档由 `hardcut_math.py` 枚举：

```bash
D:\comfyui\comfyenv\python.exe ...\comfyui-h3-seamkit\hardcut_math.py 8 4.25 --mp 1.5
```

**③ 动作要"跨切点连续，机位要断"**

| 做法 | 说明 |
|---|---|
| ✅ **动作时间线连续** | 0→15 秒是一条完整的动作线，不重复不跳跃 |
| ✅ **机位在切点明确改变** | 中景 → 过肩近景 / 侧跟 → 仰角 |
| ❌ **不要在切点重新开始动作** | 会让观感像"回放" |
| ❌ **不要在连续动作的最激烈处切** | 尽量落在节拍上（一击结束、转身瞬间） |

## 3. ★ blocking 显式化（v1.3.0 新增铁律）

**人物站位/朝向/视线必须在提示词里写死，并声明全片统一。**

```text
✅ 写法示例（放 detailed_description 开头或角色动作段）：
"Both girls stand side by side, BOTH facing the camera directly, shoulders square
 to the lens, eyes to the viewer — this exact blocking (both girls front-facing,
 side by side, neither turned in profile) is kept in every single frame of the
 video, in the wild moves and in the locked windows alike."

❌ 反例（exp_4v10a_00087 实测翻车）：
"facing slightly toward each other"   ← 语义模糊
```

**为什么**：两个采样窗口是**独立渲染**的。站位语义模糊时，窗口 A 解读成"侧身相对"、
窗口 B 解读成"正面朝镜"——锚定接缝（冻结前缀 | 新鲜渲染）处姿势跳变，**看起来像硬切**
（exp_4v10a_00087 @85/187 实测）。把 blocking 写死 = 两窗只能渲染同一站位 = 接缝自然连续。

**同族规则**：凡是"跨窗口必须一致"的视觉语义（站位、朝向、视线、与镜头的相对关系），
都写死 + 声明全片统一。模糊词（slightly / somewhat / 之类）都是接缝跳变的种子。

## 4. 相邻镜头必须换机位/景别（两镜画面太像，切点会像"崩了"）

---

# 第三部分 · 路线 A：手动硬切专用

## 3.1 `cut_1~cut_4` 与不等长分段

| 项 | 写法 |
|---|---|
| `cut_1`~`cut_4` | 填 **n（17 帧块数）**：`4` → 68 帧；`4 / 10` → 68f / 170f。`-1` = 不切 |
| 帧 → 时间戳 | `帧 / 24`，三位毫秒。`68 帧 = 00:02.833`，`170 帧 = 00:07.083` |
| 镜头数 | = 段数 = 切点数 + 1。`4` → 2 镜；`4 / 10` → 3 镜 |
| 永远合法 | n × 17 天生落在网格上，不需要 snap |

> 例：8 秒切 `68` → `[Shot 1]` 覆盖 0→68 帧（0→2.833s），`[Shot 2] At 00:02.833` 覆盖 68→192 帧。
> 第一段短、第二段长，完全合法 —— 这就是「不等长」。

## 3.2 想换切点？改这 4 处

1. `MiniMaxH3HardCutPlan` 的 `cut_1`~`cut_4`（用不上的槽填 -1）
2. `summary` 里的时间戳
3. `[Shot 2]` 开头的 `At MM:SS.mmm, `
4. `overall_soundscape` 末尾那句的时间戳

**第 2–4 处可以直接抄节点 `cut_report` 里的 `cut points` 行**，或者把 `cut_seconds` 连到
`MiniMaxH3HardCutShotPrompt` 让它自动生成 `[Shot N]` 行（推荐）。

---

# 第四部分 · 路线 B：自动版专用

## 4.1 LLM 输入格式（喂 LLM 就复制这一份）

```text
片长：10 秒
目标镜头数：2            ← 你要几个镜头就写几（决定段数，也决定跑起来的开销）
chunk_step：8            ← 必须与镜头数匹配（查 4.2 的表）
剧情要求：<你的故事>
```

## 4.2 `chunk_step` 必须与目标镜头数匹配

**段数是 `chunk_step` 的函数，不是自由变量。** 节点从 n = 1 开始试，
第一个「合法」的 n 就是结果（合法 = 每段 ≤ `chunk_step×17` 帧、尾段 ≥ 17 帧、最长段×画布MP ≤ 180）。
查表（二采 1.544 MP）：

| 片长 | 2 镜 | 3 镜 | 4 镜 | 5 镜 | 6 镜 |
|---|---|---|---|---|---|
| 8s | 6 ~ 20 | 4 ~ 5 | — | 3 | — |
| 10s | **8 ~ 20** | **5 ~ 7** | 4 | — | 3 |
| 12s | — | **6 ~ 20** | 5 | 4 | — |
| 15s | — | 8 ~ 20 | 6 ~ 7 | 5 | 4 |

**读法**：10 秒片要 2 镜 → `chunk_step` 填 **8 ~ 20**；要 3 镜 → 填 **5 ~ 7**。

- 目标镜头数落在表里 → 直接往下写
- **填的档位会切出别的镜头数** → 输出一行提醒：
  `「要 N 镜，#56 的 chunk_step 请改成 X~Y」`，并**按 N 镜照写**（用户改完参数就能跑）

> 表里没有的片长：用 4.3 的公式试 n = 1, 2, 3…，取第一个满足全部条件的。

## 4.3 切点公式（节点用同一套算法，你算的必须和它一致）

```text
① 总帧数   F = round(秒 × 24) + (5 − round(秒 × 24) % 17) % 17
② 每段上限 C = chunk_step × 17                        （帧）
③ 段数     n = 1, 2, 3… 试，取第一个满足全部四条的：
               · 按下面公式切出的段数 = n
               · 每段 ≤ C
               · 尾段 ≥ 17 帧
               · 最长段 × 1.544 ≤ 180        （1.544 = 二采画布 MP）
④ 切点帧   第 i 个 = round( round(i × F / n) / 17 ) × 17      i = 1 … n−1
⑤ 时间戳   切点帧 ÷ 24 → MM:SS.mmm（三位小数）
```

例（片长 10 秒 / `chunk_step` 5）：`F = 243`，`C = 85`，`n = 3`，
切点帧 `85 / 170` → **`00:03.542` / `00:07.083`**。

> **帧 → 秒：除以 24**（85 → 3.542s，170 → 7.083s）。**17 帧 = 0.708s**，切点永远是它的整数倍。
> 算不准就以 `#56` 节点输出的骨架为准 —— 那是同一套算法算的，100% 一致。

## 4.4 镜头数由你指定，剧情在框内安排

**镜头数 ≥ 段数**（不是相等）：

- **执行器每切一刀 = 一个硬断点**（`overlap = 0`，段之间没有上下文）→ 所以**每一刀都必须出现在你的时间戳里**
- **反过来说得通**：提示词里的时间戳**可以比执行器的刀更多** —— 那些多出来的时间戳是**同一段内模型自己切的镜头**（和一采整片生成多镜是一个机制）
- 所以「5 个镜头 + 2 段」完全合法：段边界是 2 个硬切，段内还有 3 次由模型完成的切换

> **先读剧情，然后把它讲圆在你指定的 N 个镜头里。**
> N 个镜头塞不下 → **不要偷偷加镜头**，输出一行提醒：
> `「这段剧情 N 个镜头偏挤，建议 N+1 镜（chunk_step 改 X）」`，然后按 N 镜先写一版。

**为什么镜头数由你定而不是"剧情需要多少就多少"**：镜头数直接决定开销与风险 —

| | 镜头少（段少） | 镜头多（段多） |
|---|---|---|
| 总计算量 | 一样（总帧数不变） | 一样 |
| 采样调用次数 | **少**（每段一次，含固定开销） | 多 |
| 剪辑点（接缝） | **少**（画面更连贯） | 多 |
| 单段显存压力 | 高（段更长） | **低** |

所以要**在显存扛得住的前提下，能用少镜头就别用多镜头**；只有当最长段 × 画布MP 逼近 180 时才必须多切。

## 4.5 自动版 chunk_step → 最大秒数

| chunk_step | 最大帧数 | 最大秒数 |
|---|---|---|
| 6 | 102 | 4.250s |
| 7 | 119 | 4.958s |
| 8 | 136 | 5.667s |
| 11 | 187 | 7.792s |

写 summary 时套一句：`... executed as a N-shot sequence with hard cut(s) at ..., with every segment no longer than <最大秒数> seconds.`

## 4.6 输出硬规则（B 路线追加）

- **不留任何占位符** —— `{{CUT_1}}` 这种不许出现；最终答案里不能出现任何尖括号
- **不要把机制写进提示词** —— 时间戳算准即可，不要写补偿说明（权重稀释）
- 不要把 `[Shot N]` 之外的时间戳当作切点声明（窗口内动作节拍随意写，节点不动它们）

---

# 第五部分 · 路线 C：锚定帧（帧级硬约束）

> 原理：提示词里的时间戳只能"请求"模型在某帧切镜——实测换模型后会偏 5~7 帧。
> `MiniMaxH3AddGuide` 把一张图**锚**在第 N 帧 → keyframe condition latent，
> **每一步重新注入、永不去噪** —— 模型必须照它画。切点准不准从"模型的意愿"变成"你的硬约束"。

## 5.1 你要给 LLM 的输入

```text
片长：15 秒
切点帧号：85, 187, 272        ← 从 #56 的报告里抄（planned_cuts 那一行），不要自己算
剧情要求：<你的故事；如果已经有六段提示词，直接贴六段>
角色定义：<和提示词里的 subject_definitions 保持一致，逐字>
```

> 帧号必须是 **17 的倍数**。拿不准就先跑
> `python hardcut_math.py 15 "4.25,8.5,12.75" --mp 1.5`，报告里的 `cut frames` 直接抄。

## 5.2 给 LLM 的指令（整段复制）

```text
你现在是分镜师 + 图像提示词工程师。

任务：为下面这个片子写"锚定帧画面描述"。这些描述会拿去图像模型生成单帧图片，
再用 MiniMaxH3AddGuide 锚进视频的指定帧——所以每一张都必须是那个镜头的第一帧。

硬性规则：
1. 每个切点写一张（个数 = 给的帧号个数）。若给了 0 号帧需求，再加一张。
2. 只描述"这一帧的静止画面"。禁止动作过程、禁止时间词（然后/接着/逐渐/开始/继续）。
3. 每张必须写全六要素：机位角度 · 景别 · 构图 · 人物姿态与表情 · 光线 · 环境细节。
4. 必须与该镜头开头一致——它就是这个镜头的第一帧，不是中间帧。
5. 与上一镜至少有一项大改（机位或景别），否则切点在成片里看不出来。
6. 角色外观描述与「角色定义」逐字一致，全片不得漂移。
7. 40~80 词，逗号分隔的短语堆叠（图像模型友好），不要完整句、不要连词。
8. 不出现元信息："第 N 镜"、"cut"、"hard cut"、"时间戳"。

输出严格 JSON，不要解释：

{
  "clip_seconds": <片长>,
  "cuts": [<帧号...>],
  "anchors": [
    {"frame_idx": <帧号>, "shot": "[Shot 2]", "image_prompt": "<静态画面描述>"},
    ...
  ]
}
```

## 5.3 画面描述对照表

| 要素 | ✗ 太虚 | ✓ 可出图 |
|---|---|---|
| 机位 | 看向她 | 低角度仰拍，机位在她膝盖高度 |
| 景别 | 近景 | 过肩中景，焦点在肩线前方，背景虚化 |
| 构图 | — | 人物偏左三分线，右侧留出天空与落花 |
| 姿态表情 | 她跑着 | 身体前倾，双臂摆到最高点，嘴微张，眉头紧 |
| 光线 | 白天 | 侧逆光，雨后湿地面反光，发梢有暖色轮廓光 |
| 环境 | 校园 | 樱花树下柏油小径，落花贴地，远处教室玻璃反光 |

**一句话判据**：把描述丢给图像模型，**不看上下文也能画出正确的一帧**，才算合格。

## 5.4 拿到描述之后

1. **出图**：每张 `image_prompt` 丢给图像模型（Krea2 / 你的图像链），一帧一张，按画布比例（16:9）
2. **接线**：`条件节点 ──positive──► [AddGuide frame_idx=85] ──► [AddGuide 187] ──► … ──► guider`（每个挂 vae / latent / image）
3. **校验**：`frame_idx` 必须**逐字等于** `#56` 报告里的 `planned_cuts`
4. **二采**：想让二采也遵守锚点，高分辨率条件链上也要串一遍

## 5.5 检查清单

- [ ] 帧号与 `#56` 报告的 `planned_cuts` **完全一致**（抄，不要自己算）
- [ ] 每张描述都是**静止画面**（无动作词、无时间词）
- [ ] 相邻两张的**机位或景别差异明显**
- [ ] 角色外观描述**全片统一**（与 subject_definitions 逐字一致）
- [ ] 每张描述**不看上下文也能出图**
- [ ] 张数 = 切点数（如需锚首帧再 +1）

---

# 第六部分 · 实测完整示例

## 6.1 路线 A：雷姆 · 校园武戏（8 秒 / 硬切 4.250s · 等长 2 镜）

> 配套节点参数：`total_seconds=8` · `cut_1=4.25 / cut_2~4=-1` · `chunk_step=0` · `canvas_megapixels=1.5`
> 执行器切点：**帧 102 = 4.250s**（偏差 0 帧）· 负载 153.0 ✅
> 要不等长（第一段短）：`cut_1 = 4`（68 帧）→ `[Shot 1]` 0→2.833s。

**为什么切 4.25s**：落在"起手 → 连击"的换挡处，动作不断，机位能从**侧向中景**干净换成**过肩近景**。避开 3.2–5.5 的连击最密处（那里切会像跳帧）。

```text
subject_definitions:
<Subject 1> is Rem, the anime girl in <Picture 1>: light-blue short hair with bangs covering her right eye, blue pupils, a white flower headband, a purple butterfly-ribbon hair ornament, the classic black-and-white maid outfit with a white apron front, white over-knee stockings, and black mary-jane shoes; she is rendered as a two-dimensional, cel-shaded anime character with clean lines, crisp color fields, and high-frame-rate action animation quality.
<Subject 2> is the real-world campus park path in <Picture 2>: a curved red-brick walking path, lush green trees, a mown lawn, trimmed shrub clusters, dappled bright sunlight, and a distant campus building, in photorealistic live-action quality under a clear daytime sky.
<Subject 3> is a mysterious shadow enemy: a faceless, blurred dark humanoid silhouette with soft smoky edges, kept slightly out of focus and always subordinate in framing; it never reveals facial features, never holds the foreground longer than Rem, and serves purely as her combat opponent.

summary:
[reference generation] The target video is an 8-second, 16:9, 2K clip with native stereo sound, executed as a two-shot sequence with a hard cut at 00:04.250, fusing the 2D anime heroine <Subject 1> into the photorealistic campus park path of <Subject 2>: Rem sinks into a combat stance, dashes forward with sparks scraping off the bricks, guards with her maid sleeve and sidesteps the strikes of the shadow enemy <Subject 3>, then continues into a rapid chain of kicks and knife-hand strikes that ends with the enemy knocked flying out of frame, finishing on a stable battle-end pose. Both shots share one continuous action line: the cut only changes the camera, never the fight.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_copy - her blue short hair with bangs over the right eye, blue pupils, white flower headband, purple butterfly ribbon, black-and-white maid outfit, white apron, white over-knee stockings, and black shoes stay identical in every frame of the high-speed combat across both shots; her proportions remain stable, her face never distorts, her limbs connect naturally, the outfit never clips through her body, and there is no frame flicker.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_copy - the curved red-brick path, green trees, lawn, shrubs, dappled sunlight, and distant campus building keep the same real structure, light direction, and color temperature for the entire video across both shots, remaining a stable background through the whole battle.
<Subject 3> (appears in [Shot 1], [Shot 2]): partially_preserved - stays a faceless blurred dark silhouette throughout, always out of focus and never taking the frame from Rem.

detailed_description:
The target video is a high-frame-rate Japanese action-animation scene that fuses a two-dimensional, cel-shaded anime heroine into a photorealistic real-world campus park, under strong midday sunlight where bright daylight contrasts with hard combat shadows; the palette is vivid and clean, the linework crisp, the motion fast but always readable, and the frame carries no text and no watermark from start to finish.
[Shot 1] From 00:00.000 the take opens on <Subject 1>, Rem, framed at medium shot in the middle distance on the curved red-brick path of <Subject 2>, facing the faceless blurred silhouette of <Subject 3>. Her body sinks low, weight coiling like a spring, and her eyes snap from calm to razor focus. At 00:01.200 she explodes into a forward dash, her black shoe soles scraping the bricks and throwing off faint sparks, while the camera drops to a low angle and races alongside her charge. At 00:02.200 comes the first exchange: she raises her maid sleeve to guard her forearm, absorbing a dark strike with a muffled impact, then slips sideways past a sweeping blow, her white apron and blue short hair whipped up by the wind of it. At 00:03.400 she plants her weight and pivots into her first kick of the counterattack, the camera beginning to close in on her, and the shot holds on this accelerating motion right up to the cut at 00:04.250.
[Shot 2] At 00:04.250, the shot cuts to a tight over-the-shoulder angle behind Rem, close and physical, the camera now riding low with her as the counterattack continues without the slightest pause: a rapid chain of kicks and knife-hand strikes — every motion short, clean, and decisive — streaked with brief speed lines and ghost afterimages, each impact popping a small transparent shockwave; the shadow enemy reels under the barrage, always blurred and off-focus, never stealing the frame from her. At 00:05.500 she plants her foot and whips through a half-spin, and the final blow hurls <Subject 3> flying out of frame, scattering a flurry of green leaves and a low roll of dust across the bricks. At 00:06.500 she sticks the landing in a stable battle-end pose, and the camera pulls back slightly and then eases in toward her face, holding the last stretch on her steady, focused expression amid drifting leaves — the rhythm fast but clear, her identity, outfit, and proportions intact across the cut, and the real park unchanged behind her to the last frame.

overall_soundscape:
Throughout the video: her shoe soles scraping the red bricks with short friction sparks at the dash; one muffled impact as the sleeve block absorbs the dark strike; sharp, clean air-whooshes following each kick and knife-hand; her skirt and apron fabric snapping in the wind of the fight; a small concave pop at each shockwave; one heavier thud with a rising whoosh as the shadow enemy is flung off-screen; green leaves fluttering down and dust settling afterward; a few startled birds; faint steady campus ambience running beneath it all. The ambience and the fight sounds carry straight across the cut at 00:04.250 without a break — the picture changes angle, the sound does not restart.

non_diegetic_music:
N/A
```

## 6.2 路线 B：10 秒 / `chunk_step` 5 / **3 镜**

> **实测跑通**（节点校验 `status OK`，0 错 0 警），切点 `00:03.542` + `00:07.083`。

```text
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.
<Subject 2> is the tree-lined campus courtyard in <Picture 2>, late afternoon, warm low sunlight and long shadows.

summary:
[reference generation] The target video is a 10.125-second clip with native stereo sound, executed as a three-shot sequence with hard cuts at 00:03.542 and 00:07.083, with every segment no longer than 3.542 seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_copy - face, hair, cardigan and necklace stay identical across both cuts; only the camera distance changes.
<Subject 2> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_copy - the same courtyard and the same low afternoon light continue across both cuts.

detailed_description:
The target video is in a cinematic, softly lit style with a slightly desaturated palette and shallow depth of field.
[Shot 1] A low-angle tracking shot follows the woman as she breaks into a run along the courtyard path, her cardigan lifting in the wind; dust and petals drift through the low sunlight. Her footsteps and breath are close and dry, cicadas faint in the distance.
[Shot 2] At 00:03.542, the shot cuts to a tight over-the-shoulder angle that rides just behind her, holding her stride and speed without a break; a bench and two students flash past in the foreground as the sunlight flickers through the leaves.
[Shot 3] At 00:07.083, the shot cuts to a close-up of her face from the front, still running, hair whipping across her cheek; her eyes narrow with effort, breath loud, the blurred courtyard streaming past behind her.

overall_soundscape:
Faint steady campus ambience with cicadas and a light breeze in the leaves runs beneath it all; footsteps, breath and the rustle of fabric stay continuous across the cuts at 00:03.542 and 00:07.083, with nothing duplicated or dropped at either join.

non_diegetic_music:
N/A.
```

**三段情绪正好落在三个镜头上**：起跑（急促）→ 中段坚持 → 特写收束。

### 6.2.1 变体：3 个镜头 / 2 个采样窗口（完全合法，已实测）

> 10 秒片，`chunk_step = 8` → 执行器只切 1 刀（4.958s）= 2 个窗口；提示词写 3 个镜头
> （`[Shot 3] At 00:08.500` 在窗口 2 内部，模型自己切）。真跑过：`exp_4v10a_00038.mp4`，节点 0 错 0 警。
> **何时用**：剧情要 3 镜但不想多付一次窗口开销。**何时别用**：窗口内切换是模型自由发挥；
> 要确定地切 3 段 → 查 4.2 的表调 `chunk_step`。

## 6.3 路线 C：锚定帧 JSON 示例（15 秒 / 3 切点）

**输入**：切点帧号 `85, 187, 272`；剧情 = 雷姆与高木雨后校园小径并排走，中途雷姆停下回头看镜头，最后切到高木极近特写。

```json
{
  "clip_seconds": 15,
  "cuts": [85, 187, 272],
  "anchors": [
    {
      "frame_idx": 85,
      "shot": "[Shot 2]",
      "image_prompt": "frontal medium two-shot at chest height, both girls fully in frame side by side, Rem in her blue maid outfit on the left, Takagi in school uniform on the right, bodies turned slightly toward each other, walking paused, wet asphalt path with fallen petals, overcast sky, soft even light, blurred sakura branches behind"
    },
    {
      "frame_idx": 187,
      "shot": "[Shot 3]",
      "image_prompt": "low-angle over-the-shoulder from behind Takagi, Rem in the left third stopping mid-step and turning her head back toward camera, sky-blue hair swinging, blue maid outfit, mouth slightly open, rain-wet path reflecting grey sky, shallow focus on her face, cold rim light along her hair edge"
    },
    {
      "frame_idx": 272,
      "shot": "[Shot 4]",
      "image_prompt": "extreme close-up of Takagi's face filling the frame, brown hair strands crossing her cheek, eyes wide and unblinking looking slightly off-camera, lips pressed, dark blurred background of wet branches, single soft side light, fine skin texture with raindrops"
    }
  ]
}
```

注意三张的共同点：**机位各不相同**（中景侧向 → 过肩仰角 → 极近特写）、**都是静止画面**、**每张都能独立成图** ✓

---

# 第七部分 · 节点校验与常见错误

## 7.1 节点会怎么检验（错一条就拦停）

| 检查 | 说明 |
|---|---|
| 六段齐全、`[Shot N]` 从 1 连续 | 缺段或跳号 → 错 |
| **每一刀都在提示词里** | 执行器切在 3.542s 但提示词没写那个时间戳 → 错。**提示词多出时间戳（段内模型自己切镜）不算错** |
| **每个时间戳落在真会切的那一帧** | 差 1 帧就报错，并告诉你正确值 |
| **没有 `no cuts` / `continuous take` / `unbroken`** | 与硬切天然冲突 → 错 |
| 尾段 ≥ 17 帧、每段 ≤ `chunk_step × 17` | 超了 → 错 |
| 负载（最长段 × 画布MP）≤ 180 | 超了 → 错 |
| 段数 ≤ 10 | — |

改完提示词先别急着跑：`#56` 节点上印的报告第一行 `status` 是 **OK** 才算过。

## 7.2 常见错误

| 现象 | 原因 |
|---|---|
| 切点处画面"回放"了一下 | `[Shot 2]` 把动作重新开始了一遍 —— 动作线必须连续 |
| 切点处观感像崩了而不是剪辑 | `[Shot 2]` 没换机位，两镜看起来几乎一样 |
| 模型抵抗切点、画面拖着不切 | `summary` 里还留着"一镜到底/无剪辑"的表述 |
| 切完角色长相变了 | `retention_analysis` 没列出切点后的 `[Shot N]` |
| 声音也在切点处断了 | 正常 —— 本路线音频整片透传。若不想断，检查提示词里有没有写"换环境音" |
| 切点位置和提示词差半秒 | `cut_seconds` 没连过去，或手抄错了时间戳 |
| **锚定接缝处人物姿势/朝向跳变（像硬切）** | **blocking 没写死** —— 见第二部分第 3 条，站位/朝向/视线必须显式 + 全片统一（v1.3.0 新增） |
