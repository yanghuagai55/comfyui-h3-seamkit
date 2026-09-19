# H3 硬切提示词模板（comfyui-h3-hardcut 专用）

> 配合节点 `MiniMaxH3HardCutPlan` + `MiniMaxH3HardCutShotPrompt`
> 母本：`refs\H3-R2V提示词模板-官方.md`（MiniMax 官方 R2V 指南）
> 原理说明：`refs\H3-硬切分镜提示词工程.md`

---

# 第一部分 · 通用骨架

```text
subject_definitions:
<Subject 1> is <角色/物体> in <Picture 1>, <外形特征>.
<Subject 2> is <场景> in <Picture 2>, <特征>.

summary:
[reference generation] The target video is a <N>-second <画幅> clip with native stereo
sound, executed as a <M>-shot sequence with a hard cut at <MM:SS.mmm> [, and a second hard
cut at <MM:SS.mmm>], fusing <A> into <B>: <一句话动作/情节总纲>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <要保持不变的具体特征>.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - <环境/光线/构图>.

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

## 硬切版必须改的 5 处

| # | 位置 | 怎么改 |
|---|---|---|
| 1 | `summary` | **删掉任何"一镜到底/无剪辑"的说法**，改成 `a N-shot sequence with a hard cut at MM:SS.mmm`。**这是最容易漏的一处** —— 留着它，模型会抵抗你。 |
| 2 | `retention_analysis` | 每项的 `(appears in ...)` **列出全部 `[Shot N]`**。跨切点必须写全，否则模型不知道切完还得保持长相。 |
| 3 | `detailed_description` | `[Shot 1]` 不带时间戳；`[Shot 2]` 起写 `At MM:SS.mmm, `，**时间戳 = 节点的 `cut_seconds`**。 |
| 4 | 切点的镜头语言 | `[Shot 2]` **必须明确换机位/景别/角度**，写清是"切"（`the shot cuts to ...`）。 |
| 5 | 音效 | `overall_soundscape` 加一句"跨切点连续"（本路线音频整片透传，不随视频切）。 |

## 三条时间线规则

**① 切点记法**：`00:04.250`（分:秒.毫秒三位），**秒数必须等于插件的 `cut_seconds`**。

**② 切点必须是 17 帧的整数倍**（`0.708s × k`）。可达切点由 `hardcut_math.py` 枚举：

```bash
D:\comfyui\comfyenv\python.exe ...\comfyui-h3-hardcut\hardcut_math.py 8 4.25 --mp 1.5
```

**②b 不等长分段（按帧切点）** —— `#48` 的 `cut_frames` 填帧，段长可以不相等：

| 项 | 写法 |
|---|---|
| `cut_frames` | 逗号分隔帧，如 `68`（两段）或 `68,136`（三段）。留空 = 等长（走 `cut_1`~`cut_4`） |
| 帧 → 时间戳 | `帧 / 24`，三位毫秒。`68 帧 = 00:02.833`，`136 帧 = 00:05.667` |
| 镜头数 | = 段数 = 切帧数 + 1。`cut_frames=68` → 2 镜；`68,136` → 3 镜 |
| 切帧也要 17 倍数 | 70 会被 snap 到 68；以 `cut_report` 里的 `cut points`（帧）为准 |

> 例：8 秒切 `68` → `[Shot 1]` 覆盖 0→68 帧（0→2.833s），`[Shot 2] At 00:02.833` 覆盖 68→192 帧。
> 第一段短、第二段长，完全合法 —— 这就是「不等长」。

**②c 自动版（`MiniMaxH3HardCutAuto`）的 chunk_step → 最大秒数**

自动版节点只填 `chunk_step`（档位），每段最大秒数由公式算，**LLM 自己换算**：

> **每段最大帧数 = `chunk_step × 17`**　｜　**每段最大秒数 = `chunk_step × 17 ÷ 24 = chunk_step × 0.708s`**

| chunk_step | 最大帧数 | 最大秒数 |
|---|---|---|
| 6 | 102 | 4.250s |
| 7 | 119 | 4.958s |
| 8 | 136 | 5.667s |
| 11 | 187 | 7.792s |

写 summary 时套一句：`... executed as a N-shot sequence with hard cut(s) at ..., with every segment no longer than <最大秒数> seconds.`

**③ 动作要"跨切点连续，机位要断"**

| 做法 | 说明 |
|---|---|
| ✅ **动作时间线连续** | 0→8 秒是一条完整的动作线，不重复不跳跃 |
| ✅ **机位在切点明确改变** | 中景 → 过肩近景 / 侧跟 → 仰角 |
| ❌ **不要在切点重新开始动作** | 会让观感像"回放" |
| ❌ **不要在连续动作的最激烈处切** | 尽量落在节拍上（一击结束、转身瞬间） |

---

# 第二部分 · 实例：雷姆 · 校园武戏（8 秒 / 硬切 4.250s · **等长 2 镜示例**）

> 改编自用户原稿（原稿要求"one single continuous take with no cuts anywhere"）。
> 配套节点参数：`total_seconds=8` · `cut_1=4.25 / cut_2~4=-1` · `chunk_step=0` · `canvas_megapixels=1.5`
> 执行器切点：**帧 102 = 4.250s**（偏差 0 帧）· 负载 153.0 ✅
>
> **要不等长**（第一段短、第二段长）：把 `cut_1~cut_4` 全设 `-1`，`cut_frames` 填 `68`。
> 于是 `[Shot 1]` 覆盖 0→2.833s、`[Shot 2] At 00:02.833` 覆盖 2.833→8s，其余写法完全一样。

## 时间线重新分配（关键）

原稿是一条 8 秒不间断的动作线。硬切版**动作线不动，只把镜头分成两段**：

| 时刻 | 原稿 | 硬切版 Shot 1（0 → 4.250s） |
|---|---|---|
| 00:00.000 | 中景，Rem 蓄力 | 同 —— 中景，重心下沉，眼神由静转锐 |
| 00:01.200 | 前冲，鞋底刮出火花 | 同 —— 低角度跟随冲刺 |
| 00:02.200 | 袖子格挡 + 侧闪 | 同 —— 第一次交锋 |
| 00:03.400 | （连击段中） | **起手第一脚**，镜头开始贴近 |
| **00:04.250** | — | **★ 硬切 ★** |

| 时刻 | 原稿 | 硬切版 Shot 2（4.250 → 8.000s） |
|---|---|---|
| **00:04.250** | 连击段中 | **★ 过肩近景切入**，连击继续（密集、贴身） |
| 00:05.500 | 半转 + 最后一击 | 同 —— 半转，最后一击把敌人打飞出画 |
| 00:06.500 | 落地定格 + 推近 | 同 —— 稳住架势，镜头推近 |
| 00:08.000 | 结束 | 同 —— 停在坚定表情上，落叶飘落 |

**为什么切 4.25s**：它落在"起手 → 连击"的换挡处，动作不断，但机位能从**侧向中景**干净地换成**过肩近景**。
避开 3.2–5.5 的连击最密处（那里切会像跳帧）。

## 完整提示词

```text
subject_definitions:
<Subject 1> is Rem, the anime girl in <Picture 1>: light-blue short hair with bangs covering her right eye, blue pupils, a white flower headband, a purple butterfly-ribbon hair ornament, the classic black-and-white maid outfit with a white apron front, white over-knee stockings, and black mary-jane shoes; she is rendered as a two-dimensional, cel-shaded anime character with clean lines, crisp color fields, and high-frame-rate action animation quality.
<Subject 2> is the real-world campus park path in <Picture 2>: a curved red-brick walking path, lush green trees, a mown lawn, trimmed shrub clusters, dappled bright sunlight, and a distant campus building, in photorealistic live-action quality under a clear daytime sky.
<Subject 3> is a mysterious shadow enemy: a faceless, blurred dark humanoid silhouette with soft smoky edges, kept slightly out of focus and always subordinate in framing; it never reveals facial features, never holds the foreground longer than Rem, and serves purely as her combat opponent.

summary:
[reference generation] The target video is an 8-second, 16:9, 2K clip with native stereo sound, executed as a two-shot sequence with a hard cut at 00:04.250, fusing the 2D anime heroine <Subject 1> into the photorealistic campus park path of <Subject 2>: Rem sinks into a combat stance, dashes forward with sparks scraping off the bricks, guards with her maid sleeve and sidesteps the strikes of the shadow enemy <Subject 3>, then continues into a rapid chain of kicks and knife-hand strikes that ends with the enemy knocked flying out of frame, finishing on a stable battle-end pose. Both shots share one continuous action line: the cut only changes the camera, never the fight.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her blue short hair with bangs over the right eye, blue pupils, white flower headband, purple butterfly ribbon, black-and-white maid outfit, white apron, white over-knee stockings, and black shoes stay identical in every frame of the high-speed combat across both shots; her proportions remain stable, her face never distorts, her limbs connect naturally, the outfit never clips through her body, and there is no frame flicker.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the curved red-brick path, green trees, lawn, shrubs, dappled sunlight, and distant campus building keep the same real structure, light direction, and color temperature for the entire video across both shots, remaining a stable background through the whole battle.
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

## 这份改写的要点

| 改动 | 原稿 | 硬切版 |
|---|---|---|
| `summary` | `one single continuous take with no cuts anywhere` | **`a two-shot sequence with a hard cut at 00:04.250`** + `the cut only changes the camera, never the fight` |
| `retention_analysis` | `(appears throughout [Shot 1])` | **`(appears in [Shot 1], [Shot 2])`**；`<Subject 3>` 补了一条 |
| `detailed_description` | 单个 `[Shot 1]` 覆盖 0→8 | **`[Shot 1]` 收在 3.400 的起手并"hold right up to the cut"；`[Shot 2]` `At 00:04.250` 过肩近景切入** |
| 镜头语言 | 一镜内 `the camera drops / sweeps into a lateral half-orbit` | **切成过肩近景，`the camera now riding low with her`** |
| 音效 | `Throughout the video: ...` | **加一句 `carry straight across the cut ... the sound does not restart`** |

## 想换切点？改这 4 处

1. `MiniMaxH3HardCutPlan` 的 `cut_1`~`cut_4`（用不上的槽填 -1）
2. `summary` 里的 `at 00:05.667`
3. `[Shot 2]` 开头的 `At 00:05.667, `
4. `overall_soundscape` 末尾那句的 `at 00:05.667`

**第 2–4 处可以直接抄节点 `cut_report` 里的 `cut points` 行**，或者把 `cut_seconds` 连到
`MiniMaxH3HardCutShotPrompt` 让它自动生成 `[Shot N]` 行（推荐）。

---

# 第三部分 · 常见错误

| 现象 | 原因 |
|---|---|
| 切点处画面"回放"了一下 | `[Shot 2]` 把动作重新开始了一遍 —— 动作线必须连续 |
| 切点处观感像崩了而不是剪辑 | `[Shot 2]` 没换机位，两镜看起来几乎一样 |
| 模型抵抗切点、画面拖着不切 | `summary` 里还留着"一镜到底/无剪辑"的表述 |
| 切完角色长相变了 | `retention_analysis` 没列出切点后的 `[Shot N]` |
| 声音也在切点处断了 | 正常 —— 本路线音频整片透传。若不想断，检查提示词里有没有写"换环境音" |
| 切点位置和提示词差半秒 | `cut_seconds` 没连过去，或手抄错了时间戳 |
