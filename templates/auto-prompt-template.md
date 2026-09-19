# 自动硬切 · 提示词生成（喂 LLM 就复制这一份）

> 给它**三样东西**：片长 · `chunk_step` · 剧情要求。
> 它输出的是**完整六段、时间戳已填好、可直接粘贴**的提示词 —— **没有占位符**。

---

## 一、你要给 LLM 的输入

```text
片长：10 秒
目标镜头数：2            ← 你要几个镜头就写几（决定段数，也决定跑起来的开销）
chunk_step：8            ← 必须与镜头数匹配（查第三条的表）
剧情要求：女学生穿过校园小径奔跑，最后回头看向镜头。
```

---

## 二、第一条规则：**镜头数由你指定，剧情在框内安排**

**镜头数 ≥ 段数**（不是相等）：

- **执行器每切一刀 = 一个硬断点**（`overlap = 0`，段之间没有上下文）→ 所以**每一刀都必须出现在你的时间戳里**
- **反过来说得通**：提示词里的时间戳**可以比执行器的刀更多** —— 那些多出来的时间戳是**同一段内模型自己切的镜头**（和一采整片生成多镜是一个机制）
- 所以「5 个镜头 + 2 段」完全合法：段边界是 2 个硬切，段内还有 3 次由模型完成的切换

> **先读剧情，然后把它讲圆在你指定的 N 个镜头里。**
> N 个镜头塞不下 → **不要偷偷加镜头**，输出一行提醒：
> `「这段剧情 N 个镜头偏挤，建议 N+1 镜（chunk_step 改 X）」`，然后按 N 镜先写一版。

**为什么镜头数由你定而不是"剧情需要多少就多少"**：镜头数直接决定跑起来的开销与风险 ——

| | 镜头少（段少） | 镜头多（段多） |
|---|---|---|
| 总计算量 | 一样（总帧数不变） | 一样 |
| 采样调用次数 | **少**（每段一次，含固定开销） | 多 |
| 剪辑点（接缝） | **少**（画面更连贯） | 多 |
| 单段显存压力 | 高（段更长） | **低** |

所以要**在显存扛得住的前提下，能用少镜头就别用多镜头**；只有当最长段 × 画布MP 逼近 236（要爆显存）时才必须多切。

---

## 三、第二条规则：`chunk_step` 必须与目标镜头数匹配

**段数是 `chunk_step` 的函数，不是自由变量。** 节点从 n = 1 开始试，
第一个「合法」的 n 就是结果（合法 = 每段 ≤ `chunk_step×17` 帧、尾段 ≥ 17 帧、最长段×1.544 < 236），
**试到合法的就不再往下试**。查表（二采 1.544 MP）：

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

> 表里没有的片长：用第四条公式试 n = 1, 2, 3…，取第一个满足全部条件的。

---

## 四、第三条规则：切点必须按公式算（否则节点会拦）

节点用的是同一套算法，所以**你算出来的时间戳必须和它一致**。

```text
① 总帧数   F = round(秒 × 24) + (5 − round(秒 × 24) % 17) % 17
② 每段上限 C = chunk_step × 17                        （帧）
③ 段数     n = 1, 2, 3… 试，取第一个满足全部四条的：
               · 按下面公式切出的段数 = n
               · 每段 ≤ C
               · 尾段 ≥ 17 帧
               · 最长段 × 1.544 < 236        （1.544 = 二采画布 MP）
④ 切点帧   第 i 个 = round( round(i × F / n) / 17 ) × 17      i = 1 … n−1
⑤ 时间戳   切点帧 ÷ 24 → MM:SS.mmm（三位小数）
```

例（片长 10 秒 / `chunk_step` 5）：`F = 243`，`C = 85`，`n = 3`，
切点帧 `85 / 170` → **`00:03.542` / `00:07.083`**。

> **帧 → 秒：除以 24**（85 → 3.542s，170 → 7.083s）。**17 帧 = 0.708s**，切点永远是它的整数倍。
> 算不准就以 `#56` 节点输出的骨架为准 —— 那是同一套算法算的，100% 一致。

---

## 五、输出格式：直接给可粘贴的完整六段

**硬规则**：

- **六段顺序不能乱**，全英文（`<d>` 里的对白保留原语言）
- **每一个 `[Shot N]` 都要有时间戳**（`[Shot 1]` 除外，官方模板不带）
- **不留任何占位符** —— `{{CUT_1}}` 这种不许出现
- **禁止** `no cuts` / `continuous take` / `unbroken` / `one take` / `single take`
- `summary` 里写清 **几镜 + 每个切点时间** + `every segment no longer than X seconds`
- `retention_analysis` **列全所有 `[Shot N]`**
- `overall_soundscape` 要写明声音**跨过每个切点不中断**
- **相邻镜头必须换机位/景别**（两镜画面太像，切点会像"崩了"）

```text
subject_definitions:
<Subject 1> is <角色> in <Picture 1>, <外形特征>.
<Subject 2> is <场景> in <Picture 2>, <特征>.

summary:
[reference generation] The target video is a <F/24>-second clip with native stereo sound,
executed as an <N>-shot sequence with hard cuts at <算好的时间戳逐字填>, with every
segment no longer than <C/24 秒> seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2] ...): fully_preserved - <保留了什么>.
<Subject 2> (appears in [Shot 1], [Shot 2] ...): fully_preserved - <保留了什么>.

detailed_description:
<一两句英文定调：风格/画质/色调>.
[Shot 1] <构图 + 主体位置与动作 + 环境光 + 镜头运动 + 声音>.
[Shot 2] At <时间戳>, <换机位/景别 + 动作接续>.
[Shot 3] At <时间戳>, <再换机位/景别 + 收束>.

overall_soundscape:
<环境音 + 物理音效；写明跨过切点不中断、不重复>.

non_diegetic_music:
<配乐描述，或 N/A>.
```

> ⚠️ 上面尖括号/中文里的都是**结构说明，不是让你照抄的**。
> **最终答案里不能出现任何尖括号、下划线或占位符** —— 照下面第六节那样，
> 每个数字和描述都写成实数实词，用户拿到就能直接粘贴。

---

## 六、完整示例（10 秒 / `chunk_step` 5 / **3 镜**）

> 这份是**实测跑通的**（节点校验 `status OK`，0 错 0 警），切点 `00:03.542` + `00:07.083`。

```text
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.
<Subject 2> is the tree-lined campus courtyard in <Picture 2>, late afternoon, warm low sunlight and long shadows.

summary:
[reference generation] The target video is a 10.125-second clip with native stereo sound, executed as a three-shot sequence with hard cuts at 00:03.542 and 00:07.083, with every segment no longer than 3.542 seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - face, hair, cardigan and necklace stay identical across both cuts; only the camera distance changes.
<Subject 2> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the same courtyard and the same low afternoon light continue across both cuts.

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

**三段情绪正好落在三个镜头上**：起跑（急促）→ 中段坚持 → 特写收束。这就是"剧情优先"该有的样子。

---

## 七、节点会怎么检验（错一条就拦停）

| 检查 | 说明 |
|---|---|
| 六段齐全、`[Shot N]` 从 1 连续 | 缺段或跳号 → 错 |
| **每一刀都在提示词里** | 执行器切在 3.542s 但提示词没写那个时间戳 → 错。**提示词多出时间戳（段内模型自己切镜）不算错** |
| **每个时间戳落在真会切的那一帧** | 差 1 帧就报错，并告诉你正确值 |
| **没有 `no cuts` / `continuous take` / `unbroken`** | 与硬切天然冲突 → 错 |
| 尾段 ≥ 17 帧、每段 ≤ `chunk_step × 17` | 超了 → 错 |
| 负载（最长段 × 画布MP）< 236 | 超了 → 错 |
| 段数 ≤ 10 | — |

改完提示词先别急着跑：`#56` 节点上印的报告第一行 `status` 是 **OK** 才算过。
