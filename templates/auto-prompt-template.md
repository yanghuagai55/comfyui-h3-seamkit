# 自动硬切 · 提示词模板（喂 LLM 就复制这一份）

> **用法**：把下面 ①②③ 三块复制给 LLM，再附上你的剧情。
> **分几段、切在哪一帧由节点算** —— 不要让它自己分段（它算不出 17 帧网格）。
> 节点使用说明、筛选算法、报错排查都不在这里（见 `AUTO.md`）。

---

## ① `chunk_step` → 每段最大秒数

```text
每段最大帧数 = chunk_step × 17
每段最大秒数 = chunk_step × 17 ÷ 24 = chunk_step × 0.708s
```

| chunk_step | 最大帧数 | 最大秒数 |
|---|---|---|
| 4 | 68 | 2.833s |
| 5 | 85 | 3.542s |
| 6 | 102 | 4.250s |
| 7 | 119 | 4.958s |
| 8 | 136 | 5.667s |
| 9 | 153 | 6.375s |
| 11 | 187 | 7.792s |

> 填这里：**我这次的 `chunk_step` = ______　→　每段最长 ______ 秒。**

---

## ② 三条约束（这段直接给 LLM）

1. **每个片段不超过上面算出的秒数。** 这是唯一的长度规则。
2. **`[Shot N] At MM:SS.mmm` 的时间戳用我给你的**，不要自己编。
   切点必须是 **17 帧的整数倍**（`秒 = 帧 ÷ 24`，一格 0.708s），你自己算必然对不上，会被拦停。
   我没给时间戳时，就照 ③ 的骨架把 `{{CUT_1}}` `{{CUT_2}}` 占位符**原样留着**，我去填。
3. **禁止写** `no cuts` / `continuous take` / `unbroken` / `one take` / `single take`
   —— 这条链路就是硬切，写了会跟分块打架，直接跑不起来。
   另外 `retention_analysis` **必须列全所有 `[Shot N]`**，漏一个切完会长相漂移。

---

## ③ 六段骨架（复制这段给它）

```text
subject_definitions:
<Subject 1> is <角色/物体> in <Picture 1>, <外形特征>.
<Subject 2> is <场景> in <Picture 2>, <特征>.

summary:
[reference generation] The target video is a <时长>-second clip with native stereo sound,
executed as a <N>-shot sequence with hard cuts at {{CUT_1}} and {{CUT_2}}, with every segment
no longer than <X> seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <具体保留了什么>.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - <具体保留了什么>.

detailed_description:
<一两句英文定调：风格/画质/色调>.
[Shot 1] <构图 + 主体位置与动作 + 环境光线 + 镜头运动 + 声音>.
[Shot 2] At {{CUT_1}}, <换一个明显不同的机位/景别，动作线接上一段>.

overall_soundscape:
<环境音 + 物理音效，贯穿全片；要写明它跨过切点不中断>.

non_diegetic_music:
<配乐描述，或 N/A>.
```

> 六段顺序不能乱，标签含义六段内必须一致（详见 `refs\H3-R2V提示词模板-官方.md`）。

---

## ④ 填好的示例（8 秒 / `chunk_step` 6 → 2 段 / 每段 ≤ 4.250s）

```text
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan,
and a thin silver necklace.
<Subject 2> is the tree-lined campus courtyard in <Picture 2>, late afternoon, warm
low sunlight and long shadows.

summary:
[reference generation] The target video is an 8-second clip with native stereo sound,
executed as a two-shot sequence with a hard cut at 00:04.250, with every segment no
longer than 4.250 seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - face, hair, cardigan and
necklace stay identical across the cut; only the camera distance changes.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the same courtyard and
the same low afternoon light continue across the cut.

detailed_description:
The target video is in a cinematic, softly lit style with a slightly desaturated palette
and shallow depth of field.
[Shot 1] A low-angle tracking shot follows the woman as she runs along the courtyard
path, her cardigan lifting in the wind; dust and petals drift through the low sunlight.
Her footsteps and breath are close and dry; faint cicadas sit in the distance.
[Shot 2] At 00:04.250, the shot cuts to a tight over-the-shoulder angle that rides just
behind her and pushes slightly toward the back of her head; the run continues without a
break, the same stride and speed, sunlight flickering through the leaves as she passes.

overall_soundscape:
Faint steady campus ambience with cicadas and a light breeze in the leaves runs beneath
it all; footsteps, breath and the rustle of fabric stay continuous across the cut at
00:04.250, with nothing duplicated or dropped at the join.

non_diegetic_music:
N/A.
```

> 上面这份已通过节点校验（`status OK`）。
> **两镜必须换机位/景别** —— 两镜画面太像的话，切点会像"崩了"而不像剪辑。
