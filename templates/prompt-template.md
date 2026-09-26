# 提示词模板（本插件唯一保留的一份）

> 母本：官方 `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`（中文整理见 `refs/H3-R2V提示词模板-官方.md`）。
> 本文件 = **官方六段结构 + 硬切路线的 4 条注意事项**。
> **不需要任何帧数/秒数换算** —— 分块大小由上游规划节点的 `target_segment_seconds` 决定，
> 你只要让提示词里的分镜与节点报告的切点对齐即可（见注意事项 1）。

---

## 一、六段骨架（顺序不能乱，全英文，`<d>` 内对白保留原语言）

```text
subject_definitions:
<Subject 1> is <角色/物体> in <Picture 1>, <外形特征>.
<Subject 2> is <场景> in <Picture 2>, <特征>.

summary:
[reference generation] The target video is a <时长>-second clip with native stereo sound,
executed as a <N>-shot sequence with hard cuts at <MM:SS.mmm>[, <MM:SS.mmm> ...],
with every segment no longer than <target_segment_seconds> seconds.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], ...): fully_preserved - <具体保留了什么>.
<Subject 2> (appears in [Shot 1], [Shot 2], ...): fully_preserved - <环境/光线/构图>.
<Audio 1>: reference - <参考音色/节奏，不搬原话>.

detailed_description:
<一两句定调：风格/画质/色调 —— 必须在 [Shot 1] 之前>.
[Shot 1] <构图 + 主体位置与动作 + 环境光 + 镜头运动 + 声音/对白>.
[Shot 2] At <MM:SS.mmm>, <换机位/景别 + 动作接续>.
[Shot 3] At <MM:SS.mmm>, <再换 + 收束>.

overall_soundscape:
<环境音 + 物理音效>。声音跨过每个切点不中断、不重复.

non_diegetic_music:
<配乐描述，或 N/A>.
```

**官方硬规则**（逐条别漏）：

- `[Shot 1]` **不带时间戳**；`[Shot 2]` 起每段以 `At MM:SS.mmm, ` 开头
- `summary` 以任务类型前缀开头（参考图锁角色 + 生成新视频 → `[reference generation]`）
- 只用来定义角色/场景/服装/风格的图 → **写成 `<Subject N>`**，不要单开 `<Picture N>` 条目
- 对白写 `<d>[Language] ...</d>`，说话人标 `(S1)`/`(S2)`；跨切点对白加 `<scenetrans>`
- `detailed_description` 350–500 英文词，每段写全 6 要素：构图 / 主体外观与位置 / 环境光 /
  动作与状态变化 / 镜头运动 / 当前声音
- `retention_analysis` 每项列出**全部** `[Shot N]`（漏了模型就不知道切完还要保持什么）

---

## 二、retention 标签：官方**两族**，别混

| 族 | 适用 | 合法标签 |
|---|---|---|
| **画面类** | `<Subject>` / `<Picture>` / `<Video>` | `fully_preserved` · `partially_preserved` · `attribute_transfer` · `weak_reference` |
| **音频类** | `<Audio>` | `fully_copy` · `partially_copy` · `reference` · `weak_reference` |

> 校验器（`MiniMaxH3HardCutValidate`）按**两族并集**校验；其它形如 `xxx_yyy` 的标签会警告。

---

## 三、★ 硬切路线专属注意事项（4 条）

### 1. 切点时间戳必须逐字对齐 —— 从节点报告里抄

运行一次（或看节点执行后）**上游规划节点 `MiniMax H3 Plan (upstream / first-pass)` 的报告**：

```text
cut points    : 3.750s (frame 90), 7.500s (frame 180), 11.250s (frame 270)
windows:
  #0  frames [   0 ->   85) ...
```

**把这些时间戳原样写进**提示词的 `summary` 与各 `[Shot N] At ...,`。差一帧就会被
`MiniMaxH3HardCutValidate` 拦下（报告会给出正确值）。

段长上限写进 `summary` 最后一句：`with every segment no longer than <target_segment_seconds> seconds`
（例：`target_segment_seconds = 5.0` → 每段不超过 5 秒）。

### 2. 绝对不要出现"一镜到底"措辞

`no cuts` / `continuous take` / `unbroken` / `one take` / `single take` —— 与硬切天然冲突，
模型会抵抗切点。`summary` 里写：

```text
executed as a <N>-shot sequence with hard cuts at <...>
```

### 3. ★ blocking 显式化（站位/朝向必须写死）

两个采样窗口是**独立渲染**的：站位语义模糊时，窗口 A 解读成"侧身相对"、窗口 B 解读成
"正面朝镜" → 锚定接缝处**姿势跳变**（实测 00087 翻车点）。

```text
✅ Both girls stand side by side, BOTH facing the camera directly, shoulders square to the
   lens, eyes to the viewer — this exact blocking is kept in every single frame of the video.

❌ facing slightly toward each other      ← 模糊词 = 接缝跳变的种子
```

凡"跨窗口必须一致"的视觉语义（站位、朝向、视线、与镜头的相对关系）都写死 + 声明全片统一。

### 4. 声音跨切点连续

`overall_soundscape` 末尾加一句：

```text
The ambience and effects carry straight across the cuts at <...> without a break —
the picture changes angle, the sound does not restart.
```

（本路线音频整片透传；不写这句，模型可能自己"换环境音"。）

---

## 四、开机检查清单

- [ ] 六段齐全、顺序正确、全英文（`<d>` 除外）
- [ ] `summary`：`[reference generation]` + 硬切表述 + 每个时间戳 + 段长上限
- [ ] `[Shot 1]` 无时间戳；`[Shot 2]` 起都有 `At MM:SS.mmm, `
- [ ] 时间戳**逐字**等于节点报告的 `cut points`
- [ ] `retention_analysis` 列全所有 `[Shot N]`；标签用官方两族
- [ ] 没有 `no cuts` / `continuous take` / `unbroken`
- [ ] **blocking 写死**（站位/朝向/视线 + "全片统一"声明）
- [ ] `overall_soundscape` 有"跨切点不中断"一句
- [ ] 相邻镜头**换了机位或景别**（否则切点看起来像"崩了"而不是剪辑）
- [ ] `MiniMaxH3HardCutValidate` 报告第一行 `status : OK`
