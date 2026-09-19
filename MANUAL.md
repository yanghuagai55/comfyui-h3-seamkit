# 硬切作业手册（跨模型协作流程）

> 适用：`comfyui-h3-hardcut` + MiniMax H3 双采分块
> 提示词模板：`templates/hardcut-prompt-template.md`
> 原理与依据：`refs\H3-硬切分镜提示词工程.md`

---

## 0. 全流程一览

```
① 定切点        插件算出「可达切点表」（17 帧网格），你从中挑一个
                    ↓  ← 这一步必须在写提示词之前！
② 写提示词      把「约束块 + 切点表 + 你的剧情」喂给任意 LLM
                    ↓  LLM 输出：六段完整提示词（带 At MM:SS.mmm）
③ 抄回 ComfyUI  #48 填切点 · #9 填提示词
                    ↓  #49 校验器就串在 #9 和条件节点之间，自动透传
④ 看校验        #49 节点上直接印出体检报告 —— status 必须是 OK
                    ↓  不是 OK 就别跑：校验器会直接报错拦停
⑤ 跑            再看 #42 的 cut_report，确认切点和负载
```

**为什么必须先定切点**：LLM 要写 `At 00:04.250`，而这个时间**须落在 token 边界上才可达**（1-4 帧粒度；17 帧整数倍是推荐档）。
先让 LLM 自由发挥，它会写出 00:04.500 这种落不了地的值。

---

## 1. 第一步 · 拿到可达切点表

**方法 A — 命令行（最快）**

```bash
D:\comfyui\comfyenv\python.exe D:\comfyui\ComfyUI\custom_nodes\comfyui-h3-hardcut\hardcut_math.py 8 4.25 --mp 1.5
```

第一个参数 = 时长，第二个 = 你**想要**切在哪（可以随便填，它会给菜单）。
把最后那段 `--- reachable windowing menu ---` 整块复制下来，第 2 步要用。

**方法 B — 在 ComfyUI 里**

`#48 MiniMaxH3HardCutPlan` 的 `cut_1`~`cut_4` 随便填 → 运行 → 看 `#42` 的 `cut_report`。

**看懂菜单**：

```
--- reachable windowing menu (cuts are multiples of 17 frames) ---
  2 windows / 1 cut(s):
     chunk 102  cuts @ 4.250s                 frames [102, 90]
     chunk 119  cuts @ 4.958s                 frames [119, 73]
     chunk 136  cuts @ 5.667s                 frames [136, 56]
  3 windows / 2 cut(s):
     chunk 68   cuts @ 2.833s, 5.667s         frames [68, 68, 56]
```

- **windows** = 段数（= 1 + 切点数）
- **chunk** = 每段帧长（只影响负载，不影响提示词）
- **cuts** = 可达切点 → **挑一个写进提示词**
- **frames** = 各段帧数 → 末段太短就换一行

**挑哪个**：

| 你的情况 | 建议 |
|---|---|
| 武戏 / 动作段落 | 切在**换挡处**（一次交锋结束、起手瞬间），别切在动作最密处 |
| 对话戏 | 切在**说话人切换**的地方 |
| 只想保一点连续感 | 选 2 段（1 处切） |
| 要更省显存 / 更长时长 | 选 3–4 段（负载 = 最长段 × 画布MP，段越短越省） |

---

## 2. 第二步 · 喂给 LLM（约束块，可直接复制）

**喂什么给 LLM —— 两种都行**：

| 方式 | 内容 | 适合 |
|---|---|---|
| **A（推荐）** | 本节的**约束块**（把 `<...>` 换成实际值 + 粘上切点表 + 剧情） | 一次生成，可控 |
| B | 整个 `templates/hardcut-prompt-template.md` + 你的剧情 + 切点表 | 让 LLM 参考实例风格 |

**★ 不管哪种方式，切点表都必须给它** —— 否则它会写出落不了地的时间戳。

把下面整块**连同你的剧情**发给任意大模型（GPT / Claude / 任何能写英文的模型）。

```text
你是一位 MiniMax H3「全参考模式 (R2V)」视频提示词工程师。请按下面的硬性约束写一条提示词。

【硬性约束 —— 违反任何一条都会导致生成失败或接缝难看】
1. 全英文书写。只有 <d> 标签里的对白/唱词保留原语言。
2. 必须按 6 段结构，顺序不能变，缺一不可：
   subject_definitions / summary / retention_analysis /
   detailed_description / overall_soundscape / non_diegetic_music
3. 视频总长 <总秒数> 秒（<总帧数> 帧），**在 <切点时间> 处有一次硬切**，分成 2 个镜头。
4. summary 里必须写：
   "a two-shot sequence with a hard cut at <切点时间>"
   **绝对不要出现** "one continuous take" / "no cuts" / "single take" 这类表述。
5. summary 必须以 "[reference generation]" 开头。
6. retention_analysis 每一项都要列出它出现在哪些镜头，格式：
   <Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <具体保留了什么>
7. detailed_description 的写法：
   - 先写一两句「定调句」（风格/画质/色调），放在 [Shot 1] 之前
   - [Shot 1] 不带时间戳
   - [Shot 2] 必须以 "At <切点时间>, " 开头
   - **两镜的动作线必须连续**（同一条动作，不重复、不跳跃）
   - **只有机位/景别/角度改变**（硬切的意义就在这里）
8. 每个 [Shot N] 都要写全这 6 要素：
   ① 构图 ② 主体外观与位置 ③ 环境光照 ④ 动作与状态变化
   ⑤ 镜头运动 ⑥ 当前声音
9. overall_soundscape 是「贯穿全片」的定义，时间线归 detailed_description 管。
   请加一句说明声音跨切点连续，例如：
   "The ambience carries straight across the cut without a break — the picture changes
    angle, the sound does not restart."
10. 标签规则：角色/场景/服装/风格都用 <Subject N>；只有当某张图本身就是首帧/
    关键帧时才用 <Picture N>。标签含义在 6 段里必须一致。
    首次出现重要 Subject 时写清「参考特征 + 画面中的位置 + 当前动作」，
    后续镜头沿用同一标签，不要重新定义。
11. 对白格式：<Subject 2> (S1) says, <d>[Japanese] ...</d>
12. detailed_description 篇幅 350–500 英文词。

【本次参数】
- 总时长：<总秒数> 秒（<总帧数> 帧）
- 硬切点：**<切点时间>**（这是唯一合法值，一个字都不要改）
- 画幅：<画幅>，<风格定位>

【以下是该时长下所有合法的备选切点，仅供你理解为什么必须是上面那个值】
<把可达切点表整块粘在这里>

【剧情 / 分镜意图】
<在这里描述你的片子：谁、在哪、发生什么、想怎么切>

【输出要求】
只输出提示词本体（6 段），不要任何解释、不要 markdown 代码围栏。
```

**LLM 返回后做 3 个检查** —— **这三条已经由 `#49` 校验器自动做了**（见第 3 步），
下面这张表只是让你知道它在查什么、报错时怎么读：

| 检查 | 校验器怎么报 |
|---|---|
| `summary` 里没有"不切"的表述 | `[E] wording contradicts a hard cut ... 'no cuts' / 'single continuous take'` |
| `[Shot 2]` 的时间戳**完全等于**你的切点 | `[E] [Shot 2] timestamp is frame 108 (00:04.500) but the plan cuts at frame 102 (00:04.250)` |
| `retention_analysis` 列出了 `[Shot 1], [Shot 2]` | `[W] retention_analysis mentions shots [1] but the video has [1, 2]` |

---

## 3. 第三步 · 抄回 ComfyUI

**① 填切点** —— `#48 MiniMaxH3HardCutPlan`

| 参数 | 填什么 |
|---|---|
| `total_seconds` | 片长（1–15） |
| `cut_1` ~ `cut_4` | 第 1 步挑的切点，每槽一个数；**用不上的槽填 `-1`（= 不切）**。例：8 秒单切 = `cut_1=4.25` 其余 `-1`；15 秒 4 段 = `4.25 / 8.5 / 12.75 / -1` |
| `chunk_step` | 默认 `0`（用切点槽推出的窗口长度）。想手动调窗口：**每 ±1 = ±17 帧**，报告里的 `chunk ladder` 逐档列出「会切几刀、切在哪、负载多少」 |
| `canvas_megapixels` | 你实际用的二采画布（如 `1.5`），只为估算负载 |
| `target_width` / `target_height` | **接 `#36 ResolutionSelector`**（别手填） |

**② 填提示词** —— `#9 PrimitiveStringMultiline`

把 LLM 输出的**六段全文**粘进去（覆盖原来的）。

**③ 让校验器把关（推荐，已经在链路里了）**

`#49 MiniMaxH3HardCutValidate` **串在 `#9` 和两个条件节点之间**：

```
#9 提示词 ──► #49.prompt ──► #8.prompt（低清条件）
                  │        └► #41.prompt（高清条件）
                  │
   #48.plan ──► #49.plan（窗口长度/切帧/画布全从这里读）
                  │
                  └─► 报告直接印在 #49 节点上（不需要预览节点）
```

它**透传**提示词，本身不改一个字，但会拦住所有"对不上"的情况。
**`#49` 上没有任何设置项** —— 判据就是节点上印的报告第一行 `status` 必须是 **OK**；
不是 OK 的话，**这个节点会直接抛错，跑都不会开始**。

**它查 8 类问题：**

| # | 查什么 | 严重度 |
|---|---|---|
| 1 | 六段 section 是否齐全、有没有重名 | 错 / 警 |
| 2 | `[Shot N]` 编号是否从 1 连续（只在 `detailed_description` 里数，不会把 `retention_analysis` 里的引用误算成镜头） | 错 |
| 3 | **有没有"不切"的措辞** —— `no cuts` / `without a cut` / `never cuts` / `continuous take` / `unbroken` / `in one take` / `no edit` | 错 |
| 4 | `[Shot 2]` 起是否都以 `At MM:SS.mmm,` 开头 | 错 |
| 5 | **执行器每一刀是否都出现在提示词时间戳里**（段边界是硬断点，模型必须被告知在此换镜）。**提示词时间戳多于刀数 = 段内模型自己切镜，合法** | 错（只拦"刀不在提示词里"） |
| 6 | **时间戳落不落在执行器真会切的那一帧**（计划已接线 → 计划是真值；未接线 → 用 17 帧网格判） | 错 |
| 7 | 提示词里写的时长 vs `total_seconds` | 警 |
| 8 | 段长 × 画布MP 是否越过 OOM 锚点（画布从 plan 的 target_width/height 算） | 错 / 警 |

**▲ 报错长什么样**（这是最有用的那一条）：

```
Hard-cut prompt check failed (1 error(s)) — the prompt and the plan disagree,
so the model would not cut where the executor cuts.
  - [Shot 2] timestamp is frame 108 (00:04.500) but the plan cuts at frame 102
    (00:04.250) — off by +6 frame(s)
```

**「自动匹配可分割的时间段」**：报告末尾永远附一张**可达切点菜单**，切点落不了地时直接照着换：

```
--- reachable windings for this clip (cuts are multiples of 17 frames) ---
  chunk 102  2 windows, cuts @ 4.250s
  chunk 119  2 windows, cuts @ 4.958s
  chunk 136  2 windows, cuts @ 5.667s
```

**它没有任何旋钮** —— 窗口长度、切帧、二采画布全部从 `plan` 里读（`temporal_chunk_frames` /
`temporal_overlap_frames` / `target_width/height`），所以**没有填错的可能**。
`plan` 没接线时它会降级成"只按提示词自己判"，并在报告里提示你 `no plan was supplied`。

**不想用校验器**：删掉 `#49`，把 `#9` 直接接回 `#8.prompt` 和 `#41.prompt`，
然后手动对着 `#42` 的 cut_report 核时间戳。**但那样就没人拦你了。**

**④（可选）让 `#49` 之前的自动组装节点帮你写时间戳**

`MiniMaxH3HardCutShotPrompt` 这个节点现在**不在工作流里**（`#49` 位置让给校验器了），
但你随时可以加一个：它把「定调句 + 逐镜描述」拼成 `detailed_description`，
时间戳从 `#48` 自动取，**永远抄不错**。`shots` 两种写法都吃：

| 你粘进去的 | 它怎么处理 |
|---|---|
| 裸描述：`The scene opens on Rem at medium shot...` | 自动补 `[Shot 1] ` / `[Shot 2] At 00:04.250, ` |
| LLM 的完整行：`[Shot 2] At 00:04.250, the shot cuts to...` | **原样保留**，不重复加 |

用法：把它插在 `#9` 前面（或旁边），把输出粘回 `#9` 的 `detailed_description` 段。

**⑤ 跑之前再看一眼 `#42 PreviewAny`（cut_report）**

```
cut points    : 4.250s (frame 102, 00:04.250)     ← 必须和提示词里的一致
vs requested  : +0.00f                            ← 偏差，太大就换切点
load estimate : 102f x 1.500MP = 153.0  (SAFE)    ← 不是 SAFE 就降画布或加切点
```

---

## 4. 快速检查清单

跑之前逐条过一遍：

- [ ] `#48` 的 `cut_1`~`cut_4` = 第 1 步挑的切点（用不上的槽填 `-1`），`chunk_step` 默认 `0`
- [ ] **`#49` 节点上的报告 `status` 是 `OK`** ← 不是 OK 就别跑，节点会抛错
- [ ] `#49` 报告里没有 `[E...]` 开头的行（`[W...]` 警告可以先忍着）
- [ ] `#49 plan` 已接线（报告里不该出现 `no plan was supplied`）
- [ ] `#42` cut_report 里 `cut points` 和提示词里的 `At MM:SS.mmm` 一致
- [ ] `#42` cut_report 里 `load estimate` 是 `SAFE`
- [ ] ComfyUI 重启过（新节点才在）

---

## 5. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| **`#49` 直接抛错 `Hard-cut prompt check failed`** | 校验器拦住了对不上的提示词 | 看报错里的具体条目（它就是答案）；完整报告印在 `#49` 节点上 |
| `[E] [Shot 2] timestamp ... off by +6 frame(s)` | 提示词的时间戳不在执行器会切的那一帧 | 照报告末尾的**可达切点菜单**改成合法值（如 4.500→4.250） |
| `[E] prompt describes 3 shot(s) ... 2 window(s)` | 提示词写的镜头数和 `#48` 会切出的段数不一致 | 要么提示词并成一镜，要么 `#48` 的某个 `-1` 槽改成切点 |
| `[E] wording contradicts a hard cut` | 提示词里留着"一镜到底"类表述 | 改成 `a two-shot sequence with a single hard cut at ...`（校验器会告诉你在第几行） |
| `[W] retention_analysis mentions shots [1]` | 没列全镜头 | 补成 `(appears in [Shot 1], [Shot 2])` |
| 不想被拦停 | — | 把 `#49` 从链路里摘掉（`#9` 直连两个条件节点）—— 它没有开关 |
| 切点处像"崩了"而不像剪辑 | `[Shot 2]` 没换机位，两镜画面几乎一样 | 让 LLM 给 Shot 2 换个**明显不同的机位/景别**（过肩、仰角、大特写） |
| 切完角色长相变了 | `retention_analysis` 没列出切点后的镜头 | 补上 `(appears in [Shot 1], [Shot 2])` |
| 模型抵抗切点，画面拖着不切 | `summary` 里留着"一镜到底"类表述 | 删掉，改成 `a two-shot sequence with a hard cut at ...` |
| LLM 给的切点落不了地 | 它没用你的可达切点表 | 重新发一次，把菜单**整块**粘进去，强调"唯一合法值" |
| 切点处声音也断了 | 正常 —— 音频是**一采整片透传**，不随视频切 | 想换声音只能在提示词里写（见模板 §4） |
| 想换切点怎么办 | — | 改 4 处：`#48` 的切点槽 + 提示词里的 `summary` / `[Shot N]` / `overall_soundscape` 尾句 |
| 报形状契约错误 | `target_width/height` 与喂给 HIGH conditioning 的画布不一致 | 两处必须同源（都接 `#36`） |
| OOM | 段太长 × 画布太大 | 看 cut_report 的 `canvas ceiling`；或加一处切点把段切短 |

---

## 6. 常用时长速查（1.5MP 画布）

| 时长 | 帧数 | 段数 | 切点 | 最长段 | 负载 |
|---|---|---|---|---|---|
| 5s | 124 | 2 | 2.833s | 68 | 102 ✅ |
| 8s | 192 | 2 | **4.250s** | 102 | **153 ✅** |
| 8s | 192 | 2 | 4.958s | 119 | 178.5 ✅ |
| 8s | 192 | 3 | 2.833 / 5.667s | 68 | 102 ✅ |
| 10s | 243 | 2 | 4.958s | 119 | 178.5 ✅ |
| 12s | 294 | 3 | 4.250 / 8.500s | 102 | 153 ✅ |
| **15s** | **362** | **3** | 5.667 / 11.333s | 136 | 204 ✅ |
| 15s | 362 | 4 | 4.250 / 8.500 / 12.750s | 102 | 153 ✅ |

> 表只是示例。**任何时长都以 `hardcut_math.py` 的菜单为准**（可达切点随时长变化）。

---

## 7. 一条命令搞定（不想开 ComfyUI 时）

```bash
# 看某时长的全部可行方案
D:\comfyui\comfyenv\python.exe <插件目录>\hardcut_math.py 8 4.25 --mp 1.5

# 三个常用的
...\hardcut_math.py 8  4.25 --mp 1.5      # 8 秒切 4.25
...\hardcut_math.py 15 5,10 --mp 1.5      # 15 秒想切 5/10（会告诉你实际能切哪）
...\hardcut_math.py 10                    # 留空 = 自动按 ~5 秒分段

# ★ 不开 ComfyUI 也能体检提示词（退出码 0=通过 / 1=有问题）
...\hardcut_math.py --check prompt.txt --total 8 --cuts 4.25 --mp 1.5
```

`--check` 把一份提示词文本文件当输入，跑的就是 `#49` 同一个校验器。
**写完提示词先在这里过一遍**，比在 ComfyUI 里点运行快得多。
