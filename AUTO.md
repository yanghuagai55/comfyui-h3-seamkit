# 自动硬切节点 · MiniMaxH3HardCutAuto

> 来源：插件 `comfyui-h3-hardcut`（`hardcut_math.py` 的 `auto_plan` / `resolution_for`）
> 提示词规范：`refs\H3-R2V提示词模板-官方.md`　·　负载锚点实测：RTX 4060 Laptop 8GB，二采 1664×928（1.544MP）

> **一个节点顶替「规划 + 校验 + 拼时间戳 + 两块画布 + 帧数表达式」。**
> 你只给**时长**和 **`chunk_step` 档位**；提示词从 `prompt` 口进来 → **被校验 → 被改写（时间戳 +`prompt_shift_frames` 帧）→ 再送出去**。
> 改写是节点干的，所以**写提示词时不需要知道任何机制**（见文末「时间戳自动改写」）。
> 切不出来、或提示词与切分对不上 → **直接报错拦停**，不让你白跑半小时。
>
> 📌 **喂 LLM 直接用 `templates\auto-prompt-template.md`** —— 给它「片长 + `chunk_step` + 剧情」，
> 它输出**完整六段、时间戳已算好、可直接粘贴**的提示词（不留占位符）。那份里的换算是
> **给定参数下的确定性计算**（不是让 AI 挑段数），并已验证与节点算法逐例一致。

---

## 一、它顶替了哪些节点

| 原节点 | 类型 | 原来干什么 | 现在归谁 |
|---|---|---|---|
| `#48` | `MiniMaxH3HardCutPlan` | 手写切点 | 自动算（段数最少 + 方差最小） |
| `#49` | `MiniMaxH3HardCutValidate` | 校验提示词 | **内置**（同一节点里跑） |
| `#10` | `ResolutionSelector` | 一采画布 0.4MP → 864×480 | `first_megapixels` |
| `#36` | `ResolutionSelector` | 二采画布 1.5MP → 1664×928 | `second_megapixels` |
| `#11` | `PrimitiveFloat` | 时长 8s | `total_seconds` |
| `#13` | `ComfyMathExpression` | `max(5,round(a*24))+(5-…%17)%17` = 17n+5 帧数 | `length` 输出口 |

> 画布算法与 ComfyUI 自带 `ResolutionSelector` **逐像素一致**
> （`scale = √(mp·1048576/(wr·hr))`，再按 `multiple` 取整），所以换过去不会有偏差。

---

## 二、输入

| 输入 | 默认 | 说明 |
|---|---|---|
| **`prompt`** | — | **输入口（socket）** —— 从提示词节点（`PrimitiveStringMultiline`）接线进来。接了就被校验，然后**原样透传**给两个条件节点；**不接**则 `prompt` 口改成六段骨架（`[Shot N]` 时间戳已填好），供你复制出去填剧情 |
| `total_seconds` | 8 | 片长（秒）→ 自动转 **17n+5 帧** |
| **`chunk_step`** | **6** | **chunk 档位**（每段最多几个「17 帧块」）。**这是 LLM 唯一要知道的数字** |
| `first_megapixels` | 0.4 | 一采画布 → 864×480 |
| `second_megapixels` | 1.5 | 二采画布 → 1664×928；**同时用于负载估算** |
| `aspect_ratio` | 16:9 | 两块画布共用（选项取自 ComfyUI 自带表） |
| `multiple` | 32 | 画布取整倍数（H3 要 32），一般不动 |
| `model_name` / `precision` / `release_policy` / `anchor_strength` / `second_pass_audio_policy` | — | 照抄上游 plan 节点，一般不动 |
| `second_pass_sigma0` | 0.30 | 二采 denoise（σ₀），经 `sigma0` 口接 `BasicScheduler.denoise`。**缝幅度 ∝ σ₀** |
| `seam_tolerance_frames` | 17 | `#40` 的 `auto_seam_hunt` 采纳检测结果时允许的偏差（帧） |
| **`prompt_shift_frames`** | **1** | **改写 `prompt` 输出**：把每个 `[Shot N] At MM:SS.mmm` 整体平移 N 帧（正 = 写晚一点）。见文末「时间戳自动改写」 |

---

## 三、输出

| 输出 | 接到哪 |
|---|---|
| `plan` | → 二采执行器 `MiniMaxH3HardCutUpscale`（或上游执行器） |
| `prompt` | → **两个**条件节点的 `prompt`（透传 + 已校验；没接线时是骨架） |
| `report` | → `PreviewAny`；**同一份也直接印在本节点上** |
| `first_width` / `first_height` | → **一采**条件节点的 `width` / `height` |
| `second_width` / `second_height` | → **二采**条件节点的 `width` / `height` |
| **`length`** | → **两个**条件节点的 `length`（17n+5 帧数） |

---

## 四、两条公式（`chunk_step` → 秒）

1. **每段最大帧数 = `chunk_step × 17`**
2. **每段最大秒数 = `chunk_step × 17 ÷ 24` = `chunk_step × 0.708s`**

| chunk_step | 最大帧数 | 最大秒数 |
|---|---|---|
| 4 | 68 | 2.833s |
| 6 | 102 | 4.250s |
| 7 | 119 | 4.958s |
| 8 | 136 | 5.667s |
| 11 | 187 | 7.792s |

> **没有「中间段 / 末段」之分** —— 切出来的**每一段（含末段）都不超过这个上限**。
> **切点推荐 17 帧整数倍**（`秒 = 帧 ÷ 24`：`119 帧 = 00:04.958`，独占帧 = 模型最干净的转镜位）。
> 更细的 token 边界（1-4 帧粒度）也合法——"提前切"（刀退到转镜前最近 token 边）就靠它，如 `115 帧 = 00:04.792`。

---

## 五、段数由「剧情 + 参数」共同决定，不是由 AI 心情决定

两条规则，次序不能反：

1. **镜头数按剧情定**（剧情优先）—— 剧情要 3 个镜头就是 3 个，**不许为了省事合并镜头**。
2. **参数不够就调 `chunk_step`**，不是改剧情。

给定「片长 + `chunk_step`」，节点切几段是**确定的**（算法：n 从 1 试到 10 取第一个可行的 →
切点 snap 到 17 倍数 → 每段 ≤ `chunk_step × 17` 帧 → 尾段 ≥ 17 帧 → 负载 < 236.2）。
想要 N 镜就查 `templates\auto-prompt-template.md` 里的反查表，把 `chunk_step` 填进对应区间。

> 所以喂给 LLM 的不是"筛选偏好"，而是**「给定参数怎么换算成切点」+「镜头数不可妥协」**这两条。
> **别把"段数最少"当成写作规则喂它** —— 那会让它主动删镜头。

**★ 镜头数可以多于段数**：执行器只在自己那几刀上硬切（段边界是 `overlap=0` 的硬断点，必须写在提示词里）；
**提示词里多出来的时间戳 = 同一段内模型自己完成的镜头切换**（和一采整片出多镜同机制）。
例：5 镜 + 2 段 = 2 个硬切 + 段内 3 次模型自切。校验器放行这种写法，只在「执行器的刀没写在提示词里」时报错。

---

## 六、报告长什么样（`report` 口 / 节点显示，同一份）

> ⚠️ **读报告前先分清两个词**（这是最容易误解的地方）：
>
> | 词 | 含义 |
> |---|---|
> | **`cut points`（切点）** | **执行器的分割点** —— latent 在哪里被切成两块。只落在 17 帧网格上 |
> | **实际转镜帧** | **模型自己**换镜头的那一帧，通常比切点**早 ~10 帧**（`cut_offset_frames` 记的就是这个差值）|
>
> 所以 `cut points : 4.958s (frame 119)` 不表示"画面在 4.958s 换镜头" —— 画面在 **4.542s** 就换了。
> 下文的 **ACTION BEATS** 段会把这个差值算好印出来。


```text
=== H3 HARD-CUT AUTO PLAN ===
duration      : 15.083s  (362 frames, 17n+5)
max segment   : 5.667s  ->  136 frames (17-grid); EVERY window (tail included) stays <= this
segments      : 3  (variance 5.6 — lower = more even)
windows:
  #0  frames [   0 ->  119)   119f    0.000s ->   4.958s
  #1  frames [ 119 ->  238)   119f    4.958s ->   9.917s
  #2  frames [ 238 ->  362)   124f    9.917s ->  15.083s
cut points    : 4.958s (frame 119), 9.917s (frame 238)
load estimate : 124f x 1.544MP = 191.5  (SAFE, 0.91x pass anchor)
canvas        : first 864x480 (0.4 MP)  ->  second 1664x928 (1.544 MP)
clip length   : 362 frames (17n+5) -> `length` output

=== H3 HARD-CUT PROMPT CHECK ===
status        : OK   (0 error(s), 1 warning(s))
shots found   : [Shot 1], [Shot 2], [Shot 3]
prompt cuts   : 4.958s (frame 119, 00:04.958), 9.917s (frame 238, 00:09.917)
executor grid : UNEQUAL windows, lengths [119, 119, 124], overlap 0
  #0  frames [   0 ->  119)   119f    0.000s ->   4.958s
  #1  frames [ 119 ->  238)   119f    4.958s ->   9.917s  <- CUT
  #2  frames [ 238 ->  362)   124f    9.917s ->  15.083s  <- CUT

--- WARNINGS ---
  [W1] prompt says '15-second' but the plan is 15.0833s
```

> **只有 `ERRORS` 会拦停**，`WARNINGS` 不会 —— 上面这条只是提示你片长被 snap 到了 17n+5。

| 段 | 讲什么 |
|---|---|
| `HARD-CUT AUTO PLAN` | 分块：片长 / 每段上限 / 段数 / 方差 / **逐窗帧与秒** / 切点 / 两块画布 / 帧数 |
| `HARD-CUT PROMPT CHECK` | 校验：六段齐不齐、`[Shot N]` 编号、`summary` 有没有 "no cuts"、时间戳落不落在真会切的那一帧、**每一刀是否都在提示词里**、负载 |

---

## 七、两种报错

### ① 切不出来 —— 10 段内找不到合规方案

```text
切不出来：15.08s（362 帧）在「每段 ≤ 34 帧 = 1.417s（chunk_step 2）、尾段 ≥ 17 帧、
负载 SAFE」下，10 段内无解。加大 chunk_step，或降画布 MP。
（chunk_step 太小时，末段总会剩下不足一个 17 帧块，执行器建不出窗口。）
```

> 三个约束缺一不可：**每段 ≤ `chunk_step × 17` 帧** · **尾段 ≥ 17 帧**（执行器的最小窗口粒度）·
> **负载 SAFE**。所以 `chunk_step` 调到 1～2 这种极端值会直接报"切不出来"—— 这是诚实答案，
> 不是 bug。

### ② 提示词对不上 —— 不合规直接抛错，跑都不会开始

```text
Hard-cut auto check failed (2 error(s)) — the prompt and the split this node picked
disagree, so the model would not cut where the executor cuts.
...
--- ERRORS (fix these before running) ---
```

> **报错条目就是答案**：它会写明「提示词写 `no cuts anywhere`」「`[Shot 2]` 时间戳是帧 108
> 但执行器切在帧 119」这类具体内容，照着改即可。

---

## 八、喂 LLM：只给这一句约束

> **「每个片段不超过 `chunk_step × 0.708` 秒。」**（chunk_step 6 → 4.25s；8 → 5.667s）

其余全部交给节点：

| 谁负责 | 内容 |
|---|---|
| **节点** | 切几段、切在哪一帧、每段多长、`[Shot N] At MM:SS.mmm` 时间戳 |
| **LLM** | 只写剧情：**每一段演什么、机位怎么变、声音怎么过切点** |

⚠️ **两条禁令**

1. **不要自己编时间戳** —— 时间戳由节点给；AI 写必然对不上 17 帧网格，会被拦停。
2. **不要照「段数最少 / 方差最小」去均匀分段** —— 那是**节点内部挑方案的筛选规则**（第五节），
   **不是镜头规则**。AI 自己决定"分几段、每段多长"就等于跟执行器抢方向盘。

---

## 九、与手写版的分工

| | `MiniMaxH3HardCutPlan`（手写） | `MiniMaxH3HardCutAuto`（自动） |
|---|---|---|
| 切点 | 自己填 `cut_1~cut_4`（秒）或 `cut_frames`（帧，可不等长） | 自动算（段数最少 + 方差最小） |
| 提示词 | 手动抄时间戳 | 自动生成带时间戳的骨架 |
| 校验 | 另接 `MiniMaxH3HardCutValidate` | 内置 |
| 想要**精确控制**切点 | ✅ 用这个 | — |
| 想**少填东西** | — | ✅ 用这个 |

---

## 十、检查清单

1. **`chunk_step` 填了吗** —— 唯一必须你定的数字（其余都有合理默认）
2. **`prompt` 口接线了吗** —— 不接就只出骨架（骨架是给你复制去喂 LLM 的，不是直接用的）
3. **报告 `status` 是 `OK` 吗** —— 不是就照 `--- ERRORS ---` 改，节点会拦停
4. **提示词里 `summary` 没写 "no cuts / continuous take / unbroken"** —— 最常见的拦停原因
5. **`retention_analysis` 列全所有 `[Shot N]`** —— 漏了切完会长相漂移
6. **`load estimate` 是 SAFE** —— 不是就加大 `chunk_step` 或降 `second_megapixels`

---

## 十一、时间戳自动改写（`prompt_shift_frames`）

**问题**：模型不会正好在被告知的时刻转镜。实测偏移 **−10 ~ +1 帧**（随内容 / 种子变），
而执行器的边界**只能落在 token 网格上**（粒度 1~4 帧）——
所以「让模型去够边界那一帧」是**唯一比网格更细的旋钮**。

**做法**：本节点算完切点后，把 `prompt` **输出**里的每个 `[Shot N] At MM:SS.mmm` 整体平移
`prompt_shift_frames` 帧（默认 **+1**）再送出。**输入口那份提示词不被修改。**

```
输入（你写的，声明 = 计划切点）     输出（节点改写后，送进一采 / 二采）
  [Shot 2] At 00:04.958    ──►      [Shot 2] At 00:05.000     (frame 119 → 120)
  [Shot 3] At 00:08.500    ──►      [Shot 3] At 00:08.542     (frame 204 → 205)
  执行器边界仍然是 f119 / f124（按计划切，不受改写影响）
```

**为什么放在节点里，而不是写进提示词**

> 提示词里的每一句"机制说明"都在稀释剧情描述的权重。
> 让**写提示词的 LLM 完全不知道这些**（它只管剧情与镜头），机制由节点在输出前机械地加上。

**规则**

- **只动镜头声明的秒数**；镜内动作节拍（如 `By 00:06.500 …`）**不动**
- summary / shot 行 / soundscape 里**引用的同一个时间戳会一起改**，保持自洽
- 填 `0` = 不改写；**可填负数**（模型偏晚时往回调）
- 写提示词时**照计划切点写**（差 1 帧以内都能过）。若你已自己 +1，节点再 +1 会变成偏 2 帧 →
  报告里会出 `WARNING: after the shift the declared times no longer sit within 1 frame …`

**报告里长这样**

```text
prompt time shift (+1 frame(s)) — only the `prompt` OUTPUT moves:
  the executor still splits where the plan says; the model is asked for the time it
  actually turns at, so the shot change and the window boundary line up:
    00:04.958 -> 00:05.000   (frame 119 -> 120)
    00:08.500 -> 00:08.542   (frame 204 -> 205)
  only declared shot changes move; in-shot beats keep their values, and every
  quote of a moved stamp (summary / shot line / soundscape) is rewritten together.
```

**怎么定这个值**：先跑一次 → 量「实际转镜帧 vs 执行器边界」的差 → 填进 `prompt_shift_frames`
（正 = 写晚）。量法见 README 的 `tools\analyze_cut.py` 一节。
