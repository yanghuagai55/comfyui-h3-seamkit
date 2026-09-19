# 自动硬切节点 · MiniMaxH3HardCutAuto

> 来源：插件 `comfyui-h3-hardcut`（`hardcut_math.py` 的 `auto_plan` / `resolution_for`）
> 提示词规范：`refs\H3-R2V提示词模板-官方.md`　·　负载锚点实测：RTX 4060 Laptop 8GB，二采 1664×928（1.544MP）

> **一个节点顶替「规划 + 校验 + 拼时间戳 + 两块画布 + 帧数表达式」。**
> 你只给**时长**和 **`chunk_step` 档位**；提示词从 `prompt` 口进来、原样出去（顺带被校验）。
> 切不出来、或提示词与切分对不上 → **直接报错拦停**，不让你白跑半小时。
>
> 📌 **喂 LLM 直接用 `templates\auto-prompt-template.md`** —— 六段骨架 + `chunk_step` 换算，
> 不含任何节点说明。若只想从本文截取：给**第四节**（公式表）+ **第八节**（一句约束）即可，
> 其余是使用/排查说明，喂进去反而会让它自作主张分镜头。

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
> **切点仍是 17 帧整数倍**（`秒 = 帧 ÷ 24`：`119 帧 = 00:04.958`）。

---

## 五、切分是节点自己算的（**这不是 AI 的规则**）

> **「段数最少 + 方差最小 + 负载 SAFE」是节点挑方案的筛选规则，不是你写镜头的规则。**
> AI **不需要、也不应该**照它去均匀分段 —— 它不知道 17 帧网格，自作主张写出来必然对不上。
> **写提示词要遵守的只有第八节那一句。**

（仅供排查"为什么切成这样"时参考：n 从 1 试到 10 取第一个可行的 → 切点 snap 到 17 倍数
→ 每段 ≤ `chunk_step × 17` 帧 → 负载 < 236.2。**这段喂 LLM 时请删掉。**）

---

## 六、报告长什么样（`report` 口 / 节点显示，同一份）

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
| `HARD-CUT PROMPT CHECK` | 校验：六段齐不齐、`[Shot N]` 编号、`summary` 有没有 "no cuts"、时间戳落不落在真会切的那一帧、镜头数 vs 段数、负载 |

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
