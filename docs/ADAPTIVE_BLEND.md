# 自适应接缝（Adaptive Blend）设计与实现记录

> 用途：供**外部审计**（程序合理性核查）。每条机制都给出**代码出处**与**实测数据**，
> 可逐条核对。凡未经实测的推断，均明确标注「推断」。
> 仓库：`comfyui-h3-seamkit` ｜ 分支：`feat/adaptive-blend` ｜ 基线 tag：`v1.0.0-baseline-20260921`

---

## 0. 一句话

硬切（overlap=0）把接缝交给"两条独立采样的结果硬拼"；本方案允许**把边界挪到片子里最平缓的位置，并让新段以旧段的输出为锚**，使接缝既不落在内容剧变处，也不必承担两条结果的风格断层。

---

## 1. 问题与目标

### 1.1 现象（实测）

MiniMax H3 的二采是**分窗放大**：把一采 latent 按时间切成若干窗，逐窗采样后拼回整片。
当窗口边界是"硬切"（`temporal_overlap_frames = 0`）时，相邻窗**完全独立采样**，
在边界处表现为可见的接缝（帧差/清晰度突变）。

### 1.2 关键实测：偏差来自模型，不是坐标换算

| 模型（同一套代码/同一时间轴） | 大小 | 提示词切点与模型实际转镜的偏差 |
|---|---|---|
| `Minimax-h3_Singularity_ref2va_v1.3_int8` | 31.67 GB | **1 帧** |
| `minimax_h3_ref2va_int8_convrot`（Comfy-Org） | 31.70 GB | 5-7 帧 |
| `DasiwaMinimaxH3_dasiwaHybridV2_int8` | 19.53 GB | 14-17 帧 |

**已排除**的假设（均经计算或源码核对否定）：

- **时间轴非线性** —— 否。`comfy/ldm/minimax/model.py`：
  `_video_t_spans(n) = [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] ...]`，
  `_video_t_grid(n, origin) = origin + exclusive_cumsum(spans)`；
  因 `sum(spans) = 5/3 × 总帧数`，**累积和与线性等价**。实测：帧 85 → 141.67 内部单位，
  两种算法差 **0.0**。
- **24fps ↔ 40Hz 换算误差** —— 否。`FRAME_RESCALE = 5/3`（`model.py:31`），
  取整误差 ≤ 0.6 帧。
- **提示词被截断** —— 否。`Qwen3VLSDTokenizer(..., max_length=99999999)`（`qwen3vl.py:151`），
  本地不截断（官方"≤7000 字符"是托管 API 限制）。

→ **结论：偏差是模型行为差异，无法通过坐标/补丁消除；只能在"接缝落在哪里"上做文章。**

### 1.3 目标

1. 边界**可以移动**（不再死守提示词写的时间戳）；
2. 移动的目标是**内容最平缓的位置**；
3. 接缝处**两条结果的风格不冲突**（用锚定而非硬拼）。

---

## 2. 上游机制（事实，附出处）

以下均为 `custom_nodes/comfyui-minimax-h3-audio-T8/h3_t8/chunked_two_pass_upscale_advanced.py` 的实际代码。

### 2.1 分段：`compute_temporal_segments(video_tokens, chunk_length, overlap)`

```python
if chunk_length <= 0 or overlap < 0 or chunk_length <= overlap:
    raise ValueError("chunk_length must be positive and larger than overlap")
hop = chunk_length - overlap
```
- **`hop = chunk − overlap`**：第 i 窗的起点 = `i × hop`，即**上一窗末端回退 overlap 帧**；
- 首窗必须 `start = 0`；
- 约束：**`0 ≤ overlap < chunk`**（本方案的钳制依据）。

### 2.2 锚定：`anchor_conditioning(conditioning, previous_video, start_frame, strength)`

```python
token = tokens_for_frames(start_frame)
if token >= previous_video.shape[2]:
    raise ValueError("previous chunk does not reach the next chunk anchor")
anchor = {"resolved_frame_index": 0,
          "latent": previous_video[:, :, token:token + 1].contiguous()}
updated["minimax_keyframes"] = [anchor, *keyframes]
updated["minimax_visual_cond_noise_aug"] = max(0.0, min(1.0, float(strength)))
```
- 把**上一窗在该位置的 1 个 token**，作为**本窗第 0 帧的关键帧条件**注入；
- **硬切时 `token == 上一窗长度` → 必然 ≥ → 抛错**：这就是"硬切不能锚定"的根本原因；
- 强度由 `anchor_strength`（→ `minimax_visual_cond_noise_aug`）控制。

### 2.3 拼接：两个版本

| 函数 | 重叠区处理 |
|---|---|
| `_append_video(acc, chunk, start_token)` | `overlap = acc长度 − start_token`，对重叠区做 **`_crossfade`**（混合） |
| **`_append_video_guarded_overlap(acc, chunk, start_token, locked_overlap_tokens)`** | `locked` 段**保留旧结果**、`transition` 段**取新采样**，然后 `torch.cat` |

本方案**使用后者**（可控，且能区分"冻结"与"过渡"）。

### 2.4 条件裁剪：`reanchor_conditioning(conditioning, start_frame, end_frame, spatial)`

按窗裁剪 `minimax_keyframes`（即 AddGuide 锚点会随分段自动裁剪）。

### 2.5 语义小结（重要）

> **overlap 不是"把两段画面混合"**，而是：
> **新窗回读旧窗尾部 N 帧，并把第 0 帧钉在旧窗的对应 token 上，只生成后面的新内容。**
> overlap 越"粘"，新窗自由度越低；`0` 即完全独立（硬切）。

---

## 3. 算法

### 3.1 输入

- `planned_cuts`：计划切点（帧，来自 `#56`/`#48` 的 plan）；
- `profile`：latent 变化剖面 `_latent_change_profile(video)`，逐 token 给出
  `(token_idx, local_score, global_score)`；**global 分越低 = 该处越平缓**；
- `hunt` 结果：`_align_to_profile` 已给出每个 planned cut 的
  `boundary_frame / ratio / measured_turn_frame`。

### 3.2 判定与动作

```text
for cut in planned_cuts:
    entry = hunt_entry(cut)

    # ① 可靠：hunt 的边界与【模型实际转镜】的残差在容忍内 → 硬切
    #    ★ 判据是 |boundary − measured|，不是 |measured − cut|：
    #      hunt 已经把边界吸附到独占帧网格上，所以"计划偏 15 帧"完全可能
    #      吸附后只剩 2 帧残差（计划 187 / 实测 202 → 边界 204）。
    #      只有"吸附后仍对不上"才算不可靠。
    if entry.boundary is not None and |entry.boundary − entry.measured| <= seam_tolerance:
        emit(boundary = entry.boundary, overlap = 0)
        continue

    # ② 不可靠（未采纳 / 偏差过大）→ 平缓搜索
    candidates = [t for t in tokens_in(cut ± calm_search_window)
                  if is_exclusive_frame(t)
                  and far_enough_from(other_boundaries, MIN_SEP)]
    if not candidates:
        emit(boundary = cut, overlap = 0)          # 退化：保持计划，不动
        continue

    P = argmin(global_score(t) for t in candidates)   # 最平缓的独占帧
    emit(boundary = P, overlap = calm_overlap_frames)
```

### 3.3 窗口与拼接（由 3.2 的结果驱动）

- **每一缝独立**决定 overlap；第 i 窗起点 = `上一窗末端 − overlap_i`；
- 有 overlap 的窗调用 `anchor_conditioning`，`overlap = 0` 的窗**不调用**
  （调用会因 2.2 的检查抛错）；
- 拼接统一走 `_append_video_guarded_overlap`（`locked = transition = overlap`：
  回读部分完全保留旧结果，不做混合）。

### 3.4 不变量（可写测试）

| # | 不变量 |
|---|---|
| I1 | **首窗起点恒为 0**，且 `overlap = 0` |
| I2 | 任一窗的 `overlap < 该窗长度`，且 `overlap < chunk` |
| I3 | 相邻边界间距 ≥ `MIN_SEP`（当前 34 帧），**不产生 17 帧级碎段** |
| I4 | 边界必落在**独占帧**（17 的倍数）上 |
| I5 | `overlap = 0` 时行为与本方案之前**逐比特一致**（未改动路径） |
| I6 | 输出总帧数 = 输入总帧数（不增不减） |
| I7 | 未采纳的切点**必须回退为计划切点**，不得被丢弃（历史缺陷，已修 `3853479`） |

---

## 4. 参数

| 参数 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `overlap_frames` | `#56` / `#48` | 0 | 全局段间重叠（帧；UI step=17 但**执行器不做倍数取整**，非倍数值经 token 边吸附（1–4 帧分辨率）生效；送到采样器前钳制到 `< chunk`） |
| `seam_tolerance_frames` | `#56` | **17**（复用，未新增参数） | **一值两用**：① hunt 采纳半径（检测到的转镜离计划点超过它即判误检）② **硬切 / overlap 的判据**——**hunt 吸附后的边界与模型实际转镜的残差**超过它 → 该缝转 overlap。注意它**不是**「实测与计划」的差：吸附本身会消掉大部分计划偏差（计划 187 / 实测 202 → 边界 204，残差仅 2）。想更敏感就调到 **3~4** |
| `auto_calm_search` | `#56`（新增） | `false` | 启用自适应平缓搜索 |
| `calm_search_window` | `#56`（新增） | 34 | 搜索半径（帧），`cut ± window` 内找最平缓的独占帧 |
| `calm_overlap_frames` | `#56`（新增） | 17 | 平缓缝使用的 overlap |
| `anchor_strength` | plan | 0.999 | 锚定强度（传给 `minimax_visual_cond_noise_aug`） |
| `locked_overlap_tokens` | plan | = overlap（帧） | 重叠区中"完全保留旧结果"的量；上游按 **token** 计，执行器负责帧→token 换算（`×5/17`） |

**负载**：`(最长窗 + 该窗 overlap) × 二采画布 MP`。实测 15s/1.544MP：
`overlap 0 → 153 SAFE`；`17 → 178.5 SAFE（贴 180 线）`；`34 → 204 LIKELY-OOM`。

---

## 5. 失败模式与边界情况

| 情况 | 处理 |
|---|---|
| latent 全片无平缓点（全在运动） | 退化为"保持计划切点 + 硬切"，并记日志 |
| 平缓点离相邻边界过近（< MIN_SEP） | 从候选中剔除；若候选空则退化 |
| 片长太短，切一刀后尾段 < 48 帧 | 上游拒绝建窗 → 退回单段（实测：5s 片 + `chunk_step=5` 即此情况） |
| overlap ≥ chunk | **钳制到 `chunk − 1`**（要求：可随便调，但不能把整窗吃掉），并打印 `requested → effective` |
| 某缝被 hunt 拒绝 | **回退为计划切点**（不得丢弃；丢弃会导致两窗合并 → 负载翻倍 → OOM） |
| `overlap = 0` | 与历史行为完全一致（不调用 anchor，不使用 guarded 拼接） |

---

## 6. 验证方法（可复现）

1. **单元级**：对 `explicit_segments` 构造 362 帧 + 切点 `[85,187,272]`，
   断言 §3.4 的 I1–I4；`overlap=0` 与历史输出逐元素比较（I5）。
2. **日志级**：每次运行输出
   `[HardCut] planned_cuts=... boundary_frames=...[offset] segments=... lengths=...`、
   `[HardCut]   overlap: requested Xf -> effective Yf`、
   `[HardCut]   cut planned=A -> boundary=B (offset=±Nf, moved=..., measured=..., ratio=...)`。
3. **成片级**：对成片做帧差/清晰度分析，确认接缝处**不再出现单帧突变**
   （工具：`tools/check_cut_flicker.py`）。
4. **模型侧对照**：同一提示词、同一 seed，切换 `auto_calm_search` 开关各跑一次对比。

### 6.1 常驻回归清单（每项都必须保持 PASS）

**基础用例（6）**

| # | 场景 | 期望 |
|---|---|---|
| A1 | hunt 可靠（边界落在 ±deviation 内） | 硬切，`overlap = 0` |
| A2 | hunt 被拒（`boundary = None`） | 选中窗口内最平缓的独占帧 |
| A3 | hunt 偏差超阈值 | 同样触发搜索 |
| A4 | profile 为空 | 保持计划切点，硬切 |
| A5 | 候选与相邻边界过近 | 被 `min_sep` 剔除 |
| A6 | 边界数量 = 切点数，且全为 17 的倍数 | 恒成立 |

**反例回归（3）—— 来自外部验收，必须保持**

| # | 反例构造 | 修复前（错误） | 期望（正确） |
|---|---|---|---|
| **R1** | planned `[204, 238]`（相距 34），204 的 hunt 可靠但漂到 **221**，238 走平缓搜索 | `[221, 238]` → **17 帧碎窗** | 间距 ≥ 34（实测 `[170, 238]`） |
| **R2** | planned `[85, 153]`（相距 68），中点 **119** 恰为全局最平缓帧（对两边各 34，两边都"够远"） | `[119, 119]` → 去重后**窗合并、负载翻倍** | 两缝不得选同一帧（实测 `[119, 187]`） |
| **R3** | `locked_overlap_tokens` 传参 | 传帧值 → `min(帧, token)` 恒 = token → **恒全锁** | 帧 → token 换算（17 帧 = 5 token） |

> R1/R2 是"两条缝互相影响"的边界：**候选过滤必须针对「已输出的最终边界 + 其余计划切点」**，
> 而不是只看计划切点。R3 是单位错位。三者均已修复（commit `384528f`）。


---

## 7. 已知限制

- **算法只在 latent 上工作**：它选的是"最不容易看出缝的位置"，**不改变模型自身的切点偏差**（§1.2）。
- **平缓 ≠ 视觉上无接缝**：静止画面里，接缝处的色调/细节差异反而更容易被察觉；本方案靠 anchor
  对齐风格来补偿，但**未做视觉验证**（待测）。
- **链式衰减**：启用 overlap 即回到上游的"锚定链"，长片后段可能逐段漂移
  （上游 README 记录 join1 ≈ 0.9、join2 ≈ 0.65）。
- **逐缝 overlap 需要自建分段**（上游 `compute_temporal_segments` 是等长 hop），
  本方案用自己的 `explicit_segments` 承担。

---

## 8. 改动清单

| commit | 内容 |
|---|---|
| `v1.0.0-baseline-20260921` (tag) | 基线：硬切链路冻结 |
| `56f00e9` | 执行器接受 `temporal_overlap_frames`（分段回退 + anchor + guarded 拼接） |
| `44142ca` | plan/负载/节点全链路暴露 `overlap_frames`；钳制到 `< chunk` |
| `3853479` | 修复：hunt 未采纳的切点回退为计划切点（防两窗合并） |
| `8c7be9f` | 日志补充 `offset` |
| （本次） | 本文档 + 自适应平缓搜索 |

---

## 9. 外部验收（2026-09-21）与修复

**验收报告**：`docs/ADAPTIVE_BLEND_ACCEPTANCE.html`（23 项运行时测试，结论"有条件通过，4 项缺陷，其中 1 项高"）

| 缺陷 | 内容 | 修复 | 回归证据 |
|---|---|---|---|
| **D2（高）** | `overlap_frames` widget **未接入** `build_hardcut_plan`（提交信息与本文 §4 失实） | 两处调用点 + 两处 `geometry` 硬编码 `0` 全部改为传参 | `grep` 4 处生效；`nodes.py:336` / `:696` |
| **C1（中）** | `min_sep` 只对比**计划切点** → 可靠切点的 hunt 边界不做间距检查 → 可产生 17 帧碎窗 | 过滤对象改为「**已输出的最终边界 + 其余计划切点**」 | 反例 `[204,238]` → `[170,238]`（间距 68）✓ |
| **C2（中）** | 两条缝可选中**同一平缓帧** → 去重后窗合并、负载翻倍 | 同上（`refs` 含已选边界） | 反例 `[85,153]` → `[119,187]`（无重复）✓ |
| **C3（低）** | `locked_overlap_tokens` 传的是**帧**值，上游按 **token** 计 → 恒等于"全锁"，旋钮失效 | 传参处换算 `frames × 5 // 17` | 17 帧 → 5 token ✓ |
| 文档失实 | ① `seam_tolerance_frames` 默认（实为 17）② "向下取整到 17 的倍数"（执行器无此逻辑）③ `locked_overlap_tokens` 单位 | 三处已按代码改正 | — |

**流程教训（已固化）**：上一次改动**只断言了 `count()`，没有断言替换结果**，
导致 `overlap_frames` 的接入静默失败而提交信息仍然声称完成。
现在所有脚本化编辑一律断言「**旧文本消失 且 新文本出现**」，否则报错退出。

**当前状态**：6 项原始单测 + 3 项反例回归 + 4 项残差判据 **全部通过**（见 §6.1）；
成片级（视觉）验证仍为**开放项**（§7）。
