# 参数使用指南 · comfyui-h3-seamkit

> 适用版本 `v1.2`（含自适应接缝 / jerk 判据 / 持久性判据 / 秒制分段）。
> 本文只讲"**该改哪个、改成多少**"；机制推导在 `ADAPTIVE_BLEND.md`，检测体系在 `DETECTION_STACK_REVIEW.html`。

---

## 0. 先理解两件事

**① 参数分两区。** 节点上直接显示 10 个（**主界面**），其余的折在节点标题下的**展开箭头**里（**高级区**）。**日常只动主界面那 10 个**就够了。

**② 一次只改一个，跑完看 report。** 每次运行 `#56` 都会输出一段报告，里面这两行是决策依据：

```
load estimate : 119f x 1.300MP = 154.7  (SAFE, 0.86x pass anchor)
[HardCut]   calm: cut 187: hunt unreliable (None) -> JERK peak 153 (burst 3.99), hard cut
```

- **`load estimate`** —— **负载**。看 `SAFE` / `BORDERLINE` / `LIKELY-OOM`。**超了先降 `target_segment_seconds` 或 `second_megapixels`。**
- **`calm:`** —— **每条缝的决策**。它告诉你边界挪去哪了、为什么挪。

---

## 1. 五类片子，怎么设

### ① 文戏 / 对话戏（固定机位，人物小幅动作）

| 参数 | 建议 | 为什么 |
|---|---|---|
| `target_segment_seconds` | **5.0** | 段长不是瓶颈，够用就行 |
| `seam_tolerance_frames` | **4** | 实测差 3~4 帧以上缝就看得出来，设小让程序早点接管 |
| `auto_calm_search` | **true** | 让程序自己找缝 |
| `calm_policy` | **`calm_overlap`** | 文戏里"平缓处"遍地都是，挪过去缝住最稳 |
| `profile_camera_compensate` | **false** | 固定机位，开了白费时间 |
| `calm_abstain_below` | **0** | 文戏基本不会"整片平坦"，不用它 |

**预期**：缝会落在"人物静止"的帧上，配合锚定基本看不出来。

---

### ② 武戏 / 动作戏（运动剧烈，手持）

| 参数 | 建议 | 为什么 |
|---|---|---|
| `target_segment_seconds` | **5.0** | 运动大时负载本来就更吃紧，别贪长 |
| `seam_tolerance_frames` | **4** | 同上 |
| `auto_calm_search` | **true** | — |
| **`calm_policy`** | **`jerk_hardcut`** | **切在运动最剧烈处，靠运动掩蔽 + "模型本来就在这儿糊"，两边清晰度都低** |
| `profile_camera_compensate` | **false** | 手持抖动**不是**整体平移，整数位移补偿帮不上（甚至更糟） |
| `hunt_persistence` | **true**（默认） | 排除"打一下就回"的假信号 |

**实测**：用 `jerk_hardcut` 跑武戏，**三条缝的帧差降到邻域的 1.06 / 1.06 / 1.18 倍**——**孤立尖峰消失**。

---

### ③ 运镜戏（推拉摇移、跟拍）

| 参数 | 建议 | 为什么 |
|---|---|---|
| **`profile_camera_compensate`** | **true** | **关键**：不补偿的话，相机平移会被读成"剧烈运动"，程序会避开本该可用的边界 |
| `calm_policy` | **`calm_overlap`** | 补偿后运镜段被读成"静止"——**那是理想的切点**（内容连续、可锚定） |
| `target_segment_seconds` | **5.0** | — |
| `seam_tolerance_frames` | **4** | — |

**代价**：逐帧做 49 次位移搜索（Python 循环），**token 多时有几十秒延迟**。**先单独开一版确认效果**，再决定要不要常态开。

**实测**：合成纯平移输入下，`mean |d1| ratio` 从 **1.00 → 0.00**（完全消除）。

---

### ④ 特效 / 战斗（高频闪烁、高对比、星芒）

| 参数 | 建议 | 为什么 |
|---|---|---|
| **`hunt_persistence`** | **true**（默认） | **关键**：闪烁和真转场在"局部变化"上长得一样，只有持久性能区分 |
| `calm_policy` | **`jerk_hardcut`** | 特效段运动大，掩蔽天然强 |
| `target_segment_seconds` | **5.0** | 特效段负载偏高，别加长 |
| `profile_camera_compensate` | false | 特效不是相机位移 |

**实测**：同一位置、同样的局部变化量下，**真转场 persistence = 0.997，闪烁 = 0.319**（后者与周围无法区分）。

---

### ⑤ 长片 / 多段（> 15 秒）

| 参数 | 建议 | 为什么 |
|---|---|---|
| `target_segment_seconds` | **5.0**（不要更大） | **段长是负载的乘数**，15s 片开 7s 段就接近红线 |
| **看 report 的 `load estimate`** | **> 160 就该降档** | 红线在 180；`overlap 17` 时代实测 178.5 已经是上限 |
| `hunt_persistence` | true | 段越多，假阳性越贵 |
| `calm_abstain_below` | 可试 **1.2** | 整片都很平的时候让程序别瞎折腾 |

**分段数量的直觉**：15 秒的片子，`5.0s` 段 → 4 段；`4.25s` 段 → 4 段（尾段短）；**段数少比段数多省一半时间**。

---

## 2. 全参数解读表

### 主界面（10 个，日常只用这些）

| 参数 | 默认 | 作用 | 什么时候改 |
|---|---|---|---|
| **`prompt`** | — | 提示词入口 | 每次都改 |
| **`total_seconds`** | 8.0 | 片长（秒，可填任意小数） | 每片必设 |
| **`target_segment_seconds`** | 5.0 | **想要的最长分段（秒）**。内部取最接近的 17 帧档位（4.25s → 102 帧） | **负载超了就降它**（首选） |
| **`first_megapixels`** | 0.4 | 一采画布（低清，决定"模型看到什么"） | 一般 0.4，调大更贵 |
| **`second_megapixels`** | 1.5 | 二采画布（成片分辨率） | **负载超了的第二选择**；1.3 比 1.5 省约 13% |
| **`aspect_ratio`** | 16:9 | 画幅 | 按需 |
| **`model_name`** | — | 放大器底模 | 装机后基本不动 |
| **`seam_tolerance_frames`** | 17 | **容忍半径**：hunt 吸附后的边界与"模型实际转镜"的残差超过它 → 该缝转给搜索处理 | **建议 4**（实测差 3~4 帧就看得出来） |
| **`auto_calm_search`** | false | 自适应搜索总开关（**需要 `#40` 的 `auto_seam_hunt` 一起开**） | **建议 true** |
| **`calm_policy`** | `calm_overlap` | 缝的策略：`calm_overlap`＝挪到平缓处＋锚定；`jerk_hardcut`＝切在剧烈处＋硬切 | **武戏/特效 → `jerk_hardcut`；文戏/运镜 → `calm_overlap`** |

### 高级区（13 个，装好就不用管）

| 参数 | 默认 | 作用 | 什么时候改 |
|---|---|---|---|
| `multiple` | 32 | 画布对齐到 32 的倍数（H3 要求） | 别动 |
| `precision` | bf16 | 放大器计算精度 | 显存紧可试 fp16 |
| `release_policy` | clear_after | 一次性模型的释放时机 | 别动 |
| `anchor_strength` | 0.999 | 锚定强度（overlap > 0 时生效） | 接缝仍明显可试 0.95 |
| `second_pass_audio_policy` | joint_av_preserve_input | 二采音频策略 | 别动 |
| **`second_pass_sigma0`** | 0.30 | **二采 denoise（同时有 `sigma0` 输出口接 BasicScheduler）** | **接缝幅度 ∝ σ₀**：调小缝更淡（0.20~0.25），代价是二采细节变少 |
| `overlap_frames` | 0 | 全局段间重叠（帧）。**用 calm 搜索时它只是兜底值**（每条缝用自己的） | 关掉 calm 搜索时才需要它 |
| `calm_search_window` | 34 | 平缓/剧烈搜索的半径（帧） | 0 = 不挪边界、只把该缝转成 overlap |
| `calm_overlap_frames` | 17 | `calm_overlap` 策略给每条缝的锚定重叠量 | 接缝粘不住可加到 34（**注意负载**） |
| **`profile_camera_compensate`** | false | 算剖面之前先做相机位移对齐 | **运镜片开 true**；固定机位别开（慢） |
| `profile_reduce` | mean | 空间聚合方式（mean / max / top-decile） | **收益未证实**，保持 mean |
| `calm_abstain_below` | 0.0 | 放弃门：jerk 对比度低于它 → 整片不搜索 | 整片很平时试 **1.2** |
| **`hunt_persistence`** | **true** | 用"持久性"排序候选：真转场＝停在新状态；闪烁/抖动＝回到原状态 | **默认开**；关掉退回旧的纯局部变化排序 |

---

## 3. 三个最容易踩的坑

**① `auto_calm_search` 开了但没生效**
**必须同时开 `#40` 的 `auto_seam_hunt`** —— 检测剖面（`|d1|`/`|d3|`/persistence）是它产出的，没有它整条链路不启动。

**② 提示词没写 `At MM:SS.mmm` 也能跑**
但你开着 `auto_calm_search` 时，校验器**只给 warning 不报错**——**那是设计如此**：
提示词可以写"约 5 秒切"，**实际边界由程序在附近自己找**。

**③ 改了代码没重启**
插件是进程启动时加载的。**每次改完必须重启 ComfyUI**，否则跑的还是旧逻辑。

---

## 4. 报告怎么读（一次完整运行）

```
[HardCut]   calm: cut 85: hard cut, residual 0f <= 4f (boundary 85 vs plan 85)
                     ↑ 这条缝：hunt 采纳了，残差 0，普通硬切
[HardCut]   calm: cut 187: hunt unreliable (None) -> JERK peak 153 (burst 3.99), hard cut
                     ↑ 这条缝：hunt 没找到转镜 → 按 jerk_hardcut 挪到 153
[HardCut] planned_cuts=[85, 187, 272] boundary_frames=[85, 153, 272]
          segments=4 lengths=[85, 68, 119, 90] longest=119f
                     ↑ 最终边界与每段长度（总和 = 总帧数，说明没有重叠）
[HardCut]   TE unloaded before sampling: MiniMaxH3TEModel
                     ↑ 文本编码器已释放（省 14.6GB 内存）
[HardCut]   seg 1 done: alloc 0.01GB  reserved 1.00GB  device-free 0.00GB
[HardCut]   cache released: +0.84GB free
                     ↑ 显存探针：alloc 恒定 = 无泄漏；清缓存稳定腾出约 1GB
```

**同一份数据在 `#40` 的 report JSON 里也能看到**，其中 `seam_hunt.aligned[].top_candidates` 是 **`[帧号, 局部变化, 持久性]`** 三元组，可用来核对程序的判断。
