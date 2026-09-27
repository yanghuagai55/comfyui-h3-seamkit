# comfyui-h3-seamkit

**把 MiniMax H3 分块二采的「接缝」，变成一次有意识的剪辑。**

> `seam` = 接缝，`kit` = 工具箱。硬切（把缝变成剪辑点）只是其中一种策略——缝还看得见时，
> 还有冻结锚定 / 交叉淡化 / 缝窗重去噪一整套处置办法。
> 依赖上游 [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)（GPL-3.0-or-later），运行时调用它的放大器 / 采样 / 条件重锚。
> **许可：GPL-3.0-or-later**（与上游一致）。除 ComfyUI 与 torch 外无额外 pip 依赖。

---

## 它解决什么

分块二采把长片切成窗口各自采样，段与段的 overlap 混合会产生**重影 / 发糊**。
本插件的思路：

1. 每窗独立采样，**overlap 处冻结上一窗的输出**（逐字复制，零重影）；
2. 提示词在切点要求模型换镜头 → 接缝成为**合法的剪辑语法**；
3. hunt / calm 自动决定每条缝**硬切还是锚定**；
4. 副作用是好的：每段更短，**峰值显存更低**——RTX 4060 Laptop 8GB 可以跑 15 秒 / 1.5MP。

## 架构（v1.4）

![架构一览](docs/img/arch.png)

- **上游 Plan**：画布、每段秒数、缓存开关、提示词——只留**影响一采**的东西
- **下游 Pass-2 Plan**：模型三件套、hunt/calm、锚定、拼缝、重去噪、诊断——全部二采参数
- **`calm_policy=auto`（默认）**：逐缝看邻域闹度——动作里的缝硬切藏进混乱，
  动作结束后的平静切点自动改走锚定 overlap。同一条片两种缝各走各的。
- 两个执行节点**零控件**；提示词识别只认**行首 `[Shot N] At + 时间点`**（任意浮点精度）。

## 安装

1. 克隆到 `ComfyUI/custom_nodes/`：
   ```bash
   git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git
   ```
2. 重启 ComfyUI。节点在分类 **`MiniMax H3 Hard Cut`** 下（10 个）。
3. 上游依赖同装：`comfyui-minimax-h3-audio-T8`。

### ⚠ 内存/显存报错？用 `tools/pinned_memory_patch.py`

8GB 卡 + 32GB 内存跑二采，若报 **`hostbuf_grow` / `aimdo memory compile error` /
pinned memory / CUDA OOM** 一类错误，**先用这个工具**再排查别的：

```bash
D:\comfyui\comfyenv\python.exe tools\pinned_memory_patch.py
```

交互菜单一键切换预设（推荐 **3) 新设置（稳定跑通）**：A=0.45 + B=16GiB + `--disable-async-offload`）。
它修改 `comfy/model_management.py` 的 pinned 上限与启动 bat（自动备份，可一键复原），
**改完必须重启 ComfyUI**。

## 快速上手（三步）

1. 搭最小链路（照 `examples/` 里的现成工作流）：
   `上游 Plan → 条件节点 → 一采 #93 → 二采 #40 → 解码`，`下游 Pass-2 Plan → #40`。
2. 写提示词：照 [`templates/prompt-template.md`](templates/prompt-template.md)
   （官方六段骨架 + 4 条注意事项：时间戳任意浮点、禁一镜到底措辞、**blocking 写死**、
   `[Shot N] At` 只在行首用作声明）。
3. 排队。切点时间以**上游报告的 `cut points`** 为准；校验器只警告不拦停
   （时间戳是指示值，overlap 模式由锚定吸收偏差）。

参数细节与原理：[`docs/GUIDE.md`](docs/GUIDE.md)
（第一章工作原理带线框图；第二章按组讲每个控件：画布 / 切点 / 缓存 / 提示词 / 采样 /
模型 / hunt / calm / 锚定 / 拼缝 / 重去噪 / 诊断）。

## v1.4 亮点

- **上下游拆分完成**：执行节点零控件；一采指纹只含上游 12 控件——下游 33 个控件随便改不重采
- **`calm_policy=auto`**：逐缝自动选 jerk 硬切（剧烈）/ calm 锚定（平静），一条片两种缝各走各的
- **提示词识别放宽**：时间戳任意浮点；`[Shot N]` 只认行首声明，句中引用自动警告
- **`upscale_pad_tokens=3` 默认**：治分块尾软化
- 文档全新：`docs/GUIDE.md`（原理 + 参数）+ 单页提示词模板

## 示例成片

两条 15 秒 / 四窗的实测片子，展示缝的两种结局。都在 [`examples/`](examples/)。

### `00095` — 三条缝，只有一条看得见

![00095 帧 265–278 · 缝 272 附近](docs/img/example_095_seam272.jpg)

**[`examples/exp_4v10a_00095.mp4`](examples/exp_4v10a_00095.mp4)**（15.08s / 24fps / 1664×928，含原生音频）

全片三条缝（`85 / 187 / 272`），**最明显的只有一处**：如上图，帧 265 背景还是彩色
（暖色招牌、行人），**帧 268 起背景突然褪成灰白**，到 275 才恢复——这是 overlap
交叉淡化把前后两窗平均后的痕迹，**人物主体（两位女主）的轮廓、位置、颜色全程不突变**，
眼睛捕捉到的只是背景一闪。另外两条缝（85 / 187）落在完全静止段，肉眼几乎不可见。

### `00090` — 85 帧处的直接分割

**[`examples/exp_4v10a_00090.mp4`](examples/exp_4v10a_00090.mp4)**（15.08s / 24fps）

帧 85 处是**直接分割**（上一窗未收尾就硬接下一窗），人物内容整体突变——构图/景别直接换掉。
这种**不需要重跑**：用剪辑软件**直接剪掉那几帧**即可，去掉不影响剧情与节奏。

> ⚠️ **`00090` 没有声音是意外，不是本插件的常态。** 它生成于 2026-09-26，
> 当时工作流的音频 mux 线在更早一次节点清理中被误删（已于 09-27 修复）。
> 音频一直在正常生成，只是没接进最终合成器——`00095` 就是修复后的产物，声音正常。

## 附带工具（`tools/`）

| 工具 | 用途 |
|---|---|
| `pinned_memory_patch.py` | 内存/pinned 上限补丁（**报错先用它**，交互菜单） |
| `seam_report.py` | 逐缝台阶 / 闪烁测量（单机自检 + A/B 对照） |
| `check_cut_flicker.py` | 成片缝与闪屏取证 |
| `repair_seam.py` | 像素域接缝修复 |
| `analyze_cut.py` | 真实切点分析与接触表 |
| `latent_cache.py` | 一采 latent 缓存管理（ComfyUI 之外用） |
| `check_widgets.py` | 控件槽位体检（UI 错位排查） |
| `attention_backend_patch.py` | 注意力后端补丁（A/B 可复现） |

纯计算器（不起 ComfyUI）：`hardcut_math.py`。

## 致谢与来源

### 上游依赖

- **[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)**（作者 **T8mars**，GPL-3.0-or-later）——
  二采执行器在运行时直接调用它的**放大器 / DualClock 采样 / 条件重锚 / 分段装配
  （`_append_video_guarded_overlap`）**，plan 契约与它保持逐字段兼容。本插件建立在它的基础上。
  许可证选 GPL-3.0-or-later 也是因为它：与 GPL 代码链接的衍生作品必须同许可。

### 官方素材

- **MiniMax H3** 的模型与官方提示词规范（六段式 R2V 模板）——`templates/prompt-template.md`
  按官方结构编写，只追加了硬切路线的注意事项。

### 灵感来源

一采 latent 缓存这一层，源头是 **[ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)**
（作者 **AIMixer**）。它把缓存的「身份」定义成*只影响一采的那些东西*——二采参数怎么改都不击穿缓存；
缓存该有「查看 / 清理」两个动作，也是跟它学的。本插件的 `nodes_latent_cache.py` /
`tools/latent_cache.py` 照这套语义写。谢谢。

jerk（三阶差分）剖面指标和「平淡时放弃搜索」，来自 **MAINodes · H3 Jerk Oracle**（matlowai）。
hunt 的「持续性」判据来自 **PERSIST**（arXiv:2608.29287）——真转镜会落在另一个稳态，闪烁会回落。
多 token 锚定（`anchor_tokens`）的思路来自 **StreamingT2V**：单帧条件才是分段不一致的根源。

缝窗重去噪「每一步都把已知区重注入」出自 **RePaint**（Lugmayr et al., CVPR 2022）。
它的仓库是 CC BY-NC-SA 4.0，与本包 GPL 不兼容，所以这里一行代码都没有用它；
真正执行这个机制的是 ComfyUI 核心自带的 `KSamplerX0Inpaint`。

### 协作与审计

- **Zhipu AI（GLM）**：缝窗重去噪（route ①）的首版实现与配套单测（提交 `e9ee3c0`），
  以及 token 变化检测方法的四轮文献调研。
- **外部对抗性审计**：第三方审阅者对本插件的检测栈与策略树做过逐条核查，
  未通过的缺陷（D1–D3 等）均已修复——相关结论记录在提交历史中。

### 实测环境

全部效果数字来自本机 **RTX 4060 Laptop 8GB / 32GB RAM** 上的实测；
`docs/GUIDE.md` 中标注的阈值与经验线（负载线 180/191、`calm_min_gain` 0.15 等）均为实测值。

## 许可

GPL-3.0-or-later（与上游 T8 包一致）。
