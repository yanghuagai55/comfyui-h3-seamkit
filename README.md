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
4. 副作用是好的：每段更短，**峰值显存更低**——RTX 4060 Laptop 8GB 可以跑 15 秒 / 1.3MP。

## 架构（v1.4）

```text
   上游 Plan（12 控件）──► 一采 #93（可缓存，零控件）──► 二采执行器 #40（零控件）──► 解码
        │                        ▲                            ▲
        └─ plan/prompt/尺寸      │                            │ pass2_plan
                                 │                    下游 Pass-2 Plan（33 控件）
   一采缓存：上游任何改动 → 指纹变 → 重采；下游 33 个控件随便改，永不重采
```

- **上游 Plan**：画布、每段秒数、缓存开关、提示词——只留**影响一采**的东西
- **下游 Pass-2 Plan**：模型三件套、hunt/calm、锚定、拼缝、重去噪、诊断——全部二采参数
- **`calm_policy=auto`（默认）**：逐缝看邻域闹度——动作里的缝硬切藏进混乱，
  动作结束后的平静切点自动改走锚定 overlap。同一条片两种缝各走各的。
- 两个执行节点**零控件**；提示词识别只认**行首 `[Shot N] At + 时间点`**（任意浮点精度）。

## 安装

1. 克隆到 `ComfyUI/custom_nodes/`：
   ```bash
   git clone https://github.com/<you>/comfyui-h3-seamkit.git
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

## 测试

```bash
python ../_hardcut_work/seamfix/e4_hardcut_policy_test.py   # 策略树单测
```

## 许可

GPL-3.0-or-later（与上游 T8 包一致）。
