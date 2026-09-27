# 示例工作流 · workflow.json

**v1.4 新架构**：上游 Plan（12 控件）→ 一采（零控件，可缓存）→ 二采执行器（零控件）← 下游 Pass-2 Plan（33 控件）。
15 秒 / 1.3MP / 四窗，`calm_policy=auto`（逐缝自动选硬切或锚定）。

## 快速上手（四步）

1. **导入**：把 `workflow.json` 拖进 ComfyUI 画布（插件已装并重启过）。
2. **换模型**：加载器里的模型 / LoRA 是作者本机的名字，按自己的环境重选——
   ref2va DiT（int8/fp8 皆可）· Qwen3-VL 文本编码器（`CLIPLoader` type = `minimax`）·
   video VAE (fp16) + audio VAE (fp32) **别接反** · turbo/风格 LoRA 按需。
3. **写提示词**：照 [`templates/prompt-template.md`](../templates/prompt-template.md)
   （官方六段骨架）。时间戳**任意浮点**即可，对不齐只会警告；
   **blocking（站位/朝向/视线）写死**并声明全片统一——这是接缝不跳的关键。
4. **排队**。切点由 `target_segment_seconds` 算出（报告的 `cut points` 可直接抄）；
   出片后想调二采，**只动下游 Pass-2 Plan**——改它不重采一采。

## 两个规划节点的分工

| 节点 | 管什么 | 改了会重采一采吗 |
|---|---|---|
| **上游 Plan**（12 控件） | 片长 / 画布 / 每段秒数 / 缓存 key / 提示词 | **会** |
| **下游 Pass-2 Plan**（33 控件） | 模型 / hunt / calm+overlap / 锚定 / 拼缝 / 重去噪 / 诊断 | **不会** |

## 硬件参考

8GB 级（RTX 4060 Laptop）实测通过。
负载 = **最长段帧数 × 二采画布 MP**：本机经验线 **≤ 180 通过 / ≥ 191 OOM**
（2026-09-20 加质量 LoRA 后重测）。报告里的 `load estimate` 直接给 SAFE / LIKELY-OOM 判定。

## 报错先看这里

- **pinned / hostbuf / `aimdo memory compile error` / CUDA OOM** 一类 →
  运行 [`tools/pinned_memory_patch.py`](../tools/pinned_memory_patch.py)，选预设
  **3) 新设置（稳定跑通）**，改完**重启 ComfyUI**。
- 反复崩在采样中途 + 系统出现 Kernel_117/141 报告 → GPU 驱动级故障（本机已知问题），
  重启 ComfyUI 重试；频繁复发考虑回滚显卡驱动。
- 提示词校验警告 → 按报告提示改（时间戳是指示值，不必精确）。

原理与每个参数的详解：[`docs/GUIDE.md`](../docs/GUIDE.md)。
