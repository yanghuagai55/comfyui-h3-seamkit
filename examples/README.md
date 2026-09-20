# 示例工作流

| 文件 | 说明 |
|---|---|
| [`硬切自动版.json`](硬切自动版.json) | **主示例**：15 秒三窗硬切出片链。画布上带一个 `★ 怎么用` 便签，三步开跑 |
| [`硬切自动版-怎么用.md`](硬切自动版-怎么用.md) | 配套操作手册：切点怎么定、参数在哪、排错表 |

## 导入

1. 把 `comfyui-h3-seamkit` 放进 `ComfyUI/custom_nodes/`，**重启 ComfyUI**
2. 把 `硬切自动版.json` 拖进画布（或放进 `ComfyUI/user/default/workflows/`）
3. 照画布上 `★ 怎么用` 便签跑；提示词模板见 [`templates/hardcut-prompt-template.md`](../templates/hardcut-prompt-template.md)

## 模型要求

工作流内的模型 / LoRA 文件名是作者本机环境的名字，**请按自己的环境在加载器里重新选择**：

- **ref2va DiT**（int8 / fp8 量化皆可，v3 硬切针对量化底模调的）
- **Qwen3-VL 文本编码器**（`CLIPLoader` 的 type 选 `minimax`）
- **video VAE (fp16)** + **audio VAE (fp32)** —— 两个别接反
- turbo / 风格 LoRA 按需（`video_steps` 与 LoRA 步数要匹配）

## 硬件参考

显存 8GB 级（RTX 4060 Laptop）实测通过。
负载 = **最长段帧数 × 二采画布 MP**：本机经验线 **≤ 180 通过 / ≥ 191 OOM**
（2026-09-20 加质量 LoRA 后重测；裸模型时代是 210 / 236.2）。
报告里的 `load estimate` 会直接给 SAFE / LIKELY-OOM 判定。
