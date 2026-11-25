# 示例工作流总览（v2.7）

本目录提供三套面向 8GB 消费级显卡（RTX 4060 实测 0 OOM）的 **v2.7 单窗主力工作流**，以及早期多窗硬切归档示例。

## 一、v2.7 三套主力单窗工作流（推荐直接拖入 ComfyUI）

| 文件 | 二采画布 | 最终分辨率 | 10s 全程耗时 (4060 8GB) | 推荐用途 |
|---|---|---|---:|---|
| **[`workflow_singlewin_selflora_13mp_vsr3k.json`](workflow_singlewin_selflora_13mp_vsr3k.json)** | `1.30 MP` (`1568×864`) | **`3136×1728` (真 3K)** | **`~10–11 分钟`**（2步 `~8.5m`） | **★ 日常主力甜点王**：双路 `ChunkedSage` 三刀提速 + 去油胶片 + 自训保锐 + 一步 `RelayVSR x2 (detail=0.50)` |
| **[`workflow_singlewin_selflora_1080p.json`](workflow_singlewin_selflora_1080p.json)** | `1.98 MP` (`1920×1088`) | **`1920×1088` (纯原生)** | **`~15 分钟`**（2步 `~12m` / 5s `9m`） | **纯原生 1080p 直出**：移除 VSR 分支（0 误触 4K OOM 风险），含三刀提速 + 去油双槽 |
| **[`workflow_singlewin_sharp_vsr2k.json`](workflow_singlewin_sharp_vsr2k.json)** | `1.00 MP` (`1376×768`) | **`2752×1536` (真 2.7K)** | **`~8.5–9.5 分钟`** | **经典最速 2.7K**：H3 原生 RoPE 甜点 + `LMS` + `RelayVSR x2 (detail=0.60)` |

### 通用上手要点
1. **模型替换**：加载器中的底模 / LoRA 为作者本机文件名，拖入后按本机路径重选（底模 `dasiwa_v5xs INT8` 或官方 BF16/INT8 均可，`CLIPLoader` type = `minimax`，视频 VAE 推荐 `minimax_h3_sol_vae_fp16`）。
2. **去油 + 保锐独立双旋钮**：
   - `#15` 社区去油胶片 LoRA（保留 `adaln_proj`，推荐 `0.35~0.45`；换上 Stage-2 去油终版自训 LoRA 后可降至 `0.0~0.30`）；
   - `#48` 自训 200 层无 `adaln` 稀疏保锐 LoRA（推荐 `0.55~0.75`，走 `bf16_lora=200/200` 无损旁路）。
3. **15s 超长片**：将 `#20 total_seconds` 改为 `15.0`，并将 `#18 head_chunks` 改为 `14`（1080p 用 `14~28`，`1.3MP` 用 `7~14`），`#36 MiniMaxH3SafeAVDecode` 会自动 2 段安全解码零 OOM。
4. **32GB 内存报错排查**：若遇 `pinned memory` / `hostbuf_grow`，运行 [`../tools/pinned_memory_patch.py`](../tools/pinned_memory_patch.py)（预设 3）并确保 ComfyUI 带 `--disable-async-offload`。

---

## 二、历史归档工作流
- [`workflow_singlewin_chunkedsage.json`](workflow_singlewin_chunkedsage.json)：v2.0 纯单窗最小骨架（无超分、无额外 LoRA 槽）。
- [`workflow.json`](workflow.json)：v1.4 四窗硬切示例（`15s / 1.3MP / calm_policy=auto`，详解见 [`../docs/GUIDE.md`](../docs/GUIDE.md) 与 [`../docs/README_LEGACY_HARDCUT.md`](../docs/README_LEGACY_HARDCUT.md)），附两条 15s 示例片 `exp_4v10a_00095.mp4`（含音轨）与 `exp_4v10a_00090.mp4`。
