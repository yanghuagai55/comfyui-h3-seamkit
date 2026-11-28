# 示例工作流总览（v2.9）

本目录提供四套面向 8GB–24GB 显卡（RTX 4060 8GB 实测 0 OOM、WDDM 防溢出自动显存分块 + 四相 Shifted-Morton 零接缝）的 **v2.9 单窗主力工作流**，以及早期多窗硬切归档示例。

## 一、v2.9 四套主力单窗工作流（推荐直接拖入 ComfyUI）

| 文件 | 二采画布 | 最终分辨率 | 10s 全程耗时 (4060 8GB) | 推荐用途 |
|---|---|---|---:|---|
| **[`workflow_singlewin_selflora_13mp_vsr3k.json`](workflow_singlewin_selflora_13mp_vsr3k.json)** | `1.30 MP` (`1568×864`) | **`3136×1728` (真 3K)** | **`~10–11 分钟`**（2步 `~8.5m`） | **★ 日常主力甜点王**：双路 `ChunkedSage` + 双保险消缝（`dense=48-49`）+ v2.9 自训高动态保锐 + `MiniMaxH3StreamingVSRSave` 流式一步 `RelayVSR x2 (detail=0.50)`（32GB 内存零 OOM） |
| **[`workflow_singlewin_selflora_1080p.json`](workflow_singlewin_selflora_1080p.json)** | `1.98 MP` (`1920×1088`) | **`1920×1088` (纯原生)** | **`~15–16 分钟`**（2步 `~12m` / 5s `9m`） | **文戏/通用纯原生 1080p 直出**：无 VSR 分支 + 双保险消缝 + WDDM 自动分块 (`hc=0`) |
| **[`workflow_singlewin_combat_1080p.json`](workflow_singlewin_combat_1080p.json)** | `1.98 MP` (`1920×1088`) | **`1920×1088` (纯原生)** | **`~15–16 分钟`** | **★ 武戏动作增强专用**：一采/二采彻底分路（一采动作栈；二采挂 **`minimax_h3_sol_sharp_v2_9_action_stylekeep.safetensors @ 0.80–0.90`**）+ 双保险消缝 |
| **[`workflow_singlewin_sharp_vsr2k.json`](workflow_singlewin_sharp_vsr2k.json)** | `1.00 MP` (`1376×768`) | **`2752×1536` (真 2.7K)** | **`~8.5–9.5 分钟`** | **经典最速 2.7K**：H3 原生 RoPE 甜点 + `LMS` + `RelayVSR x2 (detail=0.60)` |

### 通用上手要点
1. **模型替换**：加载器中的底模 / LoRA 为作者本机文件名，拖入后按本机路径重选（底模 `dasiwa_v5xs INT8` 或官方 BF16/INT8 均可，`CLIPLoader` type = `minimax`，视频 VAE 推荐 `minimax_h3_sol_vae_fp16`）。
2. **v2.9 二次元高动态 vs 文戏 vs 真人切换**：
   - **二次元武戏 / 高速运动（v2.9 首推）**：二采选 `minimax_h3_sol_sharp_v2_9_action_stylekeep.safetensors`（强度 `0.80–0.90`，电影质感 LoRA 保持 Bypass 关闭；实测高速帧 `LapVar +30.9%`）；
   - **二次元文戏 / 极细笔触**：选 `minimax_h3_sol_sharp_v2_9_stylekeep.safetensors`（强度 `0.65–0.75`）；
   - **真人 / 写实**：换 `minimax_h3_sol_sharp_v2_9_clean.safetensors`（强度 `0.60–0.75`）并 `Ctrl+B` 开启电影质感 LoRA。
   - 全部 v2.9 自训权重统一托管在 **[GitHub Releases · v2.9](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/tag/v2.9)**，下载后放入 `ComfyUI/models/loras/`。
3. **10s / 15s 自动显存分块**：`#16/#18` 默认 **`head_chunks = 0`（自适应 WDDM 防溢出公式）** + **`sol_dense_blocks = "48-49"`**（叠四相 Shifted-Morton 双保险消缝），只需改 `#20 total_seconds = 15.0`，8GB 自动切 `hc=14`、`#36` 自动 2 段解码零 OOM。
4. **32GB 内存与锁页内存说明**：Windows 下 ComfyUI 锁页内存上限为 `0.45 × 物理内存`（32GB 内存 = `14.2 GiB`）；20GB INT8 DiT + LoRA 吃满不回落约需 `16–18 GiB` 锁页内存（48–64GB RAM 最佳），32GB 跑片时建议关闭高占内存后台并保留 `--disable-async-offload`。

---

## 二、历史归档工作流
- [`workflow_singlewin_chunkedsage.json`](workflow_singlewin_chunkedsage.json)：v2.0 纯单窗最小骨架（无超分、无额外 LoRA 槽）。
- [`workflow.json`](workflow.json)：v1.4 四窗硬切示例（`15s / 1.3MP / calm_policy=auto`，详解见 [`../docs/GUIDE.md`](../docs/GUIDE.md) 与 [`../docs/README_LEGACY_HARDCUT.md`](../docs/README_LEGACY_HARDCUT.md)），附两条 15s 示例片 `exp_4v10a_00095.mp4`（含音轨）与 `exp_4v10a_00090.mp4`。
