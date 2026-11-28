# comfyui-h3-seamkit (v2.9)

**面向 8GB 消费级显卡（RTX 4060 Laptop 实测 10s/15s 原生 1080p 与真 2.7K/3K，全程 0 OOM）的 MiniMax H3 单窗加速、高锐度推理与 v2.9 异构扩秩高动态自蒸馏 LoRA 套件。** 同一套 `head_chunks=0` 显存求解器按 6/8/10/12/16/24GB 自动选档，A100 / H100 / 4090 / 5090 / RTX PRO 6000 走 `exact` 逐位一致路径。

> 推理节点、安全解码与 [`train/`](train/) 都在本仓库内完成，不改 `ComfyUI`、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 的源码。32GB 内存撞上 `hostbuf_grow` 时，再用 [`tools/pinned_memory_patch.py`](tools/pinned_memory_patch.py) 把 ComfyUI 的 pinned 上限调到预设 `stable`。
>
> - **原理与公式**：[`docs/PRINCIPLES_AND_FORMULAS.md`](docs/PRINCIPLES_AND_FORMULAS.md)（显存削峰、WDDM 激活预算、Roofline、Packed-Rank BF16 旁路、`0.57` 断崖、Split-Band RoPE、四相 Shifted-Morton、`Sol-H3-VAE` 分块解码、Scheme 2A 去油）。
> - **自蒸馏训练方案（AMD MI300X ROCm & NVIDIA A100/G4）**：[`docs/AMD_MI300X_V3_ACTION_LORA_PLAN.md`](docs/AMD_MI300X_V3_ACTION_LORA_PLAN.md)、[`docs/A100_LORA_TRAINING_PLAN.md`](docs/A100_LORA_TRAINING_PLAN.md) 与 [`train/`](train/)。
> - **下一代 v3.0 底层协同与千步抗错位训练蓝图**：[`docs/H3_ANIME_ENGINE_V3_ARCHITECTURE.md`](docs/H3_ANIME_ENGINE_V3_ARCHITECTURE.md)、[`docs/V3_LONG_HORIZON_1000STEP_AND_LIMB_FIX_PLAN.md`](docs/V3_LONG_HORIZON_1000STEP_AND_LIMB_FIX_PLAN.md)。
> - **工作流参数**：[`examples/README.md`](examples/README.md)。旧版多窗硬切：[`docs/README_LEGACY_HARDCUT.md`](docs/README_LEGACY_HARDCUT.md)、[`docs/GUIDE.md`](docs/GUIDE.md)。
> - 运行时依赖上游 [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)（GPL-3.0-or-later）。本仓库许可 **GPL-3.0-or-later**。

---

## 核心特性（v2.9）

1. **分块 QKV + WDDM 防溢出显存求解器（`MiniMaxH3ChunkedSageAttentionPatch`）**
   - Token 分块 ConvRot 预量化、头分组零拷贝、首尾 FP32 I/O 分块。`streaming` 去掉整段 `[S, 7168]` 输出缓冲：10s `864p` 激活从 `6.70 GiB` 降到 **`1.98 GiB`**，15s 原生 1080p（负载 `756.2`）单窗 0 OOM。
   - `head_chunks=0` 时在均匀 ConvRot 档 `{1, 2, 4, 7, 14, 28}` 里选最快且放进激活预算的档。预算 \(B_{\text{act}}=\max(2.0\text{ GiB},\;0.88\,V-0.90\text{ GiB})\)。8GB 上预算是 **`6.14 GiB`**：10s 1080p 取 `hc=4, token_chunk=16384`（`6.00 GiB`），15s 1080p 取 `hc=14`（`5.20 GiB`）。同卡上 `15s hc=7`（`6.64 GiB`）会把专用显存顶到 `7.6/8.0 GB`，WDDM 静默溢到系统内存（不报 OOM，功耗掉到约 `29W`）。
   - `out_proj_mode=auto`：序列 \(\le 64\text{k}\) 或显存 \(\ge 16\text{GB}\) 走 `exact`（与 KJ SageAttention2++ 逐位一致）；8–12GB 长序列走 `streaming`。
   - v2.7 三刀仍在：一采也走 ChunkedSage（去掉 200 层 INT8 反量化，约 `-60s`）；自训 LoRA 下全步稀疏 `sol_min_sigma=0`（约 `-85s`）；`token_chunk_size=16384` 把 `streaming` 残差读写降到约 `1/3.5`（约 `-40s`）。8GB 10s 上求解器选出的就是 `hc=4`。

2. **四相 Shifted-Morton + 末两层稠密收口**
   - Sol-Attn 在 RoPE 之后按 `block_idx & 3` 在 `(0,0), (4,4), (2,6), (6,2)` 之间轮换 `2d_frame` Morton，注意力后再逆置换。固定原点时，`1920×1088` 上 `y=512` / `y=1024` 始终压在 `8×8` 块边界，50 层尾项偏置叠成贯通割裂（实测跳变 `+2.69` / `+2.80`）。四相错位让这些 `256/512/1024 px` 网格线在 4 层里有 3 层落在块中心，峰值显存不增加，每步约 `+2–4s`。
   - 当前三套 v2.8 工作流再设 `sol_dense_blocks="48-49"`。文本、参考图、音频走 `exact_kv_and_rows` sink。`S≈147k` 时稀疏注意力相对稠密约 **`8×`**（`285s → 36s/步`）。

3. **Packed-Rank BF16 LoRA 旁路**
   - 拦截标准 2D LoRA（`LoraLoaderModelOnly` / `LoraLoaderBypassModelOnly`，可双槽），200 层线性增量走一次 cuBLAS `addmm`，不再每步 INT8 反量化再量化。
   - 弱 LoRA（如 `LMS @ 0.18`）相对 FP32：INT8 重定量化 `cos=0.016`（相对 L2 `422%`），BF16 旁路 **`cos=0.999996`**（`0.27%`）。多挂一张 `r=32/64` 全程增加不到 `1.9s`。
   - 社区去油胶片 LoRA 保留 `adaln_proj`，与自训 200 层、`0 adaln` 的保锐 LoRA 并挂。`train/slice_lora_stylekeep.py` 按层切 MLP：`action_stylekeep` 专为二次元武戏/高速运动设计（MLP 早/中/晚层 `0.30 / 0.70 / 0.25`，释放中层动作结构重建力同时锁住首尾笔触），`stylekeep` 留给二次元日常文戏细线（`0.25 / 0.45 / 0.15`），`clean` 给真人抗油（`0–35` 层 `0.80`，`36–49` 层 `0.30`），`attn_only` 去掉全部 MLP。

4. **Split-Band RoPE、Sol-H3-VAE 安全解码与流式 3K 超分落盘**
   - `rope_hf_scale` 只放大视频空间高频：\((\sqrt{WH}/1024)^{0.80}\)。工作流里 `1.30MP → 1.10`，原生 1080p `→ 1.28`，武戏 1080p `→ 1.20`。自训 LoRA 下 `attn_temp_scale` 收到 `1.02`（武戏工作流为 `1.0`）。
   - `MiniMaxH3SafeAVDecode`：10s 1080p 及以下单遍解码；潜空间 token 超过 `630000`（15s/20s 1080p）时按 5-latent 网格分段，重叠块上交叉融合。
   - `MiniMaxH3StreamingVSRSave`（[`vsr_stream_save.py`](vsr_stream_save.py)）：逐帧执行与原版 `RelayFlash` 逐位一致的超分解码并直通 `ffmpeg` 封装 H.264/AAC MP4，彻底省去原版 `VSRDecode` 一次性分配 `14.7–21.9 GiB` 连续 FP32 内存导致的 32GB 宿主机崩溃，峰值额外内存 `< 0.2 GiB`。

5. **v2.9 异构扩秩高动态自蒸馏（AMD MI300X ROCm 7.2 & NVIDIA CUDA 双栈，[`train/`](train/)）**
   - **无损异构扩秩热启动（`--rank-attn 64 --rank-mlp 32`）**：从 v2 母版（`r=32`）逐位等价热启，将 100 层 Attention 跨帧 `<Picture 1/2>` 五官与墨线检索容量翻倍至 `r=64`。
   - **运动门控高低频解耦（`--motion-gate-tau 0.75`）+ 潜空间一阶梯度损失（`--grad-xy-weight 0.15`）**：静止区 100% 锁死冻结基座 `5×5` 低频包络（零塑料油光）；高速挥砍/转身区域自动放开中低频向 28 步零快门模糊 `39f` R2VA 教师对齐，根治大动作色块涂抹。
   - **推理同构 `64`-token `2d_frame` Morton 稀疏路由**：支持 NVIDIA `flex_attention` 与 AMD Instinct MI300X `sdpa_mask` 双后端及跨日 `--auto-resume` 无损接力。

---

## 工作流

日常拖 [`examples/`](examples/) 里的 JSON。主力单窗工作流均为双路 ChunkedSage：`head_chunks=0`、`token_chunk_size=16384`、`sol_dense_blocks=48-49`、一采 `sol_tau=1.20` / `sol_max_sigma=0.90`，二采 `sol_tau=1.35`。耗时是 RTX 4060 Laptop 8GB、10s 的全程估计。

| 工作流 | 二采画布 | 成片 | 10s 全程 | 用途 |
|---|---|---|---:|---|
| **[`workflow_singlewin_selflora_13mp_vsr3k.json`](examples/workflow_singlewin_selflora_13mp_vsr3k.json)** | `1.30 MP`（`1568×864`） | **`3136×1728`** | **`~10–11 分钟`**（2 步 `~8.5 分钟`） | 日常主力。二采 `rope_hf_scale=1.10`，一步 `RelayVSR x2 (detail=0.50)` |
| **[`workflow_singlewin_selflora_1080p.json`](examples/workflow_singlewin_selflora_1080p.json)** | `1.98 MP`（`1920×1088`） | **`1920×1088`** | **`~15–16 分钟`**（2 步 `~12 分钟`，5s `~9 分钟`） | 文戏 / 通用原生 1080p。二采 `rope=1.28`，`attn_temp=1.02`，无 VSR |
| **[`workflow_singlewin_combat_1080p.json`](examples/workflow_singlewin_combat_1080p.json)** | `1.98 MP`（`1920×1088`） | **`1920×1088`** | **`~15–16 分钟`** | 武戏。一采挂 Combat / Motion Repair / speed，二采挂 **`v2_9_action_stylekeep`（`0.80–0.90`）** 或 `clean`。二采 `rope=1.20`。高速帧不要再叠 LMS |
| [`workflow_singlewin_sharp_vsr2k.json`](examples/workflow_singlewin_sharp_vsr2k.json) | `1.00 MP`（`1376×768`） | `2752×1536` | `~8.5–9.5 分钟` | 较早的最速 2.7K：单路 `hc=7`、`token_chunk=8192`、末步 Tight-Sol `0.90`，没有 `48-49` 稠密层 |

二次元高动态 / `dasiwa` 首选 `minimax_h3_sol_sharp_v2_9_action_stylekeep.safetensors`（日常慢节奏文戏可选 `..._v2_9_stylekeep.safetensors`），电影质感 LoRA 关掉。真人换成 `..._v2_9_clean.safetensors` 并打开电影质感。底模文件名是作者本机的，拖进去后重选；`CLIPLoader` 类型用 `minimax`，视频 VAE 用 `minimax_h3_sol_vae_fp16`。

### 自训高动态保锐 / 去油 LoRA 下载（GitHub Releases `v2.9`）

全部 5 个权重托管在 **[Releases · v2.9](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/tag/v2.9)**，下载后直接放入 `ComfyUI/models/loras/`（均为 `0-adaln` 200 层纯线性 BF16 异构秩 `attn=64 / mlp=32` 权重，走本仓库 BF16 旁路，与 `dasiwa` / 官方底模及其他风格 LoRA 无损并挂）：

| 文件 | 体积 | 推荐强度 | 用途（普通用户按题材选前 3 个之一） |
|---|---:|---:|---|
| [`minimax_h3_sol_sharp_v2_9_action_stylekeep.safetensors`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/minimax_h3_sol_sharp_v2_9_action_stylekeep.safetensors) | `404.8 MiB` | `0.80–0.90` | **★ 二次元武戏 / 高速运动主力（v2.9 新增首推）**：`MLP 0.30/0.70/0.25`，峰值运动帧清晰度较 v2 提升 **`+30.9%`**，锁墨线不抢画风 |
| [`minimax_h3_sol_sharp_v2_9_stylekeep.safetensors`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/minimax_h3_sol_sharp_v2_9_stylekeep.safetensors) | `404.8 MiB` | `0.65–0.75` | **★ 二次元日常 / 文戏保细线**：`MLP 0.25/0.45/0.15`，更温和地保留原底模极细赛璐璐笔触 |
| [`minimax_h3_sol_sharp_v2_9_clean.safetensors`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/minimax_h3_sol_sharp_v2_9_clean.safetensors) | `404.8 MiB` | `0.60–0.75` | **★ 真人 / 写实高动态抗油**：`0–35` 层 `0.80` + `36–49` 层 `0.30`（建议配合电影质感 LoRA `@ 0.35–0.42`） |
| [`minimax_h3_sol_sharp_v2_9_attn_only.safetensors`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/minimax_h3_sol_sharp_v2_9_attn_only.safetensors) | `240.7 MiB` | `0.75–0.90` | 纯 Attention `r=64` 轻量化版（零 MLP，少跑 100 层 BF16 GEMM，100% 零笔触干涉） |
| [`minimax_h3_sol_sharp_v2_9_final.safetensors`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/minimax_h3_sol_sharp_v2_9_final.safetensors) | `404.8 MiB` | `0.55–0.75` | 完整母版（`step 140`，供 [`train/slice_lora_stylekeep.py`](train/slice_lora_stylekeep.py) 自定义各层 MLP 比例） |

*(完整校验哈希见 Release 附件 [`SHA256SUMS.txt`](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/download/v2.9/SHA256SUMS.txt)；上一代 `r=32` 轻量权重仍保留在 [Releases · v2.8](https://github.com/yanghuagai55/comfyui-h3-seamkit/releases/tag/v2.8)。)*

归档仍在仓库里：[`workflow_singlewin_chunkedsage.json`](examples/workflow_singlewin_chunkedsage.json)（v2.0 最小单窗）、[`workflow.json`](examples/workflow.json)（v1.4 四窗硬切，附 `exp_4v10a_00095.mp4` / `exp_4v10a_00090.mp4`）。

---

## 实测成绩单（RTX 4060 Laptop 8GB / 32GB RAM，`dasiwa` INT8）

### 0. v2.9 高动态双人剑戟同源 A/B 实测（`1536×864`，90 帧，`dasiwa v5xs INT8`）

固定同一冻结一采潜空间缓存（`seed=20261128`，双角色 `<Picture 1/2>` 参考图、未见过的竹林双人刀剑交锋与 360° 旋身劈砍提示词），仅在二采切换 `v2_final` 与 **`v2.9 action_stylekeep`**（同强度 `0.90`，`sol_dense_blocks="48-49"`）：

| 对照组 | Attention / MLP 秩 | 二采激活峰值 | 全片均值 `LapVar` | 全片均值 `Tenengrad` | **最快 25% 动作帧 `LapVar`** | **最快 25% 动作帧 `Tenengrad`** | 极限旋身帧 `f084` `LapVar` |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (`v2_final @ 0.90`) | `32 / 32` | `0.80 GiB` | `176.08` | `4960.3` | `113.54` | `3536.0` | `91.4` |
| **v2.9 (`v2_9_action_stylekeep @ 0.90`)** | **`64 / 32`** | **`0.80 GiB`** | **`224.75` (`+27.6%`)** | **`5992.8` (`+20.8%`)** | **`148.58` (`+30.9%`)** | **`4337.3` (`+22.7%`)** | **`120.7` (`+32.1%`)** |

下面两张表为 v2.6–v2.8 引擎的 10s 长片同底基准记录，全程 0 OOM。

### 1. 10s `1568×864`（243 帧）

| 组别 | 核心配置 | 二采 3 步 (`s/it`) | 二采耗时 | 显存峰值 | `LapVar` | 高频 `>0.25%` | `Halo` | `Jitter` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Dense 旧基线 | 全稠密 + 旧 `fp16` 放大器 + `0.57` 断崖 | `219 / 219 / 219` | `11m10s` | `7.3 GiB` | `125.3` | `0.231%` | `1.01×` | `3.51` |
| P1-A | BF16 旁路 + 融合残差 + `hc=7` + 末步稠密 | `119.1 / 113.0 / 231.0` | `7m48s` | `7653 MiB` | `200.4` | `0.458%` | `1.00×` | `3.33` |
| P1-C | P1-A + `LMS 0.18` + `shift=6.5` + `0.80/0.48/0.18` | `121.6 / 113.9 / 231.9` | `7m52s` | `7471 MiB` | **`216.6`** | **`0.493%`** | **`1.02×`** | **`3.28`** |

### 2. 同缩到 `1920×1088` 的路线对比（10s / 243 帧）

| 路线 | 原生画布 | 二采 + 解码 (+VSR) | `@1080p LapVar` | `@1080p HF>0.25%` | `Face120` | `Halo` | `Jitter` |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧版原生 1080p (All-Sol) | `1920×1088` | `~13m` | `57.0` | `0.096%` | `161.3` | `20.3` | `3.85` |
| Exp 1 原生 1080p+（Split-RoPE `1.32` + `LMS 0.50` + Sol-VAE） | `1984×1120` | `20m 22s`（v2.7 全步稀疏后约 `~11m` 二采量级，全程见上表 `~15–16m`） | **`122.1`** | `0.265%` | `226.9` | **`20.2`** | **`3.14`** |
| `1.30MP` + `RelayVSR x2 (d=0.50)` | `3136×1728` | `~10m 50s`（v2.7 后全程 **`~10–11m`**） | **`146.4`** | `0.249%` | **`331.2`** | **`21.7`** | `3.56` |
| `1.00MP` + `RelayVSR x2 (d=0.60)` | `2752×1536` | **`9m 10s`**（无背景参考 `8m10s`） | **`179.2`** | **`0.299%`** | **`328.3`** | `23.3` | `3.30` |

- **15s 原生 `1920×1088`（362 帧）记录跑**：`hc=28` + Tight-Sol `0.90` + 两段 `SafeAVDecode`，峰值 `7909 / 7535 MiB`，二采+解码 `25m 38s`，`@1080p LapVar = 111.6`。现在的 v2.8 工作流把片长改成 15s 时，8GB 求解器会改取 `hc=14`（激活约 `5.20 GiB`），`SafeAVDecode` 仍按 token 阈值自动分段。这组 25 分钟是 Tight-Sol 记录，不是 `hc=14` 全步稀疏的新计时。
- **去油**：本地 `真实电影质感V0.1 @ 0.42` + 自训 `v2_step200 @ 0.55` + `temp=1.02`，高光峰 `hi95/hi99` 从 `0.884/0.971` 降到 `0.857/0.952`。云端 Scheme 2A 的 `loss_lf` 见上一节。

---

## 快速上手

1. 装好 `comfyui-minimax-h3-audio-T8`、`sageattention`、当前所用的 `comfy_kitchen` 后，克隆到 `ComfyUI/custom_nodes/`：
   ```bash
   git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git
   ```
2. 把上表里的 JSON 拖进 ComfyUI。日常用 **`workflow_singlewin_selflora_13mp_vsr3k.json`**，要原生 1080p 用文戏版或武戏版。
3. 二次元高速动作首选 `action_stylekeep`（日常文戏用 `stylekeep`）。真人把保锐 LoRA 换成 `clean`，并打开电影质感 LoRA。离线切片：
   ```bash
   python train/slice_lora_stylekeep.py --input <step.safetensors> --output <out.safetensors> --preset action_stylekeep
   ```
   `--preset` 还可以是 `stylekeep`、`clean` 或 `attn_only`。
4. 大显存卡保持 `out_proj_mode=auto`、`head_chunks=0`，求解器会落到 `exact` 和小 `hc`。
5. 8GB + 32GB 内存若报 `hostbuf_grow`：运行 `tools/pinned_memory_patch.py`，选预设 **`stable`**（pinned 比例 `0.45`，并写入 `--disable-async-offload`），然后重启 ComfyUI。
6. 拍 15s：三套主力工作流只改上游 Plan 的 `total_seconds=15`。8GB 会自动用 `hc=14`，解码节点按 5-latent 分段。

---

## 节点一览（18 个）

| 类别 | 节点 | 职责 |
|---|---|---|
| **MiniMax H3 Hard Cut** | `MiniMaxH3ChunkedSageAttentionPatch` | 分块 QKV、BF16 LoRA 旁路、Sol-Attn、四相 Morton、Split-Band RoPE。一采/二采各挂一路 |
| | `MiniMaxH3SafeAVDecode` | Sol-H3-VAE / INT8 VAE 音视频解码；长片按 5-latent 分段 |
| | `MiniMaxH3StreamingVSRSave` | 流式 RelayVSR Flash 解码并直通 `ffmpeg` 封装 H.264/AAC MP4（32GB 内存零 OOM，额外内存 `< 0.2 GiB`） |
| | `MiniMaxH3HardCutFirstPassPlan` / `Pass2Plan` | 片长、`17n+5` 帧网格、一采/二采分辨率（0.02 MP 步进） |
| | `MiniMaxH3FirstPassSampler` | 可缓存的一采 |
| | `MiniMaxH3HardCutUpscale` | 二采执行（单窗或多窗） |
| | `MiniMaxH3AVLatentSave` / `Load` | 冻结联合 AV latent |
| **MiniMax H3 Distill** | `MiniMaxH3TeacherProbe` / `ConditioningDump` / `TeacherLatentDump` / `LatentDownsample2x` | 教师条件与潜空间导出、`z_up` 下采样 |
| | `MiniMaxH3DenseRopePatch` | 教师机上的稠密 SDPA + 同一套 Split-Band RoPE（不依赖新版 `sageattention` API） |
| **MiniMaxH3/route3** | `Route3AutoMask` / `Apply` / `Assemble` / `Probe` | 早期前/背景双流实验。常规出片用 ChunkedSage |

---

## 仓库目录与测试

```text
comfyui-h3-seamkit/
├── chunked_sage.py                 # 分块 QKV、BF16 LoRA、Sol-Attn、显存求解器
├── sol_vae_decode.py               # Sol-H3-VAE 安全分块解码
├── vsr_stream_save.py              # 流式 RelayVSR Flash 解码直出 MP4（32GB 防内存 OOM）
├── h3_upscale.py                   # 二采执行与 [SeamKit] 日志
├── nodes.py / nodes_firstpass.py   # 上下游规划、一采采样
├── nodes_latent_cache.py           # AV latent 存读
├── hardcut_math.py / bridge.py     # 帧网格与 T8 运行时桥
├── train/                          # 教师、train_h3_lora.py、评测、slice_lora_stylekeep.py
├── examples/                       # 三套 v2.8 工作流 + 2.7K 旧预设 + v1.4 归档
├── route3/                         # 双流实验与 route3/tests/
├── docs/                           # 公式、训练计划、v1.4 指南与归档
└── tools/                          # pinned_memory_patch、切镜/缓存/接缝分析
```

```bash
for f in route3/tests/test_*.py; do PYTHONPATH=. python "$f" || exit 1; done
python -m py_compile train/*.py
```

`route3/tests/` 里有 7 个测试文件：分块注意力、Sol-VAE 解码、装配、双流数学、引擎、掩码、集成。

---

## 致谢与许可

- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)**（**Kijai**）：int64-safe per-thread INT8 Triton 量化，以及 SM75–SM120 的 SageAttention2++ 分发约定。
- **ComfyUI / `comfy_kitchen`、Sol-Attn (arXiv:2607.24027)、Sol-H3-VAE**：INT8 ConvRot CUTLASS（含原地残差 epilogue）、`rms_adaln`、`sol_attn`，以及 24 层非因果 FP16 视频解码器。
- **[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)**（**T8mars**）：H3 音视频双时钟与 3D 潜空间放大器。

**License**: GPL-3.0-or-later（见 [`LICENSE`](LICENSE)）。
