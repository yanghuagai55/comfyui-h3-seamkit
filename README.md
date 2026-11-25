# comfyui-h3-seamkit (v2.7)

**面向 8GB 消费级显卡（RTX 4060 Laptop 实测验证 10s/15s 原生 1080p 与真 2.7K/3K 全程 0 OOM，同时无缝兼容 Colab A100 / H100 / 4090 / 5090 / RTX PRO 6000）的 MiniMax H3 单窗加速、高锐度推理与自蒸馏去油 LoRA 训练套件。**

> **100% 独立自包含 · 零侵入第三方插件**：本仓库**不修改** `ComfyUI` 核心、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 或任何第三方包的源码；全部推理内核、安全解码与 A100 训练代码均自包含在仓库内，上游更新零冲突。
>
> - **核心原理与完整数学公式（v2.7）**：[`docs/PRINCIPLES_AND_FORMULAS.md`](docs/PRINCIPLES_AND_FORMULAS.md)（显存削峰、Roofline 耗时拟合与三刀提速、Packed-Rank BF16 旁路、`0.57` 断崖修复、Split-Band RoPE、`Sol-H3-VAE` 5-latent 分块解码、以及 **Scheme 2A 高低频解耦去油自蒸馏公式**）。
> - **Colab A100 / G4 自蒸馏 LoRA 训练指南**：[`docs/A100_LORA_TRAINING_PLAN.md`](docs/A100_LORA_TRAINING_PLAN.md)（配合 [`train/`](train/) 完整端到端脚本）。
> - **示例工作流总览**：[`examples/README.md`](examples/README.md) · **旧版 v1.4 多窗硬切归档**：[`docs/README_LEGACY_HARDCUT.md`](docs/README_LEGACY_HARDCUT.md)。
> - 运行时依赖上游 [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)（GPL-3.0-or-later）；本仓库许可 **GPL-3.0-or-later**。

---

## 核心特性（v2.7）

1. **双路分块 QKV 低显存引擎 + 三刀提速（`MiniMaxH3ChunkedSageAttentionPatch`）**：
   - 一次性 Token 分块 ConvRot 预量化 + 头分组零拷贝切片（自动按 `2 头 = 256` ConvRot 对齐）+ 首尾 FP32 I/O 层分块，10s `864p` 激活峰值从 `6.70 GiB` 压至 **`1.98 GiB`**，15s 原生 1080p（负载 `756.2`）单窗 0 OOM；
   - **v2.7 三刀消除 `~3.5–4 分钟` 软件水分**：① 一采同样走 `ChunkedSage (hc=7, sol_max_sigma=0.90)` 消除 200 层 INT8 反量化（`-60s`）；② 配合自训 LoRA 放开全步稀疏（`sol_min_sigma=0, tau=1.40`，`-85s`）；③ `head_chunks=4, token_chunk_size=16384` 将 `streaming` 模式残差读写降 `3.5×`（100% 数值等价，`-40s`）。
2. **Packed-Rank 16-bit BF16 低秩 LoRA 旁路 + 算子融合**：
   - 自动拦截标准 2D LoRA（兼容 `LoraLoaderModelOnly` / `LoraLoaderBypassModelOnly` 与双槽并挂），免掉每步 200 层 VBAR INT8 反量化/重定量化；
   - 微弱锐化/自训 LoRA 的 16-bit 保真度从 INT8 重定量化的 **`cos = 0.016`（422% 误差）** 提升至 **`cos = 0.999996`（0.27% 误差）**，叠挂多张 LoRA 合并进同一次 cuBLAS `addmm` 仅增 `<1.9s`；
   - 原生支持**社区去油胶片 LoRA（保留 `adaln_proj`）+ 自训 200 层无 `adaln` 保锐 LoRA 双槽无损并挂**，独立调节去油胶片感与 1080p 锐度。
3. **Sol-Attn 稀疏加速 + 面积归一化 Split-Band RoPE + `Sol-H3-VAE` 8GB 安全分块解码**：
   - 内置 `2d_frame` Morton 原地重排与条件/音频 Sink 保护（$S=147\text{k}$ 时注意力提速 **`8×`**）；
   - `rope_hf_scale` 按 $(\sqrt{W\cdot H}/1024)^{0.80}$ 零耗时补偿高分辨率空间相位压缩（`1.30MP → 1.10`，`1.98MP → 1.28~1.32`）；
   - `MiniMaxH3SafeAVDecode` 在 `15s 1080p` 按 5-latent 块网格自动 2 段解码并保护块融合，0 接缝 0 OOM。
4. **端到端 A100 自蒸馏去油保锐 LoRA 训练管线（[`train/`](train/)）**：
   - 官方满血 BF16 24 步教师离线生成 + Zstd 分片 HF 同步 + 可微静态 `flex_attention`（空间邻域 `±2` + 质心余弦路由）+ 保方向 L2 `_GradClamp` + 强制 `400/400` 活层门禁；
   - **Scheme 2A 高低频解耦去油目标（`--lf-base-anchor --lf-kernel 5`）**：低频 `5×5` 空间包络 100% 锚定冻结基座 `v_base`（`0.0 GB` 额外峰值显存）、高频 100% 吸收满血教师 1080p 细节，A100 实测仅 25 步即将低频油光误差 `loss_lf` 压降 **`91%`**！

---

## 三套开箱即用工作流（直接拖入 ComfyUI）

| 工作流文件 | 二采画布 (`second_megapixels`) | 最终成品分辨率 | 4060 8GB 10s 全程耗时 | 核心特点与定位 |
|---|---|---|---:|---|
| **[`examples/workflow_singlewin_selflora_13mp_vsr3k.json`](examples/workflow_singlewin_selflora_13mp_vsr3k.json)** | **`1.30 MP` (`1568×864`)** | **`3136×1728` (真 3K)** | **`~10–11 分钟`**（2步 `~8.5m`） | **★ 日常主力甜点王**：三刀提速 + 去油胶片 + 自训保锐 + 一步 `RelayVSR x2 (d=0.50)` |
| **[`examples/workflow_singlewin_selflora_1080p.json`](examples/workflow_singlewin_selflora_1080p.json)** | **`1.98 MP` (`1920×1088`)** | **`1920×1088` (纯原生)** | **`~15 分钟`**（2步 `~12m` / 5s `9m`） | **纯原生 1080p 直出**（移除 VSR，0 误触 4K OOM 风险） |
| **[`examples/workflow_singlewin_sharp_vsr2k.json`](examples/workflow_singlewin_sharp_vsr2k.json)** | **`1.00 MP` (`1376×768`)** | **`2752×1536` (真 2.7K)** | **`~8.5–9.5 分钟`** | **经典最速 2.7K**：H3 原生 RoPE 甜点 + `LMS` + `RelayVSR x2 (d=0.60)` |

---

## 实测成绩单（RTX 4060 Laptop 8GB / 32GB RAM，真实 `dasiwa` INT8）

### 1. 10s `1568×864`（243 帧）严格同底 A/B 实测（全部 0 OOM）

| 组别 | 核心配置 | 二采 3 步 (`s/it`) | 二采耗时 | 显存峰值 | `LapVar` 锐度 | 高频 `>0.25%` | 光晕 `Halo` | 抖动 `Jitter` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Dense 旧基线** | 全稠密 + 旧 `fp16` 放大器 + `0.57` 断崖 | `219 / 219 / 219` | `11m10s` | `7.3 GiB` | `125.3` | `0.231%` | `1.01×` | `3.51` |
| **P1-A（纯引擎）** | BF16 旁路 + 融合残差 + `hc=7` + 末步稠密 | `119.1 / 113.0 / 231.0` | `7m48s` (`-54.4s`) | `7653 MiB` | `200.4` (`+59.9%`) | `0.458%` | `1.00×` | `3.33` |
| **P1-C（默认推荐）** | P1-A + `LMS 0.18` (`+1.9s`) + `shift=6.5` + `0.80/0.48/0.18` | `121.6 / 113.9 / 231.9` | `7m52s` | `7471 MiB` | **`216.6` (`+72.9%`)** | **`0.493%`** | **`1.02×`** | **`3.28`** |

### 2. 原生 1080p vs 2.7K / 3K 超分实测对比（全部同缩至 `1920×1088` 评测，0 OOM）

| 路线（10s / 243 帧） | 原生输出画布 | 二采 + 解码 (+VSR) | `@1080p LapVar` | `@1080p HF>0.25%` | `Face120` | `Halo` | `Jitter` |
|---|---|---:|---:|---:|---:|---:|---:|
| **旧版原生 1080p (All-Sol)** | `1920×1088` | `~13m` | `57.0` | `0.096%` | `161.3` | `20.3` | `3.85` |
| **Exp 1 原生 1080p+**（Split-RoPE `1.32` + `LMS 0.50` + `Sol-VAE`） | `1984×1120` | `20m 22s`（v2.7 全步稀疏 **`~11m`**） | **`122.1` (`2.14×`)** | `0.265%` | `226.9` | **`20.2`** | **`3.14`** |
| **`1.30MP (864p)` + `RelayVSR x2 (d=0.50)` 真 3K** | **`3136×1728`** | **`~10m 50s`**（v2.7 **`~10m`**） | **`146.4` (`2.57×`)** | `0.249%` | **`331.2`** | **`21.7`** | `3.56` |
| **Exp 2 `1.00MP (768p)` + `RelayVSR x2 (d=0.60)` 真 2.7K** | **`2752×1536`** | **`9m 10s`**（无背景参考 `8m10s`） | **`179.2` (`3.14×`)** | **`0.299%`** | **`328.3`** | `23.3` | `3.30` |

- **15s 原生 `1920×1088`（362 帧单窗，负载 `756.2`）**：`hc=28` + `Tight-Sol 0.90` + `MiniMaxH3SafeAVDecode` 2 段解码全程 **0 OOM**，`@1080p LapVar = 111.6`（旧版 15s 的 `2.20×`），过渡帧零接缝。
- **v2.6/v2.7 去油实证**：本地双槽并挂 `真实电影质感V0.1 @ 0.42 + 自训 v2_step200 @ 0.55 + temp=1.02` 将高光过曝峰 `hi95/hi99` 从 `0.884/0.971` 稳降至 `0.857/0.952`；云端 Stage-2 开启 `--lf-base-anchor --lf-kernel 5` 后 25 步内 `loss_lf` 从 `0.872` 降至 `0.079`（`-91%`）。

---

## 快速上手

1. 克隆到 `ComfyUI/custom_nodes/`（需已安装 `comfyui-minimax-h3-audio-T8`、`sageattention` 与新版 `comfy_kitchen`）：
   ```bash
   git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git
   ```
2. 按上表将 `examples/` 中对应的 `.json` 拖入 ComfyUI（推荐日常直接用 **`workflow_singlewin_selflora_13mp_vsr3k.json`**，纯原生 1080p 用 **`workflow_singlewin_selflora_1080p.json`**）。
3. **大显存卡（A100 / H100 / 4090 / 5090）**：`out_proj_mode="auto"` 自动切 `exact` 逐位一致模式，建议 `head_chunks = 1` 或 `2`。
4. **8GB 卡 + 32GB 内存如遇 `hostbuf_grow` / `pinned memory` 报错**：运行 `tools/pinned_memory_patch.py`（预设 3），并确保启动参数带 `--disable-async-offload`。

---

## 节点一览（共 12 个）

| 类别 | 节点 | 核心职责 |
|---|---|---|
| **主线加速** | **`MiniMaxH3ChunkedSageAttentionPatch`** | 分块 QKV + BF16 低秩 LoRA 旁路 + CUTLASS 融合残差 + Sol-Attn / Tight-Sol + 分频段 RoPE + 首尾 FP32 分块（支持一采/二采独立双路挂载） |
| **安全解码** | **`MiniMaxH3SafeAVDecode`** | `Sol-H3-VAE` 8GB 守护神：10s 单遍直出，15s/20s 自动按 5-latent 块分段并保护块融合，0 OOM 0 接缝 |
| **双采规划与缓存** | `MiniMaxH3HardCutFirstPassPlan` / `Pass2Plan` / `FirstPassSampler` / `HardCutUpscale` / `AVLatentSave` / `Load` | 上下游分辨率（0.02 步进精确两位小数）与 17n+5 帧长规划、一采/二采 Latent 存读档、3D 锐度放大器 `bf16 + pad=3` 调度 |
| **实验套件** | `MiniMaxH3Route3AutoMask` / `Apply` / `Assemble` / `Probe` | 早期前/背景双流压缩实验套件（常规场景已由 `ChunkedSage + Sol-Attn` 替代） |

---

## 仓库目录与回归测试

```text
comfyui-h3-seamkit/
├── chunked_sage.py              # 核心：分块 QKV + BF16 LoRA 旁路 + 融合核 + Sol-Attn + Split-RoPE
├── sol_vae_decode.py            # Sol-H3-VAE 5-latent 块对齐 8GB 安全分块解码节点
├── h3_upscale.py                # 二采单窗/多窗执行器与统一 [SeamKit] 日志
├── nodes.py                     # 上下游双采规划节点（两位小数像素数精度）
├── nodes_firstpass.py           # 一采可缓存采样器
├── nodes_latent_cache.py        # 冻结 AV latent 存/读档节点
├── hardcut_math.py / bridge.py  # 帧网格数学与 T8 运行时桥接
├── train/                       # A100/G4 端到端离线教师 + 高低频解耦去油 LoRA 训练与评测套件
├── examples/                    # 三套 v2.7 主力工作流（1.3MP VSR3K / 原生1080p / 1.0MP VSR2.7K）
├── route3/tests/                # 7 套全绿单元与 GPU 逐位一致性回归测试
├── docs/
│   ├── PRINCIPLES_AND_FORMULAS.md  # 核心原理、数学推导与全线实测总账（v2.7）
│   ├── A100_LORA_TRAINING_PLAN.md  # Colab A100/G4 自蒸馏 LoRA 任务书与排障矩阵
│   └── README_LEGACY_HARDCUT.md    # 旧版 v1.4 多窗硬切文档归档
└── tools/                       # pinned_memory_patch / 帧分析 / 缓存管理工具
```

运行全部单元与回归测试：
```bash
for f in route3/tests/test_*.py; do PYTHONPATH=. python "$f" || exit 1; done
python -m py_compile train/*.py
```

---

## 致谢与许可

- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)**（**Kijai**）：int64-safe per-thread INT8 Triton 量化核与 SM75–SM120 SageAttention2++ 分发契约；
- **ComfyUI / `comfy_kitchen` & Sol-Attn (arXiv:2607.24027) & Sol-H3-VAE**：底层 INT8 ConvRot CUTLASS 核（含原地残差 epilogue）、`rms_adaln`、`sol_attn` 稀疏核及 24L 非因果 FP16 视频解码器；
- **[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)**（**T8mars**）：上游 H3 音视频双时钟与 3D 潜空间放大器运行时。

**License**: GPL-3.0-or-later（见 [`LICENSE`](LICENSE)）。
