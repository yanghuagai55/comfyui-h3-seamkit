# comfyui-h3-seamkit (v2.0)

**让 8GB 消费级显卡也能单窗直出 MiniMax H3 全程 10s / 15s 与原生 1080p（并向 2K 演进）——零分窗接缝、100% 兼容 KJ SageAttention2++ 与 `comfy_kitchen` Sol-Attn 稀疏加速。**

> **完全独立封装 · 零侵入第三方插件**：本仓库**不修改** `ComfyUI` 核心、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 或任何第三方包的源码；所有内核封装（INT8 ConvRot 预量化、分块 QKV、int64-safe Triton 量化、`2d_frame` Morton 原地重排、`FinalLayer` / `video_patch_proj` 分块）均自包含在 [chunked_sage.py](chunked_sage.py) 内，上游任意更新均不会被覆盖。
>
> - 旧版 v1.4「多窗硬切分块」文档已归档至 [`docs/README_LEGACY_HARDCUT.md`](docs/README_LEGACY_HARDCUT.md)（多窗与硬切节点仍完整保留，可供 >15s 或极端负载回退）。
> - 运行时依赖上游 [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)（GPL-3.0-or-later）；本仓库采用相同许可 **GPL-3.0-or-later**。

---

## 为什么从「分窗硬切」进化到「单窗分块 QKV + 稀疏 Sage」？

MiniMax H3 原生支持最长 15s，且 50 层 DiT 中有 **72%–84% 的注意力质量分布在跨帧全局交互上**（实测探针统计）。一旦把 10s/15s 硬拆成多个独立窗口分别去噪，即使采用硬切或 overlap 锚定，仍难免出现跨窗姿态/曝光跳变（例如 10s 实片 `f118 → f119` 帧间 MAE 飙升至 `10.63`）。

此前 8GB 显卡被迫分窗（经验 OOM 锚 `frames × MP = 191.0`），真正元凶并非权重或注意力本身，而是 **激活峰值被三处整序列缓冲撑爆**：
1. **`qkv_proj(h)` 一次性物化 `[S, 21504]` BF16**：10s (`S≈96k`) 单张就吃 **3.85 GiB**，15s (`S≈144k`) 吃 **5.78 GiB**；
2. **`norm1/2 + SwiGLU MLP` 整序列物化两份 `[S, 5376]`**（各 `0.98–1.48 GiB`）；
3. **首尾 FP32 层 `video_patch_proj` 与 `FinalLayer._modulate`** 一次性物化 `[N_v, 5376]` FP32（`1.96–2.94 GiB`）。

### 核心方案：`MiniMaxH3ChunkedSageAttentionPatch`

```text
x [S, 5376] bf16
 ├─ Step A: Token 分块 norm1 + AdaLN + INT8 ConvRot 预量化 ──▶ (x_qdata [S,5376] int8, x_scale [S,1])
 │          （永不物化整条 BF16 h，且全层仅量化/旋转 1 次）
 ├─ Step B: 头分组循环（按 2 头 = 256 ConvRot 块对齐，如 7 / 14 / 28 组）：
 │    ├─ 零拷贝行切片 Q/K 权重 ──▶ 预量化 INT8 GEMM ──▶ 原地 RMS-RoPE
 │    ├─ 模式 1 (dense_sage): KJ SageAttention2++（先量化释放 BF16 Q/K，再投影 V）
 │    ├─ 模式 2 (sol_attn_sparse): comfy_kitchen Sol-Attn 稀疏 Sage（2d_frame Morton + 质心尾项 + 条件/音频 sink + token_aug=64）
 │    └─ out_proj:
 │         • exact 模式：写 [S, 7168] 后分块 out_proj（与原版 KJ 100.00% 逐位一致）
 │         • streaming 模式：按头组列切片 [5376, g×128] 分块直累残差 x（零 [S, 7168] 输出缓冲！）
 ├─ Step C: 融合 Token 分块 norm2 + AdaLN + SwiGLU MLP + gate ──▶ 原地写回 x（100.00% 逐位一致）
 └─ I/O 层: video_patch_proj 与 FinalLayer 逐位一致 Token 分块（消除首尾 FP32 显存尖峰）
```

**兼容性**：原生覆盖 **SM75 / SM80 (A100) / SM86 / SM89 (4060/4090) / SM90 (H100) / SM120 (5090)**、ComfyUI `aimdo` VBAR 动态显存与 `malloc_graph`（0 rogues）、`LoraLoaderModelOnly` / 低显存 LoRA、BF16 未量化权重，以及 Route 3 外部注意力拦截。

---

## 实测数据（RTX 4060 Laptop 8GB / 32GB RAM，真实 `dasiwa` INT8 ConvRot）

### 1. 单层峰值激活显存 & 数值一致性（`blocks.0..4` 实测）

| 片长（序列 $S$） | 原版 KJ (`h=6, mlp=4`) | **`exact` (`h=14`，100% 逐位一致)** | **`streaming` (`h=14`，零输出缓冲)** | **`Sol-Attn 稀疏` (`tau=1.3, aug=64`)** |
|---|---:|---:|---:|---:|
| **5s @ 864p (`48k`)** | `3.35 GiB` \| `66.6s/步` | **`1.63 GiB` \| `61.3s/步`（快 8%）** | **`1.21 GiB` \| `64.5s/步`** | — |
| **10s @ 864p (`94k~96k`)** | `6.70 GiB`（爆/抖 `564s/步`） | **`3.26 GiB` \| `193.1s/步`** | **`1.91~1.98 GiB` \| `192.8s/步`** | **`1.99 GiB` \| `81.1s/步`（`2.38×`）** |
| **15s @ 864p (`144k`)** | `10.04 GiB`（爆） | `4.89 GiB` \| `402.8s/步` | **`2.97 GiB` \| `414.5s/步`** | `~3.05 GiB` \| **`~130s/步`（`~3.1×`）** |
| **10s @ 真 1080p (`147.6k`)** | 爆 | — | `2.99 GiB` \| `426.4s/步` | **`3.14 GiB` \| `137.8s/步`（`3.09×`）** |
| **15s @ 真 1080p (`219k`)** | 爆 | — | `3.87 GiB` \| `901.6s/步` | **`3.99 GiB` \| `269.2s/步`（`3.35×`）** |

- **精度验证**：
  - `exact` vs 原版 KJ SageAttention2++（连续 5 层真实权重）：`max_abs = 0.000000`、**`100.00%` 逐位完全一致**；
  - `streaming` vs BF16 反量化真值（连续 5 层）：`cos = 0.999965`（原版 INT8 本身 `cos = 0.999981`）；
  - `Sol-Attn (tau=1.3, token_aug=64)` vs 稠密 KJ SageAttention2++（结构化视频 token）：**`cos = 0.999184`**（`rel_L2 = 4.05%`）。

### 2. 端到端 ComfyUI 实片验证（RTX 4060 Laptop 8GB，含 Pika LoRA）

| 实测任务 | 窗口 / 负载（旧 OOM 锚 191） | 二采单步耗时 | `nvidia-smi` / Alloc 峰值 | 稳定性 & 画质/跳变指标 |
|---|---|---:|---:|---|
| **10s @ 1568×864（稠密单窗）** | 1 窗 `243f` / **`329.2`（1.72×）** | `223.5s/步`（3 步 `11m10s`） | `7309 MiB` / `1.017 GiB` | 0 OOM / 0 rogues；旧接缝 `f118→f119` MAE **`10.63 → 2.38`** |
| **10s @ 1568×864（Sol-Attn 稀疏单窗）** | 1 窗 `243f` / **`329.2`（1.72×）** | **`119.1s/步`（3 步 `5m57s`，提速 `1.88×`！）** | **`7325 MiB` / `1.017 GiB`** | 相对稠密 PSNR `26.31 dB`（背景几乎零差、肉眼难辨）；`f118→119` MAE `2.50` |
| **15s @ 1568×864（稠密单窗）** | 1 窗 `362f` / **`490.4`（2.57×）** | `451.5s/步`（3 步 `22m34s`） | `7909 MiB` / `1.134 GiB` | 0 OOM / 0 rogues；全程 3 镜零人工缝，原生硬切 `f119→120` / `f243→244` 干脆利落 |
| **10s @ 原生 1080p (`1920×1088`, Sol-Attn 单窗)** | 1 窗 `243f` / **`507.6`（2.66×）** | **`217.4s/步`（3 步 `10m52s`，比 864p 稠密还快！）** | **`7835 MiB` / `1.172 GiB`** | 0 OOM；`960×544` 一采 → 精确 `2.0×` 放大 → 真 1080p，线条/发丝/眸光显著更锐，全程 `17m36s` |

---

## 耗时经验拟合公式（RTX 4060 Laptop 8GB · 50 层 INT8 DiT）

基于上述单层常驻显存基准（`S = 48k / 96k / 144k` 分别对应 `64.5s / 199.3s / 414.5s`，零 PCIe）与 ComfyUI `aimdo` 实跑（10s 单步 `223.5s`）的联立反解，稠密模式下单步耗时满足：

$$T_{\text{step}}^{\text{dense}}(S) \approx 24\text{s}_{\text{PCIe}} + 29.4\text{s} \cdot \left(\frac{S}{48\text{k}}\right)_{\text{Linear/MLP}} + 35.2\text{s} \cdot \left(\frac{S}{48\text{k}}\right)^2_{\text{稠密 SageAttn}}$$

- **公式推导**：Transformer 计算严格拆分为 $O(S)$ 线性项（Norm / QKV / OutProj / MLP）与 $O(S^2)$ 注意力二次项；由 $a + b = 64.5$ 与 $2a + 4b = 199.3$ 直接解得 $a = 29.35\text{s}$、$b = 35.15\text{s}$，代回 $x = 3\,(144\text{k})$ 得 $404.4\text{s}$（与实测 `414.5s` 误差仅 2.4%）；实跑减去纯 GPU 计算得 Aimdo PCIe 权重搬运约 `24s/步`（15s 内存偏紧时约 `35s/步`）。
- **核心启示**：
  - 在 **10s 864p (`96k`)**，注意力占 **63%**；
  - 在 **15s 864p (`144k`) / 10s 1080p (`147k`)**，注意力占 **74%**；
  - 在 **15s 1080p (`218k`)**，注意力暴涨至 **82%**（`725s/步`）；
  - 因此启用 `sol_attn_sparse` 将二次项压低 **3×–5×**，才能同时撬动**提速**与**高分辨率（1080p / 2K）**；而在二次项压低后，剩余大头变为线性项 + PCIe（对应路线图阶段三的步数优化、CUTLASS 残差融合与中间层缓存）。

---

## 节点一览（共 11 个）

### 核心主线（`MiniMax H3 Hard Cut`）
| 节点 | 用途 |
|---|---|
| **`MiniMaxH3ChunkedSageAttentionPatch`** | **核心主力**：分块 QKV + KJ SageAttention2++ + `comfy_kitchen.sol_attn` 稀疏加速 + 首尾 FP32 分块 |
| `MiniMaxH3HardCutFirstPassPlan` | 上游规划：片长（17n+5 网格）、一采/二采画布、单窗上限（设 `target_segment_seconds ≥ 片长+1` 即单窗）、一采缓存 |
| `MiniMaxH3HardCutPass2Plan` | 下游二采规划：3D 放大器、精度、padding（改下游不击穿一采缓存） |
| `MiniMaxH3FirstPassSampler` | 一采可缓存采样器（内置 SageAttention 补丁护栏，调二采参数 0 秒跳过一采） |
| `MiniMaxH3HardCutUpscale` | 二采执行器（自动识别 `ChunkedSage` 并抬升 8GB 负载安全线至 `510 / 620`；同时保留旧版多窗硬切能力） |
| `MiniMaxH3AVLatentSave` / `Load` | 冻结联合 AV latent 存/读档 |

### 实验/专项保留（`MiniMaxH3/route3`）
| 节点 | 定位 |
|---|---|
| `MiniMaxH3Route3AutoMask` / `Apply` / `Assemble` / `Probe` | 早期前景/背景双流压缩实验套件（适用于极静态背景/纯绿幕场景研究；常规高分单窗场景已由 `ChunkedSage + Sol-Attn` 全面替代） |

---

## 安装与快速上手

1. 克隆到 `ComfyUI/custom_nodes/` 并确保已安装上游 `comfyui-minimax-h3-audio-T8`、`sageattention` 与新版 `comfy_kitchen`：
   ```bash
   git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git
   ```
2. 重启 ComfyUI，直接拖入示例工作流 [`examples/workflow_singlewin_chunkedsage.json`](examples/workflow_singlewin_chunkedsage.json)。
3. 在二采模型链上挂载 **`MiniMaxH3ChunkedSageAttentionPatch`**（可直接替代 `MiniMaxLowVRAMAttention` + `MiniMaxChunkFeedForward` + `SolAttnMiniMax`）：
   - **追求极限提速 / 1080p**：`attention_mode = sol_attn_sparse`，`sol_tau = 1.3`，`sol_token_aug = 64`，`out_proj_mode = auto`，`head_chunks = 14`（15s 1080p 可设 `28`）；
   - **追求与原版 KJ 100% 逐位一致**：`attention_mode = dense_sage`，`out_proj_mode = exact`（≤10s）或 `auto`。
4. **8GB 卡 + 32GB 内存如遇 `hostbuf_grow` / `pinned memory` 报错**：先运行 `tools/pinned_memory_patch.py`（推荐预设 3，并确保启动参数带 `--disable-async-offload`）。

---

## 演进路线图（聚焦「提速 + 冲高分辨率 1080p → 2K」）

MiniMax H3 的 3D RoPE（`_frame_grid`）采用**面积归一化坐标**，放大到 1080p / 2K 不会产生 RoPE 外推崩坏；因此未来演进全面聚焦两条主线：

- [x] **Step 1（已完成）**：分块 QKV + KJ SageAttention2++（`exact` 逐位一致 + `streaming` 零输出缓冲），RTX 4060 8GB 单窗实片通过 **10s (`329.2`)** 与 **15s (`490.4`)**，彻底消灭跨窗跳变；
- [x] **Phase 1（已完成并实片验收）**：头分组内原生融合 **`comfy_kitchen.sol_attn` 稀疏 SageAttention** + 零峰值列分块 `2d_frame` Morton 重排——10s @ 864p 二采端到端提速 **`1.88×`**（`11m10s → 5m57s`，肉眼几乎无差），并已成功解锁 **8GB 本地单窗原生 1080p (`1920×1088 = 2.09MP`, 负载 `507.6`, `217.4s/步` 比 864p 稠密还快！)**；
- [ ] **Phase 2（继续向 2K `2560×1440` 突破）**：
  - [x] 原生单窗 **1080p (`1920×1088 = 2.09MP`)**：`0.5MP (960×544)` 一采 → 精确 `2.0×` 3D Upscaler → 1080p 单窗二采（已实片通过！）；
  - [ ] 突破 **2K (`2560×1440 = 3.69MP`)** 双路径对比：① **三阶段金字塔**（1080p 锁全局 + 2K 仅 1 步 `denoise=0.15~0.18` 细节抛光）；② **1080p/864p 单窗 + `RelayVSR x2` 一步视频超分**直出 2K/3K；
- [ ] **Phase 3（向 Linear/MLP 与步数再要 1.5×–2× 加速）**：
  - `cutlass_int8_dequant_residual` Epilogue 残差融合（砍掉 `streaming` 模式重复读写残差流带宽）；
  - 二采 `2 步 vs 3 步` 画质 A/B + 低噪步「中间层残差复用（Mid-Block Delta Cache）」；
- [ ] **Phase 4（A100-40GB 云端量产档）**：全权重常驻零 PCIe + `exact` 逐位无损 + Sol-Attn 稀疏，实现 **10s @ 1080p < 1 分钟、15s @ 1080p < 2 分钟、2K 2–3 分钟**产出。

---

## 仓库目录结构

```text
comfyui-h3-seamkit/
├── chunked_sage.py          # 核心：MiniMaxH3ChunkedSageAttentionPatch（分块 QKV + KJ Sage + Sol-Attn）
├── h3_upscale.py            # 二采执行器（单窗/多窗 + 自动识别 ChunkedSage 负载线）
├── nodes.py                 # 上下游双采规划节点
├── nodes_firstpass.py       # 一采可缓存采样器
├── nodes_latent_cache.py    # 冻结 AV latent 存/读档节点
├── hardcut_math.py          # 17n+5 帧网格与负载计算器（纯 Python）
├── bridge.py                # 上游 T8 运行时桥接
├── examples/                # 示例工作流（含单窗 ChunkedSage 工作流与旧版示例）
├── route3/                  # 实验性双流引擎与全套单测（route3/tests/test_*.py，6 套全绿）
├── docs/                    # 原理文档、任务书与旧版 README 归档（README_LEGACY_HARDCUT.md）
├── templates/               # 官方六段式提示词模板
└── tools/                   # pinned_memory_patch / 帧分析 / 缓存管理工具
```

运行全套单元测试（零回归校验）：
```bash
for f in route3/tests/test_*.py; do PYTHONPATH=. python "$f" || exit 1; done
```

---

## 致谢与许可

- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)**（**Kijai**）：`chunked_sage.py` 中的 int64-safe per-thread INT8 Triton 量化核与 SM75–SM120 SageAttention2++ 分发契约严格对齐 KJNodes 的 `MiniMaxH3MemoryEfficientSageAttentionPatch` 设计；
- **ComfyUI / `comfy_kitchen` & Sol-Attn (arXiv:2607.24027)**：底层 INT8 ConvRot CUTLASS 核、`sol_attn` 稀疏 SageAttention CUDA 核及 H3 `2d_frame` Morton 对齐思路；
- **[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)**（**T8mars**）：上游 H3 音视频双时钟与 3D 潜空间放大器运行时；
- 历史分块硬切与缓存设计致谢（AIMixer / MAINodes / PERSIST / StreamingT2V 等）完整保留于 [`docs/README_LEGACY_HARDCUT.md`](docs/README_LEGACY_HARDCUT.md)。

**License**: GPL-3.0-or-later（见 [`LICENSE`](LICENSE)）。
