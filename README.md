# comfyui-h3-seamkit (v2.3)

**面向 8GB 消费级显卡（RTX 4060 Laptop 实测验证 10s/15s 原生 1080p 与真 2.7K 全程 0 OOM，同时无缝兼容 Colab G4 / A100 / H100 / 4090 / 5090）的 MiniMax H3 单窗加速与高锐度推理套件：零分窗拼缝、16-bit BF16 低秩 LoRA 旁路、Sol-Attn 稀疏加速 + 末步稠密/Tight-Sol、1080p 分频段 RoPE 解压缩、`Sol-H3-VAE` 8GB 安全分块解码与 RelayVSR x2 真 2.7K 工作流。**

> **100% 独立自包含 · 零侵入第三方插件**：本仓库**不修改** `ComfyUI` 核心、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 或任何第三方包的源码；全部内核与调度逻辑封装在 [`chunked_sage.py`](chunked_sage.py) 与同目录节点内，上游更新零冲突。
>
> - **数学原理与全套推导公式**：[`docs/PRINCIPLES_AND_FORMULAS.md`](docs/PRINCIPLES_AND_FORMULAS.md)（显存削峰、耗时拟合、BF16 LoRA 旁路精度实证、`0.57` 断崖修复、1080p 分频段 RoPE 解压缩与 `Sol-H3-VAE` 原理）。
> - **A100 24 小时小 LoRA / 放大器训练任务书**：[`docs/A100_LORA_TRAINING_PLAN.md`](docs/A100_LORA_TRAINING_PLAN.md)（80GB / 40GB 双排班与现成训练器指南）。
> - **旧版 v1.4 多窗硬切文档归档**：[`docs/README_LEGACY_HARDCUT.md`](docs/README_LEGACY_HARDCUT.md)（多窗与硬切能力仍完整保留）。
> - 运行时依赖上游 [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)（GPL-3.0-or-later）；本仓库许可 **GPL-3.0-or-later**。

---

## 核心特性（v2.2）

1. **单窗分块 QKV 低显存引擎（`MiniMaxH3ChunkedSageAttentionPatch`）**：
   - 一次性 Token 分块 ConvRot 预量化 + 头分组零拷贝切片（自动按 `2 头 = 256` ConvRot 块对齐）+ 首尾 FP32 I/O 层分块，10s `864p` 激活峰值从 `6.70 GiB` 压至 **`1.98 GiB`**；
   - `exact` 模式（显存 $\ge 16\text{ GiB}$ 如 A100/4090 或短序列自动启用）与原版 KJ **100.00% 逐位一致**；`streaming` 模式（8GB 卡长序列自动启用）零 `[S, 7168]` 输出缓冲，15s 单窗直出零拼缝。
2. **Packed-Rank 16-bit BF16 低秩 LoRA 旁路 + 融合内核（第一拳）**：
   - 自动拦截标准 2D LoRA（兼容前/后接 `LoraLoaderModelOnly` 与 `LoraLoaderBypassModelOnly`），彻底免掉每步 200 层 VBAR INT8 反量化/重定量化（10s `864p` 实测提速 **`-54.4s`**）；
   - 将微弱锐化 LoRA（如 `minimax_h3_lms_v1.0_r64_r @ 0.18`）的 16-bit 保真度从旧版 INT8 重定量化的 **`cos = 0.016`（422% 误差）** 拉升至 **`cos = 0.999996`（0.27% 误差）**，叠挂 `1.18 GB` rank-64 LoRA 全程仅增加 **`+1.9s`**！
   - 集成原地 CUTLASS `cutlass_int8_dequant_residual` 残差融合、权重折叠 `_ck.rms_adaln`（`3.0×` 提速）、单遍融合 Triton SwiGLU 与 `torch.addmm` 门控累加。
3. **Sol-Attn 稀疏加速 + `sol_min_sigma` 末步全稠密回收**：
   - 内置零峰值列分块 `2d_frame` Morton 重排与条件/音频 Sink 保护；前 2 步稀疏加速 + 末步自动切回 100% 稠密 KJ SageAttention2++，兼得速度与锐度。
4. **1080p 分频段空间 RoPE 解压缩 & 注意力温度补偿（可选高级参数）**：
   - `rope_hf_scale`（1080p 推荐 `1.32`）：仅解压缩视频高频空间频段 `k=0..5`（`k=6..7` 平滑过渡），不动低频构图、时间轴与双声道音频对齐；
   - `attn_temp_scale`（1080p `154k` token 推荐 `1.08`）：抑制超长序列 softmax 熵稀释；零额外耗时与显存开销。
5. **统一极简 `[SeamKit]` 日志体系 & 硬件跨平台兼容**：
   - 全仓日志前缀统一为 `[SeamKit]`，单窗模式下自动静默历史切片/overlap 冗余打印；
   - 原生覆盖 **SM75 / SM80 (A100) / SM86 / SM89 (4060/4090) / SM90 (H100) / SM120 (5090)**，支持 `minimax_h3_sol_vae_fp16` 与全精度 BF16 底模。

---

## 实测成绩单（RTX 4060 Laptop 8GB / 32GB RAM，真实 `dasiwa` INT8）

### 1. 10s `1568×864`（243 帧）严格同底 A/B 实测（全部 0 OOM）

| 组别 | 核心配置 | 二采 3 步耗时 (`s/it`) | 二采总耗时 | 显存峰值 | `LapVar` 锐度 | 高频 `>0.25%` | 边缘光晕 `Halo` | 时序抖动 `Jitter` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Dense 旧基线** | 全稠密 + 旧 `fp16` 放大器 + `0.57` 断崖 | `219 / 219 / 219` | `11m10s` | `7.3 GiB` | `125.3` | `0.231%` | `1.01×` | `3.51` |
| **Exp B（旧引擎）** | 锐度放大器 + `0.85/0.55/0.25` + 末步稠密 (`hc=14`) | `136.7 / 134.6 / 245.0` | `8m43s` | `7295 MiB` | `196.9` (`+57.1%`) | `0.450%` | `1.00×` | `3.32` |
| **P1-A（第一拳纯引擎）** | 同 Exp B，切 BF16 旁路 + 融合残差 + `hc=7` | **`119.1 / 113.0 / 231.0`** | **`7m48s` (`-54.4s`)** | `7653 MiB` | `200.4` (`+59.9%`) | `0.458%` | `1.00×` | `3.33` |
| **P1-B（+二采 `LMS 0.18`）** | P1-A + 叠挂 `1.18GB` rank-64 `LMS 0.18` | **`119.1 / 114.7 / 232.2`** | **`7m50s` (仅 `+1.9s`!)** | `7415 MiB` | `202.7` (`+61.8%`) | `0.463%` | `1.00×` | `3.31` |
| **P1-C（默认推荐）** | P1-B + `shift=6.5` + `0.80/0.48/0.18` + `sol_min=0.25` + `ref=match` | **`121.6 / 113.9 / 231.9`** | **`7m52s` (`-50.8s`)** | `7471 MiB` | **`216.6` (`+72.9%`)** | **`0.493%` (`2.13×`)** | **`1.02×`（零白边）** | **`3.28`（最稳）** |

### 2. 原生 1080p (`1984×1120`) 破壁实测 & 2K/3K 超分正面 PK（RTX 4060 8GB，全部同缩至 `1920×1088` 评测）

| 路线（10s / 243 帧，全部 0 OOM） | 原生画布 | 二采 + 解码 (+VSR) | `@1080p LapVar` | `@1080p HF>0.25%` | `Face120` | 边缘光晕 `Halo` | 时序抖动 `Jitter` |
|---|---|---:|---:|---:|---:|---:|---:|
| **旧版原生 1080p (All-Sol)** | `1920×1088` | `~13m` | `57.0` | `0.096%` | `161.3` | `20.3` | `3.85` |
| **P1-C 864p → 拉伸 1080p** | `1568×864` | `~9m 20s` | `104.3` | `0.207%` | `183.8` | `22.6` | `3.28` |
| **Exp 1 新版原生 1080p+**（Split-RoPE `1.32` + Temp `1.08` + `LMS 0.50` + `Sol-H3-VAE` + 末步全稠密） | **`1984×1120` (`2.22MP`, 负载 `540`)** | `17m 07s` + `1m 42s` (`20m 22s`) | **`122.1` (`2.14×` 旧1080p, `+17.1%` 超864p)** | `0.265%` (`2.76×` 旧1080p) | `226.9` (`+23.4%`) | **`20.2`（最低）** | **`3.14`（最稳）** |
| **864p + 一步 `RelayVSR x2 (d=0.50)` → 降采 1080p** | `3136×1728` | `~10m 50s` (`VSR 1m 45s`) | `146.4` (`2.57×`) | `0.249%` | **`331.2`** | `21.7` | `3.56` |
| **Exp 2 性价比之王：`768p` + `LMS 0.50` + `Sol-H3-VAE` + `RelayVSR x2 (d=0.60)` → 降采 1080p** | **`2752×1536` (真 2.7K)** | **`5m 29s` + `48s` + `1m 40s` (`9m 10s`, 比 Exp 1 快 `2.23×`!)** | **`179.2` (`3.14×` 旧1080p, `+46.8%` 超 Exp 1!)** | **`0.299%`（全场最高!）** | **`328.3`** | `23.3` | **`3.30`（稳于 864p VSR）** |

- **15s 原生 `1920×1088`（362 帧单窗，负载 `756.2` = 旧 8GB OOM 锚的 `3.96×`！）**：开启 `hc=28` + `sol_low_sigma_tau=0.90`（Tight-Sol 末步精修）+ `Split-RoPE 1.32` + `LMS 0.50` + `/free` + `MiniMaxH3SafeAVDecode` 2 段 `Sol-H3-VAE` 解码，全程 **0 OOM**（DiT 峰值 `7909 MiB` / `22m 51s`，2 段解码峰值 `7535 MiB` / `2m 47s`），`@1080p LapVar = 111.6`（旧版 15s `864p` 的 **`2.20×`**）、`HF>0.25% = 0.242%`（`2.75×`）、`Jitter = 2.77` 且 `[170..186]` 过渡帧零接缝！
- **零背景参考（仅留人物参考图 `deepseek.png`，新场景玻璃花房 `beta57` 一采 + 完整日语对白）**：
  - `NoBg-1`（10s 原生 `1920×1088`）：二采提速至 `14m 59s`，`@1080p LapVar` 暴涨至 **`188.3`（较带糊背景参考图的 Exp 1 再涨 `+54.2%`！）**、`Jitter = 1.13` 且清晰保留远景灯塔；
  - `NoBg-2`（10s `768p` + `RelayVSR x2 d=0.60` → `2752×1536`）：全程仅 **`8m 10s`**，`@1080p LapVar = 258.8`、`HF>0.25% = 0.346%`；
  - `NoBg-3`（15s 原生 `1920×1088` 3 镜）：`f117` / `f233` 精准双切、`@1080p LapVar = 145.3`、`Halo = 20.8`、`Jitter = 1.22`，2 段 `Sol-VAE` 零接缝。
- 耗时经验拟合公式与全部推导过程请见 [`docs/PRINCIPLES_AND_FORMULAS.md`](docs/PRINCIPLES_AND_FORMULAS.md#2-rtx-4060-8gb-实测耗时拟合模型与1080p-为什么慢的定量解剖)。

---

## 快速上手

1. 克隆到 `ComfyUI/custom_nodes/`（需已安装 `comfyui-minimax-h3-audio-T8`、`sageattention` 与新版 `comfy_kitchen`）：
   ```bash
   git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git
   ```
2. **推荐直接拖入 v2.3 封装工作流**：[`examples/workflow_singlewin_sharp_vsr2k.json`](examples/workflow_singlewin_sharp_vsr2k.json)
   - 默认冠军管线：`0.48MP (960×544)` `beta57` 纯稠密一采 → 锐度版 3D 放大器 → `1.00MP (1376×768)` 单窗二采（含二采专用 `LMS 0.42` BF16 旁路、`hc=7`、前 2 步稀疏/末步全稠密、内置 `Split-RoPE` 与 `Tight-Sol` 原生 1080p 旋钮）→ `MiniMaxH3SafeAVDecode` (`Sol-H3-VAE FP16`) → 一步 `RelayVSR x2 (d=0.60)` → **`2752×1536` 真 2.7K + H3 原生音频**；
   - 10s 二采+解码+超分在 4060 上仅约 **`8–9 分钟`**，32GB 内存峰值仅 `~26.5 GB`；
   - 也可使用纯单窗无超分工作流 [`examples/workflow_singlewin_chunkedsage.json`](examples/workflow_singlewin_chunkedsage.json)。
3. **大显存卡（Colab G4 / A100 / H100 / 4090 / 5090）调优建议**：
   - `out_proj_mode="auto"` 会自动切回 `exact` 逐位一致模式；建议将 `head_chunks` 设为 `1` 或 `2`，大显存卡亦可直接换用官方 BF16 剪枝底模与 `minimax_h3_sol_vae_fp16.safetensors`。
4. **8GB 卡 + 32GB 内存如遇 `hostbuf_grow` / `pinned memory` 报错**：运行 `tools/pinned_memory_patch.py`（预设 3），并确保 ComfyUI 启动参数带 `--disable-async-offload`。

---

## 节点一览（共 12 个）

| 类别 | 节点 | 核心职责 |
|---|---|---|
| **主线加速** | **`MiniMaxH3ChunkedSageAttentionPatch`** | **核心主力**：分块 QKV + BF16 低秩 LoRA 旁路 + CUTLASS 融合残差 + Sol-Attn 稀疏/末步稠密/Tight-Sol + 分频段 RoPE 解压缩 + 首尾 FP32 分块 |
| **安全解码** | **`MiniMaxH3SafeAVDecode`** | **Sol-H3-VAE 8GB 守护神**：10s 1080p 单遍零开销直出；15s/20s 1080p 自动按 5-latent 块网格分 2~3 段并在保护块平滑融合，彻底消灭 8GB Sol-VAE OOM |
| **双采规划与缓存** | `MiniMaxH3HardCutFirstPassPlan` / `Pass2Plan` / `FirstPassSampler` / `HardCutUpscale` / `AVLatentSave` / `Load` | 上下游分辨率与 17n+5 帧长规划、一采/二采 Latent 存读档（支持两段 `/free` 内存解耦）、3D 锐度放大器 `bf16 + pad=3` 调度与单窗/多窗执行 |
| **实验套件** | `MiniMaxH3Route3AutoMask` / `Apply` / `Assemble` / `Probe` | 早期前/背景双流压缩实验套件（常规场景已由 `ChunkedSage + Sol-Attn` 全面替代） |

---

## 仓库目录与回归测试

```text
comfyui-h3-seamkit/
├── chunked_sage.py              # 核心：分块 QKV + BF16 LoRA 旁路 + 融合核 + Sol-Attn + Tight-Sol + 分频段 RoPE
├── sol_vae_decode.py            # Sol-H3-VAE 5-latent 块对齐 8GB 安全分块解码节点
├── h3_upscale.py                # 二采单窗/多窗执行器与统一 [SeamKit] 日志
├── nodes.py                     # 上下游双采规划节点
├── nodes_firstpass.py           # 一采可缓存采样器
├── nodes_latent_cache.py        # 冻结 AV latent 存/读档节点（支持 output 节点两段 /free 解耦）
├── hardcut_math.py / bridge.py  # 帧网格数学与 T8 运行时桥接
├── examples/                    # workflow_singlewin_sharp_vsr2k.json（v2.3 推荐）与旧版示例
├── route3/tests/                # 7 套全绿单元与 GPU 逐位一致性回归测试
├── docs/
│   ├── PRINCIPLES_AND_FORMULAS.md  # 核心原理、推导公式与全线实测总账（v2.3）
│   ├── A100_LORA_TRAINING_PLAN.md  # Colab G4/A100 双 Agent 协同蒸馏 LoRA 任务书
│   └── README_LEGACY_HARDCUT.md    # 旧版 v1.4 多窗硬切文档归档
└── tools/                       # pinned_memory_patch / 帧分析 / 缓存管理工具
```

运行全部 7 套单元与 GPU 回归测试：
```bash
for f in route3/tests/test_*.py; do PYTHONPATH=. python "$f" || exit 1; done
```

---

## 致谢与许可

- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)**（**Kijai**）：int64-safe per-thread INT8 Triton 量化核与 SM75–SM120 SageAttention2++ 分发契约；
- **ComfyUI / `comfy_kitchen` & Sol-Attn (arXiv:2607.24027) & Sol-H3-VAE**：底层 INT8 ConvRot CUTLASS 核（含原地残差 epilogue）、`rms_adaln`、`sol_attn` 稀疏核及 24L 非因果 FP16 视频解码器；
- **[comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)**（**T8mars**）：上游 H3 音视频双时钟与 3D 潜空间放大器运行时。

**License**: GPL-3.0-or-later（见 [`LICENSE`](LICENSE)）。
