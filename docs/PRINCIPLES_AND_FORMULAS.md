# comfyui-h3-seamkit 核心原理与数学公式详解（v2.7）

> **AI / 开发者 30 秒快速接手摘要**：
> - **项目目标**：在 **RTX 4060 Laptop 8GB / 32GB RAM**（同时原生兼容 **A100 / H100 / 4090 / 5090 / RTX PRO 6000**）上，实现 MiniMax H3 **10s–15s 单窗零拼缝直出（最高原生 `1920×1088` 与真 `2.7K / 3K`）**，并在高锐度、去油腻胶片质感与音画同步的前提下把 8GB 推理速度压到硬件物理极限。
> - **已攻克的六大核心模块**：
>   1. **8GB 单窗显存破壁**（[`chunked_sage.py`](../chunked_sage.py)）：分块 ConvRot 预量化 + 头分组零拷贝 QKV + `streaming` 零输出缓冲，`15s @ 1920×1088`（负载 `756.2 = 3.96×` 旧 OOM 红线）0 OOM；
>   2. **双路 16-bit BF16 低秩 LoRA 旁路 + 算子融合**：同时覆盖一采与二采（消除 200 层 INT8 VBAR 反量化/重量化，保真度 `cos: 0.016 → 0.999996`），配合 `hc=4, token_chunk=16384` 消除 `1.1 TB` 残差带宽碎片，10s 原生 1080p 直降 `~4 分钟`；
>   3. **Sol-Attn 稀疏加速 + `2d_frame` Morton 重排 + 条件 Sink 保护**：在 `S = 147k` 时将注意力提速 **`8×`**（`285s → 36s/步`），并支持末步全稠密 / Tight-Sol / 自训 LoRA 全步稀疏三档策略；
>   4. **面积归一化 Split-Band RoPE 解压缩 & 温度补偿**：按 $\gamma_{\text{hf}} = (\sqrt{W\cdot H}/1024)^{0.80}$ 仅补偿视频高频空间频段，零额外耗时；
>   5. **`Sol-H3-VAE (FP16)` 5-Latent 块对齐安全分块解码**（[`sol_vae_decode.py`](../sol_vae_decode.py)）：`MiniMaxH3SafeAVDecode` 在 `15s 1080p` 自动 2 段解码并在保护块交叉融合，0 接缝 0 OOM；
>   6. **通用 200 层稀疏保锐 + 高低频解耦去油自蒸馏 LoRA**（[`train/train_h3_lora.py`](../train/train_h3_lora.py)）：`0` `adaln` 全底座通用、可微静态 `flex_attention` 空间邻域+质心余弦路由、保方向 L2 `_GradClamp`、以及 **Scheme 2A 潜空间 `5×5` 低频基座锚定去油目标**（`0.0 GB` 额外峰值显存，25 步即把低频油光误差压降 `91%`）。
> - **零侵入原则**：本仓库不修改 `ComfyUI` 核心、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 任何一行源码。

---

## 1. 为什么 8GB 显存跑 10s/15s 单窗会 OOM？分块 QKV 与 Streaming 显存公式

MiniMax H3 主干包含 50 层 DiT Block（隐藏维 $H = 5376$，注意力头数 $N_h = 56$，每头维度 $D = 128$，内维 $I = N_h \cdot D = 7168$，SwiGLU 中间维 $F = 14336$）。
视频潜空间下采样倍率为空间 $16\times$、时间 $17k+5 \leftrightarrow 5k+2$，DiT 空间 patch 大小为 $2\times 2$（每 DiT 视频 token 对应 $32\times 32$ 像素），因此单帧视频 token 数与总序列长度 $S$ 为：

$$N_{\text{tok/frame}} = \frac{W_{\text{px}}}{32} \cdot \frac{H_{\text{px}}}{32}, \qquad S = N_{\text{text}} + N_{\text{ref}} + N_{\text{audio}} + T_{\text{lat}} \cdot N_{\text{tok/frame}}$$

典型画布的精确网格、序列长度与负载值（$\text{Load} = \text{Frames} \times \text{MP}$）：
- `5s @ 1920×1088 (124f, T_lat=37)`：$S \approx 75{,}480$，Load `259`
- `10s @ 1376×768 (1.00MP 档, 243f, T_lat=72)`：$S \approx 78{,}000$，Load `257`（H3 原生 RoPE 甜点）
- `10s @ 1568×864 (1.30MP 档, 243f, T_lat=72)`：$S \approx 96{,}000$，Load `329`（**10 分钟高速高清甜点 / VSR 3K 底座**）
- `10s @ 1920×1088 (1.98MP 档, 243f, T_lat=72)`：$S \approx 146{,}880$（多图参考可达 `154k`），Load `508`
- `15s @ 1920×1088 (362f, T_lat=107)`：**$S \approx 222{,}000$**，Load **`756.2`（旧版 8GB OOM 锚 `191` 的 `3.96×`！）**

### 1.1 原版 KJ 低显存补丁的三大显存尖峰（BF16 = 2 字节，FP32 = 4 字节）
1. **整序列 QKV 缓冲区**：`qkv_proj(h)` 一次性物化 `[S, 3I] = [S, 21504]` BF16：
   $$M_{\text{QKV}} = 2 \cdot S \cdot 21504 \approx \mathbf{3.85\text{ GiB}}\;(96\text{k}),\quad \mathbf{5.89\text{ GiB}}\;(147\text{k}),\quad \mathbf{8.90\text{ GiB}}\;(222\text{k})$$
2. **残差 + 归一化 + 输出缓冲并发驻留**：`x [S, 5376]` + `h [S, 5376]` + `out [S, 7168]`（$S=222\text{k}$ 时仅此三块即超 `7.5 GiB`）。
3. **首尾 FP32 I/O 层**：`video_patch_proj` 与 `FinalLayer._modulate` 一次性物化 `[N_v, 5376]` FP32（`1.96–4.37 GiB`）。

### 1.2 `chunked_sage.py` 的三级显存削峰与 ConvRot-256 对齐条件
1. **一次性 Token 分块 ConvRot 预量化**：按 `token_chunk_size`（推荐 `16384`）分块执行 `RMSNorm + AdaLN + Hadamard 旋转 + 行级 INT8 量化`，只落盘 `(x_qdata [S, 5376] int8, x_scale [S, 1] fp32)`（`96k` 仅 `0.49 GiB`，`147k` 仅 `0.74 GiB`，永不物化整条 BF16 `h`）。
2. **头分组切片 QKV + 延迟投影 V**：把 56 个头分为 $G$ 组（每组 $g$ 个头），严格满足 Hadamard 块对齐约束：
   $$(g \cdot 128) \bmod 256 = 0 \iff g \text{ 为偶数（} G=4 \Rightarrow g=14,\; G=7 \Rightarrow g=8,\; G=14 \Rightarrow g=4,\; G=28 \Rightarrow g=2\text{）}$$
   每组零拷贝切片投影 `Q_g, K_g` → 原地 RMS-RoPE → INT8 量化并释放 BF16 `Q_g, K_g`，再投影并量化 `V_g` → 调起 SageAttention2++ / Sol-Attn。
3. **双 `out_proj` 模式与带宽优化（`hc=4` vs `hc=14`）**：
   - **`exact` 模式**（显存 $\ge 16\text{ GiB}$ 如 A100/4090 或 $S \le 64\text{k}$ 自动启用）：保留 `[S, 7168]` 输出缓冲，与原版 KJ **100.00% 逐位一致**（`max_abs = 0.0`）。
   - **`streaming` 模式**（8GB 卡长序列自动启用）：每组头输出 `o_g [S, g·128]` 直接乘 `out_proj` 列切片 `W[:, c0:c1]` 并原地累加进残差 `x`，彻底消灭 `[S, 7168]` 缓冲（单层余弦精度 `cos = 0.999965`）。
   - **碎片带宽公式（v2.7 核心提速发现）**：在 `streaming` 模式下，每组头都会对残差 `x`（`1.58 GB @ 147k`）做一次读写累加，因此 50 层每步由 `out_proj` 产生的额外残差 HBM 流量与核函数启动次数为：
     $$\text{Traffic}_{\text{resid}}(G) = 50 \cdot G \cdot 2 \cdot (2 S H), \qquad N_{\text{launches}}(G, C_{\text{tok}}) = 50 \cdot G \cdot \left\lceil\frac{S}{C_{\text{tok}}}\right\rceil$$
     - 旧配 `G=14, C_tok=8192`（$S=147\text{k}$）：每步高达 **`12,600` 次**窄矩阵（512 列）核启动、**`2.2 TB`** 残差读写；
     - 升级 **`G=4, C_tok=16384`**（每组 14 头 = `1792` 列 = `7×256` 严格对齐）：核启动骤降至 **`1,800` 次（`-85.7%`）**、残差读写降 **`3.5×`**，100% 数值等价下 3 步立省 **`~40s`**。

### 1.3 精确单块峰值显存解析公式与全规格显卡自适应求解器（v2.8）
[`_estimate_block_peak_bytes`](../chunked_sage.py) 与 [`_resolve_vram_plan`](../chunked_sage.py) 直接对单层 DiT Block 所有并发驻留张量逐字节精确求和（并零显存调用 `comfy_kitchen._C.sol_attn_plan(1, S, g, token_aug)` 读取真实 Sol-Attn 工作区字节数）：

$$M_{\text{base}}(S) = S \cdot \big(2H + H + 4 + 768 + 2(R_{\text{qkv}} + R_{\text{out}})\big) \approx 16{,}900\, S + 2 R_{\text{sum}} S$$

$$M_{\text{attn}}(S, g) = M_{\text{base}}(S) + \mathbb{I}[\text{exact}]\cdot 2 N_h D S + \begin{cases} 4 \cdot 2 g D S + W_{\text{sol}}(S, g, \text{aug}) & (\text{Sol-Attn}) \\ 6 g D S + 16 g S & (\text{Dense Sage}) \end{cases}$$

$$M_{\text{mlp}}(S, C_{\text{tok}}) = 2 H S + 2 \cdot (3 F) \cdot \min(S, C_{\text{tok}}) = 10{,}752\, S + 86{,}016\, \min(S, C_{\text{tok}})$$

主流显存规格 **防 WDDM 共享显存静默溢出（Sysmem Fallback）** 激活预算——实测在 RTX 4060 8GB 上，`10s hc=4 (6.00 GiB)` 与 `15s hc=14 (5.20 GiB)` 均满功耗（`82°C`）运行，而 `15s hc=7 (6.64 GiB)` 会把专用显存顶至 `7.6/8.0 GB` 并触发 NVIDIA WDDM 静默溢出到系统内存（不报 OOM 但 GPU 跌至 `29W / 58°C` 死等 PCIe）：

$$B_{\text{act}}(V_{\text{total}}) = \max\!\big(2.0\text{ GiB},\; 0.88\, V_{\text{total}} - 0.90\text{ GiB}\big)$$

求解器在均匀 ConvRot-256 约数集合 $G \in \{1, 2, 4, 7, 14, 28\}$（保证每组张量形状完全一致、零 `cudaMallocAsync` 池碎片）中选择满足 $M_{\text{attn}}(S, 56/G) \le B_{\text{act}}$ 的**最小（最快）$G$**，且仅当 $M_{\text{mlp}} > B_{\text{act}}$ 时才缩减 `token_chunk_size`（设 `head_chunks = 0` 即全自动最优）：

| 显卡规格 ($V_{\text{total}}$) | 激活预算 $B_{\text{act}}$ | 5s 1080p ($79\text{k}$) | 10s 1080p ($153\text{k}$) | 15s 1080p ($224\text{k}$) | 20s 1080p ($296\text{k}$) |
|---|---:|---|---|---|---|
| **6 GB** | `4.38 GiB` | `hc=4` (`3.06 GiB`) | `hc=14` (`3.61 GiB`) | `hc=28` (`4.61 GiB`) | 需降分辨率 |
| **8 GB (4060/3070)** | **`6.14 GiB`** | **`hc=2`** (`4.76 GiB`) | **`hc=4, tc=16k`** (`6.00 GiB`) | **`hc=14, tc=16k`** (`5.20–5.32 GiB`) | **`hc=28, tc=16k`** (`6.11 GiB`) |
| **10 GB (3080)** | `7.90 GiB` | `hc=2` (`4.76 GiB`) | `hc=4` (`6.00 GiB`) | **`hc=7, tc=16k`** (`6.76 GiB`) | `hc=14` (`7.08 GiB`) |
| **12 GB (4070/3060)** | `9.66 GiB` | `hc=1` (`8.14 GiB`) | **`hc=2`** (`9.35 GiB`) | **`hc=4, tc=16k`** (`8.92 GiB`) | `hc=7` (`9.02 GiB`) |
| **16 GB (4080/4060Ti)** | `13.18 GiB` | `exact hc=1` (`9.20 GiB`) | **`exact hc=2`** (`11.39 GiB`) | **`exact hc=4`** (`11.91 GiB`) | `streaming hc=4` (`11.93 GiB`) |
| **24 GB (3090/4090)** | `20.22 GiB` | `exact hc=1` (`9.20 GiB`) | **`exact hc=1`** (`18.09 GiB`) | **`exact hc=2`** (`16.94 GiB`) | `exact hc=4` (`15.88 GiB`) |

*(上表按 `Sol-Attn (aug=64)` + 重载 LoRA 栈 `R_sum = 864` 计算；无 LoRA 或 `dense_sage` 时峰值还会更低。)*

---

## 2. RTX 4060 8GB 实测耗时拟合与「为什么原生 10s 1080p 有硬件物理地板」

Transformer 单步前向严格由 **PCIe 卸载搬运常数项**、**$O(S)$ 线性投影/MLP/LoRA 项** 与 **$O(S^2)$ 注意力二次项** 构成。由 RTX 4060 Laptop 8GB 在真实 `dasiwa` 50 层 INT8 模型上的常驻基准与 `comfyui.log` 实测反解：

$$T_{\text{step}}^{\text{dense}}(S) \approx 24\text{s}_{\text{PCIe}} + 29.4\text{s} \cdot \left(\frac{S}{48\text{k}}\right)_{\text{Linear/MLP}} + 35.2\text{s} \cdot \left(\frac{S}{48\text{k}}\right)^2_{\text{Dense SageAttn}}$$

开启 **`Sol-Attn (tau=1.3~1.4, aug=64)`** 后，稀疏注意力项降低 **`4×–8×`**（$S=147\text{k}$ 时从 `~285s` 压至 `~36s`）：

$$T_{\text{step}}^{\text{sol}}(S) \approx 24\text{s}_{\text{PCIe}} + 31\text{s} \cdot \left(\frac{S}{48\text{k}}\right)_{\text{Linear/MLP}} + T_{\text{LoRA}}(S, R) + (4\text{–}9)\text{s} \cdot \left(\frac{S}{48\text{k}}\right)^2_{\text{Sol-Attn}}$$

### 2.1 `10s @ 1920×1088`（$S=146{,}880$）单步 `215s` 的硬件算力硬账（Roofline 解剖）
50 层 DiT 在 $S = 146{,}880$ 下**每跑 1 步**必须完成固定量的矩阵乘（与注意力是否稀疏无关）：
- **50 层 INT8 线性投影**（`qkv [21504,5376] + out [5376,7168] + fc1 [28672,5376] + fc2 [5376,14336]`）：
  $$\text{OPS}_{\text{INT8}} = 50 \cdot 2 \cdot S \cdot \big(5376(21504+7168+28672+14336)\big) \approx \mathbf{5{,}660\text{ TOPS}}$$
  在 4060 Laptop 持续功耗墙下（实测有效 ~44 TOPS INT8）硬耗时 **`~129s/步`**；
- **打包 BF16 LoRA 旁路**（`pika r=384` + `电影质感 r=16` + `自训 r=32`，合计 `R_qkv = 432`）：约 **`343 TFLOPS` BF16** + Norm/SwiGLU/Morton，耗时 **`~26s/步`**；
- **20GB INT8 权重 PCIe 卸载搬运**：`19.53 GiB` 每步硬下限 **`~24s/步`**；
- **Sol-Attn 稀疏注意力 (`tau=1.30`)**：仅占 **`~36s/步`**（相比稠密 `~285s` 已提速 **`8×`**！）。

> **物理结论**：在 RTX 4060 Laptop 8GB 上，就算注意力降为 `0 秒`，`146,880 tokens` 单步的线性层+LoRA+PCIe 物理地板也高达 **`~175s`（~3 分钟/步）**！因此：
> - **10s 真原生 `1920×1088`** 物理最快是 **`~14.5–15.3 分钟`（3 步）** / **`~11.5–12 分钟`（2 步）**；
> - **想要 10s 进 `~9–11 分钟` 大关**：必须降低二采 token 数——最香甜点是 **`1.30 MP (1568×864, S≈96k)` + 一步 `RelayVSR x2` 直出 `3136×1728` 真 3K（`~10–11 分钟`）**，或 **`1.00 MP (1376×768, S≈78k)` + `RelayVSR x2` 真 2.7K（`~8–9 分钟`）**！

### 2.2 v2.7 三刀消除 `~3.5–4 分钟` 纯软件水分（`19m01s → ~15m`）
1. **第一刀 — 一采补齐 BF16 LoRA 旁路（`-55~60s`）**：旧版一采 `#16` 走 KJ 原版补丁，导致 `pika + 电影质感` 在一采挂载 `258 patches`、4 步触发 800 次 INT8 VBAR 反量化/重量化（`4m08s`）。替换为独立一采分支 `ChunkedSage (hc=7, tc=16384, sol_tau=1.20, sol_max_sigma=0.90, rope=1.0, temp=1.0)` 后，第 0 步 `σ=1.0` 仍 100% 稠密，`patches` 从 `258 → 58`，一采降至 **`~3m10s`**。
2. **第二刀 — 自训 LoRA 解锁全步统一稀疏（`-85s`）**：旧版 `sol_min_sigma=0.25, low_tau=0.90` 在第 3 步（`σ=0.20`）强切 Tight-Sol（`tau=0.90, aug=128` + 首尾 6 层稠密），使第 3 步暴涨 `+63.4s`（`215s → 278.4s`）。挂载专门在稀疏下训的自训 LoRA 后设为 `sol_min_sigma=0.0, low_tau=0.0, sol_tau=1.40`，立省 `~85s`。
3. **第三刀 — 消除 `out_proj` 碎片带宽（`-40s`）**：`hc=14 → 4`、`token_chunk_size=8192 → 16384`，100% 数值等价。

---

## 3. Sol-Attn 稀疏加速、`2d_frame` Morton 对齐与三档低噪调度

在 [`chunked_sage.py`](../chunked_sage.py) 中，`comfy_kitchen.sol_attn` 直接嵌入头分组循环内：
1. **零额外显存的 `2d_frame` Shifted-Morton 四相错位重排（v2.8 治本消缝）**：
   - 视频 token 按帧内二维 Z-order 曲线重排使空间邻近 patch 落入同一 64-token（`8×8` patch = `256×256 px`）块；按列分块（`col_chunk = 1344`）原地置换，结束时逆置换还原。
   - **四相错位消缝（Swin 思想）**：若连续 50 层固定 `(ox, oy) = (0, 0)`，在 `1920×1088`（`60×34` patch）上 `y = 256 / 512 / 768 / 1024` 与 `x = 256 / 512 / 1024 / 1536` 永远处于超级象限块边界，全步稀疏下 `sol_attn` 块级尾项偏置 $\bar{v}_{\text{tail}}(B)$ 叠加 50 层会在固定像素线上形成肉眼可见的贯通阶梯割裂（实测 `y=512` 跳变 `+2.69`、`y=1024` `+2.80`）。现按 `phase = block_idx & 3` 在 `(0,0), (4,4), (2,6), (6,2)` 四相间轮换相对置换（RoPE 之后原地分头置换 `Q/K/V`、注意后逆置换 `O`，置换与注意力 100% 代数自洽），使任意 `256/512/1024 px` 网格线在 **4 层中有 3 层位于 `8×8` 块正中心**，零峰值显存增量且每步仅增 `~2–4s`（配合 `sol_dense_blocks="48-49"` 末 2 层稠密收口可 100% 根治）。
2. **条件与音频 Sink 严格保护**：视频起始偏移 `va` 按 64-token 向上对齐，通过 `sink_blocks` 与 `sink_q` 保证文本、参考图与音频查询行 100% 稠密注意，杜绝音画漂移。
3. **三档低噪调度**：
   - **自训 LoRA 全步稀疏（v2.7 默认，`sol_min_sigma=0.0, low_tau=0.0, sol_tau=1.40`）**：全 3 步极速稀疏，靠自训 LoRA 补齐高频锐度；
   - **无自训 LoRA 末步全稠密回收（`sol_min_sigma=0.25, low_tau=0.0`）**：前 2 步稀疏定型 + 末步 100% 稠密精修（适合 `768p/864p` 裸跑）；
   - **15s 超长片 Tight-Sol（`sol_min_sigma=0.25, low_tau=0.90`）**：首尾各 3 层全稠密 + 中间 44 层 `tau=0.90, aug>=128`，15s 1080p 末步较全稠密省 `7.2 分钟`。

---

## 4. Packed-Rank BF16 低秩 LoRA 旁路与算子融合

### 4.1 为什么 ComfyUI 原生 INT8 重定量化会“碾碎”微弱锐化/蒸馏 LoRA？
低显存模式下 ComfyUI 每步对 200 个 INT8 线性层重复执行：

$$\widetilde{W}_{\text{int8}} = \mathcal{Q}_{\text{INT8,ConvRot}}\!\left(\mathcal{D}(W_{\text{int8}}) + \sum_i s_i B_i A_i\right)$$

除白烧 `10–16s/步` 与 `~350 MiB` 缓存外，当 LoRA 幅度较弱时（如 `LMS @ 0.18` 或自训 LoRA），增量 $\|s B A\|$ 远小于底模通道 INT8 量化步长 $\Delta_q = \max|W_{\text{row}}|/127$，重定量化后被舍入噪声淹没：

| 实现方式（`dasiwa blocks.0.attn.qkv_proj` + `LMS @ 0.18` vs 精确 FP32） | 余弦相似度 $\cos$ | 相对 $L_2$ 误差 |
|---|---:|---:|
| 旧版 ComfyUI INT8 重定量化 (`want_requant=True`) | **`0.016058`** | **`422.25%`**（几乎纯噪声） |
| **本仓库 BF16 低秩旁路 (`bf16_lora=200/200`)** | **`0.999996`** | **`0.27%`**（无损保真） |

### 4.2 代数等价的 Packed-Rank 旁路与四项 CUDA/Triton 融合
自动拦截 200 个 DiT 线性层（`blocks.{0..49}.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}`）的标准 2D LoRA，拼接为单个紧凑低秩对：

$$A_{\text{cat}} = \begin{bmatrix} s_1 A_1 \\ s_2 A_2 \\ \vdots \end{bmatrix} \in \mathbb{R}^{R \times K}, \qquad B_{\text{cat}} = \begin{bmatrix} B_1 & B_2 & \cdots \end{bmatrix} \in \mathbb{R}^{N \times R}$$

1. **原地 `torch.addmm` 累加**：$Y \leftarrow Y + (X A_{\text{cat}}^\top) B_{\text{cat}}^\top$ 直接在 cuBLAS epilogue 写回，消除 `448 MiB` 临时缓冲，多挂一张 `r=32/64` LoRA 全程仅增 `<1.9s`；
2. **门控折叠残差累加**（`out_proj / mlp.fc2`）：$X \leftarrow X + \text{addmm}(X,\; M,\; (B_{\text{cat}} \odot g_r[:, \text{None}])^\top)$；
3. **单遍融合 Triton SwiGLU** + **原地 CUTLASS `cutlass_int8_dequant_residual`** + **权重折叠 `_ck.rms_adaln`**（$s_{\text{eff}} = w \odot (1+s) - 1$，提速 `3.0×`）。
4. **双路链式继承**：一采与二采各自独立挂载 `ChunkedSage` 或串行挂载时，`_is_chunked_sage_run_blocks` 会自动继承上游 `static_loras` 且永不双重包裹 `_run_blocks`。

---

## 5. 二采画质破壁与长片解码的四大物理公式

### 5.1 Flow-Matching 偏移公式与「`0.57` 断崖」
MiniMax H3 使用带偏移 $s$ 的 Flow Matching 噪声表：

$$\sigma(t) = \frac{s \cdot t}{1 + (s - 1)\, t}, \qquad t \in [0, 1]$$

旧版 `BasicScheduler(steps=3, denoise=0.30, shift=12)` 采样 $\sigma = [0.8372,\; 0.7500,\; \mathbf{0.5714},\; 0.0]$，末步直接从 `57.1%` 噪声硬跳 `0.0` 导致 MMSE 糊底。
- **修复**：显式低尾段噪声表 `ManualSigmas("0.85, 0.54, 0.20, 0.0")`（强贴复杂背景参考图时可用 `"0.80, 0.48, 0.18, 0.0"`；极速 2 步可用 `"0.78, 0.30, 0.0"`）+ `shift_video = 6.5` + 锐度版 3D 放大器 `h3_upscaler_sharpness_2000steps_v0.1_fp32 (bf16, pad_t=3)`。

### 5.2 面积归一化 RoPE 与分频段解压缩（Split-Band RoPE）
源码 `comfy/ldm/minimax/model.py` 将整帧 $2\times 2$ 潜空间 patch 映射到面积归一化坐标框（`FRAME_SCALE = 32.0`），相邻 patch 空间坐标步长为：

$$\Delta x = \Delta y = \frac{2 \times 16 \times 32}{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}} = \frac{1024}{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}}$$

- `1344×768 (1.03MP)`：$\Delta x = 1.008\text{ rad}$（对齐最高频 `inv_freq[0]=1.0`）；
- `1568×864 (1.35MP)`：$\Delta x = 0.879\text{ rad}$ → 推荐 **`rope_hf_scale = 1.10`**；
- `1920×1088 (2.09MP)`：$\Delta x = 0.709\text{ rad}$（二维相位密度跌 `-50.6%`）→ 推荐 **`rope_hf_scale = 1.28~1.32`**。

通用标定公式（仅作用于视频行 `va:vb` 的高频空间频段 $k \in [0, 5]$，$k=6,7$ 经 `0.75/0.25` 平滑过渡回 `1.0`，不动低频、时间轴与音频）：

$$\gamma_{\text{hf}} = \left(\frac{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}}{1024}\right)^{0.80}$$

同时 `attn_temp_scale` 补偿长序列 softmax 熵稀释：裸跑 1080p 用 `1.08`；**叠加自训强晚层 LoRA 后收回至 `1.02`**，防止高光过冲与反差过重。

### 5.3 `Sol-H3-VAE (FP16)` 5-Latent 块对齐安全分块解码
24 层非因果 `minimax_h3_sol_vae_fp16` 帧长映射遵循 `5-latent (17 帧)` 局部窗结构：

$$F(T_{\text{lat}}) = 17\left\lfloor\frac{T_{\text{lat}}}{5}\right\rfloor + \max\!\big(0,\; 4(T_{\text{lat}}\bmod 5) - 3\big)$$

[`MiniMaxH3SafeAVDecode`](../sol_vae_decode.py) 在潜空间总 token $> 630{,}000$（如 15s 1080p $873{,}120$ token）时，自动按 `5` 的整数倍分段解码并在中间保护块做 17 帧线性交叉融合，峰值显存稳控 `7,535 MiB`（0 OOM，过渡帧零接缝）。

### 5.4 `1.30 MP / 1.00 MP` + 一步 `RelayVSR x2` 真 3K / 2.7K 路线
- **`1.30 MP (1568×864)` + `RelayVSR x2 (detail=0.50)` → `3136×1728`（真 3K）**：比原生 1080p 少 `35%` DiT token，配三刀提速 + 自训 LoRA + 电影质感，全程仅 **`~10–11 分钟`**，`Face120 = 331.2`、`Halo = 21.7`；
- **`1.00 MP (1376×768)` + `RelayVSR x2 (detail=0.60)` → `2752×1536`（真 2.7K）**：全程 **`~8–9 分钟`**，`@1080p LapVar = 179.2–258.8`。

---

## 6. 通用稀疏保锐 + 高低频解耦去油自蒸馏 LoRA 原理（v2.6/v2.7）

完整离线教师与 A100 训练代码见 [`train/`](../train/)（核心入口 [`train/train_h3_lora.py`](../train/train_h3_lora.py)）。

### 6.1 满血自蒸馏速度目标与 200 层 `0-adaln` 跨底座解耦契约
用官方未蒸馏 `minimax_h3_ref2va_pruned_bf16`（24 步 CFG5 全稠密）生成原生 `1920×1088` 潜空间真值 $z_{\text{native}}$，经 `Sol-H3-VAE` 480p 往返再过 `h3_upscaler_sharpness_2000steps_v0.1` 得到与 4060 二采起跑线同分布的糊底 $z_{\text{up}}$。训练步在 $\sigma$ 上构造加噪输入并回归指向 $z_{\text{native}}$ 的整流速度：

$$x_\sigma = (1 - \sigma) z_{\text{up}} + \sigma \varepsilon, \qquad v^*_{\text{teacher}} = \frac{x_\sigma - z_{\text{native}}}{\sigma}$$

- **为何严格只训 200 线性层（`0` `adaln`）？**
  - 市面加速底座（`dasiwa_v5xs` 内置 4 步 Turbo、`curveproj1025` Turbo 等）各自重塑了 `adaln_t_table` 与时间步门控，修改 `adaln_proj` 会破坏跨底座兼容性；
  - `0-adaln` 纯线性 LoRA 100% 命中 `bf16_lora=200/200` 旁路，可与任意当前/未来加速底座和风格 LoRA 无损并挂。
- **为何社区去油 LoRA（如 `Minimax H3真实电影质感V0.1`）不可剥离 `adaln` 合并，而应双槽并挂？**
  - 实测 `真实电影质感V0.1 (r=16)` 的 50 层 `adaln_proj.linear` (`[96768, 8]`) Frobenius 范数均值达 **`9.655`（单输入维能量密度是 `attn/mlp` 的 `15×` 以上）** —— 胶片暗部滚降与去塑料高光的核心正位于 `adaln_proj`；
  - 好在 `dasiwa` 的 `adaln_proj` 为 `1.5 MB` 轻量浮点层，双槽并挂时 `200` 层线性层自动并入同一次 BF16 `addmm`（`<0.4s/步`），`50` 层 `adaln` 原生运行（`<0.05s/步`），同时保留「去油胶片感」与「1080p 锐度」两个独立旋钮！

### 6.2 可微稀疏 `flex_attention` 代理：空间邻域 `±2` + 质心余弦路由
`comfy_kitchen._C.sol_attn` 无反向传播，训练端在 [`train/train_h3_lora.py`](../train/train_h3_lora.py) 使用 `torch.compile(flex_attention, dynamic=False)` + `BlockMask.from_kv_blocks`（配模块级 `_keep_mod` 与预分配 `_KEEP_GPU` 彻底杜绝动态重编译回退 eager 56GB OOM）：
1. **必留局部骨架**：`prefix sink`（text/ref/audio）+ 同帧空间邻块 `b±1, b±2`（约 `±136px` 连续带，彻底根治光栅条带横纹撕裂）+ 邻帧 `±1, ±2` 对应块；
2. **内容感知质心余弦路由**（替代纯随机块，彻底根治竖纹鬼影）：在 `no_grad` 下将进入 block 0 的隐状态 $h$ 按 128-token 块求均值并单位化 $\hat{h}_b = \text{norm}(\text{mean}(h_b))$，按相似度选 top-$k$（`topk ∈ [0.15, 0.30]`）：
   $$S_{ij} = \langle \hat{h}_i, \hat{h}_j \rangle + \eta \, U_{ij}, \qquad U_{ij} \sim \mathcal{U}(0, 1),\; \eta = 0.05$$
   该改进单步将 `step 0` 学生-教师对齐度从 `cos=0.7315 / mse=1.6806` 跃升至 **`cos=0.8782 / mse=0.7729`**。

### 6.3 纯 BF16 梯度稳定：保方向 L2 `_GradClamp` 与 `400/400` 活层门禁
1. **保方向 L2 梯度截断**：H3 `adaln` 门控极大（`blk0 ~ 9e4`, `blk49 ~ 1.7e7`），纯 BF16 反向穿越 50 层必指数溢出；逐元素 `clamp` 会扭曲梯度方向，因此在每层边界施加保方向 L2 缩放（`limit = 10.0`）：
   $$g' = \begin{cases} g \cdot \dfrac{L}{\|g\|_2 + \epsilon}, & \|g\|_2 > L \\ g, & \|g\|_2 \le L \end{cases}$$
2. **闭包晚绑定防呆 & 强制 `[grad-check]`**：在 `for i, block in enumerate(self.blocks)` 的 `checkpoint` 包装函数内**必须显式默认捕获 `_blk=block`**（漏绑会导致 50 层重算全部落到末层 `blocks.49`，仅 `4/200` 层有梯度、`gn=1e8–1e11`）；脚本内置 Step 1/2 强制断言 `400/400` 参数均获非零有限梯度（实测 `gn` 稳控在 `0.10–0.67`）。

### 6.4 高低频解耦去油蒸馏（Scheme 2A `--lf-base-anchor`，`0.0 GB` 额外峰值显存！）
- **油腻感根因**：官方 9 月 H3 教师在高 CFG 下天生带重镜面漆光与暖色 HDR 微反差，且潜空间 `85%` 方差位于低频光影；全频谱 MSE + 高噪 `σ=0.85` 会把官方低频漆光蒸馏进 LoRA，叠到已去油的 `dasiwa` 上会导致油腻“死灰复燃”（高光 `p99: 0.922 → 0.971`）。
- **⚠️ 数学陷阱**：绝不可在教师仍是官方 H3 时，把社区去油 LoRA 挂到冻结学生底模上训练！因最优解满足 $v_{\text{base}} + \Delta v_{\text{去油}} + \Delta v_{\text{LoRA}} = v^*_{\text{官方}}$，即 $\Delta v_{\text{LoRA}} = (v^*_{\text{官方}} - v_{\text{base}}) - \Delta v_{\text{去油}}$，LoRA 会反向学出 **`双倍增油`**！
- **Scheme 2A 数学解（零重跑教师、零额外峰值显存）**：
  每步先以 `with _disable_lora(), torch.no_grad():` 跑一次冻结基座参考 $v_{\text{base}}$（自动跳过 `checkpoint` 与 `_GradClamp`，峰值仅 `~43 GB < 56~65 GB`，**额外峰值显存严格 = `0.0 GB`**；并复用同一 sparse `BlockMask` 零抖动）。对视频潜空间 $[B, C, T, H, W]$ 施加逐帧 `5×5` `replicate` 边界低通滤波 $L_5(\cdot)$（对应约 `80×80 px` 大尺度光影包络，零边缘暗角），构造**纯净去油速度目标**：
  $$v_{\text{tgt}}^{\text{clean}} = \underbrace{L_5(v_{\text{base}})}_{\text{100\% 锁死基座低频光影}} + \underbrace{\big(v^*_{\text{teacher}} - L_5(v^*_{\text{teacher}})\big)}_{\text{100\% 吸收教师 1080p 高频锐度}}$$
  在唯一全局最优点 $v_{\text{pred}}^* = v_{\text{tgt}}^{\text{clean}}$ 处，LoRA 学到的精确增量严格满足：
  $$\Delta v_{\text{LoRA}}^* = v_{\text{pred}}^* - v_{\text{base}} = \big(v^*_{\text{teacher}} - L_5(v^*_{\text{teacher}})\big) - \big(v_{\text{base}} - L_5(v_{\text{base}})\big) = \text{HP}_5\!\left(v^*_{\text{teacher}} - v_{\text{base}}\right)$$
  同步**彻底砍掉高噪 `σ=0.85`**（改为 $\sigma \in [0.14, 0.62]$ + 35% 锚点 $\{0.54, 0.20\}$）并启用**按桶内样本数加权抽样**（$P(g) = |g| / \sum_j |g_j|$，把小桶占比从 `33.3%` 压至 `<4.5%`）。
- **实测验证**：
  - 合成验证：低频油光偏置误差从 `0.1899` 压至 **`0.000464`（清除 `99.76%`）**，高频棋盘纹理几乎 100% 保留；
  - A100-80GB Stage-2 (`lora_run_v2_s2`, 72 样本 `22f+39f`, `mem_gb=65.2 GB`, `46.6s/step`) 实盘监控：从 `step_000200` 热启，**仅 25 步低频油光误差 `loss_lf` 即从 `0.745 / 0.872` 暴降至 `0.079`（洗油 `-91%`！）**。

---

## 7. RTX 4060 Laptop 8GB / 32GB 全线实测总账（真实 `dasiwa` INT8，全部 0 OOM）

### 7.1 历史 A/B 画质实测基准（同缩 `1920×1088` 评测）

| 组别（除注明外均为 10s / 243f） | 原生画布 | 二采 3 步 (`s/it`) | 总耗时（二采+解码+VSR） | 显存峰值 | `@1080p LapVar` | `HF>0.25%` | `Halo` | `Jitter` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **旧版原生 1080p (All-Sol)** | `1920×1088` | — | `~13m` | — | `57.0` | `0.096%` | `20.3` | `3.85` |
| **P1-C `864p` 甜点默认** | `1568×864` | `121.6 / 113.9 / 231.9` | `~9m 20s` | `7471 MiB` | `104.3` (原生 `216.6`) | `0.207%` | `22.6` | `3.28` |
| **Exp 1 新版原生 1080p+** (`RoPE 1.32 + LMS 0.50 + Sol-VAE`) | `1984×1120` | `244.7 / 238.5 / 538.1` | `20m 22s` | `7845 MiB` | **`122.1` (`2.14×`)** | `0.265%` | **`20.2`** | `3.14` |
| **`864p` + `RelayVSR x2 (d=0.50)` 真 3K** | **`3136×1728` (3K)** | `~115 / 110 / 230` (旧) | `~10m 50s`（v2.7 提速后 **`~10m`**） | `~7.5 GiB` | **`146.4` (`2.57×`)** | `0.249%` | **`21.7`** | `3.56` |
| **Exp 2 `768p` + `LMS 0.50` + `RelayVSR x2 (d=0.60)` 真 2.7K** | **`2752×1536` (2.7K)** | `89.4 / 82.0 / 153.2` | **`9m 10s`**（NoBg `8m10s`） | `7645 MiB` | **`179.2` (`3.14×`)** | **`0.299%`** | `23.3` | `3.30` |
| **Exp 15s 原生 1080p 单窗** (`362f, hc=28, Tight-Sol 0.90`, 2 段 `Sol-VAE`) | **`1920×1088` (15s)** | `384.7 / 378.9 / 546.8` | `25m 38s` | `7909 / 7535 MiB` | **`111.6` (`2.20×`)** | `0.242%` | `20.6` | **`2.77`** |

### 7.2 v2.7 全工作流端到端选档表（含 Qwen3VL + 一采 + 二采 + 解码 + VSR 全程耗时）

| 推荐工作流 | 二采画布 (`second_megapixels`) | 最终分辨率 | 步数 | 4060 8GB 全程耗时 | 推荐场景 |
|---|---|---|---|---:|---|
| **[`workflow_singlewin_selflora_13mp_vsr3k.json`](../examples/workflow_singlewin_selflora_13mp_vsr3k.json)** | **`1.30` (`1568×864`, `96k`)** | **`3136×1728` (真 3K)** | 3 步 / 2 步 | **`~10–11m`** / **`~8.5m`** | **★ 日常主力甜点王**（高锐+去油+3K） |
| **[`workflow_singlewin_selflora_1080p.json`](../examples/workflow_singlewin_selflora_1080p.json)** | **`1.98` (`1920×1088`, `147k`)** | **`1920×1088` (纯原生)** | 3 步 / 2 步 | **`~14.5–15.3m`** / **`~12m`**（5s 仅 **`9m`**） | 零超分纯原生 1080p 极限质感 |
| **[`workflow_singlewin_sharp_vsr2k.json`](../examples/workflow_singlewin_sharp_vsr2k.json)** | **`1.00` (`1376×768`, `78k`)** | **`2752×1536` (真 2.7K)** | 3 步 | **`~8.5–9.5m`** | 经典最速 2.7K 高性价比路线 |
