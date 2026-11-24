# comfyui-h3-seamkit 核心原理与数学公式详解（v2.3）

> **AI / 开发者 30 秒快速接手摘要**：
> - **项目目标**：在 **RTX 4060 Laptop 8GB / 32GB RAM**（同时原生兼容 **A100 / H100 / 4090 / 5090**）上，实现 MiniMax H3 **10s–15s 单窗零拼缝直出（最高原生 `1920×1088` 与真 `2.7K`）**，并在保证画质与音画同步的前提下最大化推理提速。
> - **当前已攻克**：① 8GB 显存单窗破壁（`15s @ 1920×1088` 负载 `756.2 = 3.96×` 旧 OOM 红线，0 OOM）；② 16-bit BF16 低秩 LoRA 旁路与算子融合（第一拳，防 INT8 重定量化碾碎微弱 LoRA）；③ 原生 1080p 清晰度翻倍（`Split-Band RoPE 1.32` + `attn_temp 1.08` + 低尾段噪声表 + `Sol-H3-VAE`，`@1080p LapVar` 达旧版 `2.14×–2.20×`）；④ `Sol-H3-VAE` 5-latent 块对齐安全分块解码（`MiniMaxH3SafeAVDecode`，0 接缝 0 OOM）。
> - **当前唯一剩余大瓶颈（即 A100 训练核心靶点）**：**原生 1080p 推理速度**！
>   - 10s 原生 1080p（`1984×1120`）二采耗时 **`17m 07s`**，其中 **末步全稠密（Step 3 Dense）独占 `538.1s`（`52.7%`）**；
>   - 15s 原生 1080p（`1920×1088`）开启 Tight-Sol 后二采仍需 **`21m 44s`**（Step 1/2/3 = `384.7s / 378.9s / 546.8s`）；
>   - 若通过 A100 蒸馏训出 **「Sol-Attn 稀疏恢复 + 1080p 2-Step 直线化」200 层线性 LoRA**（挂载到本仓库 16-bit BF16 旁路仅 `+1.5s` 开销），可把 10s 原生 1080p 从 `17m` 压至 **`~6–9m`（提速 `1.9×–2.8×`）**，把 15s 从 `22m` 压至 **`~9–14m`**！
> - **零侵入原则**：本仓库不改动 `ComfyUI` 核心、`ComfyUI-KJNodes`、`comfyui-minimax-h3-audio-T8` 任何一行代码；核心实现全在 [`chunked_sage.py`](../chunked_sage.py) 与 [`sol_vae_decode.py`](../sol_vae_decode.py)。

---

## 1. 为什么 8GB 显存跑 10s/15s 单窗会 OOM？分块 QKV 与 Streaming 的显存公式

MiniMax H3 主干包含 50 层 DiT Block（隐藏维 $H = 5376$，注意力头数 $N_h = 56$，每头维度 $D = 128$，内维 $I = N_h \cdot D = 7168$，SwiGLU 中间维 $F = 14336$）。
视频潜空间下采样倍率为空间 $16\times$、时间 $17k+5 \leftrightarrow 5k+2$，DiT 空间 patch 大小为 $2\times 2$（即每 DiT 视频 token 对应 $32\times 32$ 像素），因此一帧视频的 token 数与总序列长度 $S$ 为：

$$N_{\text{tok/frame}} = \frac{W_{\text{px}}}{32} \cdot \frac{H_{\text{px}}}{32}, \qquad S = N_{\text{text}} + N_{\text{ref}} + N_{\text{audio}} + T_{\text{lat}} \cdot N_{\text{tok/frame}}$$

典型 10s（243 帧，$T_{\text{lat}} = 72$）与 15s（362 帧，$T_{\text{lat}} = 107$）的序列长度与 8GB 负载值（$\text{Load} = \text{Frames} \times \text{MP}$）：
- `10s @ 1280×736 (0.90MP)`：$S \approx 67{,}000$，Load `219`
- `10s @ 1376×768 (1.06MP)`：$S \approx 78{,}000$，Load `257`（H3 原生 RoPE 甜点）
- `10s @ 1568×864 (1.35MP)`：$S \approx 96{,}000$，Load `329`
- `10s @ 1920×1088 (2.09MP)`：$S \approx 154{,}000$（`1984×1120` 时 `163k`），Load `508`
- `15s @ 1920×1088 (2.09MP)`：**$S \approx 222{,}000$**（视频 `218,280` tok），Load **`756.2`（旧版 8GB OOM 锚 `191` 的 `3.96×`！）**

### 1.1 原版 KJ 低显存补丁的三大显存尖峰（BF16 = 2 字节，FP32 = 4 字节）
1. **整序列 QKV 缓冲区**：`qkv_proj(h)` 一次性物化 `[S, 3I] = [S, 21504]` BF16：
   $$M_{\text{QKV}} = 2 \cdot S \cdot 21504 \approx \mathbf{3.85\text{ GiB}}\;(96\text{k}),\quad \mathbf{6.17\text{ GiB}}\;(154\text{k}),\quad \mathbf{8.90\text{ GiB}}\;(222\text{k})$$
2. **残差 + 归一化 + 输出缓冲并发驻留**：`x [S, 5376]` + `h [S, 5376]` + `out [S, 7168]`（在 $S=222\text{k}$ 单是这三块就超过 `7.5 GiB`）。
3. **首尾 FP32 I/O 层**：`video_patch_proj` 与 `FinalLayer._modulate` 一次性物化 `[N_v, 5376]` FP32（`1.96–4.37 GiB`）。

### 1.2 `chunked_sage.py` 的三级显存削峰与 ConvRot-256 对齐条件
1. **一次性 Token 分块 ConvRot 预量化**：按 `token_chunk = 8192` 分块执行 `RMSNorm + AdaLN + Hadamard 旋转 + 行级 INT8 量化`，只落盘 `(x_qdata [S, 5376] int8, x_scale [S, 1] fp32)`（`96k` 仅 `0.49 GiB`，`222k` 仅 `1.14 GiB`，永不物化整条 BF16 `h`）。
2. **头分组切片 QKV + 延迟投影 V**：把 56 个头分成 $G$ 组（每组 $g$ 个头），严格满足 Hadamard 块对齐约束：
   $$(g \cdot 128) \bmod 256 = 0 \iff g \text{ 为偶数（例如 } G=7 \Rightarrow g=8,\; G=14 \Rightarrow g=4,\; G=28 \Rightarrow g=2\text{）}$$
   每组零拷贝行切片投影 `Q_g, K_g` → 原地 RMS-RoPE → 量化为 INT8 并立即释放 BF16 `Q_g, K_g`，再投影并量化 `V_g` → 调起 SageAttention2++ / Sol-Attn。
3. **双 `out_proj` 模式**：
   - **`exact` 模式**（显存 $\ge 16\text{ GiB}$ 如 A100/4090 或 $S \le 64\text{k}$ 自动启用）：保留 `[S, 7168]` 输出缓冲，再按 token 分块执行 `out_proj` —— 与原版 KJ **100.00% 逐位一致**（`max_abs = 0.0`）。
   - **`streaming` 模式**（8GB 卡长序列自动启用）：利用 Hadamard 旋转的分块正交性，每组头输出 `o_g [S, g·128]` 直接乘以 `out_proj` 对应列切片 `W[:, c0:c1]`（宽度 `g·128` 恰为 256 整数倍）并原地累加进残差 `x` —— **彻底消灭 `[S, 7168]` 输出缓冲**，10s 864p 峰值激活从 `6.70 GiB` 降至 **`1.98 GiB`**（`-70%`），15s 1080p (`hc=28, g=2`) 硬张量总量仅 `4.70 GiB`（实测总显存峰值 `7,909 / 8,188 MiB`，0 OOM！），单层余弦精度仍达 `cos = 0.999965`。

---

## 2. RTX 4060 8GB 实测耗时拟合模型与「1080p 为什么慢」的定量解剖

Transformer 单步前向计算严格分为 **PCIe 权重搬运常数项**、**$O(S)$ 线性投影/MLP 项** 与 **$O(S^2)$ 注意力二次项**。由 RTX 4060 Laptop 8GB 在真实 `dasiwa` 50 层 INT8 模型上的常驻显存基准（$S = 48\text{k} / 96\text{k} / 144\text{k}$ 耗时 `64.5s / 199.3s / 414.5s`）与 ComfyUI `aimdo` 实跑联立反解，得到稠密单步经验公式：

$$T_{\text{step}}^{\text{dense}}(S) \approx 24\text{s}_{\text{PCIe}} + 29.4\text{s} \cdot \left(\frac{S}{48\text{k}}\right)_{\text{Linear/MLP}} + 35.2\text{s} \cdot \left(\frac{S}{48\text{k}}\right)^2_{\text{Dense SageAttn}}$$

而开启 **`Sol-Attn (tau=1.3, aug=64)`** 后，稀疏注意力项降低约 `3.2×–4.0×`：

$$T_{\text{step}}^{\text{sol}}(S) \approx 24\text{s}_{\text{PCIe}} + 31\text{s} \cdot \left(\frac{S}{48\text{k}}\right)_{\text{Linear/MLP+Morton}} + (9\text{–}12)\text{s} \cdot \left(\frac{S}{48\text{k}}\right)^2_{\text{Sol-Attn}}$$

### 实测 1080p 耗时占比诊断（为什么 1080p 慢 & A100 蒸馏能省多少？）

| 场景（RTX 4060 8GB 实测） | $S$ | Step 1 (Sol) | Step 2 (Sol) | Step 3 | 二采总计 | 瓶颈归因 |
|---|---:|---:|---:|---:|---:|---|
| **10s `1376×768`** (`Exp 2a`) | `78k` | `89.4s` | `82.0s` | `153.2s` (Dense) | **`5m 29s`** | 极快（RoPE 甜点） |
| **10s `1568×864`** (`P1-C`) | `96k` | `121.6s` | `113.9s` | `231.9s` (Dense) | **`7m 52s`** | Step 3 Dense 占 `49%` |
| **10s `1984×1120`** (`Exp 1`) | `163k` | `244.7s` | `238.5s` | **`538.1s` (Dense!)** | **`17m 07s`** | **Step 3 Dense 独占 `52.7%`!** |
| **15s `1920×1088`** (`Exp 15s`) | `222k` | `384.7s` | `378.9s` | **`546.8s` (Tight-Sol)** | **`21m 44s`** | Tight-Sol 比 Dense(`~980s`) 省 `7.2m`，但仍占 `42%` |

- **提速杠杆计算（指导 A100 LoRA 训练目标）**：
  1. **杠杆 ①（消灭 Step 3 Dense + 关闭 `aug=64→0`、推高 `tau=1.3→1.5`）**：经 **Sol-Attn → Dense 稀疏恢复蒸馏 LoRA** 补偿后，10s 原生 1080p 3 步从 `245 + 239 + 538 = 1022s` 压至 **`~185 × 3 = 555s`（`9m 15s`，提速 `1.84×`）**；15s 从 `1310s` 压至 **`~840s`（`14m 00s`，提速 `1.56×`）**。
  2. **杠杆 ②（叠加 `3 步 → 2 步` 直线化轨迹蒸馏：`0.78 → 0.35 → 0.0`）**：10s 原生 1080p 再砍整整 1 步 → **`~370s`（`6m 10s`，提速 `2.76×`！）**；15s → **`~560s`（`9m 20s`，提速 `2.34×`！）**。
  3. **在 A100 40GB/80GB 推理端**：HBM2e `1.55–2.0 TB/s` 且模型全常驻（$T_{\text{PCIe}}=0$），单步比 4060 快 `6–8×`（10s 1080p 稠密单步仅 `~25–35s`），非常合适用来做离线教师轨迹生成与蒸馏训练。

---

## 3. Sol-Attn 稀疏加速、`2d_frame` Morton 对齐与三档低噪调度

在 [`chunked_sage.py`](../chunked_sage.py) 中，`comfy_kitchen.sol_attn` 直接嵌入头分组循环内，包含四项工程设计：
1. **零额外显存的 `2d_frame` Morton 原地重排**：将视频 token 按帧内二维 Z-order 曲线重排，使空间邻近 patch 落在同一个 64-token Sol-Attn 块内，提高稀疏路由命中率；重排按列分块（`col_chunk = 1344`）原地完成，scratch 峰值降 `4×`，推理结束逆置换还原。
2. **条件与音频 Sink 严格保护**：视频起始偏移 `va` 按 64-token 向上对齐，通过 `sink_blocks` 与 `sink_q` 强制文本、参考图与音频查询行保持 100% 稠密注意，杜绝音画漂移与提示词失效。
3. **`sol_min_sigma` 末步全稠密回收（默认 `sol_low_sigma_tau = 0.0`，推荐 10s 及以下）**：
   - 全程 Sol-Attn 虽提速 `1.88×`，但末步低噪阶段稀疏路由会损失约 `12%` 高频微纹理；
   - 设置 `sol_min_sigma = 0.25` 后，前两步走 Sol-Attn 极速定型，末步 $\sigma < 0.25$ 自动切回 **100% 稠密 KJ SageAttention2++** 精修，把高频锐度完整找回。
4. **`sol_low_sigma_tau` Tight-Sol 末步精修（15s 1080p 专属，推荐 `0.90`）**：
   - 当 $S \ge 200\text{k}$（如 15s 1080p）时，末步全稠密耗时暴涨至 `~980s`；
   - 设置 `sol_low_sigma_tau = 0.90` 后，末步自动切为 **首尾各 3 层全稠密（`0..2, 47..49`）+ 中间 44 层超紧稀疏（`tau=0.90, token_aug>=128`）**；实测 15s 1080p 末步压至 `546.8s`（狂省 `7.2 分钟`），同时 `@1080p LapVar` 仍高达 `111.6`（旧版 15s `864p` 的 `2.20×`）。

> **⚠️ 给 A100 训练代码的关键提示**：`comfy_kitchen._C.sol_attn` 经由 DLPack 调用底层 CUDA 内核，**不支持 PyTorch Autograd 反向传播**！因此 A100 训练稀疏学生模型时，必须用 **无梯度 64-token 质心路由 + `sink_blocks` + `2d_frame` Morton 算出等价块掩码**，再送入 PyTorch 原生可微 **`torch.nn.attention.flex_attention` (`create_block_mask`)** 完成前向与反向传播！

---

## 4. 第一拳核心：Packed-Rank BF16 低秩 LoRA 旁路与算子融合

### 4.1 为什么旧版 ComfyUI INT8 重定量化会“摧毁”微弱锐化/蒸馏 LoRA？
ComfyUI 在 `ModelPatcherDynamic` 低显存模式下挂载 LoRA 到 INT8 权重时，每步对全部 200 个 DiT 线性层重复执行：

$$\widetilde{W}_{\text{int8}} = \mathcal{Q}_{\text{INT8,ConvRot}}\!\left(\mathcal{D}(W_{\text{int8}}) + \sum_i s_i B_i A_i\right)$$

这带来双重灾难：
1. **速度与显存灾难**：每步 200 次全矩阵反量化 + `B@A` 合并 + Hadamard 重定量化，白烧 `10–16s/步` 与 `~350 MiB` `AIMDO_CAST_BUFFER`；
2. **精度灾难（实测证伪）**：当 LoRA 幅度较弱时（如 `minimax_h3_lms_v1.0_r64_r @ 0.18`），增量 $\|s B A\|$ 远小于底模通道 INT8 步长 $\Delta_q = \max|W_{\text{row}}|/127$，重定量化后被舍入噪声彻底淹没：

| 实现方式（`dasiwa blocks.0.attn.qkv_proj` + `LMS @ 0.18` vs 精确 FP32） | 余弦相似度 $\cos$ | 相对 $L_2$ 误差 |
|---|---:|---:|
| 旧版 ComfyUI INT8 重定量化 (`want_requant=True`) | **`0.016058`** | **`422.25%`**（几乎纯噪声） |
| **本仓库 BF16 低秩旁路 (`bf16_lora=200/200`)** | **`0.999996`** | **`0.27%`**（无损保真） |

### 4.2 代数等价的 Packed-Rank 旁路与四项 CUDA/Triton 融合
我们自动拦截 200 个 DiT 线性层（`blocks.{0..49}.attn.{qkv_proj,out_proj}` 与 `mlp.{fc1,fc2}`）的标准 2D LoRA，冻结的 INT8 主干保持零修改直走最快 CUTLASS 内核，所有叠挂 LoRA 拼接为单个紧凑低秩对：

$$A_{\text{cat}} = \begin{bmatrix} s_1 A_1 \\ s_2 A_2 \end{bmatrix} \in \mathbb{R}^{R \times K}, \qquad B_{\text{cat}} = \begin{bmatrix} B_1 & B_2 \end{bmatrix} \in \mathbb{R}^{N \times R}$$

1. **原地 `torch.addmm` 累加**：$Y \leftarrow Y + (X A_{\text{cat}}^\top) B_{\text{cat}}^\top$ 直接在 cuBLAS epilogue 内写回，消除 `448 MiB` 临时缓冲，单层 `fc1` 旁路提速 `4.5×`（实测叠挂 `1.18 GB` rank-64 `LMS` 全程仅多花 **`+1.9s`**！）；
2. **门控折叠残差累加**（`out_proj` / `mlp.fc2`）：将 `mod_segment` 门控 $g_r$ 预乘入极小的 $B_{\text{cat}}$（仅 `0.8 MB`）：
   $$X \leftarrow X + g_r \odot (M B_{\text{cat}}^\top) \equiv \text{addmm}\!\left(X,\; M,\; (B_{\text{cat}} \odot g_r[:, \text{None}])^\top\right)$$
3. **单遍融合 Triton SwiGLU**：一次合并访存完成 `silu(gate.float()) * up.float()`；
4. **原地 CUTLASS 残差 Epilogue + 权重折叠 `rms_adaln`**：`_C.cutlass_int8_dequant_residual` 实现 **`0.0` 逐位一致**原地残差累加；`streaming` 模式预折叠 $s_{\text{eff}} = w \odot (1 + s) - 1$ 走 `_ck.rms_adaln`（提速 `3.0×`），`exact` 模式保持与原版 KJ `torch.equal` 100.00% 逐位一致。

> **⚠️ 给 A100 训练的接口铁律**：只要新训的 LoRA **仅包含上述 200 个线性层（严格跳过 `adaln_proj`）** 且采用标准 `lora_down.weight / lora_up.weight / alpha` 键名，即可 100% 命中 `bf16_lora=200/200` 旁路，回到 4060 零转换、无精度损失、耗时仅 `+1.5s`！

---

## 5. 二采画质发虚与长片解码的四大物理破壁公式

### 5.1 Flow-Matching 偏移公式与「`0.57` 断崖」
MiniMax H3 使用带偏移 $s$（默认视频 `shift_video = 12.0`）的 Flow Matching 噪声表：

$$\sigma(t) = \frac{s \cdot t}{1 + (s - 1)\, t}, \qquad t \in [0, 1]$$

旧版 `BasicScheduler(simple, steps=3, denoise=0.30, shift=12)` 只采样 $t \in \{0.30, 0.20, 0.10, 0\}$，导致 $\sigma = [0.8372,\; 0.7500,\; \mathbf{0.5714},\; 0.0]$ —— **末步直接从 `57.1%` 噪声跳到 `0.0`**，跳过整个 `[0.05, 0.55]` 纹理雕刻区而输出 MMSE 糊底。
- **修复**：显式低尾段噪声表 `ManualSigmas("0.80, 0.48, 0.18, 0.0")`（无背景参考场景推高至 `"0.85, 0.54, 0.20, 0.0"` 以抹除 85% 一采平墙模糊）+ `shift_video = 6.5` + 锐度版 3D 放大器 `h3_upscaler_sharpness_2000steps_v0.1_fp32 (bf16, pad_t=3)`。

### 5.2 面积归一化 RoPE 与 1080p 分频段解压缩（Split-Band RoPE）
源码 `comfy/ldm/minimax/model.py:77-99` 将整帧所有 $2\times 2$ 潜空间 patch 映射到面积归一化坐标框（`FRAME_SCALE = 32.0`），相邻 patch 空间坐标步长为：

$$\Delta x = \Delta y = \frac{2 \times 16 \times 32}{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}} = \frac{1024}{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}}$$

- H3 原生甜点 `1344×768 (1.03MP)` 下 $\Delta x = \mathbf{1.0079\text{ rad}}$（对齐最高频 `inv_freq[0] = 1.0`）；
- 拉高到原生 `1920×1088 (2.09MP)` 后 $\Delta x$ 压缩到 **`0.7085 rad`**（二维相位密度暴跌 **`-50.6%`**，等效局部感受野虚胖 `1.42×`），叠加 `154k–222k` 长序列 softmax 熵稀释，导致裸跑 1080p 发虚。
- **零耗时修复（`chunked_sage.py`，`0.00s / 0 MiB` 开销）**：
  1. **`rope_hf_scale = 1.32`**：仅放大视频行 `va:vb` 的空间高频频段 $k \in [0, 5]$（$k=6,7$ 经 `0.75/0.25` 平滑过渡回 `1.0`），绝不碰低频 $k \in [8, 15]$、时间轴、参考图与双声道音频：
     $$\gamma_{\text{hf}} = \left(\frac{\sqrt{W_{\text{px}} \cdot H_{\text{px}}}}{1024}\right)^{0.80} \approx \mathbf{1.32} \quad (@\;1920\times 1088)$$
  2. **`attn_temp_scale = 1.08`**：将 `sm_scale = D^{-1/2}` 乘以 $\alpha \approx \ln(S)/\ln(S_0) \approx 1.08$ 补偿长序列熵稀释。

### 5.3 `Sol-H3-VAE (FP16)` 5-Latent 块对齐安全分块解码与 32GB 内存解耦
- **双重 INT8 替代**：旧版 `minimax_h3_video_vae_int8_convrot` 对 36 层解码器全程跑 `int8_attention`（慢且软）；换用 24 层非因果 **`minimax_h3_sol_vae_fp16.safetensors`** 后，10s `768p` 解码仅 **`47.95s`**、10s `1080p+` 仅 **`101.8s`**，且画质更透彻。
- **15s 1080p 两大红线与破解（[`sol_vae_decode.py`](../sol_vae_decode.py)）**：
  1. **显存红线**：`Sol-H3-VAE` 整段解码在 `15s 1080p`（$107\times 68\times 120 = 873{,}120$ token）预估需 `8.46 GiB > 8GB`，且原生 `decode_tiled` 忽略分块直接硬崩。利用其原生 `5-latent (17 帧)` 局部窗结构：
     $$F(T_{\text{lat}}) = 17\left\lfloor\frac{T_{\text{lat}}}{5}\right\rfloor + \max\!\big(0,\; 4(T_{\text{lat}}\bmod 5) - 3\big)$$
     [`MiniMaxH3SafeAVDecode`](../sol_vae_decode.py) 在总 token $> 630{,}000$ 时自动按 `5` 的整数倍切片（15s 1080p 切成 `[0:60] -> f[0..203]` 与 `[45:107] -> f[153..361]`，3 块重叠 = 51 帧，仅在中间保护块 `[170..186]` 做 17 帧线性交叉融合），将峰值显存压至 **`7,535 MiB`**（0 OOM，过渡帧零接缝！）。
  2. **32GB 主机内存红线**：`362f @ 1088×1920` FP32 图像缓冲占 `9.07 GB`，叠加 `dasiwa` 常驻 `14 GiB` `HostBuffer` 会顶破 `31.63 GiB`。通过 `MiniMaxH3AVLatentSave`（Prompt 1 采样落盘 `84 MB`）→ `POST /free {"unload_models": true, "free_memory": true}`（内存回落至 `11.2 GB`）→ Prompt 2 `MiniMaxH3SafeAVDecode` 解码，峰值内存稳控在 **`25.12 GB`**！

### 5.4 甜点 H3 + 一步 `RelayVSR x2` 真 2.7K 路线
让 H3 停在 RoPE 最锐甜点 `1376×768`（`1.06MP`, $\Delta x \approx 1.0$），叠 `LMS 0.50` + `Sol-H3-VAE` 后经 1 步扩散超分 `RelayVSR x2 (detail=0.60)` 直出 **`2752×1536` 真 2.7K**，全程仅 **`9m 10s`**（比原生 1080p 快 `2.23×`），`@1080p LapVar = 179.2`（原生旧 1080p 的 **`3.14×`**）！

---

## 6. RTX 4060 Laptop 8GB / 32GB 全线实测总账（真实 `dasiwa` INT8，全部 0 OOM）

| 组别（除注明外均为 10s / 243f，同缩 `1920×1088` 评测） | 原生画布 | 二采 3 步 (`s/it`) | 总耗时（二采+解码+VSR） | 显存峰值 | `@1080p LapVar` | `HF>0.25%` | `Halo` | `Jitter` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **旧版原生 1080p (All-Sol)** | `1920×1088` | — | `~13m` | — | `57.0` | `0.096%` | `20.3` | `3.85` |
| **P1-C `864p` 甜点默认** | `1568×864` | `121.6 / 113.9 / 231.9` | `~9m 20s` | `7471 MiB` | `104.3` (原生 `216.6`) | `0.207%` | `22.6` | `3.28` |
| **Exp 1 新版原生 1080p+** (`RoPE 1.32 + LMS 0.50 + Sol-VAE`) | `1984×1120` | `244.7 / 238.5 / 538.1` | **`20m 22s`** | `7845 MiB` | **`122.1` (`2.14×`)** | `0.265%` | **`20.2`** | `3.14` |
| **Exp 2 性价比之王：`768p` + `LMS 0.50` + `Sol-VAE` + `RelayVSR x2`** | **`2752×1536` (2.7K)** | `89.4 / 82.0 / 153.2` | **`9m 10s` (`2.23×` 快!)** | `7645 MiB` | **`179.2` (`3.14×`)** | **`0.299%`** | `23.3` | `3.30` |
| **Exp 15s 原生 1080p 单窗** (`362f`, `hc=28`, `Tight-Sol 0.90`, 2 段 `Sol-VAE`) | **`1920×1088` (15s)** | `384.7 / 378.9 / 546.8` | **`25m 38s`** (`22m51s+2m47s`) | `7909 / 7535 MiB` | **`111.6` (`2.20×` 旧15s)** | `0.242%` | `20.6` | **`2.77`** |
