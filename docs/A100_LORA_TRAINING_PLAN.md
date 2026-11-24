# Colab G4 / A100 双 Agent 协同任务书：MiniMax H3 原生 1080p 逼近满血 + 提速蒸馏 LoRA 训练

> **本文档面向接手本项目的两个自主 AI Agent（DeepSeek 部署工程师 + Grok Bot 夜间值守员）**。
> 请在动手前先通读 [`docs/PRINCIPLES_AND_FORMULAS.md`](PRINCIPLES_AND_FORMULAS.md)，以彻底理解本仓库在 RTX 4060 8GB 上的显存、耗时、Sol-Attn 与 16-bit BF16 LoRA 旁路机制。
> **硬件硬门槛**：本项目**仅在 Google Colab G4（NVIDIA RTX PRO 6000 96GB Blackwell）或 A100（40GB / 80GB）上运行**；若分到 T4 / L4 / V100，开机第一步立即退出，绝不浪费时间降级硬跑！

---

## 0. 项目核心目的与跨模型通用性评估（战略必读）

### 0.1 我们到底要解决什么问题？
经过前序工程攻坚，[`comfyui-h3-seamkit`](../README.md) 已经在消费级 **RTX 4060 Laptop 8GB / 32GB RAM** 上打通了 **10s 与 15s 原生 `1920×1088` 单窗直出（0 OOM、0 接缝、清晰度较旧版翻倍）**。
但目前**唯一剩下的核心痛点是「原生 1080p 推理速度慢」**：
- `10s @ 1984×1120` 二采 3 步耗时 `244.7s (Sol) + 238.5s (Sol) + 538.1s (Dense) = 17m 07s` —— **末步被迫退回 100% 稠密注意力独占 `52.7%` 耗时**！
- `15s @ 1920×1088` 即使切到 Tight-Sol (`low_tau=0.90`) 仍需 `384.7s + 378.9s + 546.8s = 21m 44s`。

### 0.2 本次 G4 / A100 跨号训练的**第一性目标（一石四鸟：画质逼近满血原生 + 原生 1080p 提速 `1.9×–2.8×`）**
**使用 7h 教师号 + 24h 主训练号（Colab G4 96GB 优先 / A100 40GB-80GB 保底），零外部视频数据集，以「官方满血 BF16 24 步 CFG 全稠密原生 1080p」为教师，向「`z_up` 糊底输入 + Sol-Attn 稀疏 2–3 步学生」做自蒸馏，同时攻克画质与速度双瓶颈！**

| 交付产物 | 核心机理（满血教师 → 稀疏少步学生） | 回到 4060 原生 1080p 效果 | 所需数据 | 推荐优先级 |
|---|---|---|---|---|
| **主产物 A（必拿 · 通用画质+提速基石）**：`h3_1080p_native_sol_restore_r32.safetensors` | **官方满血 BF16 (24 步 CFG 全稠密) 教师 → 3 步全稀疏学生（`keep_ratio 15%–30%`）+ 成对 `z_up → z_native_bf16` 去糊 LoRA**（取消 Step 3 Dense 与 `aug=64`，顺带把 24 步原生微观纹理注入 3 步！） | 10s 1080p：`17m07s → ~9m15s` (**`-46%`**)<br>15s 1080p：`21m44s → ~14m00s` (**`-36%`**)<br>画质从 Turbo 75% → 逼近原生满血 **~88%–92%** | **0 外部视频**（7h 教师号自生成 150–200 组满血 1080p 教师包） | **P0 必完成** |
| **主产物 B（进阶 · 终极提速）**：`h3_1080p_native_sol2step_r32.safetensors` | 复用同一批满血教师包，做 **「满血原生 `z_native_bf16` → 2 步全稀疏直线化 (`0.78 → 0.35 → 0.0`)」** 速度匹配（无多步 BPTT，单步 MSE 稳收敛！） | 10s 1080p：`17m07s → ~6m10s` (**`2.8×` 提速!**)<br>15s 1080p：`21m44s → ~9m20s` (**`2.3×` 提速!**) | **复用主产物 A 同一批离线教师包**（0 额外成本） | **P1 强烈推荐** |
| **可选加分项 C（4–6h 强化学习对齐）**：`h3_1080p_native_sol2step_dpo.safetensors` | 在 Phase B 上做 **Iterative Flow-DPO**（5 项防作弊组合奖励：`LapVar + HF - Halo - Jitter + SSIM`） | 消除 $L_2$ 轻微平滑，发丝/远景再提一截且零白边 | 仅需冻结 `Sol-H3-VAE` 抽帧打分 | **P2 可选** |

### 0.3 换模型后的通用性评估（实话实说，避免盲投）

| 未来换模型场景 | 主产物 A（稀疏恢复 + 1080p 锐度 LoRA） | 主产物 B（2 步轨迹直线化 LoRA） | 可选产物 C（`LearnedResizer3D v2`） |
|---|---|---|---|
| **① 同底模换量化格式**（如 A100 用 `dasiwa BF16` 训 → 4060 挂 `dasiwa INT8 / FP8 / NVFP4`） | **95%–100% 通用**（本仓库 16-bit BF16 旁路与底模量化完全解耦） | **95%–100% 通用** | **100% 通用** |
| **② 换同架构新版 H3 社区微调 / Turbo 底模**（如未来 `dasiwa v6`、官方新权重） | **75%–90% 可迁移**（**前提：仅训 200 线性层并跳过 `adaln_proj`**！正如原版 H3 训出的 `LMS` 挂到 `dasiwa v5xs` 依然极强有效，通常只需调 `strength=0.6~1.0`） | **同 Turbo 家族 70%–85% 通用**；若换成未蒸馏 50 步原版 H3 则 2 步失效（但稀疏补偿仍部分有效） | **100% 通用**（所有 H3 共用同一 24ch VAE 潜空间） |
| **③ 彻底跨架构换代**（如改用 Wan / LTX / Seedance 等新底座） | 权重不可直接复用；**但整套自蒸馏脚本、可微 `flex_attention` 稀疏代理与双 Agent 运维架构 90% 可直接迁移** | 同左 | 权重不可复用 |

> **战略结论**：把 **主产物 A（3 步全稀疏恢复，高通用性）作为前半夜必拿保底里程碑（如 Step 1500/2000 单独锁定最佳权重）**，再在同一批教师缓存上跑 **主产物 B（2 步直线化，极致 1080p 提速）**，既保证换同系新底模不白费，又把当前原生 1080p 提速榨干到极致！

### 0.4 底模与精确 HuggingFace 路径（官方剪枝原版优先，已核实）
- **准确 HF Repo**：[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)（国内/备用镜像见 ModelScope 同名仓；⚠️ 务必拉 `ref2va` 而非 `fl2va`！）
  - **G4 96GB / A100 80GB（首选）**：`diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors`（约 **`40.2 GB`**，无损剪枝冗余 AdaLN 投影表；原生 BF16 精确梯度，G4 96GB 整载后仍余 `~55 GB`！）。
  - **A100 40GB（备选）**：`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`（约 **`20–21 GB`**，配合排障条目 #6 的 `FrozenINT8LinearFn` 或 `ai-toolkit qint8`）。
- **为什么用官方剪枝原版（共同祖先）既提上限又保通用性？**
  1. **教师画质上限最高**：官方 `pruned_bf16` **未经过 Turbo 少步蒸馏压缩**，跑 20–28 步完整 CFG (`4.5–6.0`) 全稠密即可产出真正的满血原生 1080p 潜空间 `z_native_bf16`；
  2. **跨模型通用性最高（`85%–95%`）**：`dasiwa v5xs` 与未来 `v6`/写实/动漫分支均从官方剪枝底模分叉；仅训 200 线性层并跳过 `adaln_proj`，回到 `dasiwa INT8` / 未来新底模通过本仓库 16-bit BF16 旁路（`bf16_lora=200/200`）几乎零损迁移！
  - *(注：若 24h 主训练号直接针对 `dasiwa` 做 Phase B 2 步极速蒸馏，也可以官方 `pruned_bf16` 教师的 `z_native_bf16` 为真值、以 `dasiwa`（或官方 BF16 + 4 步 Turbo LoRA）为学生底座来训，兼具满血画质目标与 Turbo 2 步动力学！)*

### 0.5 Google Colab 跨账号接力架构 & 可选 Phase C 强化学习（Flow-DPO）对齐

**完全支持拆开换号接力！而且在 Google Colab（常见 A100-40GB、单次运行 12–24h、断连清 `/content`）下，把「教师备料」与「主训练 / RL 对齐」拆成多号接力是最高效、最抗断连的打法**：

- **为什么跨号交接极轻？**
  - 150 组 `1080p` (`22f, T_lat=7`) 离线教师包（含 `x_082/050/018`、`v_dense`、`z_teacher_clean` 与 `text_emb`）每组仅约 `21 MB`，总计仅 **`~3.1 GB`（压缩后 `~1.8 GB`）**；即便 `39f` 150 组也仅 **`~5.4 GB`**！
  - **跨账号中转强烈推荐用「私有 HuggingFace Repo（单枚 `HF_TOKEN`）」**（而非绑定单账号的 Google Drive）：Colab 机房拉/推 HF 私有库速率高达 `200–500 MB/s`，任意新号填同一个 `HF_TOKEN` 2 分钟即可无缝接棒！
- **三段式跨号接力排班**：
  1. **Colab 号 1（离线教师备料 + 烟雾验收 · 约 2.5–3.5 小时）**：
     - 下载 `Qwen3-VL-32B` + `Sol-H3-VAE` + 官方剪枝底模 → 生成 150–200 组高质量 1080p 教师包 → 跑通 20 步烟雾测试确认代码 0 bug → 将 `teacher_cache_v1.tar.zst`（`~2–4 GB`）与训练脚本上传至私有 HF Repo！
  2. **Colab 号 2（满血纯训练 Phase A → Phase B · 12–24 小时零浪费）**：
     - **彻底不用下 `Qwen3-VL-32B`（省 15GB）也不用下 `Sol-H3-VAE`（省 3.4GB）**！开机 10 分钟内拉下 `teacher_cache_v1.tar.zst` + 官方剪枝底模直接满载开训；
     - 每 **`150–200` 步（~25–40 分钟）** 自动把轻量 `.safetensors`（`150–300 MB`）+ `latest_state.pt`（`~600 MB`）+ `logs/metrics.jsonl` 后台同步到私有 HF Repo —— 即使 Colab 随时回收，最多损失 `<30` 分钟且换号秒级 `--resume`！
  3. **Colab 号 3（可选 · Phase C 强化学习 / 偏好对齐 · 4–6 小时）**：
     - **为什么省出 4–6 小时跑 RL 有价值？** Phase A/B 的 $L_2$ 速度匹配天然带轻微均值平滑；Phase C 在像素域直接对齐人眼锐度与防光晕，把发丝、睫毛与远景微纹理再拉一截！
     - **推荐算法：Iterative Flow-DPO（极稳、绝不训崩；优于裸 PPO/Online GRPO）**：
       - 固定从 Phase B 最优 `best_sol_2step.safetensors` 起步，拉回冻结 `Sol-H3-VAE`，每条样本用 2 步稀疏学生采 $K=4$ 个随机噪声分支，解码 3–4 个关键帧按**本仓库已验证的多指标防作弊奖励**打分选出 `(Winner y_w, Loser y_l)` 对：
         $$R(V) = +1.0\,\widetilde{\text{LapVar}}_{1080} + 0.8\,\widetilde{\text{HF}}_{>0.25\%} - 1.5\,\max(0,\; \text{Halo}_{99-1} - 22.0) - 1.2\,\text{Jitter} + 0.5\,\text{SSIM}(V, V_{\text{teacher}})$$
       - 其中 **光晕罚项（`Halo > 22`）+ 时序抖动罚项 + 教师 SSIM 锚定** 从数学上封死「过锐白边 / 闪烁」的 Reward Hacking！
       - 用 `lr = 2e-5`、KL 强锚定 $\beta_{\text{DPO}} = 1000\text{–}2000$ 跑 `300–500` 步 DPO，显存开销与普通 LoRA 完全相同，4–6 小时稳稳出片 `best_sol_2step_dpo.safetensors`！

---

## 1. 给 DeepSeek 的铁律与硬架构契约（动手前必看）

任何情况下都不得违反以下 **5 条硬红线**：

1. **A100 (`SM80` Ampere) 严禁开启 FP8 / NVFP4**：
   - A100 没有原生 FP8 Tensor Core，训练必须走 **原生 `bfloat16`**（80GB 卡加载剪枝版 `dasiwa BF16` `39.1 GiB`；40GB 卡加载冻结 `INT8` 底模 `19.53 GiB` + 激活梯度检查点，详见排障条目 #8）。
2. **目标模块严格限定为 200 个 DiT 线性层，绝不碰 `adaln_proj`！**
   - 仅注入以下 `50 × 4 = 200` 层：
     - `diffusion_model.blocks.{0..49}.attn.qkv_proj`
     - `diffusion_model.blocks.{0..49}.attn.out_proj`
     - `diffusion_model.blocks.{0..49}.mlp.fc1`
     - `diffusion_model.blocks.{0..49}.mlp.fc2`
   - 保存为标准 ComfyUI `.safetensors` 键名：
     - `diffusion_model.blocks.{i}.{suffix}.lora_down.weight`（`[rank, in_features]`, `bf16/fp16`）
     - `diffusion_model.blocks.{i}.{suffix}.lora_up.weight`（`[out_features, rank]`, `bf16/fp16`）
     - `diffusion_model.blocks.{i}.{suffix}.alpha`（标量 `float32`，如 `rank=32, alpha=32.0`）
   - **验收意义**：100% 命中 `chunked_sage.py` 的 `bf16_lora=200/200` 旁路，回到 4060 零转换、无 INT8 精度损失、耗时仅 `+1.5s`！
3. **`comfy_kitchen._C.sol_attn` 没有反向传播 —— 稀疏学生训练必须走可微代理！**
   - 推理内核 `_C.sol_attn` 通过 DLPack 调 C++ CUDA 核，PyTorch Autograd 无法求导。
   - **正确训练实现（防短片外推失效 + 防 `flex_attention` 动态 shape 崩溃！）**：
     1. 按与 `chunked_sage.py` 完全一致的 **`2d_frame` Morton 重排**排列视频 token，并将每条样本 padding 到 **3 个静态固定长度桶**（均严格为 `64` 倍数，如 `16384 / 28672 / 49152`），编译 `torch.compile(flex_attention, dynamic=False)`（全程最多编译 3 次，零动态 shape 重编译风暴！）；
     2. 在 `torch.no_grad()` 下按 64-token 块计算质心路由分数，**严格保留 `sink_blocks` + 帧内/邻帧局部块 + 随机抖动 `topk_ratio ∈ [0.15, 0.30]`**（逼着短/中桶模拟 10s/15s 长片的真实 `15%–25%` 极低保留率，避免外推失效！）；
     3. 用 PyTorch 原生 **`BlockMask.from_kv_blocks(...)`** 直接吃入稀疏块索引（`mask_mod` 内严禁写 Python `if`！），送入 **`flex_attention`** 执行完整反向传播；若 SM80 (A100 164KB SMEM) 报共享内存超限，传 `kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}` 根治！（极端报错兜底见排障条目 #2，**严禁静默退回无掩码全稠密！**）
4. **必须先离线生成教师包，卸载大编码器后再开训**：
   - 前 ~1.5–2.5 小时把教师 `(z_up, text_emb, ref_cond, x_sigma, v_dense, z_teacher_clean)` 预编码存盘；
   - 随后彻底卸载 `Qwen3-VL-32B`（`14.9 GB`）与 `VAE`（`3.4 GB`），保证后 20 小时净训练期显存纯净稳定。
5. **所有状态必须按标准可观测契约落盘（给 Grok Bot 夜间值守）**：
   - 训练脚本每 `10` 步向 `logs/metrics.jsonl` 追加一行 JSON，并维护 `logs/train.pid` 与 `logs/train.log`（详见第 2.4 节）。

---

## 2. 阶段一：DeepSeek 部署与启动全流程（Step 0 → Step 5）

> **DeepSeek 角色定位**：自主完成环境体检 → 依赖部署 → 离线教师轨迹缓存 → 20 步烟雾测试验收 → 拉起正式训练并移交 Grok Bot 监控。**遇真实报错时按第 2.5 节决策矩阵结合现场日志灵活处置，切忌死循环重试同一命令。**

### Step 0：硬件硬门槛核验与自动分档（耗时 ~2 分钟）
开机第一步运行硬件探测，**非 G4 / A100 立即中止（绝不在 T4/L4/V100 上凑合！）**，并写入 `logs/env_snapshot.json`：
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /content /tmp
python3 -c "import torch, sys; n=torch.cuda.get_device_name(0); m=torch.cuda.get_device_properties(0).total_memory/1024**3; print(n, f'{m:.1f}GB', torch.__version__, torch.version.cuda, torch.cuda.get_device_capability()); assert ('A100' in n or 'PRO 6000' in n or 'RTX 6000' in n or m >= 38), f'ABORT: unsupported GPU {n} ({m:.1f}GB)'"
```
**自动决策分支**：
- **Colab G4（`NVIDIA RTX PRO 6000 96GB Blackwell SM120`，最优！）**：原生 BF16 整载 `minimax_h3_ref2va_pruned_bf16.safetensors`（`40.2 GB`），剩余 `~55 GB` 显存直接开启 **三长度静态桶混训** `22f (T=7, 40%) + 39f (T=12, 35%) + 73f (T=22, S≈45k, 25%)`，`rank=32`，`grad_ckpt=True`！
- **A100 80GB (`SM80`)**：同上原生 BF16，开启 `22f (50%) + 39f (50%)`（可选 `56f`），`rank=32`，`grad_ckpt=True`。
- **A100 40GB (`SM80`)**：加载 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（`~20 GB`）+ 排障条目 #6 冻结 INT8 反向，切片 `17f / 22f / 39f` 静态桶，`rank=16` 或 `32`，`grad_ckpt=True`。
- 磁盘检查：Colab `/content` 约 `235 GB` 本地盘，工作目录全部建在 `/content/work/`，确保剩余 $\ge 100\text{ GB}$。

### Step 1：隔离环境与依赖安装（Google Colab 特调，耗时 ~10 分钟）
- **Colab 铁律 1（别重装 PyTorch！）**：Colab 预装的 PyTorch + CUDA 轮子已高度适配，先运行 `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` —— 若已满足 `torch >= 2.5`，**绝不执行 `pip install --upgrade torch`**（极易破坏 Colab cuDNN/NCCL 且浪费 15 分钟），仅轻装缺失依赖：`pip install -q safetensors einops bitsandbytes huggingface_hub hf_transfer tqdm zstandard`。
- **Colab 铁律 2（高速本地盘 + 异步持久化）**：所有底模、教师缓存、临时文件必须放在高速本地盘 **`/content/work/`**（绝不要直接在 `/content/drive/MyDrive/` 上跑 DataLoader 读写，FUSE 会拖死吞吐！），仅把最终紧凑产物与每 150–200 步 checkpoint 异步同步到 **私有 HuggingFace Repo（推荐，跨 Colab 账号通用！）** 或 Google Drive。
- **Colab 铁律 3（规避 `/dev/shm` 崩溃）**：DataLoader 统一设为 `num_workers=0, pin_memory=True`（本地 NVMe 读取单个 `21 MB` 教师张量仅需 `5 ms`，相对 `18,000 ms` 单步耗时零开销，彻底杜绝 Colab 容器 `/dev/shm` OOM）。
- 拉取本仓库 `comfyui-h3-seamkit`（复用 `Split-Band RoPE` 公式、`2d_frame` Morton 置换与 200 层键名校验）及精简版 `ComfyUI` 模型定义（`comfy/ldm/minimax/model.py` + `vae.py`）以保证 100% 字节级对齐！

### Step 2：7h 教师号离线生成「官方满血 BF16 24 步 CFG 教师包」（耗时 ~3–4 小时，0 外部视频集！）
**为什么必须用「官方满血 BF16 24 步 CFG 全稠密」当教师，而不用 3 步 Turbo 当教师？**
因为**蒸馏 LoRA 的画质上限等于教师上限**！用 `dasiwa turbo 3 步` 当教师只能提速却永远跨不过 Turbo 涂抹感；而在 **7h 教师号（G4 96GB / A100 80GB）** 上用官方 **`minimax_h3_ref2va_pruned_bf16.safetensors`** 跑 **24 步完整 CFG (`4.5–6.0`) + 100% 稠密 + `Split-RoPE 1.32`**，既能产成真正满血原生 1080p 真值 **`z_native_bf16`**，又能顺带构造 4060 二采真实面对的糊底输入 **`z_up`**，一举解决 **画质逼近原生 + 去放大器模糊 + 稀疏提速**！

1. **直接复用公开高质量提示词集（5 分钟拼出 160 条，免手写！）**：
   - 从 HF 拉取 **`BestiVictoryLab/VGA-Bench`**（`prompts/200-aesthetic-prompts.csv`，覆盖人像/构图/光影/建筑/微距/自然植被）+ **`meta-ai-for-media-research/movie_gen_video_bench_no_generations`**（`["test"]`，补齐运镜/夜景/文字/高动态物理），均衡抽样 160 条（其中 30–40 条挂开源高清单人物图按官方 H3 6 段 R2V 模板包装，其余按 T2VA 模板包装），预留 **8 条固定验证样本**不参与训练。
2. **多长度桶满血生成并落盘（存至 `/content/work/teacher_cache/sample_XXXX.pt`）**：
   - 按多长度桶分配帧长（**G4 96GB**：`22f 40% / 39f 35% / 73f 25%`；**A100 40GB**：`17f 40% / 22f 40% / 39f 20%`），彻底杜绝单一短片外推到 10s/15s 失效！
   - 用官方 `minimax_h3_ref2va_pruned_bf16` 跑 **20–28 步全稠密 + CFG `5.0` + `Split-RoPE 1.32` + `attn_temp 1.08`** 直接生成满血原生 `1920×1088` 干净潜空间 **`z_native_bf16`**（G4 96GB 单条仅 ~45–75 秒，160 条约 3 小时）；
   - **构造真实 4060 二采糊底 `z_up`**：将 `z_native_bf16` 经 `Sol-H3-VAE` 解码 → 双三次降采到 `960×544` → 重新编码为 `480p` 潜空间 → 经 **`h3_upscaler_sharpness_2000steps_v0.1_fp32 (bf16)`** 精确 `2.0×` 放大回 `1920×1088`，得到与 4060 二采起跑线 100% 同分布的 **`z_up`**！
   - 落盘字段：`z_native_bf16`（BF16）、`z_up`（BF16）、`text_emb`（Qwen3-VL-32B Layer-50）、`ref_cond`（若有）、`meta`（帧长/桶号/prompt）；训练期按任意 $\sigma \in [0.12, 0.85]$ 在线采 $\varepsilon \sim \mathcal{N}(0, I)$，构造 $x_\sigma = (1 - \sigma) z_{\text{up}} + \sigma \varepsilon$（及 30% 干净自回归混合 $(1-\sigma) z_{\text{native\_bf16}} + \sigma\varepsilon$ 防止低噪漂移），精确回归指向满血原生 `z_native_bf16` 的速度目标：
     $$v^*_{\text{native}}(x_\sigma, \sigma) = \frac{x_\sigma - z_{\text{native\_bf16}}}{\sigma}$$
3. **打包推送至私有 HF Repo（供 24h 主训练号直接接棒！）**：
   - 跑完 Step 3 烟雾测试确认代码 0 bug 后，将 `/content/work/teacher_cache/` 压为 `teacher_cache_v1.tar.zst`（约 `3–6 GB`），用 `HF_HUB_ENABLE_HF_TRANSFER=1` 推至私有 HF Repo；**24h 主训练号开机无需再下 `Qwen3-VL-32B`（15GB）与 `VAE`（3.4GB）**，10 分钟内满血开训！

### Step 3：5 分钟烟雾测试（Smoke Test，6 项全绿才允许进正式长训！）
在正式跑 20 小时前，**强制执行 20 步短跑验证**，以下 6 项必须全部通过：
1. **稀疏性真生效验证**：打印实际稀疏保留率 `keep_ratio`（目标在 `tau=1.50` 下视频块保留率约 `18%–35%`，若 `>90%` 说明稀疏掩码没生效，立即停机修 bug！）。
2. **梯度回传验证**：检查 `blocks.0.attn.qkv_proj.lora_down.weight.grad` 与 `blocks.49.mlp.fc2.lora_up.weight.grad` 均非 `None` 且范数有限（确保 `grad_ckpt` 与冻结底模没截断前层梯度！）。
3. **20 步 Loss 单调下降 & 0 NaN**：`loss_step20 < loss_step1` 且 `grad_norm < 10.0`。
4. **显存安全余量**：峰值显存 $\le 85\%$ 总显存（80GB 卡 $\le 68\text{ GiB}$，40GB 卡 $\le 34\text{ GiB}$），确保连跑 20 小时零内存泄漏。
5. **落盘键名与 `chunked_sage.py` 旁路兼容性校验**：导出一份测试 `.safetensors`，运行断言脚本确认：
   - 总计恰好 `200 × 3 = 600` 个张量（`lora_down`, `lora_up`, `alpha`），**0 个 `adaln_proj` 键**；
   - 键名全部匹配 `diffusion_model.blocks.{0..49}.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2}.lora_*`；
   - 无 `NaN/Inf`，形状与 H3 维度严格一致（`qkv_proj: [21504, 5376]`, `out_proj: [5376, 7168]`, `fc1: [28672, 5376]`, `fc2: [5376, 14336]`）。
6. **验证集基线打点**：记录 `step=0`（LoRA=0 初始化）时稀疏学生在 8 条验证样本上相对稠密教师的余弦相似度 `val_cos_step0`（通常在 `0.94–0.98`），用于夜间监控验证真实收益！

### Step 4：正式训练超参、分阶段交付与标准可观测契约
- **推荐训练配置（20 小时净训练，预计 `3,000–4,000` 有效优化步）**：
  - **优化器**：`AdamW8bit`，`lr = 1e-4`（100 步 warmup + cosine 衰减至 `1e-5`），`weight_decay = 0.01`，`grad_clip = 1.0`，`grad_accum = 4`。
  - **LoRA 规格**：`rank = 32, alpha = 32.0`（若 40GB 偏紧改 `rank = 16, alpha = 16.0`，仅 `148 MB`），仅 200 DiT 线性层。
  - **损失函数（速度匹配 + 高频加权）**：
    $$\mathcal{L} = w(\sigma)\,\|v_{\text{student, sparse}}(x_\sigma, \sigma) - v_{\text{teacher}}^*(x_\sigma, \sigma)\|_2^2 + 0.15\,\|\nabla_{xy} v_{\text{student}} - \nabla_{xy} v_{\text{teacher}}^*\|_1$$
    其中空间梯度项 $\nabla_{xy}$ 显式惩罚低噪阶段的纹理发糊，并给低噪步 $\sigma \in [0.15, 0.35]$ 更高权重 $w(\sigma) = 1.5$（因为这正是消除 Step 3 Dense 的最关键区间！）。
  - **两阶段稳妥排班（一晚拿双产物）**：
    - **Phase A（Step `0 → 2000`，~11 小时）**：专攻 **主产物 A（3 步全稀疏恢复：`x_082 / x_050 / x_018` → `v_dense`）**，每 250 步保存 checkpoint 并算验证集 `val_cos`，锁定最优 `best_sol_restore.safetensors`（跨 H3 同系新模型通用性最高！）。
    - **Phase B（Step `2000 → 3500`，~8–9 小时，从 Phase A 权重热启并另存目录！）**：切换/混入 **2 步直线化目标（$\sigma \in \{0.78, 0.35\} \to z_{\text{teacher\_clean}}$）**，产出 **主产物 B `best_sol_2step.safetensors`**（实现原生 1080p 极限 `2.8×` 提速！）。
- **标准可观测输出（供 Grok Bot 夜间值守）**：
  - `logs/train.pid`：当前训练进程 PID
  - `logs/train.log`：标准 stdout/stderr 日志
  - `logs/metrics.jsonl`：每 10 步追加一行（严格 JSON）：
    ```json
    {"ts": 1763980000, "phase": "A", "step": 250, "loss": 0.0184, "grad_norm": 0.82, "lr": 9.6e-5, "keep_ratio": 0.24, "vram_gb": 54.2, "step_sec": 18.6, "val_cos": 0.9912, "val_l2_rel": 0.094, "eta_h": 14.2}
    ```
  - `checkpoints/phaseA/` 与 `checkpoints/phaseB/`：每 250 步落盘 `.safetensors` + 优化器断点 `latest_state.pt`（支持秒级 `--resume`）。

### Step 5：DeepSeek 现场报错排障决策矩阵（真实报错怎么自主判断！）

> **总原则**：先看最后 50 行 traceback 定位真实根因；每类错误最多尝试 **2 次有依据的修正**；严禁为了“跑通”而违背第 1 节的 5 条红线（如偷偷关掉稀疏变成稠密学生，或把 `adaln_proj` 塞进 LoRA）。

| # | 典型报错现象 / 症状 | 根本原因 | 现场诊断命令 / 判断依据 | DeepSeek 标准处置动作 |
|---|---|---|---|---|
| **1** | `torch.OutOfMemoryError: CUDA out of memory` | ① 未开 `grad_ckpt`；② 样本帧长 `39f` 过长；③ 编码器没卸载 | `nvidia-smi` 看是否残留旧进程；检查 `T_lat` 与 `activation_checkpointing` | 按顺序降级：① 确认 `Qwen3-VL`/`VAE` 已 `del` + `empty_cache()`；② 开全层 `torch.utils.checkpoint`；③ 切片从 `39f` 降至 `22f`（$T_{\text{lat}}=7, S\approx 14.5\text{k}$）；④ 仍爆则 `rank 32→16` 或切冻结 INT8 底模。 |
| **2** | `flex_attention` 编译失败 / `Triton out of shared memory` / `InductorError` | PyTorch/Triton 版本与 SM80 共享内存上限冲突 | 看报错是否含 `triton` / `inductor` / `flex_attention` | ① 设 `TORCH_LOGS="+inductor"` 检查是否因动态 shape 触发重编译（强制 padding 到固定 `64` 倍数 + `dynamic=False`）；② 改用 `torch.compile(flex_attention, mode="default")`；③ 若彻底不可用，因切片 $S \le 15\text{k}$ 仅 `240` 个 64-block，可退用 **`torch.nn.functional.scaled_dot_product_attention(..., attn_mask=block_expanded_mask)`（Flash/MemEff 后端，逐块等价且支持完整反向！）**，**绝不退回无掩码全稠密**！ |
| **3** | `loss = NaN / Inf` 或 `grad_norm > 50` | ① FP16 下溢/上溢；② 学习率过高；③ 个别教师缓存含脏值 | 检查 `torch.isfinite(x_sigma).all()` 与 `torch.isfinite(v_target).all()` | ① 强制模型/激活走 **`bfloat16`**、LoRA 主参数与优化器状态走 **`float32`**；② 过滤掉非有限值的损坏缓存样本；③ 从上一个健康 ckpt `--resume`，`lr` 减半（`1e-4 → 5e-5`），确保 `grad_clip=1.0`。 |
| **4** | `keep_ratio > 0.90` 或 `< 0.03` | 质心路由 `tau` 阈值尺度与 `sm_scale` 不匹配（差了 $1/\sqrt{128}$） | 在 `step=0` 打印 `scores.min/max/mean` 与各层平均 `keep_ratio` | 核对块分数公式必须带 `1 / math.sqrt(head_dim)`！若用 `tau` 难以标定，直接切为等价 **`topk_ratio = 0.25`**（保留 `sink_blocks` + top `25%` 空间块，与 `chunked_sage.py` 的 `sol_topk_ratio` 100% 对齐！）。 |
| **5** | `blocks.0` LoRA 梯度全 `0` 或 `None`（仅末层有梯度） | 在冻结底模上误用了 `with torch.no_grad():` 包住主干线性层 | 检查 `blocks.0.attn.qkv_proj.lora_down.weight.grad` | 移除主干前向的 `no_grad()`，改为仅冻结权重 `p.requires_grad_(False)` 并保持输入 `x.requires_grad` 贯通！ |
| **6** | **40GB 卡加载 `dasiwa INT8` 反向报错**（`QuantizedTensor` 不支持 backward） | ComfyUI `QuantizedTensor` 仅实现推理前向，未定义 `backward` | 报错含 `QuantizedTensor` / `does not require grad` / `NotImplementedError` | 编写 20 行自定义 `FrozenINT8LinearFn(torch.autograd.Function)`：前向走原生 INT8 GEMM，反向 `dL/dx = grad_out @ dequant(W_int8)`（逐层即时反量化、用完即弃，`dL/dW` 返回 `None`），LoRA 分支正常走标准 BF16 `F.linear` 自动求导！ |
| **7** | `Qwen3-VL-32B` 文本缓存阶段 40GB OOM | 32B 模型全精度约 `64 GB`，INT8/NF4 约 `15–18 GB` | 看是否以 BF16 加载了 `Qwen3-VL-32B` | 必须加载社区量化版 `Qwen3-VL-32B`（与本地同款 `14.9 GB`）或开 `bitsandbytes` 4-bit/8-bit，逐条编码完立刻落盘并彻底卸载。 |
| **8** | 模型下载极慢 / 超时 / 403 | 云端机房连 HuggingFace 限速或镜像问题 | `curl -I` 测连通性 | 按机房区域切换 `HF_ENDPOINT=https://hf-mirror.com` 或开启 `HF_HUB_ENABLE_HF_TRANSFER=1` + 断点续传下载；优先从用户指定网盘/镜像拉取。 |
| **9** | 训练 300 步后 `val_cos` 停滞不升甚至低于 `step=0` | ① 学生和教师的 `rope_hf_scale` / `Morton` 逆置换没对齐；② `v_target` 符号反了（`ε - z` vs `z - ε`） | **黄金自检**：把学生临时切稠密 + LoRA=0，算 `cos(v_student, v_dense)` 必须 **`> 0.999`**！ | 若稠密自检 `< 0.999`，立刻停机检查 RoPE / Morton / `sigma` 传入格式（对齐 `model.py` 直到 `> 0.999` 才可开训！）。 |
| **10** | 磁盘空间报警 / Colab 断连或宿主机回收 (`SIGTERM`) | ① 保留过多优化器快照；② 未及时把 ckpt 同步出 `/content` | `df -h /content` | 本地只留**最近 1 份** `latest_state.pt`（覆盖写），每 `150–200` 步自动后台推送 `.safetensors` + `latest_state.pt` + `logs/metrics.jsonl` 到私有 HF Repo；换号新开 Colab 后拉下 `teacher_cache_v1.tar.zst` + `latest_state.pt` 直接 `--resume` 秒续！ |

---

## 3. 阶段二：Grok Bot 夜间值守监控简明要求（部署后按实际微调）

> **设计原则**：Grok Bot 不改动训练代码架构、不乱调超参，只做 **轻量低耗的周期巡检 + 明确止损 + 次日晨报**。待 DeepSeek 完成 Step 0–4 正式拉起训练后，再根据实际日志路径和步速把轮询周期与告警阈值微调定稿。

### 3.1 常规巡检（每 15–30 分钟一次，只读 `logs/metrics.jsonl`、`logs/train.pid`、`nvidia-smi`、`df -h`）
- **进程与算力健康**：`train.pid` 存活；GPU 利用率持续 $> 70\%$；显存稳定无单调爬坡泄露；`step_sec` 无异常突增（$> 2.5\times$ 基线）。
- **收敛健康**：`loss` 总体呈下降趋势；`val_cos`（稀疏学生 vs 稠密教师）高于 `step=0` 基线并持续爬升；每 250 步有新 `.safetensors` 落盘且磁盘剩余 $> 20\text{ GB}$。
- **阶段切换确认**：在预定里程碑（如 `step=2000`）确认 Phase A 最优权重已安全拷贝冻结，再进入 Phase B（2 步直线化）。

### 3.2 异常分级与自动化止损动作
- **黄灯（记录并通知，继续观察）**：单次 `grad_norm` 尖峰但未 `NaN`；磁盘剩余 `10–20 GB`（清理过期临时文件，**绝不删 `.safetensors` 产物**）；单次 `val_cos` 微幅震荡。
- **红灯（立即介入或安全停机通知主人）**：
  1. 连续 3 次采样 `loss` 为 `NaN/Inf` 或较历史最低暴涨 $> 3\times$；
  2. 进程意外退出，或 `metrics.jsonl` 超过 `15 分钟` 无新记录且 GPU 利用率持续 `0%`；
  3. 磁盘剩余 $< 5\text{ GB}$；
  - **自动恢复上限**：仅允许从最后一个健康 checkpoint 自动执行 **最多 1 次 `--resume`**（若是 `NaN` 则自动把 `lr` 减半）；若再次触发红灯，**立即保存最后 200 行日志、停止进程保留已有最优 `.safetensors` 产物并向主人报警**。

### 3.3 次日 24h 结束前的收尾晨报
输出一份极简交付清单，方便主人直接下载回本地 RTX 4060 验收：
1. 最终推荐下载的 **2–3 个 `.safetensors` 权重路径与大小**（Phase A 最优 `best_sol_restore.safetensors` + Phase B 最优 `best_sol_2step.safetensors`）；
2. 训练全程核心曲线摘要（总步数、起止 `loss`、`val_cos` 从 `step=0` 到终点的涨幅、耗时与总重启次数）；
3. 回到本地 RTX 4060 的 **3 步验收指令**（挂载到封装工作流 `#19` → 确认日志出现 `[SeamKit] ... | bf16_lora=200/200` → 跑 `10s 1920×1088` 对比 `Exp 1` 的耗时与 `@1080p LapVar / Halo / Jitter`）。
