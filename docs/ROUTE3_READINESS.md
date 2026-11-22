# Route 3 开机前审计（2026-11-22）

> 审计对象：`docs/ROUTE3_A100_TASK_SPEC.md`
> 审计方式：读 `comfy/ldm/minimax/model.py` 全文逐条对行号 + 本机 torch 2.14 实测 + 对 `HANDOVER_DOCUMENT.md` / `.workbuddy/memory/` 交叉核对
> 结论一句话：**任务书是设计稿，不是可执行包。现在开机 = 拿 A100 的小时数写代码。**

---

## 0. 结论

| 项 | 现状 |
|---|---|
| 任务书要求的 8 个脚本 | 原状 **0 个**；v2 已交付 4 个（内核 / 可训练块 / 预检 / 单测） |
| 引擎数学 | **本机已验证成立**（LSE 合并误差 3.3e-7；18 项单测全绿） |
| 训练可行性 | **原 DiTBlock 直接反传会崩、融合 RoPE 无 autograd**（均本机复现）→ v2 已把训练整体砍掉 |
| Oracle 权重 | 云上只有 INT8 量化 pruned 底模，**没有任务书要求的 BF16 稠密** |
| 平台假设 | 任务书写 AutoDL `/root/autodl-tmp`，实际是 **Colab，`/content` 是临时盘** |
| 测试片 | 3 条 Case 一条都不存在，需现生成 |

按原任务书开机，9.25h 里预计 4–5h 花在写代码和修 bug。
**v2 已把预算压到 3h、砍掉训练、把「要不要继续」变成一个 20 分钟就能回答的探针问题。**
仍缺 `engine.py` 与 `probe_attn.py` —— 这两个不写完，开机仍然跑不出任何有意义的结论。

---

## 1. 源码级事实核对：任务书的 5 个地雷，逐条对上了

| # | 任务书说法 | 实读结论 | 影响 |
|---|---|---|---|
| 1 | 3D RoPE：`head_dim=128`，前 96 维 split-half（步距 48），后 32 维 NoPE；t=(0..15,48..63)、h=(16..31,64..79)、w=(32..47,80..95) | ✅ **完全正确**。`rope_freqs()` L558-566 返回 `[S,96] = cat([t16,h16,w16])` 再自拼一倍；`rope_rotation_table()` L156-162 取 `angles[:, :48]`，配对步距 48 | 引擎按这个索引写即可，不用试错 |
| 2 | `FRAME_PER_TOKEN=(1,4,4,4,4)`、`FRAME_RESCALE=5/3` | ✅ L32-33。**时间轴单位是 40Hz 音频帧**（17 帧 = 28.33 单位 = 0.708s）。因此 T=37→124 帧、T=107→362 帧，与任务书"5s/T=37""15s/T=107"精确吻合 | 任务书的帧数自洽，可信 |
| 3 | 非均匀 `denoise_mask` 触发 `vecs[row]` 物化 `[N,5376]` | ✅ L254-270 `_mod_row` / L781-796 `rows_to_mod_index`。per-token LongTensor 索引确实会展开整块 | 必须按连续段传标量 int，任务书方案对 |
| 4 | PDD 蒸馏头按 `sigma_v` 选头 | ✅ L349-380，`pdd_grid()` L44-46 | mask 必须二值 |
| 5 | KJNodes 签名冲突（Stage 4） | ✅ `minimax_block_lowmem_forward` 确实把 `h` 装 list、缺 `attention=` 参数 | 只在回迁本机时处理 |

**结论：任务书的技术部分经得起查，问题全在"没实现"和"平台假设错位"。**

---

## 2. 本机已实测（不是推断，可复现）

测试环境：`D:\comfyui\comfyenv\python.exe`（Python 3.12 / torch 2.14.0+cu130 / RTX 4060 Laptop）

| # | 结论 | 证据 |
|---|---|---|
| A | `flex_attention` 能返 LSE，且数值精确 | `return_lse=True`，与 `logsumexp` 最大误差 **4.8e-7** |
| B | `torch.compile(flex_attention, dynamic=False)` 可用 | 首次编译 5.9s；S=8192/56头 **68ms ≈ 28 TF**（同尺寸 SDPA 103ms）；S=48k 推算单层 2.3s |
| B2 | **零编译 LSE 存在**（cudnn / memeff） | cudnn 8192²×56头 **71.9ms**，Q≠K 误差 9.5e-7，`out` 可反传；memeff 95.7ms / 20.1 TF。见 §4 修订表 |
| C | **LSE 合并数学成立** | "BG 键复制 S=5 次" ≡ "加 `log(5)` 偏置后两分支 LSE 合并"，最大误差 **7.2e-7** |
| D | **β 不可省** | 同一组数据不加 β 误差 **0.391**，而输出量级仅 0.093 → 不加偏置等于把背景全丢掉 |
| E | **原 `DiTBlock` 无法反传** | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [207, 256] ... is at version 15; expected 12`（`_mod_gate` 的 `x[a:b].addcmul_` 改的正是 `norm1(x)` 存给反向的 x） |
| F | 不置 `in_training` 连前向都过不去 | `RuntimeError: in-place RoPE operations are inference-only and do not support autograd` |
| G | **即使置了 `in_training`，backward 仍然过不去** | `RuntimeError: Trying to backward through comfy_kitchen.rms_rope_split_half.default but no autograd formula was registered` —— 融合 RoPE 是 custom op，**根本没注册反向** |
| H | 纯 torch 版 RoPE 已写好并通过等价性检验 | 与 `comfy_kitchen` 融合算子最大差 **9.54e-7**；后 32 维 NoPE 原样透传 |
| I | out-of-place block + LoRA 反传已跑通 | tiny 模型 2 层 × rank8：**8 个 LoRA 张量全部拿到梯度**（梯度和 2.19e+10，随机初始化下属正常量级） |
| J | 任务书的 `T1/T2 < 1e-3` 门禁在 bf16 下**不可达** | fp32 实测 2.98e-7（稳过）；bf16 实测 **1.92e-3** —— bf16 只有 8 位尾数，两条不同计算路径天然差 ~2e-3 |

→ **E/F/G 是这次审计最重要的产出**。任何 H3 训练（不只是 Route 3）在这个版本上都会连炸两次：先撞原地算子，改完再撞无 autograd 的 custom op。v2 因此直接砍掉 Stage 3。
→ **J 会直接导致误判停机**：按任务书的 1e-3 卡，T1 永远不过，Agent 会以为算法写错了。


---

## 3. 缺失清单

### P0 —— 不补就会烧卡

| # | 缺什么 | 现状 / 依据 | 补法 |
|---|---|---|---|
| P0-1 | **全部执行脚本** | `preflight_check.py`、`watchdog.sh`、`run_pipeline.sh`、`splitstream_h3_engine.py`、Stage 0–4 驱动，一个都没有 | 本机写，本机单测 |
| P0-2 | **LSE 通路** | `optimized_attention` 最终走 SDPA，**不返回 LSE**；`flash-attn` 在 torch 2.14+cu130 上无轮子，编译 30–60min（明令禁止） | 用 `flex_attention(return_lse=True)` + `torch.compile`，已验证；flash 作可选加速 |
| P0-3 | **training-safe DiTBlock** | 见 §2-E。**v2：训练已砍，本轮不启用**，但代码已交付（万一要训，不用再查一遍坑） | 已写 out-of-place 版 + 可反传 RoPE + LoRA 包装，tiny 模型反传绿 |
| P0-4 | **前景 mask `M_fg(t)` 生成方案** | 任务书**完全没写**。这是整个算法的第一输入，没有它连 token 划分都做不了 | `comfyui-easy-use` 自带 `easy imageRemBg`（BriaRMBG，纯 PyTorch，无需新 pip 包）。预把 RMBG-1.4 权重（~176MB）放 Drive，脚本直接 import `BriaRMBG` |
| P0-5 | **Oracle 权重口径** | 云上只有 `dasiwa_..._v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed`（20.97G，**pruned + INT8 + turbo**）。任务书要求"纯净 BF16 稠密" | 见 §4 决策 1。**preflight 必须读 header：层数是不是 50、hidden 是不是 5376、heads 是不是 56、有没有内嵌 `comfy_attention.config = comfy_kitchen_int8`** |
| P0-6 | **Colab 无持久盘** | 任务书写 `$WORK_ROOT=/root/autodl-tmp/...`；实际 `/content` 关机即清，Drive 才是持久的 | `WORK_ROOT=/content/h3_route3`（快）+ 产物落盘后 `rsync` 小文件到 Drive（大 ckpt 只留最新一份） |
| P0-7 | **1.3MP 从未实测** | 现网定型是 1280×736 一采；二采 1920×1104 在 **12s** 已用 66.8/79.25GB，交接文档原话"15 秒在 80G 上二采会撞墙"。1536×864 是 +41% token | 全程改 1280×736（见 §4 决策 2） |
| P0-8 | **`deploy_v3.py` 还带 `--highvram`** | 第 197 行；交接文档 §5 明确写"别再上 `--highvram`，二采必 OOM" | 已在本轮修掉（见 §6） |
| P0-9 | **训练路径根本没有可用的 RoPE** | `comfy_kitchen.rms_rope_split_half` 未注册 autograd（见 §2-G）。ComfyUI 内置 `TrainLoraNode` 对 MiniMax H3 也走这条路 → **任何 H3 LoRA 训练在这个版本上都跑不通**，不只是我们 | 已写 `route3_block.rms_rope_split_half_ref`（纯 torch，实测差 9.5e-7）+ `patch_attention()` |
| P0-10 | **T1/T2 门禁阈值写错** | 见 §2-J。按 1e-3 卡 bf16 必误报 | 改成"fp32 < 1e-5 / bf16 < 5e-3"，并在报告里写明口径 |

### P1 —— 会超预算 / 结论不可信

| # | 问题 | 算账 |
|---|---|---|
| ~~P1-1~~ | ~~Stage 3 的 4.5h 只够 200–300 step~~ | **已砍**（v2）：5s/1280×736 = 34k token，单步前反向仍要 ~35s；43M 参数 × 300 step 训不动，且 BF16 训练 / INT8 推理不同源 |
| P1-2 | 门控校准仍保留 | β/α 从 `[50,56]` 升级成相位感知 `[50,56,5]`，共 **14,000 参数**，产物 < 80KB，可直接在 INT8 模型上零开销加载 |
| P1-3 | Stage 0 的"50×56 头 FG→BG 注意力质量" | 48.7k² × 56 头 无法物化。必须设计成：抽样 head（如 8 头）+ 抽样 query 行（如 512 行/层）+ fp32 |
| P1-4 | 3 条测试片不存在 | 需现生成（t2va），且 A/B 必须**同 seed 同 prompt**；1536×864 下 4 步 ≈ 3min/条，可接受 |
| P1-5 | Drive 上原子写不可靠 | FUSE 上 `os.replace` 跨目录语义不明。状态文件一律"先写本地 `.tmp` → `cp` 到 Drive → 比 size" |
| P1-6 | `AUTO_SHUTDOWN_CMD` 在 Colab 未验证 | Colab 没有计费停止 API；`shutdown -h now` 只关 VM。**不要盲填**，先测一次 |

### P2 —— 收尾

- `_r` 底模软链等价性仍未证（交接文档 §8.1）—— Route 3 全程应手动选 `..._turbo_int8_...` 原名，不走 `_r`。
- Stage 4 的 8GB 显存账本 / Dasiwa INT8 回归，依赖引擎定稿后再做。
- 音频：Route 3 只改视频 token，音频在 Prefix 里不动。对比成片建议**直接 mux Oracle 的音轨**，省掉音频 VAE 解码和口型验收。

---

## 4. v2 修订：逐条处置外部审阅意见

| 意见 | 处置 | 依据 |
|---|---|---|
| 砍掉 Stage 3 的 43M LoRA（过拟合 + 与本机 INT8 convrot 冲突） | **采纳** | 43M 参数 × 300 步不可能收敛；且训练走 BF16 路径、本机推理走 INT8 convrot，两条路径不同源，LoRA 的收益无法保证迁移 |
| 总预算 9.25h → 3h 以内 | **采纳** | 见 §7 |
| LSE 公式笔误 `exp(L2+β-L)` | **采纳（是我文字写错，代码一直对）** | `merge_lse` 先做 `l2 += β` 再算权重，等价于 `exp(L2+β-L)`；单测 T2 误差 3.3e-7 |
| 1280×736 = 920 token/帧 | **采纳** | 1280/16/2=40、736/16/2=23 → **920**；T37=34,040、T107=98,440。我此前写的 1296 是 1536×864 的口径 |
| β 升级为相位感知 `[50,56,5]` | **采纳** | 组内 5 个相位的起始像素帧 = **(0,1,5,9,13)**，最远 Δt=**21.667**。已实现 `phase_of_rows` / `beta_for_sequence`，`merge_lse` 支持 `[1,H,S]` 逐 query 的 β |
| Stage 0 优先实测 RoPE 在 Δt=0..21.667 的衰减 | **采纳** | 见 §5 探针规程 |
| 高频时间头占比极高 → 止损转路线 A | **采纳（写成硬门禁）** | 见 §6 |
| 放弃固定坐标 lerp 残差 | **改为 A/B 对照，不删** | 任务书 v3 §4.2.3 把法 A（lerp）留作对照组、法 B（光流 warp）主推 —— 这是对的：没有对照组就说不清 warp 到底赚了多少。已实现 `assembly.py`：已知 2×4 平移下 **法 B 误差 0.073 vs 法 A 0.294（低 75%）** |
| 放弃 flex_attention+compile，用免编译原生 FlashAttention LSE | **采纳 —— 我上一条判断错了，撤回** | ATen 直接暴露了 `_scaled_dot_product_cudnn_attention` / `_scaled_dot_product_efficient_attention`，**零编译、支持 Q≠K、返回 fp32 自然对数 LSE**。本机实测：cudnn 误差 **9.5e-7**、`out` 可反传（q/k 都有梯度）、非连续输入也接受；8192²×56 头 **71.9ms / 26.8 TF**，与 flex 的 68.2ms 基本持平但省掉首编。`_scaled_dot_product_flash_attention` 在 Windows 版 torch 上确实报 `USE_FLASH_ATTENTION was not enabled for build` |
| 运动差分自适应掩码 | **采纳（但必须先做全局补偿）** | 任务书 v3 的 `Active = M_fg ∪ HighMotionBG ∪ Ring` 已经把「局部」写清楚了。已实现 `masks.py`：先估分块位移把全局运镜补偿掉，再看残差。实测：纯推镜 **0.9%** 活跃，加一个不跟镜头走的物体 → **9.5%**（只抓物体） |
| INT8 权重直测 | **采纳** | 与本机 4060 推理同源，结论才能迁移 |

### 4.1 LSE 公式（正写）

设键集 K = `[Prefix | FG] ∪ BG_strided`，BG 侧整体加偏置 β（逐 layer/head/相位）：

```
L1 = logsumexp(QK1ᵀ/√d)                    # branch1
L2 = logsumexp(QK2ᵀ/√d)                    # branch2
L  = logaddexp(L1,  L2 + β)
O  = exp(L1 - L) · O1  +  exp(L2 + β - L) · O2
```

代码里 `beta` 在合并前就加进 `l2`，所以第二项写 `exp(l2 - L)` 是同一个东西。单测 T2（BG 键复制 5 次 vs β=log5 合并）误差 **3.3e-7**。

### 4.1b 二采必须保留（v3 任务书的核心修订）

我上一版把「lerp 残差」连二采一起简化掉了，这是错的。任务书 v3 把二采写成承重件：

1. **BG 关键帧默认 `m = 1` 参与完整二采重绘**（`denoise = 0.30~0.40`）。也就是说背景不是锁死在
   一采 4 步 0.4MP 的糊底上，而是真的被 H3 按 1.3MP 重画一遍 —— 只是把「重绘自由度」从
   「每帧瞎画」收束到「每 17 帧画准一次」。这一步决定了 `bg_hf_sharpness ≥ 0.90× Oracle` 能不能成立。
2. **中间帧靠法 A / 法 B 从关键帧残差还原**：`z_p1_up`（一采上采样，全帧、自带 24fps 运镜）
   + 关键帧细节残差。法 A 是固定坐标 lerp，法 B 是 latent 空间分块位移场双向 warp。
3. **静机位档（T_bg=1）** 直接把二采画好的 `z_bg_out(0)` 铺满背景区（仅当外部给了 4K 底图才 `m=0` 锁定）。

**代价必须写进预算**：二采 5 步 @1920×1104（patch 网格 60×35 = 2100/帧）比一采贵得多。
稠密 Oracle 的二采单条约 5 分钟；压缩后（token 降到 ~40%）约 1 分钟。
所以 **Oracle 也必须跑二采** —— 否则没有 1.3MP 的基准，`bg_hf_sharpness` 无从比较。

已实现：`route3/assembly.py`（法 A / 法 B / 静机位复用）。

### 4.2 三个仍然生效的决策

1. **Oracle 用现有 INT8 dasiwa，不下 BF16**。没有可直链的公开 BF16 权重；preflight 读 header 核对是否 50L/5376/56×128，若内嵌 `comfy_kitchen_int8` 注意力则加 `--use-pytorch-cross-attention` 对齐口径。
2. **全程 1280×736**（920 tok/帧）。1536×864 只在结论站得住后作加测。
3. **训练彻底退出本轮**：只校准 β/α 门控（`50×56×5`，14,000 参数，产物 < 80KB），不做 LoRA。

---

## 5. Stage 0 探针规程（决定继续 or 止损）

**要回答的问题**：H3 的 50 层 56 头里，有多少头的注意力能在 **Δt=21.667**（13 帧，组内最远相位到锚点）上仍然有效？如果绝大多数头只在 Δt≈0 工作，那么把中间帧 BG 换成 17 帧前的锚点，等于给它们喂了错的键——β 补得回「质量」，补不回「相位」。

**做法**（单条 5s 稠密 Oracle 上跑，约 20 分钟）：

1. 在注意力 patch 里挂探针：每层取 **q/k 经 RoPE 之后、attn 之前**的张量。
2. 每个视频相位类（p=0..4）抽 256 行 query，对一个 17 帧窗口内的全部 BG 键算 fp32 logits（`256 × 920 × 17 ≈ 4M`，56 头分批）。
3. 按 Δt 分桶（像素帧）：`0 / 1 / 5 / 9 / 13`，对每 (layer, head) 统计 softmax 质量占比。
4. 定义 **时间头分数** `T[h] = mass(Δt≥5) / (mass(Δt=0)+ε)`，并给每层算中位数。
5. 产物：`probe_attn_stats.json`（50×56 全表 + 每层中位数）+ 一张 50×56 热图。

**判定**（写进 `run_state.json`，由 watchdog 读）：

| 观测 | 结论 |
|---|---|
| 时间头分数 ≥ 0.3 的头占多数（>50%） | 跨帧锚定有效，**继续** S_bg=5 |
| 0.1–0.3 为主 | 降档到 S_bg=3（Δt 上限 8.33）再测 |
| < 0.1 且集中在少数层 | 这些层单独用 S_bg=1（层间混合档位），其余层压缩 |
| 全局 < 0.1 且 Δt=0 质量 > 0.7 | **止损**，转路线 A |

---

## 6. 路线 A（止损备选）

不改良 DiT 内部，改用 H3 已有的 **ref2va** 能力：先把长片切成 N 窗，用第一窗生成的高清背景帧作为后续窗的 reference/cond 锚定，抑制跨窗背景跳变。成本低、无新数学，但只能治「跳变」，治不了「显存」。

触发条件：§5 判定为「止损」时。届时不再动 Route 3 的引擎，本轮剩余预算全部用于跑 4 窗 A/B。

---

## 7. 3 小时时间线（Stage 3 已移除，二采保留）

| 分钟 | 动作 | 预算 |
|---|---|---|
| 0–3 | `deploy_v3.py`（已去 `--highvram`） | 3 |
| 3–13 | 只本地化 5 个模型（≈41.9G，Drive 60–70MB/s） | 10 |
| 13–15 | `preflight_check.py` + flex kernel 预热 | 2 |
| 15–33 | **3 条 Oracle 的一采 + 二采**（5s / 1280×736；稠密二采约 5min/条） | 18 |
| 33–53 | §5 RoPE 衰减探针（抽 8 头 × 50 层） | 20 |
| 53–63 | 活跃掩码（全局补偿差分 + 可选抠像）+ 真实权重上的 T1/T2 | 10 |
| 63–100 | 消融 12 条：`S_bg=5/3 × ring=0/1` + `T_bg=1` × 3 Case（每条一采+二采 ≈ 2.2min） | 37 |
| 100–118 | 法 A vs 法 B 的中间帧 A/B（复用已有 latent，只加解码与拼板） | 18 |
| 118–140 | `bg_hf_sharpness` / `seam_grad_ratio` / 拼板图 / 报告 | 22 |
| 140–160 | 同步 Drive + 原子校验 | 20 |
| 160–180 | 停机缓冲 | 20 |

**合计 3.0h**。二采占了大头（Oracle 18min + 消融内嵌），所以消融只做 4 个档位、不做 6 个。
若探针判定「止损」，跳过 63–118，直接转路线 A（省 55 分钟）。

## 8. 一句话

> 这版改动把「训一个 43M 的 LoRA」换成「花 20 分钟问模型一个问题：你的注意力跨 13 帧还认不认得自己的背景？」
> 认得 → 压缩成立，继续；不认得 → 立刻转路线 A，不再往引擎里投时间。

---

## 9. 还差什么（按开机顺序）

| # | 模块 | 作用 | 状态 |
|---|---|---|---|
| 1 | 掩码 → `(fg, bg, ring, phase)` | `route3/masks.py` 已实现全局补偿差分 + ring + 降采样；`route3_plan` 吃 `fg_mask` | **已完成**（抠像源待接） |
| 2 | `route3/engine.py` | 用 `[Prefix \| FG \| BG]` 自建打包序列的 forward | **已完成**，`s_bg=1` 与原生 forward 逐位一致（diff=0.00e+00） |
| 3 | `route3/probe_attn.py` | §5 探针，产出 `probe_attn_stats.json` | **已完成**，机制在 tiny 模型上跑通 |
| 4 | `route3/stage0_oracle.py` | 3 条 Case 的稠密 Oracle + 落 Drive | 待写（需云） |
| 5 | `route3/stage1_ablate.py` | 6 组消融 + 指标 + 拼板 | 待写（需云） |
| 6 | `route3/assembly.py` | 法 A / 法 B / 静机位复用 | **已完成**（法 B 比法 A 低 75%） |
| 7 | `watchdog.sh` / `run_pipeline.sh` | Colab 版：`/content` 本地盘 + Drive 原子同步 | 待写 |

还缺的两件**非代码**东西：
- 掩码源：RMBG-1.4 权重（176MB，云端拉）或运动补偿差分的实现；
- 3 条 Case 的 prompt 定稿（决定探针在什么画面上量）。

---

## 10. 本轮已改

| 文件 | 改动 |
|---|---|
| `route3/splitstream_attn.py` | 新增 `PHASES` / `phase_of_rows` / `beta_for_sequence` / `warmup`；`merge_lse` 支持逐 query 的 `[1,H,S]` β；`Route3Plan` 带 `phase` 与 `full_phase()` |
| `route3/tests/test_splitstream_math.py` | 口径改为 1280×736=920（并保留 1536×864=1296 对照）；新增相位测试；**18 项全绿** |
| `docs/ROUTE3_READINESS.md` | §4 逐条处置审阅意见；§5 探针规程；§6 路线 A；§7 3h 时间线 |
| `D:\dev\colab\deploy_v3.py` | 去掉 `--highvram`（二采必 OOM） |

本机实测（`D:/comfyui/comfyenv/python.exe`，RTX 4060 8GB）：

```
[PASS] 3D RoPE 通道切分 (t=0..15/h=16..31/w=32..47，48 步距复制)
[PASS] T1 两分支合并 == 整段 SDPA (fp32)    max_err=2.98e-07
[PASS] T1 同上 (bf16)                       max_err=1.92e-03   ← 任务书 1e-3 卡不住
[PASS] T2 BG 键 x5 == beta=log(5)           max_err=3.28e-07
[PASS] T2' 不加 beta 明显错                  no_beta_err=0.460
[PASS] 端到端 flex provider (S=3, bf16)     max_err=1.81e-03
[PASS] 1280x736 = 920 token/帧   /  1536x864 = 1296 token/帧
[PASS] 5s/1280x736 几何：34040 行 = 26680 FG + 7360 BG
[PASS] 最远相位 Δt = 21.667
[PASS] phase_of_rows = t % 5  /  beta 按相位展开  /  逐行 beta 可进 merge
[PASS] 纯 torch RoPE == 融合算子             max_err=9.54e-07
[PASS] 原 DiTBlock 反传（预期失败，已复现）   RuntimeError(inplace)
[PASS] out-of-place block + LoRA 反传        8 个 LoRA 张量有梯度
```

### 10.1 v2 期间新发现的坑（本机实测）

| 现象 | 说明 |
|---|---|
| **压缩率取决于前景掩码，不是 `S_bg`** | `video_split` 最初只按时间切分 → 所有非锚点行都落进 FG，一行都删不掉（自检直接报了"降幅 0%"）。改成「FG 掩码全帧保留 + BG 只在锚点帧保留」后：24% 前景 / S_bg=5 → 保持 13856/34040，**降 59%** |
| **MemEff 的 LSE 会被 padding 到 8 的倍数** | `S=95 → lse 96`、`S=17 → lse 32`，而 `out` 保持真实长度。不切掉 `[..., :S]` 就会和 β 形状对不上；若 S 恰好对齐则**静默错位**。cudnn 不 padding |
| **flex 的 autotune 可能挑到超 smem 的 config** | 4060 上实测报 `No valid triton configs ... Required 131072 / limit 101376`。已加降级链「默认 config → `BLOCK_M/N=64, num_stages=1` → manual」，并缓存成功的那档，避免 50 层 × 2 分支重复试探 |

### 10.2 provider 落地顺序（实测后定稿）

```
推理：cudnn  →  memeff  →  flash(Linux)  →  flex  →  manual
训练：flex（需要 lse 可反传时；但 β/α 校准不需要 —— β 是加在 merge 里的外部标量，
      梯度由 merge 的解析式给出，与内核是否给 lse 梯度无关）
```

`cudnn` 与 `memeff` 速度基本持平（26.8 vs 20.1 TF），但 cudnn 的 LSE 不 padding、`out` 可反传，
所以排第一。第一次成功后缓存，50 层 × 2 分支不再重复试探；失败时清缓存重新探。

