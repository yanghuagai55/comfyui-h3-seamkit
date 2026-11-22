# Route 3 开机前审计（2026-11-22）

> 审计对象：`docs/ROUTE3_A100_TASK_SPEC.md`
> 审计方式：读 `comfy/ldm/minimax/model.py` 全文逐条对行号 + 本机 torch 2.14 实测 + 对 `HANDOVER_DOCUMENT.md` / `.workbuddy/memory/` 交叉核对
> 结论一句话：**任务书是设计稿，不是可执行包。现在开机 = 拿 A100 的小时数写代码。**

---

## 0. 结论

| 项 | 现状 |
|---|---|
| 任务书要求的 8 个脚本 | **仓库里 0 个**（`git status` 只有这份未提交的 md） |
| 引擎数学 | **本机已验证成立**（LSE 合并误差 7e-7） |
| 训练可行性 | **原 DiTBlock 直接反传会崩**（本机复现），必须先写 training-safe 版 |
| Oracle 权重 | 云上只有 INT8 量化 pruned 底模，**没有任务书要求的 BF16 稠密** |
| 平台假设 | 任务书写 AutoDL `/root/autodl-tmp`，实际是 **Colab，`/content` 是临时盘** |
| 测试片 | 3 条 Case 一条都不存在，需现生成 |

按现状开机，9.25h 预算里预计 4–5h 花在写代码和修 bug，剩下 4h 跑不完 Stage 0–2。
**必须先在本机把 ①预检脚本 ②注意力内核 ③training-safe block 做完并单测绿，再开机。**

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
| C | **LSE 合并数学成立** | "BG 键复制 S=5 次" ≡ "加 `log(5)` 偏置后两分支 LSE 合并"，最大误差 **7.2e-7** |
| D | **β 不可省** | 同一组数据不加 β 误差 **0.391**，而输出量级仅 0.093 → 不加偏置等于把背景全丢掉 |
| E | **原 `DiTBlock` 无法反传** | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [207, 256] ... is at version 15; expected 12`（`_mod_gate` 的 `x[a:b].addcmul_` 改的正是 `norm1(x)` 存给反向的 x） |
| F | 不置 `in_training` 连前向都过不去 | `RuntimeError: in-place RoPE operations are inference-only and do not support autograd` |
| G | **即使置了 `in_training`，backward 仍然过不去** | `RuntimeError: Trying to backward through comfy_kitchen.rms_rope_split_half.default but no autograd formula was registered` —— 融合 RoPE 是 custom op，**根本没注册反向** |
| H | 纯 torch 版 RoPE 已写好并通过等价性检验 | 与 `comfy_kitchen` 融合算子最大差 **9.54e-7**；后 32 维 NoPE 原样透传 |
| I | out-of-place block + LoRA 反传已跑通 | tiny 模型 2 层 × rank8：**8 个 LoRA 张量全部拿到梯度**（梯度和 2.19e+10，随机初始化下属正常量级） |
| J | 任务书的 `T1/T2 < 1e-3` 门禁在 bf16 下**不可达** | fp32 实测 2.98e-7（稳过）；bf16 实测 **1.92e-3** —— bf16 只有 8 位尾数，两条不同计算路径天然差 ~2e-3 |

→ **E/F/G 是这次审计最重要的产出**。Stage 3 一开跑就会在第一次 `backward()` 炸，而且是烧着卡炸两次（先撞原地算子，改完再撞无 autograd 的 custom op）。
→ **J 会直接导致误判停机**：按任务书的 1e-3 卡，T1 永远不过，Agent 会以为算法写错了。


---

## 3. 缺失清单

### P0 —— 不补就会烧卡

| # | 缺什么 | 现状 / 依据 | 补法 |
|---|---|---|---|
| P0-1 | **全部执行脚本** | `preflight_check.py`、`watchdog.sh`、`run_pipeline.sh`、`splitstream_h3_engine.py`、Stage 0–4 驱动，一个都没有 | 本机写，本机单测 |
| P0-2 | **LSE 通路** | `optimized_attention` 最终走 SDPA，**不返回 LSE**；`flash-attn` 在 torch 2.14+cu130 上无轮子，编译 30–60min（明令禁止） | 用 `flex_attention(return_lse=True)` + `torch.compile`，已验证；flash 作可选加速 |
| P0-3 | **training-safe DiTBlock** | 见 §2-E，原版必崩 | 写 out-of-place 版：`x = x + attn_out*gate`、`h = h*(1+scale)+shift`；单测要覆盖 backward + LoRA 梯度非零 |
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
| P1-1 | **Stage 3 的 4.5h 只够 200–300 step** | 5s/1.3MP ≈ 48.7k token；teacher 前向 ~27s（A100 40% MFU），student 前反向 ~24s → 单步 50s+。rank-16 × 50 层 = **43M 参数**，300 step 训不动 |
| P1-2 | 同上，改法 | **离线预生成 teacher 目标**（32–64 个 `(x_t, t)` 对，约 15–30min），训练期只跑 student → 单步 ~24s → 4.5h 可到 **650–700 step**。这是唯一能在预算内"训得动"的路径 |
| P1-3 | Stage 0 的"50×56 头 FG→BG 注意力质量" | 48.7k² × 56 头 无法物化。必须设计成：抽样 head（如 8 头）+ 抽样 query 行（如 512 行/层）+ fp32 |
| P1-4 | 3 条测试片不存在 | 需现生成（t2va），且 A/B 必须**同 seed 同 prompt**；1536×864 下 4 步 ≈ 3min/条，可接受 |
| P1-5 | Drive 上原子写不可靠 | FUSE 上 `os.replace` 跨目录语义不明。状态文件一律"先写本地 `.tmp` → `cp` 到 Drive → 比 size" |
| P1-6 | `AUTO_SHUTDOWN_CMD` 在 Colab 未验证 | Colab 没有计费停止 API；`shutdown -h now` 只关 VM。**不要盲填**，先测一次 |

### P2 —— 收尾

- `_r` 底模软链等价性仍未证（交接文档 §8.1）—— Route 3 全程应手动选 `..._turbo_int8_...` 原名，不走 `_r`。
- Stage 4 的 8GB 显存账本 / Dasiwa INT8 回归，依赖引擎定稿后再做。
- 音频：Route 3 只改视频 token，音频在 Prefix 里不动。对比成片建议**直接 mux Oracle 的音轨**，省掉音频 VAE 解码和口型验收。

---

## 4. 三个必须先拍板的决策

### 决策 1：Oracle 用 INT8 还是下载 BF16？

**建议：用 INT8，不下载。**

理由：
1. T1/T2 比的是"同一个权重下，合并注意力 vs 稠密注意力"，量化误差在两边同时出现并抵消 —— 数学等价性检验**不需要** BF16。
2. 真正的风险不是量化，而是 **pruned**：文件名里 `v5xs_pruned`。若层数不是 50，"50 层注意力探针"整节作废。**这条必须由 preflight 读 header 证实**，不能靠猜。
3. BF16 原版 DiT 估 40–45GB，下载 + 落盘 ≈ 30–40min，占预算 6%，且 Colab 磁盘只剩 ~166GB（70G 模型已占）。
4. 若 header 显示层数/hidden/heads 与源码默认值一致 → 直接开测；不一致 → 改任务书的层级结论，再决定要不要下原版。

附带条件：若 checkpoint 内嵌 `comfy_attention.config = comfy_kitchen_int8`，启动必须加 `--use-pytorch-cross-attention`，让 Oracle 和自研分支走同一条注意力；否则 Oracle 是 int8 近似，T1/T2 永远过不了 1e-3。

### 决策 2：分辨率

**建议：全程 1280×736（0.94MP），1536×864 只作为 Stage 4 的加测档。**

理由：1536×864 从未跑过；0.94→1.33MP 是 +41% token；现网 12s/二采已顶到 79.25GB。先把算法打通，再谈分辨率——这比"按任务书上 1.3MP 然后在第一步 OOM"划算得多。任务书"严禁偷偷降分辨率"的用意是防作弊，我们在报告里明示即可。

### 决策 3：第一轮范围

**建议：第一轮只跑 Stage 0 + 1（含 T1/T2 与 3 档消融），≤2.5h，然后落盘停机审图。**
Stage 2 的 β/α 校准放第二轮，和 Stage 3 一起。理由：Stage 2 的"提前胜利条件"（`seam_grad_ratio ≤ 1.20×`）没有 Stage 1 的基线数据就没法定标。

---

### 决策 4：大文件全部云端自取（本机不传）

前提已确认：**本机传不上去，大文件只能云端直接下载**。这决定了几件事：

| 东西 | 大小 | 怎么来 | 备注 |
|---|---|---|---|
| 现有 19 个模型 / 70.46 GB | — | **已在 Drive `h3_assets/models/`** | 不需要下载，也不需要上传。Route 3 只需其中 5 个 |
| Route 3 必需模型 5 个 | ≈40.5 GB | Drive → `/content/local_models`（`deploy_v3.py` 第 8 步） | 只本地化这 5 个，其余保持软链，省 10 分钟 |
| RMBG-1.4 抠像权重 | 176 MB | 云端拉 `huggingface.co/briaai/RMBG-1.4` | `easy imageRemBg` 会自己下；**preflight `--check-net` 已加直链探活** |
| 我写的全部代码 | < 200 KB | `push_file.py`（gzip+base64 走 bridge） | 这个体量传得动，没问题 |
| BF16 稠密底模 | ≈40 GB | **没有可直链的公开权重** | 见决策 1：不赌。真要 BF16，只能云端从 Civitai/HF 现下，且地址得先验证 |

**推论：整条流水线的所有"大件"必须走 Drive 或公网直链，脚本里不许出现任何"从本机传模型"的假设。**
产物也一样：`output/` 已软链到 Drive，LoRA/ckpt 落 Drive，回传本机时按需单拉。

---

## 5. 开机前工作单（本机可做，不需要 GPU 小时）

| # | 事项 | 本机可验证 | 状态 |
|---|---|---|---|
| 1 | `route3/splitstream_attn.py`：三 provider（flex/flash/manual）+ LSE 合并 + β | ✅ | **已完成** |
| 2 | `route3/tests/test_splitstream_math.py`：T1/T2 + 3D RoPE 通道 + 反传 | ✅ 14 项全绿 | **已完成** |
| 3 | `route3/preflight_check.py`：60s 静态体检，写 `preflight_ok.json` | ✅ 本机逻辑已验证 | **已完成** |
| 4 | `route3/route3_block.py`：training-safe DiTBlock + 可反传 RoPE + LoRA | ✅ tiny 模型反传 | **已完成** |
| 5 | `route3/plan.py`：mask → token 计划（FG/BG 划分、`S_bg`、ring 外扩） | ✅ | 待写 |
| 6 | `route3/engine.py`：自建 `[Prefix\|FG\|BG]` 布局的 `_forward` 分支 | ✅ tiny 模型 | 待写 |
| 7 | `route3/stage0_dump.py` / `stage1_ablate.py`：ComfyUI API 驱动 + 产物落 Drive | ❌ 需云 | 待写 |
| 8 | `watchdog.sh` + `run_pipeline.sh`（Colab 版） | ⚠️ 逻辑可验 | 待写 |
| 9 | RMBG-1.4 云端拉取（176MB）+ 3 条 Case prompt 定稿 | ❌ 需网络 | 待定 |

---

## 6. 本轮已改

| 文件 | 改动 |
|---|---|
| `D:\dev\colab\deploy_v3.py` | 第 197 行去掉 `--highvram` —— 与交接文档 §5「定型配置：什么都不加」一致。这条不改，每次开机第一次二采就 OOM |
| `route3/splitstream_attn.py` | 新增。三 provider + LSE 合并 + token 几何划分 |
| `route3/route3_block.py` | 新增。out-of-place block / 可反传 RoPE / LoRA 包装 |
| `route3/preflight_check.py` | 新增。60s 静态体检（含模型 header 结构核对、flex LSE 自检、直链探活） |
| `route3/tests/test_splitstream_math.py` | 新增。14 项，本机全绿 |
| `docs/ROUTE3_READINESS.md` | 本文 |

本机实测汇总（`D:/comfyui/comfyenv/python.exe`，RTX 4060）：

```
[PASS] 3D RoPE 通道切分 (t=0..15/h=16..31/w=32..47，48 步距复制)
[PASS] T1 两分支合并 == 整段 SDPA (fp32)   max_err=2.98e-07
[PASS] T1 同上 (bf16)                      max_err=1.92e-03   ← 任务书 1e-3 卡不住
[PASS] T2 BG 键 x5 == beta=log(5)          max_err=3.28e-07
[PASS] T2' 不加 beta 明显错                 no_beta_err=0.460
[PASS] 端到端 flex provider (S=3, bf16)    max_err=1.81e-03
[PASS] 5s/1280x736 几何                    fg=37584 bg=10368
[PASS] 纯 torch RoPE == 融合算子            max_err=9.54e-07
[PASS] 原 DiTBlock 反传（预期失败，已复现）  RuntimeError(inplace)
[PASS] out-of-place block + LoRA 反传       8 个 LoRA 张量有梯度
```

---

## 7. 开机后的时间线（前提：§5 的 1/2/3/4 已绿）

| 分钟 | 动作 | 预期 |
|---|---|---|
| 0–3 | `deploy_v3.py`（已去 `--highvram`） | ComfyUI 起来，HTTP 200 |
| 3–13 | 只本地化 4 个必需模型（底模 21G / text enc 15.7G / video vae 3.2G / audio vae 0.6G ≈ 40.5G，Drive 60–70MB/s） | ~10min；其余保持软链 |
| 13–15 | `preflight_check.py` | 60s 出 `preflight_ok.json`；任何 fatal → 立刻停机 |
| 15–25 | Stage 0：3 条 Case 的稠密 Oracle（4 步） | 3×3min ≈ 9min |
| 25–45 | Stage 0：注意力探针（抽样 8 头 × 50 层 × 512 query） | 20min |
| 45–150 | Stage 1：T1/T2 单测 + 3 档消融 | ≈100min |
| 150–160 | 落盘 + 同步 Drive + 停机 | 等审图 |

**Stage 3 不在第一轮**。第一轮的价值是拿到"FG/BG 到底该怎么切、β 该多大"的实测数据；没有这组数据，训 LoRA 是在盲训。

---

## 8. 一句话给决策者

> 现在开 A100，最可能的结局是：前三小时在写 attention patch、被 in-place 报错打回来、发现 mask 没方案、最后只跑出两条 1280×736 的样片。
>
> 把 §5 的 1–5 做完再开，前 15 分钟就能拿到 `preflight_ok.json` 和 3 条 Oracle，当天就能出消融表。
