# 任务书（v2 生产容错版）：MiniMax H3 路线 3 —— DiT 内部「多速率前/背景双流 Token 引擎」验证与校准

> **算力预算**：云端 **A100 80GB（本月剩余 10 小时，硬停机红线 9.25 小时）**
> **设计定位**：MiniMax H3 是完整的 **50 层视听叙事大模型**（支持运镜推拉摇移、景深视差、背景动态演进、多镜叙事与原生音画同步），**不是静态动图模型**。本任务书验证的路线 3 必须同时覆盖 **「动态运镜叙事（5× 关键帧压缩）」** 与 **「静机位（极限压缩）」** 两种场景，最终产物迁回本机 **RTX 4060 Laptop 8GB** 运行。
> **监控对象**：面向自动化代理（如 DeepSeek）执行与监控；**内建独立于 Agent 的系统级防断连/防空烧看门狗**，确保即便长时间无人值守或 SSH/API 断开，也绝不空烧卡时、绝不丢产物。

---

## 🔴 一、执行红线与重点标注（DeepSeek 必读，违反即终止）

1. **🔴 严禁不看管直接前台跑**：所有实验必须跑在 `tmux` 会话内，且**必须先启动独立后台看门狗 `watchdog.sh`**（不依赖 DeepSeek 在线）。
2. **🔴 严禁在 A100 计费期间折腾环境超过 10 分钟**：开机先跑 2 分钟 `preflight_check.py`；若缺权重、缺 CUDA/FlashAttn、磁盘 `< 80 GB`，**立即退出并停机**，改用无卡模式/CPU 盘补齐后再开 A100。
3. **🔴 严禁两段式大烧卡无审批直冲（强烈推荐分两段跑）**：
   - 默认设 `ALLOW_STAGE3_UNATTENDED=0`：**第一轮只跑 Stage 0 + 1 + 2（总计 ≤ 3.5 小时）**，完成后自动落盘对比视频、拼板图、指标 JSON 并触发停机，等你人工看一眼效果再决定是否开第二轮（Stage 3 LoRA + Stage 4，≤ 5.5 小时）。
   - 只有你显式设置 `ALLOW_STAGE3_UNATTENDED=1` 才允许单轮直通 Stage 3。
4. **🔴 严禁 Agent 为了“跑通”作弊或放宽门禁**：
   - 严禁放宽单测 `T1 / T2` 数值等价阈值（必须 `< 1e-3`）；
   - 严禁偷偷降分辨率（必须跑 1.3MP `1536×864`）或删掉动态运镜/多镜叙事测试用例；
   - 同一 Stage 连续失败 **2 次** 必须立刻中止写 `ABORT_REPORT.md`，**严禁无限重试烧卡**。
5. **🔴 严禁把产物存在容器临时盘（`/tmp`）**：全部输出、缓存、checkpoint 必须落在平台持久化数据盘 `$WORK_ROOT`（如 AutoDL `/root/autodl-tmp/h3_route3`），且所有状态与权重必须**原子写盘**（先写 `.tmp` 再 `os.replace`）。
6. **🔴 云端纯净基线原则**：A100 上跑 Stage 0–3 时**严禁挂载** `MiniMaxLowVRAMAttention`、`MiniMaxChunkFeedForward`、`MiniMaxH3MemoryEfficientSageAttentionPatch`、`SolAttnMiniMax`；必须用纯净 BF16 稠密模型作为唯一可信 Oracle（仅在 Stage 4 回归时再测 8GB 补丁与 Dasiwa INT8 兼容性）。

---

## 🛡 二、断连 / 无人值守防事故架构（长时间不操作怎么办）

云端最常见的 4 种事故：**① SSH/API 断开进程被杀；② 进程死锁/下载卡死导致 A100 0% 空烧 8 小时；③ 容器偶发重启丢进度；④ LoRA 训崩/发散浪费 4.5 小时**。通过以下 4 重机制兜底：

### 2.1 启动方式（防 SSH / DeepSeek 断连杀进程）
```bash
# 持久化目录（按实际云平台修改，如 AutoDL 用 /root/autodl-tmp/h3_route3）
export WORK_ROOT="/root/autodl-tmp/h3_route3"
export ALLOW_STAGE3_UNATTENDED=0          # 默认跑完 Stage 0-2 自动停机等你审图；想全自动再改 1
export AUTO_SHUTDOWN_CMD=""               # 可选：AutoDL 可填 "shutdown -h now"（测试确认平台支持后再填）
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/artifacts"

# 在独立 tmux 会话内同时拉起看门狗与主流水线
tmux new-session -d -s route3_dog  "bash $WORK_ROOT/scripts/watchdog.sh 2>&1 | tee -a $WORK_ROOT/logs/watchdog.log"
tmux new-session -d -s route3_main "bash $WORK_ROOT/scripts/run_pipeline.sh 2>&1 | tee -a $WORK_ROOT/logs/pipeline.log"
```

### 2.2 独立硬件看门狗 `watchdog.sh`（四重自动熔断，不依赖 DeepSeek 在线）
看门狗每 **60 秒** 独立巡检一次，触发任意一条立即落盘诊断并止损：

| 熔断规则 | 触发条件 | 自动处置动作 |
|---|---|---|
| **① 总时长软/硬停机** | 运行 ≥ `540 min (9.0h)` 软停；≥ `555 min (9.25h)` 硬杀 | 先写 `$WORK_ROOT/STOP_REQUESTED` 让主进程保存当前最优 ckpt；15 分钟后未退出则 `kill -9`；如配了 `$AUTO_SHUTDOWN_CMD` 则自动关机释放计费 |
| **② GPU 空转 / 卡死熔断** | 连续 **15 分钟** `nvidia-smi` 利用率 `< 5%` **且** `pipeline.log` 无新增字节 | 判定死锁 / 网络挂起 / 僵死 → 抓取进程列表与最后 200 行日志到 `ABORT_REPORT.md` → 强杀主任务 → 触发停机 |
| **③ 单阶段超时锁** | 每个 Stage 由 `timeout --signal=TERM --kill-after=60 <上限秒>` 强制封顶 | 防止某一步死循环吃光后续全部预算 |
| **④ 磁盘爆满保护** | `$WORK_ROOT` 可用空间 `< 10 GB` | 写 `LOW_DISK_FLAG`，自动跳过冗余中间 latent dump，只保留核心 `.pt`/`.safetensors` 与报告 |

### 2.3 断点续跑（Idempotent Resume）与状态文件规范
- 每个 Stage 成功验收后，**原子写入** `$WORK_ROOT/artifacts/stage{N}_DONE.ok`（含时间戳、核心指标、产物路径）。
- `run_pipeline.sh` 启动或断线重连后**自动扫描 `_DONE.ok`**，已完成阶段直接跳过，绝不重烧。
- Stage 3（若触发）每 `250 steps` 原子保存 `latest_ckpt.pt` 与 `best_lora.safetensors`，支持从最新 step 无缝续训。
- DeepSeek 仅需每 5 分钟轮询 `$WORK_ROOT/run_state.json`（只读心跳，不阻塞主进程）：
```json
{
  "stage": "stage1",
  "status": "running",
  "elapsed_minutes": 48.2,
  "hard_limit_minutes": 540,
  "retry_count": 0,
  "gpu_util_pct": 94,
  "vram_peak_gb": 41.2,
  "metrics": {
    "t1_lse_max_err": 0.0004,
    "t2_rope_max_err": 0.0003,
    "case2_moving_seam_grad_ratio": 1.19
  },
  "last_heartbeat_utc": "2026-11-22T06:30:00Z"
}
```

---

## ⚠️ 三、必须规避的 MiniMax H3 源码实查地雷（编码必看）

以下全部为本机实读源码确认的约束，实现引擎脚本时逐条核对：

1. **🔴 3D RoPE 真实通道切分（写错必崩）**（`comfy/ldm/minimax/model.py` L156-162, L558-566）：
   - `head_dim = 128`，前 `96` 维为 **split-half 旋转**（配对步距 `48`），后 `32` 维（`96..127`）为 **NoPE**。
   - **时间 `t`**：16 对 = `(0..15, 48..63)`；**高度 `h`**：`(16..31, 64..79)`；**宽度 `w`**：`(32..47, 80..95)`。
   - 目标视频时间起点 `t_0 = cursor`（紧跟 `text + refs`，**不是 0**），时间步长遵循 H3 原生非均匀组 `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`、`FRAME_RESCALE = 5/3`。
2. **🔴 非均匀 `denoise_mask` 的 `+2.1 GB` 显存尖峰**（`model.py` L254-270, L781-796）：
   - 散乱的 per-token `mod_segments` 会触发每层 6 次 `vecs[row]` 物化 `[N_video, 5376]`。
   - **必须**在 Block 0 入口把序列整理为连续段 `[Prefix | BG_strided | FG_full]`，每段只用标量 `int` 索引。
3. **🔴 原地算子与 `malloc_graph` 陷阱**（`model.py` L196, L262, L269, L686-700）：
   - `rms_rope_split_half_` 原地改写 `q, k`；若做跨帧旋转复用必须先 `.clone()`。
   - 本机回迁启用动态显存图时，跨块持久张量必须包在 `comfy_kitchen.allocation.allocation_context()` 中（优先采用下方「锁步多速率架构」，**根本无需跨 50 层常驻大缓存**）。
4. **🟡 PDD 蒸馏头不兼容半透明 mask**（`model.py` L349-380）：
   - Turbo/Dasiwa（`pdd_steps > 1`）按全局 `sigma_v` 选输出头；因此所有前景掩码严格用 **二值 `{0, 1}` + 外扩 `r` 格阴影环归入 `FG`**，避免 `0 < m < 1` 扰动输出头。
5. **🟡 KJNodes 签名冲突（仅 Stage 4 回迁本机时注意）**（`ComfyUI-KJNodes/nodes/minimax_nodes.py` L141-197）：
   - `minimax_block_lowmem_forward` 缺少 `attention=None` 且把 `h` 装在单元素 `list` 里传入 `attn.forward`。最终交付的插件模块不得依赖 `blocks_replace` 的 `attention` 参数，且必须兼容 `isinstance(x, list)`。

---

## 🧬 四、路线 3 升级版架构：锁步多速率双流引擎（兼容动态运镜与完整叙事）

为了不把 MiniMax H3 阉割成“静图背景 + 贴纸人物”，本任务书采用 **「锁步多速率双流（Lockstep Multi-Rate Split-Stream）」**，天然贴合 H3 原生的 `5 token = 17 帧` 时间块结构：

### 4.1 两种背景压缩档位（同一套引擎统一支持）
| 档位 | 适用场景 | 背景时间步长 `S_bg` | 15s (107 latent) 序列总长变化 | 额外 50 层缓存 |
|---|---|---|---|---|
| **叙事运镜档（主测重点）** | 推拉摇移、景深视差、背景光影演变、多镜切换 | **`S_bg = 5`**（每 17 帧独占帧 `5k` 保留 1 份 BG，共 22 帧；也可测 `S_bg = 3`；**背景关键帧默认 `m=1` 参与完整二采高清重绘**） | `138k → ~67k`（**token 砍半，Attention FLOPs 降至 ~23%**） | **0 GB**（50 层与 FG 锁步前向，双向光影/阴影耦合） |
| **静机位极限档** | 固定机位、极慢微动镜头 | **`T_bg = 1`**（整片仅 1 份代表帧 BG，**默认 `m=1` 随二采完整重绘成 1.3MP 高清**；仅在外部提供现成 4K 底图时才设 `m=0` 锁定） | `138k → ~49k`（**token 砍 65%，Attention FLOPs 降至 ~13%**） | **0 GB**（锁步）或 `1.2 GB`（全层静态 KV 复用） |

### 4.2 核心前向、背景重绘余地与两块 Token 合并数学
1. **Block 0 入口重排（运动自适应活跃区，严禁一刀切死背景）**：
   - **活跃全速率流 `FG_full`**：不仅包含人物 `M_fg(t)` 与外扩 `r`-patch 阴影/边缘环，还并入一采中**局部高频运动的背景区域**（`Active(t) = M_fg(t) ∪ HighMotionBG(t) ∪ Ring(r)`，如剧烈飘动窗帘、水流、近景穿行物）——只对真正变化平缓的背景降频，绝不硬切。
   - **慢速背景关键帧流 `BG_strided`**：按 `S_bg` 抽样的背景关键帧（保留各自真实 3D RoPE 坐标 `(t_bg, h, w)`），**在二采中默认保持 `m = 1`（参与完整 `denoise = 0.30~0.40` 高清重绘！）**——绝不把背景锁死在 4 步 0.4 MP 糊底上，而是让 H3 实打实把背景画成 1.3 MP 锐利质感，只是把“重绘自由度”从每 4 帧瞎画一次收束到每 17 帧（或单帧）画准一次。
   - 拼成紧凑单序列：`H = [Prefix | BG_strided | FG_full]`，50 层锁步前进，双向感知阴影、反光与运镜视差，零跨层 KV 内存爆炸。
2. **Attention 内的精确质量守恒合并（Online-Softmax LSE Merge）**：
   - **为什么不能直接拼完就跑普通 SDPA？** 因为 `BG_strided` 被压缩了 `S_bg` 倍（`5×` 或 `107×`），在 Softmax 分母里权重暴跌 `1 / S_bg`，会导致背景光影约束变弱、边缘泛灰。
   - **精确解法（双分支 FlashAttention LSE 合并，全程零 `mask` 张量）**：
     - **Branch 1**：`Q` 对 `[Prefix | FG_full]` 算标准 FlashAttention → `(O_1, L_1)`
     - **Branch 2**：`Q` 对 `BG_strided` 算跨注意力 → `(O_2, L_2)`
       - 当 `S_bg = 5` 时：`BG_strided` 带着真实 `(t_5k, h, w)` 3D RoPE，任意中间帧 `FG(5k+j)` 距最近背景关键帧 `|Δt| ≤ 8` 帧（`≤ 0.33s`），直接跨注意力即可；
       - 当 `T_bg = 1` 时：支持 **Mode A**（把 `Q_fg(t)` 的 `(0..15, 48..63)` 时间通道逆旋转 `-Δt(t)`，获得**逐位等价于全视频同帧 `Δt=0`** 的空间点积）与 **Mode C**（同帧 `Δt=0` + 合成 `K=8` 个稀疏时层键）。
     - **LSE 质量守恒合并**（用每层每头偏置 `β[l, h]` 补回折叠的 `log(S_bg)` 注意力质量）：
       ```python
       L_bg_eff = L_2 + beta[l, h]
       L_new = torch.logaddexp(L_1, L_bg_eff)
       O = torch.exp(L_1 - L_new) * O_1 + torch.exp(L_bg_eff - L_new) * O_2
       ```
3. **动态背景中间帧高清还原（运动对齐残差传播，避免快速运镜重影）**：
   - 一采 3D Upscaler 输出 `z_p1_up`（全 `T` 帧）提供流畅的 24fps 运镜结构底子；二采的 `BG_strided` 在每 17 帧关键帧 `5k` 上**真正重绘出 1.3 MP 锐利细节增量** `δ_bg(5k) = z_bg_out(5k) - z_p1_up(5k)`。
   - **Stage 1 必做 A/B**：对比两种中间帧 `5k+j` 还原法：
     - **法 A（直接线性插值）**：`z_bg_final(5k+j) = z_p1_up(5k+j) + lerp(δ_bg(5k), δ_bg(5k+5), j/5)`（慢镜极快，快摇镜可能重影）；
     - **法 B（Latent 空间光流/位移对齐双向 Warp，主推）**：在 `96×54` latent 网格上对 `z_p1_up` 估轻量帧间位移场，将高清残差 `δ_bg(5k)` 与 `δ_bg(5k+5)` 沿镜头运动轨迹双向 warp 对齐后再融合，使二采画出的砖纹/树叶锐度死死贴着运动物体走、不拖影；
     - 静机位档（`T_bg = 1`）直接将二采刚画好的高清 `z_bg_out(0)`（或外部 4K 底图）复用到全片背景区；人物与高频运动区填入全速率 `z_fg_out(t)`，整体送入 Video VAE 解码。

---

## 📋 五、分阶段任务表与严格 Go/No-Go 门禁（总计 ≤ 9.0h）

> **推荐两轮执行**：**第一轮（Stage 0–2，约 3.0–3.5h）自动停机审图**；确认有价值后再开**第二轮（Stage 3–4，约 5.0–5.5h）**。

| 阶段 | 时间上限 | 核心目标 | 必须产物 | 🟡 Go / No-Go 门禁（不达标立即止损） |
|---|---|---|---|---|
| **Pre-flight** | **0.15h** (10m) | 校验 GPU、磁盘、权重、FlashAttn LSE 接口 | `preflight_ok.json` | 任何缺件/报错立刻退出，严禁现场 pip 编译折腾 |
| **Stage 0** | **0.5h** | 跑 3 个叙事 Case 的 Dense Oracle + 50 层注意力探针 | `probe_attn_stats.json` + 3 条 Oracle mp4 | 测清 50×56 头中 `FG→BG` 的 `Δt=0` vs `Δt≤8f` vs `Δt>8f` 占比，以及动态/静态 BG 跨层余弦相似度 |
| **Stage 1** | **1.5h** | 数学等价单测 `T1/T2` + 免训练消融（`S_bg=5 / S_bg=3 / T_bg=1` × `ring=0,1,2` × `lerp vs flow-warp` × `β`） | `stage1_ablation.csv` + 对比视频与边界切片图 | **硬门槛**：`T1/T2` 误差 `< 1e-3` 才许开消融；消融至少 1 组满足 `fg_psnr ≥ 24 dB`、**`bg_hf_sharpness ≥ 0.85× Oracle`**（确认背景真画出 1.3MP 细节而非 4 步糊底）且 `seam_grad_ratio ≤ 1.8×`，否则停机排查 |
| **Stage 2** | **1.0h** | 轻量 5,600 参数门控校准（`β[50,56] + α[50,56]` 速度场自蒸馏，产物 `< 50 KB`） | `splitstream_gates.pt` + `stage2_contact_sheet.png` | **提前胜利条件**：动态运镜 `Case-2` 与强光 `Case-3` 的 `seam_grad_ratio ≤ 1.20×`、`bg_hf_sharpness ≥ 0.90× Oracle` 且肉眼无接缝/重影 → **跳过 Stage 3，省 4.5h！**（若 `ALLOW_STAGE3_UNATTENDED=0`，此处落盘并自动停机等你审图） |
| **Stage 3** *(条件)* | **4.5h** | 自蒸馏轻量 Attention LoRA（`rank=16`，仅 `qkv/out`，Teacher=Dense H3，Student=Split-Stream） | `splitstream_lora_r16.safetensors` | 仅当 Stage 2 光影/边缘/一采融合仍不达标时启动；每 250 step 评估并原子存 ckpt；连续 2 次 eval 恶化或 20 步 NaN 立即早停 |
| **Stage 4** | **1.0h** | **15s / 1.3MP 完整叙事长片压测** + 8GB 显存账本 + Dasiwa INT8 兼容回归 | `splitstream_h3_engine.py` + `FINAL_REPORT.md` + 15s 对比 mp4 | 15s 长片相对原 4 窗分块：背景跨窗跳变显著下降（或单/双窗直出）、音频口型与叙事动作无崩坏、8GB 峰值估算达标 |

---

## 🎯 六、核心测试用例规格（拒绝玩具静图，贴近真实 MiniMax H3 叙事）

Stage 0–2 必须在以下 **3 条固定 5s（124 帧, `T=37`, 1.3MP `1536×864`, 含原生音频）** 测试片上同种子对比：

1. **`Case-1`（静机位表演叙事）**：固定机位，单人中近景做连续手势/转身 + 对白/环境音（标定 `T_bg = 1` 极限压缩上限与发丝/衣纹细节保真）。
2. **`Case-2`（动态运镜 + 人物大位移叙事 — 重点拷问）**：缓慢推进/横移跟拍镜头（含明显前景-背景景深视差）+ 人物从画面左侧走到右侧 + 脚步声/环境音（检验 `S_bg = 5` 残差传播与动态每帧 `M_fg(t)` 是否能保住真实电影运镜，对比暴露 `T_bg = 1` 的适用边界）。
3. **`Case-3`（强光影交互 / 多镜切换叙事 — 重点拷问）**：暖色夜景霓虹或强侧逆光室内，含 1 次镜头景别切换（如中景切特写）+ 地面/墙面投影交互（检验跨镜切换、人物环境反光与脚底阴影融合、以及音频跨镜连贯性）。

Stage 4 升级到 **1 条 15s（362 帧, `T=107`, 1.3MP `1536×864`）多镜叙事长片**，直接与你现有 `comfyui-h3-seamkit` 原 4 窗分块二采做 A/B 验收。

---

## 📦 七、最终必须回传本地的交付物清单（全部在 `$WORK_ROOT/artifacts/`）

1. `FINAL_REPORT.md`（含 Stage 0–4 全指标表、加速比、峰值显存、失败点复盘与本机迁回指南）
2. `probe_attn_stats.json`（50 层 × 56 头 FG→BG 注意力质量与时间跨度统计）
3. `stage1_ablation.csv` + `stage2_contact_sheet.png`（可直接肉眼检验边界与光影融合）
4. `splitstream_gates.pt`（`< 50 KB`，校准后的 `β/α` 门控参数，支持任意量化模型零开销加载）
5. `splitstream_lora_r16.safetensors`（若 Stage 3 触发，~160 MB）
6. `splitstream_h3_engine.py`（纯净独立 Python 模块，规避了第三章全部 5 个源码地雷，可直接封装进 `comfyui-h3-seamkit`）
7. 15s 对比成片：`oracle_15s.mp4`、`seamkit_4chunk_15s.mp4`、`route3_splitstream_15s.mp4`
