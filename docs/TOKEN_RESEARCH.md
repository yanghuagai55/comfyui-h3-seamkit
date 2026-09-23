# Token 变化研究报告：可测量签名、判据标定与可复现性

> 回应 `TOKEN_RESEARCH_REQUEST.md`（2026-09-23）。方法：四轮泛式文献与开源项目调研
> （镜头边界检测 / 频域长视频 / 时序指标 / 确定性推理 / 许可证核实），来源以论文与仓库
> 原文为准，逐条给出引用；无法核实的标注 [未验]。所有「迁移到本仓库」的建议均考虑
> RTX 4060 Laptop 8GB 约束。

---

## 0. 执行摘要（先给答案）

1. **你们缺的不是新启发式，而是「标定」。** 请求书 §4.1 的三个分——局部变化、
   瞬态脉冲（jerk）、持续性——与 PERSIST（BMVC 2026）论文的三个语义门控
   **逐一对得上**：local change / transient impulse / return-to-trend [cite:3]。
   你们的直觉方向是对的；差异在于 PERSIST 把三信号喂给 FiLM 条件化的 SIREN
   潜在状态分类器、在标注数据上训练出权重，而你们是三个未标定的手工分。
   → 第一优先级不是发明新量，是给现有三信号建**标注集 + 权重/阈值**。
2. **判别框架应该是「对应性 × 持久性」两维**（详见 §2）：镜头边界检测文献里，
   硬切/相机运动/局部物体/闪烁的区分本质是**帧间像素对应性**的三态
   （无对应 / 全局一致对应 / 局部对应破坏）[cite:25][cite:8]，latent 域信号
   （一阶差/jerk/持续性）只覆盖了「变化多大」，缺了「对应性」这个正交维度——
   这正是手挥特写被误判的病因。
3. **「细节密度」是错误的镜头，频域才是对的。** FreeLong 实测：窗口化生成的
   高频分量不稳定——高频 SNR 从 16 帧的 1.0 降到 128 帧的 0.73 [cite:9]。
   你们的观察（细节密区重生成注入伪纹理 +23~35%）与它是同一枚硬币的两面：
   **窗口化高频不稳定**既可能退化也可能过激。判据 B 应为
   **重生成前后高频带能量差（Δ 高频能量）**，而非重生成前的高频密度。
4. **可复现性可以基本修好**：ComfyUI 有官方 `--deterministic` 旗标 [cite:16]；
   cuBLAS 在**同一 GPU + 同一 CUDA 工具链 + 单流**下保证逐位一致 [cite:15]；
   注意力后端按可复现性排序：SDPA-MATH > Flash > Sage [cite:14][cite:18]。
   同会话缓存技巧可以保留，但能升级为「同启动参数即逐位复现」。
5. **许可证红线**：**MAINodes 是 GPL-3.0** [cite:19]——你们借过它的 jerk oracle
   **想法**（想法不受版权保护，合规）；但**永远不要拷它的代码**进本仓库，
   除非整个 seamkit 转 GPL。FreeLong **没有官方代码仓库** [未验]，只能用思想
   （FFT 一行代码自己写，无风险）。其余主力工具全是 MIT/Apache（§5）。

---

## 1. 方法与来源

四轮检索：① 镜头边界检测（SBD）文献与工具；② 频域长视频生成与时序指标；
③ 扩散推理确定性；④ 逐仓库 LICENSE 核实（以仓库 LICENSE 文件原文为准）。
引用编号见文末 Sources。凡 [未验] = 检索不可核实或来源不足。

---

## 2. 第一层：分类学——token 变化的可测量签名

### 2.1 核心发现：判别框架是「对应性 × 持久性」

SBD 文献把"切 vs 动"的区分归结为**帧间像素对应性**（correspondence）：

- **硬切**：相邻帧无真实像素对应 → 光流匹配置信度塌、方向一致性差 [cite:25]；
- **相机运动（推/摇/移）**：光流整体一致，可被全局仿射模型解释，
  RANSAC 内点率高、拟合残差小 [cite:8][cite:25]；
- **局部快速物体（手挥/掠过）**：**全局对应保持、局部对应破坏**——
  残差集中在运动区域，其余区域仿射拟合良好 [cite:8][cite:25]；
- **渐变过渡（fade 等）**：多帧内缓变、整体匹配置信度中等 [cite:1]。

这个框架直接解释你们的失败案例（请求书 §3.2.2）：**连续特写 + 手挥**在
latent 域产生大的一阶差与 jerk（局部变化确实大、脉冲确实尖锐），
所以你们的三个分全部报警；但它的**全局对应性完好**（背景持续可匹配），
真硬切则对应性塌零。缺的就是这一维。

### 2.2 六类变化的签名表（文献信号 → latent 域映射）

| 类别 | 像素域签名（文献） | latent 域映射（本仓库可测） | 持久性 |
|---|---|---|---|
| 真硬切 | 无对应：光流内点率→0、方向一致性差 [cite:25] | d1 大且**状态持久跳变**（persistence 高） | 高 |
| 渐变过渡 | 多帧缓变、匹配置信度中等 [cite:1] | d1 中等、分布**平缓爬坡** | 中 |
| 相机运动 | 全局一致光流、仿射拟合残差小 [cite:8][cite:25] | d1 大但**相机补偿后消失**（你们已有 `_camera_compensate`） | 高 |
| 局部快速物体 | 全局对应保持、残差局部集中 [cite:8][cite:25] | d1/jerk 大，但**空间上稀疏**（token 内通道子集）、补偿后残余 | 低（震荡） |
| 纹理闪烁/细节抖动 | 高频不稳定、低频稳定（FreeLong 视角 [cite:9]） | 高频带能量抖动、低频带平稳 | 低 |
| 量化棘轮（本地特有） | 近静止段「三帧相同+一帧轻推」 | token 周期性签名，见 §2.4 | 周期性 |

关键行是第 4 行：**latent 的空间稀疏性**是你们现在没测的量——手挥只动
token 的局部通道/空间 patch，真转镜动整块。这在 latent 上可用
**变化的空间集中度**（把 token 差按空间 patch 展开，看能量集中度）测量，
与光流 RANSAC 内点率在像素域的角色同构，但不需要解码。

### 2.3 三个分与 PERSIST 三门控的逐项对应

| 你们的分（`_latent_change_profile`） | PERSIST 语义门控 [cite:3] | 差距 |
|---|---|---|
| 一阶差 d1 | local change（局部变化） | 同构 |
| 三阶差分 jerk | transient impulse（瞬态脉冲） | 同构 |
| persistence 窗口均值差 | return-to-trend（回归趋势） | PERSIST 是「边界后状态必须持久更新，而非震荡回原趋势」的**判别式公式**，你们是朴素窗口均值差 |

PERSIST 报告：在 2,727 个视频的子类型诊断上移除 33~80% 的闪烁/字幕/档案
素材误报；与 TransNetV2 同召回下伪事件误报约减半 [cite:3]。**代码已开源
（Apache 2.0），含验证阈值标注的 checkpoint** [cite:4]——即它的「阈值」
是验证集 F1 选出来的，不是拍的。这正是请求书 §6 第二层要的东西的形态。

### 2.4 量化棘轮的频谱签名（本报告新提案，[未验]）

公开文献没有「量化棘轮」的先例（检索无果，这是 H3 因果 VAE 的本地现象）。
但它的结构可测：FRAME_PER_TOKEN = (1,4,4,4,4)，即**每 17 帧一个完整周期、
每 4 帧一个小台阶**。提案：

- 对近静止段的**逐帧亮度/latent 均值序列**做自相关：棘轮应显示
  lag≈4 与 lag≈17 的峰（token 边界与组边界）；
- 或对时间序列做 FFT：**棘轮周期是 4 帧 → 24/4 = 6 Hz**（谐波 3 Hz、12 Hz）；
  **分组周期是 17 帧 → 24/17 = 1.41 Hz**。棘轮段应有 6 Hz 谱线，真静止段没有。
  （原稿写「token 节奏 = 5/17 token/帧 ≈ 7.06 Hz」，那是把「token 数/帧」当成了频率 ——
   **量纲滑移**，数值上也偏 18%。上面自相关的 **lag≈4 / lag≈17 两个数是对的**，只有 Hz 换算错。）
- 临界速度问题（请求书 §3.2.5）由此可答：当运动能量把 7 Hz 谱线
  淹没到信噪比以下，棘轮不可见——**临界速度 = 谱线 SNR 的函数**，
  可在现有素材上标定（同 §3.3 协议）。

---

## 3. 第二层：判据——两个有物理意义的量

### 3.1 判据 A：转镜置信度 = 持久性 × 无对应性

把现有三信号 + 一个新维度压成一个置信分：

```text
confidence(切点) = persistence(状态跳变幅度,你们已有)
                 × non_correspondence(空间集中度/补偿后残余, §2.2 新测)
```

- persistence 高 + 无对应 → 真转镜 → **硬切**（支持请求书 §5 策略表第一行）；
- persistence 高 + 有全局对应 → 相机运动 → 挪边界或补偿（不是切）；
- persistence 低（震荡回归） + 局部对应破坏 → 手挥/物体掠过 → **不切**；
- 这正是 PERSIST 的判别式结构 [cite:3]，但把输入从 RGB 换成你们的
  latent 画像——**latent 域 SBD 没有公开先例**（检索确认，这是空白区），
  你们的素材就是第一份标注集。

### 3.2 判据 B：重生成安全度 = Δ 高频能量

FreeLong 的证据链 [cite:9]：窗口化生成的高频分量不稳定（SNR 1.0→0.73，
空间高频→0.68）。你们的 +23~35% 伪纹理注入是同一现象的过激分支。
判据不是重生成前的细节密度，而是：

```text
safety(区域) = f( Δ高频能量, 锚间高频一致性 )
Δ高频能量   = 高频带能量(重生成后) − 高频带能量(重生成前)   # 同区域同帧
锚间高频一致性 = corr(左锁区Laplacian能量谱, 右锁区Laplacian能量谱)
```

- **Δsharpness 而非绝对阈值**：sharpness 文献的绝对阈值全部是任务特定的
  （例如文档扫描 <200 模糊 / >900 过锐 [cite:22]），不可迁移；
  前后差值是自校准的 [cite:22]；
- **锚间高频一致性是新假设**（[未验]，现有 n=2）：若两侧锚的高频结构
  本身不一致，自由区重生成会"造"出折中纹理——这解释了为什么细节密区
  先出事（高频基数大、折中空间大）；
- 高频量度用 Laplacian 方差（你们已有）或小波高频系数能量 [cite:22]，
  CPU 毫秒级，8GB 显存无压力。

### 3.3 标定协议（具体到数据与指标）

| 项 | 方案 | 来源/许可证 |
|---|---|---|
| 标注工具（对照真值） | **TransNetV2** 在你们成片上跑出转镜帧作为参照真值（F1 96.2 BBC / 77.9 ClipShots [cite:1][cite:2]） | MIT [cite:2] |
| 公开数据集（域外加固） | ClipShots（4,039 视频、128k 硬切 + 38k 渐变 [cite:6]）、AutoShot/SHOT（853 视频、11,606 标注 [cite:7]） | MIT / MIT [cite:6][cite:7] |
| 域内标注集 | ★ **正样本必须来自独立来源**（TransNetV2 输出 + 人工复核）。**不能**用「hunt 验证过的真转镜」当正样本 —— hunt 正是**被标定的对象**，拿它的判定当真值是循环，ROC 会好看但没有意义。负样本 = 手挥特写等失败案例 + hunt 误判为正的样本；另用 `analyze_cut.py`（像素域，独立链路）交叉校验 | 本地 |
| 指标 | 按帧二分类的 **F1**（SBD 标准指标 [cite:1][cite:7]），阈值在验证集上扫出而非拍定（PERSIST 的 checkpoint 即带验证阈值 [cite:4]） | — |
| 残差问题（§3.2.4） | 用干净 A/B 协议测「残差 0/2/5/8 × 内容类型」的台阶矩阵，直接回答"残差 5 与 2 差多少"；门限从矩阵读出 | 本地 |

注意：TransNetV2 仓库本身不带预训练权重 [未验——README 未声明]，推理实现
（`transnetv2-pytorch`，MIT [cite:2]）与权重要分开确认；27 帧滑窗 CPU 推理
成本需实测（论文报告 1.4ms/次但未注明设备 [cite:1]，不可直接采信）。

### 3.4 与「vibe」工具的关系

VBench 的 temporal flickering（相邻帧像素 MAE，极便宜）与 subject consistency
（DINO 特征余弦，中等成本）[cite:13] 可作为**旁路验收指标**（seamfix 已是
像素域旁路，正合）；但它们是整帧全局量，不能定位"某缝是否注入伪纹理"——
定位仍靠 §3.2 的局部 Δ 高频能量。VBench 为 Apache 2.0 [cite:13]。

---

## 4. 第三层：自动化与可复现

### 4.1 策略表自动化

请求书 §5 的表由两个判据驱动，零手调形态：

```text
对每个候选切点：
  conf = 判据A(持久性 × 无对应性)
  若 conf > θ_hardcut（验证集标定）           → 硬切（缝挪到最近独占帧）
  否则若 安静(低频平稳, 棘轮谱线不可见)        → 锚定重叠（已验证有效）
  否则若 判据B1(锚间高频一致性, 事前可算) 达标  → 缝窗重去噪（E-3）
  否则                                        → 挪到 conf 次高的候选点

重去噪跑完之后：
  若 判据B2(Δ高频能量) 超限  →  丢弃重生成结果、回退到上一版
```

**★ 判据 B 必须拆成两半（原稿这里循环依赖）**：
`Δ高频能量 = 高频(重生成后) − 高频(重生成前)` **只有重生成之后才算得出来**，
所以它**不能**用来决定「要不要重生成」。原稿把它放在决策树的门口是自相矛盾的。
正确拆法：

- **B1 = 锚间高频一致性**（两侧锁区的 Laplacian/小波高频相关系数）—— **事前就能算**，
  用它决定「这个区域值不值得重生成」；
- **B2 = Δ高频能量** —— **事后否决**：重生成是允许失败的，超限就丢掉、
  保留上一版（代价只是那一次采样的时间，不会污染成片）。

θ 不再是 6~7 个拍的门，而是 2 个在标注集上扫出的 F1 最优点
（+1 个负载护栏保留）。这是"压成 1~2 个有物理意义的量"的具体答案。

### 4.2 可复现性：能修的与不能修的

**能修的**（同机同版本前提下）[cite:14][cite:15][cite:16]：

| 措施 | 效果 | 代价 |
|---|---|---|
| ComfyUI `--deterministic` | 官方旗标：尽可能用确定性 PyTorch 算法 [cite:16] | 官方声明"不保证所有情况" |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 固定 cuBLAS workspace 配置 [cite:14]。**本机实测：`--deterministic` 已自动设，无需手设** | 近零 |
| 注意力后端钉死 **MATH** | 文档明确前向确定性，且 upcast FP32 计算 [cite:14][cite:24]。**⚠️ 本机实测：没有对应旗标** —— 只能改 `comfy/ops.py:66-72` 的偏好列表，或改用 `--use-split-cross-attention` | 慢（量级待实测） |
| 关闭 `--fast`（fp16_accumulation/fp8_matrix_mult/cublas_ops/autotune） | 排除实验性数值路径 [cite:16] | 慢 |
| 关闭 stochastic rounding | 排除随机舍入 [cite:16] | 近零 |
| 固定 seed/权重文件/量化级别/workflow 节点顺序 | ComfyUI FAQ 要求项 [cite:16] | 无 |

cuBLAS 官方文档：**同一 GPU 架构 + 同一 SM 数 + 同一工具链 + 单流 =
逐位一致** [cite:15]——你们"同会话缓存"技巧本质是凑齐了这些条件；
显式固定后可升级为"同启动参数即复现"。

**不能修的**：

- FlashAttention 前向无逐位保证（确定性模式存在但吞吐 -37.9% [cite:17]）；
  SageAttention 为 int8/fp8 量化 kernel，无逐位承诺 [cite:18]——
  **二选一：复现性（MATH）或速度（Sage），A/B 实验期间用前者**；
- 跨 PyTorch/CUDA 版本、跨硬件、驱动更新：不保证 [cite:14][cite:15]；
- int8 量化推理（bitsandbytes/torchao 路径）的 kernel 选择差异 [cite:15]。

**无法复现的根因候选排序**（供你们逐项排查）：SageAttention 量化 kernel >
**模型补丁 / 动态加载**（每段重新挂 208 个补丁、aimdo 分片 staging —— **本机特有，
原稿没列**）> `--fast`/stochastic rounding > cuBLAS 多流算法选择 > cuDNN benchmark
> bf16 累积顺序（单流同版本下 cuBLAS 本身逐位一致 [cite:15]）。

### 4.3 本机实测补充（原稿没写，但决定阶段 0 成败）

在 RTX 4060 Laptop 上的 ComfyUI v0.34.0 实测：

| 项 | 实测结果 |
|---|---|
| `--deterministic` 做什么 | `torch.use_deterministic_algorithms(True, **warn_only=True**)`（`comfy/model_management.py:109`） **且自动设** `CUBLAS_WORKSPACE_CONFIG=:4096:8`（`main.py:109-111`） |
| 所以 `CUBLAS_WORKSPACE_CONFIG` 要手设吗 | **不必**，`--deterministic` 已经设了（原稿那条冗余） |
| `--fast` | 旗标存在，子项名与报告一致；**但本机 bat 里本来就没开** → 暂时空操作 |
| **「钉 SDPA-MATH」** | **✗ 没有这个旗标**。注意力只有 split / quad / pytorch / sage / flash / ck 六种。要钉 MATH 必须**改 `comfy/ops.py:66-72` 的后端偏好列表**，或改用 `--use-split-cross-attention`（纯 PyTorch 实现，确定但慢） |

**★★ 机制关键**：`warn_only=True` 意味着**不报错、只警告**；
而 SageAttention 是**自定义 CUDA 扩展**（`comfy/ldm/modules/attention.py:679` 直调 `sageattn`），
**不经过 PyTorch 算子注册** → **`--deterministic` 对它完全无效，连警告都不会有**。

所以报告 §4.2 那句「Sage/Flash 是速度与复现二选一」是对的，但**理由比它写的更强**：
不是"效果打折"，而是**这个旗标压根看不见 Sage**。**关 Sage 是前提，不是可选项。**

**阶段 0 的验证要这样设计**：同会话会被 ComfyUI 的节点缓存掩盖（一采命中缓存 → 两次跑的本质是同一份 latent），
所以必须**中间重启**。为省时间与 I/O，用**短片**验证即可（见 §6 阶段 0）。

---

## 5. 开源许可证一览（以仓库 LICENSE 原文核实）

| 项目 | 用途 | 许可证 | 风险/边界 |
|---|---|---|---|
| **ComfyUI-MAINodes** | jerk oracle 想法来源 | **GPL-3.0** [cite:19] | ★ **红线**：想法（算法思路）可借鉴，**代码一行都不能拷**进非 GPL 仓库；拷了整个 seamkit 需转 GPL |
| **RePaint**（Lugmayr et al., CVPR 2022, arXiv:2201.09865）<br>← **原稿漏了这一条** | 缝窗重去噪（E-3）的「逐步锚定」语义 | **CC BY-NC-SA 4.0**（**非商用**＋相同方式共享） | ★★ **最该列的红线**：与本仓库的 GPL-3.0 **不兼容**（GPL 允许商用，它不允许）。本仓库**未使用其任何代码** —— 依赖的是 ComfyUI 核心自带的 `KSamplerX0Inpaint` / `MiniMaxH3.scale_latent_inpaint`（GPL-3.0），只借用论文的方法思想（思想不受版权保护）。**永远不要移植 RePaint 代码进本仓库** |
| TransNetV2 | 标定对照真值 | **MIT** [cite:2] | 可用可改；权重条款需单独确认 [未验] |
| PERSIST | 判别式结构 + 潜在状态思路 | **Apache 2.0** [cite:4] | 可用可改可商用（留 NOTICE） |
| PySceneDetect | AdaptiveDetector 思路 | **BSD-3** [cite:5] | 可用 |
| ClipShots | 标定数据集 | **MIT** [cite:6] | 可用 |
| AutoShot/SHOT | 标定数据集 | **MIT** [cite:7] | 可用 |
| vidstab | 光流/RANSAC 相机运动判别 | **MIT** [cite:8] | 可用 |
| VBench | 时序指标定义 | **Apache 2.0** [cite:13] | 指标自己实现一行 MAE 亦可，无依赖必要 |
| FreeLong / FreeLong++ | 频域视角 | **无官方代码仓库** [未验] | **只用思想**（FFT 自写，一行级）；不存在拷代码的可能 |
| FreeSpec / FreqForcing / FlowLong | 相邻方法 | 代码许可证未验 [未验] | 只引论文结论 |
| MiniMax H3（官方仓库/权重） | 模型本体 | MiniMax H3 Community License [cite:20][cite:21] | 与本仓库无代码关系；H3-VisualVAE 公开规格为因果 16×空间/4×时间/24 通道 [cite:20]——**17帧=5token(1+4+4+4+4) 是本地实现层结构，非官方公开规格** |

原则：**无 LICENSE 文件的仓库 = 默认保留所有权利**，与 GPL 同样不可拷。

---

## 6. 落地路线（RTX 4060 8GB，按依赖排序）

**阶段 0（半天）——先把 A/B 变成逐位复现**

本机现状：`启动ComfyUI.bat` 跑的是 `--use-sage-attention --output-directory "D:\共享"`。
实验会话改成（**已在本机逐个核过旗标存在性，见 §4.3**）：

```bat
set COMMANDLINE_ARGS=--deterministic --use-split-cross-attention --output-directory "D:\共享"
rem ① 去掉 --use-sage-attention —— 这是前提不是可选项：
rem    --deterministic 是 warn_only，且 Sage 是自定义 CUDA 扩展，旗标看不见它
rem ② --deterministic 会自动设 CUBLAS_WORKSPACE_CONFIG，不用手设
rem ③ --fast 本来就没开，无需处理
rem ④ 若 --use-split-cross-attention 太慢：改回 --use-pytorch-cross-attention，
rem    并改 comfy/ops.py:66-72 把 SDPABackend.MATH 提到首位（没有旗标能做这件事）
```

**★ 验证设计（这一步最容易做错）**：同一会话内连跑会被 ComfyUI 的节点缓存掩盖 ——
一采命中缓存后，两次跑本质是同一份 latent，**永远测不出复现性**。
所以必须**两次跑之间重启 ComfyUI**。为省时间与 I/O，用短片验就够：

```
total_seconds = 5      target_segment_seconds ≈ 1.2      → 约 4 段，有 hunt 行可比
每跑十几分钟；I/O 从几百 GB 降到几十 GB
```

判据：两次的 `cut planned=… -> boundary=… measured=…` 几行**逐字相同 = 复现成功**
（一采一变 hunt 必变，所以这几行就是 latent 的廉价指纹）。
若仍不齐，按 §4.2 根因排序逐项排查 —— 第一嫌疑是 Sage 残留与**模型补丁路径**。

**阶段 1（1~2 天，CPU 为主）——标注集**
用 TransNetV2（MIT）在现有成片上生成参照转镜标签 + 人工复核手挥特写等
失败案例 → 存成逐帧标签表（正/负/拒判）。这份数据同时标定判据 A
与「残差-台阶」矩阵（请求书 §3.2.4 一并回答）。

**阶段 2（1 天）——两个判据的量 + ROC**
实现 `空间集中度`（token 差按空间 patch 展开的能量集中度）与
`Δ高频能量`（重生成前后 Laplacian/小波带能量差），在标注集上画
F1/ROC 曲线，扫出 θ_hardcut 与安全阈值。棘轮谱线（§2.4）加进
`find_calm_boundaries` 的过静门替代拍的 0.05。

**阶段 3（1 天）——策略表接线**
把 §4.1 的决策树接进现有 hunt/blend/redenoise 分派逻辑；
旧 6~7 个门退位为负载护栏。

**判定器形态**：不引入 PERSIST 的 SIREN/FiLM 网络也可先落地
（判别式公式 + 标定阈值已能跑）；若线性组合不够，再训一个
logistic 头（输入 3+2 个信号，标注集上百样本即可）——比 FiLM 轻几个量级，
CPU 推理。

---

## 7. 缺口声明（检索确认的空白）

1. **latent 域 SBD 标定无公开先例**——判据 A 的迁移是空白区研究，
   你们的标注集是首批数据；
2. **量化棘轮无公开文献**——§2.4 的频谱签名是推导提案，[未验]；
3. **重生成安全预测器无现成校准**（检索确认 [未验]）——判据 B 的
   锚间高频一致性假设需要 n≥20 的样本检验；
4. FreeLong 原始代码仓库不存在/不可验 [未验]，频域结论以论文数据为准；
5. TransNetV2 预训练权重的分发条款未验 [未验]，落地前需确认。

---

## Sources

1. TransNet V2 论文：https://arxiv.org/abs/2008.04838
2. TransNetV2 仓库（MIT）：https://github.com/soCzech/TransNetV2 ｜ PyTorch 推理实现：https://pypi.org/project/transnetv2-pytorch/
3. PERSIST 论文：https://arxiv.org/abs/2608.29287
4. PERSIST 仓库（Apache 2.0）：https://github.com/linty5/PERSIST
5. PySceneDetect 文档：https://www.scenedetect.com/docs/0.6.3/api/detectors.html
6. ClipShots 仓库（MIT）：https://github.com/Tangshitao/ClipShots
7. AutoShot 论文：https://arxiv.org/abs/2304.06116 ｜ 仓库（MIT）：https://github.com/wentaozhu/AutoShot
8. vidstab 仓库（MIT）：https://github.com/AdamSpannbauer/python_video_stab
9. FreeLong 论文：https://arxiv.org/abs/2407.19918 ｜ FreeLong++：https://arxiv.org/abs/2507.00162 ｜ 项目页：https://freelongvideo.github.io/
10. FreeSpec 论文：https://arxiv.org/abs/2605.06509
11. FreqForcing 论文：https://arxiv.org/abs/2607.27110
12. FlowLong 论文：https://arxiv.org/abs/2605.20910
13. VBench 论文：https://arxiv.org/abs/2311.17982 ｜ PyPI（Apache 2.0）：https://pypi.org/project/vbench/
14. PyTorch 可复现性文档：https://docs.pytorch.org/docs/main/notes/randomness.html ｜ 数值精度：https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html
15. NVIDIA cuBLAS 12.9 文档（同卡同工具链单流逐位一致）：https://docs.nvidia.com/cuda/archive/12.9.0/pdf/CUBLAS_Library.pdf
16. ComfyUI 启动旗标（--deterministic / --fast / attention 后端）：https://docs.comfy.org/development/comfyui-server/startup-flags ｜ FAQ：https://comfyanonymous.github.io/ComfyUI_examples/faq/
17. FlashAttention-3 确定性模式（吞吐 -37.9%）：https://arxiv.org/abs/2601.21824
18. SageAttention 仓库：https://github.com/thu-ml/SageAttention
19. ComfyUI-MAINodes 仓库（GPL-3.0）：https://github.com/matlowai/ComfyUI-MAINodes
20. MiniMax H3 官方仓库：https://github.com/MiniMax-AI/MiniMax-H3
21. MiniMax H3 模型卡：https://huggingface.co/MiniMaxAI/MiniMax-H3
22. Laplacian 方差等 sharpness 指标：https://docs.mbari.org/triton-video-sampling/algorithm/metrics/ ｜ 阈值示例（任务特定）：https://shiwanshu.in/projects/image-quality-checker
23. CamFlow（2D 相机运动估计）：https://arxiv.org/abs/2507.22480
24. PyTorch CUDA 语义：https://docs.pytorch.org/docs/2.2/notes/cuda.html
25. 相机切 vs 快速运动（光流对应性判别）：https://dev.to/multigrid/telling-a-camera-cut-from-fast-motion-in-raw-footage-3h8i
