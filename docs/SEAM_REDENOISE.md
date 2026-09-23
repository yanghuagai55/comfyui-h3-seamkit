# 缝窗重去噪（Seam Re-Denoise）与多 token 锚定：原理与实现记录

> 用途：供**外部审计**（程序合理性核查）。每条机制都给出**代码出处**与**实测数据**，
> 可逐条核对。凡未经实测的推断，均明确标注「推断」。
> 仓库：`comfyui-h3-seamkit` ｜ 分支：`feat/adaptive-blend` ｜ 提交：`e9ee3c0`
> 配套调研：`D:\comfyui\_hardcut_work\SAMPLER_RESEARCH_20260923.md`（论文线）
> 配套单测：`D:\comfyui\_hardcut_work\seamfix\e1_temporal_mask_test.py`（真实 comfy 链路）、
> `e3_redenoise_unit_test.py`（几何与接线）

---

## 0. 一句话

分块采样产生曝光台阶的根因是**每块收敛到自己的局部最优**，拼接（crossfade/冻结/挪边界）只能藏台阶、不能消台阶；本方案把过渡**交给模型重新生成**——缝两侧锁死为已发布 latent（每个采样步以键帧语义注入），中段在双侧上下文里重去噪。

---

## 1. 为什么拼接路线到头了

### 1.1 已排除的假设（源码核对）

- **初始噪声差异** —— 否。上游 `_build_global_target_av_noise` 生成**单一全局噪声张量**，
  每块按 token 坐标精确切片（`GLOBAL_NOISE_POLICY = "one_full_target_video_noise_then_exact_coordinate_slices"`，
  `chunked_two_pass_upscale_advanced.py:45`）。噪声共享已完整实现，不是台阶来源。

- **时间轴坐标换算** —— 否（见 `ADAPTIVE_BLEND.md` §1.2，累积和与线性等价，偏差 0.0）。

### 1.2 根因（推断，有旁证）

每块在**自己的条件上下文**里独立去噪，收敛点由局部动力学决定——即使起始噪声相同，
两条轨迹在有限步内各自收敛到**不同的局部最优**（曝光/色彩漂移随块累积）。
旁证：同一全局噪声下，段首单帧跳变仍达 **35.84 灰阶**（缝 187，实测，基线 tag
`v1.0.0-baseline-20260921`）。

### 1.3 既有工具都在"拼"的层面

| 工具 | 做什么 | 层面 |
|---|---|---|
| `auto_seam_hunt` | 把边界挪到内容最平缓处 | 拼前 |
| guarded overlap | 冻结前块尾部，新块从锚后接管 | 拼时 |
| `seam_blend` | overlap 区 crossfade | 拼后 |

它们的共同上限：**两块各自的收敛结果已经定型**，任何拼接都在两个"既成事实"之间取舍。
本方案补上最后一层：**拼后把过渡区重新交给模型**，让它在双侧约束下生成一个一致的解。

---

## 2. 原理：comfy 原生的 RePaint 逐步锚定

### 2.1 `KSamplerX0Inpaint`（comfy 核心，未做任何修改）

出处：`comfy/samplers.py:634-644`。每个采样步做两件事：

```python
# 输入侧：锁区（mask=0）的 x 替换为加噪锚
x = x * denoise_mask + scale_latent_inpaint(x, sigma, noise,
                                            latent_image, denoise_mask) * (1 - denoise_mask)
# 模型前向（自由区照常，锁区被上一行污染为锚）
out = self.inner_model(x, sigma, ...)
# 输出侧：锁区的去噪预测钉死为干净锚
out = out * denoise_mask + self.latent_image * (1 - denoise_mask)
```

这正是 RePaint 论文（arXiv:2201.09865）的逐步条件化：**已知区域在每一步被重置**，
模型每一步都在"已知边界条件"下解自由区域，而不是只在 t=0 注入一次。

### 2.2 数学：锁区轨迹精确落在锚上（实测）

Euler 更新 `x' = x + (x - denoised)/σ · (σ' - σ)`。锁区里 `denoised ≡ anchor`，
归纳可得锁区轨迹恒为 `anchor + σ · noise`；末步 σ'→0 时
`x' = x - (x - anchor) = anchor`——**精确相等，与模型输出无关**。

E-1 实测（CPU、真实 `sample_euler` + 真实 `MiniMaxH3.scale_latent_inpaint`、模型本体 mock）：

| 验证 | 结果 |
|---|---|
| 模型每步收到的锁区输入 | `== 0.999·anchor + 0.001·noise` ✓ |
| Euler 轨迹锁区 | `== anchor + σ·noise`（键帧同款噪声轨迹）✓ |
| 最终输出锁区 | `== anchor`，**err 2.4e-7** ✓ |
| `audio_scale=2.0` AV 双时钟分支 | 锁区仍精确收敛（err 3.0e-7）✓ |

### 2.3 H3 特有：锁区注入语义 = 键帧条件

出处：`comfy/model_base.py:2251-2275`（`MiniMaxH3.scale_latent_inpaint`）。

```python
cleans[0] = VISUAL_COND_TIMESTEP * cleans[0] + (1 - VISUAL_COND_TIMESTEP) * noises[0]
```

`VISUAL_COND_TIMESTEP = 0.999`（`comfy/ldm/minimax/model.py:25`）。锁区的"加噪锚"
**不随 σ 变化**，恒为 `0.999·anchor + 0.001·noise`——与 `minimax_keyframes` 键帧的
噪声增广是**同一语义**（条件时间步）。即：**锁区在模型眼里始终是"几乎干净的键帧"，
且是每一步都注入的连续键帧**——比一次性的 `anchor_conditioning` 键帧更强。

子 token 边界：`_pool_masks_to_token_grid`（`model_base.py:2218`）只做**空间** 2x2 patch
amax 池化，时间维完整保留；部分覆盖的 patch 由 `x_blend_weight` 线性混合。

### 2.4 时间维掩码的数据通路（E-1 逐段验证）

```
(1,1,T,H,W) 时间维掩码
  → guider.sample 逐流 prepare_mask（samplers.py:1297-1314）
  → comfy.utils.reshape_mask：trilinear，T 相等时零插值（E-1 T1 实测逐元素相等）
  → pack_latents：与 packed AV latent 逐元素对齐（E-1 T2 实测）
  → KSamplerX0Inpaint 逐步应用（E-1 T4 实测）
```

mask 支持 0~1 连续值（线性混合），时间维渐变（overlap 从 0 升到 1）是合法输入——
这是路线③（生成期 latent crossfade）的直接表达，尚未启用（推断：E-3 通过后可试）。

---

## 3. 实现：路线① 缝窗重去噪（E-3）

### 3.1 流程（`h3_upscale.py:510` `_redenoise_seam_windows`）

主循环（所有分块采样、拼接完成）之后、输出构造之前：

```
已发布时间线（accumulated latent）
        [....前块内容....][缝][....新块内容....]
                        ↓ 取窗（默认 10 token ≈ 34 帧，缝居中）
        [ lock 3 tok ][  free 4 tok  ][ lock 3 tok ]
        锁(mask=0)      重去噪(mask=1)     锁(mask=0)
        每步注入 0.999 键 │ 模型自由生成   每步注入 0.999 键
        帧语义(§2.3)      │ (双侧上下文)   帧语义(§2.3)
                        ↓ 回写中段（锁区按构造不变）
```

窗内 piece 构造与主循环 `_sample_fullframe`（`h3_upscale.py:460`）完全同构：
全局噪声切片（`global_video_noise[:, :, w0:w1]`）、conditioning 重锚到窗帧区间
（`reanchor_conditioning`）、双时钟采样器重绑；唯一区别是 `noise_mask`
从"空间全 1"变为"时间维锁两端"。**上游零改动**。

### 3.2 缝的位置（实测核对）

- **guarded overlap 路径**：缝 = `start_token + _locked_tokens`（冻结段结束处，
  即已发布内容与新块新鲜采样的交界；`h3_upscale.py:1462`）。
- **crossfade 路径**：缝记为 overlap 中点（近似；blend 本身已把台阶摊开）。

实测映射：缝 187（帧）= token 55，窗 `[50,60)` token = 帧 `[170,204)`。

### 3.3 参数（节点输入，默认关闭）

| 输入 | 默认 | 含义 |
|---|---|---|
| `seam_redenoise` | `false` | 总开关；false 时行为与旧版**逐位一致** |
| `seam_redenoise_frames` | `""` | 只处理指定帧附近的缝（如 `"187"`）；空 = 全部内缝；按 17 帧 token 网格吸附 |
| `seam_window_tokens` | `10` | 窗宽（token，1 token ≈ 3.4 帧），缝居中，边界钳制 |
| `seam_lock_tokens` | `3` | 两端各锁多少 token 为已发布 latent |

report JSON 新增 `seam_redenoise` 段：每缝的 window/lock 记录与 skip 原因。

---

## 4. 实现：路线② 多 token 锚定（E-2）

### 4.1 机制（`h3_upscale.py:621` `_anchor_conditioning_multi`）

上游 `anchor_conditioning`（`chunked_two_pass_upscale_advanced.py:277`）把上一窗
**恰好 1 个 token** 作为本窗第 0 帧的键帧条件——StreamingT2V（arXiv:2403.08312）
点名批评的单帧条件正是块间不一致来源。keyframe latent 格式**原生支持多 token**：
上游 `_trim_keyframe`（`chunked_two_pass_upscale_advanced.py:204-236`）逐 token 按
`FRAME_PER_TOKEN` 计算帧跨度、保留落在窗内的连续段。因此扩锚只是**切片宽度**变化：

```python
anchor = {"resolved_frame_index": 0,
          "latent": previous_video[:, :, token : token + k].contiguous()}
```

### 4.2 参数

| 输入 | 默认 | 含义 |
|---|---|---|
| `anchor_tokens` | `1` | 锚的 token 宽度；1 = 上游原版；2/5 = 扩锚（需 overlap > 0，硬切无锚可锚） |

`k` 超出前块剩余 token 时钳制（实测：前块剩 2 token、k=5 → 锚宽 2）。

---

## 5. 单元测试（全过，CPU，不占 GPU 成片）

`seamfix/e3_redenoise_unit_test.py`（真实上游 token 数学 + mock `sample_piece`）：

| # | 验证 |
|---|---|
| G1 | 窗居中且边界钳制（片尾左移） |
| G2 | 掩码形状 `(1,1,span,H,W)`，两端 3 token 为 0、中段为 1，audio mask 全 1 |
| G3 | 锁区逐 token 精确回写，中段替换，窗外原样 |
| G4 | `only_frames=[187]` 只命中 token 55 的缝 |
| G5 | prepared_noise = 全局噪声切片（视频 `[w0:w1]`、音频 `[a0:a1]`），conditioning 重锚到窗帧区间 |
| G6 | 多缝顺序处理、互不覆盖 |
| G7 | 窗过小（span ≤ 2×lock）安全跳过 |
| E2 | 多 token 锚：k=5 切 `[55:60)`、替换 index-0 键帧、非零索引键帧保留、尾端钳制 |

`seamfix/e1_temporal_mask_test.py`（真实 comfy 链路）见 §2.2、§2.4。

---

## 6. 实验协议（待 GPU 验证）

| 实验 | 参数 | 验收口径 |
|---|---|---|
| E-3 | `seam_redenoise=true` + `seam_redenoise_frames="187"` | 缝 187 单帧跳变 **35.84 → 目标 < 5** |
| E-2 | `anchor_tokens` ∈ {1,2,5}，固定种子 | 段首曝光差随 k 的收益曲线 |
| 叠加 | E-2 + E-3 | 生成期锚多强 × 缝期补多深的配比 |

测量：`seamfix` 的 luma / `find_calm_runs` 工具（像素域单帧跳变）。

---

## 7. 设计决策与已知边界

1. **纯 latent 锁定，不加显式 keyframe 双锚**——E-1 发现锁区注入即键帧语义（§2.3），
   且是每步连续注入，强于一次性键帧。最小实现先不带 keyframe，A/B 干净。
2. **全局噪声切片而非新噪声**——与主管线确定性哲学一致（噪声共享已完整，见 §1.1），
   且缝窗内容是"已有内容的重生成"，新噪声会引入不必要方差（推断）。
3. **sigma 复用主循环的表**——二采本身已是低 sigma 表（`PLAN_SCHEMA_LOW_SIGMA_V3`），
   缝窗重去噪的力度与分块采样同级，锁端约束其不越界。
4. **audio mask 全 1、只取视频输出**——与 `_sample_fullframe` 一致：audio 联合采样
   但最终输出仍用原始 audio latent。
5. **crossfade 路径的缝标记是 overlap 中点**（近似）；缝窗重去噪主要服务 guarded 路径。
6. **VRAM**：窗 10 token ≈ 主循环最长段（~45 token）的 1/4，单窗采样在已验证预算内
   （推断：实测待 E-3 GPU 跑确认）。

## 8. 与路线③（生成期版）的关系

同一机制（时间维 denoise_mask）的**部署位置不同**：E-3 是"拼后补"（输出期），
路线③是"拼时锁"（每块采样时把 overlap 区 mask 设 0/渐变，锚 = 前块已发布输出）。
E-3 验证模型补过渡的能力后，生成期版只是把同一开关前移（约 20 行，`_sample_fullframe`
构造 `video_mask` 处），不需要任何上游/采样器改动。

## 9. 参考

- RePaint: Inpainting using Denoising Diffusion Probabilistic Models（arXiv:2201.09865）
- StreamingT2V（arXiv:2403.08312）——单帧条件导致块间不一致
- Towards Chunk-Wise Generation for Long Videos（arXiv:2411.18668）
- 调研全文：`D:\comfyui\_hardcut_work\SAMPLER_RESEARCH_20260923.md`
