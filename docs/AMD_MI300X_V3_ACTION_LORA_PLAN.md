# DeepSeek 上机执行任务书：魔搭 AMD MI300X (192GB) · MiniMax H3 `v3 Action-Sharp` 首日 5h 出片 + 每日接力训练单

> **本文档直接交付给运行在「魔搭（ModelScope）× AMD 开发者计划 192G 实例」上的执行 Agent（DeepSeek）**。  
> 本地所有 `v3` 训练脚本、ROCm 兼容补丁、一键装机脚本与 32 条零动态模糊武戏提示词**已在仓库 `train/` 目录全部就位并通过自动化单测**，上机无需手改算法代码，直接按本单顺序执行即可。

---

## 0. 任务总览与两段式/每日接力节奏

- **硬件环境**：AMD Instinct MI300X / MI308X（`192GB HBM3` 显存 / `200GB` 内存 / `ubuntu22.04-rocm7.2.3-py312-torch2.12.0` / 总额度 `100h`，单次容器限时约 `6–10h`）。
- **核心攻克目标**：在保留 `v2` 静止文戏零油光与本地 RTX 4060 8GB `bf16_lora=200/200` 旁路 100% 兼容的前提下，通过 **「同构真一采退化糊底 `real_pass1` + 运动门控解耦 `Motion-Gated Scheme 2A` + 空间梯度损失 $\nabla_{xy}$ + Attention 无损扩秩热启（`r=32 → attn 64 / mlp 32`）+ 32 条零动态模糊 Sakuga 教师」**，彻底根治大幅跳跃/旋身/甩发镜头的五官融化与色块涂抹。

| 批次 | 目标与产出 | 教师任务 | 训练任务 | 预计总耗时 |
|---|---|---|---|---:|
| **Day 1（首轮 5h 极速见效班）** | 当天先拿出一版可回 4060 实测 `00:07` 大动作的 `v3_step220` | 跑 `prompts_sakuga_action_32.jsonl` **前 20 条 `39f (1.6s)`**（`--limit 20`） | 从 `v2_final` 无损扩秩热启，混训 `v2+v3` 至 **`step 220`** | **`~4.3 – 4.8 小时`** |
| **Day 2（次日满血封版接力班）** | 补齐 3 秒长连招大视差记忆，推进至满血收敛 | 自动跳过前 20 条，跑完**后 12 条 `73f (3.0s)`**（去掉 `--limit`） | 自动从 `step 220` 断点续训（保留动量）至 **`step 420`** | **`~3.8 – 4.3 小时`** |
| **Day 3+（日常自由接力）** | 随时追加新动作 JSONL 或引入二期 MAGI-2 视频桥 | 仅跑新增未完成 ID，每 5 条自动推 Zstd 分片 | 每次把 `--train-steps` `+150~200` 一键续跑 | 每次 `2–4 小时` 按需 |

---

## 1. 给 DeepSeek 的 5 条硬红线（严禁违反！）

1. **🚨 防爆盘红线（`/mnt/workspace` 持久化盘 vs 临时盘分离）**：
   - `/mnt/workspace` 是跨日关机唯一保留目录，但配额仅约 `50–100 GB`！
   - **严禁把 `40GB` 主模型和 `32–64GB` 文本编码器下载到 `/mnt/workspace`！** 必须运行 [`train/setup_amd_mi300x.sh`](../train/setup_amd_mi300x.sh)，自动将 `~90GB` 只读大模型放到空间最大的临时盘（如 `/tmp/h3_rocm_models` → 软链 `/content/local_models`），`/mnt/workspace/h3_v3/work`（→ 软链 `/content/work`）只存代码、压缩后的 `teacher_cache`（`< 8GB`）和 LoRA 断点。
2. **🚨 ROCm 算子三不碰**：
   - **绝不重装 PyTorch**（预装 `torch 2.12 + rocm 7.2.3` 已深度适配 `gfx942`）；
   - **绝不用 `nvfp4_awq` 文本编码器**（`setup_amd_mi300x.sh` 会自动枚举并下载 `Comfy-Org/MiniMax-H3` 中的非 FP4 原生 `bf16/fp8/int8` 版 `Qwen3-VL-32B`，并将文件名写入 `/content/work/rocm_clip_name.txt`）；
   - **绝不用 `bitsandbytes` 或 `flex_attention` 编译模式**：本仓库脚本已内置 ROCm 自动识别（`_is_rocm() == True` 时自动选原生 `torch.optim.AdamW` + `--attn-backend sdpa_mask`，走 `gfx942` 原生 AOTriton/CK Flash SDPA，彻底规避 `64KB LDS` 共享内存报错）。
3. **🚨 严守 `200` 层 `0-adaln` 纯线性契约**：
   - 仅训 `blocks.{0..49}.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}`（`attn r=64, mlp r=32`，导出恰好 `600` 个张量），绝不碰 `adaln_proj`。
4. **🚨 严禁手动中断未上传的分片或 Checkpoint**：
   - 停训练前先 `tail -n 20 /content/work/lora_run_v3/logs/hf_sync.log` 确认最新 `step_XXXXXX.safetensors` 与 `latest_state.pt` 已同步至 HuggingFace 私有仓。
5. **🚨 跑完必须提醒或停止实例防空扣 100h 额度**：
   - 魔搭免费实例只要处于运行态就会持续扣减 100h 总额度；当日任务与上传全部收尾后，第一时间通知主人点击魔搭控制台 **「停止」** 按钮！

---

## 2. 上机前主人本地 1 分钟准备清单（仅首次）

在启动魔搭 AMD 实例前，确保私有 HuggingFace 仓库 `coolwindtang/minimax-h3-teacher-cache` 中存有以下文件（`setup_amd_mi300x.sh` 开机即自动拉取，省去网页手动传图）：
1. 两张角色设定参考图上传至私有仓 `refs/` 目录：
   - `refs/deepseek.png`（本地 `D:\comfyui\ComfyUI\input\deepseek.png`）
   - `refs/claude_sheet.png`（本地 `D:\comfyui\ComfyUI\input\claude_sheet.png`，已从 `克劳德酱三视图与细节设定稿.png` 创建纯英文别名防 Linux 编码乱码）
2. 确认私有仓已有热启动母版：`final/minimax_h3_sol_sharp_v2_final.safetensors`（上次 A100 Step 4 已上传）。
3. 将最新 `comfyui-h3-seamkit` 代码 `git push`（或打成压缩包传上实例 `/mnt/workspace/h3_v3/`）。

---

## 3. Day 1 执行全流程（首轮 ~4.5–5 小时极速出片单）

### Step 1.1：写入 Token 并一键装机拉模（约 `25–35 分钟`）
在魔搭 AMD 192G 实例终端（Bash）执行：

```bash
mkdir -p /mnt/workspace/h3_v3/work
# 填入你的 HuggingFace 读写 Token（切勿明文提交进公开 Git）
echo "hf_你的真实Token" > /mnt/workspace/h3_v3/work/hftoken.txt
chmod 600 /mnt/workspace/h3_v3/work/hftoken.txt

# 拉取最新 comfyui-h3-seamkit 代码到持久化盘
if [ ! -d /mnt/workspace/h3_v3/comfyui-h3-seamkit/.git ]; then
  git clone https://github.com/yanghuagai55/comfyui-h3-seamkit.git /mnt/workspace/h3_v3/comfyui-h3-seamkit
else
  git -C /mnt/workspace/h3_v3/comfyui-h3-seamkit pull
fi

# 一键完成：ROCm 核验 + 最大临时盘分配 + ComfyUI/T8 部署 + 自动选非FP4编码器与BF16主模下载 + 参考图就位
bash /mnt/workspace/h3_v3/comfyui-h3-seamkit/train/setup_amd_mi300x.sh
```

**Step 1.1 验收门禁**（必须全绿才往下走）：
- 终端最后输出 `PREFLIGHT OK (teacher, backend=rocm)`；
- `ls -lh /content/ComfyUI/input/deepseek.png /content/ComfyUI/input/claude_sheet.png` 两个文件均存在且 `> 1 MB`（若未放私有仓 `refs/`，直接通过魔搭左侧文件树把这两张图拖进 `/mnt/workspace/h3_v3/ComfyUI/input/`）。

### Step 1.2：跑 10 步快速烟雾测试验证 ROCm 反向全绿（约 `5–8 分钟`，仅首日跑一次）
在正式花 2 小时跑新教师前，**先从云端拉 1 片现成 `v2` 分片跑 6 步训练烟雾自检**，100% 确认 MI300X `sdpa_mask` + 无损扩秩 `64/32` 反向传播零 bug：

```bash
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
cd /content/ComfyUI/custom_nodes/comfyui-h3-seamkit

# 拉第 0 号旧分片做 6 步极速验证
python3 - <<'PY'
from hf_shards import ShardStore
s = ShardStore("coolwindtang/minimax-h3-teacher-cache", "teacher_cache_v1", "/content/work/hftoken.txt")
m = s.fetch_manifest()
assert m["shards"], "teacher_cache_v1 无分片"
s.pull_all("/tmp/smoke_cache", "/tmp/smoke_dl")
PY

python3 train/train_h3_lora.py \
  --comfy /content/ComfyUI \
  --base /content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
  --cache /tmp/smoke_cache \
  --out /tmp/smoke_out \
  --frames 22 --steps 6 \
  --rank-attn 64 --rank-mlp 32 \
  --init-lora-hf final/minimax_h3_sol_sharp_v2_final.safetensors \
  --hf-repo coolwindtang/minimax-h3-teacher-cache \
  --token-file /content/work/hftoken.txt \
  --attn-backend sdpa_mask --block-size 64 \
  --lf-base-anchor --lf-kernel 5 --motion-gate-tau 0.35 --grad-xy-weight 0.15 \
  --optim adamw --lr 8e-5 --grad-clamp 10 --log-every 2 --save-every 6

rm -rf /tmp/smoke_dl /tmp/smoke_out
```

**烟雾门禁核对（4 项全中即刻进正赛）**：
1. `[init] warm-start ... exact=200 rank_expanded=200 (step=0)`（200 个 MLP 参数精确继承 + 200 个 Attention 参数无损扩至 `r=64`）；
2. `[sanity] sigma=0.5 cos > 0.75`；
3. `[grad-check] step 1` 与 `step 2` 均为 **`400/400`** 参数非零有限梯度；
4. 6 步跑完 `0 NaN`，日志打出 `loss_lf`, `loss_hf`, `loss_grad_xy`, `mot_gate`！

### Step 1.3：一键启动首日「前 20 条 `39f` 零残影教师 + 220 步 `v3` 主训练」（约 `3.8 小时` 全自动串联）
```bash
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
cd /content/ComfyUI/custom_nodes/comfyui-h3-seamkit
CLIP_NAME="$(cat /content/work/rocm_clip_name.txt)"

nohup python3 train/boot.py \
  --mode auto \
  --repo coolwindtang/minimax-h3-teacher-cache \
  --prefix teacher_cache_v3_action \
  --train-shard-prefixes teacher_cache_v1,teacher_cache_v3_action \
  --token-file /content/work/hftoken.txt \
  --prompts train/prompts_sakuga_action_32.jsonl \
  --limit 20 \
  --clip "${CLIP_NAME}" \
  --teacher-steps 28 --teacher-cfg 5.5 \
  --zup-mode hybrid --min-motion-lapvar 75.0 \
  --shard-size 5 \
  --teacher-budget-min 150 \
  --then-train --train-min 40 --train-steps 220 \
  --frames 22,39 \
  --rank-attn 64 --rank-mlp 32 \
  --lr 8e-5 --warmup 20 \
  --lf-base-anchor --lf-kernel 5 --motion-gate-tau 0.35 --grad-xy-weight 0.15 --action-boost 2.5 \
  --init-lora-hf final/minimax_h3_sol_sharp_v2_final.safetensors \
  --hf-prefix lora_run_v3 \
  > /content/work/logs/day1_auto.log 2>&1 &

echo "PID=$!"
```

**巡检命令（每 20–30 分钟看一眼即可，零人工干预）**：
```bash
tail -n 25 /content/work/logs/day1_auto.log
tail -n 10 /content/work/lora_run_v3/logs/metrics.jsonl 2>/dev/null || true
```

### Step 1.4：首日云端动态帧 A/B 验收 + 切出 `action_stylekeep` 回传本地（约 `15 分钟`）
待 `day1_auto.log` 打印 `[done]` 后：

```bash
cd /content/ComfyUI/custom_nodes/comfyui-h3-seamkit

# 1. 在 39f 高动态样本上跑 5 栏对比图（聚焦峰值运动帧 1:1 裁切）+ 三联 MP4 并上传 HF
python3 train/eval_lora_a100.py \
  --comfy /content/ComfyUI \
  --base /content/local_models/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
  --vae /content/local_models/vae/minimax_h3_sol_vae_fp16.safetensors \
  --cache /content/work/teacher_cache \
  --lora /content/work/lora_run_v3/steps/step_000220.safetensors \
  --eval-frames 39 --samples 2 \
  --out /content/work/eval_v3_day1 \
  --hf-repo coolwindtang/minimax-h3-teacher-cache \
  --hf-eval-prefix eval_v3_day1 --upload-hf

# 2. 一键切出二次元武戏专用版与真人版并上传 HF 供本地 4060 直接下载实测！
python3 train/slice_lora_stylekeep.py \
  --input /content/work/lora_run_v3/steps/step_000220.safetensors \
  --output /content/work/lora_run_v3/minimax_h3_sol_sharp_v3_d1_action_stylekeep.safetensors \
  --preset action_stylekeep

python3 - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
tok = Path("/content/work/hftoken.txt").read_text().strip()
api = HfApi(token=tok)
for f in (
    "/content/work/lora_run_v3/steps/step_000220.safetensors",
    "/content/work/lora_run_v3/minimax_h3_sol_sharp_v3_d1_action_stylekeep.safetensors",
):
    p = Path(f)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"candidates_v3/{p.name}",
                    repo_id="coolwindtang/minimax-h3-teacher-cache", repo_type="model")
    print("[UPLOADED]", p.name)
PY
```
> **🛑 首日收工必做**：确认打印 `[UPLOADED]` 后，**立即在魔搭网页端点击「停止实例」**！把 `minimax_h3_sol_sharp_v3_d1_action_stylekeep.safetensors` 拉回本地 `D:\comfyui\ComfyUI\models\loras\` 实测 `D2_combat_chapel` `00:07`。

---

## 4. Day 2 及后续每日接力标准指令（每次仅需 2 条命令！）

次日（或任意一天）重新开启魔搭 AMD 192G 实例后，`/mnt/workspace/h3_v3/` 下的代码、`ComfyUI`、`teacher_cache` 和 `lora_run_v3/resume/latest_state.pt` 全部完好保留！只需跑以下两步：

```bash
# 1. 自动恢复软链并补拉被清空的临时盘只读大模型（内网约 10~15 分钟）
bash /mnt/workspace/h3_v3/comfyui-h3-seamkit/train/setup_amd_mi300x.sh

# 2. 去掉 --limit：自动跳过已完成的 20 条，生成后 12 条 73f (3.0s) 长连招教师，
#    并自动从 step 220 断点无缝续训至 step 420（保留优化器动量、平滑延展余弦学习率！）
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
cd /content/ComfyUI/custom_nodes/comfyui-h3-seamkit
CLIP_NAME="$(cat /content/work/rocm_clip_name.txt)"

nohup python3 train/boot.py \
  --mode auto \
  --repo coolwindtang/minimax-h3-teacher-cache \
  --prefix teacher_cache_v3_action \
  --train-shard-prefixes teacher_cache_v1,teacher_cache_v3_action \
  --token-file /content/work/hftoken.txt \
  --prompts train/prompts_sakuga_action_32.jsonl \
  --clip "${CLIP_NAME}" \
  --teacher-steps 28 --teacher-cfg 5.5 \
  --zup-mode hybrid --min-motion-lapvar 75.0 \
  --shard-size 4 \
  --teacher-budget-min 140 \
  --then-train --train-min 40 --train-steps 420 \
  --frames 22,39,73 \
  --rank-attn 64 --rank-mlp 32 \
  --lr 6e-5 --warmup 20 \
  --lf-base-anchor --lf-kernel 5 --motion-gate-tau 0.35 --grad-xy-weight 0.15 --action-boost 2.5 \
  --hf-prefix lora_run_v3 \
  > /content/work/logs/day2_relay.log 2>&1 &
```
*(若后续 Day 3 还要继续加练，只需把 `--prompts` 指向新增提示词文件或设 `--mode train`，并把 `--train-steps` 改为 `600` 即可无限接力！)*

---

## 5. DeepSeek 现场异常排障决策矩阵（ROCm 7.2 专版）

| # | 现场报错 / 现象 | 根因定位 | 标准自愈处置动作（最多重试 2 次） |
|---|---|---|---|
| **1** | `No space left on device` | 误把大模型或原始未压缩 `.z_native.pt` 写进了 `/mnt/workspace` | `df -h` 查满盘分区；确认 `/content/local_models` 指向大空间临时盘（如 `/tmp/h3_rocm_models`）且 `--out` 指向 `/tmp/h3_native_raw`，清理 `rm -rf ~/.cache/huggingface/hub` 后续跑。 |
| **2** | 教师阶段 `CLIPLoader` 报 `nvfp4` / `comfy_kitchen` 缺失 | 误传了默认 `nvfp4_awq` 编码器名 | 读取 `cat /content/work/rocm_clip_name.txt` 传入 `--clip`，确保加载的是非 FP4 的 `bf16 / fp8 / int8` 版 `Qwen3-VL-32B`。 |
| **3** | `73f` 桶训练报 HIP OOM（极罕见，192GB 下理论峰值 `<145GB`） | `73f` 序列长 `S≈46k` 且多进程残留占显存 | ① 先 `pkill -f 'ComfyUI/main.py'` 确认 ComfyUI 已彻底退出（腾出 `60GB+`）；② 若仍超限，只训 `--frames 22,39` 先拿稳产物。 |
| **4** | 单次实例即将到达平台强制回收时限（如运行满 5.5h） | 魔搭 PAI-DSW 单次容器限时 | 提前调小 `--teacher-budget-min`；若已在训练中，等下一个 `50` 步整数倍 `[ckpt]` 上传完成（核实 `hf_sync.log`）后优雅 `pkill -TERM -f train_h3_lora.py`，关机重开实例直接无缝续跑！ |
| **5** | `grad-check` 失败或 `cos < 0.7` | 基座路径或 `ComfyUI` 版本不匹配 | 核对 `--base` 必须是 `minimax_h3_ref2va_pruned_bf16.safetensors`（严禁用 `fl2va` 或社区改过 `adaln` 的魔改量化底模当教师/训练底座）。 |
