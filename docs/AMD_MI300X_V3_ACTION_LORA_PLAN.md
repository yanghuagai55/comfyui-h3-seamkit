# 第二轮 7 小时接力执行书：魔搭 AMD MI300X (192GB) · MiniMax H3 `v3 Action-Sharp` Session 2

> **首夜（Night 1）验收结果（全部绿灯通过 ✅）**：
> - **教师落盘**：10/10 条精选零模糊 `39f` 作画样本（`0001,0002,0004,0008,0009,0010,0012,0013,0015,0018`）全部在 185 分钟内完成，公共盘 `/mnt/data/h3_v3/work/teacher_cache` 现已有 **`82` 条 `.pt`**（72 条 `v1` + 10 条 `v3`）。
> - **训练与权重审计**：训练稳跑至 `step 67`（`112.5s/步`，显存恒定 `72.2 GB`，`0 NaN`），`step_000050.safetensors` 与 `minimax_h3_sol_sharp_v3_night1_action_stylekeep.safetensors` 均严格通过 **600 张量 / 0 `adaln_proj` / 异构秩 `64/32` / 新增 `[32:64]` 扩秩 `100/100` 全活（`||B[:,32:]||_F=0.372`）** 校验。
> - **本轮（Session 2 · 下一个 7 小时）目标**：补齐剩余核心 **8 条 `39f` 教师**（凑齐全部 14/14 R2VA 角色动作 + 4 条 T2VA，总库 `82 → 90` 条，并用 Python `zstandard` 修复首夜教师包备份上传）→ 从 `resume/latest_state.pt`（`step=50`）**无缝继承权重与 AdamW 动量续训至 `step 180~190`**（优化 `--save-every 20`、校准 `--motion-gate-tau 0.75`）→ 导出 `s120` 中段与最新终版双档切片回传魔搭私有仓 `coolwind123/h3-teacher-cache`。

---

## 1. 首夜实测复盘与第二轮 3 项精准调优

1. **校准 `--motion-gate-tau`：`0.35 → 0.75`**
   - 首夜 `metrics_night1.jsonl` 实测发现：因 `Sol-VAE` 时间步长为 6 帧，`tau=0.35` 令 `sakuga39_*` 的 `mot_gate` 均值高达 `0.95~0.995`（连慢动背景也放开了低频约束，`loss_lf` 升至 `1.1~2.5`）。调至 **`0.75`** 后，静止/慢速背景恢复 `70%+` 基座低频锁定（杜绝油光/画风漂移），而高速挥尾/甩发区域仍保持 `90%~100%` 满血轮廓重塑。
2. **加密保存间隔：`--save-every 25 → --save-every 20`**
   - 首夜 `112.5s/步` 跑到第 67 步触发 `160m` 限时，因未满 75 步回退取了 `step_000050`。改为 **`--save-every 20`** 后，从 `step 50` 续训会依次落盘 **`60, 80, 100, 120, 140, 160, 180, 190`**（每 ~37 分钟一存，尾部损失最多 `<15` 分钟）。
3. **修复教师分片云端备份打包（`tar --zstd` → Python `zstandard + tarfile`）**
   - 首夜容器无 `/usr/bin/zstd` 二进制导致 `tar --zstd` 跳过上传；`train/run_day2_7h.sh` 已改用容器已装的 Python `zstandard` 模块直接流式压缩并把全部 18 条 `sakuga*.pt` 备份至 `teacher_cache_v3_action/sakuga_v3_all.tar.zst`。

---

## 2. 第二轮 7 小时上机步骤（开机仅需 ~3 分钟装轻量 pip 包即可开跑）

### Step 0 · 启动 AMD 实例并恢复软链与系统盘轻量依赖（约 3 分钟）

由于 `/mnt/data/h3_v3/{models,ComfyUI,work,comfyui-h3-seamkit}` 均已在跨实例持久盘就绪，新开 AMD 实例**只需重装随系统盘重置的轻量 pip 依赖与 `/content` 软链**（严禁碰预装 ROCm PyTorch）：

```bash
export PERSIST_ROOT=/mnt/data/h3_v3
export WORK_DIR=/mnt/data/h3_v3/work
export MODEL_ROOT=/mnt/data/h3_v3/models
export COMFY_DIR=/mnt/data/h3_v3/ComfyUI
export CODE_DIR=/mnt/data/h3_v3/comfyui-h3-seamkit

# 1) 核对首夜资产与断点
ls -lh "${RUN_DIR:-/mnt/data/h3_v3/work/lora_run_v3}/resume/latest_state.pt" \
       "${RUN_DIR:-/mnt/data/h3_v3/work/lora_run_v3}/steps/"
echo "teacher_cache total=$(ls /mnt/data/h3_v3/work/teacher_cache/*.pt | wc -l) sakuga=$(ls /mnt/data/h3_v3/work/teacher_cache/sakuga*.pt | wc -l)"

# 2) 恢复轻量 pip 包（过滤 torch 三件套）与 /content 软链
pip install -q safetensors einops modelscope zstandard pillow tqdm
for req in "${COMFY_DIR}/requirements.txt" "${COMFY_DIR}/custom_nodes/comfyui-minimax-h3-audio-T8/requirements.txt"; do
  [ -f "${req}" ] || continue
  grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "${req}" > /tmp/req_filtered.txt || true
  pip install -q -r /tmp/req_filtered.txt || true
done

mkdir -p /content
ln -sfn "${COMFY_DIR}" /content/ComfyUI
ln -sfn "${WORK_DIR}" /content/work
ln -sfn "${MODEL_ROOT}" /content/local_models
```

---

### Step 1 · 同步最新 `run_day2_7h.sh` 与安全兜底脚本到云端并核验全层 checkpoint

1. 确认云端 `/mnt/data/h3_v3/comfyui-h3-seamkit/train/train_h3_lora.py` 保持全 50 层 `if use_ckpt:`（跑一次 `python3 /mnt/data/h3_v3/comfyui-h3-seamkit/train/revert_full_ckpt.py` 确保 `[SKIP] 已是全量 checkpoint` 或 `[OK]`），并将 `dense_rope_patch.py` / `run_teacher.py` 同步至 `/content/ComfyUI/custom_nodes/comfyui-h3-seamkit/train/`。
2. 将本地仓库最新的 [run_day2_7h.sh](file:///d:/comfyui/ComfyUI/custom_nodes/comfyui-h3-seamkit/train/run_day2_7h.sh) 写入 `/mnt/data/h3_v3/work/run_day2_7h.sh`（同时 `chmod +x` 并通过 `bash -n` 校验）。

---

### Step 2 · 后台启动第二轮 7 小时接力总控（预计总耗时 ~6.4 小时）

- **时间分配**：教师 8 条 `39f` 约 `140~150` 分钟（硬上限 `150m`）+ 训练 `step 50 → 180~190` 约 `235` 分钟（硬上限 `240m`）+ 双档切片与回传 `3` 分钟 = **总计约 `390` 分钟（`6.5` 小时），预留充足余量在 `7.5h` 回收前优雅收官**！

```bash
chmod +x /mnt/data/h3_v3/work/run_day2_7h.sh
setsid nohup bash /mnt/data/h3_v3/work/run_day2_7h.sh \
  > /mnt/data/h3_v3/logs/day2_7h.log 2>&1 < /dev/null &
echo "DAY2_PID=$!"
```

**建议同步启动独立时间兜底守护**（按开机时间 `+ 6 小时 45 分钟` 动态设定截止时间戳，避免跨时区误差，先给主脚本 `180s` 正常切片窗口，超时才接管回传）：

```bash
cat > /mnt/data/h3_v3/work/finish_day2_fallback.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
PERSIST="/mnt/data/h3_v3"; WORK="${PERSIST}/work"; RUN_DIR="${WORK}/lora_run_v3"; LOG_DIR="${PERSIST}/logs"
DONE="${LOG_DIR}/DONE_DAY2.txt"; MARK="${LOG_DIR}/DONE_DAY2_FALLBACK.txt"
# 从启动起算 405 分钟（6小时45分），彻底免疫容器 UTC/CST 时区差异！
DEADLINE=$(( $(date +%s) + 405 * 60 ))
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  [ -f "${DONE}" ] && exit 0
  sleep 120
done
[ -f "${DONE}" ] || [ -f "${MARK}" ] && exit 0
echo "[fallback $(date '+%F %T')] 达到 6h45m 护栏，发 SIGTERM 触发收尾" >> "${LOG_DIR}/fallback_day2.log"
pkill -TERM -f "train_h3_lora.py" 2>/dev/null || true
sleep 180
[ -f "${DONE}" ] && exit 0
pkill -f "ComfyUI/main.py" 2>/dev/null || true
LATEST=$(ls -1 "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | sort | tail -n 1)
[ -z "${LATEST}" ] && exit 4
OUT_SLICE="${RUN_DIR}/minimax_h3_sol_sharp_v3_day2_fallback_stylekeep.safetensors"
python3 "${PERSIST}/comfyui-h3-seamkit/train/slice_lora_stylekeep.py" \
  --input "${LATEST}" --output "${OUT_SLICE}" --preset action_stylekeep >> "${LOG_DIR}/fallback_day2.log" 2>&1
LATEST_STEP="${LATEST}" OUT_SLICE="${OUT_SLICE}" RUN_DIR="${RUN_DIR}" \
TOKEN_FILE="${WORK}/mstoken.txt" MS_REPO="coolwind123/h3-teacher-cache" \
python3 - >> "${LOG_DIR}/fallback_day2.log" 2>&1 <<'PY'
import os
from pathlib import Path
tok = Path(os.environ["TOKEN_FILE"]).read_text(encoding="utf-8").strip()
repo = os.environ["MS_REPO"]; run_dir = Path(os.environ["RUN_DIR"])
try:
    from modelscope_hub.api import HubApi
    api = HubApi(token=tok)
except ImportError:
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(tok)
for lp, rp in [
    (Path(os.environ["OUT_SLICE"]), "candidates_v3/minimax_h3_sol_sharp_v3_day2_fallback_stylekeep.safetensors"),
    (Path(os.environ["LATEST_STEP"]), f"candidates_v3/{Path(os.environ['LATEST_STEP']).name}"),
    (run_dir / "logs" / "metrics.jsonl", "candidates_v3/metrics_day2_fallback.jsonl"),
]:
    if lp.is_file():
        api.upload_file(repo_id=repo, repo_type="model", path_or_fileobj=str(lp), path_in_repo=rp, commit_message=f"day2 fb {lp.name}")
        print("[UPLOADED]", rp)
PY
echo "[ALL_DAY2_DONE_FALLBACK] $(date -Iseconds)" > "${MARK}"
EOF
chmod +x /mnt/data/h3_v3/work/finish_day2_fallback.sh
setsid nohup bash /mnt/data/h3_v3/work/finish_day2_fallback.sh > /dev/null 2>&1 < /dev/null &
```

---

### Step 3 · 巡检与收工停机判据

每 25～30 分钟执行一次轻量巡检：

```bash
cat /mnt/data/h3_v3/logs/milestones_day2.log 2>/dev/null
tail -n 8 /mnt/data/h3_v3/work/lora_run_v3/logs/metrics.jsonl 2>/dev/null
cat /mnt/data/h3_v3/logs/DONE_DAY2.txt /mnt/data/h3_v3/logs/DONE_DAY2_FALLBACK.txt 2>/dev/null
```

**验收与立即停机标准**：
1. 训练启动日志出现 `[init] resumed step 50 -> target 190 (exact=200, expanded=0)`；
2. `metrics.jsonl` 从 `step 55` 连续推进至 `170~190`，`0 NaN`，`mot_gate` 回落至合理区间（`sakuga` 约 `0.55~0.80`、常规约 `0.25~0.45`），`loss_lf` 较首夜明显下降；
3. `upload_day2.log` 出现 `[UPLOADED] minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors` 且 `DONE_DAY2.txt` 写出 `[ALL_DAY2_DONE]` 后，**立即在魔搭网页停止该 AMD 实例**。
