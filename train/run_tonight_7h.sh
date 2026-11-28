#!/usr/bin/env bash
# 今晚 7 小时黄金配比（Gemini 方案）：分层精选 10×39f 教师 → 150 步训练 → action_stylekeep 切片 → 魔搭回传
# 纯本地模式；硬截断保护：教师预算 185 分钟、训练 timeout 160 分钟，确保 06:35 前进入切片上传。
set -euo pipefail

export PERSIST_ROOT="/mnt/data/h3_v3"
export WORK_DIR="${PERSIST_ROOT}/work"
export MODEL_ROOT="${PERSIST_ROOT}/models"
export CODE_DIR="${PERSIST_ROOT}/comfyui-h3-seamkit"
export COMFY_DIR="/content/ComfyUI"
export CACHE_DIR="${WORK_DIR}/teacher_cache"
export RUN_DIR="${WORK_DIR}/lora_run_v3"
export LOG_DIR="${PERSIST_ROOT}/logs"
export MS_REPO="coolwind123/h3-teacher-cache"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MODELSCOPE_API_TOKEN="$(cat "${WORK_DIR}/mstoken.txt")"
CLIP_NAME="$(cat "${WORK_DIR}/rocm_clip_name.txt")"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}" "${RUN_DIR}"
rm -f "${LOG_DIR}/DONE_TONIGHT.txt"
log() { echo "[tonight $(date '+%F %T')] $*" | tee -a "${LOG_DIR}/milestones.log"; }

# 运行期防 1h 空闲回收心跳
(
  while kill -0 $$ 2>/dev/null; do
    date -Iseconds > /mnt/workspace/.overnight_heartbeat 2>/dev/null || true
    curl -fsS --max-time 3 http://127.0.0.1:8888/ >/dev/null 2>&1 || true
    sleep 600
  done
) &
HB_PID=$!
trap 'kill "${HB_PID}" 2>/dev/null || true; pkill -f "ComfyUI/main.py" 2>/dev/null || true' EXIT

cd "${CODE_DIR}"

# 1) 分层精选 10 条 39f（跳过 cache 已有）
python3 - <<'PY'
import json
from pathlib import Path

wanted = {
    "sakuga39_0001", "sakuga39_0002", "sakuga39_0004", "sakuga39_0008",
    "sakuga39_0009", "sakuga39_0010", "sakuga39_0012", "sakuga39_0013",
    "sakuga39_0015", "sakuga39_0018",
}
src = Path("/mnt/data/h3_v3/comfyui-h3-seamkit/train/prompts_sakuga_action_32.jsonl")
cache = Path("/mnt/data/h3_v3/work/teacher_cache")
done = {p.stem for p in cache.glob("*.pt") if p.stat().st_size > 1024}
out = []
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    sid = str(d["id"])
    if sid in wanted and sid not in done:
        out.append(line)
Path("/tmp/tonight_sakuga_10.jsonl").write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
print(f"[queue] cached={len(done)} tonight_pending={len(out)}")
PY

PENDING_N=$(wc -l < /tmp/tonight_sakuga_10.jsonl | tr -d ' ')
if [ "${PENDING_N}" -gt 0 ]; then
  log "=== STAGE 1: 生成精选 ${PENDING_N} 条 39f 零残影教师（硬上限 185 分钟） ==="
  python3 train/boot.py \
    --mode teacher --repo "" \
    --prompts /tmp/tonight_sakuga_10.jsonl \
    --limit 0 \
    --clip "${CLIP_NAME}" \
    --teacher-steps 28 --teacher-cfg 5.5 \
    --zup-mode hybrid --min-motion-lapvar 0 \
    --shard-size 2 \
    --teacher-budget-min 185 \
    --cache "${CACHE_DIR}" \
    --out /tmp/h3_native_raw \
    --work-root "${WORK_DIR}" \
    --model-root "${MODEL_ROOT}" || log "[WARN] teacher 非零退出，继续用已落盘样本开训"
fi

log "=== STAGE 1B: 杀净 ComfyUI + 备份今晚新教师至魔搭私有仓 ==="
pkill -f "ComfyUI/main.py" 2>/dev/null || true
pkill -f "main.py --listen 127.0.0.1 --port 8188" 2>/dev/null || true
sleep 6
rm -rf /tmp/h3_native_raw /tmp/h3_zup_tmp

SAKUGA_N=$(find "${CACHE_DIR}" -maxdepth 1 -name 'sakuga*.pt' | wc -l | tr -d ' ')
log "teacher_cache 当前 sakuga=${SAKUGA_N}, total=$(find "${CACHE_DIR}" -maxdepth 1 -name '*.pt' | wc -l)"
if [ "${SAKUGA_N}" -gt 0 ]; then
  tar --zstd -cf /tmp/sakuga_n1.tar.zst -C "${CACHE_DIR}" $(cd "${CACHE_DIR}" && ls sakuga*.pt) 2>/dev/null && {
    TARBALL=/tmp/sakuga_n1.tar.zst TOKEN_FILE="${WORK_DIR}/mstoken.txt" MS_REPO="${MS_REPO}" \
    python3 - <<'PY' || log "[WARN] 教师备份上传失败（NAS 已存妥，不阻塞训练）"
import os
from pathlib import Path
src = os.environ["TARBALL"]
dst = "teacher_cache_v3_action/sakuga_night1.tar.zst"
tok = Path(os.environ["TOKEN_FILE"]).read_text(encoding="utf-8").strip()
from modelscope_hub.api import HubApi
api = HubApi(token=tok)
api.upload_file(repo_id=os.environ["MS_REPO"], repo_type="model",
                path_or_fileobj=src, path_in_repo=dst, commit_message="sakuga night1 teachers")
print("[UPLOADED_TEACHER]", dst)
PY
    rm -f /tmp/sakuga_n1.tar.zst
  } || log "[WARN] 教师备份打包失败（不阻塞训练）"
fi

log "=== STAGE 2: 训练 150 步（22,39 混桶, action-boost=4.0, save-every=25） ==="
# timeout 160m 保底，确保最晚 06:25 前进入切片与上传，避免 07:00 强收丢盘
timeout -s TERM 160m python3 train/train_h3_lora.py \
  --comfy "${COMFY_DIR}" \
  --base "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" \
  --cache "${CACHE_DIR}" \
  --out "${RUN_DIR}" \
  --phase A \
  --frames 22,39 \
  --steps 150 \
  --rank-attn 64 --rank-mlp 32 \
  --init-lora "${PERSIST_ROOT}/assets/final/minimax_h3_sol_sharp_v2_final.safetensors" \
  --attn-backend sdpa_mask --block-size 64 \
  --lf-base-anchor --lf-kernel 5 \
  --motion-gate-tau 0.35 --grad-xy-weight 0.15 --action-boost 4.0 \
  --optim adamw --lr 8e-5 --warmup 15 --grad-clamp 10 \
  --save-every 25 --log-every 5 \
  --auto-resume >> "${LOG_DIR}/train_tonight.log" 2>&1 \
  || log "[WARN] train 触发限时或退出，取最新已保存 step 切片上传"

LATEST_STEP=$(ls -1 "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | sort | tail -n 1)
if [ -z "${LATEST_STEP}" ] || [ ! -s "${LATEST_STEP}" ]; then
  log "[FAIL] 没有任何 step 权重，终止"; echo "[TONIGHT_ABORTED]" | tee -a "${LOG_DIR}/DONE_TONIGHT.txt"; exit 4
fi
log "最新权重: ${LATEST_STEP}"

log "=== STAGE 3: 导出 action_stylekeep 切片 + 回传魔搭私有仓 ==="
OUT_SLICE="${RUN_DIR}/minimax_h3_sol_sharp_v3_night1_action_stylekeep.safetensors"
python3 train/slice_lora_stylekeep.py \
  --input "${LATEST_STEP}" \
  --output "${OUT_SLICE}" \
  --preset action_stylekeep >> "${LOG_DIR}/train_tonight.log" 2>&1 || log "[WARN] 切片失败"

LATEST_STEP="${LATEST_STEP}" OUT_SLICE="${OUT_SLICE}" RUN_DIR="${RUN_DIR}" \
TOKEN_FILE="${WORK_DIR}/mstoken.txt" MS_REPO="${MS_REPO}" \
python3 - <<'PY' 2>&1 | tee "${LOG_DIR}/upload_night1.log"
import os
from pathlib import Path

tok = Path(os.environ["TOKEN_FILE"]).read_text(encoding="utf-8").strip()
repo = os.environ["MS_REPO"]
run_dir = Path(os.environ["RUN_DIR"])
files = [
    (Path(os.environ["OUT_SLICE"]), "candidates_v3/minimax_h3_sol_sharp_v3_night1_action_stylekeep.safetensors"),
    (Path(os.environ["LATEST_STEP"]), f"candidates_v3/{Path(os.environ['LATEST_STEP']).name}"),
    (run_dir / "logs" / "metrics.jsonl", "candidates_v3/metrics_night1.jsonl"),
]
from modelscope_hub.api import HubApi
api = HubApi(token=tok)
for lp, rp in files:
    if not lp.is_file():
        print("[SKIP]", lp.name)
        continue
    for attempt in range(3):
        try:
            api.upload_file(repo_id=repo, repo_type="model",
                            path_or_fileobj=str(lp), path_in_repo=rp,
                            commit_message=f"night1 {lp.name}")
            print(f"[UPLOADED] {lp.name} -> {rp} ({lp.stat().st_size/1024/1024:.1f} MiB)")
            break
        except Exception as e:
            print(f"[RETRY {attempt+1}] {lp.name}: {e}")
PY

{
  echo "FINISHED_AT=$(date -Iseconds)"
  ls -lh "${OUT_SLICE}" "${RUN_DIR}/steps/" "${RUN_DIR}/resume/" 2>/dev/null
  echo "[ALL_TONIGHT_DONE]"
} | tee "${LOG_DIR}/DONE_TONIGHT.txt"

log "[ALL_TONIGHT_DONE] 今晚产物已落仓，请立即停止实例！"
