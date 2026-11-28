#!/usr/bin/env bash
# 第二轮 7 小时接力总控脚本（v3 Session 2）：
# 1) 补齐剩余核心 8 条 39f 零模糊教师（完成全部 14/14 R2VA 角色动作 + 4 条 T2VA，总库 82 -> 90 条）
# 2) 用 Python zstandard+tarfile 打包备份全部 sakuga*.pt 至魔搭私有仓（修复首夜容器无 zstd CLI 跳过问题）
# 3) 从 resume/latest_state.pt (step=50) 无缝继承权重与 AdamW 动量续训至 190 步（save-every=20，校准 motion-gate-tau=0.75）
# 4) 导出最新步 (s180/s190) 与中段 (s120) 双档 action_stylekeep 切片并回传魔搭私有仓
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
rm -f "${LOG_DIR}/DONE_DAY2.txt"
log() { echo "[day2 $(date '+%F %T')] $*" | tee -a "${LOG_DIR}/milestones_day2.log"; }

# 运行期防 1h 空闲回收心跳（退出即停）
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

# 0) 确认续训断点 step >= 50 存在
if [ ! -s "${RUN_DIR}/resume/latest_state.pt" ]; then
  log "[FATAL] 缺少续训断点 ${RUN_DIR}/resume/latest_state.pt，中止防覆盖！"
  exit 3
fi

# 1) 生成第二轮精选 8 条 39f 队列（补齐剩余 4 条双人交锋 + 2 条单人角色 + 2 条无参考 T2VA）
python3 - <<'PY'
import json
from pathlib import Path

wanted = {
    # 首夜 10 条（若个别未在 cache 也会自动补漏，已在 cache 则秒跳过）
    "sakuga39_0001", "sakuga39_0002", "sakuga39_0004", "sakuga39_0008",
    "sakuga39_0009", "sakuga39_0010", "sakuga39_0012", "sakuga39_0013",
    "sakuga39_0015", "sakuga39_0018",
    # 第二轮新增 8 条（凑齐全部 14 条 R2VA 角色样本 + 4 条 T2VA）
    "sakuga39_0003", "sakuga39_0005", "sakuga39_0006", "sakuga39_0007",
    "sakuga39_0011", "sakuga39_0014", "sakuga39_0016", "sakuga39_0017",
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
Path("/tmp/day2_sakuga_8.jsonl").write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
print(f"[queue-day2] cached_total={len(done)} day2_pending={len(out)}")
PY

PENDING_N=$(wc -l < /tmp/day2_sakuga_8.jsonl | tr -d ' ')
if [ "${PENDING_N}" -gt 0 ]; then
  log "=== STAGE 1: 生成第二轮 ${PENDING_N} 条 39f 零残影教师（硬上限 150 分钟） ==="
  python3 train/boot.py \
    --mode teacher --repo "" \
    --prompts /tmp/day2_sakuga_8.jsonl \
    --limit 0 \
    --clip "${CLIP_NAME}" \
    --teacher-steps 28 --teacher-cfg 5.5 \
    --zup-mode hybrid --min-motion-lapvar 0 \
    --shard-size 2 \
    --teacher-budget-min 150 \
    --cache "${CACHE_DIR}" \
    --out /tmp/h3_native_raw \
    --work-root "${WORK_DIR}" \
    --model-root "${MODEL_ROOT}" || log "[WARN] teacher 非零退出，继续用已落盘样本续训"
else
  log "18 条目标 sakuga 教师均已在 cache 中，直接进入续训"
fi

log "=== STAGE 1B: 杀净 ComfyUI + 用 Python zstandard 打包备份全部 sakuga*.pt 至魔搭私有仓 ==="
pkill -f "ComfyUI/main.py" 2>/dev/null || true
pkill -f "main.py --listen 127.0.0.1 --port 8188" 2>/dev/null || true
sleep 6
rm -rf /tmp/h3_native_raw /tmp/h3_zup_tmp

TOKEN_FILE="${WORK_DIR}/mstoken.txt" MS_REPO="${MS_REPO}" CACHE_DIR="${CACHE_DIR}" \
python3 - <<'PY' || log "[WARN] sakuga 教师备份上传非致命失败（NAS 已存妥，继续续训）"
import os, tarfile
from pathlib import Path

cache = Path(os.environ["CACHE_DIR"])
sakuga_files = sorted(p for p in cache.glob("sakuga*.pt") if p.stat().st_size > 1024)
all_files = list(cache.glob("*.pt"))
print(f"[backup] sakuga={len(sakuga_files)} total={len(all_files)}")
if sakuga_files:
    tar_out = Path("/tmp/sakuga_v3_all.tar.zst")
    try:
        import zstandard
        cctx = zstandard.ZstdCompressor(level=6, threads=4)
        with tar_out.open("wb") as raw_fh, cctx.stream_writer(raw_fh, closefd=False) as zfh:
            with tarfile.open(fileobj=zfh, mode="w|") as tf:
                for p in sakuga_files:
                    tf.add(p, arcname=p.name)
        dst = "teacher_cache_v3_action/sakuga_v3_all.tar.zst"
    except Exception:
        tar_out = Path("/tmp/sakuga_v3_all.tar.gz")
        with tarfile.open(tar_out, "w:gz") as tf:
            for p in sakuga_files:
                tf.add(p, arcname=p.name)
        dst = "teacher_cache_v3_action/sakuga_v3_all.tar.gz"

    tok = Path(os.environ["TOKEN_FILE"]).read_text(encoding="utf-8").strip()
    repo = os.environ["MS_REPO"]
    try:
        from modelscope_hub.api import HubApi
        api = HubApi(token=tok)
    except ImportError:
        from modelscope.hub.api import HubApi
        api = HubApi(); api.login(tok)
    api.upload_file(repo_id=repo, repo_type="model",
                    path_or_fileobj=str(tar_out), path_in_repo=dst,
                    commit_message=f"backup {len(sakuga_files)} sakuga v3 teachers")
    print(f"[UPLOADED_TEACHER] {dst} ({tar_out.stat().st_size/1024/1024:.1f} MiB)")
    tar_out.unlink(missing_ok=True)
PY

log "=== STAGE 2: 从 step 50 无缝续训至 190 步（save-every=20, motion-gate-tau=0.75, lr=6.8e-5） ==="
# 实测 112.5s/步，140 步约 262 分钟；设 timeout 240m 硬护栏确保必在 7h 窗口内留足切片上传缓冲
timeout -s TERM 240m python3 train/train_h3_lora.py \
  --comfy "${COMFY_DIR}" \
  --base "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" \
  --cache "${CACHE_DIR}" \
  --out "${RUN_DIR}" \
  --phase A \
  --frames 22,39 \
  --steps 190 \
  --rank-attn 64 --rank-mlp 32 \
  --init-lora "${PERSIST_ROOT}/assets/final/minimax_h3_sol_sharp_v2_final.safetensors" \
  --attn-backend sdpa_mask --block-size 64 \
  --lf-base-anchor --lf-kernel 5 \
  --motion-gate-tau 0.75 --grad-xy-weight 0.15 --action-boost 3.2 \
  --optim adamw --lr 6.8e-5 --warmup 10 --grad-clamp 10 \
  --save-every 20 --log-every 5 \
  --auto-resume >> "${LOG_DIR}/train_day2.log" 2>&1 \
  || log "[WARN] train 触发限时或退出，取最新已保存 step 切片上传"

LATEST_STEP=$(ls -1 "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | sort | tail -n 1)
if [ -z "${LATEST_STEP}" ] || [ ! -s "${LATEST_STEP}" ]; then
  log "[FAIL] 未找到有效 step 权重"; echo "[DAY2_ABORTED]" | tee -a "${LOG_DIR}/DONE_DAY2.txt"; exit 4
fi
log "第二轮最新权重: ${LATEST_STEP}"

log "=== STAGE 3: 导出双档 action_stylekeep 切片 (最新终版 + step_000120 中段对照) + 回传魔搭仓 ==="
OUT_FINAL="${RUN_DIR}/minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors"
python3 train/slice_lora_stylekeep.py \
  --input "${LATEST_STEP}" \
  --output "${OUT_FINAL}" \
  --preset action_stylekeep >> "${LOG_DIR}/train_day2.log" 2>&1

OUT_MID="${RUN_DIR}/minimax_h3_sol_sharp_v3_s120_action_stylekeep.safetensors"
if [ -s "${RUN_DIR}/steps/step_000120.safetensors" ]; then
  python3 train/slice_lora_stylekeep.py \
    --input "${RUN_DIR}/steps/step_000120.safetensors" \
    --output "${OUT_MID}" \
    --preset action_stylekeep >> "${LOG_DIR}/train_day2.log" 2>&1 || true
fi

LATEST_STEP="${LATEST_STEP}" OUT_FINAL="${OUT_FINAL}" OUT_MID="${OUT_MID}" RUN_DIR="${RUN_DIR}" \
TOKEN_FILE="${WORK_DIR}/mstoken.txt" MS_REPO="${MS_REPO}" \
python3 - <<'PY' 2>&1 | tee "${LOG_DIR}/upload_day2.log"
import os, subprocess
from pathlib import Path

tok = Path(os.environ["TOKEN_FILE"]).read_text(encoding="utf-8").strip()
repo = os.environ["MS_REPO"]
run_dir = Path(os.environ["RUN_DIR"])
files = [
    (Path(os.environ["OUT_FINAL"]), "candidates_v3/minimax_h3_sol_sharp_v3_day2_action_stylekeep.safetensors"),
    (Path(os.environ["LATEST_STEP"]), f"candidates_v3/{Path(os.environ['LATEST_STEP']).name}"),
    (Path(os.environ["OUT_MID"]), "candidates_v3/minimax_h3_sol_sharp_v3_s120_action_stylekeep.safetensors"),
    (run_dir / "logs" / "metrics.jsonl", "candidates_v3/metrics_day2.jsonl"),
]

api = None
try:
    from modelscope_hub.api import HubApi
    api = HubApi(token=tok)
except Exception:
    try:
        from modelscope.hub.api import HubApi
        api = HubApi(); api.login(tok)
    except Exception:
        api = None

ok_cnt = 0
for lp, rp in files:
    if not lp.is_file():
        print("[SKIP]", lp.name)
        continue
    done = False
    if api is not None:
        for attempt in range(3):
            try:
                api.upload_file(repo_id=repo, repo_type="model",
                                path_or_fileobj=str(lp), path_in_repo=rp,
                                commit_message=f"day2 {lp.name}")
                done = True
                break
            except Exception as e:
                print(f"[RETRY {attempt+1}] {lp.name}: {e}")
    if not done:
        subprocess.run(["modelscope", "upload", repo, str(lp), rp, "--token", tok], check=True)
    ok_cnt += 1
    print(f"[UPLOADED] {lp.name} -> {rp} ({lp.stat().st_size/1024/1024:.1f} MiB)")

assert ok_cnt >= 2, f"上传成功数不足: {ok_cnt}"
PY

{
  echo "FINISHED_AT=$(date -Iseconds)"
  ls -lh "${OUT_FINAL}" "${RUN_DIR}/steps/" "${RUN_DIR}/resume/" 2>/dev/null
  echo "[ALL_DAY2_DONE]"
} | tee "${LOG_DIR}/DONE_DAY2.txt"

log "[ALL_DAY2_DONE] 第二轮 7h 接力全部落仓，请立即停止实例！"
