#!/usr/bin/env bash
# Route 3 一键容错流水线：
#   Preflight -> Stage 0 (稠密 Oracle + 诊断探针) -> Stage 1a (1.5min 金丝雀真实成片门禁)
#             -> Stage 1b (通用动态档全量消融 + 静机位档仅跑 case1_static + 拼板)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-/content/h3_route3}"
API_JSON="${API_JSON:-${REPO_ROOT}/route3/workflows/local_dualclock_api.json}"
SERVER="${SERVER:-http://127.0.0.1:8188}"
PY="${PY:-python3}"
COMBOS="${COMBOS:-3:1:flow,3:1:lerp,3:0:flow,4:1:flow}"
CANARY_CASE="${CANARY_CASE:-case2_dolly}"
CANARY_COMBO="${CANARY_COMBO:-3:1:flow}"
SEED="${SEED:-20261122}"
DURATION="${DURATION:-5.0}"
MEGAPIXELS="${MEGAPIXELS:-1.3}"   # 二采最终画质；720p 不够看，1080p 本地跑不到
STRIP_LOWVRAM="${STRIP_LOWVRAM:-1}"
EXTRA_FLAGS=""
if [ "${STRIP_LOWVRAM}" = "1" ]; then
    EXTRA_FLAGS="--strip-lowvram"
fi

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/artifacts/stage0" "${WORK_ROOT}/artifacts/stage1"

echo "=== [0] Preflight (timeout 3m) ==="
if [ ! -f "${WORK_ROOT}/preflight_ok.json" ]; then
    timeout --signal=TERM --kill-after=30 180 \
        "${PY}" "${REPO_ROOT}/route3/preflight_check.py" --work-root "${WORK_ROOT}"
fi

echo "=== [1] Stage 0: Dense Oracle + Diagnostic Probe + Freeze Pass-1 AV Latent (timeout 50m) ==="
if [ ! -f "${WORK_ROOT}/artifacts/stage0/stage0_DONE.ok" ]; then
    timeout --signal=TERM --kill-after=60 3000 \
        "${PY}" "${REPO_ROOT}/route3/stage0_oracle.py" \
        --api "${API_JSON}" \
        --out "${WORK_ROOT}/artifacts/stage0" \
        --server "${SERVER}" \
        --seed "${SEED}" \
        --duration "${DURATION}" \
        --second-mp "${MEGAPIXELS}" \
        ${EXTRA_FLAGS} \
        --probe --resume
fi

# 探针仅作诊断排序器与 U 型层画像（取消拍脑袋 rel_err>0.20 盲杀；真正止损交给下方 1.5 分钟金丝雀成片！）
PROBE_JSON="${WORK_ROOT}/artifacts/stage0/probe_attn_stats.json"
if [ -f "${PROBE_JSON}" ]; then
    "${PY}" -c "import json,sys; s=json.load(open(sys.argv[1])).get('summary',{}); print('[diag] probe:', s.get('verdict'), 'bands=', s.get('layer_bands'))" "${PROBE_JSON}"
fi

if [ -f "${WORK_ROOT}/STOP_REQUESTED" ]; then
    echo "[gate] STOP_REQUESTED detected before Stage 1, exiting cleanly."
    date -u > "${WORK_ROOT}/PIPELINE_FINISHED.ok"
    exit 0
fi

echo "=== [2a] Stage 1a: 1-Clip Canary Gate (${CANARY_CASE} x ${CANARY_COMBO}, timeout 15m) ==="
CANARY_DIR="${WORK_ROOT}/artifacts/stage1_canary"
mkdir -p "${CANARY_DIR}"
timeout --signal=TERM --kill-after=60 900 \
    "${PY}" "${REPO_ROOT}/route3/stage1_ablate.py" \
    --api "${API_JSON}" \
    --stage0 "${WORK_ROOT}/artifacts/stage0" \
    --out "${CANARY_DIR}" \
    --server "${SERVER}" \
    --only "${CANARY_CASE}" \
    --combos "${CANARY_COMBO}" \
    --seed "${SEED}" \
    ${EXTRA_FLAGS} \
    --resume

# 用真实端到端成片指标判止损（fg_ssim>=0.82 或 fg_psnr>=22，且 bg_hf 在 [0.80,1.05] 双向防重影，seam<=1.8）
CANARY_OK=$("${PY}" -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8'))) if sys.argv[1] else []
if not rows:
    print('NO_ROWS'); sys.exit(0)
r = rows[0]
hf = float(r.get('bg_hf_sharpness') or 0)
ok = ((float(r.get('fg_ssim') or 0) >= 0.82 or float(r.get('fg_psnr') or 0) >= 22.0)
      and (0.80 <= hf <= 1.05) and float(r.get('seam_grad_ratio') or 9) <= 1.8)
print('PASS' if ok else 'FAIL', r)
" "${CANARY_DIR}/stage1_ablation.csv" 2>/dev/null || echo "FAIL")
echo "[gate] Stage 1a Canary result: ${CANARY_OK}"
if echo "${CANARY_OK}" | grep -q "^FAIL"; then
    echo "[gate] CANARY STOP-LOSS TRIGGERED: 跳过全量消融，请查 probe_attn_stats.json U型层画像（尝试保留末5层稠密或降 S_bg=3，否则转路线 A）。"
    date -u > "${WORK_ROOT}/PIPELINE_FINISHED.ok"
    exit 0
fi

echo "=== [2b] Stage 1b: Full Narrative Ablation (${COMBOS}) + Static-Only (case1_static x 0:1:flow) ==="
if [ ! -f "${WORK_ROOT}/artifacts/stage1/stage1_DONE.ok" ]; then
    timeout --signal=TERM --kill-after=60 5400 \
        "${PY}" "${REPO_ROOT}/route3/stage1_ablate.py" \
        --api "${API_JSON}" \
        --stage0 "${WORK_ROOT}/artifacts/stage0" \
        --out "${WORK_ROOT}/artifacts/stage1" \
        --server "${SERVER}" \
        --combos "${COMBOS}" \
        --seed "${SEED}" \
        ${EXTRA_FLAGS} \
        --resume

    # s_bg=0 仅静机位专享档（非通用动态重绘），单独仅在 case1_static 上标定极限压缩上限
    timeout --signal=TERM --kill-after=60 900 \
        "${PY}" "${REPO_ROOT}/route3/stage1_ablate.py" \
        --api "${API_JSON}" \
        --stage0 "${WORK_ROOT}/artifacts/stage0" \
        --out "${WORK_ROOT}/artifacts/stage1_static_only" \
        --server "${SERVER}" \
        --only "case1_static" \
        --combos "0:1:flow" \
        --seed "${SEED}" \
        ${EXTRA_FLAGS} \
        --resume
fi

date -u > "${WORK_ROOT}/PIPELINE_FINISHED.ok"
echo "=== Route 3 Pipeline Finished Cleanly ==="
