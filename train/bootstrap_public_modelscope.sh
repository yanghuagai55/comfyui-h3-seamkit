#!/usr/bin/env bash
# 魔搭（ModelScope）实例恢复训练环境：HF 不可达时，用「公共 NAS /mnt/data/h3_v3 + 魔搭私有仓」补齐
# 用法：PERSIST_ROOT=/mnt/data/h3_v3 bash train/bootstrap_public_modelscope.sh
# 说明：模型权重（~96GB）已由 fetch_models_modelscope.sh 落在 ${PERSIST_ROOT}/models，本脚本只补私有资源与代码。
set -euo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/mnt/data/h3_v3}"
WORK_DIR="${PERSIST_ROOT}/work"
MODEL_ROOT="${PERSIST_ROOT}/models"
COMFY_DIR="${COMFY_DIR:-/content/ComfyUI}"
MS_REPO="${MS_REPO:-coolwind123/h3-teacher-cache}"
MS_TOKEN_FILE="${MS_TOKEN_FILE:-${WORK_DIR}/mstoken.txt}"
ASSET_DIR="${PERSIST_ROOT}/assets"
CODE_DIR="${PERSIST_ROOT}/comfyui-h3-seamkit"

mkdir -p "${WORK_DIR}" "${ASSET_DIR}"
if [ ! -s "${MS_TOKEN_FILE}" ]; then
  echo "[boot] FATAL: 缺少魔搭访问令牌 ${MS_TOKEN_FILE}（token 形如 ms-xxxxxxxx）" >&2
  exit 2
fi
export MODELSCOPE_API_TOKEN="$(cat "${MS_TOKEN_FILE}")"

if [ ! -f "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ]; then
  echo "[boot] FATAL: 公共目录缺少主模型，先执行 train/fetch_models_modelscope.sh" >&2
  exit 3
fi

echo "[boot] 拉取私有资源（代码包 / v2 热启动 / 参考图）..."
modelscope download --model "${MS_REPO}" \
  bundle/comfyui-h3-seamkit-train.tar.gz \
  final/minimax_h3_sol_sharp_v2_final.safetensors \
  refs/deepseek.png \
  refs/claude_sheet.png \
  --local_dir "${ASSET_DIR}"

if [ ! -f "${CODE_DIR}/train/train_h3_lora.py" ]; then
  echo "[boot] 解包训练代码 -> ${CODE_DIR}"
  mkdir -p "${CODE_DIR}"
  tar -xzf "${ASSET_DIR}/bundle/comfyui-h3-seamkit-train.tar.gz" -C "${CODE_DIR}"
fi

echo "[boot] 拉取并解压 teacher_cache_v1 分片..."
SHARD_ARGS=("${MS_REPO}" "teacher_cache_v1/manifest.json")
for i in 0 1 2 3 4 5 6 7 8; do
  SHARD_ARGS+=("teacher_cache_v1/shards/shard_$(printf '%03d' "$i").tar.zst")
done
if modelscope download --model "${SHARD_ARGS[@]}" --local_dir "${ASSET_DIR}/teacher_repo"; then
  mkdir -p "${WORK_DIR}/teacher_cache"
  for f in "${ASSET_DIR}"/teacher_repo/teacher_cache_v1/shards/*.tar.zst; do
    [ -e "$f" ] || continue
    if tar --zstd -xf "$f" -C "${WORK_DIR}/teacher_cache" 2>/dev/null; then
      continue
    fi
    # 镜像里可能没有 zstd 二进制：退回 python zstandard
    python3 - "$f" "${WORK_DIR}/teacher_cache" <<'PY' || echo "[boot] WARN 分片解压失败: $f"
import sys, tarfile
from pathlib import Path
src, dst = sys.argv[1], sys.argv[2]
try:
    import zstandard
except ImportError:
    raise SystemExit("[boot] 缺少 zstandard 模块，跳过该分片")
with open(src, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        tf.extractall(dst)
PY
  done
  echo "[boot] 本地教师样本: $(find "${WORK_DIR}/teacher_cache" -maxdepth 1 -name '*.pt' | wc -l) 条"
else
  echo "[boot] 警告: teacher_cache_v1 未取到（私有仓尚未上传分片），先跳过；不影响 v3 新样本从零开训"
fi

# /content 软链（与训练脚本默认路径一致）
mkdir -p /content
ln -sfn "${WORK_DIR}" /content/work
ln -sfn "${MODEL_ROOT}" /content/local_models
[ -d "${COMFY_DIR}" ] && ln -sfn "${COMFY_DIR}" /content/ComfyUI

# 参考图就位（ComfyUI 已部署时）
if [ -d "${COMFY_DIR}/input" ]; then
  cp -f "${ASSET_DIR}/refs/deepseek.png" "${COMFY_DIR}/input/deepseek.png"
  cp -f "${ASSET_DIR}/refs/claude_sheet.png" "${COMFY_DIR}/input/claude_sheet.png"
  echo "[boot] 参考图已复制到 ${COMFY_DIR}/input/"
fi

[ -s "${WORK_DIR}/rocm_clip_name.txt" ] || printf '%s' "qwen3vl_32b_minimax_h3_bf16.safetensors" > "${WORK_DIR}/rocm_clip_name.txt"

cat <<EOF
[boot] 完成。HF 不可达环境请使用「纯本地模式」（不传 --repo / --hf-repo / --pull-shards）：
  教师: python3 ${CODE_DIR}/train/run_teacher.py --prompts ${CODE_DIR}/train/prompts_sakuga_action_32.jsonl \\
          --cache ${WORK_DIR}/teacher_cache --repo "" --limit 20 ...
  训练: python3 ${CODE_DIR}/train/train_h3_lora.py --base ${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \\
          --cache ${WORK_DIR}/teacher_cache --out ${WORK_DIR}/lora_run_v3 \\
          --init-lora ${ASSET_DIR}/final/minimax_h3_sol_sharp_v2_final.safetensors ...
EOF
