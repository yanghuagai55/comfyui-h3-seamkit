#!/usr/bin/env bash
# 魔搭内网直装 ComfyUI + T8 节点（当 gh-proxy/ghfast 全挂时使用）
# 依赖：bundle/ComfyUI-master.tar.gz 与 bundle/T8-minimax-h3-audio.tar.gz 已上传到魔搭私有仓
set -euo pipefail

PERSIST="${PERSIST_ROOT:-/mnt/data/h3_v3}"
WORK="${PERSIST}/work"
MODEL_ROOT="${PERSIST}/models"
COMFY="${PERSIST}/ComfyUI"
MS_REPO="${MS_REPO:-coolwind123/h3-teacher-cache}"
SRC_CODE="${PERSIST}/comfyui-h3-seamkit"
export MODELSCOPE_API_TOKEN="$(cat "${WORK}/mstoken.txt")"

echo "[deploy] 1/6 从魔搭仓拉安装包 ..."
mkdir -p "${PERSIST}/assets/bundle"
modelscope download --model "${MS_REPO}" \
  bundle/ComfyUI-master.tar.gz bundle/T8-minimax-h3-audio.tar.gz \
  --local_dir "${PERSIST}/assets"

echo "[deploy] 2/6 解包 ComfyUI ..."
mkdir -p "${COMFY}"
tar -xzf "${PERSIST}/assets/bundle/ComfyUI-master.tar.gz" -C "${COMFY}" --strip-components=1

echo "[deploy] 3/6 解包 T8 音频节点 + 同步本仓库 ..."
mkdir -p "${COMFY}/custom_nodes/comfyui-minimax-h3-audio-T8"
tar -xzf "${PERSIST}/assets/bundle/T8-minimax-h3-audio.tar.gz" \
  -C "${COMFY}/custom_nodes/comfyui-minimax-h3-audio-T8" --strip-components=1
DST_NODE="${COMFY}/custom_nodes/comfyui-h3-seamkit"
mkdir -p "${DST_NODE}"
if [ "${SRC_CODE}" != "${DST_NODE}" ]; then
  cp -rf "${SRC_CODE}"/* "${DST_NODE}/"
fi

echo "[deploy] 4/6 安装依赖（过滤 torch 三件套，保护 ROCm 预装）..."
for req in "${COMFY}/requirements.txt" "${COMFY}/custom_nodes/comfyui-minimax-h3-audio-T8/requirements.txt"; do
  [ -f "${req}" ] || continue
  grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "${req}" > /tmp/req_filtered.txt || true
  pip install -q -r /tmp/req_filtered.txt 2>>"${PERSIST}/logs/deploy_comfy.log" || echo "[deploy] WARN 依赖安装部分失败: ${req}"
done

echo "[deploy] 5/6 软链模型与 /content ..."
mkdir -p /content "${COMFY}/input"
for sub in diffusion_models text_encoders vae upscale_models loras; do
  mkdir -p "${COMFY}/models/${sub}"
  for f in "${MODEL_ROOT}/${sub}"/*.safetensors; do
    [ -e "$f" ] && ln -sfn "$f" "${COMFY}/models/${sub}/$(basename "$f")"
  done
done
ln -sfn "${COMFY}" /content/ComfyUI
ln -sfn "${WORK}" /content/work
ln -sfn "${MODEL_ROOT}" /content/local_models

echo "[deploy] 6/6 私有资源复原 + ROCm Preflight ..."
bash "${SRC_CODE}/train/bootstrap_public_modelscope.sh" \
  || echo "[deploy] WARN bootstrap 非致命失败（分片解压/网络），继续 preflight"
python3 "${SRC_CODE}/train/preflight.py" --role teacher --min-vram-gb 70 \
  --work-root "${WORK}" --model-root "${MODEL_ROOT}"

echo "[PREP_ALL_OK] $(date -Iseconds)"
