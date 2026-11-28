#!/usr/bin/env bash
# 魔搭（ModelScope）镜像源一键拉取 MiniMax H3 训练所需只读权重
# 背景：魔搭实例内 huggingface.co 不可达、hf-mirror 返回 0 字节；裸 curl 直连魔搭 CDN 仅 ~1.3MB/s，
#       而 modelscope download 在 DSW 内网走加速通道，实测 150-266MB/s（96GB 约 5-10 分钟）。
# 特性：内网加速下载 + 断点续传 + 字节数校验 + safetensors 头校验 + 软链就位；失败自动回退 aria2/wget。
set -euo pipefail

REPO="${MS_REPO:-Comfy-Org/MiniMax-H3}"
UP_REPO="${MS_UP_REPO:-LBH-123-AI/Minimax_h3_latent_Upscaler}"
MS_BASE="${MS_BASE:-https://www.modelscope.cn/api/v1/models}"
MODEL_ROOT="${MODEL_ROOT:-}"
WORK_DIR="${WORK_DIR:-/content/work}"
COMFY_DIR="${COMFY_DIR:-/content/ComfyUI}"
THREADS="${THREADS:-16}"
WITH_UPSCALER="${WITH_UPSCALER:-0}"

# 训练/教师生成所需的只读权重（仓库内路径|精确字节数）
FILES=(
  "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors|40225724176"
  "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors|51506295256"
  "vae/minimax_h3_sol_vae_fp16.safetensors|3596335728"
  "vae/minimax_h3_audio_vae_fp32.safetensors|605254808"
)
UP_FILE="minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
UP_BYTES=690592992
UP_DST_NAME="h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"

while [ $# -gt 0 ]; do
  case "$1" in
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --comfy) COMFY_DIR="$2"; shift 2 ;;
    --with-upscaler) WITH_UPSCALER=1; shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

# 未指定时：优先复用已有 /content/local_models 软链，其次公共 NAS /mnt/data，最后挑最大分区
if [ -z "${MODEL_ROOT}" ]; then
  if [ -L /content/local_models ]; then
    MODEL_ROOT="$(readlink -f /content/local_models)"
  elif [ -d /mnt/data ]; then
    MODEL_ROOT="/mnt/data/h3_v3/models"
  else
    BEST_TMP="/tmp"; BEST_KB=0
    for cand in /tmp /root /var/tmp /; do
      [ -d "$cand" ] || continue
      avail_kb=$(df -Pk "$cand" | tail -1 | awk '{print $4}')
      if [ "${avail_kb:-0}" -gt "$BEST_KB" ]; then BEST_KB="$avail_kb"; BEST_TMP="$cand"; fi
    done
    MODEL_ROOT="${BEST_TMP%/}/h3_rocm_models"
  fi
fi
mkdir -p "${MODEL_ROOT}"/{diffusion_models,text_encoders,vae,upscale_models,loras}
mkdir -p "${WORK_DIR}/logs"
LOG="${WORK_DIR}/logs/ms_download.log"
: > "${LOG}"

need_bytes=0
for item in "${FILES[@]}"; do need_bytes=$((need_bytes + ${item##*|})); done
[ "$WITH_UPSCALER" = "1" ] && need_bytes=$((need_bytes + UP_BYTES))
avail_gb=$(( $(df -Pk "${MODEL_ROOT}" | tail -1 | awk '{print $4}') / 1024 / 1024 ))
echo "[ms] 目标目录: ${MODEL_ROOT}（分区剩余 ~${avail_gb} GiB，本次约需 $((need_bytes / 1024 / 1024 / 1024)) GiB）"

# 优先 modelscope 内网加速；缺失时 pip 安装（PyPI 镜像可用，不重装 torch）
if ! command -v modelscope >/dev/null 2>&1; then
  echo "[ms] 未检测到 modelscope CLI，pip 安装（不触碰预装 torch）..."
  pip install -q modelscope >> "${LOG}" 2>&1 || true
fi
USE_MS_CLI=1
command -v modelscope >/dev/null 2>&1 || USE_MS_CLI=0
[ "$USE_MS_CLI" = "1" ] || echo "[ms] modelscope CLI 不可用，回退 aria2/wget 直连 CDN（约 1.3MB/s）"

DL_FALLBACK=""
if [ "$USE_MS_CLI" = "0" ]; then
  if command -v aria2c >/dev/null 2>&1; then
    DL_FALLBACK="aria2c"
  elif command -v wget >/dev/null 2>&1; then
    DL_FALLBACK="wget"
  else
    echo "[ms] FATAL: modelscope / aria2c / wget 均不可用" >&2; exit 3
  fi
fi

file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

check_safetensors() {
  python3 - "$1" <<'PY'
import json, os, struct, sys
p = sys.argv[1]
size = os.path.getsize(p)
with open(p, "rb") as f:
    raw = f.read(8)
    if len(raw) != 8:
        raise SystemExit(f"[verify] 文件过短: {p}")
    n = struct.unpack("<Q", raw)[0]
    if not (0 < n < 200 * 1024 * 1024) or 8 + n > size:
        raise SystemExit(f"[verify] safetensors 头长度异常({n}): {p}")
    hdr = json.loads(f.read(n))
    if not isinstance(hdr, dict) or not hdr:
        raise SystemExit(f"[verify] safetensors 头不是有效 JSON 对象: {p}")
print(f"[verify] OK {p} ({size/1024**3:.2f} GiB, {len(hdr)} tensors)")
PY
}

download_one() {  # $1=repo $2=repo内路径 $3=本地绝对路径 $4=期望字节
  local repo="$1" rel="$2" dst="$3" expect="$4"
  local dir base url t0 t1 got
  dir="$(dirname "$dst")"; base="$(basename "$dst")"
  mkdir -p "$dir"
  if [ -f "$dst" ] && [ "$(file_size "$dst")" = "$expect" ]; then
    echo "[ms] 已存在且大小一致，跳过: $rel"
    check_safetensors "$dst"
    return 0
  fi
  echo "[ms] 开始下载: $rel（期望 $((expect / 1024 / 1024)) MiB）"
  t0=$(date +%s)
  if [ "$USE_MS_CLI" = "1" ]; then
    # modelscope 保留仓库内子路径：<local_dir>/<rel>
    modelscope download --model "$repo" "$rel" --local_dir "${MODEL_ROOT}" 2>&1 | tee -a "${LOG}"
  elif [ "$DL_FALLBACK" = "aria2c" ]; then
    url="${MS_BASE}/${repo}/repo?FilePath=${rel}"
    aria2c -c -x"${THREADS}" -s"${THREADS}" -k 1M --file-allocation=none \
      --max-redirect=10 --retry-wait=3 --max-tries=0 \
      --console-log-level=warn --summary-interval=30 --auto-file-renaming=false \
      -d "$dir" -o "$base" "$url" 2>&1 | tee -a "${LOG}"
  else
    url="${MS_BASE}/${repo}/repo?FilePath=${rel}"
    wget -c --tries=0 --timeout=30 --progress=dot:giga -O "$dst" "$url" 2>&1 | tee -a "${LOG}"
  fi
  # 目标文件名与仓库路径不一致时（如备用放大器），把落盘文件拷到训练脚本约定的路径
  if [ "$USE_MS_CLI" = "1" ] && [ "${MODEL_ROOT}/${rel}" != "$dst" ] && [ -f "${MODEL_ROOT}/${rel}" ]; then
    cp -f "${MODEL_ROOT}/${rel}" "$dst"
  fi
  t1=$(date +%s); got=$(file_size "$dst")
  if [ "$got" != "$expect" ]; then
    echo "[ms] FATAL: $rel 大小不符（期望 ${expect}，实际 ${got}）。重跑本脚本可断点续传。" >&2
    exit 4
  fi
  local secs=$((t1 - t0)); [ "$secs" -lt 1 ] && secs=1
  echo "[ms] 完成 $rel : $((got / 1024 / 1024)) MiB, ${secs}s, ~$((got / secs / 1024 / 1024)) MiB/s"
  check_safetensors "$dst"
}

for item in "${FILES[@]}"; do
  rel="${item%%|*}"; expect="${item##*|}"
  download_one "$REPO" "$rel" "${MODEL_ROOT}/${rel}" "$expect"
done

if [ "$WITH_UPSCALER" = "1" ]; then
  # 私有仓缺失时的官方备用 3D 潜空间放大器；沿用 setup 约定文件名，供 gen_teacher_native 直接调用
  download_one "$UP_REPO" "$UP_FILE" "${MODEL_ROOT}/upscale_models/${UP_DST_NAME}" "$UP_BYTES"
fi

# 记录本次选定的 ROCm 非-FP4 文本编码器名（教师/训练统一读取该文件）
printf '%s' "qwen3vl_32b_minimax_h3_bf16.safetensors" > "${WORK_DIR}/rocm_clip_name.txt"
echo "[ms] rocm_clip_name.txt = $(cat "${WORK_DIR}/rocm_clip_name.txt")"

# 软链到 ComfyUI/models（ComfyUI 已部署时）
if [ -d "${COMFY_DIR}" ]; then
  for sub in diffusion_models text_encoders vae upscale_models; do
    mkdir -p "${COMFY_DIR}/models/${sub}"
    for f in "${MODEL_ROOT}/${sub}"/*.safetensors; do
      [ -e "$f" ] && ln -sfn "$f" "${COMFY_DIR}/models/${sub}/$(basename "$f")"
    done
  done
  echo "[ms] 已软链至 ${COMFY_DIR}/models/{diffusion_models,text_encoders,vae,upscale_models}"
fi

echo "[ms] 全部就绪。日志: ${LOG}"
