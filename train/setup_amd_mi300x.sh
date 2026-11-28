#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-late
# 魔搭 (ModelScope) × AMD Instinct MI300X/MI308X (192GB ROCm 7.2) 一键环境初始化与极速拉模脚本
# 支持首日 5h 快速出片与次日/每日接力重开（持久化代码与断点存 /mnt/workspace，百 GB 只读底模存最大临时分区）

set -euo pipefail

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1

PERSIST_ROOT="${PERSIST_ROOT:-/mnt/workspace/h3_v3}"
WORK_DIR="${PERSIST_ROOT}/work"
COMFY_DIR="${PERSIST_ROOT}/ComfyUI"
HF_REPO_CACHE="${HF_REPO_CACHE:-coolwindtang/minimax-h3-teacher-cache}"

echo "=================================================================="
echo "[1/6] 探测 AMD ROCm / GPU 显存与挂载分区容量 ..."
echo "=================================================================="
df -h

python3 - <<'PY'
import sys, torch
print("PyTorch:", torch.__version__, "| HIP:", getattr(torch.version, "hip", None), "| CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "FATAL: torch.cuda.is_available() == False! 请确认已选择 AMD GPU 192G 实例。"
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024 ** 3)
arch = getattr(props, "gcnArchName", "")
print(f"GPU: {props.name} ({arch}) | VRAM: {vram_gb:.1f} GiB")
assert vram_gb >= 70.0, f"FATAL: 显存不足 ({vram_gb:.1f} GiB < 70 GiB)"
PY

# 自动挑选可用空间最大的本地临时盘存放 ~90GB 只读大模型（保护 /mnt/workspace 100GB 持久化配额不爆盘！）
BEST_TMP="/tmp"
BEST_KB=0
for cand in /tmp /root /var/tmp /; do
  if [ -d "$cand" ]; then
    avail_kb=$(df -Pk "$cand" | tail -1 | awk '{print $4}')
    if [ "${avail_kb:-0}" -gt "$BEST_KB" ]; then
      BEST_KB="$avail_kb"
      BEST_TMP="$cand"
    fi
  fi
done
MODEL_ROOT_EXPLICIT="${MODEL_ROOT:-}"
MODEL_ROOT="${MODEL_ROOT:-${BEST_TMP%/}/h3_rocm_models}"
# 公共 NAS（/mnt/data，跨实例共享）已部署好只读权重时直接复用，避免每次开机重下 ~96GB
SHARED_MODEL_ROOT="${SHARED_MODEL_ROOT:-/mnt/data/h3_v3/models}"
if [ -z "${MODEL_ROOT_EXPLICIT}" ] && [ -f "${SHARED_MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ] \
   && [ -f "${SHARED_MODEL_ROOT}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors" ]; then
  MODEL_ROOT="${SHARED_MODEL_ROOT}"
  echo "[disk] 检测到公共目录已就绪权重，直接复用: ${MODEL_ROOT}"
fi
AVAIL_GB=$((BEST_KB / 1024 / 1024))
echo "[disk] 选定只读大模型目录: ${MODEL_ROOT} (所在分区剩余 ~${AVAIL_GB} GiB)"
echo "[disk] 持久化断点与代码目录: ${PERSIST_ROOT} (跨日关机保留)"

mkdir -p "${WORK_DIR}"/{logs,teacher_cache,lora_run_v3}
mkdir -p "${MODEL_ROOT}"/{diffusion_models,text_encoders,vae,upscale_models,loras}
mkdir -p /content
ln -sfn "${WORK_DIR}" /content/work
ln -sfn "${MODEL_ROOT}" /content/local_models

echo "=================================================================="
echo "[2/6] 安装轻量纯 Python 依赖（严禁重装预装 ROCm PyTorch！）..."
echo "=================================================================="
pip install -q safetensors einops huggingface_hub hf_transfer modelscope zstandard pillow tqdm

# 自动探测最快 HuggingFace 端点（国内魔搭容器自动切 hf-mirror）
if [ -z "${HF_ENDPOINT:-}" ]; then
  if curl -fsSL --connect-timeout 4 https://huggingface.co/api/models/Comfy-Org/MiniMax-H3 >/dev/null 2>&1; then
    export HF_ENDPOINT="https://huggingface.co"
  else
    export HF_ENDPOINT="https://hf-mirror.com"
  fi
fi
echo "[net] HF_ENDPOINT=${HF_ENDPOINT}"

echo "=================================================================="
echo "[3/6] 部署持久化 ComfyUI + T8 节点 + 本仓库 comfyui-h3-seamkit ..."
echo "=================================================================="
# 魔搭实例直连 github.com 会超时挂起：优先走 gh-proxy / ghfast 镜像拉 tarball
GH_MIRRORS="${GH_MIRRORS:-https://gh-proxy.com/|https://ghfast.top/}"
fetch_repo_tarball() {
  # $1=owner/repo  $2=目标目录
  local slug="$1" dst="$2" tmp
  tmp="$(mktemp -d)"
  local ok=0
  local IFS='|'
  for m in ${GH_MIRRORS}; do
    rm -rf "${tmp:?}"/* 2>/dev/null || true
    echo "[git-mirror] 尝试 ${m}${slug} ..."
    if curl -fsSL --connect-timeout 8 --max-time 600 \
        -o "${tmp}/repo.tar.gz" "${m}https://github.com/${slug}/archive/refs/heads/master.tar.gz" \
       || curl -fsSL --connect-timeout 8 --max-time 600 \
        -o "${tmp}/repo.tar.gz" "${m}https://github.com/${slug}/archive/refs/heads/main.tar.gz"; then
      if [ "$(stat -c%s "${tmp}/repo.tar.gz" 2>/dev/null || echo 0)" -gt 100000 ]; then
        tar -xzf "${tmp}/repo.tar.gz" -C "${tmp}" && ok=1 && break
      fi
    fi
  done
  unset IFS
  if [ "$ok" != "1" ]; then
    echo "[git-mirror] 全部镜像失败，最后尝试直连 git clone ..."
    timeout 90 git clone --depth 1 "https://github.com/${slug}.git" "${dst}" || return 1
    rm -rf "${tmp}"; return 0
  fi
  mkdir -p "${dst}"
  cp -rf "${tmp}"/*/* "${dst}/" 2>/dev/null || true
  cp -rf "${tmp}"/*/.[!.]* "${dst}/" 2>/dev/null || true
  rm -rf "${tmp}"
  return 0
}

if [ ! -f "${COMFY_DIR}/main.py" ]; then
  fetch_repo_tarball "comfyanonymous/ComfyUI" "${COMFY_DIR}"
  # 过滤 torch 三件套，严禁把预装 ROCm PyTorch 替换成 CUDA 版
  grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "${COMFY_DIR}/requirements.txt" \
    > /tmp/comfy_req_rocm.txt || true
  pip install -q -r /tmp/comfy_req_rocm.txt
fi
if [ ! -f "${COMFY_DIR}/custom_nodes/comfyui-minimax-h3-audio-T8/__init__.py" ]; then
  fetch_repo_tarball "T8mars/comfyui-minimax-h3-audio-T8" \
    "${COMFY_DIR}/custom_nodes/comfyui-minimax-h3-audio-T8"
  if [ -f "${COMFY_DIR}/custom_nodes/comfyui-minimax-h3-audio-T8/requirements.txt" ]; then
    grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' \
      "${COMFY_DIR}/custom_nodes/comfyui-minimax-h3-audio-T8/requirements.txt" \
      > /tmp/t8_req_rocm.txt || true
    pip install -q -r /tmp/t8_req_rocm.txt || true
  fi
fi
ln -sfn "${COMFY_DIR}" /content/ComfyUI
mkdir -p "${COMFY_DIR}/input"

# 若当前脚本在外部目录执行，则自动同步本仓库到 ComfyUI/custom_nodes/comfyui-h3-seamkit
SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_NODE="${COMFY_DIR}/custom_nodes/comfyui-h3-seamkit"
if [ "${SCRIPT_REPO}" != "${TARGET_NODE}" ]; then
  mkdir -p "${TARGET_NODE}"
  cp -rf "${SCRIPT_REPO}"/* "${TARGET_NODE}/"
fi

echo "=================================================================="
echo "[4/6] 自动枚举并下载 ROCm 兼容版 BF16 底模 / 非-NVFP4 CLIP / Sol-VAE / Upscaler ..."
echo "=================================================================="
# 魔搭实例出网白名单：huggingface.co / hf-mirror 均不可达时，自动改走魔搭 HD mirror 仓（阿里 CDN）
MODEL_SOURCE="${H3_MODEL_SOURCE:-auto}"
HF_REACHABLE=0
if [ "${MODEL_SOURCE}" != "modelscope" ] && curl -fsS --connect-timeout 5 --max-time 8 \
     https://huggingface.co/api/models/Comfy-Org/MiniMax-H3 >/dev/null 2>&1; then
  HF_REACHABLE=1
fi
if [ "${HF_REACHABLE}" = "1" ]; then
MODEL_ROOT="${MODEL_ROOT}" WORK_DIR="${WORK_DIR}" python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

model_root = Path(os.environ["MODEL_ROOT"])
work_dir = Path(os.environ["WORK_DIR"])
repo = "Comfy-Org/MiniMax-H3"

files = list_repo_files(repo)
text_encs = [f for f in files if f.startswith("text_encoders/") and f.endswith(".safetensors") and "qwen3" in f.lower()]
# 严格排除 N 卡专属 nvfp4/fp4，按 bf16 > fp16 > fp8 > int8 优先级选择 ROCm 原生可读编码器
non_fp4 = [f for f in text_encs if "fp4" not in f.lower() and "awq" not in f.lower()]
candidates = non_fp4 if non_fp4 else text_encs

def _rank_clip(name: str) -> int:
    n = name.lower()
    if "bf16" in n:
        return 0
    if "fp16" in n:
        return 1
    if "fp8" in n:
        return 2
    if "int8" in n:
        return 3
    return 4

candidates.sort(key=_rank_clip)
if not candidates:
    raise SystemExit(f"未在 {repo} 找到 text_encoders 权重: {text_encs}")
chosen_clip = candidates[0]
clip_basename = Path(chosen_clip).name
(work_dir / "rocm_clip_name.txt").write_text(clip_basename, encoding="utf-8")
print(f"[model] 自动选定 ROCm 文本编码器: {chosen_clip}")

want_main = [
    (repo, "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"),
    (repo, chosen_clip),
    (repo, "vae/minimax_h3_sol_vae_fp16.safetensors"),
    (repo, "vae/minimax_h3_audio_vae_fp32.safetensors"),
]

# 查找锐度 3D upscaler（优先官方/社区仓）
upscaler_target = model_root / "upscale_models" / "h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"
for r_id, rel_path in want_main:
    dst = model_root / rel_path
    if dst.is_file() and dst.stat().st_size > 1024 * 1024:
        print(f"[skip] 已存在 {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
        continue
    print(f"[dl] 正在拉取 {r_id} :: {rel_path} ...")
    hf_hub_download(r_id, rel_path, local_dir=str(model_root))

if not upscaler_target.is_file():
    # 尝试从私有备份仓或公开仓拉取 3D 放大器
    token_file = work_dir / "hftoken.txt"
    tok = token_file.read_text(encoding="utf-8").strip() if token_file.is_file() else os.environ.get("HF_TOKEN")
    loaded_up = False
    if tok:
        for cand_up in (
            "models/h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors",
            "upscale_models/h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors",
        ):
            try:
                got = hf_hub_download(os.environ.get("HF_REPO_CACHE", "coolwindtang/minimax-h3-teacher-cache"), cand_up, token=tok)
                upscaler_target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(got, upscaler_target)
                loaded_up = True
                print(f"[dl] 从私有仓拉取 3D 放大器成功 -> {upscaler_target}")
                break
            except Exception:
                pass
    if not loaded_up:
        print("[dl] 拉取公开 3D 潜空间放大器备用 -> latent_upscaler_3d_conv_v1_bf16.safetensors")
        got = hf_hub_download(
            "LBH-123-AI/Minimax_h3_latent_Upscaler",
            "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        )
        import shutil
        upscaler_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(got, upscaler_target)
PY
else
  echo "[model] HuggingFace 不可达（或 H3_MODEL_SOURCE=modelscope），改走魔搭镜像仓 ${MS_REPO:-Comfy-Org/MiniMax-H3}"
  MODEL_ROOT="${MODEL_ROOT}" WORK_DIR="${WORK_DIR}" COMFY_DIR="${COMFY_DIR}" WITH_UPSCALER=1 \
    bash "${SCRIPT_REPO}/train/fetch_models_modelscope.sh" \
      --model-root "${MODEL_ROOT}" --work-dir "${WORK_DIR}" --comfy "${COMFY_DIR}"
fi

echo "=================================================================="
echo "[5/6] 软链模型至 ComfyUI/models 各子目录 ..."
echo "=================================================================="
for sub in diffusion_models text_encoders vae upscale_models loras; do
  mkdir -p "${COMFY_DIR}/models/${sub}"
  for f in "${MODEL_ROOT}/${sub}"/*.safetensors; do
    [ -e "$f" ] && ln -sfn "$f" "${COMFY_DIR}/models/${sub}/$(basename "$f")"
  done
done

# 若已配置 hftoken.txt，尝试自动从私有仓拉取角色参考图（deepseek.png / claude_sheet.png）与 v2 热启动母版
if [ -s "${WORK_DIR}/hftoken.txt" ]; then
  WORK_DIR="${WORK_DIR}" COMFY_DIR="${COMFY_DIR}" HF_REPO_CACHE="${HF_REPO_CACHE}" python3 - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

work = Path(os.environ["WORK_DIR"])
comfy = Path(os.environ["COMFY_DIR"])
repo = os.environ["HF_REPO_CACHE"]
tok = (work / "hftoken.txt").read_text(encoding="utf-8").strip()

for ref_name in ("deepseek.png", "claude_sheet.png"):
    dst = comfy / "input" / ref_name
    if dst.is_file():
        continue
    for prefix in ("refs/", "input/", ""):
        try:
            got = hf_hub_download(repo, f"{prefix}{ref_name}", token=tok)
            shutil.copy2(got, dst)
            print(f"[refs] 已拉取参考图 -> {dst}")
            break
        except Exception:
            pass
PY
fi

echo "=================================================================="
echo "[6/6] 执行 ROCm Preflight 体检 ..."
echo "=================================================================="
python3 "${TARGET_NODE}/train/preflight.py" --role teacher --min-vram-gb 70 --work-root "${WORK_DIR}" --model-root "${MODEL_ROOT}"

CLIP_NAME="$(cat "${WORK_DIR}/rocm_clip_name.txt")"
echo ""
echo "✅ AMD MI300X 环境初始化全部完成！"
echo "   - 检测到的 ROCm 文本编码器文件名: ${CLIP_NAME}"
echo "   - 请确认两张参考图已在位: ls -lh /content/ComfyUI/input/{deepseek.png,claude_sheet.png}"
echo "   - 请确认 HF Token 已写入: /content/work/hftoken.txt"
