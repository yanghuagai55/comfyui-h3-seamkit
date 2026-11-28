#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# H3-AnimeEngine v3.0 夜间通宵总控（魔搭 AMD MI300X / ROCm 7.2）
#
# 设计目标（幂等，可重复执行）：
#   Stage 0  清 NAS 配额 -> 同步代码 -> 解压 veo_cast -> 建 /content 软链 -> 开防回收心跳
#   Stage 1  Phase 2B Zero-Gather Triton 真机 A/B 门禁；通过用 triton_bs，异常无感回落 sdpa_mask
#   Stage 2  Veo 240 首批黄金教师；首条测速自动定 best-of-K；按片直推 teacher_cache_v3_veo/
#   Stage 3  208 层 v3.0 训练；每 50 步切片回传 candidates_v3/；NAS 滚动保留最近 2 份 step
#
# 启动示例（见交付说明）：
#   curl 拉取本脚本到 /tmp/v3boot 后 nohup bash /tmp/v3boot/run_v3_overnight.sh >> .../v3_console.log 2>&1 &
#
# 常用环境变量覆盖：
#   V3_HARD_STOP_MIN=400      全局硬停止（分钟，默认 6h40m）
#   V3_TEACHER_BUDGET_MIN=140 教师阶段预算（含首条测速）
#   V3_TRAIN_FRAMES=22,39,73  训练帧桶（73f OOM 时自动回落 22,39）
#   V3_TRAIN_STEPS=1100       目标总步数（跨夜接力递增）
#   V3_SKIP_CLEAN=1 / V3_SKIP_BENCH=1 / V3_SKIP_TEACHER=1

set -uo pipefail
umask 022

export PERSIST_ROOT="${PERSIST_ROOT:-/mnt/data/h3_v3}"
export WORK_DIR="${PERSIST_ROOT}/work"
export MODEL_ROOT="${PERSIST_ROOT}/models"
export CODE_DIR="${PERSIST_ROOT}/comfyui-h3-seamkit"
export COMFY_DIR="${COMFY_DIR:-/content/ComfyUI}"
export LOG_DIR="${PERSIST_ROOT}/logs"
export RUN_DIR="${RUN_DIR:-${WORK_DIR}/lora_run_v3_208}"
export EXPORT_DIR="${EXPORT_DIR:-${RUN_DIR}/export}"
export TEACHER_CACHE_NEW="${TEACHER_CACHE_NEW:-/tmp/v3_teacher_cache_veo}"
export TRAIN_CACHE="${TRAIN_CACHE:-/tmp/v3_train_cache}"
export TEACHER_RAW="${TEACHER_RAW:-/tmp/v3_veo_raw}"
export ZUP_TMP="${ZUP_TMP:-/tmp/v3_zup_tmp}"
export MS_REPO="${MS_REPO:-coolwind123/h3-teacher-cache}"
export MS_TOKEN_FILE="${MS_TOKEN_FILE:-${WORK_DIR}/mstoken.txt}"
export START_EPOCH="$(date +%s)"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/tmp/v3_ms_cache}"
export HF_HOME="${HF_HOME:-/tmp/v3_hf_cache}"
export PYTHONUNBUFFERED=1

HARD_STOP_MIN="${V3_HARD_STOP_MIN:-400}"
TEACHER_BUDGET_MIN="${V3_TEACHER_BUDGET_MIN:-140}"
TRAIN_FRAMES="${V3_TRAIN_FRAMES:-22,39,73}"
TRAIN_TARGET_STEPS="${V3_TRAIN_STEPS:-1100}"
TRAIN_LR="${V3_TRAIN_LR:-8e-5}"
ATTN_BACKEND="${V3_ATTN_BACKEND:-auto}"

MAIN_LOG="${LOG_DIR}/v3_overnight.log"
TRAIN_LOG="${LOG_DIR}/train_v3_overnight.log"
TEACHER_LOG="${LOG_DIR}/teacher_v3_veo.log"
BENCH_LOG="${LOG_DIR}/phase2b_bench_v3.log"
DONE_FILE="${LOG_DIR}/DONE_V3_OVERNIGHT.txt"
HELPER="/tmp/v3_ms.py"
WATCH_PID=""
UPLOAD_PID=""
HB_PID=""

mkdir -p "${LOG_DIR}" "${WORK_DIR}" "${EXPORT_DIR}" "${TEACHER_CACHE_NEW}" "${TRAIN_CACHE}" "${TEACHER_RAW}" "${ZUP_TMP}"
rm -f "${DONE_FILE}"

log() { echo "[v3 $(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"; }
elapsed_min() { echo $(( ( $(date +%s) - START_EPOCH ) / 60 )); }
remaining_min() { echo $(( HARD_STOP_MIN - $(elapsed_min) )); }

fatal() {
  log "[FATAL] $*"
  {
    echo "FINISHED_AT=$(date -Iseconds)"
    echo "ELAPSED_MIN=$(elapsed_min)"
    echo "[V3_ABORTED] $*"
  } | tee "${DONE_FILE}"
  exit 1
}

ms() { python3 "${HELPER}" "$@"; }
ms_status() { ms status "$1" "${2:-}" >> "${MAIN_LOG}" 2>&1 || true; }

cleanup_children() {
  [ -n "${HB_PID}" ] && kill "${HB_PID}" 2>/dev/null || true
  [ -n "${WATCH_PID}" ] && kill "${WATCH_PID}" 2>/dev/null || true
  [ -n "${UPLOAD_PID}" ] && kill "${UPLOAD_PID}" 2>/dev/null || true
  pkill -f "ComfyUI/main.py" 2>/dev/null || true
}
trap cleanup_children EXIT

# 防 1h 空闲回收心跳：本脚本存活期间每 5 分钟触碰一次
(
  while kill -0 $$ 2>/dev/null; do
    date -Iseconds > /mnt/workspace/.overnight_heartbeat 2>/dev/null || true
    curl -fsS --max-time 3 http://127.0.0.1:8888/ >/dev/null 2>&1 || true
    sleep 300
  done
) &
HB_PID=$!

log "================ H3 v3.0 OVERNIGHT START ================"
log "HARD_STOP_MIN=${HARD_STOP_MIN} TEACHER_BUDGET_MIN=${TEACHER_BUDGET_MIN} TRAIN_FRAMES=${TRAIN_FRAMES} STEPS=${TRAIN_TARGET_STEPS}"
log "PERSIST_ROOT=${PERSIST_ROOT} RUN_DIR=${RUN_DIR}"

# ============================================================================
# 内嵌 ModelScope 助手：下载/解压/分片同步/分片上传/里程碑切片/状态上报
# ============================================================================
cat > "${HELPER}" <<'PY'
#!/usr/bin/env python3
import hashlib, json, os, re, shutil, subprocess, sys, tarfile, time, urllib.parse, urllib.request
from pathlib import Path

REPO = os.environ.get("MS_REPO", "coolwind123/h3-teacher-cache")
TOKEN_FILE = os.environ.get("MS_TOKEN_FILE", "/mnt/data/h3_v3/work/mstoken.txt")
PERSIST_ROOT = Path(os.environ.get("PERSIST_ROOT", "/mnt/data/h3_v3"))
API = "https://www.modelscope.cn/api/v1/models"
VEO_PREFIX = "teacher_cache_v3_veo"


def _tok():
    return Path(TOKEN_FILE).read_text(encoding="utf-8").strip()


def _log(msg):
    print(f"[ms {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_bytes(remote, timeout=180):
    url = f"{API}/{REPO}/repo?Revision=master&FilePath={urllib.parse.quote(remote)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_tok()}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download(remote, dest, timeout=600):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/{REPO}/repo?Revision=master&FilePath={urllib.parse.quote(remote)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_tok()}"})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)
    _log(f"downloaded {remote} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _api():
    try:
        from modelscope_hub.api import HubApi
        return HubApi(token=_tok())
    except Exception:
        from modelscope.hub.api import HubApi
        hub = HubApi()
        hub.login(_tok())
        return hub


def _upload(local, remote, message=""):
    local = Path(local)
    hub = _api()
    for attempt in range(3):
        try:
            hub.upload_file(
                repo_id=REPO, repo_type="model", path_or_fileobj=str(local),
                path_in_repo=remote,
                commit_message=message or f"v3 overnight {local.name}",
            )
            _log(f"uploaded {local.name} -> {remote}")
            return 0
        except Exception as exc:
            _log(f"upload retry {attempt + 1}/3 {local.name}: {exc}")
            time.sleep(3)
    return 1


def _manifest(prefix):
    try:
        man = json.loads(_get_bytes(f"{prefix}/manifest.json", timeout=60))
    except Exception as exc:
        _log(f"manifest {prefix} unavailable: {exc}")
        man = {}
    man.setdefault("version", 1)
    man.setdefault("shards", [])
    man["count"] = sum(len(s.get("ids") or []) for s in man["shards"])
    return man


def _tar_zst(files, out):
    import zstandard
    out = Path(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    cctx = zstandard.ZstdCompressor(level=6, threads=4)
    with tmp.open("wb") as raw, cctx.stream_writer(raw, closefd=False) as zfh:
        with tarfile.open(fileobj=zfh, mode="w|") as tf:
            for p in files:
                tf.add(str(p), arcname=Path(p).name)
    tmp.replace(out)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_dl(remote, dest):
    _download(remote, dest)
    return 0


def cmd_extract_veo(tar_path, input_dir):
    dst = Path(input_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(dst, filter="data")
        except TypeError:
            tf.extractall(dst)
    n = len(list((dst / "veo_cast").glob("*.png")))
    _log(f"veo_cast extracted png={n} -> {dst / 'veo_cast'}")
    return 0 if n >= 24 else 5


def cmd_sync_teachers(cache, state):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    man = _manifest(VEO_PREFIX)
    done = set()
    for shard in man.get("shards", []):
        name = shard.get("name", "")
        ids = [str(x) for x in (shard.get("ids") or [])]
        if not name:
            continue
        if ids and all((cache / f"{i}.pt").is_file() for i in ids):
            done.update(ids)
            continue
        local = Path("/tmp/v3_ms_dl") / name
        try:
            _download(f"{VEO_PREFIX}/shards/{name}", local)
            import zstandard
            with local.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    for member in tf:
                        if not member.isfile() or not member.name.endswith(".pt"):
                            continue
                        target = cache / Path(member.name).name
                        fh = tf.extractfile(member)
                        if fh is None:
                            continue
                        tmp = target.with_suffix(".pt.tmp")
                        with tmp.open("wb") as out:
                            shutil.copyfileobj(fh, out, length=1 << 20)
                        tmp.replace(target)
            done.update(ids)
            _log(f"sync {name} +{len(ids)}")
        except Exception as exc:
            _log(f"sync {name} FAIL: {exc}")
        finally:
            try:
                local.unlink()
            except Exception:
                pass
    Path(state).write_text("\n".join(sorted(done)) + ("\n" if done else ""), encoding="utf-8")
    _log(f"sync complete cache_pt={len(list(cache.glob('*.pt')))} cloud_ids={len(done)}")
    return 0


def cmd_upload_shards(cache, state, batch=5):
    cache = Path(cache)
    state = Path(state)
    done = set(state.read_text(encoding="utf-8").split()) if state.is_file() else set()
    files = [p for p in sorted(cache.glob("v3veo_*.pt"))
             if p.is_file() and p.stat().st_size > 1024 and p.stem not in done]
    if not files:
        _log("no pending teacher .pt")
        return 0
    man = _manifest(VEO_PREFIX)
    idx = 0
    for sh in man.get("shards", []):
        m = re.search(r"shard_v3veo_(\d+)", sh.get("name", ""))
        if m:
            idx = max(idx, int(m.group(1)) + 1)
    tmp_tar = Path("/tmp/v3_teacher_shard.tar.zst")
    while files:
        chunk, files = files[:batch], files[batch:]
        name = f"shard_v3veo_{idx:03d}.tar.zst"
        try:
            _tar_zst(chunk, tmp_tar)
            if _upload(tmp_tar, f"{VEO_PREFIX}/shards/{name}",
                       f"teacher shard {name} ({len(chunk)} samples)") != 0:
                return 1
            digest = _sha256(tmp_tar)
            man = _manifest(VEO_PREFIX)
            man["shards"] = [s for s in man["shards"] if s.get("name") != name]
            man["shards"].append({
                "name": name, "ids": [p.stem for p in chunk],
                "bytes": tmp_tar.stat().st_size, "sha256": digest,
            })
            man["shards"].sort(key=lambda s: s.get("name", ""))
            man["count"] = sum(len(s.get("ids") or []) for s in man["shards"])
            man["updated"] = time.strftime("%F %T")
            mp = Path("/tmp/v3_manifest.json")
            mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
            if _upload(mp, f"{VEO_PREFIX}/manifest.json",
                       f"manifest +{name} total={man['count']}") != 0:
                return 1
            done.update(p.stem for p in chunk)
            state.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")
            _log(f"[UPLOADED_TEACHER] {name} n={len(chunk)} total={man['count']}")
            idx += 1
        except Exception as exc:
            _log(f"shard {name} FAIL: {exc}")
            return 1
        finally:
            tmp_tar.unlink(missing_ok=True)
    return 0


def cmd_upload(local, remote):
    return _upload(local, remote)


def cmd_status(stage, detail=""):
    run = Path(os.environ.get("RUN_DIR", "/mnt/data/h3_v3/work/lora_run_v3_208"))
    cache_new = Path(os.environ.get("TEACHER_CACHE_NEW", "/tmp/v3_teacher_cache_veo"))
    cache_train = Path(os.environ.get("TRAIN_CACHE", "/tmp/v3_train_cache"))
    last = {}
    mp = run / "logs" / "metrics.jsonl"
    if mp.is_file():
        try:
            lines = [x for x in mp.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if lines:
                last = json.loads(lines[-1])
        except Exception:
            pass
    steps = sorted(p for p in (run / "steps").glob("step_*.safetensors")
                   if not p.stem.endswith("_ema")) if (run / "steps").is_dir() else []
    payload = {
        "stage": stage,
        "detail": detail,
        "updated": time.strftime("%F %T"),
        "elapsed_min": int((time.time() - int(os.environ.get("START_EPOCH", "0"))) / 60),
        "latest_step": steps[-1].stem if steps else None,
        "metrics": last,
        "teacher_new_pt": len(list(cache_new.glob("v3veo_*.pt"))) if cache_new.is_dir() else 0,
        "train_pool_pt": len(list(cache_train.glob("*.pt"))) if cache_train.is_dir() else 0,
    }
    local = Path("/tmp/v3_status.json")
    local.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"status {stage}: {detail}")
    _upload(local, "candidates_v3/status_v3.json", f"status {stage}")
    return 0


def cmd_milestone(run_dir, export_dir, state):
    run = Path(run_dir)
    exp = Path(export_dir)
    exp.mkdir(parents=True, exist_ok=True)
    steps_dir = run / "steps"
    if not steps_dir.is_dir():
        return 0
    state_p = Path(state)
    done = set(state_p.read_text(encoding="utf-8").split()) if state_p.is_file() else set()
    code_dir = PERSIST_ROOT / "comfyui-h3-seamkit"

    def _step_id(path):
        m = re.search(r"step_(\d+)", path.name)
        return m.group(1) if m else ""

    def _slice(src, out):
        if out.is_file():
            return True
        _log(f"slice {src.name} -> {out.name}")
        rc = subprocess.call([
            sys.executable, str(code_dir / "train" / "slice_lora_stylekeep.py"),
            "--input", str(src), "--output", str(out), "--preset", "action_stylekeep",
        ])
        if rc != 0:
            _log(f"slice {src.name} FAIL rc={rc}")
            return False
        return True

    plain = sorted([p for p in steps_dir.glob("step_*.safetensors") if not p.stem.endswith("_ema")],
                   key=lambda p: p.stat().st_mtime)
    for p in plain:
        sid = _step_id(p)
        if not sid or p.stat().st_size < 1024:
            continue
        if sid not in done:
            out = exp / f"minimax_h3_sol_sharp_v3_overnight_s{sid}_action_stylekeep.safetensors"
            if _slice(p, out) and _upload(out, f"candidates_v3/{out.name}") == 0:
                metrics = run / "logs" / "metrics.jsonl"
                if metrics.is_file():
                    _upload(metrics, "candidates_v3/metrics_v3_overnight.jsonl")
                with state_p.open("a", encoding="utf-8") as fh:
                    fh.write(sid + "\n")
                done.add(sid)
                _log(f"[MILESTONE] s{sid} sliced+uploaded")
        ema_p = steps_dir / f"step_{sid}_ema.safetensors"
        if ema_p.is_file() and ema_p.stat().st_size >= 1024 and f"{sid}_ema" not in done:
            eout = exp / f"minimax_h3_sol_sharp_v3_overnight_s{sid}_ema_action_stylekeep.safetensors"
            if _slice(ema_p, eout) and _upload(eout, f"candidates_v3/{eout.name}") == 0:
                with state_p.open("a", encoding="utf-8") as fh:
                    fh.write(f"{sid}_ema\n")
                done.add(f"{sid}_ema")
                _log(f"[MILESTONE] s{sid} EMA sliced+uploaded")

    # 最新完整未切片权重：固定远端路径覆盖，任意时刻可跨实例无损热启
    if plain:
        latest = plain[-1]
        sid = _step_id(latest)
        full_state = Path("/tmp/v3_full_uploaded_step.txt")
        prev = full_state.read_text(encoding="utf-8").strip() if full_state.is_file() else ""
        if sid and prev != sid:
            if _upload(latest, "candidates_v3/v3_208L_latest_full.safetensors") == 0:
                full_state.write_text(sid + "\n", encoding="utf-8")
                _log(f"[FULL] v3_208L_latest_full <- {latest.name}")

    # NAS 滚动：只删已上传成功的旧文件，保留最近 2 份 step / ema / export
    plain_now = sorted([p for p in steps_dir.glob("step_*.safetensors") if not p.stem.endswith("_ema")],
                       key=lambda p: p.stat().st_mtime)
    ema_now = sorted(steps_dir.glob("step_*_ema.safetensors"), key=lambda p: p.stat().st_mtime)
    for p in plain_now[:-2]:
        if _step_id(p) in done:
            try:
                p.unlink()
                _log(f"prune {p.name}")
            except Exception:
                pass
    for p in ema_now[:-2]:
        if f"{_step_id(p)}_ema" in done:
            try:
                p.unlink()
                _log(f"prune {p.name}")
            except Exception:
                pass
    exports = sorted(exp.glob("*_action_stylekeep.safetensors"), key=lambda p: p.stat().st_mtime)
    for p in exports[:-2]:
        m = re.search(r"_s(\d+)(_ema)?_action_stylekeep\.safetensors$", p.name)
        key = (m.group(1) + ("_ema" if m.group(2) else "")) if m else ""
        if key and key in done:
            try:
                p.unlink()
                _log(f"prune {p.name}")
            except Exception:
                pass
    return 0


def main():
    if len(sys.argv) < 2:
        print("usage: v3_ms.py <dl|extract-veo|sync-teachers|upload-shards|upload|status|milestone> ...")
        return 2
    cmd = sys.argv[1]
    if cmd == "dl":
        return cmd_dl(sys.argv[2], sys.argv[3])
    if cmd == "extract-veo":
        return cmd_extract_veo(sys.argv[2], sys.argv[3])
    if cmd == "sync-teachers":
        return cmd_sync_teachers(sys.argv[2], sys.argv[3])
    if cmd == "upload-shards":
        batch = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        return cmd_upload_shards(sys.argv[2], sys.argv[3], batch)
    if cmd == "upload":
        return cmd_upload(sys.argv[2], sys.argv[3])
    if cmd == "status":
        return cmd_status(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    if cmd == "milestone":
        return cmd_milestone(sys.argv[2], sys.argv[3], sys.argv[4])
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
PY

[ -s "${MS_TOKEN_FILE}" ] || fatal "缺少魔搭令牌 ${MS_TOKEN_FILE}"
export MODELSCOPE_API_TOKEN="$(cat "${MS_TOKEN_FILE}")"

ms_status stage0 "清理 NAS + 同步代码"

# ============================================================================
# Stage 0：清理 NAS 配额 / 同步代码 / 软链 / 参考图
# ============================================================================
quota_probe() {
  dd if=/dev/zero of="${WORK_DIR}/.quota_probe" bs=1M count=64 status=none >/dev/null 2>&1
  local rc=$?
  rm -f "${WORK_DIR}/.quota_probe"
  return "${rc}"
}

cleanup_nas() {
  local before_mb after_mb
  before_mb=$(df -Pm /mnt/data 2>/dev/null | awk 'END{print $4}')
  log "[clean] NAS free before=${before_mb:-?}MB"

  # 旧 run 的 step/ema 先搬到 /tmp 归档（本机仍可查），再从 NAS 释放配额
  python3 - "${WORK_DIR}/lora_run_v3/steps" <<'PY' || true
import shutil, sys
from pathlib import Path
d = Path(sys.argv[1])
arch = Path("/tmp/v3_old_run_archive/steps")
arch.mkdir(parents=True, exist_ok=True)
if not d.is_dir():
    raise SystemExit(0)
for p in sorted(d.glob("step_*.safetensors"), key=lambda p: p.stat().st_mtime):
    try:
        dst = arch / p.name
        shutil.move(str(p), str(dst))
        print(f"[clean] archived {p.name} ({dst.stat().st_size/1e6:.0f} MB)", flush=True)
    except Exception as exc:
        print(f"[clean] archive fail {p.name}: {exc}", flush=True)
PY

  # 已回传魔搭的旧成品切片（candidates_v3/ 已有同名文件）
  mkdir -p /tmp/v3_old_run_archive/slices
  for f in "${WORK_DIR}/lora_run_v3/minimax_h3_sol_sharp_v3_"*_action_stylekeep.safetensors; do
    [ -e "$f" ] || continue
    mv -f "$f" /tmp/v3_old_run_archive/slices/ 2>/dev/null || rm -f "$f"
  done
  find "${WORK_DIR}/lora_run_v3" -maxdepth 1 -type f -name 'minimax_h3_sol_sharp_v3_*_action_stylekeep.safetensors' -delete 2>/dev/null || true

  # 临时 raw / 打包副本 / 下载暂存（严禁碰 models、assets/final、teacher_cache/*.pt）
  rm -rf "${WORK_DIR}/h3_native_raw" "${WORK_DIR}/h3_zup_tmp" "${WORK_DIR}/zup_tmp" \
         "${WORK_DIR}/shard_dl"* "${WORK_DIR}/v3_ms_dl" \
         "${WORK_DIR}/teacher_cache/shard_"*.tar.zst "${WORK_DIR}/teacher_cache/_manifest.json" \
         "${WORK_DIR}/"*.tar.zst "${PERSIST_ROOT}/assets/teacher_repo"
  find "${LOG_DIR}" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.log' \) -mtime +3 -delete 2>/dev/null || true
  find "${CODE_DIR}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

  # 旧 200 层续训包（Day2 step140）与新 v3.0 208 层目录不兼容；先归档到 /tmp 再释放 NAS
  if [ -s "${WORK_DIR}/lora_run_v3/resume/latest_state.pt" ]; then
    mkdir -p /tmp/v3_old_run_archive/resume
    mv -f "${WORK_DIR}/lora_run_v3/resume/latest_state.pt" /tmp/v3_old_run_archive/resume/ 2>/dev/null \
      || rm -f "${WORK_DIR}/lora_run_v3/resume/latest_state.pt"
    log "[clean] 旧 200 层续训包已归档至 /tmp/v3_old_run_archive/（NAS 已释放）"
  fi
  after_mb=$(df -Pm /mnt/data 2>/dev/null | awk 'END{print $4}')
  log "[clean] NAS free after=${after_mb:-?}MB"
  if quota_probe; then
    log "[clean] NAS 配额探针通过（64MB 可写）"
  else
    log "[WARN] NAS 用户配额仍满（64MB 写入失败）；若训练首个 checkpoint 失败，请手动清理旧产物"
  fi
}

if [ "${V3_SKIP_CLEAN:-0}" = "1" ]; then
  log "[clean] 跳过（V3_SKIP_CLEAN=1）"
elif pgrep -f "train_h3_lora.py" >/dev/null 2>&1; then
  log "[WARN] 检测到已有 train_h3_lora.py 进程，跳过 NAS 清理"
else
  cleanup_nas
fi

pkill -f "ComfyUI/main.py" 2>/dev/null || true
pkill -f "main.py --listen 127.0.0.1 --port 8188" 2>/dev/null || true
sleep 3

sync_code() {
  local mode="${V3_CODE_MODE:-auto}" ok=0 m cfg
  if [ "${mode}" != "bundle" ] && [ -d "${CODE_DIR}/.git" ]; then
    for m in "" "https://ghfast.top/https://github.com/" "https://gh-proxy.com/https://github.com/"; do
      cfg=()
      [ -n "${m}" ] && cfg=(-c "url.${m}https://github.com/.insteadOf=https://github.com/")
      if timeout 120 git -C "${CODE_DIR}" "${cfg[@]}" fetch --depth 1 origin main >>"${MAIN_LOG}" 2>&1; then
        if git -C "${CODE_DIR}" reset --hard FETCH_HEAD >>"${MAIN_LOG}" 2>&1; then ok=1; break; fi
      fi
    done
  fi
  if [ "${mode}" = "git" ] && [ "${ok}" -ne 1 ] && [ ! -d "${CODE_DIR}/.git" ]; then
    for m in "https://ghfast.top/https://github.com/" "https://github.com/" "https://gh-proxy.com/https://github.com/"; do
      rm -rf /tmp/v3_clone
      if timeout 300 git clone --depth 1 "${m}yanghuagai55/comfyui-h3-seamkit.git" /tmp/v3_clone >>"${MAIN_LOG}" 2>&1; then
        mkdir -p "${CODE_DIR}"
        cp -a /tmp/v3_clone/. "${CODE_DIR}/" && ok=1 && break
      fi
    done
  fi
  if [ "${ok}" -ne 1 ]; then
    log "[code] 使用魔搭代码包（V3_CODE_MODE=${mode}；git 克隆在魔搭常被限速）"
    ms dl "bundle/comfyui-h3-seamkit-train_v3_overnight.tar.gz" /tmp/v3boot/code.tar.gz || return 1
    mkdir -p "${CODE_DIR}"
    tar -xzf /tmp/v3boot/code.tar.gz -C "${CODE_DIR}" || return 1
  fi
  return 0
}

if ! sync_code; then fatal "代码同步失败（git + 魔搭代码包均不可用）"; fi
log "[code] HEAD=$(git -C "${CODE_DIR}" rev-parse --short HEAD 2>/dev/null || echo no-git) $(git -C "${CODE_DIR}" log -1 --pretty='%s' 2>/dev/null || true)"

for f in train/run_v3_overnight.sh train/train_h3_lora.py train/run_teacher.py train/gen_teacher_native.py \
         train/bench_phase2_mi300x.py train/triton_block_sparse_attn.py train/slice_lora_stylekeep.py \
         train/prompts_v3_veo_240.jsonl train/hf_shards.py train/pack_teachers.py; do
  [ -s "${CODE_DIR}/${f}" ] || fatal "代码文件缺失：${CODE_DIR}/${f}"
done

# /content 软链
mkdir -p /content
link_dir() {
  local target="$1" link="$2"
  if [ -L "${link}" ] || [ ! -e "${link}" ]; then
    ln -sfn "${target}" "${link}"
  else
    log "[link] ${link} 是真实目录，保留不覆盖"
  fi
}
link_dir "${PERSIST_ROOT}/ComfyUI" /content/ComfyUI
link_dir "${WORK_DIR}" /content/work
link_dir "${MODEL_ROOT}" /content/local_models

# ComfyUI 模型软链 + latent_upscale_models 修复（上次 z_up 缺失根因）
if [ -d "${COMFY_DIR}/models" ]; then
  for sub in diffusion_models text_encoders vae upscale_models loras; do
    mkdir -p "${COMFY_DIR}/models/${sub}"
    for f in "${MODEL_ROOT}/${sub}"/*.safetensors; do
      [ -e "$f" ] && ln -sfn "$f" "${COMFY_DIR}/models/${sub}/$(basename "$f")"
    done
  done
  mkdir -p "${COMFY_DIR}/models/latent_upscale_models"
  UP_FILE="${MODEL_ROOT}/upscale_models/h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors"
  [ -s "${UP_FILE}" ] && ln -sfn "${UP_FILE}" "${COMFY_DIR}/models/latent_upscale_models/$(basename "${UP_FILE}")"
fi

# ComfyUI 教师节点同步（不覆盖整个 custom_node，只更新教师依赖）
NODE_DIR="${COMFY_DIR}/custom_nodes/comfyui-h3-seamkit"
if [ ! -e "${NODE_DIR}" ]; then ln -sfn "${CODE_DIR}" "${NODE_DIR}"; fi
if [ -d "${NODE_DIR}/train" ]; then
  cp -f "${CODE_DIR}/train/teacher_capture.py" "${NODE_DIR}/train/" 2>/dev/null || true
  cp -f "${CODE_DIR}/train/dense_rope_patch.py" "${NODE_DIR}/train/" 2>/dev/null || true
fi

# 依赖自检（不重装 torch）
MISSING=$(python3 - <<'PY'
import importlib.util
need = ["torch", "safetensors", "zstandard", "modelscope"]
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))
PY
)
if [ -n "${MISSING}" ]; then
  case " ${MISSING} " in
    *" torch "*) fatal "torch 缺失；禁止自动重装 ROCm torch，请先恢复镜像环境" ;;
  esac
  log "[deps] 缺少 ${MISSING}，尝试 pip 安装（不触碰 torch）"
  python3 -m pip install -q ${MISSING} >>"${MAIN_LOG}" 2>&1 || log "[deps] pip 安装失败：${MISSING}"
fi
python3 - <<'PY' >>"${MAIN_LOG}" 2>&1
import importlib.util, torch
print(f"[deps] torch={torch.__version__} hip={getattr(torch.version,'hip',None)} cuda_avail={torch.cuda.is_available()}")
print(f"[deps] triton={importlib.util.find_spec('triton') is not None}")
PY

# ComfyUI 运行依赖（新实例系统盘会丢 pip 包：comfy-aimdo 等必须补齐）
if [ -d "${COMFY_DIR}" ] && [ -s "${COMFY_DIR}/requirements.txt" ]; then
  if ! ( cd "${COMFY_DIR}" && python3 -c "import comfy.model_management" >/dev/null 2>&1 ); then
    log "[deps] ComfyUI 依赖缺失，安装 requirements（过滤 torch）"
    grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "${COMFY_DIR}/requirements.txt" > /tmp/v3_req_filtered.txt 2>/dev/null || true
    if [ -s /tmp/v3_req_filtered.txt ]; then
      python3 -m pip install -q -r /tmp/v3_req_filtered.txt >>"${MAIN_LOG}" 2>&1 || log "[deps] ComfyUI requirements 安装失败"
    fi
  fi
fi

# veo_cast 参考图
VEO_PNG=$(ls "${COMFY_DIR}/input/veo_cast"/*.png 2>/dev/null | wc -l | tr -d ' ')
if [ "${VEO_PNG}" -lt 24 ]; then
  log "[veo] 本地仅 ${VEO_PNG} 张，开始从魔搭拉取 assets/veo_cast.tar.gz"
  if ms dl "assets/veo_cast.tar.gz" /tmp/v3boot/veo_cast.tar.gz >>"${MAIN_LOG}" 2>&1; then
    ms extract-veo /tmp/v3boot/veo_cast.tar.gz "${COMFY_DIR}/input" >>"${MAIN_LOG}" 2>&1 || log "[veo] 解压失败"
  else
    log "[WARN] veo_cast 下载失败，教师阶段将跳过 ref2va"
  fi
fi
VEO_PNG=$(ls "${COMFY_DIR}/input/veo_cast"/*.png 2>/dev/null | wc -l | tr -d ' ')
log "[veo] ready png=${VEO_PNG}"

ms_status stage0_done "代码=$(git -C "${CODE_DIR}" rev-parse --short HEAD 2>/dev/null || echo no-git) veo_png=${VEO_PNG} nas_free=$(df -Pm /mnt/data | awk 'END{print $4}')MB"

# ============================================================================
# Stage 1：Phase 2B Triton 真机门禁
# ============================================================================
if [ "${V3_SKIP_BENCH:-0}" = "1" ]; then
  ATTN_BACKEND="${V3_ATTN_BACKEND:-sdpa_mask}"
  log "[stage1] 跳过（V3_SKIP_BENCH=1），backend=${ATTN_BACKEND}"
else
  log "=== STAGE 1: Phase 2B Zero-Gather Triton 真机 A/B 门禁 ==="
  ms_status stage1 "Phase2B bench running"
  rm -f /tmp/phase2_v3_bench.json
  timeout -s TERM 1500 python3 "${CODE_DIR}/train/bench_phase2_mi300x.py" \
    --seq-len 26624 --base-steps 2 --p2-steps 3 \
    --cache "${WORK_DIR}/teacher_cache" \
    --base "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" \
    --out-json /tmp/phase2_v3_bench.json --upload-ms >>"${BENCH_LOG}" 2>&1
  BENCH_RC=$?
  SPEED="n/a"
  GATE_OK=0
  if [ "${BENCH_RC}" -eq 0 ] && [ -s /tmp/phase2_v3_bench.json ]; then
    if python3 - /tmp/phase2_v3_bench.json <<'PY'
import json, math, sys
try:
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    s = r["selftest"]
    e = r["e2e_mi300x"]
    p = e["phase2"]
    vals = [float(s.get(k, 0.0)) for k in ("cos_out", "cos_dq", "cos_dk", "cos_dv")]
    ok = (
        all(math.isfinite(v) and v >= 0.999 for v in vals)
        and p.get("grad_alive") == "416/416"
        and float(e.get("e2e_speedup", 0.0)) >= 1.0
        and float(p.get("peak_vram_gb", 9999.0)) <= 150.0
    )
    print(
        f"[gate] cos_out={s.get('cos_out')} cos_dq={s.get('cos_dq')} "
        f"grads={p.get('grad_alive')} speedup={e.get('e2e_speedup')} "
        f"vram={p.get('peak_vram_gb')}GB -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    sys.exit(0 if ok else 1)
except Exception as exc:
    print(f"[gate] parse/verify fail: {exc}", flush=True)
    sys.exit(1)
PY
    then GATE_OK=1; fi
    SPEED=$(python3 -c "import json;print(json.load(open('/tmp/phase2_v3_bench.json'))['e2e_mi300x']['e2e_speedup'])" 2>/dev/null || echo "n/a")
  fi
  if [ "${GATE_OK}" = "1" ]; then
    ATTN_BACKEND="triton_bs"
  else
    ATTN_BACKEND="sdpa_mask"
  fi
  log "[stage1] rc=${BENCH_RC} gate=${GATE_OK} backend=${ATTN_BACKEND} (speedup=${SPEED})"
  ms_status stage1_done "backend=${ATTN_BACKEND} speedup=${SPEED} rc=${BENCH_RC} gate=${GATE_OK}"
fi

# ============================================================================
# Stage 2：Veo 240 首批黄金教师
# ============================================================================
TEACHER_ENABLED=1
if [ "${V3_SKIP_TEACHER:-0}" = "1" ]; then
  TEACHER_ENABLED=0
  log "[stage2] 跳过（V3_SKIP_TEACHER=1）"
fi
[ -s "${COMFY_DIR}/main.py" ] || { TEACHER_ENABLED=0; log "[stage2] ComfyUI 缺失，跳过教师"; }
[ -s "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ] || { TEACHER_ENABLED=0; log "[stage2] 主模型缺失，跳过教师"; }
[ "${VEO_PNG}" -ge 24 ] || { TEACHER_ENABLED=0; log "[stage2] veo_cast 不足 24 张，跳过教师"; }

CLIP_NAME="$(cat "${WORK_DIR}/rocm_clip_name.txt" 2>/dev/null || true)"
[ -n "${CLIP_NAME}" ] || CLIP_NAME="qwen3vl_32b_minimax_h3_bf16.safetensors"
[ -s "${MODEL_ROOT}/text_encoders/${CLIP_NAME}" ] || { TEACHER_ENABLED=0; log "[stage2] CLIP 缺失 ${CLIP_NAME}，跳过教师"; }

ZUP_MODE="hybrid"
if [ ! -s "${MODEL_ROOT}/upscale_models/h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors" ]; then
  ZUP_MODE="downsample"
  log "[stage2] 锐度放大器缺失，zup-mode 回落 downsample"
fi

start_comfy() {
  if curl -fsS --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then return 0; fi
  ( cd "${COMFY_DIR}" && nohup python3 main.py --listen 127.0.0.1 --port 8188 --bf16-unet --disable-auto-launch >>/tmp/v3_comfyui.log 2>&1 & )
  local i
  for i in $(seq 1 100); do
    if curl -fsS --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then return 0; fi
    sleep 3
  done
  return 1
}

if [ "${TEACHER_ENABLED}" = "1" ]; then
  log "=== STAGE 2: Veo 240 首批黄金教师（预算 ${TEACHER_BUDGET_MIN}m，zup=${ZUP_MODE}） ==="
  ms_status stage2 "ComfyUI starting"
  if start_comfy; then
    log "[stage2] ComfyUI ready"
    ms sync-teachers "${TEACHER_CACHE_NEW}" /tmp/v3_veo_uploaded.txt >>"${TEACHER_LOG}" 2>&1 || log "[stage2] 云端教师同步失败，继续本地生成"

    TEACHER_ARGS=(--prompts "${CODE_DIR}/train/prompts_v3_veo_240.jsonl"
      --out "${TEACHER_RAW}" --cache "${TEACHER_CACHE_NEW}"
      --shard-size 1 --timeout 2400 --steps 28 --cfg 5.5
      --zup-mode "${ZUP_MODE}" --zup-tmp-dir "${ZUP_TMP}"
      --min-motion-lapvar 75.0 --max-qa-retries 2
      --clip "${CLIP_NAME}" --api http://127.0.0.1:8188 --comfy "${COMFY_DIR}" --repo "")

    # 首条测速，自动定 best-of-K
    rm -f "${TEACHER_RAW}/teacher_summary.json"
    log "[stage2] 首条测速（best-of-1，硬上限 25m）"
    timeout -s TERM 1500 python3 "${CODE_DIR}/train/run_teacher.py" "${TEACHER_ARGS[@]}" \
      --limit 1 --best-of-k 1 --budget-min 25 >>"${TEACHER_LOG}" 2>&1
    PROBE_RC=$?
    P_OK=0
    P_MIN=9999
    read -r P_OK P_MIN < <(python3 - "${TEACHER_RAW}/teacher_summary.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    d = {}
print(int(d.get("ok") or 0), float(d.get("elapsed_min") or 9999))
PY
)
    P_MIN_I=${P_MIN%.*}
    [ "${PROBE_RC}" -ne 0 ] && P_MIN_I=25
    K=1
    if [ "${PROBE_RC}" -eq 0 ] && [ "${P_OK:-0}" -ge 1 ] && [ "${P_MIN_I:-9999}" -le 13 ]; then
      K=2
    elif [ "${PROBE_RC}" -eq 0 ] && [ "${P_OK:-0}" -eq 0 ] && [ "${P_MIN_I:-9999}" -le 1 ]; then
      K=2   # 首条已在 cache 中，说明链路可用，直接按质量优先
    fi
    if [ "${PROBE_RC}" -ne 0 ]; then
      curl -fsS -X POST --max-time 10 http://127.0.0.1:8188/interrupt >/dev/null 2>&1 || true
    fi
    MAIN_BUDGET=$(( TEACHER_BUDGET_MIN - P_MIN_I ))
    [ "${MAIN_BUDGET}" -lt 40 ] && MAIN_BUDGET=40
    LEFT=$(( $(remaining_min) - 55 ))
    [ "${MAIN_BUDGET}" -gt "${LEFT}" ] && MAIN_BUDGET="${LEFT}"
    [ "${MAIN_BUDGET}" -lt 25 ] && MAIN_BUDGET=25
    log "[stage2] probe rc=${PROBE_RC} ok=${P_OK} min=${P_MIN_I} -> best-of-k=${K} budget=${MAIN_BUDGET}m"
    ms_status stage2 "teacher probe ok=${P_OK} t=${P_MIN_I}m k=${K} budget=${MAIN_BUDGET}m"

    python3 "${CODE_DIR}/train/run_teacher.py" "${TEACHER_ARGS[@]}" \
      --best-of-k "${K}" --budget-min "${MAIN_BUDGET}" >>"${TEACHER_LOG}" 2>&1 &
    TEACHER_PID=$!
    (
      n=0
      while kill -0 "${TEACHER_PID}" 2>/dev/null; do
        python3 "${HELPER}" upload-shards "${TEACHER_CACHE_NEW}" /tmp/v3_veo_uploaded.txt 5 >>"${TEACHER_LOG}" 2>&1 || true
        n=$(( n + 1 ))
        if [ $(( n % 10 )) -eq 0 ]; then
          ms_status stage2 "teacher running cached=$(ls "${TEACHER_CACHE_NEW}"/v3veo_*.pt 2>/dev/null | wc -l) budget_left=${MAIN_BUDGET}m"
        fi
        sleep 60
      done
      python3 "${HELPER}" upload-shards "${TEACHER_CACHE_NEW}" /tmp/v3_veo_uploaded.txt 5 >>"${TEACHER_LOG}" 2>&1 || true
    ) &
    UPLOAD_PID=$!
    wait "${TEACHER_PID}"; TEACHER_RC=$?
    wait "${UPLOAD_PID}"; UPLOAD_PID=""
    log "[stage2] run_teacher rc=${TEACHER_RC} cached=$(ls "${TEACHER_CACHE_NEW}"/v3veo_*.pt 2>/dev/null | wc -l)"
  else
    log "[WARN] ComfyUI 启动失败，跳过教师阶段"
  fi
  pkill -f "ComfyUI/main.py" 2>/dev/null || true
  pkill -f "main.py --listen 127.0.0.1 --port 8188" 2>/dev/null || true
  sleep 6
  rm -rf "${TEACHER_RAW}" "${ZUP_TMP}"
  VEO_NEW=$(ls "${TEACHER_CACHE_NEW}"/v3veo_*.pt 2>/dev/null | wc -l | tr -d ' ')
  log "[stage2] done new_teacher_pt=${VEO_NEW}"
  ms_status stage2_done "new_teacher_pt=${VEO_NEW}"
else
  log "[stage2] 教师阶段未启用，仅使用历史 teacher_cache"
fi

# ============================================================================
# Stage 3：208 层 v3.0 长程训练 + 里程碑切片回传
# ============================================================================
log "=== STAGE 3: 208 层 v3.0 训练 ==="
mkdir -p "${TRAIN_CACHE}" "${RUN_DIR}/steps" "${RUN_DIR}/resume" "${RUN_DIR}/logs" "${EXPORT_DIR}"

# 组装训练池：历史 87 条 + 新 veo 教师
for f in "${TEACHER_CACHE_NEW}"/v3veo_*.pt; do
  [ -e "$f" ] || continue
  ln -f "$f" "${TRAIN_CACHE}/$(basename "$f")" 2>/dev/null || cp -f "$f" "${TRAIN_CACHE}/"
done
for f in "${WORK_DIR}/teacher_cache"/*.pt; do
  [ -e "$f" ] || continue
  [ -e "${TRAIN_CACHE}/$(basename "$f")" ] && continue
  ln -sf "$f" "${TRAIN_CACHE}/$(basename "$f")" 2>/dev/null || cp -f "$f" "${TRAIN_CACHE}/"
done
POOL_N=$(ls "${TRAIN_CACHE}"/*.pt 2>/dev/null | wc -l | tr -d ' ')
log "[train] pool=${POOL_N} pt  (veo_new=${VEO_NEW:-0})"
[ "${POOL_N}" -ge 4 ] || fatal "训练池样本不足（${POOL_N}）"

# 热启动优先级：V3_INIT_LORA > Day2 step_000140（/tmp 归档 / NAS / 魔搭）> v2 final
choose_init() {
  local c
  INIT_LORA=""
  if [ -n "${V3_INIT_LORA:-}" ]; then INIT_LORA="${V3_INIT_LORA}"; return; fi
  for c in "/tmp/v3_old_run_archive/steps/step_000140.safetensors" \
           "${WORK_DIR}/lora_run_v3/steps/step_000140.safetensors"; do
    if [ -s "$c" ]; then INIT_LORA="$c"; return; fi
  done
  if [ ! -s "/tmp/v3_assets/step_000140.safetensors" ]; then
    log "[train] 本地无 step_000140，先尝试魔搭最新完整 v3.0 权重"
    ms dl "candidates_v3/v3_208L_latest_full.safetensors" "/tmp/v3_assets/v3_208L_latest_full.safetensors" >>"${MAIN_LOG}" 2>&1 || true
  fi
  if [ -s "/tmp/v3_assets/v3_208L_latest_full.safetensors" ]; then
    INIT_LORA="/tmp/v3_assets/v3_208L_latest_full.safetensors"
    return
  fi
  if [ ! -s "/tmp/v3_assets/step_000140.safetensors" ]; then
    log "[train] 无最新完整权重，回退魔搭 candidates_v3/step_000140.safetensors"
    ms dl "candidates_v3/step_000140.safetensors" "/tmp/v3_assets/step_000140.safetensors" >>"${MAIN_LOG}" 2>&1 || true
  fi
  if [ -s "/tmp/v3_assets/step_000140.safetensors" ]; then
    INIT_LORA="/tmp/v3_assets/step_000140.safetensors"
    return
  fi
  for c in "${PERSIST_ROOT}/assets/final/minimax_h3_sol_sharp_v2_final.safetensors" \
           "/tmp/v3_assets/minimax_h3_sol_sharp_v2_final.safetensors"; do
    if [ -s "$c" ]; then INIT_LORA="$c"; return; fi
  done
  log "[train] step_000140 不可用，回退 v2 final 热启动"
  ms dl "final/minimax_h3_sol_sharp_v2_final.safetensors" "/tmp/v3_assets/minimax_h3_sol_sharp_v2_final.safetensors" >>"${MAIN_LOG}" 2>&1 || true
  INIT_LORA="/tmp/v3_assets/minimax_h3_sol_sharp_v2_final.safetensors"
}

if [ -s "${RUN_DIR}/resume/latest_state.pt" ]; then
  INIT_LORA="${V3_INIT_LORA:-}"
  log "[train] 检测到本地续训包，将优先 --auto-resume（init 不生效）"
else
  choose_init
fi
if [ ! -s "${RUN_DIR}/resume/latest_state.pt" ] && [ ! -s "${INIT_LORA}" ]; then
  fatal "缺少热启动权重：${INIT_LORA}（且无本地续训包）"
fi
log "[train] init=${INIT_LORA:-none} resume=$([ -s "${RUN_DIR}/resume/latest_state.pt" ] && echo yes || echo no)"
echo "rank=64/32/32 refiner=1 frames=${TRAIN_FRAMES} backend=${ATTN_BACKEND} lr=${TRAIN_LR} steps=${TRAIN_TARGET_STEPS}" > "${RUN_DIR}/.v3_config"

TRAIN_CMD=(python3 "${CODE_DIR}/train/train_h3_lora.py"
  --comfy "${COMFY_DIR}"
  --base "${MODEL_ROOT}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"
  --cache "${TRAIN_CACHE}" --out "${RUN_DIR}"
  --phase A --frames "${TRAIN_FRAMES}" --steps "${TRAIN_TARGET_STEPS}"
  --rank-attn 64 --rank-mlp 32 --train-refiner --rank-refiner 32
  --vel-t-weight 0.20 --orth-weight 0.02 --ema-decay 0.995
  --lf-base-anchor --lf-kernel 5 --motion-gate-tau 0.75 --grad-xy-weight 0.15
  --action-boost 3.0 --optim adamw --lr "${TRAIN_LR}" --grad-clamp 10
  --save-every 50 --log-every 5
  --attn-backend "${ATTN_BACKEND}" --block-size 64
  --auto-resume)
[ -s "${INIT_LORA}" ] && TRAIN_CMD+=(--init-lora "${INIT_LORA}")

WATCH_STOP="${RUN_DIR}/.v3_watch_stop"
rm -f "${WATCH_STOP}" /tmp/v3_milestones_done.txt
(
  n=0
  while [ ! -f "${WATCH_STOP}" ]; do
    python3 "${HELPER}" milestone "${RUN_DIR}" "${EXPORT_DIR}" /tmp/v3_milestones_done.txt >>"${TRAIN_LOG}" 2>&1 || true
    n=$(( n + 1 ))
    if [ $(( n % 20 )) -eq 0 ]; then
      ms_status train "running latest=$(ls "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | grep -v _ema | sort | tail -1 | xargs -r basename)" || true
    fi
    sleep 30
  done
) &
WATCH_PID=$!

TRAIN_TIMEOUT_MIN=$(( $(remaining_min) - 25 ))
[ "${TRAIN_TIMEOUT_MIN}" -lt 20 ] && TRAIN_TIMEOUT_MIN=20
log "[train] timeout=${TRAIN_TIMEOUT_MIN}m backend=${ATTN_BACKEND} frames=${TRAIN_FRAMES}"
ms_status train "started timeout=${TRAIN_TIMEOUT_MIN}m backend=${ATTN_BACKEND} pool=${POOL_N}"

cd "${CODE_DIR}" || fatal "无法进入 ${CODE_DIR}"
touch "${TRAIN_LOG}"
TRAIN_LOG_OFFSET=$(wc -l < "${TRAIN_LOG}" | tr -d ' ')
timeout -s TERM "${TRAIN_TIMEOUT_MIN}m" "${TRAIN_CMD[@]}" >>"${TRAIN_LOG}" 2>&1
TRAIN_RC=$?
log "[train] rc=${TRAIN_RC} latest=$(ls "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | grep -v _ema | sort | tail -1 || true)"

# 73f 显存超限自动回落（只重试一次）
if printf '%s' "${TRAIN_FRAMES}" | grep -q "73" \
   && tail -n +$(( TRAIN_LOG_OFFSET + 1 )) "${TRAIN_LOG}" | grep -qiE "out of memory|OutOfMemoryError|HIP out of memory"; then
  log "[WARN] 检测到 73f OOM，回落 --frames 22,39 重试"
  TRAIN_FRAMES="22,39"
  # 重建 frames 参数
  NEW_CMD=()
  skip=0
  for a in "${TRAIN_CMD[@]}"; do
    if [ "${skip}" = "1" ]; then skip=0; continue; fi
    if [ "${a}" = "--frames" ]; then NEW_CMD+=("--frames" "22,39"); skip=1; continue; fi
    NEW_CMD+=("${a}")
  done
  TRAIN_CMD=("${NEW_CMD[@]}")
  TRAIN_TIMEOUT_MIN=$(( $(remaining_min) - 20 ))
  [ "${TRAIN_TIMEOUT_MIN}" -lt 20 ] && TRAIN_TIMEOUT_MIN=20
  log "[train-retry] frames=22,39 timeout=${TRAIN_TIMEOUT_MIN}m"
  ms_status train "retry no-73 frames timeout=${TRAIN_TIMEOUT_MIN}m"
  timeout -s TERM "${TRAIN_TIMEOUT_MIN}m" "${TRAIN_CMD[@]}" >>"${TRAIN_LOG}" 2>&1
  TRAIN_RC=$?
  log "[train-retry] rc=${TRAIN_RC}"
fi

touch "${WATCH_STOP}"
wait "${WATCH_PID}" 2>/dev/null || true
WATCH_PID=""
python3 "${HELPER}" milestone "${RUN_DIR}" "${EXPORT_DIR}" /tmp/v3_milestones_done.txt >>"${TRAIN_LOG}" 2>&1 || true

LATEST_STEP=$(ls -1 "${RUN_DIR}"/steps/step_*.safetensors 2>/dev/null | grep -v _ema | sort | tail -1)
if [ -z "${LATEST_STEP}" ] || [ ! -s "${LATEST_STEP}" ]; then
  log "[FAIL] 未找到有效 step 权重（训练未到首个 save-every=50 或启动失败）"
  ms_status failed "no checkpoint"
  {
    echo "FINISHED_AT=$(date -Iseconds)"
    echo "ELAPSED_MIN=$(elapsed_min)"
    echo "[V3_ABORTED_NO_CKPT] train_rc=${TRAIN_RC}"
    tail -n 40 "${TRAIN_LOG}"
  } | tee "${DONE_FILE}"
  exit 4
fi

LATEST_NAME=$(basename "${LATEST_STEP}")
LATEST_SID=$(printf '%s' "${LATEST_NAME}" | sed -n 's/^step_\([0-9]*\)\.safetensors$/\1/p')
SLICE_FILE="${EXPORT_DIR}/minimax_h3_sol_sharp_v3_overnight_s${LATEST_SID}_action_stylekeep.safetensors"
log "[final] latest=${LATEST_NAME} slice=${SLICE_FILE}"

{
  echo "FINISHED_AT=$(date -Iseconds)"
  echo "ELAPSED_MIN=$(elapsed_min)"
  echo "TRAIN_RC=${TRAIN_RC}"
  echo "ATTN_BACKEND=${ATTN_BACKEND}"
  echo "TRAIN_FRAMES=${TRAIN_FRAMES}"
  echo "POOL_SAMPLES=${POOL_N}"
  echo "NEW_VEO_TEACHERS=${VEO_NEW:-0}"
  echo "LATEST_STEP=${LATEST_NAME}"
  echo "LATEST_SLICE=${SLICE_FILE}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "--- exports ---"
  ls -lh "${EXPORT_DIR}" 2>/dev/null || true
  echo "--- steps ---"
  ls -lh "${RUN_DIR}/steps" 2>/dev/null || true
  echo "--- resume ---"
  ls -lh "${RUN_DIR}/resume" 2>/dev/null || true
  echo "[ALL_V3_OVERNIGHT_DONE]"
} | tee "${DONE_FILE}"

ms_status done "latest=${LATEST_NAME} rc=${TRAIN_RC} backend=${ATTN_BACKEND}"
log "[ALL_V3_OVERNIGHT_DONE] 训练与回传完成，请核对 candidates_v3/ 后停止实例"
