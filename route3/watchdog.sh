#!/usr/bin/env bash
# Route 3 独立系统级防空烧看门狗（不依赖 DeepSeek / SSH 在线）
# 三重硬熔断：
#   1. 总墙钟时间上限（默认 170 分钟软停，180 分钟硬杀；3 小时预算）
#   2. GPU 连续 15 分钟空转 (<5%) 且日志无新增字节 -> 判定僵死立即强杀
#   3. 持久化盘余量 < 10GB -> 写 LOW_DISK_FLAG
set -u

WORK_ROOT="${WORK_ROOT:-/content/h3_route3}"
DRIVE_SYNC_DIR="${DRIVE_SYNC_DIR:-${DRIVE_BACKUP:-/content/drive/MyDrive/h3_assets/route3_runs}}"
SOFT_LIMIT_MIN="${SOFT_LIMIT_MIN:-165}"
HARD_LIMIT_MIN="${HARD_LIMIT_MIN:-178}"
IDLE_LIMIT_MIN="${IDLE_LIMIT_MIN:-15}"
LOG_FILE="${WORK_ROOT}/logs/pipeline.log"
AUTO_SHUTDOWN_CMD="${AUTO_SHUTDOWN_CMD:-}"

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/artifacts"
START_TS=$(date +%s)
IDLE_STREAK=0
LAST_LOG_SIZE=0

log() {
    echo "[watchdog $(date -u +%H:%M:%S)] $*"
}

sync_artifacts() {
    if [ -d "$(dirname "${DRIVE_SYNC_DIR}")" ]; then
        mkdir -p "${DRIVE_SYNC_DIR}"
        rsync -av --include='*/' --include='*.json' --include='*.csv' --include='*.ok' \
              --include='*.png' --include='*.md' --include='*.pt' --include='*.mp4' \
              --exclude='*' "${WORK_ROOT}/artifacts/" "${DRIVE_SYNC_DIR}/" >/dev/null 2>&1 || true
        log "artifacts synced -> ${DRIVE_SYNC_DIR}"
    fi
}

kill_pipeline() {
    local reason="$1"
    log "ABORT TRIGGERED: ${reason}"
    {
        echo "# Route 3 Watchdog Abort Report"
        echo "- Time (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "- Reason: ${reason}"
        echo "- Elapsed min: $(( ($(date +%s) - START_TS) / 60 ))"
        echo ""
        echo "## nvidia-smi"
        nvidia-smi 2>&1 || true
        echo ""
        echo "## Last 80 lines of pipeline.log"
        tail -n 80 "${LOG_FILE}" 2>/dev/null || true
    } > "${WORK_ROOT}/artifacts/ABORT_REPORT.md"
    pkill -f "route3/(stage0_oracle|stage1_ablate|run_pipeline)" 2>/dev/null || true
    sleep 5
    pkill -9 -f "route3/(stage0_oracle|stage1_ablate|run_pipeline)" 2>/dev/null || true
    sync_artifacts
    if [ -n "${AUTO_SHUTDOWN_CMD}" ]; then
        log "Running AUTO_SHUTDOWN_CMD: ${AUTO_SHUTDOWN_CMD}"
        eval "${AUTO_SHUTDOWN_CMD}" || true
    fi
    exit 1
}

log "Started: WORK_ROOT=${WORK_ROOT} soft=${SOFT_LIMIT_MIN}m hard=${HARD_LIMIT_MIN}m idle=${IDLE_LIMIT_MIN}m"

while true; do
    sleep 60
    NOW=$(date +%s)
    ELAPSED_MIN=$(( (NOW - START_TS) / 60 ))

    if [ -f "${WORK_ROOT}/PIPELINE_FINISHED.ok" ]; then
        log "PIPELINE_FINISHED.ok detected at ${ELAPSED_MIN}m. Syncing and exiting cleanly."
        sync_artifacts
        exit 0
    fi

    # 1. 墙钟熔断
    if [ "${ELAPSED_MIN}" -ge "${HARD_LIMIT_MIN}" ]; then
        kill_pipeline "Hard wall-clock limit reached (${ELAPSED_MIN}m >= ${HARD_LIMIT_MIN}m)"
    elif [ "${ELAPSED_MIN}" -ge "${SOFT_LIMIT_MIN}" ] && [ ! -f "${WORK_ROOT}/STOP_REQUESTED" ]; then
        log "Soft limit ${SOFT_LIMIT_MIN}m reached -> writing STOP_REQUESTED"
        date -u > "${WORK_ROOT}/STOP_REQUESTED"
        sync_artifacts
    fi

    # 2. GPU 空转 + 日志停滞检测
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || echo "0")
    CUR_LOG_SIZE=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo "0")
    if [ "${GPU_UTIL:-0}" -lt 5 ] && [ "${CUR_LOG_SIZE}" -eq "${LAST_LOG_SIZE}" ]; then
        IDLE_STREAK=$((IDLE_STREAK + 1))
        log "Idle streak ${IDLE_STREAK}/${IDLE_LIMIT_MIN}m (gpu=${GPU_UTIL}% log=${CUR_LOG_SIZE}B)"
    else
        IDLE_STREAK=0
    fi
    LAST_LOG_SIZE="${CUR_LOG_SIZE}"
    if [ "${IDLE_STREAK}" -ge "${IDLE_LIMIT_MIN}" ]; then
        kill_pipeline "GPU idle <5% and log stalled for ${IDLE_STREAK} consecutive minutes"
    fi

    # 3. 磁盘余量检查
    AVAIL_GB=$(df -BG "${WORK_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || echo "99")
    if [ "${AVAIL_GB:-99}" -lt 10 ]; then
        log "Low disk (${AVAIL_GB}GB < 10GB) -> writing LOW_DISK_FLAG"
        echo "${AVAIL_GB}GB" > "${WORK_ROOT}/LOW_DISK_FLAG"
    fi
done
