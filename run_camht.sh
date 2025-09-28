#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(dirname "${SCRIPT_PATH}")"

DEFAULT_CFG="camht/configs/camht.yaml"
RTX4090_CFG="camht/configs/camht_rtx4090.yaml"
RTX5090_CFG="camht/configs/camht_rtx5090.yaml"

USE_TMUX=0
TMUX_SESSION="camht"
PROFILE=""
CFG_PATH=""
PASSTHROUGH=()

show_help() {
    cat <<'EOH'
Usage: run_camht.sh [options]

Options:
  --profile NAME       预置配置：可选 rtx4090 / rtx5090
  --cfg PATH           指定自定义配置文件路径，覆盖 profile
  --tmux [SESSION]     如在非 tmux 环境，自动创建/附加到指定 SESSION（默认 camht）
  -h, --help           显示此帮助

示例：
  ./run_camht.sh --profile rtx4090 --tmux camht4090
EOH
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -lt 2 ]] && { echo "--profile 需要参数"; exit 1; }
            PROFILE="$2"
            PASSTHROUGH+=("$1" "$2")
            shift 2
            ;;
        --cfg)
            [[ $# -lt 2 ]] && { echo "--cfg 需要参数"; exit 1; }
            CFG_PATH="$2"
            PASSTHROUGH+=("$1" "$2")
            shift 2
            ;;
        --tmux)
            USE_TMUX=1
            if [[ $# -ge 2 && "$2" != --* ]]; then
                TMUX_SESSION="$2"
                shift 2
            else
                shift
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

if (( USE_TMUX )) && [[ -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "未检测到 tmux，请安装后再使用 --tmux 选项" >&2
        exit 1
    fi
    if tmux has-session -t "${TMUX_SESSION}" >/dev/null 2>&1; then
        tmux attach -t "${TMUX_SESSION}"
    else
        tmux new-session -d -s "${TMUX_SESSION}" "${SCRIPT_PATH} ${PASSTHROUGH[*]}"
        tmux attach -t "${TMUX_SESSION}"
    fi
    exit 0
fi

cd "${PROJECT_ROOT}"

case "${PROFILE}" in
    rtx4090)
        CFG_PATH="${CFG_PATH:-${RTX4090_CFG}}"
        ;;
    rtx5090)
        CFG_PATH="${CFG_PATH:-${RTX5090_CFG}}"
        ;;
esac

CFG_PATH="${CFG_PATH:-${DEFAULT_CFG}}"

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "配置文件不存在: ${CFG_PATH}" >&2
    exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

START_TIME="$(date +%s)"
RUN_STATUS="success"
EXIT_REASON=""

send_report() {
    if [[ "${CAMHT_DISABLE_EMAIL:-0}" == "1" ]]; then
        echo "[info] 邮件通知被禁用"
        return
    fi
    local smtp_host="${CAMHT_EMAIL_SMTP:-}"
    local smtp_port="${CAMHT_EMAIL_PORT:-587}"
    local email_user="${CAMHT_EMAIL_USER:-}"
    local email_pass="${CAMHT_EMAIL_PASSWORD:-}"
    local email_to="${CAMHT_EMAIL_TO:-willetjiang000@gmail.com}"
    if [[ -z "${smtp_host}" || -z "${email_user}" || -z "${email_pass}" ]]; then
        echo "[warn] 邮件凭据不完整，跳过邮件发送"
        return
    fi

    local end_time="$(date +%s)"
    local duration=$(( end_time - START_TIME ))
    local tail_log
    tail_log=$(tail -n 80 "${LOG_FILE}" 2>/dev/null || echo "日志不可用")

    local body
    printf -v body '运行状态: %s\n原因: %s\n配置文件: %s\n快照: %s\n日志位置: %s\n耗时: %ds\n启动时间: %s\n结束时间: %s\n\n最近80行日志:\n%s\n' \
        "${RUN_STATUS}" "${EXIT_REASON:-无}" "${CFG_PATH}" "${SNAPSHOT_PATH:-未生成}" "${LOG_FILE}" "${duration}" "$(date -d @${START_TIME} '+%F %T' 2>/dev/null || date '+%F %T')" "$(date -d @${end_time} '+%F %T' 2>/dev/null || date '+%F %T')" "${tail_log}"

    RUN_STATUS_MSG="${RUN_STATUS}" \
    CAMHT_EMAIL_BODY="${body}" \
    CAMHT_EMAIL_TO="${email_to}" \
    CAMHT_EMAIL_SMTP="${smtp_host}" \
    CAMHT_EMAIL_PORT="${smtp_port}" \
    CAMHT_EMAIL_USER="${email_user}" \
    CAMHT_EMAIL_PASSWORD="${email_pass}" python - <<'PY'
import os, smtplib, ssl, sys
from email.message import EmailMessage

body = os.environ.get("CAMHT_EMAIL_BODY")
if not body:
    sys.exit(0)
smtp_host = os.environ.get("CAMHT_EMAIL_SMTP")
smtp_port = int(os.environ.get("CAMHT_EMAIL_PORT", "587"))
user = os.environ.get("CAMHT_EMAIL_USER")
password = os.environ.get("CAMHT_EMAIL_PASSWORD")
to = os.environ.get("CAMHT_EMAIL_TO")
status = os.environ.get("RUN_STATUS_MSG", "unknown")
if not (smtp_host and user and password and to):
    sys.exit(0)
msg = EmailMessage()
msg["Subject"] = f"[CAMHT] Run status: {status}"
msg["From"] = user
msg["To"] = to
msg.set_content(body)

context = ssl.create_default_context()
with smtplib.SMTP(smtp_host, smtp_port, timeout=45) as server:
    server.starttls(context=context)
    server.login(user, password)
    server.send_message(msg)
PY
}

shutdown_machine() {
    if [[ "${CAMHT_DISABLE_SHUTDOWN:-0}" == "1" ]]; then
        echo "[info] 自动关机已禁用"
        return
    fi
    echo "[info] 任务结束，系统将在 15 秒后关机 (设置 CAMHT_DISABLE_SHUTDOWN=1 可禁用)"
    sleep 15 || true
    if command -v shutdown >/dev/null 2>&1; then
        nohup shutdown -h now >/dev/null 2>&1 &
    elif command -v poweroff >/dev/null 2>&1; then
        nohup poweroff >/dev/null 2>&1 &
    else
        echo "[warn] 未找到 shutdown/poweroff 命令，无法自动关机"
    fi
}

cleanup() {
    local exit_code=$?
    if [[ "${RUN_STATUS}" == "success" ]]; then
        if [[ ${exit_code} -ne 0 ]]; then
            RUN_STATUS="failure"
            EXIT_REASON="exit code ${exit_code}"
        else
            EXIT_REASON="completed successfully"
        fi
    fi
    send_report
    shutdown_machine
}

on_interrupt() {
    RUN_STATUS="interrupted"
    EXIT_REASON="terminated by signal"
    exit 1
}

trap cleanup EXIT
trap on_interrupt INT TERM

SNAPSHOT_PATH=$(python - "$CFG_PATH" <<'PY' 2>/dev/null || true
import sys
try:
    from omegaconf import OmegaConf
except Exception:
    raise SystemExit
cfg = OmegaConf.load(sys.argv[1])
ckpt_dir = cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
if not isinstance(ckpt_dir, str):
    ckpt_dir = "checkpoints"
print(f"{ckpt_dir.rstrip('/')}/best.ckpt")
PY
)

if [[ -z "${SNAPSHOT_PATH}" ]]; then
    SNAPSHOT_PATH="checkpoints/best.ckpt"
fi

TOTAL_STEPS=3
BAR_WIDTH=30
CURRENT_STEP=0

draw_bar() {
    local done=$((CURRENT_STEP * BAR_WIDTH / TOTAL_STEPS))
    local remain=$((BAR_WIDTH - done))
    local filled=""
    local todo=""
    if (( done > 0 )); then
        filled=$(printf '%*s' "${done}" "" | tr ' ' '#')
    fi
    if (( remain > 0 )); then
        todo=$(printf '%*s' "${remain}" "" | tr ' ' '.')
    fi
    printf "[%s%s] %d/%d | %s" "${filled}" "${todo}" "${CURRENT_STEP}" "${TOTAL_STEPS}" "$1"
}

run_step() {
    local label="$1"
    shift
    CURRENT_STEP=$((CURRENT_STEP + 1))
    draw_bar "${label}..."
    printf '\n'
    "$@"
    draw_bar "${label} 完成"
    printf '\n'
}

echo "使用配置: ${CFG_PATH}"

run_step "预训练 TiMAE 编码器" python pretrain_timae.py --cfg "${CFG_PATH}"
run_step "监督训练 CAMHT 主干" python train_camht.py --cfg "${CFG_PATH}"

run_step "离线推理生成提交模板" python infer_submit.py --snapshot "${SNAPSHOT_PATH}" --data data/test.csv --target_pairs data/target_pairs.csv --date_col date_id

printf '\n全部流程完成：最新模型保存在 %s，推理结果位于 preds/\n' "${SNAPSHOT_PATH}"
