#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(dirname "${SCRIPT_PATH}")"

DEFAULT_CFG="camht/configs/camht.yaml"
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
  --profile NAME       预置配置：可选 rtx5090（将自动使用 camht/configs/camht_rtx5090.yaml）
  --cfg PATH           指定自定义配置文件路径，覆盖 profile
  --tmux [SESSION]     如在非 tmux 环境，自动创建/附加到指定 SESSION（默认 camht）
  -h, --help           显示此帮助

示例：
  ./run_camht.sh --profile rtx5090 --tmux camht5090
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

case "${PROFILE}" in
    rtx5090)
        CFG_PATH="${CFG_PATH:-${RTX5090_CFG}}"
        ;;
esac

CFG_PATH="${CFG_PATH:-${DEFAULT_CFG}}"

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "配置文件不存在: ${CFG_PATH}" >&2
    exit 1
fi

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
