#!/usr/bin/env bash

set -euo pipefail

CFG_PATH=${1:-camht/configs/camht.yaml}

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

run_step "预训练 TiMAE 编码器" python pretrain_timae.py --cfg "${CFG_PATH}"
run_step "监督训练 CAMHT 主干" python train_camht.py --cfg "${CFG_PATH}"
run_step "离线推理生成提交模板" python infer_submit.py --snapshot checkpoints/best.ckpt --data data/test.csv --target_pairs data/target_pairs.csv --date_col date_id

printf '\n全部流程完成：checkpoints/best.ckpt 与 preds/ 目录已经更新\n'
