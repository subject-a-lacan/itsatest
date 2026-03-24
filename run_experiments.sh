#!/usr/bin/env bash

# 简单的超参扫描脚本示例
# 用法示例：
#   bash run_experiments.sh
#   或者自定义设备： DEVICE=cuda:0 bash run_experiments.sh

set -e

# 可在这里统一设置默认设备和随机种子
# 注意：具体使用哪块 GPU 由 CUDA_VISIBLE_DEVICES 控制，
# 这里的 DEVICE 一般固定为 cuda:0（相对当前可见设备）。
DEVICE=${DEVICE:-"cuda:0"}
SEED=${SEED:-42}

# 要扫描的超参列表（按需修改）
LR_LIST=(1e-4)
BATCH_LIST=(128)
LAYER_LIST=(3 5 7)
EPOCH_LIST=(100)

# 其它固定参数（按需修改）
SIGNAL_DIM=63
RANK=2
NUM_SAMPLES=8192

# 日志目录
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# 最大并行任务数
MAX_JOBS=2

# 可用 GPU 列表（按需修改，例如只用 0 和 1）
GPU_LIST=(0 1)

# 全局任务计数，用于轮流分配 GPU
JOB_IDX=0

# 等待直到当前后台任务数小于 MAX_JOBS
wait_for_slot() {
  while : ; do
    # 统计正在运行的后台作业数
    local njobs
    njobs=$(jobs -r | wc -l)
    if [ "${njobs}" -lt "${MAX_JOBS}" ]; then
      break
    fi
    sleep 1
  done
}

for lr in "${LR_LIST[@]}"; do
  for bs in "${BATCH_LIST[@]}"; do
    for K in "${LAYER_LIST[@]}"; do
      for nepoch in "${EPOCH_LIST[@]}"; do
        exp_name="sd${SIGNAL_DIM}_r${RANK}_K${K}_bs${bs}_lr${lr}_ep${nepoch}_$(date +"%Y%m%d_%H%M%S")"
        log_file="${LOG_DIR}/${exp_name}.log"

        # 控制最大并行数
        wait_for_slot
        # 为该任务选择 GPU：在 GPU_LIST 中轮流分配
        gpu_idx=$(( JOB_IDX % ${#GPU_LIST[@]} ))
        gpu=${GPU_LIST[$gpu_idx]}
        JOB_IDX=$((JOB_IDX + 1))

        {
          echo "[RUN] ${exp_name} (CUDA_VISIBLE_DEVICES=${gpu}, device=${DEVICE}, seed=${SEED})"

          CUDA_VISIBLE_DEVICES="${gpu}" python3 -u main.py \
            --signal_dim "${SIGNAL_DIM}" \
            --rank "${RANK}" \
            --num_layers "${K}" \
            --num_samples "${NUM_SAMPLES}" \
            --num_epochs "${nepoch}" \
            --lr "${lr}" \
            --batch_size "${bs}" \
            --seed "${SEED}" \
            --device "${DEVICE}"

          echo "[DONE] ${exp_name}"
          echo
        } 2>&1 | tee -a "${log_file}" &
      done
    done
  done
done

# 等待所有后台任务结束
wait
