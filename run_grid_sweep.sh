#!/usr/bin/env bash

set -u

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}

SIGNAL_DIM=63
RANK=3
NUM_SAMPLES=20000
NUM_EPOCHS=30
BATCH_SIZE=128
LR=5e-4

LOG_DIR=${LOG_DIR:-logs_grid}
MAX_JOBS=${MAX_JOBS:-1}

LAMBDA_LIST_DEFAULT="0.005 0.02 0.03 6 8 20 45 55 90"
K_LIST_DEFAULT="5 6 7 8 9 10"

read -r -a LAMBDAS <<< "${LAMBDA_LIST:-$LAMBDA_LIST_DEFAULT}"
read -r -a KS <<< "${K_LIST:-$K_LIST_DEFAULT}"
read -r -a GPUS <<< "${GPU_LIST:-0}"

mkdir -p "${LOG_DIR}"

wait_for_slot() {
  while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done
}

job_idx=0

for lam in "${LAMBDAS[@]}"; do
  for K in "${KS[@]}"; do
    wait_for_slot

    gpu_idx=$((job_idx % ${#GPUS[@]}))
    gpu=${GPUS[$gpu_idx]}
    job_idx=$((job_idx + 1))

    exp_name="lambda${lam}_K${K}_n${NUM_SAMPLES}_ep${NUM_EPOCHS}_$(date +"%Y%m%d_%H%M%S")"
    log_file="${LOG_DIR}/${exp_name}.log"

    {
      echo "[RUN] ${exp_name}"
      echo "[GPU] CUDA_VISIBLE_DEVICES=${gpu}"
      echo "[TIME] $(date +"%F %T")"
      echo "[CMD] ${PYTHON_BIN} -u main.py --loss_lambda ${lam} --num_layers ${K} --lr ${LR} --batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --num_epochs ${NUM_EPOCHS}"

      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u main.py \
        --signal_dim "${SIGNAL_DIM}" \
        --rank "${RANK}" \
        --num_layers "${K}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_epochs "${NUM_EPOCHS}" \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --loss_lambda "${lam}" \
        --seed "${SEED}" \
        --device "${DEVICE}"

      echo "[DONE] ${exp_name}"
      echo "[TIME] $(date +"%F %T")"
    } > "${log_file}" 2>&1 &

    echo "Started ${exp_name}, log: ${log_file}"
  done
done

wait
echo "All grid-sweep experiments finished."
