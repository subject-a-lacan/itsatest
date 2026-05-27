#!/usr/bin/env bash

set -u

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}

SIGNAL_DIM=63
RANK=3
NUM_SAMPLES=40000
NUM_EPOCHS=100
BATCH_SIZE=128

LOG_DIR=${LOG_DIR:-logs_sweep2}
MAX_JOBS=${MAX_JOBS:-1}

LAMBDA_LIST_DEFAULT="0.03 3 6 55"
read -r -a LAMBDAS <<< "${LAMBDA_LIST:-$LAMBDA_LIST_DEFAULT}"
read -r -a GPUS <<< "${GPU_LIST:-0}"

mkdir -p "${LOG_DIR}"

wait_for_slot() {
  while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done
}

job_idx=0

for lam in "${LAMBDAS[@]}"; do
  for K in 8 9 10; do

    if [ "${K}" -eq 8 ]; then
      LR_LIST="5e-4 2e-4"
    else
      LR_LIST="2e-4"
    fi

    for lr in ${LR_LIST}; do
      wait_for_slot

      gpu_idx=$((job_idx % ${#GPUS[@]}))
      gpu=${GPUS[$gpu_idx]}
      job_idx=$((job_idx + 1))

      exp_name="lambda${lam}_K${K}_lr${lr}_n${NUM_SAMPLES}_ep${NUM_EPOCHS}_$(date +"%Y%m%d_%H%M%S")"
      log_file="${LOG_DIR}/${exp_name}.log"

      {
        echo "[RUN] ${exp_name}"
        echo "[GPU] CUDA_VISIBLE_DEVICES=${gpu}"
        echo "[TIME] $(date +"%F %T")"
        echo "[CMD] ${PYTHON_BIN} -u main.py --loss_lambda ${lam} --num_layers ${K} --lr ${lr} --batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --num_epochs ${NUM_EPOCHS}"

        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u main.py \
          --signal_dim "${SIGNAL_DIM}" \
          --rank "${RANK}" \
          --num_layers "${K}" \
          --num_samples "${NUM_SAMPLES}" \
          --num_epochs "${NUM_EPOCHS}" \
          --lr "${lr}" \
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
done

wait
echo "All sweep2 experiments finished."
