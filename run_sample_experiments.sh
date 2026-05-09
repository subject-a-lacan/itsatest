#!/usr/bin/env bash

set -u

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}

SIGNAL_DIM=${SIGNAL_DIM:-63}
RANK=${RANK:-3}
NUM_LAYERS=${NUM_LAYERS:-5}
NUM_EPOCHS=${NUM_EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-3e-5}

LOG_DIR=${LOG_DIR:-logs_samples}
MAX_JOBS=${MAX_JOBS:-1}

SAMPLE_LIST_DEFAULT="10000 20000 30000 40000 50000 60000 70000 80000 90000 100000"
read -r -a SAMPLE_LIST <<< "${SAMPLE_LIST:-$SAMPLE_LIST_DEFAULT}"
read -r -a GPUS <<< "${GPU_LIST:-0}"

mkdir -p "${LOG_DIR}"

wait_for_slot() {
  while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done
}

job_idx=0

for num_samples in "${SAMPLE_LIST[@]}"; do
  wait_for_slot

  gpu_idx=$((job_idx % ${#GPUS[@]}))
  gpu=${GPUS[$gpu_idx]}
  job_idx=$((job_idx + 1))

  exp_name="n${num_samples}_lr${LR}_K${NUM_LAYERS}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_$(date +"%Y%m%d_%H%M%S")"
  log_file="${LOG_DIR}/${exp_name}.log"

  {
    echo "[RUN] ${exp_name}"
    echo "[GPU] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[TIME] $(date +"%F %T")"
    echo "[CMD] ${PYTHON_BIN} -u main.py --num_samples ${num_samples} --lr ${LR} --batch_size ${BATCH_SIZE} --num_layers ${NUM_LAYERS} --num_epochs ${NUM_EPOCHS}"

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u main.py \
      --signal_dim "${SIGNAL_DIM}" \
      --rank "${RANK}" \
      --num_layers "${NUM_LAYERS}" \
      --num_samples "${num_samples}" \
      --num_epochs "${NUM_EPOCHS}" \
      --lr "${LR}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --device "${DEVICE}"

    echo "[DONE] ${exp_name}"
    echo "[TIME] $(date +"%F %T")"
  } > "${log_file}" 2>&1 &

  echo "Started ${exp_name}, log: ${log_file}"
done

wait
echo "All sample-size experiments finished."
