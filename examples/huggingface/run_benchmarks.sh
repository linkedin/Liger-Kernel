#!/bin/bash

## Benchmarking Script
## Runs the training script with different configurations and logs the results

MODEL_TYPE="mistral"
MODEL_PATH="mistralai/Mistral-7B-v0.1"
USE_LIGER_VALUES=("True" "False")
BATCH_SIZE_VALUES=(64 128 192)
NUM_REP=5
MAX_STEPS=20
DATASET_PATH="tatsu-lab/alpaca"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/results"

for USE_LIGER in "${USE_LIGER_VALUES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
        echo "Running with use_liger=$USE_LIGER and batch_size=$BATCH_SIZE"

        for ((i=1; i<=NUM_REP; i++)); do

            LOG_FILE="${SCRIPT_DIR}/results/${MODEL_TYPE}_use_liger_${USE_LIGER}_batch_size_${BATCH_SIZE}_rep_${i}.log"

            torchrun --nnodes=1 --nproc-per-node=4 training.py \
                --bf16 \
                --num_train_epochs 1 \
                --max_steps $MAX_STEPS \
                --model_name $MODEL_PATH \
                --dataset $DATASET_PATH \
                --per_device_train_batch_size $BATCH_SIZE \
                --per_device_eval_batch_size 16 \
                --eval_strategy "no" \
                --save_strategy "no" \
                --learning_rate 6e-6 \
                --weight_decay 0.05 \
                --warmup_ratio 0.1 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --include_num_input_tokens_seen \
                --report_to none \
                --fsdp "full_shard auto_wrap" \
                --fsdp_config config/fsdp_config.json \
                --seed 42 \
                --use_liger $USE_LIGER \
                --output_dir model_output_dir \
                > $LOG_FILE

            sleep 5
        done
    done
done