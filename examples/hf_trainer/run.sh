#!/bin/sh

export PATH=$HOME/.local/bin:$PATH

source /home/jobuser/mldev-scripts/setup_mlflow_hf.sh

OUTPUT_DIR="$FLYTE_INTERNAL_EXECUTION_ID-$(date -Iseconds)"

echo "Model output path is: $OUTPUT_FULL_PATH"
echo "Tensorboard log path is: $LOG_FULL_PATH"

export HF_DATASETS_OFFLINE=1

export GPUS_PER_NODE=$(nvidia-smi --list-gpus|wc -l)
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE

# WORD_SIZE is incorrectly set as number of nodes by Flyte pytorch plugin
export NUM_NODES=$WORLD_SIZE

# This sets correct for the world size
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))


torchrun --nnodes=$NUM_NODES --nproc-per-node=$GPUS_PER_NODE training.py \
      --model_path /shared/public/models/Meta-Llama-3-8B \
      --data_path /shared/public/data/mmlu \
      --max_steps 600 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --max_seq_length 2048 \
      --logging_steps 1 \
      --include_num_input_tokens_seen \
      --save_strategy "no" \
      --learning_rate 6e-6 \
      --optim adamw_torch_fused \
      --warmup_ratio 0.1 \
      --weight_decay 0.05 \
      --report_to none \
      --output_dir $OUTPUT_DIR \
      --logging_dir /dev/null \
      --fsdp "full_shard auto_wrap" \
      --fsdp_config config/fsdp_config.json \
      
