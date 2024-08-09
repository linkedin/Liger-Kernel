#!/bin/sh

export HF_DATASETS_OFFLINE=1

# We use sgd to save memory on 1 GPU
python training.py \
      --model_path /shared/public/models/Meta-Llama-3-8B \
      --data_path /shared/public/data/booksum-complete-cleaned/chapters \
      --max_steps 10 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --max_seq_length 2048 \
      --logging_steps 1 \
      --include_num_input_tokens_seen \
      --save_strategy "no" \
      --learning_rate 1e-6 \
      --optim sgd \
      --report_to none \
      --output_dir "/tmp/output/" \
      --logging_dir /dev/null \
      --gradient_checkpointing \
      --liger_kernel
      
