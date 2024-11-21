#!/bin/bash

torchrun --nnodes=1 --nproc-per-node=4 training_multimodal.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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
    --use_liger True \
    --output_dir multimodal_finetuning
