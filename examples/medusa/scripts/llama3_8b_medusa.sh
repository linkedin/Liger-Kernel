#!/bin/sh

export GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export NUM_NODES=$WORLD_SIZE
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
echo "Starting training... Num nodes: $NUM_NODES, Num workers: $WORLD_SIZE"

export OUTPUT_DIR="./llama3-8b-medusa-liger"

export LOCAL_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export LR=1e-5

export MEDUSA_NUM_HEADS=5
export MEDUSA_NUM_LAYERS=1
export MEDUSA_HEADS_COEFFICIENT=0.2
export MEDUSA_DECAY_COEFFICIENT=0.8
export MEDUSA_SCHEDULER=constant
export MEDUSA_LR_MULTIPLIER=4.0

accelerate launch --config_file fsdp/acc-fsdp.conf \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    train.py \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size $LOCAL_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --prediction_loss_only \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --report_to none \
    --include_num_input_tokens_seen \
    --medusa_num_heads $MEDUSA_NUM_HEADS \
    --medusa_num_layers $MEDUSA_NUM_LAYERS \
    --medusa_heads_coefficient $MEDUSA_HEADS_COEFFICIENT \
    --medusa_decay_coefficient $MEDUSA_DECAY_COEFFICIENT \
    --medusa_scheduler $MEDUSA_SCHEDULER \
    --medusa_lr_multiplier $MEDUSA_LR_MULTIPLIER \
    --medusa_only_heads False \
    --medusa_return True \
    --use_liger True