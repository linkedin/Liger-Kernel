## Benchmarking Script
## Runs the training script with different configurations and logs the results

# MODEL_TYPE="llama"
# MODEL_PATH="/shared/public/models/Meta-Llama-3-8B"
# MODEL_TYPE="mistral"
# MODEL_PATH="/shared/public/models/mistralai/Mistral-7B-v0.1"
# MODEL_TYPE="gemma"
# MODEL_PATH="/shared/public/models/gemma-7b-it"
# MODEL_TYPE="gemma2"
# MODEL_PATH="/shared/public/models/google/gemma-2-2b-it"
# MODEL_TYPE="qwen2"
# MODEL_PATH="/shared/public/models/Qwen/Qwen2-7B-Instruct"
# MODEL_TYPE="phi3"
# MODEL_PATH="/shared/public/models/microsoft/Phi-3.5-mini-instruct"
# MODEL_TYPE="mixtral"
# MODEL_PATH="/shared/public/models/Mixtral-8x7B-v0.1"
MODEL_TYPE="qwen2_vl"
MODEL_PATH="/shared/public/elr-models/Qwen/Qwen2-VL-2B-Instruct/3c86da475a9bcc0876910f022ecdd476e621e636"

# USE_LIGER_VALUES=("True" "False")
# PATCHING_TYPE_VALUES=("pre_init" "post_init_class" "post_init_instance")
USE_LIGER_VALUES=("False" "True")
PATCHING_TYPE_VALUES=("pre_init" "post_init_instance" "post_init_class")
MAX_STEPS=10
BATCH_SIZE=32
DATASET_PATH="/shared/public/data/tatsu-lab"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/results"

for USE_LIGER in "${USE_LIGER_VALUES[@]}"; do

    # Only run patching types if USE_LIGER is "True"
    if [ "$USE_LIGER" == "True" ]; then
        PATCHING_TYPES=("${PATCHING_TYPE_VALUES[@]}")
    else
        PATCHING_TYPES=("None")
    fi

    for PATCHING_TYPE in "${PATCHING_TYPES[@]}"; do
        echo "Running with use_liger=$USE_LIGER and patching_type=$PATCHING_TYPE"
        
        LOG_FILE="${SCRIPT_DIR}/results/${MODEL_TYPE}_use_liger_${USE_LIGER}_patching_type_${PATCHING_TYPE}.log"

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
            --patching_type $PATCHING_TYPE \
            --output_dir model_output_dir \
            > $LOG_FILE

        sleep 5
    done
done