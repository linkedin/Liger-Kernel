


| **Use Case**                                    | **Description**                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

## HuggingFace Trainer

## How to Run

### Locally on a GPU machine
You can run the example locally on a GPU machine. The default hyperparameters and configurations work on single node with 4xA100 80GB GPUs.

```bash
pip install -r requirements.txt
sh run_{MODEL}.sh
```

### Remotely on Modal
If you do not have access to a GPU machine, you can run the example on Modal. Modal is a serverless platform that allows you to run your code on a remote GPU machine. You can sign up for a free account at [Modal](https://www.modal.com/).

```bash
pip install modal
modal setup  # authenticate with Modal
modal run launch_on_modal.py --script "run_qwen2_vl.sh"
```

**Notes**

1. This example uses an optional `use_liger` flag. If true, it does a 1 line monkey patch to apply liger kernel.

2. The example uses Llama3 model that requires community license agreement and HuggingFace Hub login. If you want to use Llama3 in this example, please make sure you have done the following:
    * Agree on the [community license agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
    * Run `huggingface-cli login` and enter your HuggingFace token.

3. The default hyperparameters and configurations work on single node with 4xA100 80GB GPUs. For running on device with less GPU RAM, please consider reducing the per-GPU batch size and/or enable `CPUOffload` in FSDP.


## Benchmark Result

### LLaMA
Benchmark conditions: 

LLaMA 3-8B, Alpaca Dataset, Max seq len = 512, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 4 A100s.

Throughput improves by around 20%, while GPU memory usage drops by 40%. This allows you to train the model on smaller GPUs, use larger batch sizes, or handle longer sequence lengths without incurring additional costs.

![Throughput](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/llama_tps.png)
![GPU Memory Allocated](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/llama_mem_alloc.png)

### QWEN
Benchmark conditions:

Qwen2-7B, Alpaca Dataset, Max seq len = 512, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 4 A100s.

Throughput improves by around 10%, while GPU memory usage drops by 50%.

![Throughput](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/qwen_tps.png)
![GPU Memory Allocated](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/qwen_mem_alloc.png)


### GEMMA 7B
Benchmark conditions:

Gemma-7B, Alpaca Dataset, Max seq len = 512, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 4 A100s.

Throughput improves by around 24%, while GPU memory usage drops by 33%.

![Throughput](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/gemma_7b_mem.png)
![GPU Memory Allocated](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/examples/huggingface/img/gemma_7b_tp.png)

# Lightning Trainer

## How to Run
```bash
pip install -r requirements.txt

# For single L40 48GB GPU
python training.py --model Qwen/Qwen2-0.5B-Instruct --num_gpu 1 --max_length 1024

# For 8XA100 40GB
python training.py --model meta-llama/Meta-Llama-3-8B --strategy deepspeed
```

**Notes**

1. The example uses Llama3 model that requires community license agreement and HuggingFace Hub login.

If you want to use Llama3 in this example, please make sure you have done the following:
    * Agree on the [community license agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
    * Run `huggingface-cli login` and enter your HuggingFace token.
2. The default hyperparameters and configurations for gemma works on single L40 48GB GPU and config for llama work on single node with 8xA100 40GB GPUs. For running on device with less GPU RAM, please consider reducing the per-GPU batch size and/or enable `CPUOffload` in FSDP.

# Medusa

Medusa is a simple framework that democratizes the acceleration techniques for LLM generation with multiple decoding heads. [[repo](https://arxiv.org/abs/2401.10774)], [[paper](https://arxiv.org/abs/2401.10774)]

During training, Medusa requires adding \(k\) decoding heads to the hidden states right before the regular LM head \(h_t\). The \(k\)-th head is used to predict the token in the \((t + k + 1)\)-th position of the next tokens (the original language model head is used to predict the \((t + 1)\)-th position). 

The Liger fused CE kernel is highly effective in this scenario, eliminating the need to materialize logits for each head, which usually consumes a large volume of memory due to the extensive vocabulary size (e.g., for LLaMA-3, the vocabulary size is 128k). The introduction of multiple heads can easily lead to OOM (Out of Memory) issues. However, thanks to the efficient Liger fused CE, which calculates the gradient in place and doesn't materialize the logits, we have observed very effective results. This efficiency opens up more opportunities for multi-token prediction research and development.


## Instructions to Run the Training Script

```
git clone git@github.com:linkedin/Liger-Kernel.git
cd {PATH_TO_Liger-Kernel}/Liger-Kernel/
pip install -e .
cd {PATH_TO_Liger-Kernel}/Liger-Kernel/examples/medusa
pip install -r requirements.txt
sh scripts/llama3_8b_medusa.sh
```

**Notes**
1. This example uses an optional `use_liger` flag. If true, it does a monkey patch to apply liger kernel with medusa heads.
2. The example uses Llama3 model that requires community license agreement and HuggingFace Hub login. If you want to use Llama3 in this example, please make sure you have done the followings:
    * Agree on the community license agreement https://huggingface.co/meta-llama/Meta-Llama-3-8B
    * Run `huggingface-cli login` and enter your HuggingFace token
3. The default hyperparameters and configurations work on single node with 8xA100 GPUs. For running on device with less GPU RAM, please consider reducing the per-GPU batch size and/or enable `CPUOffload` in FSDP.
4. We are using a smaller sample of shared GPT data primarily to benchmark performance. The example requires hyperparameter tuning and dataset selection to work effectively, also ensuring the dataset has the same distribution as the LLaMA pretraining data. Welcome contribution to enhance the example code.


# Memory Profiling Result

> **Note:**  
> 1. Benchmark conditions: LLaMA 3-8B, Batch Size = 6, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.

## Stage1

Stage1 refers to Medusa-1 where the backbone model is frozen and only weights of LLM heads are updated.

```
# Modify this flag in llama3_8b_medusa.sh to True enables stage1 
--medusa_only_heads True
```

### num_head = 3

![Memory](./docs/images/Memory_Stage1_num_head_3.png)
![Throughput](./docs/images/Throughput_Stage1_num_head_3.png)

### num_head = 5

![Memory](./docs/images/Memory_Stage1_num_head_5.png)
![Throughput](./docs/images/Throughput_Stage1_num_head_5.png)

## Stage2

```
# Modify this flag to False in llama3_8b_medusa.sh enables stage2
--medusa_only_heads False
```

Stage2 refers to Medusa-2 where all the model weights are updated incuding backbone model and llm heads.

### num_head = 3

![Memory](./docs/images/Memory_Stage2_num_head_3.png)
![Throughput](./docs/images/Throughput_Stage2_num_head_3.png)

### num_head = 5

![Memory](./docs/images/Memory_Stage2_num_head_5.png)
![Throughput](./docs/images/Throughput_Stage2_num_head_5.png)

