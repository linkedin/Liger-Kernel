
| **Use Case**                                    | **Description**                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |