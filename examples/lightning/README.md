# Liger-Kernel Example with Lightning Trainer

## How to Run
```bash
pip install -r requirements.txt
python training.py
```

**Notes**
1. The example uses Llama3 model that requires community license agreement and HuggingFace Hub login. If you want to use Llama3 in this example, please make sure you have done the followings:
    * Agree on the community license agreement https://huggingface.co/meta-llama/Meta-Llama-3-8B
    * Run `huggingface-cli login` and enter your HuggingFace token
2. The default hyperparameters and configurations work on single node with 8xA100 80GB GPUs. For running on device with less GPU RAM, please consider reducing the per-GPU batch size and/or enable `CPUOffload` in FSDP.


<!-- Benchmark TBD -->