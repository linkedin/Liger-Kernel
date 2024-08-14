# Liger-Kernel Example with Lightning Trainer

## How to Run
```bash
pip install -r requirements.txt
python training.py
```

The default hyperparameters and configurations work on single node with 8xA100 GPUs. For running on device with less GPU RAM, please consider reducing the per-GPU batch size and/or enable `CPUOffload` in FSDP.


<!-- Benchmark TBD -->
