#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hpc-low
#SBATCH --time=00:10:00

# Print GPU info
echo "GPU Info:"
nvidia-smi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Run the test
echo "Running DPO test..."
python test_triton_dpo_simple.py 