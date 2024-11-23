# train_orpo.py
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
sys.path.insert(0, "/home/jobuser/Liger-Kernel/examples/alignment")
from orpo_trainer import LigerORPOTrainer  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("./orpo_testing_data", split="train")
# get first 1000 examples
train_dataset = train_dataset.select(range(1000))
train_dataset = train_dataset.map(
    lambda example: {
        "prompt": example["prompt"],
        "chosen": example["chosen"][0]['content'],
        "rejected": example["rejected"][0]['content']
    }
)
training_args = ORPOConfig(output_dir="Llama3.2_1B_Instruct", beta=0.1, per_device_train_batch_size=32, max_steps=25)
trainer = LigerORPOTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=train_dataset)
trainer.train()
