import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import ORPOConfig  # noqa: F401

from liger_kernel.transformers.trainer import LigerORPOTrainer  # noqa: F401

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    max_length=512,
    padding="max_length",
)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("trl-lib/tldr-preference", split="train")

training_args = ORPOConfig(
    output_dir="Llama3.2_1B_Instruct",
    beta=0.1,
    max_length=128,
    per_device_train_batch_size=32,
    max_steps=100,
    save_strategy="no",
)

trainer = LigerORPOTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=train_dataset)

trainer.train()
