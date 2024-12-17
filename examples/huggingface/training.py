from dataclasses import dataclass

import datasets
import torch
import transformers

from callback import EfficiencyCallback
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

from liger_kernel.transformers import AutoLigerKernelForCausalLM


@dataclass
class CustomArguments:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    dataset: str = "tatsu-lab/alpaca"
    max_seq_length: int = 512
    use_liger: bool = False


def formatting_prompts_func(example):
    return example["text"]


def train():
    parser = transformers.HfArgumentParser((transformers.TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        custom_args.model_name,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset(custom_args.dataset)["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    response_prompt = tokenizer.encode("### Response:\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_prompt,
        pad_to_multiple_of=16,
    )

    if custom_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            custom_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            # These args will get passed to the appropriate apply_liger_kernel_to_* function
            # to override the default settings
            # cross_entropy=True,
            # fused_linear_cross_entropy=False,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            custom_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        max_seq_length=custom_args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        callbacks=[EfficiencyCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    train()
