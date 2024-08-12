from dataclasses import dataclass, field

import datasets
import torch
import transformers
import trl
from liger_kernel.transformers import apply_liger_kernel_to_llama
from profiler import ProfilerCallback
import time
from efficiency import EfficiencyCallback
import random

def set_seed(seed: int = 11):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)  # This sets the seed for transformers-specific code.

@dataclass
class CustomArguments:
    model_path: str
    data_path: str
    max_seq_length: int = field(default=512)
    liger_kernel: bool = field(default=False)


QUESTION = "<Question>"
CHOICES = "<Choices>"
ANSWER = "<Answer>"

def formatting_func(example):

    output_texts = []
    
    for i in range(len(example['question'])):
        choices = ""
        for j in range(len(example["choices"][i])):
            choices += f"{j+1}. {example['choices'][i][j]}; "
        s = "Below is a question and multiple choice answers, choices separated by a semicolon. Please select the best answer for the question. "
        s += f"{QUESTION}{example['question'][i]} "
        s += f"{CHOICES}{choices} "
        s += f"{ANSWER}{example['answer'][i]}"
        output_texts.append(s)
    
    return output_texts


def main():
    set_seed()

    parser = transformers.HfArgumentParser(
        (transformers.TrainingArguments, CustomArguments)
    )
    training_args, custom_args = parser.parse_args_into_dataclasses()

    if custom_args.liger_kernel:
        apply_liger_kernel_to_llama(
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )

    dataset = datasets.load_from_disk(custom_args.data_path)
    dataset_train, dataset_eval = dataset["auxiliary_train"], dataset["test"]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        custom_args.model_path,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        custom_args.model_path, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token


    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        formatting_func=formatting_func,
        max_seq_length=custom_args.max_seq_length,
        args=training_args,
        callbacks=[EfficiencyCallback(), transformers.integrations.MLflowCallback(),],
    )
    trainer.train()


if __name__ == "__main__":
    main()
