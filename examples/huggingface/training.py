from dataclasses import dataclass

import datasets
import torch
import transformers
from callback import EfficiencyCallback
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from liger_kernel.transformers import (
    AutoLigerKernelForCausalLM,
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_gemma2,
)
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel, _apply_liger_kernel_to_instance


@dataclass
class CustomArguments:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    dataset: str = "tatsu-lab/alpaca"
    max_seq_length: int = 512
    use_liger: bool = False
    patching_type: str = "pre_init" # pre_init, post_init_class, post_init_instance

# bos_token = '<|begin_of_text|>' # llama
# bos_token = '<s>' # mistral
# bos_token = '<bos>' # gemma
bos_token = '<|endoftext|>'

def formatting_prompts_func(example):
    return [text.replace("### Response:", bos_token) for text in example["text"]]


def train():
    parser = transformers.HfArgumentParser(
        (transformers.TrainingArguments, CustomArguments)
    )
    training_args, custom_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        custom_args.model_name,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset(custom_args.dataset)["train"].train_test_split(
        test_size=0.1
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    response_prompt = bos_token
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_prompt,
        pad_to_multiple_of=16,
    )

    
    if custom_args.use_liger:
        if custom_args.patching_type == "pre_init":
            print("********** Pre-Init Patching ***********")
            # apply_liger_kernel_to_gemma2()
            # model = transformers.AutoModelForCausalLM.from_pretrained(
            #     custom_args.model_name,
            #     trust_remote_code=True,
            #     use_cache=False,
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation='eager', # for gemma2
            # )
            model = AutoLigerKernelForCausalLM.from_pretrained(
                custom_args.model_name,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch.bfloat16,
                # attn_implementation='eager', # for gemma2
            )
        elif custom_args.patching_type == "post_init_class":
            print("********** Post-Init Class Patching ***********")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                custom_args.model_name,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch.bfloat16,
                # attn_implementation='eager', # for gemma2
            )
            model_type = getattr(model, "config", None) and getattr(
                model.config, "model_type", None
            )
            _apply_liger_kernel(model_type=model_type)
        elif custom_args.patching_type == "post_init_instance":
            print("********** Post-Init Instance Patching ***********")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                custom_args.model_name,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch.bfloat16,
                # attn_implementation='eager', # for gemma2
            )
            _apply_liger_kernel_to_instance(model)
        else:
            raise ValueError(f"Invalid patching type: {custom_args.patching_type}")
    else:
        print("********** No Patching ***********")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            custom_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            # attn_implementation='eager', # for gemma2
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
