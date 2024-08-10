from dataclasses import dataclass, field

import datasets
import torch
import transformers
import trl
from liger_kernel.transformers import apply_liger_kernel_to_llama
from profiler import ProfilerCallback
import time


@dataclass
class CustomArguments:
    model_path: str
    data_path: str
    max_seq_length: int = field(default=512)
    liger_kernel: bool = field(default=False)


def formatting_func(example):
    output = ""
    output += f"TEXT: {example['text']} "

    for i in range(len(example["summary"])):
        output += f"SUMMARY:{i} {example['summary'][i]} "

    return [output]


class EfficiencyCallback(transformers.TrainerCallback):

    def __init__(self, n_warmup_steps=2):
        self.n_warmup_steps = n_warmup_steps


    def on_train_begin(self, args, state, control, **kwargs):
        print("warm up step starts...")

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        
        if state.global_step == self.n_warmup_steps:
            print("warm up step ends...")
            self.start_time = time.time()
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training.
        """
        print(f"Training took {time.time() - self.start_time:.2f} seconds")
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Peak memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")



def main():
    parser = transformers.HfArgumentParser(
        (transformers.TrainingArguments, CustomArguments)
    )
    training_args, custom_args = parser.parse_args_into_dataclasses()

    if custom_args.liger_kernel:
        apply_liger_kernel_to_llama()

    dataset = datasets.load_dataset(path=custom_args.data_path)
    dataset_train, dataset_eval = dataset["train"], dataset["test"]

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
        callbacks=[EfficiencyCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    main()
