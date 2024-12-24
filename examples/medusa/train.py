# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

import json
import os
import pathlib

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Optional

import torch
import transformers

from callback import EfficiencyCallback
from medusa_util import add_medusa_heads
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from liger_kernel.transformers import apply_liger_kernel_to_llama

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")


@dataclass
class DataArguments:
    data_path: str = field(
        default="Aeala/ShareGPT_Vicuna_unfiltered",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    medusa_heads_coefficient: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the Medusa heads."},
    )
    medusa_decay_coefficient: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the Medusa heads."},
    )
    medusa_scheduler: str = field(
        default="constant",
        metadata={"help": "Scheduler for the Medusa heads."},
    )
    medusa_lr_multiplier: float = field(
        default=0.0,
        metadata={"help": "Learning rate multiplier for the Medusa heads."},
    )
    medusa_return: bool = field(
        default=False,
        metadata={
            "help": "If medusa is not applied, the default is False, and the regular lm_head will be used for single-token prediction."
        },
    )
    medusa_only_heads: bool = field(
        default=False,
        metadata={"help": "If train medusa heads only, default is False, the whole model will be trained"},
    )
    use_liger: bool = field(
        default=False,
        metadata={"help": "If apply liger kernel to the model."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # import pdb; pdb.set_trace()
    for conversation in sources[:50]:
        tokenizer_compatible_conv = [
            {
                "role": "user" if c["from"] == "human" else "assistant",
                "content": c["value"],
            }
            for c in conversation["conversations"]
        ]
        prompt = tokenizer.apply_chat_template(tokenizer_compatible_conv, tokenize=False)
        prompts.append(prompt)
        conversations.append(tokenizer_compatible_conv)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts, strict=False)):
        # print(conv_index)
        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = prompt.index(content.strip())
                # stop = start + len(content)
                indices = []
                for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                    if tok_stop >= start or tok_start < tok_stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, test_size=0.05) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.
        test_size: evaluation data ratio (default: 0.05)

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    rank0_print("Loading data...")

    # Load the entire dataset
    train_json = json.load(open(data_args.data_path, "r"))

    # Perform a train-test split based on test_size
    train_data, eval_data = train_test_split(train_json, test_size=test_size, random_state=42)
    # Create the train and eval datasets
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    # Making sure the tokenizer works before loading the model.
    print(tokenizer(["This is a test", "secondary"], padding=True))
    print(tokenizer.apply_chat_template([{"role": "user", "content": "This is a test"}]))

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    if training_args.use_liger is True:
        apply_liger_kernel_to_llama()

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    add_medusa_heads(
        model,
        training_args.medusa_num_heads,
        training_args.medusa_num_layers,
        training_args.medusa_return,
        training_args.medusa_only_heads,
        training_args.use_liger,
    )
    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EfficiencyCallback()],
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if training_args.medusa_return and training_args.medusa_only_heads:
        # Save only the updated head without saving the backbone model
        if hasattr(model, "module"):
            lm_head = model.module.medusa_head
        else:
            lm_head = model.medusa_head

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = lm_head.state_dict()

        # Save Medusa heads
        if local_rank == 0:
            save_file(
                state_dict,
                os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
            )
    else:
        # Save the whole model weight
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
