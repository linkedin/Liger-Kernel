import os

from dataclasses import dataclass

import datasets
import torch
import transformers

from callback import EfficiencyCallback
from datasets import Image as ImageFeature
from trl import SFTTrainer

from liger_kernel.transformers import monkey_patch


@dataclass
class CustomArguments:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset: str = "HuggingFaceM4/the_cauldron"
    dataset_subset: str = "ai2d"
    dataset_split: str = "train"
    max_seq_length: int = 512
    dataset_text_field: str = "texts"
    use_liger: bool = False


def construct_model_and_processor(model_name: str, use_liger: bool) -> torch.nn.Module:
    if "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration

        # These settings are used to reduce the memory footprint of the Qwen2-VL model,
        # which supports training/inferences on images in their native resolution. Large
        # images -> many visual tokens (a max of 16384) -> large memory consumption.
        # If fine-tuning for a real-world application, consider these values carefully.
        min_visual_tokens_per_image = 256
        max_visual_tokens_per_image = 256

        processor = transformers.AutoProcessor.from_pretrained(
            model_name,
            padding_side="left",
            truncation_side="left",
            min_pixels=min_visual_tokens_per_image * 28 * 28,  # patch size is 14x14
            max_pixels=max_visual_tokens_per_image * 28 * 28,  # 4 patches / token
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        if use_liger:
            print("Applying Liger Kernel to Qwen2-VL model")
            monkey_patch.apply_liger_kernel_to_qwen2_vl(
                # These args can be used to override the default Liger settings
                # cross_entropy=True,
                # fused_linear_cross_entropy=False,
            )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        return model, processor, image_token_id

    raise NotImplementedError(f"Model {model_name} not supported")


def _validate_and_extract_the_cauldron(examples) -> dict[str, list]:
    batch_texts = []
    batch_images = []
    for images, texts in zip(examples["images"], examples["texts"], strict=False):
        if not images:
            raise ValueError("No image found in example from the_cauldron dataset")
        if len(images) > 1:
            raise ValueError("Only one image per example is supported")
        batch_texts.extend(texts)
        batch_images.extend([images[0]] * len(texts))
    return {"texts": batch_texts, "images": batch_images}


def _format_for_convo(example, tokenizer):
    # cauldron data is already in message format {"user": ..., "assistant": ...}
    text = example["texts"]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text["user"]}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": text["assistant"]}]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"texts": text}


def train():
    parser = transformers.HfArgumentParser((transformers.TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False  # required to not drop the image column
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    model, processor, image_token_id = construct_model_and_processor(custom_args.model_name, custom_args.use_liger)

    dataset = (
        datasets.load_dataset(
            custom_args.dataset,
            custom_args.dataset_subset,
            split=custom_args.dataset_split,
        )
        .map(
            _validate_and_extract_the_cauldron,
            batched=True,
            num_proc=min(os.cpu_count(), 16),
            desc="Extracting text and images",
        )
        .map(
            _format_for_convo,
            fn_kwargs={"tokenizer": processor.tokenizer},
            desc="Formatting for convo",
        )
        .cast_column("images", ImageFeature())
        .train_test_split(test_size=0.1)
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def collate_fn(examples):
        """
        Taken directly from the TRL documentation with minor modifications:
        https://huggingface.co/docs/trl/en/sft_trainer#a-custom-collator-for-processing-multi-modal-data

        Modifications:
        1. `apply_chat_template` is used to preprocess the texts before training begins (see above)
        2. `example["messages"]` -> `example["texts"]` to conform with the_cauldron dataset schema
        3. Ignoring image tokens in the loss computation
        """
        # Get the texts and images
        texts = [example["texts"] for example in examples]
        images = [example["images"] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        max_seq_length=custom_args.max_seq_length,
        dataset_text_field=custom_args.dataset_text_field,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        callbacks=[EfficiencyCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    train()
