import os
import torch
import transformers
import datasets
from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
from trl.trainer import ConstantLengthDataset
from datasets import Image as ImageFeature
from liger_kernel.transformers import monkey_patch


@dataclass
class CustomArguments:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset: str = "HuggingFaceM4/the_cauldron"
    dataset_subset: str = "ai2d"
    dataset_split: str = "train"
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

    model, processor, image_token_id = construct_model_and_processor(
        custom_args.model_name, custom_args.use_liger
    )

    dataset = datasets.load_dataset(
        custom_args.dataset, 
        custom_args.dataset_subset, 
        split=custom_args.dataset_split
    )

    train_dataset, eval_dataset = prepare_dataset(dataset, processor, image_token_id)

    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.num_train_epochs,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    trainer.train()


if __name__ == "__main__":
    train()