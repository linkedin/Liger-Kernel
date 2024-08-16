from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer_path="/shared/public/models/Mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the shakespeare dataset
def prepare_dataset(tokenizer, file_path="/home/jobuser/resources/liger-kernel/test/convergence/tiny_shakespeare.txt"):
    # Each line is a different example
    dataset = load_dataset("text", data_files={"train": file_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset["train"]

train_dataset = prepare_dataset(tokenizer)

train_dataset.save_to_disk("/home/jobuser/Liger-Kernel/test/resources/tiny_shakespeare_tokenized")
