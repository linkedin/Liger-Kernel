import argparse

from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_dataset(tokenizer, text_file_path: str):
    """
    Tokenizes a text file where each line is a different example.
    Padding is applied to each example.
    """
    # Each line is a different example
    dataset = load_dataset("text", data_files={"train": text_file_path})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset["train"]


def generate_tokenized_dataset(tokenizer_path: str, text_file_path: str, output_dir: str) -> None:
    """
    Generate tokenized dataset from a text file, where each line is a different example.

    Args:
        tokenizer_path (str): Path to the directory containing the tokenizer files.
        text_file_path (str): Path to the text file to tokenize.
        output_dir (str): Directory where the tokenized dataset will be saved
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(tokenizer, text_file_path)
    train_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    # Example usage:
    # python generate_tokenized_dataset.py --tokenizer_path /shared/public/models/Mistral-7B --text_file_path ./../../resources/tiny_shakespeare.txt --output_dir ./../../resources/tiny_shakespeare_tokenized
    parser = argparse.ArgumentParser(description="Generate tokenized dataset from a text file.")

    # Add arguments
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the directory containing the tokenizer files.",
    )
    parser.add_argument(
        "--text_file_path",
        type=str,
        required=True,
        help="Path to the text file to tokenize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the tokenized dataset will be saved.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_tokenized_dataset(
        tokenizer_path=args.tokenizer_path,
        text_file_path=args.text_file_path,
        output_dir=args.output_dir,
    )
