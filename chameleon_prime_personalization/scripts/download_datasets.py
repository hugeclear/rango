import argparse
from pathlib import Path

from datasets import load_dataset


def download_dataset(dataset_name: str, output_dir: str):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from HF")
    parser.add_argument("dataset_name", help="Dataset name on HuggingFace")
    parser.add_argument(
        "--output_dir",
        default="./data/raw",
        help="Directory to save dataset",
    )
    args = parser.parse_args()
    download_dataset(args.dataset_name, args.output_dir)
