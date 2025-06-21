import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(model_name: str, output_dir: str):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=dest, resume_download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF model")
    parser.add_argument("model_name", help="Model name on HuggingFace")
    parser.add_argument(
        "--output_dir",
        default="./models/base_model",
        help="Directory to store downloaded model",
    )
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir)
