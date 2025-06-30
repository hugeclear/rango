import argparse
from pathlib import Path

from datasets import load_dataset
import subprocess
import tempfile
import shutil
import sys


def download_dataset(dataset_name: str, output_dir: str):
    if dataset_name == "LaMP-2":
        raw_task_dir = Path(output_dir) / dataset_name
        raw_task_dir.mkdir(parents=True, exist_ok=True)
        ques_url = (
            "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/new/dev/dev_questions.json"
        )
        ans_url = (
            "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/new/dev/dev_outputs.json"
        )
        qpath = raw_task_dir / "questions.json"
        apath = raw_task_dir / "answers.json"
        if not qpath.exists() or not apath.exists():
            try:
                import urllib.request

                print("Downloading LaMP-2 dev questions and outputs...")
                urllib.request.urlretrieve(ques_url, qpath)
                urllib.request.urlretrieve(ans_url, apath)
            except Exception as e:
                print(f"Error downloading LaMP-2 dev files: {e}")
                sys.exit(1)
        repo_dir = Path(tempfile.mkdtemp(prefix="LaMP-"))
        try:
            subprocess.run(
                [
                    "git", "clone", "https://github.com/LaMP-Benchmark/LaMP.git", str(repo_dir)
                ],
                check=True,
            )
            raw_task_dir.mkdir(parents=True, exist_ok=True)
            cwd = repo_dir / "LaMP"
            # patch rank_profiles.py: unpack the extra return value from classification_movies_query_corpus_maker
            rank_py = cwd / "rank_profiles.py"
            subprocess.run([
                "sed", "-i",
                "s/\(corpus, *query\) = classification_movies_query_corpus_maker/\1, _ = classification_movies_query_corpus_maker/",
                str(rank_py),
            ], check=True)
            subprocess.run(
                [
                    "python", "rank_profiles.py",
                    "--input_data_addr", str((raw_task_dir / "questions.json").resolve()),
                    "--output_ranking_addr", str((raw_task_dir / "profile_rankings.json").resolve()),
                    "--task", dataset_name,
                    "--ranker", "bm25",
                ],
                cwd=str(cwd),
                check=True,
            )
            processed_task_dir = Path(output_dir).parent / "processed" / dataset_name
            processed_task_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "python", "utils/merge_with_rank.py",
                    "--lamp_questions_addr", str((raw_task_dir / "questions.json").resolve()),
                    "--lamp_output_addr", str((raw_task_dir / "answers.json").resolve()),
                    "--profile_ranking_addr", str((raw_task_dir / "profile_rankings.json").resolve()),
                    "--merged_output_addr", str((processed_task_dir / "merged.json").resolve()),
                ],
                cwd=str(cwd),
                check=True,
            )
        finally:
            shutil.rmtree(str(repo_dir))
        return
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
