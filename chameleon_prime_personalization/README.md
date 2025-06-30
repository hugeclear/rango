# Chameleon Prime Personalization

This project provides a template for experimenting with large language models and evolutionary model merging.

## Directory Structure

```
chameleon_prime_personalization/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── base_model/
│   └── user_adapters/
├── notebooks/
├── scripts/
│   ├── download_models.py
│   ├── download_datasets.py
│   └── setup_environment.sh
```

## Setup

1. Run the environment setup script:

```bash
bash scripts/setup_environment.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. Download a model (e.g., LLaMA 3.2 3B Instruct) into `models/base_model`:

```bash
python scripts/download_models.py meta-llama/Llama-3.2-3B-Instruct
```

（必要に応じて他のモデル名に置き換えてください）

3. Download a dataset

- **Hugging Face データセット** を取得する場合:
  ```bash
  python scripts/download_datasets.py <HF-dataset-name>
  ```
  （例: `imdb`, `glue` など）

**LaMP-2 (映画タグ分類タスク)** を取得・前処理する場合:
```bash
python scripts/download_datasets.py LaMP-2  # questions.json と answers.json、profile_rankings.json、merged.json を取得・生成します
```

4. Generate insights and tags via LLM

下記のように、raw データ配下の merged.json を入力にして、self_generate.py を実行します:

```bash
python ../self_generate.py \
  --input data/raw/LaMP-2/merged.json \
  --outdir processed/LaMP-2/selfgen/
```
