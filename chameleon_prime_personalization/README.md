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

2. Download a model (e.g., LLaMA 2 7B) into `models/base_model`:

```bash
python scripts/download_models.py meta-llama/Llama-2-7b-hf
```

3. Download a dataset (e.g., LaMP) into `data/raw`:

```bash
python scripts/download_datasets.py LaMP
```

You can replace the model or dataset names with any available on the HuggingFace Hub.
