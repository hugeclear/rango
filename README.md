# rango
Personalized LLM via Chameleon approach

 🧠 Chameleon + PriME: Personalized & Privacy-Preserving LLM Optimization

This repository contains code, configuration, and experimental framework for a three-month research project that integrates **Chameleon-style embedding personalization** and **PriME-style evolutionary model merging** to achieve **efficient, private, and high-performance LLM-based recommendation systems**.

## 🚀 Project Overview

This project aims to:
- Personalize LLMs using user-specific directions in embedding space (via Chameleon)
- Optimize merged models via evolutionary algorithms (PriME)
- Support multi-user inference with limited data and strong privacy constraints

## 📁 Project Structure

chameleon_prime_personalization/
├── configs/ # YAML/JSON configs for models, training, evaluation
├── data/
│ ├── raw/ # Raw datasets (LaMP-2, LaMP-3, etc.)
│ └── processed/ # Tokenized or embedded versions
├── models/
│ ├── base_model/ # Base LLM (e.g., LLaMA 2 7B or Mistral)
│ └── user_adapters/ # LoRA/PEFT modules for users
├── notebooks/ # Experiment tracking or visualization notebooks
├── scripts/
│ ├── download_models.py
│ ├── download_datasets.py
│ └── setup_environment.sh
└── README.md




## 🛠️ Setup

### 1. Environment
```bash
bash scripts/setup_environment.sh
source env/bin/activate
2. Download base models

python scripts/download_models.py --model mistralai/Mistral-7B-v0.1
3. Download datasets

python scripts/download_datasets.py --dataset lamp2
🧩 Method Summary
Embedding Personalization (Chameleon)
SVD-based decomposition of user history embeddings

Extracts user-specific direction vs general direction

Adjust embeddings by shifting along user-personalized vector

Evolutionary Merge (PriME)
Each user gets LoRA/IA3 fine-tuned PEFT modules

Cosine similarity used to identify similar shared users

Evolutionary strategy (CMA-ES or NSGA-II) to merge modules

Objective: maximize user utility (F1, ROUGE), minimize privacy leakage

📊 Evaluation Metrics
ROUGE / F1 / BLEU (text output quality)

Cosine similarity / KL divergence (privacy leakage)

Model diversity and memorization metrics (advanced)

📄 License
This project is licensed under the Apache License 2.0. See LICENSE for details.

📚 References
Chameleon: Personalized Prompt Editing for Large Language Models

PriME: Personalized Model Merging via Evolutionary Search

✍️ Acknowledgments
This repository was developed for academic research on privacy-preserving personalization using large language models, with support for datasets such as LaMP-2 and LaMP-3, and tested on NVIDIA A100×2 environment.
