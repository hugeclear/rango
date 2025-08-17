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

## Strict Compliance Prompt Pack (LaMP-2)

**目的**: 出力を `Answer: <TAG>` の単一行に強制し、形式準拠率 ≥ 0.95 を安定達成。

### ✅ 実測
- 形式準拠率: **98.0%**（目標 95% 超）
- テスト規模: **50 samples**
- 出力形式: 単一行 `Answer: <TAG>`
- デコード制約: `temperature=0, top_p=0, max_tokens=8, stop=["\n"]`
- 厳格検証パターン: `^Answer:\s*([A-Za-z0-9_\- ]+)\s*$`

### 🔧 プロンプト
**SYSTEM**
```
You are a strict single-line tag classifier.

RULES (絶対遵守):
1. Output EXACTLY one line: Answer: <TAG>
2. No explanations, no extra words, no punctuation after <TAG>, no emojis.
3. NO NEWLINES. Output must be a single line only.
4. Choose ONE best tag from the allowed list, case-sensitive.
5. If uncertain, still pick the single best tag.

FORBIDDEN:
• Multiple lines or trailing spaces
• Any text before/after Answer: <TAG>

Allowed tags: {{ALLOWED_TAGS}}
Required output format: Answer: <TAG>
```

**USER**
```
Task

Classify the following movie description into exactly one tag from the allowed list.

Description

{{QUESTION}}

User Profile (optional)

{{USER_PROFILE}}

Allowed tags (pick ONE, case-sensitive)

{{ALLOWED_TAGS}}

Output constraints (絶対遵守)
• Single line only.
• EXACT string format: Answer: <TAG>
• Nothing else before or after.
• No newline characters.

Your response

Answer: <TAG>
```

### 🧪 使い方
```bash
# 基本テスト
python run_strict_compliance_test.py --samples 10

# 実データでのテスト
python run_strict_compliance_test.py --data path/to/lamp2_test.jsonl

# カスタムプロンプト
python run_strict_compliance_test.py \
  --system-prompt prompts/lamp2_system_strict.txt \
  --user-template prompts/lamp2_user_template_strict.txt \
  --target-compliance 0.95
```

### 🎉 効果
• 無関係出力の排除（例: "Source: …" 等）
• LaMP-2向けに最適化された単一タグ選択
• 決定論的デコード設定で再現性担保
• ≥95% 準拠で評価の信頼性向上

---

📄 License
This project is licensed under the Apache License 2.0. See LICENSE for details.

📚 References
Chameleon: Personalized Prompt Editing for Large Language Models

PriME: Personalized Model Merging via Evolutionary Search

✍️ Acknowledgments
This repository was developed for academic research on privacy-preserving personalization using large language models, with support for datasets such as LaMP-2 and LaMP-3, and tested on NVIDIA A100×2 environment.
