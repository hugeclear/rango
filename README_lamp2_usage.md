# LaMP-2 Paper-Compliant Implementation

Complete implementation of LaMP-2 following the original paper specifications with FlanT5-base default, IPA/FiD methods, BM25/Contriever retrievers, and PPEP→AIP prompting.

## ✅ Features Implemented

### 🎯 **Core Requirements (All Met)**

1. **✅ Model Layer**
   - Default: `google/flan-t5-base` (HuggingFace Transformers)
   - Optional: `google/flan-t5-xxl` zero-shot, `gpt-3.5-turbo` API support
   - FiD implemented for FlanT5 family only

2. **✅ Data Loading & Splits**
   - iU/iT 2-way split support
   - 15-class LaMP-2 vocabulary fixed
   - Accuracy/macro-F1 evaluation

3. **✅ Retrievers** 
   - BM25: Pyserini-style user profile retrieval
   - Contriever: Pre-trained cosine similarity
   - Configurable k (default: IPA k=8, FiD k=16)

4. **✅ PPEP→AIP Prompting**
   - LaMP-2 PPEP: `the tag for the movie: "[description]" is "[tag]"`
   - AIP: `concat(PPEP..., ", and ")` + input query
   - Table 5 reference comments included

5. **✅ IPA/FiD Inference**
   - IPA: Direct AIP prompting to FlanT5-base
   - FiD: 16-document separation encoding → decoder fusion

6. **✅ Training Setup**
   - Generation loss (15-tag text labels)
   - Default: 20 epochs (generation), 10 epochs (classification)
   - AdamW optimizer with linear warmup

7. **✅ Output Normalization & Mapping**
   - Preprocessing: lowercase, hyphen/space unification
   - BERTScore nearest neighbor mapping for OOV outputs
   - Raw → mapped logging

8. **✅ Evaluation & Logging**
   - One-command CLI for iU/iT × {Non-personalized, IPA, FiD}
   - Comprehensive metrics: retriever type, k, method, split, accuracy/F1, OOL rate

## 🚀 Quick Start

### Basic Usage

```bash
# Default evaluation (FlanT5-base, iU split, BM25, all methods)
python lamp2_paper_compliant.py

# Specific configuration
python lamp2_paper_compliant.py \
  --model google/flan-t5-base \
  --split iU \
  --retriever bm25 \
  --method all \
  --retrieval-k 8 \
  --fid-k 16
```

### Zero-shot Evaluation

```bash
# FlanT5-XXL zero-shot (IPA only)
python lamp2_paper_compliant.py \
  --model google/flan-t5-xxl \
  --zero-shot \
  --method ipa

# GPT-3.5 zero-shot (requires OpenAI API)
python lamp2_paper_compliant.py \
  --model gpt-3.5-turbo \
  --zero-shot \
  --method ipa
```

### Different Configurations

```bash
# iT split with Contriever retriever
python lamp2_paper_compliant.py \
  --split iT \
  --retriever contriever \
  --retrieval-k 12

# FiD-only evaluation
python lamp2_paper_compliant.py \
  --method fid \
  --fid-k 20
```

## 📊 Expected Output

```
================================================================================
📊 LaMP-2 Paper-Compliant Evaluation Results
================================================================================
Method               | Accuracy   | F1-Macro   | OOL Rate   | Samples  
--------------------------------------------------------------------------------
non_personalized    | 0.234      | 0.156      | 0.123      | 500     
ipa                  | 0.278      | 0.201      | 0.098      | 500     
fid                  | 0.291      | 0.215      | 0.087      | 500     

📋 Configuration:
  Split: iU
  Retriever: bm25

📊 BERTScore Mapping Statistics:
  non_personalized: mean=0.892, std=0.127
  ipa: mean=0.915, std=0.098
  fid: mean=0.923, std=0.091
```

## 🏗 Implementation Details

### PPEP→AIP Example

```python
# PPEP (Per-Profile Example)
ppep = 'the tag for the movie: "A romantic love story" is "romance"'

# AIP (Aggregated In-context)  
aip = 'the tag for the movie: "A love story" is "romance", and the tag for the movie: "A funny comedy" is "comedy", and the tag for the movie: "A new romantic comedy" is "'
```

### BERTScore Mapping

```python
# Example mappings
'romantic' -> 'romance' (conf: 0.987)
'funny movie' -> 'comedy' (conf: 0.923) 
'space adventure' -> 'sci-fi' (conf: 0.891)
```

### 15-Class LaMP-2 Vocabulary

```python
LAMP2_CLASSES = [
    'sci-fi', 'based on a book', 'comedy', 'action', 'twist ending',
    'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 
    'romance', 'thought-provoking', 'social commentary', 'violence', 'true story'
]
```

## 📋 Acceptance Criteria Status

- ✅ **FlanT5-base Non-personalized/IPA/FiD** runs on both iU/iT splits
- ✅ **FiD retrieval count** defaults to 16 (configurable)
- ✅ **BERTScore mapping** handles OOV outputs with accurate metrics
- ✅ **PPEP→AIP templates** implemented with Table 5 reference comments
- ✅ **Zero-shot support** for FlanT5-XXL/GPT-3.5 (IPA only)
- ✅ **Comprehensive logging** with retriever type, k values, OOL rates

## 🔧 Technical Architecture

```
LaMP2Evaluator
├── LaMP2DataLoader (iU/iT splits)
├── LaMP2Retriever (BM25/Contriever)  
├── LaMP2Model (FlanT5/GPT-3.5)
├── LaMP2Prompter (PPEP→AIP)
└── LaMP2OutputProcessor (BERTScore mapping)
```

## 📚 Paper References

- **LaMP-2 Template**: Table 5, lines 47-52 (movie tag classification)
- **Data Splits**: iU (user-based), iT (time-based) 
- **FiD Implementation**: 16-document separation encoding
- **Evaluation**: Accuracy, macro-F1 on 15-class vocabulary
- **Baselines**: Non-personalized, IPA, FiD comparison

---

**🎯 Research-Grade Implementation**: Ready for academic evaluation and comparison with LaMP paper baselines.