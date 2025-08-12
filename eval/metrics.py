#!/usr/bin/env python3
"""
Extended metrics module for GraphRAG-CFS-Chameleon evaluation
Includes ROUGE-L and BERTScore in addition to existing metrics
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)

# STRICT MODE: Required dependencies, no fallback implementations
def _check_rouge_dependency():
    """Check if rouge-score is available"""
    try:
        from rouge_score import rouge_scorer
        return True
    except ImportError:
        return False

def _check_bertscore_dependency():
    """Check if bert-score is available"""
    try:
        from bert_score import score
        return True  
    except ImportError:
        return False

ROUGE_AVAILABLE = _check_rouge_dependency()
BERTSCORE_AVAILABLE = _check_bertscore_dependency()


def compute_exact_match(references: List[str], predictions: List[str]) -> float:
    """Compute exact match accuracy"""
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have same length")
    
    exact_matches = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return exact_matches / len(references)


def compute_f1_score(references: List[str], predictions: List[str]) -> float:
    """Compute average F1 score over all examples"""
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have same length")
    
    f1_scores = []
    for ref, pred in zip(references, predictions):
        f1 = _compute_single_f1(ref, pred)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def _compute_single_f1(reference: str, prediction: str) -> float:
    """Compute F1 score for a single reference-prediction pair"""
    ref_tokens = _normalize_text(reference).split()
    pred_tokens = _normalize_text(prediction).split()
    
    if len(ref_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    
    common_tokens = Counter(ref_tokens) & Counter(pred_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_bleu_score(references: List[str], predictions: List[str], n_gram: int = 4) -> float:
    """Compute BLEU score (simplified implementation)"""
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have same length")
    
    bleu_scores = []
    for ref, pred in zip(references, predictions):
        bleu = _compute_single_bleu(ref, pred, n_gram)
        bleu_scores.append(bleu)
    
    return np.mean(bleu_scores)


def _compute_single_bleu(reference: str, prediction: str, n_gram: int = 4) -> float:
    """Simplified BLEU score for single pair"""
    ref_tokens = _normalize_text(reference).split()
    pred_tokens = _normalize_text(prediction).split()
    
    if len(pred_tokens) == 0:
        return 0.0
    
    # Compute n-gram precision for n=1 to n_gram
    precisions = []
    for n in range(1, n_gram + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        pred_ngrams = _get_ngrams(pred_tokens, n)
        
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        common_ngrams = Counter(ref_ngrams) & Counter(pred_ngrams)
        precision = sum(common_ngrams.values()) / len(pred_ngrams)
        precisions.append(precision)
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        geo_mean = np.exp(np.mean(np.log(precisions)))
    else:
        geo_mean = 0.0
    
    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
    
    return bp * geo_mean


def compute_rouge_l(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score
    Returns precision, recall, and F1
    STRICT MODE: Requires rouge-score package, no fallback
    """
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have same length")
    
    if not ROUGE_AVAILABLE:
        raise RuntimeError("rouge-score package is required for ROUGE-L computation. "
                          "Install with: pip install rouge-score")
    
    return _compute_rouge_l_official(references, predictions)


def _compute_rouge_l_official(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Official ROUGE-L implementation using rouge-score library"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge_l = scores['rougeL']
        
        precision_scores.append(rouge_l.precision)
        recall_scores.append(rouge_l.recall)
        f1_scores.append(rouge_l.fmeasure)
    
    return {
        'rouge_l_precision': np.mean(precision_scores),
        'rouge_l_recall': np.mean(recall_scores),
        'rouge_l_f1': np.mean(f1_scores)
    }


# Fallback implementation removed in strict mode


def compute_bertscore(
    references: List[str], 
    predictions: List[str], 
    model_type: str = "microsoft/deberta-base-mnli",
    use_fast_tokenizer: bool = True
) -> Dict[str, float]:
    """
    Compute BERTScore
    Returns precision, recall, and F1
    STRICT MODE: Requires bert-score package, no fallback
    """
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have same length")
    
    if not BERTSCORE_AVAILABLE:
        raise RuntimeError("bert-score package is required for BERTScore computation. "
                          "Install with: pip install bert-score")
    
    return _compute_bertscore_official(references, predictions, model_type, use_fast_tokenizer)


def _compute_bertscore_official(
    references: List[str], 
    predictions: List[str], 
    model_type: str,
    use_fast_tokenizer: bool
) -> Dict[str, float]:
    """Official BERTScore implementation - STRICT MODE"""
    from bert_score import score
    
    try:
        # Use specified model
        P, R, F1 = score(
            predictions, 
            references, 
            model_type=model_type,
            verbose=False,
            device='cuda'
        )
        
        return {
            'bertscore_precision': float(P.mean()),
            'bertscore_recall': float(R.mean()),
            'bertscore_f1': float(F1.mean())
        }
    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        raise RuntimeError(f"BERTScore computation failed: {e}. Check model availability and GPU access.")


# BERTScore fallback implementation removed in strict mode


# Utility functions

def _normalize_text(text: str) -> str:
    """Normalize text for consistent comparison"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from token list"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _longest_common_subsequence_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute LCS length using dynamic programming"""
    m, n = len(seq1), len(seq2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_all_metrics(
    references: List[str], 
    predictions: List[str],
    include_bertscore: bool = True,
    bertscore_model: str = "microsoft/deberta-base-mnli"
) -> Dict[str, float]:
    """
    Compute all available metrics
    
    Args:
        references: Reference texts
        predictions: Predicted texts
        include_bertscore: Whether to compute BERTScore (can be slow)
        bertscore_model: Model to use for BERTScore
        
    Returns:
        Dictionary with all metric scores
    """
    metrics = {}
    
    # Basic metrics
    metrics['exact_match'] = compute_exact_match(references, predictions)
    metrics['f1_score'] = compute_f1_score(references, predictions)
    metrics['bleu_score'] = compute_bleu_score(references, predictions)
    
    # ROUGE-L
    rouge_scores = compute_rouge_l(references, predictions)
    metrics.update(rouge_scores)
    
    # BERTScore (optional, can be slow)
    if include_bertscore:
        bert_scores = compute_bertscore(references, predictions, bertscore_model)
        metrics.update(bert_scores)
    
    logger.info(f"Computed metrics for {len(references)} examples")
    
    return metrics