#!/usr/bin/env python3
"""
Diversity selection module for GraphRAG-CFS-Chameleon
Implements MMR (Maximal Marginal Relevance) and quantile filtering
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def quantile_filter(scores: np.ndarray, candidates: List[Any], q: float = 0.8) -> Tuple[np.ndarray, List[Any]]:
    """
    Filter candidates by keeping only top q quantile
    
    Args:
        scores: Array of relevance scores
        candidates: List of candidate items
        q: Quantile threshold (0.8 = keep top 20%)
        
    Returns:
        Filtered scores and candidates
    """
    if q >= 1.0:
        return scores, candidates
        
    threshold = np.quantile(scores, q)
    mask = scores >= threshold
    
    filtered_scores = scores[mask]
    filtered_candidates = [candidates[i] for i in range(len(candidates)) if mask[i]]
    
    logger.debug(f"Quantile filter q={q}: {len(candidates)} -> {len(filtered_candidates)} candidates")
    
    return filtered_scores, filtered_candidates


def select_with_diversity(
    candidates: List[Any],
    scores: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    method: str = "mmr",
    lambda_div: float = 0.3,
    min_similarity_gap: float = 0.1
) -> Tuple[List[Any], List[int]]:
    """
    Select diverse candidates using MMR or greedy diversity
    
    Args:
        candidates: List of candidate items
        scores: Relevance scores for each candidate
        embeddings: Embedding vectors for similarity computation
        k: Number of items to select
        method: Selection method ("mmr" or "greedy")
        lambda_div: Diversity weight (0=pure relevance, 1=pure diversity)
        min_similarity_gap: Minimum similarity gap between selected items
        
    Returns:
        Selected candidates and their indices
    """
    if len(candidates) <= k:
        return candidates, list(range(len(candidates)))
        
    if method == "mmr":
        return _select_mmr(candidates, scores, embeddings, k, lambda_div)
    elif method == "greedy":
        return _select_greedy_diversity(candidates, scores, embeddings, k, min_similarity_gap)
    else:
        raise ValueError(f"Unknown diversity method: {method}")


def _select_mmr(
    candidates: List[Any],
    scores: np.ndarray, 
    embeddings: np.ndarray,
    k: int,
    lambda_div: float
) -> Tuple[List[Any], List[int]]:
    """Maximal Marginal Relevance selection"""
    n_candidates = len(candidates)
    selected_indices = []
    remaining_indices = list(range(n_candidates))
    
    # Select first item with highest relevance
    first_idx = np.argmax(scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    for _ in range(k - 1):
        if not remaining_indices:
            break
            
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance score
            relevance = scores[idx]
            
            # Max similarity to already selected items
            max_sim = max(sim_matrix[idx, sel_idx] for sel_idx in selected_indices)
            
            # MMR score
            mmr_score = lambda_div * relevance - (1 - lambda_div) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    selected_candidates = [candidates[i] for i in selected_indices]
    
    logger.debug(f"MMR selection: {len(candidates)} -> {len(selected_candidates)} with λ={lambda_div}")
    
    return selected_candidates, selected_indices


def _select_greedy_diversity(
    candidates: List[Any],
    scores: np.ndarray,
    embeddings: np.ndarray, 
    k: int,
    min_gap: float
) -> Tuple[List[Any], List[int]]:
    """Greedy diversity selection with minimum similarity gap"""
    n_candidates = len(candidates)
    selected_indices = []
    remaining_indices = list(range(n_candidates))
    
    # Sort by relevance score descending
    sorted_indices = np.argsort(scores)[::-1]
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    for idx in sorted_indices:
        if len(selected_indices) >= k:
            break
            
        if idx not in remaining_indices:
            continue
            
        # Check diversity constraint
        can_select = True
        for sel_idx in selected_indices:
            if sim_matrix[idx, sel_idx] > (1.0 - min_gap):
                can_select = False
                break
                
        if can_select:
            selected_indices.append(idx)
            remaining_indices.remove(idx)
    
    # Fill remaining slots if needed (relax constraint)
    while len(selected_indices) < k and remaining_indices:
        # Pick highest scoring remaining item
        remaining_scores = [scores[i] for i in remaining_indices]
        best_remaining = remaining_indices[np.argmax(remaining_scores)]
        selected_indices.append(best_remaining)
        remaining_indices.remove(best_remaining)
    
    selected_candidates = [candidates[i] for i in selected_indices]
    
    logger.debug(f"Greedy diversity: {len(candidates)} -> {len(selected_candidates)} with gap={min_gap}")
    
    return selected_candidates, selected_indices


def compute_diversity_metrics(selected_embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute diversity metrics for selected candidates
    
    Args:
        selected_embeddings: Embeddings of selected candidates
        
    Returns:
        Dictionary with diversity metrics
    """
    if len(selected_embeddings) <= 1:
        return {"avg_pairwise_similarity": 0.0, "min_pairwise_similarity": 0.0}
    
    sim_matrix = cosine_similarity(selected_embeddings)
    
    # Get upper triangle (exclude diagonal and duplicates)
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    metrics = {
        "avg_pairwise_similarity": float(np.mean(upper_triangle)),
        "min_pairwise_similarity": float(np.min(upper_triangle)),
        "max_pairwise_similarity": float(np.max(upper_triangle)),
        "std_pairwise_similarity": float(np.std(upper_triangle))
    }
    
    return metrics