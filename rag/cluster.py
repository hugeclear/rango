#!/usr/bin/env python3
"""
Clustering module for GraphRAG-CFS-Chameleon diversity control
Implements HDBSCAN and K-means clustering with intra-cluster diversity
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available, clustering features disabled")
    CLUSTERING_AVAILABLE = False


def cluster_assign(
    embeddings: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Assign cluster labels to embeddings
    
    Args:
        embeddings: Input embedding vectors
        algorithm: Clustering algorithm ("kmeans" or "hdbscan")
        n_clusters: Number of clusters for k-means
        min_cluster_size: Minimum cluster size for HDBSCAN
        random_state: Random seed
        
    Returns:
        Cluster labels for each embedding
    """
    if not CLUSTERING_AVAILABLE:
        # Fallback: assign all to cluster 0
        logger.warning("Clustering not available, assigning all items to cluster 0")
        return np.zeros(len(embeddings), dtype=int)
    
    if algorithm == "kmeans":
        if n_clusters is None:
            # Heuristic: sqrt(n) clusters
            n_clusters = max(2, int(np.sqrt(len(embeddings))))
            
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        labels = clusterer.fit_predict(embeddings)
        
    elif algorithm == "hdbscan":
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine',
            cluster_selection_epsilon=0.1
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Handle noise points (-1 label)
        noise_mask = labels == -1
        if noise_mask.any():
            # Assign noise points to nearest cluster
            valid_labels = labels[~noise_mask]
            if len(valid_labels) > 0:
                # Simple assignment: all noise points to most common cluster
                most_common_cluster = np.bincount(valid_labels).argmax()
                labels[noise_mask] = most_common_cluster
            else:
                # All points are noise, assign to cluster 0
                labels[:] = 0
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    logger.debug(f"Clustering with {algorithm}: {len(embeddings)} items -> {len(set(labels))} clusters")
    
    return labels


def intra_cluster_diversity(
    candidates: List[Any],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    sim_matrix: Optional[np.ndarray] = None,
    min_gap: float = 0.1
) -> List[Any]:
    """
    Enforce diversity within each cluster
    
    Args:
        candidates: List of candidate items
        embeddings: Embedding vectors
        cluster_labels: Cluster assignment for each candidate
        sim_matrix: Precomputed similarity matrix (optional)
        min_gap: Minimum similarity gap within cluster
        
    Returns:
        Filtered candidates maintaining intra-cluster diversity
    """
    if sim_matrix is None:
        sim_matrix = cosine_similarity(embeddings)
    
    selected_indices = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) <= 1:
            selected_indices.extend(cluster_indices)
            continue
        
        # Within this cluster, select diverse items
        selected_in_cluster = []
        remaining_in_cluster = list(cluster_indices)
        
        # Start with first item (could be sorted by relevance score)
        selected_in_cluster.append(remaining_in_cluster[0])
        remaining_in_cluster.remove(remaining_in_cluster[0])
        
        # Add diverse items
        for candidate_idx in remaining_in_cluster:
            can_add = True
            for selected_idx in selected_in_cluster:
                if sim_matrix[candidate_idx, selected_idx] > (1.0 - min_gap):
                    can_add = False
                    break
            
            if can_add:
                selected_in_cluster.append(candidate_idx)
        
        selected_indices.extend(selected_in_cluster)
    
    selected_candidates = [candidates[i] for i in selected_indices]
    
    logger.debug(f"Intra-cluster diversity: {len(candidates)} -> {len(selected_candidates)} candidates")
    
    return selected_candidates


def compute_cluster_distribution(cluster_labels: np.ndarray) -> Dict[str, Union[int, float, List[int]]]:
    """
    Compute statistics about cluster distribution
    
    Args:
        cluster_labels: Cluster assignment for each item
        
    Returns:
        Dictionary with cluster statistics
    """
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    stats = {
        "n_clusters": len(unique_labels),
        "cluster_sizes": counts.tolist(),
        "avg_cluster_size": float(np.mean(counts)),
        "std_cluster_size": float(np.std(counts)),
        "largest_cluster_size": int(np.max(counts)),
        "smallest_cluster_size": int(np.min(counts))
    }
    
    return stats


def balance_cluster_selection(
    candidates: List[Any],
    cluster_labels: np.ndarray,
    scores: np.ndarray,
    max_per_cluster: int = 10
) -> Tuple[List[Any], List[int]]:
    """
    Balance selection across clusters to avoid single-cluster dominance
    
    Args:
        candidates: List of candidate items
        cluster_labels: Cluster assignment for each candidate
        scores: Relevance scores for each candidate
        max_per_cluster: Maximum items to select from each cluster
        
    Returns:
        Balanced candidates and their indices
    """
    selected_indices = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_scores = scores[cluster_indices]
        
        # Sort by score within cluster
        sorted_cluster_indices = cluster_indices[np.argsort(cluster_scores)[::-1]]
        
        # Take top items from this cluster
        n_select = min(max_per_cluster, len(sorted_cluster_indices))
        selected_indices.extend(sorted_cluster_indices[:n_select])
    
    selected_candidates = [candidates[i] for i in selected_indices]
    
    logger.debug(f"Balanced cluster selection: {len(candidates)} -> {len(selected_candidates)} "
                f"from {len(unique_clusters)} clusters (max {max_per_cluster} per cluster)")
    
    return selected_candidates, selected_indices