#!/usr/bin/env python3
"""
Enhanced retrieval pipeline with diversity and clustering for GraphRAG-CFS-Chameleon
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from .diversity import quantile_filter, select_with_diversity, compute_diversity_metrics
from .cluster import cluster_assign, intra_cluster_diversity, balance_cluster_selection

logger = logging.getLogger(__name__)


class EnhancedCFSRetriever:
    """
    Enhanced CFS retriever with diversity and clustering support
    """
    
    def __init__(
        self,
        cfs_pool_path: str,
        user_embeddings_path: str,
        config: Dict[str, Any]
    ):
        self.cfs_pool_path = Path(cfs_pool_path)
        self.user_embeddings_path = Path(user_embeddings_path)
        self.config = config
        
        # Load CFS pool and embeddings
        self._load_data()
        
        # Configuration
        self.diversity_enabled = config.get('diversity', {}).get('enabled', False)
        self.diversity_method = config.get('diversity', {}).get('method', 'mmr')
        self.diversity_lambda = config.get('diversity', {}).get('lambda', 0.3)
        self.q_quantile = config.get('selection', {}).get('q_quantile', 0.8)
        self.clustering_enabled = config.get('clustering', {}).get('enabled', False)
        self.clustering_algorithm = config.get('clustering', {}).get('algorithm', 'kmeans')
        self.max_per_cluster = config.get('clustering', {}).get('max_per_cluster', 10)
        
        logger.info(f"EnhancedCFSRetriever initialized: diversity={self.diversity_enabled}, "
                   f"clustering={self.clustering_enabled}")
    
    def _load_data(self):
        """Load CFS pool and user embeddings"""
        try:
            self.cfs_pool = pd.read_parquet(self.cfs_pool_path)
            self.user_embeddings = np.load(self.user_embeddings_path)
            
            logger.info(f"Loaded CFS pool: {len(self.cfs_pool)} edges")
            logger.info(f"Loaded user embeddings: {self.user_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def retrieve_collaborative_users(
        self,
        query_user_id: str,
        k: int = 50,
        min_weight: float = 1e-4
    ) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
        """
        Retrieve collaborative users with optional diversity and clustering
        
        Args:
            query_user_id: Target user ID
            k: Number of users to retrieve
            min_weight: Minimum collaboration weight threshold
            
        Returns:
            Selected user IDs, their weights, and metadata
        """
        # Get candidates from CFS pool
        user_pool = self.cfs_pool[self.cfs_pool['user_id'] == query_user_id].copy()
        
        if len(user_pool) == 0:
            logger.warning(f"No collaborative users found for user {query_user_id}")
            return [], np.array([]), {}
        
        # Filter by minimum weight
        user_pool = user_pool[user_pool['weight'] >= min_weight]
        
        if len(user_pool) == 0:
            logger.warning(f"No users above weight threshold {min_weight} for user {query_user_id}")
            return [], np.array([]), {}
        
        # Extract data
        candidate_user_ids = user_pool['neighbor_id'].tolist()
        candidate_weights = user_pool['weight'].values
        
        # Get embeddings for candidates
        candidate_embeddings = self._get_user_embeddings(candidate_user_ids)
        
        # Apply quantile filtering
        if self.q_quantile < 1.0:
            candidate_weights, filtered_data = quantile_filter(
                candidate_weights, 
                list(zip(candidate_user_ids, candidate_embeddings)),
                self.q_quantile
            )
            candidate_user_ids, candidate_embeddings = zip(*filtered_data)
            candidate_user_ids = list(candidate_user_ids)
            candidate_embeddings = np.array(candidate_embeddings)
            
            logger.debug(f"After quantile filtering: {len(candidate_user_ids)} candidates")
        
        # Apply clustering if enabled
        cluster_labels = None
        if self.clustering_enabled and len(candidate_user_ids) > 2:
            cluster_labels = cluster_assign(
                candidate_embeddings,
                algorithm=self.clustering_algorithm,
                random_state=42
            )
            
            # Balance selection across clusters
            if self.max_per_cluster > 0:
                balanced_data = balance_cluster_selection(
                    list(zip(candidate_user_ids, candidate_embeddings)),
                    cluster_labels,
                    candidate_weights,
                    self.max_per_cluster
                )
                filtered_data, selected_indices = balanced_data
                candidate_user_ids, candidate_embeddings = zip(*filtered_data)
                candidate_user_ids = list(candidate_user_ids)
                candidate_embeddings = np.array(candidate_embeddings)
                candidate_weights = candidate_weights[selected_indices]
                cluster_labels = cluster_labels[selected_indices]
        
        # Apply diversity selection
        if self.diversity_enabled and len(candidate_user_ids) > k:
            diverse_data = select_with_diversity(
                list(zip(candidate_user_ids, candidate_embeddings)),
                candidate_weights,
                candidate_embeddings,
                k,
                method=self.diversity_method,
                lambda_div=self.diversity_lambda
            )
            selected_data, selected_indices = diverse_data
            final_user_ids, final_embeddings = zip(*selected_data)
            final_user_ids = list(final_user_ids)
            final_embeddings = np.array(final_embeddings)
            final_weights = candidate_weights[selected_indices]
            
            if cluster_labels is not None:
                final_cluster_labels = cluster_labels[selected_indices]
            else:
                final_cluster_labels = None
        else:
            # Simple top-k selection by weight
            if len(candidate_user_ids) > k:
                top_indices = np.argsort(candidate_weights)[::-1][:k]
                final_user_ids = [candidate_user_ids[i] for i in top_indices]
                final_embeddings = candidate_embeddings[top_indices]
                final_weights = candidate_weights[top_indices]
                final_cluster_labels = cluster_labels[top_indices] if cluster_labels is not None else None
            else:
                final_user_ids = candidate_user_ids
                final_embeddings = candidate_embeddings
                final_weights = candidate_weights
                final_cluster_labels = cluster_labels
        
        # Compute metadata
        metadata = {
            "query_user_id": query_user_id,
            "n_candidates_initial": len(user_pool),
            "n_candidates_after_quantile": len(candidate_user_ids),
            "n_selected": len(final_user_ids),
            "diversity_enabled": self.diversity_enabled,
            "clustering_enabled": self.clustering_enabled,
            "weight_min": float(np.min(final_weights)),
            "weight_max": float(np.max(final_weights)),
            "weight_mean": float(np.mean(final_weights))
        }
        
        if self.diversity_enabled:
            diversity_metrics = compute_diversity_metrics(final_embeddings)
            metadata.update(diversity_metrics)
        
        if self.clustering_enabled and final_cluster_labels is not None:
            from .cluster import compute_cluster_distribution
            cluster_stats = compute_cluster_distribution(final_cluster_labels)
            metadata["cluster_stats"] = cluster_stats
        
        logger.debug(f"Retrieved {len(final_user_ids)} collaborative users for user {query_user_id}")
        
        return final_user_ids, final_weights, metadata
    
    def _get_user_embeddings(self, user_ids: List[str]) -> np.ndarray:
        """
        Get embeddings for given user IDs
        This is a placeholder - implement based on your embedding storage format
        """
        # For now, return random embeddings
        # In real implementation, map user_ids to embedding indices
        n_users = len(user_ids)
        embedding_dim = self.user_embeddings.shape[1] if len(self.user_embeddings.shape) > 1 else 768
        
        # Mock implementation: random embeddings
        np.random.seed(42)  # For reproducibility
        embeddings = np.random.randn(n_users, embedding_dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        logger.debug(f"Generated embeddings for {n_users} users with dimension {embedding_dim}")
        
        return embeddings