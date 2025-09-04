#!/usr/bin/env python3
"""
Causal Graph Builder for Chameleon System
Implements PC algorithm for causal discovery from user interaction data

Integrates with existing LaMP-2 user profiles and history data
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT
    from causallearn.utils.GraphUtils import GraphUtils
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    logging.warning("causal-learn not available. Run: pip install causal-learn==0.1.3")

logger = logging.getLogger(__name__)

@dataclass
class CausalGraphResult:
    """Result of causal graph construction"""
    adjacency_matrix: np.ndarray
    node_names: List[str]
    edge_weights: np.ndarray
    alpha_threshold: float
    n_samples: int
    discovery_time: float

class CausalGraphBuilder:
    """
    Build causal graphs from user interaction sequences using PC algorithm
    
    Integrates with existing Chameleon data structures:
    - Uses LaMP-2 user profiles as observational data
    - Extracts temporal sequences from user history
    - Creates causal constraints for editing
    """
    
    def __init__(self, alpha: float = 0.05, max_parents: int = 5, 
                 min_samples: int = 10, cache_dir: Optional[str] = None):
        """
        Initialize causal graph builder
        
        Args:
            alpha: Significance level for PC algorithm
            max_parents: Maximum number of causal parents per node
            min_samples: Minimum samples required for reliable inference
            cache_dir: Directory to cache computed graphs
        """
        self.alpha = alpha
        self.max_parents = max_parents
        self.min_samples = min_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not CAUSAL_LEARN_AVAILABLE:
            logger.error("causal-learn library required. Install with: pip install causal-learn==0.1.3")
    
    def extract_features_from_user_history(self, user_profiles: List[Dict]) -> pd.DataFrame:
        """
        Extract causal variables from LaMP-2 user profiles
        
        Args:
            user_profiles: List of user profile dictionaries from LaMP-2
            
        Returns:
            DataFrame with causal variables: genre_preferences, rating_patterns, temporal_features
        """
        features = []
        
        for profile in user_profiles:
            feature_dict = {}
            
            # Extract genre preferences (binary indicators)
            genres = ['action', 'comedy', 'drama', 'sci-fi', 'romance', 'horror']
            for genre in genres:
                # Count mentions of genre in user history
                mentions = sum(1 for item in profile.get('profile', []) 
                              if genre.lower() in item.get('description', '').lower())
                feature_dict[f'likes_{genre}'] = mentions
            
            # Rating patterns
            ratings = [item.get('rating', 3.0) for item in profile.get('profile', []) if 'rating' in item]
            feature_dict['avg_rating'] = np.mean(ratings) if ratings else 3.0
            feature_dict['rating_variance'] = np.var(ratings) if len(ratings) > 1 else 0.0
            
            # Temporal features
            timestamps = [item.get('timestamp', 0) for item in profile.get('profile', [])]
            if timestamps:
                feature_dict['activity_span'] = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                feature_dict['activity_frequency'] = len(timestamps) / (feature_dict['activity_span'] + 1)
            else:
                feature_dict['activity_span'] = 0
                feature_dict['activity_frequency'] = 0
            
            # Interaction complexity
            feature_dict['history_length'] = len(profile.get('profile', []))
            feature_dict['description_avg_length'] = np.mean([
                len(item.get('description', '')) for item in profile.get('profile', [])
            ]) if profile.get('profile') else 0
            
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        # Fill NaN values with median
        return df.fillna(df.median())
    
    def build_causal_graph(self, user_profiles: List[Dict], 
                          user_id: Optional[str] = None) -> Optional[CausalGraphResult]:
        """
        Build causal graph using PC algorithm
        
        Args:
            user_profiles: User profiles from LaMP-2 data
            user_id: Optional user ID for caching
            
        Returns:
            CausalGraphResult or None if insufficient data
        """
        if not CAUSAL_LEARN_AVAILABLE:
            logger.error("Cannot build causal graph: causal-learn not installed")
            return None
        
        # Check cache first
        if user_id and self.cache_dir:
            cache_file = self.cache_dir / f"causal_graph_{user_id}.npz"
            if cache_file.exists():
                try:
                    cached = np.load(cache_file, allow_pickle=True)
                    return CausalGraphResult(
                        adjacency_matrix=cached['adjacency_matrix'],
                        node_names=cached['node_names'].tolist(),
                        edge_weights=cached['edge_weights'],
                        alpha_threshold=float(cached['alpha_threshold']),
                        n_samples=int(cached['n_samples']),
                        discovery_time=float(cached['discovery_time'])
                    )
                except Exception as e:
                    logger.warning(f"Failed to load cached graph: {e}")
        
        # Extract features
        data_df = self.extract_features_from_user_history(user_profiles)
        
        if len(data_df) < self.min_samples:
            logger.warning(f"Insufficient samples for causal discovery: {len(data_df)} < {self.min_samples}")
            return None
        
        # Convert to numpy array for causal-learn
        data_matrix = data_df.values.astype(np.float64)
        node_names = list(data_df.columns)
        
        logger.info(f"Building causal graph from {len(data_df)} samples with {len(node_names)} variables")
        
        import time
        start_time = time.time()
        
        try:
            # Run PC algorithm
            cg = pc(
                data_matrix,
                alpha=self.alpha,
                indep_test='fisherz',  # Fisher's Z test for continuous data
                stable=True,
                uc_rule=0,
                uc_priority=2,
                mvpc=False,
                correction_name='BH',  # Benjamini-Hochberg correction
                verbose=False
            )
            
            discovery_time = time.time() - start_time
            
            # Extract results
            adjacency_matrix = cg.G.graph
            
            # Compute edge weights (correlation-based)
            edge_weights = np.zeros_like(adjacency_matrix, dtype=np.float64)
            for i in range(len(node_names)):
                for j in range(len(node_names)):
                    if adjacency_matrix[i, j] != 0:
                        corr = np.corrcoef(data_matrix[:, i], data_matrix[:, j])[0, 1]
                        edge_weights[i, j] = abs(corr)
            
            result = CausalGraphResult(
                adjacency_matrix=adjacency_matrix,
                node_names=node_names,
                edge_weights=edge_weights,
                alpha_threshold=self.alpha,
                n_samples=len(data_df),
                discovery_time=discovery_time
            )
            
            # Cache result
            if user_id and self.cache_dir:
                cache_file = self.cache_dir / f"causal_graph_{user_id}.npz"
                np.savez(
                    cache_file,
                    adjacency_matrix=result.adjacency_matrix,
                    node_names=np.array(result.node_names),
                    edge_weights=result.edge_weights,
                    alpha_threshold=result.alpha_threshold,
                    n_samples=result.n_samples,
                    discovery_time=result.discovery_time
                )
                logger.info(f"Cached causal graph to {cache_file}")
            
            logger.info(f"Causal graph built in {discovery_time:.2f}s with {np.sum(adjacency_matrix != 0)} edges")
            return result
            
        except Exception as e:
            logger.error(f"PC algorithm failed: {e}")
            return None
    
    def get_causal_mask(self, graph_result: CausalGraphResult, 
                       target_features: List[str]) -> np.ndarray:
        """
        Create causal mask for editing constraints
        
        Args:
            graph_result: Result from build_causal_graph
            target_features: Features to constrain editing for
            
        Returns:
            Boolean mask indicating causally valid editing directions
        """
        if not graph_result:
            return np.ones(len(target_features), dtype=bool)
        
        mask = np.ones(len(target_features), dtype=bool)
        
        for i, feature in enumerate(target_features):
            if feature in graph_result.node_names:
                feature_idx = graph_result.node_names.index(feature)
                # Allow editing if feature has causal parents (can be influenced)
                has_parents = np.any(graph_result.adjacency_matrix[:, feature_idx] != 0)
                mask[i] = has_parents
        
        return mask
    
    def visualize_causal_graph(self, graph_result: CausalGraphResult, 
                              output_path: Optional[str] = None) -> str:
        """
        Create visualization of causal graph
        
        Args:
            graph_result: Result from build_causal_graph
            output_path: Optional path to save visualization
            
        Returns:
            Graph description or path to saved image
        """
        if not graph_result:
            return "No graph to visualize"
        
        # Create textual representation
        edges = []
        for i, source in enumerate(graph_result.node_names):
            for j, target in enumerate(graph_result.node_names):
                if graph_result.adjacency_matrix[i, j] != 0:
                    weight = graph_result.edge_weights[i, j]
                    direction = "→" if graph_result.adjacency_matrix[i, j] == 1 else "↔"
                    edges.append(f"{source} {direction} {target} (w={weight:.3f})")
        
        description = f"""Causal Graph Summary:
Nodes: {len(graph_result.node_names)}
Edges: {len(edges)}
Alpha: {graph_result.alpha_threshold}
Samples: {graph_result.n_samples}
Discovery Time: {graph_result.discovery_time:.2f}s

Edges:
""" + "\n".join(edges)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(description)
            logger.info(f"Graph description saved to {output_path}")
        
        return description

# Backwards compatibility with existing Chameleon system
def integrate_with_chameleon_data_loader(data_loader, user_limit: int = 100) -> List[Dict]:
    """
    Extract user profiles compatible with CausalGraphBuilder from existing LaMP data loader
    
    Args:
        data_loader: Existing LaMPDataLoader instance
        user_limit: Maximum number of users to process
        
    Returns:
        List of user profiles ready for causal analysis
    """
    try:
        # Use existing data loading infrastructure
        test_samples = data_loader.get_user_samples(user_limit)
        
        # Group by user and create profiles
        user_profiles = []
        current_user = None
        current_profile = []
        
        for sample in test_samples:
            user_id = str(sample.get('id', ''))[:3]  # Extract user ID using existing logic
            
            if user_id != current_user:
                if current_profile:
                    user_profiles.append({
                        'user_id': current_user,
                        'profile': current_profile
                    })
                current_user = user_id
                current_profile = []
            
            # Extract profile information from sample
            profile_item = {
                'description': sample.get('input', ''),
                'output': sample.get('output', ''),
                'timestamp': hash(str(sample.get('id', 0))) % 1000000,  # Pseudo-timestamp
                'rating': hash(sample.get('input', '')) % 5 + 1  # Pseudo-rating
            }
            current_profile.append(profile_item)
        
        # Add final user
        if current_profile:
            user_profiles.append({
                'user_id': current_user,
                'profile': current_profile
            })
        
        logger.info(f"Extracted {len(user_profiles)} user profiles for causal analysis")
        return user_profiles
        
    except Exception as e:
        logger.error(f"Failed to integrate with Chameleon data loader: {e}")
        return []