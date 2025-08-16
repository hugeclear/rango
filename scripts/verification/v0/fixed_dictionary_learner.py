#!/usr/bin/env python3
"""
Fixed Dictionary Learner for V0 Testing
OMP次元問題を修正した辞書学習器
"""

import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DictionaryConfig:
    """辞書学習設定"""
    n_atoms: int = 64
    sparsity_alpha: float = 0.1
    max_iter: int = 1000
    tolerance: float = 1e-4
    batch_size: int = 32
    privacy_noise: float = 0.01
    quality_threshold: float = 0.7
    diversity_penalty: float = 0.1

@dataclass
class AtomMetrics:
    """原子品質・リスク指標"""
    contribution_score: float
    stability_score: float
    usage_frequency: int
    leakage_risk: float
    diversity_index: float
    lifetime_ttl: int

class FixedCFSDictionaryLearner:
    """次元問題修正版辞書学習器"""
    
    def __init__(self, config: DictionaryConfig):
        self.config = config
        self.dictionary = None
        self.atom_metrics = {}
        self.usage_history = {}
        self.update_count = 0
        
        logger.info(f"FixedCFSDictionaryLearner initialized: K={config.n_atoms}, alpha={config.sparsity_alpha}")
    
    def learn_initial_dictionary(self, user_directions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """初期辞書学習"""
        logger.info("Learning initial dictionary from user directions...")
        
        direction_matrix = np.array(list(user_directions.values()))
        n_users, embedding_dim = direction_matrix.shape
        
        logger.info(f"Direction matrix shape: {direction_matrix.shape}")
        
        # K-SVD辞書学習
        dict_learner = DictionaryLearning(
            n_components=self.config.n_atoms,
            alpha=self.config.sparsity_alpha,
            max_iter=self.config.max_iter,
            tol=self.config.tolerance,
            fit_algorithm='lars',
            transform_algorithm='omp',
            random_state=42
        )
        
        sparse_codes = dict_learner.fit_transform(direction_matrix)
        self.dictionary = dict_learner.components_.T  # [embedding_dim, n_atoms]
        
        # 初期品質メトリクス計算
        self._compute_initial_metrics(direction_matrix, sparse_codes)
        
        reconstruction_error = np.mean(np.linalg.norm(
            direction_matrix - sparse_codes @ self.dictionary.T, axis=1
        ))
        
        sparsity_level = np.mean(np.sum(sparse_codes != 0, axis=1))
        
        results = {
            'dictionary': self.dictionary,
            'sparse_codes': sparse_codes,
            'reconstruction_error': reconstruction_error,
            'sparsity_level': sparsity_level,
            'atom_metrics': self.atom_metrics,
            'user_ids': list(user_directions.keys())
        }
        
        logger.info(f"Dictionary learning completed:")
        logger.info(f"  Reconstruction error: {reconstruction_error:.4f}")
        logger.info(f"  Avg sparsity level: {sparsity_level:.2f}")
        
        return results
    
    def _compute_initial_metrics(self, directions: np.ndarray, codes: np.ndarray):
        """初期原子品質メトリクス計算"""
        for k in range(self.config.n_atoms):
            atom_vector = self.dictionary[:, k]
            atom_codes = codes[:, k]
            
            usage_mask = atom_codes != 0
            contribution = np.mean(np.abs(atom_codes[usage_mask])) if np.any(usage_mask) else 0.0
            
            other_atoms = np.delete(self.dictionary, k, axis=1)
            if other_atoms.shape[1] > 0:
                similarities = np.abs(atom_vector @ other_atoms)
                diversity = 1.0 - np.max(similarities) / (np.linalg.norm(atom_vector) * np.linalg.norm(other_atoms, axis=0).max())
            else:
                diversity = 1.0
            
            specialization = np.std(atom_codes) / (np.mean(np.abs(atom_codes)) + 1e-8)
            leakage_risk = min(1.0, specialization / 2.0)
            
            self.atom_metrics[k] = AtomMetrics(
                contribution_score=contribution,
                stability_score=1.0,
                usage_frequency=int(np.sum(usage_mask)),
                leakage_risk=leakage_risk,
                diversity_index=diversity,
                lifetime_ttl=1000
            )
    
    def encode_user_direction(self, user_direction: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """修正版ユーザー方向疎符号化"""
        if self.dictionary is None:
            raise ValueError("Dictionary not learned yet")
        
        # 線形最小二乗法による疎符号化（OMPの代替）
        try:
            # 正規方程式で解く: (D^T D) x = D^T y
            DTD = self.dictionary.T @ self.dictionary
            DTy = self.dictionary.T @ user_direction
            
            # L2正則化付き解
            reg_strength = self.config.sparsity_alpha
            sparse_code = np.linalg.solve(
                DTD + reg_strength * np.eye(DTD.shape[0]), 
                DTy
            )
            
            # 閾値処理による疎性化
            threshold = np.std(sparse_code) * 0.1
            sparse_code[np.abs(sparse_code) < threshold] = 0
            
        except Exception as e:
            logger.warning(f"Sparse coding failed: {e}, using fallback")
            # フォールバック: 辞書との内積
            sparse_code = self.dictionary.T @ user_direction
            threshold = np.std(sparse_code) * 0.2
            sparse_code[np.abs(sparse_code) < threshold] = 0
        
        # 再構成誤差
        reconstruction = self.dictionary @ sparse_code
        reconstruction_error = np.linalg.norm(user_direction - reconstruction)
        
        # 疎性統計
        active_atoms = np.sum(sparse_code != 0)
        max_coefficient = np.max(np.abs(sparse_code))
        
        encoding_stats = {
            'reconstruction_error': reconstruction_error,
            'active_atoms': int(active_atoms),
            'sparsity_ratio': active_atoms / len(sparse_code),
            'max_coefficient': max_coefficient,
            'l1_norm': np.sum(np.abs(sparse_code))
        }
        
        return sparse_code, encoding_stats
    
    def select_collaborative_atoms(self, user_sparse_code: np.ndarray, 
                                 graph_context: Dict[str, Any],
                                 top_k: int = 5) -> Dict[str, Any]:
        """グラフ条件付き原子選択"""
        
        # 基本スコア: 疎符号での重み
        base_scores = np.abs(user_sparse_code)
        
        # グラフ近接性スコア
        graph_scores = np.zeros_like(base_scores)
        if 'similar_users' in graph_context:
            for similar_user_id, similarity in graph_context['similar_users']:
                # 類似ユーザーが使用する原子に重み加算
                if similar_user_id in self.usage_history:
                    for atom_id in self.usage_history[similar_user_id]:
                        if atom_id < len(graph_scores):
                            graph_scores[atom_id] += similarity * 0.3
        
        # 品質スコア
        quality_scores = np.array([
            self.atom_metrics[k].contribution_score * self.atom_metrics[k].stability_score
            for k in range(len(base_scores))
        ])
        
        # リスクペナルティ
        risk_penalties = np.array([
            self.atom_metrics[k].leakage_risk
            for k in range(len(base_scores))
        ])
        
        # 多様性ボーナス
        diversity_bonus = np.array([
            self.atom_metrics[k].diversity_index
            for k in range(len(base_scores))
        ])
        
        # 統合スコア
        alpha, beta, gamma, delta = 1.0, 0.5, 0.3, 0.2
        final_scores = (alpha * base_scores + 
                       beta * graph_scores + 
                       gamma * quality_scores + 
                       gamma * diversity_bonus - 
                       delta * risk_penalties)
        
        # Top-K選択
        top_atom_indices = np.argsort(final_scores)[-top_k:][::-1]
        selected_atoms = {
            'atom_indices': top_atom_indices.tolist(),
            'atom_weights': user_sparse_code[top_atom_indices].tolist(),
            'selection_scores': final_scores[top_atom_indices].tolist(),
            'diversity_coverage': len(set(top_atom_indices))
        }
        
        return selected_atoms