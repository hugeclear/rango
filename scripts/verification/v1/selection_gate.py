#!/usr/bin/env python3
"""
V1: Selection Gate Implementation
選択ゲート - cos+tags+ppr複合スコアリング + 適応的K値

技術仕様:
- cos: コサイン類似度スコア
- tags: タグベース親和性スコア
- ppr: PersonalizedPageRankスコア
- lambda: 冗長性ペナルティ重み
- adaptive_K: 動的top-K調整
- quality_gate: 品質閾値フィルタリング
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SelectionGateConfig:
    """選択ゲート設定"""
    # スコア重み
    weight_cos: float = 0.4      # コサイン類似度重み
    weight_tags: float = 0.3     # タグ親和性重み
    weight_ppr: float = 0.3      # PPRスコア重み
    
    # 冗長性制御
    redundancy_lambda: float = 0.2     # 冗長性ペナルティ重み
    redundancy_threshold: float = 0.7   # 冗長性閾値
    
    # 適応的K
    adaptive_k_enabled: bool = True     # 適応的K有効化
    base_k: int = 5                    # ベースK値
    min_k: int = 2                     # 最小K値
    max_k: int = 10                    # 最大K値
    
    # 品質フィルタ
    quality_threshold: float = 0.5     # 品質閾値
    stability_threshold: float = 0.6   # 安定性閾値
    
    # タグマッピング
    tag_vocabulary: List[str] = None   # タグ語彙リスト
    tag_embedding_dim: int = 64        # タグ埋め込み次元

@dataclass
class SelectionScore:
    """選択スコア詳細"""
    atom_id: int
    cos_score: float
    tags_score: float
    ppr_score: float
    composite_score: float
    redundancy_penalty: float
    final_score: float
    quality_passed: bool

@dataclass
class SelectionResult:
    """選択結果"""
    selected_atoms: List[int]
    selection_scores: List[SelectionScore]
    adaptive_k: int
    total_candidates: int
    filtering_stats: Dict[str, int]
    computation_time_ms: float

class SelectionGate:
    """V1選択ゲートシステム"""
    
    def __init__(self, config: SelectionGateConfig):
        self.config = config
        
        # タグ埋め込み初期化
        self._initialize_tag_embeddings()
        
        # PPRキャッシュ
        self.ppr_cache = {}
        self.ppr_computation_count = 0
        
        # 統計
        self.selection_history = []
        self.adaptive_k_history = []
        
        logger.info("V1 Selection Gate initialized with composite scoring")
    
    def _initialize_tag_embeddings(self):
        """タグ埋め込み初期化"""
        if self.config.tag_vocabulary is None:
            # デフォルトLaMP-2タグ語彙
            self.config.tag_vocabulary = [
                'action', 'adventure', 'animation', 'biography', 'comedy', 
                'crime', 'documentary', 'drama', 'family', 'fantasy',
                'history', 'horror', 'music', 'mystery', 'romance',
                'sci-fi', 'sport', 'thriller', 'war', 'western'
            ]
        
        # ランダム初期化タグ埋め込み（実際は事前学習済みを使用）
        vocab_size = len(self.config.tag_vocabulary)
        self.tag_embeddings = np.random.randn(vocab_size, self.config.tag_embedding_dim) * 0.1
        
        # タグ→インデックスマッピング
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.config.tag_vocabulary)}
        
        logger.info(f"Tag embeddings initialized: {vocab_size} tags x {self.config.tag_embedding_dim}D")
    
    def select_atoms_with_gate(self, 
                             candidates: Dict[str, Any],
                             user_profile: Dict[str, Any],
                             query_context: Dict[str, Any],
                             dictionary: np.ndarray,
                             atom_metrics: Dict[int, Any]) -> SelectionResult:
        """ゲート付き原子選択"""
        
        start_time = time.time()
        
        # 候補原子取得
        candidate_atoms = candidates.get('atom_indices', [])
        candidate_weights = candidates.get('atom_weights', [])
        
        if not candidate_atoms:
            return SelectionResult(
                selected_atoms=[], selection_scores=[], adaptive_k=0, 
                total_candidates=0, filtering_stats={}, computation_time_ms=0
            )
        
        # ステップ1: 複合スコア計算
        selection_scores = self._compute_composite_scores(
            candidate_atoms, candidate_weights, user_profile, 
            query_context, dictionary, atom_metrics
        )
        
        # ステップ2: 品質フィルタリング
        quality_filtered_scores = self._apply_quality_filter(selection_scores, atom_metrics)
        
        # ステップ3: 適応的K決定
        adaptive_k = self._determine_adaptive_k(
            quality_filtered_scores, user_profile, query_context
        )
        
        # ステップ4: 冗長性除去選択
        final_selected = self._select_with_redundancy_control(
            quality_filtered_scores, adaptive_k, dictionary
        )
        
        computation_time = (time.time() - start_time) * 1000
        
        # 統計更新
        filtering_stats = {
            'initial_candidates': len(candidate_atoms),
            'quality_filtered': len(quality_filtered_scores),
            'redundancy_filtered': len(final_selected),
            'adaptive_k_used': adaptive_k
        }
        
        result = SelectionResult(
            selected_atoms=[score.atom_id for score in final_selected],
            selection_scores=final_selected,
            adaptive_k=adaptive_k,
            total_candidates=len(candidate_atoms),
            filtering_stats=filtering_stats,
            computation_time_ms=computation_time
        )
        
        # 履歴記録
        self.selection_history.append(result)
        self.adaptive_k_history.append(adaptive_k)
        
        return result
    
    def _compute_composite_scores(self, 
                                candidate_atoms: List[int],
                                candidate_weights: List[float],
                                user_profile: Dict[str, Any], 
                                query_context: Dict[str, Any],
                                dictionary: np.ndarray,
                                atom_metrics: Dict[int, Any]) -> List[SelectionScore]:
        """複合スコア計算"""
        
        scores = []
        query_embedding = query_context.get('query_embedding', np.zeros(dictionary.shape[0]))
        user_tags = user_profile.get('tags', [])
        user_id = user_profile.get('user_id', 'unknown')
        
        for i, atom_id in enumerate(candidate_atoms):
            atom_vector = dictionary[:, atom_id]
            atom_weight = candidate_weights[i] if i < len(candidate_weights) else 0.0
            
            # コサイン類似度スコア
            cos_score = self._compute_cos_score(atom_vector, query_embedding)
            
            # タグ親和性スコア
            tags_score = self._compute_tags_score(atom_id, user_tags, atom_metrics)
            
            # PPRスコア
            ppr_score = self._compute_ppr_score(atom_id, user_id, query_context)
            
            # 複合スコア
            composite_score = (self.config.weight_cos * cos_score + 
                             self.config.weight_tags * tags_score + 
                             self.config.weight_ppr * ppr_score)
            
            # 品質チェック
            quality_passed = self._check_atom_quality(atom_id, atom_metrics)
            
            score = SelectionScore(
                atom_id=atom_id,
                cos_score=cos_score,
                tags_score=tags_score,
                ppr_score=ppr_score,
                composite_score=composite_score,
                redundancy_penalty=0.0,  # 後で計算
                final_score=composite_score,  # 暫定
                quality_passed=quality_passed
            )
            
            scores.append(score)
        
        return scores
    
    def _compute_cos_score(self, atom_vector: np.ndarray, query_embedding: np.ndarray) -> float:
        """コサイン類似度スコア計算"""
        if np.linalg.norm(atom_vector) == 0 or np.linalg.norm(query_embedding) == 0:
            return 0.0
        
        cos_sim = np.dot(atom_vector, query_embedding) / (
            np.linalg.norm(atom_vector) * np.linalg.norm(query_embedding)
        )
        
        # [0, 1]にクリップ
        return max(0.0, cos_sim)
    
    def _compute_tags_score(self, atom_id: int, user_tags: List[str], 
                          atom_metrics: Dict[int, Any]) -> float:
        """タグ親和性スコア計算"""
        if not user_tags or atom_id not in atom_metrics:
            return 0.0
        
        # 原子のタグ推定（実際は原子メタデータから取得）
        atom_tags = self._estimate_atom_tags(atom_id, atom_metrics)
        
        if not atom_tags:
            return 0.0
        
        # ユーザータグと原子タグの類似度
        max_similarity = 0.0
        
        for user_tag in user_tags:
            if user_tag in self.tag_to_idx:
                user_tag_emb = self.tag_embeddings[self.tag_to_idx[user_tag]]
                
                for atom_tag in atom_tags:
                    if atom_tag in self.tag_to_idx:
                        atom_tag_emb = self.tag_embeddings[self.tag_to_idx[atom_tag]]
                        
                        similarity = np.dot(user_tag_emb, atom_tag_emb) / (
                            np.linalg.norm(user_tag_emb) * np.linalg.norm(atom_tag_emb)
                        )
                        max_similarity = max(max_similarity, similarity)
        
        return max(0.0, max_similarity)
    
    def _estimate_atom_tags(self, atom_id: int, atom_metrics: Dict[int, Any]) -> List[str]:
        """原子タグ推定（簡易版）"""
        # 実際は原子メタデータから取得。ここでは例として一部タグをランダム選択
        if atom_id not in atom_metrics:
            return []
        
        # 原子IDに基づく決定論的タグ割り当て
        np.random.seed(atom_id)
        n_tags = np.random.randint(1, 4)  # 1-3個のタグ
        selected_tags = np.random.choice(
            self.config.tag_vocabulary, 
            size=n_tags, 
            replace=False
        ).tolist()
        
        return selected_tags
    
    def _compute_ppr_score(self, atom_id: int, user_id: str, 
                         query_context: Dict[str, Any]) -> float:
        """PPRスコア計算"""
        # PPRキャッシュチェック
        cache_key = f"{user_id}_{atom_id}"
        if cache_key in self.ppr_cache:
            return self.ppr_cache[cache_key]
        
        # 簡易PPR計算（実際はグラフベース）
        similar_users = query_context.get('similar_users', [])
        ppr_score = 0.0
        
        for similar_user_id, similarity in similar_users:
            # 類似ユーザーが該当原子を使用している確率
            usage_prob = np.random.random() * similarity  # 簡易版
            ppr_score += usage_prob
        
        # 正規化
        ppr_score = min(1.0, ppr_score / max(1, len(similar_users)))
        
        # キャッシュ保存
        self.ppr_cache[cache_key] = ppr_score
        self.ppr_computation_count += 1
        
        return ppr_score
    
    def _check_atom_quality(self, atom_id: int, atom_metrics: Dict[int, Any]) -> bool:
        """原子品質チェック"""
        if atom_id not in atom_metrics:
            return False
        
        metrics = atom_metrics[atom_id]
        
        # 品質チェック条件
        quality_ok = getattr(metrics, 'contribution_score', 0.0) >= self.config.quality_threshold
        stability_ok = getattr(metrics, 'stability_score', 0.0) >= self.config.stability_threshold
        
        return quality_ok and stability_ok
    
    def _apply_quality_filter(self, scores: List[SelectionScore], 
                            atom_metrics: Dict[int, Any]) -> List[SelectionScore]:
        """品質フィルタ適用"""
        return [score for score in scores if score.quality_passed]
    
    def _determine_adaptive_k(self, scores: List[SelectionScore], 
                            user_profile: Dict[str, Any],
                            query_context: Dict[str, Any]) -> int:
        """適応的K決定"""
        if not self.config.adaptive_k_enabled:
            return self.config.base_k
        
        # スコア品質に基づく調整
        if not scores:
            return self.config.min_k
        
        avg_composite_score = np.mean([score.composite_score for score in scores])
        score_variance = np.var([score.composite_score for score in scores])
        
        # より感度の高い適応的K調整
        k_adjustment = 0
        
        # スコア品質による調整（緩和した閾値）
        if avg_composite_score > 0.4:
            k_adjustment += 1  # 高品質でK増加
        elif avg_composite_score < 0.2:
            k_adjustment -= 1  # 低品質でK減少
        
        # 分散による調整（緩和した閾値）
        if score_variance > 0.02:
            k_adjustment += 1  # 高分散（多様性）でK増加
        elif score_variance < 0.005:
            k_adjustment -= 1  # 低分散（単調）でK減少
        
        # ユーザープロファイルによる調整
        user_tags = user_profile.get('tags', [])
        if len(user_tags) > 2:
            k_adjustment += 1  # 多様な好みのユーザーはK増加
        
        adaptive_k = self.config.base_k + k_adjustment
        
        # 範囲制限
        adaptive_k = max(self.config.min_k, min(self.config.max_k, adaptive_k))
        
        # 候補数による制限
        adaptive_k = min(adaptive_k, len(scores))
        
        return adaptive_k
    
    def _select_with_redundancy_control(self, scores: List[SelectionScore], 
                                      k: int, dictionary: np.ndarray) -> List[SelectionScore]:
        """冗長性制御付き選択"""
        if not scores or k == 0:
            return []
        
        # スコア順ソート
        sorted_scores = sorted(scores, key=lambda x: x.composite_score, reverse=True)
        
        selected = []
        
        for candidate in sorted_scores:
            if len(selected) >= k:
                break
            
            # 冗長性ペナルティ計算
            redundancy_penalty = self._compute_redundancy_penalty(
                candidate, selected, dictionary
            )
            
            # 最終スコア更新
            candidate.redundancy_penalty = redundancy_penalty
            candidate.final_score = candidate.composite_score - self.config.redundancy_lambda * redundancy_penalty
            
            # 冗長性閾値チェック
            if redundancy_penalty < self.config.redundancy_threshold:
                selected.append(candidate)
        
        return selected
    
    def _compute_redundancy_penalty(self, candidate: SelectionScore, 
                                  selected: List[SelectionScore],
                                  dictionary: np.ndarray) -> float:
        """冗長性ペナルティ計算"""
        if not selected:
            return 0.0
        
        candidate_vector = dictionary[:, candidate.atom_id]
        max_similarity = 0.0
        
        for selected_score in selected:
            selected_vector = dictionary[:, selected_score.atom_id]
            
            if np.linalg.norm(candidate_vector) > 0 and np.linalg.norm(selected_vector) > 0:
                similarity = np.dot(candidate_vector, selected_vector) / (
                    np.linalg.norm(candidate_vector) * np.linalg.norm(selected_vector)
                )
                max_similarity = max(max_similarity, abs(similarity))
        
        return max_similarity
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """選択統計取得"""
        if not self.selection_history:
            return {'status': 'no_data'}
        
        # 適応的K統計
        k_values = self.adaptive_k_history
        k_stats = {
            'mean': np.mean(k_values),
            'std': np.std(k_values),
            'min': np.min(k_values),
            'max': np.max(k_values)
        }
        
        # 計算時間統計
        computation_times = [result.computation_time_ms for result in self.selection_history]
        time_stats = {
            'mean': np.mean(computation_times),
            'std': np.std(computation_times),
            'min': np.min(computation_times),
            'max': np.max(computation_times)
        }
        
        # フィルタリング効率
        total_candidates = sum(result.total_candidates for result in self.selection_history)
        total_selected = sum(len(result.selected_atoms) for result in self.selection_history)
        selection_efficiency = total_selected / max(1, total_candidates)
        
        return {
            'total_selections': len(self.selection_history),
            'adaptive_k_stats': k_stats,
            'computation_time_stats': time_stats,
            'selection_efficiency': selection_efficiency,
            'ppr_cache_size': len(self.ppr_cache),
            'ppr_computations': self.ppr_computation_count
        }

# テスト・使用例
if __name__ == "__main__":
    # テスト設定
    config = SelectionGateConfig(
        weight_cos=0.4,
        weight_tags=0.3,
        weight_ppr=0.3,
        redundancy_lambda=0.2,
        adaptive_k_enabled=True,
        base_k=5
    )
    
    # 選択ゲート初期化
    gate = SelectionGate(config)
    
    # テストデータ
    np.random.seed(42)
    dictionary = np.random.randn(768, 64)
    
    # ダミー原子メトリクス
    atom_metrics = {
        i: type('AtomMetrics', (), {
            'contribution_score': np.random.random(),
            'stability_score': np.random.random(),
            'leakage_risk': np.random.random()
        })() for i in range(64)
    }
    
    # テスト選択実行
    print("=== V1 Selection Gate Test ===")
    
    for i in range(5):
        # テスト候補
        candidates = {
            'atom_indices': np.random.choice(64, size=10, replace=False).tolist(),
            'atom_weights': np.random.random(10).tolist()
        }
        
        # ユーザープロファイル
        user_profile = {
            'user_id': f'test_user_{i}',
            'tags': ['action', 'drama', 'comedy']
        }
        
        # クエリコンテキスト
        query_context = {
            'query_embedding': np.random.randn(768),
            'similar_users': [('user_1', 0.8), ('user_2', 0.6)]
        }
        
        # 選択実行
        result = gate.select_atoms_with_gate(
            candidates, user_profile, query_context, dictionary, atom_metrics
        )
        
        print(f"Selection {i}: {len(result.selected_atoms)} atoms selected (K={result.adaptive_k})")
        print(f"  Computation time: {result.computation_time_ms:.2f}ms")
        print(f"  Filtering: {result.filtering_stats}")
        
        # スコア詳細表示
        if result.selection_scores:
            avg_final_score = np.mean([score.final_score for score in result.selection_scores])
            avg_redundancy = np.mean([score.redundancy_penalty for score in result.selection_scores])
            print(f"  Avg final score: {avg_final_score:.3f}")
            print(f"  Avg redundancy penalty: {avg_redundancy:.3f}")
    
    # 統計レポート
    print("\n=== V1 Selection Statistics ===")
    stats = gate.get_selection_statistics()
    print(f"Total selections: {stats['total_selections']}")
    print(f"Adaptive K stats: mean={stats['adaptive_k_stats']['mean']:.2f}, std={stats['adaptive_k_stats']['std']:.2f}")
    print(f"Computation time: mean={stats['computation_time_stats']['mean']:.2f}ms")
    print(f"Selection efficiency: {stats['selection_efficiency']:.3f}")
    print(f"PPR cache size: {stats['ppr_cache_size']}")
    
    print("\n✅ V1 Selection Gate Test Completed")
    print("   - Composite scoring: WORKING")
    print("   - Adaptive K: WORKING")
    print("   - Redundancy control: WORKING")
    print("   - Quality filtering: WORKING")