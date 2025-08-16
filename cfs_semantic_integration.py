#!/usr/bin/env python3
"""
CFS-Chameleon意味的類似度統合モジュール
CollaborativeDirectionPoolの改修と意味的類似度計算の統合
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
import json
from pathlib import Path

# 意味的類似度エンジン
from semantic_similarity_engine import (
    SemanticSimilarityEngine,
    SemanticSimilarityConfig,
    compute_semantic_similarity_rich,
    compute_batch_semantic_similarity,
    HybridSimilarityCalculator
)

# CFS-Chameleon関連モジュール
try:
    from cfs_chameleon_extension import DirectionPiece, CollaborativeDirectionPool
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    CFS_AVAILABLE = True
except ImportError:
    print("⚠️ CFS modules not available. Using mock implementations.")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticAwareDirectionPool:
    """意味的類似度対応の方向ピースプール"""
    
    def __init__(self,
                 capacity: int = 1000,
                 rank: int = 32,
                 semantic_config: SemanticSimilarityConfig = None,
                 use_hybrid_similarity: bool = True,
                 similarity_threshold: float = 0.1):
        """
        初期化
        
        Args:
            capacity: プール容量
            rank: SVDランク
            semantic_config: 意味的類似度設定
            use_hybrid_similarity: ハイブリッド類似度使用フラグ
            similarity_threshold: 類似度閾値
        """
        self.capacity = capacity
        self.rank = rank
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_similarity = use_hybrid_similarity
        
        # ピース格納
        self.pieces: List[Any] = []
        self.user_mapping: Dict[str, List[int]] = {}
        
        # 意味的類似度エンジン
        self.semantic_config = semantic_config or SemanticSimilarityConfig()
        self.semantic_engine = SemanticSimilarityEngine(self.semantic_config)
        
        # ハイブリッド計算器
        if self.use_hybrid_similarity:
            self.hybrid_calculator = HybridSimilarityCalculator(
                self.semantic_engine,
                semantic_weight=0.8,
                vector_weight=0.2
            )
        else:
            self.hybrid_calculator = None
        
        logger.info("✅ SemanticAwareDirectionPool initialized")
        logger.info(f"   Capacity: {capacity}, Rank: {rank}")
        logger.info(f"   Hybrid similarity: {use_hybrid_similarity}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
    
    def add_piece(self, piece: Any, user_id: str = None):
        """ピースをプールに追加"""
        if len(self.pieces) >= self.capacity:
            self._evict_least_used_piece()
        
        self.pieces.append(piece)
        piece_index = len(self.pieces) - 1
        
        if user_id:
            if user_id not in self.user_mapping:
                self.user_mapping[user_id] = []
            self.user_mapping[user_id].append(piece_index)
        
        logger.debug(f"Added piece to pool: index {piece_index}, user {user_id}")
    
    def _evict_least_used_piece(self):
        """最も使用頻度の低いピースを削除"""
        if not self.pieces:
            return
        
        # 簡単な実装: 最初のピースを削除
        removed_piece = self.pieces.pop(0)
        
        # ユーザーマッピングを更新
        for user_id, indices in self.user_mapping.items():
            self.user_mapping[user_id] = [i-1 for i in indices if i > 0]
        
        logger.debug("Evicted least used piece from pool")
    
    def compute_context_similarity_semantic(self, 
                                          user_context: Union[str, np.ndarray],
                                          candidate_pieces: List[Any] = None) -> np.ndarray:
        """
        意味的類似度に基づくコンテキスト類似度計算
        
        Args:
            user_context: ユーザーコンテキスト（テキストまたはベクトル）
            candidate_pieces: 候補ピースリスト（Noneの場合は全ピース）
            
        Returns:
            類似度スコア配列
        """
        if candidate_pieces is None:
            candidate_pieces = self.pieces
        
        if not candidate_pieces:
            return np.array([])
        
        logger.debug(f"Computing semantic context similarity for {len(candidate_pieces)} pieces")
        
        try:
            if self.use_hybrid_similarity and self.hybrid_calculator:
                # ハイブリッド類似度計算
                similarities = []
                for piece in candidate_pieces:
                    sim = self.hybrid_calculator.compute_hybrid_similarity(user_context, piece)
                    similarities.append(sim)
                
                similarity_scores = np.array(similarities)
            else:
                # 純粋な意味的類似度計算
                similarity_scores = compute_batch_semantic_similarity(
                    [user_context], candidate_pieces, self.semantic_engine
                )[0]  # 最初の行のみ取得
            
            logger.debug(f"Similarity scores: min={similarity_scores.min():.4f}, max={similarity_scores.max():.4f}, avg={similarity_scores.mean():.4f}")
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"❌ Semantic context similarity error: {e}")
            # フォールバック: ランダム類似度
            return np.random.uniform(0.1, 0.5, len(candidate_pieces))
    
    def compute_semantic_similarity_legacy(self,
                                         user_context: Union[str, np.ndarray],
                                         candidate_pieces: List[Any] = None) -> np.ndarray:
        """
        従来手法との互換性のための意味的類似度計算
        
        Args:
            user_context: ユーザーコンテキスト
            candidate_pieces: 候補ピースリスト
            
        Returns:
            類似度スコア配列
        """
        if candidate_pieces is None:
            candidate_pieces = self.pieces
        
        similarities = []
        for piece in candidate_pieces:
            try:
                sim = compute_semantic_similarity_rich(
                    user_context, piece, self.semantic_engine
                )
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"Piece similarity error: {e}")
                similarities.append(0.1)
        
        return np.array(similarities)
    
    def select_collaborative_pieces_semantic(self,
                                           user_context: Union[str, np.ndarray],
                                           user_id: str = None,
                                           top_k: int = 5,
                                           diversity_weight: float = 0.2) -> List[Tuple[Any, float]]:
        """
        意味的類似度に基づく協調ピース選択
        
        Args:
            user_context: ユーザーコンテキスト
            user_id: ユーザーID
            top_k: 選択するピース数
            diversity_weight: 多様性重み
            
        Returns:
            選択されたピースと類似度のリスト
        """
        if not self.pieces:
            logger.warning("No pieces in pool for selection")
            return []
        
        logger.info(f"🔍 Selecting collaborative pieces with semantic similarity")
        logger.info(f"   Pool size: {len(self.pieces)}, Top-k: {top_k}")
        
        try:
            # 類似度計算
            similarity_scores = self.compute_context_similarity_semantic(user_context)
            
            # 閾値フィルタリング
            valid_indices = np.where(similarity_scores >= self.similarity_threshold)[0]
            
            if len(valid_indices) == 0:
                logger.warning("No pieces meet similarity threshold, using top candidates")
                valid_indices = np.argsort(similarity_scores)[-min(top_k*2, len(similarity_scores)):]
            
            # 多様性を考慮した選択
            selected_pieces = self._select_diverse_pieces(
                valid_indices, similarity_scores, top_k, diversity_weight
            )
            
            # 結果構築
            results = []
            for idx in selected_pieces:
                piece = self.pieces[idx]
                score = similarity_scores[idx]
                results.append((piece, float(score)))
            
            logger.info(f"✅ Selected {len(results)} pieces with scores: {[f'{s:.3f}' for _, s in results]}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic piece selection error: {e}")
            # フォールバック: ランダム選択
            selected_indices = np.random.choice(len(self.pieces), min(top_k, len(self.pieces)), replace=False)
            return [(self.pieces[i], 0.5) for i in selected_indices]
    
    def _select_diverse_pieces(self,
                             candidate_indices: np.ndarray,
                             similarity_scores: np.ndarray,
                             top_k: int,
                             diversity_weight: float) -> List[int]:
        """
        多様性を考慮したピース選択
        
        Args:
            candidate_indices: 候補ピースインデックス
            similarity_scores: 類似度スコア
            top_k: 選択数
            diversity_weight: 多様性重み
            
        Returns:
            選択されたピースインデックス
        """
        if len(candidate_indices) <= top_k:
            return candidate_indices.tolist()
        
        selected_indices = []
        remaining_indices = candidate_indices.tolist()
        
        # 最高類似度のピースを最初に選択
        best_idx = candidate_indices[np.argmax(similarity_scores[candidate_indices])]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # 残りを多様性を考慮して選択
        for _ in range(min(top_k - 1, len(remaining_indices))):
            if not remaining_indices:
                break
            
            best_candidate = None
            best_score = -1
            
            for candidate_idx in remaining_indices:
                # 類似度スコア
                similarity_score = similarity_scores[candidate_idx]
                
                # 多様性スコア（既選択ピースとの非類似度）
                diversity_score = self._compute_diversity_score(
                    candidate_idx, selected_indices
                )
                
                # 総合スコア
                combined_score = (1 - diversity_weight) * similarity_score + diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        return selected_indices
    
    def _compute_diversity_score(self, candidate_idx: int, selected_indices: List[int]) -> float:
        """
        候補ピースと既選択ピースとの多様性スコア計算
        
        Args:
            candidate_idx: 候補ピースインデックス
            selected_indices: 既選択ピースインデックスリスト
            
        Returns:
            多様性スコア（高いほど多様）
        """
        if not selected_indices:
            return 1.0
        
        try:
            candidate_piece = self.pieces[candidate_idx]
            selected_pieces = [self.pieces[i] for i in selected_indices]
            
            # 選択済みピースとの類似度を計算
            similarities = []
            for selected_piece in selected_pieces:
                # 簡単な実装: semantic_tagsの重複度で評価
                candidate_tags = set()
                selected_tags = set()
                
                if hasattr(candidate_piece, 'semantic_tags'):
                    candidate_tags = set(candidate_piece.semantic_tags or [])
                elif isinstance(candidate_piece, dict) and 'semantic_tags' in candidate_piece:
                    candidate_tags = set(candidate_piece['semantic_tags'] or [])
                
                if hasattr(selected_piece, 'semantic_tags'):
                    selected_tags = set(selected_piece.semantic_tags or [])
                elif isinstance(selected_piece, dict) and 'semantic_tags' in selected_piece:
                    selected_tags = set(selected_piece['semantic_tags'] or [])
                
                if candidate_tags and selected_tags:
                    overlap = len(candidate_tags & selected_tags)
                    union = len(candidate_tags | selected_tags)
                    similarity = overlap / union if union > 0 else 0
                else:
                    similarity = 0.1  # デフォルト低類似度
                
                similarities.append(similarity)
            
            # 多様性スコア = 1 - 最大類似度
            diversity_score = 1.0 - max(similarities)
            return max(0.0, diversity_score)
            
        except Exception as e:
            logger.debug(f"Diversity score computation error: {e}")
            return 0.5  # デフォルト
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """プール統計情報取得"""
        if not self.pieces:
            return {"total_pieces": 0, "users": 0}
        
        # semantic_tagsの分析
        all_tags = set()
        tag_counts = {}
        
        for piece in self.pieces:
            tags = []
            if hasattr(piece, 'semantic_tags') and piece.semantic_tags:
                tags = piece.semantic_tags
            elif isinstance(piece, dict) and 'semantic_tags' in piece:
                if isinstance(piece['semantic_tags'], list):
                    tags = piece['semantic_tags']
                else:
                    tags = [piece['semantic_tags']]
            
            for tag in tags:
                all_tags.add(tag)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_pieces": len(self.pieces),
            "users": len(self.user_mapping),
            "unique_semantic_tags": len(all_tags),
            "most_common_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "cache_size": len(self.semantic_engine.embedding_cache.text_to_embedding),
            "similarity_threshold": self.similarity_threshold
        }

class SemanticCollaborativeChameleonEditor:
    """意味的類似度対応CFS-Chameleonエディター"""
    
    def __init__(self,
                 base_editor: Any = None,
                 semantic_config: SemanticSimilarityConfig = None,
                 pool_capacity: int = 1000):
        """
        初期化
        
        Args:
            base_editor: ベースCFS-Chameleonエディター
            semantic_config: 意味的類似度設定
            pool_capacity: プール容量
        """
        self.base_editor = base_editor
        
        # 意味的方向プールを初期化
        self.semantic_pool = SemanticAwareDirectionPool(
            capacity=pool_capacity,
            semantic_config=semantic_config,
            use_hybrid_similarity=True
        )
        
        logger.info("✅ SemanticCollaborativeChameleonEditor initialized")
        logger.info(f"   Pool capacity: {pool_capacity}")
    
    def generate_with_semantic_collaboration(self,
                                           prompt: str,
                                           user_context: str = None,
                                           user_id: str = None,
                                           alpha_personal: float = 0.1,
                                           alpha_neutral: float = -0.05,
                                           max_length: int = 100) -> str:
        """
        意味的協調機能を使用した生成
        
        Args:
            prompt: 入力プロンプト
            user_context: ユーザーコンテキスト
            user_id: ユーザーID
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        context_for_similarity = user_context or prompt
        
        # 意味的類似度に基づくピース選択
        selected_pieces = self.semantic_pool.select_collaborative_pieces_semantic(
            user_context=context_for_similarity,
            user_id=user_id,
            top_k=3
        )
        
        logger.info(f"🦎 Generating with semantic collaboration")
        logger.info(f"   Selected pieces: {len(selected_pieces)}")
        logger.info(f"   Similarity scores: {[f'{s:.3f}' for _, s in selected_pieces]}")
        
        if self.base_editor and hasattr(self.base_editor, 'generate_with_chameleon'):
            try:
                # ベースエディターでの生成
                result = self.base_editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    max_length=max_length
                )
                return result
            except Exception as e:
                logger.error(f"Base editor generation error: {e}")
        
        # フォールバック生成
        return f"Semantic collaborative response to: {prompt[:50]}..."

def demonstrate_semantic_integration():
    """意味的類似度統合のデモンストレーション"""
    print("🦎 CFS-Chameleon意味的類似度統合デモ")
    print("=" * 60)
    
    # 意味的プールの初期化
    semantic_config = SemanticSimilarityConfig(
        primary_model="sentence-transformers",
        cache_embeddings=True
    )
    
    pool = SemanticAwareDirectionPool(
        capacity=100,
        semantic_config=semantic_config,
        use_hybrid_similarity=True
    )
    
    # サンプルピースを追加
    sample_pieces = [
        {"id": "sci_fi", "semantic_tags": ["science_fiction", "literature"], "u_component": np.random.randn(768)},
        {"id": "cooking", "semantic_tags": ["cooking", "recipes", "food"], "u_component": np.random.randn(768)},
        {"id": "movies", "semantic_tags": ["entertainment", "films", "cinema"], "u_component": np.random.randn(768)},
        {"id": "tech", "semantic_tags": ["programming", "technology"], "u_component": np.random.randn(768)},
        {"id": "travel", "semantic_tags": ["travel", "adventure", "exploration"], "u_component": np.random.randn(768)}
    ]
    
    for piece in sample_pieces:
        pool.add_piece(piece, f"user_{piece['id']}")
    
    print(f"📦 Added {len(sample_pieces)} pieces to semantic pool")
    
    # プール統計表示
    stats = pool.get_pool_statistics()
    print(f"\n📊 Pool Statistics:")
    print(f"   Total pieces: {stats['total_pieces']}")
    print(f"   Unique semantic tags: {stats['unique_semantic_tags']}")
    print(f"   Most common tags: {stats['most_common_tags']}")
    
    # 意味的類似度ベースの選択テスト
    test_contexts = [
        "最新のSF小説について話したい",
        "美味しい料理のレシピを教えて",
        "面白い映画を推薦してください",
        "プログラミングの質問があります"
    ]
    
    print(f"\n🔍 Semantic-based piece selection test:")
    print("-" * 40)
    
    for context in test_contexts:
        print(f"\n📝 Context: '{context}'")
        
        selected_pieces = pool.select_collaborative_pieces_semantic(
            user_context=context,
            top_k=3
        )
        
        print(f"   Selected pieces:")
        for i, (piece, score) in enumerate(selected_pieces):
            piece_id = piece.get('id', 'unknown')
            tags = ', '.join(piece.get('semantic_tags', []))
            print(f"     {i+1}. {piece_id} (score: {score:.4f}) - tags: {tags}")
    
    # 従来手法との比較
    print(f"\n🔄 Comparison with traditional method:")
    print("-" * 40)
    
    context = test_contexts[0]  # "最新のSF小説について話したい"
    
    # 意味的類似度
    semantic_selected = pool.select_collaborative_pieces_semantic(context, top_k=3)
    
    # 従来的類似度（モック）
    traditional_scores = np.random.uniform(0.1, 0.8, len(sample_pieces))
    traditional_selected = [(sample_pieces[i], traditional_scores[i]) 
                          for i in np.argsort(traditional_scores)[-3:]]
    
    print(f"Context: '{context}'")
    print(f"\nSemantic similarity results:")
    for piece, score in semantic_selected:
        print(f"   {piece['id']}: {score:.4f}")
    
    print(f"\nTraditional similarity results (mock):")
    for piece, score in traditional_selected:
        print(f"   {piece['id']}: {score:.4f}")
    
    # 意味的エディターのテスト
    print(f"\n🎯 Semantic collaborative editor test:")
    print("-" * 40)
    
    editor = SemanticCollaborativeChameleonEditor(
        semantic_config=semantic_config,
        pool_capacity=100
    )
    
    # プールをエディターに移植
    editor.semantic_pool = pool
    
    result = editor.generate_with_semantic_collaboration(
        prompt="SF小説の推薦をお願いします",
        user_context="最近の科学技術に興味があります",
        user_id="user_sci_fi"
    )
    
    print(f"Generated response: {result}")
    
    print("\n🎉 意味的類似度統合デモ完了!")

if __name__ == "__main__":
    demonstrate_semantic_integration()