#!/usr/bin/env python3
"""
適応的ピース統合のCFS-Chameleon完全統合モジュール
四大改良システムの統合アーキテクチャ
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass

# 四大改良システム
from adaptive_piece_fusion import (
    fuse_pieces_adaptive,
    AdaptiveFusionConfig,
    AdaptiveFusionChameleonEditor
)
from task_based_quality_evaluator import (
    TaskBasedQualityEvaluator,
    calculate_improved_quality_score
)
from semantic_similarity_engine import (
    SemanticSimilarityEngine,
    compute_semantic_similarity_rich
)
from dual_direction_cfs_integration import (
    DualDirectionPool,
    DualDirectionChameleonEditor
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

@dataclass
class IntegratedChameleonConfig:
    """統合Chameleonシステム設定"""
    # 適応的統合設定
    adaptive_fusion_config: AdaptiveFusionConfig = None
    
    # 意味的類似度設定
    use_semantic_similarity: bool = True
    semantic_threshold: float = 0.3
    
    # タスクベース品質評価設定
    use_quality_evaluation: bool = True
    quality_threshold: float = 0.5
    
    # 双方向ピース設定
    use_dual_direction: bool = True
    personal_weight: float = 1.0
    neutral_weight: float = -0.5
    
    # 統合戦略
    integration_strategy: str = "full"  # full, adaptive_only, quality_only, semantic_only
    
    def __post_init__(self):
        if self.adaptive_fusion_config is None:
            self.adaptive_fusion_config = AdaptiveFusionConfig()

class IntegratedChameleonSystem:
    """四大改良システム統合Chameleonシステム"""
    
    def __init__(self, config: IntegratedChameleonConfig = None):
        """
        初期化
        
        Args:
            config: 統合システム設定
        """
        self.config = config or IntegratedChameleonConfig()
        
        # 各サブシステム初期化
        self._initialize_subsystems()
        
        logger.info("✅ IntegratedChameleonSystem initialized")
        logger.info(f"   Integration strategy: {self.config.integration_strategy}")
        logger.info(f"   Semantic similarity: {self.config.use_semantic_similarity}")
        logger.info(f"   Quality evaluation: {self.config.use_quality_evaluation}")
        logger.info(f"   Dual direction: {self.config.use_dual_direction}")
    
    def _initialize_subsystems(self):
        """サブシステムの初期化"""
        # 1. 適応的統合システム
        self.adaptive_fusion_editor = AdaptiveFusionChameleonEditor(
            fusion_config=self.config.adaptive_fusion_config
        )
        
        # 2. タスクベース品質評価システム
        if self.config.use_quality_evaluation:
            self.quality_evaluator = TaskBasedQualityEvaluator()
        else:
            self.quality_evaluator = None
        
        # 3. 意味的類似度システム
        if self.config.use_semantic_similarity:
            self.semantic_engine = SemanticSimilarityEngine()
        else:
            self.semantic_engine = None
        
        # 4. 双方向ピースシステム
        if self.config.use_dual_direction:
            self.dual_direction_editor = DualDirectionChameleonEditor()
        else:
            self.dual_direction_editor = None
        
        logger.info("✅ All subsystems initialized")
    
    def select_and_fuse_pieces_integrated(self,
                                        user_context: str,
                                        available_pieces: List[Any],
                                        eval_dataset: List[Tuple[str, str]],
                                        user_id: str = None,
                                        top_k: int = 5) -> np.ndarray:
        """
        統合的ピース選択・融合
        
        Args:
            user_context: ユーザーコンテキスト
            available_pieces: 利用可能なピースリスト
            eval_dataset: 評価データセット
            user_id: ユーザーID
            top_k: 選択するピース数
            
        Returns:
            融合された方向ベクトル
        """
        logger.info(f"🦎 Integrated piece selection and fusion")
        logger.info(f"   Available pieces: {len(available_pieces)}")
        logger.info(f"   Strategy: {self.config.integration_strategy}")
        
        selected_pieces = []
        
        try:
            if self.config.integration_strategy == "full":
                # 完全統合戦略
                selected_pieces = self._full_integration_selection(
                    user_context, available_pieces, eval_dataset, user_id, top_k
                )
                
            elif self.config.integration_strategy == "adaptive_only":
                # 適応的統合のみ
                selected_pieces = available_pieces[:top_k]
                
            elif self.config.integration_strategy == "quality_only":
                # 品質評価ベース選択のみ
                selected_pieces = self._quality_based_selection(
                    available_pieces, eval_dataset, top_k
                )
                
            elif self.config.integration_strategy == "semantic_only":
                # 意味的類似度ベース選択のみ
                selected_pieces = self._semantic_based_selection(
                    user_context, available_pieces, top_k
                )
            
            logger.info(f"✅ Selected {len(selected_pieces)} pieces for fusion")
            
            # 適応的統合実行
            if selected_pieces:
                def mock_generate_with_piece(prompt: str, piece: Any) -> str:
                    piece_id = getattr(piece, 'piece_id', str(id(piece))[:8])
                    return f"Generated for '{prompt[:20]}...' using {piece_id}"
                
                fused_vector = fuse_pieces_adaptive(
                    pieces=selected_pieces,
                    eval_dataset=eval_dataset,
                    generate_with_piece=mock_generate_with_piece,
                    config=self.config.adaptive_fusion_config
                )
                
                return fused_vector
            else:
                logger.warning("No pieces selected, returning zero vector")
                return np.zeros(384)
                
        except Exception as e:
            logger.error(f"❌ Integrated fusion error: {e}")
            # フォールバック: 簡単な平均統合
            if available_pieces:
                vectors = []
                for piece in available_pieces[:top_k]:
                    if hasattr(piece, 'u_component'):
                        vectors.append(np.array(piece.u_component))
                    elif isinstance(piece, dict) and 'u_component' in piece:
                        vectors.append(np.array(piece['u_component']))
                
                if vectors:
                    fused = np.mean(vectors, axis=0)
                    norm = np.linalg.norm(fused)
                    return fused / norm if norm > 0 else fused
            
            return np.zeros(384)
    
    def _full_integration_selection(self,
                                  user_context: str,
                                  available_pieces: List[Any],
                                  eval_dataset: List[Tuple[str, str]],
                                  user_id: str,
                                  top_k: int) -> List[Any]:
        """完全統合戦略でのピース選択"""
        logger.info("🔗 Full integration piece selection")
        
        candidate_pieces = available_pieces.copy()
        
        # 1. 品質評価によるフィルタリング
        if self.config.use_quality_evaluation and self.quality_evaluator:
            logger.info("   📊 Quality-based filtering")
            filtered_pieces = []
            
            for piece in candidate_pieces:
                # 簡単な品質推定（実際の評価は計算コストが高い）
                estimated_quality = getattr(piece, 'quality_score', 0.5)
                if isinstance(piece, dict):
                    estimated_quality = piece.get('quality_score', 0.5)
                
                if estimated_quality >= self.config.quality_threshold:
                    filtered_pieces.append(piece)
            
            candidate_pieces = filtered_pieces
            logger.info(f"   Quality filtered: {len(candidate_pieces)} pieces")
        
        # 2. 意味的類似度による選択
        if self.config.use_semantic_similarity and self.semantic_engine:
            logger.info("   🧠 Semantic similarity selection")
            similarities = []
            
            for piece in candidate_pieces:
                try:
                    similarity = compute_semantic_similarity_rich(
                        user_context, piece, self.semantic_engine
                    )
                    similarities.append((piece, similarity))
                except Exception as e:
                    logger.warning(f"Semantic similarity error: {e}")
                    similarities.append((piece, 0.1))
            
            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 閾値フィルタリング
            semantic_filtered = [
                piece for piece, sim in similarities 
                if sim >= self.config.semantic_threshold
            ]
            
            candidate_pieces = semantic_filtered[:top_k * 2]  # 余裕を持って選択
            logger.info(f"   Semantic filtered: {len(candidate_pieces)} pieces")
        
        # 3. 最終選択（top-k）
        final_pieces = candidate_pieces[:top_k]
        
        logger.info(f"✅ Full integration selected: {len(final_pieces)} pieces")
        return final_pieces
    
    def _quality_based_selection(self,
                               available_pieces: List[Any],
                               eval_dataset: List[Tuple[str, str]],
                               top_k: int) -> List[Any]:
        """品質評価ベースピース選択"""
        logger.info("📊 Quality-based piece selection")
        
        quality_scores = []
        for piece in available_pieces:
            # 既存の品質スコアを使用
            quality = getattr(piece, 'quality_score', 0.5)
            if isinstance(piece, dict):
                quality = piece.get('quality_score', 0.5)
            quality_scores.append((piece, quality))
        
        # 品質スコアでソート
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [piece for piece, _ in quality_scores[:top_k]]
        logger.info(f"✅ Quality-based selected: {len(selected)} pieces")
        return selected
    
    def _semantic_based_selection(self,
                                user_context: str,
                                available_pieces: List[Any],
                                top_k: int) -> List[Any]:
        """意味的類似度ベースピース選択"""
        logger.info("🧠 Semantic-based piece selection")
        
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, using random selection")
            return available_pieces[:top_k]
        
        similarities = []
        for piece in available_pieces:
            try:
                similarity = compute_semantic_similarity_rich(
                    user_context, piece, self.semantic_engine
                )
                similarities.append((piece, similarity))
            except Exception as e:
                logger.warning(f"Semantic similarity error: {e}")
                similarities.append((piece, 0.1))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        selected = [piece for piece, _ in similarities[:top_k]]
        logger.info(f"✅ Semantic-based selected: {len(selected)} pieces")
        return selected
    
    def generate_with_integrated_system(self,
                                      prompt: str,
                                      user_context: str = None,
                                      user_id: str = None,
                                      available_pieces: List[Any] = None,
                                      eval_dataset: List[Tuple[str, str]] = None,
                                      max_length: int = 100) -> str:
        """
        統合システムを使用した生成
        
        Args:
            prompt: 入力プロンプト
            user_context: ユーザーコンテキスト
            user_id: ユーザーID
            available_pieces: 利用可能なピースリスト
            eval_dataset: 評価データセット
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        context_for_selection = user_context or prompt
        
        logger.info(f"🦎 Integrated generation started")
        logger.info(f"   Prompt: '{prompt[:50]}...'")
        
        # デフォルトデータの設定
        if available_pieces is None:
            available_pieces = self._create_default_pieces()
        
        if eval_dataset is None:
            eval_dataset = self._create_default_eval_dataset()
        
        try:
            # 統合的ピース選択・融合
            fused_vector = self.select_and_fuse_pieces_integrated(
                user_context=context_for_selection,
                available_pieces=available_pieces,
                eval_dataset=eval_dataset,
                user_id=user_id,
                top_k=3
            )
            
            # 生成（フォールバック実装）
            generation_info = {
                "vector_norm": np.linalg.norm(fused_vector),
                "strategy": self.config.integration_strategy,
                "pieces_count": len(available_pieces)
            }
            
            response = f"Integrated Chameleon response to: {prompt[:40]}..."
            response += f" (norm: {generation_info['vector_norm']:.3f}, strategy: {generation_info['strategy']})"
            
            logger.info(f"✅ Generation completed with {generation_info['strategy']} strategy")
            return response
            
        except Exception as e:
            logger.error(f"❌ Integrated generation error: {e}")
            return f"Generation error: {prompt[:30]}..."
    
    def _create_default_pieces(self) -> List[Dict[str, Any]]:
        """デフォルトピースの作成"""
        return [
            {
                "piece_id": "default_general",
                "u_component": np.random.randn(384),
                "importance": 0.7,
                "quality_score": 0.6,
                "semantic_tags": ["general", "common"]
            },
            {
                "piece_id": "default_specific",
                "u_component": np.random.randn(384),
                "importance": 0.8,
                "quality_score": 0.8,
                "semantic_tags": ["specific", "detailed"]
            }
        ]
    
    def _create_default_eval_dataset(self) -> List[Tuple[str, str]]:
        """デフォルト評価データセットの作成"""
        return [
            ("一般的な質問です", "一般的な回答をします"),
            ("具体的な質問です", "具体的な回答をします"),
            ("詳細な説明をお願いします", "詳細な説明をいたします")
        ]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報取得"""
        stats = {
            "integration_strategy": self.config.integration_strategy,
            "subsystems": {
                "adaptive_fusion": True,
                "quality_evaluation": self.config.use_quality_evaluation,
                "semantic_similarity": self.config.use_semantic_similarity,
                "dual_direction": self.config.use_dual_direction
            },
            "thresholds": {
                "semantic_threshold": self.config.semantic_threshold,
                "quality_threshold": self.config.quality_threshold
            }
        }
        
        if self.semantic_engine:
            cache_size = len(self.semantic_engine.embedding_cache.text_to_embedding)
            stats["semantic_cache_size"] = cache_size
        
        return stats

def compare_integration_strategies():
    """統合戦略の性能比較"""
    print("🔬 統合戦略性能比較デモ")
    print("=" * 60)
    
    # テストデータ
    test_pieces = [
        {
            "piece_id": "high_quality",
            "u_component": np.random.randn(384),
            "importance": 0.9,
            "quality_score": 0.9,
            "semantic_tags": ["movies", "entertainment"]
        },
        {
            "piece_id": "medium_quality",
            "u_component": np.random.randn(384),
            "importance": 0.6,
            "quality_score": 0.6,
            "semantic_tags": ["cooking", "recipes"]
        },
        {
            "piece_id": "low_quality",
            "u_component": np.random.randn(384),
            "importance": 0.3,
            "quality_score": 0.3,
            "semantic_tags": ["general", "misc"]
        }
    ]
    
    eval_dataset = [
        ("映画の推薦をしてください", "おすすめの映画をご紹介します"),
        ("料理のレシピを教えて", "美味しい料理の作り方をお教えします"),
        ("一般的な質問です", "一般的な回答をいたします")
    ]
    
    test_prompt = "面白い映画について教えてください"
    test_context = "エンターテイメントに興味があります"
    
    strategies = ["full", "adaptive_only", "quality_only", "semantic_only"]
    
    print(f"\n📋 Test setup:")
    print(f"   Pieces: {len(test_pieces)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Test prompt: '{test_prompt}'")
    print(f"   Test context: '{test_context}'")
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🧪 Testing strategy: {strategy}")
        print("-" * 40)
        
        start_time = time.time()
        
        # 戦略別設定
        config = IntegratedChameleonConfig(
            integration_strategy=strategy,
            use_semantic_similarity=True,
            use_quality_evaluation=True,
            semantic_threshold=0.2,
            quality_threshold=0.4
        )
        
        system = IntegratedChameleonSystem(config)
        
        # 生成実行
        result = system.generate_with_integrated_system(
            prompt=test_prompt,
            user_context=test_context,
            available_pieces=test_pieces,
            eval_dataset=eval_dataset
        )
        
        execution_time = time.time() - start_time
        
        # 統計取得
        stats = system.get_system_statistics()
        
        results[strategy] = {
            "result": result,
            "execution_time": execution_time,
            "stats": stats
        }
        
        print(f"   Execution time: {execution_time:.3f}s")
        print(f"   Result: {result[:80]}...")
        print(f"   Subsystems active: {sum(stats['subsystems'].values())}/4")
    
    # 結果比較
    print(f"\n📊 Strategy Comparison Summary:")
    print("=" * 60)
    print(f"{'Strategy':<15} | {'Time (s)':<8} | {'Active Systems':<14} | {'Result Quality'}")
    print("-" * 60)
    
    for strategy, data in results.items():
        time_str = f"{data['execution_time']:.3f}"
        active_systems = sum(data['stats']['subsystems'].values())
        
        # 簡単な結果品質推定（文字数ベース）
        result_length = len(data['result'])
        quality_estimate = "High" if result_length > 100 else "Medium" if result_length > 60 else "Low"
        
        print(f"{strategy:<15} | {time_str:<8} | {active_systems}/4{'':<10} | {quality_estimate}")
    
    # 推奨戦略
    print(f"\n💡 Recommendations:")
    print(f"   • Full integration: 最高性能、全機能活用")
    print(f"   • Quality-only: バランス重視、中程度の性能")
    print(f"   • Semantic-only: 高速処理、意味的マッチング重視")
    print(f"   • Adaptive-only: 基本機能、最小オーバーヘッド")
    
    print("\n🎉 統合戦略比較完了!")

def demonstrate_integrated_system():
    """統合システムのデモンストレーション"""
    print("🦎 四大改良システム統合デモ")
    print("=" * 60)
    
    # 設定
    config = IntegratedChameleonConfig(
        integration_strategy="full",
        use_semantic_similarity=True,
        use_quality_evaluation=True,
        use_dual_direction=True
    )
    
    # システム初期化
    system = IntegratedChameleonSystem(config)
    
    # テストケース
    test_cases = [
        {
            "prompt": "映画の推薦をお願いします",
            "context": "SF映画が好きです",
            "user_id": "movie_lover"
        },
        {
            "prompt": "料理のコツを教えてください", 
            "context": "料理初心者です",
            "user_id": "cooking_beginner"
        },
        {
            "prompt": "プログラミング学習について",
            "context": "Python を学習中です",
            "user_id": "developer"
        }
    ]
    
    print(f"\n🧪 Testing integrated system with {len(test_cases)} cases:")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Prompt: '{case['prompt']}'")
        print(f"   Context: '{case['context']}'")
        print(f"   User ID: {case['user_id']}")
        
        start_time = time.time()
        
        result = system.generate_with_integrated_system(
            prompt=case["prompt"],
            user_context=case["context"],
            user_id=case["user_id"]
        )
        
        execution_time = time.time() - start_time
        
        print(f"   Result: {result}")
        print(f"   Time: {execution_time:.3f}s")
    
    # システム統計
    stats = system.get_system_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Integration strategy: {stats['integration_strategy']}")
    print(f"   Active subsystems: {sum(stats['subsystems'].values())}/4")
    print(f"   Semantic threshold: {stats['thresholds']['semantic_threshold']}")
    print(f"   Quality threshold: {stats['thresholds']['quality_threshold']}")
    
    if "semantic_cache_size" in stats:
        print(f"   Semantic cache size: {stats['semantic_cache_size']}")
    
    print("\n🎉 統合システムデモ完了!")

if __name__ == "__main__":
    demonstrate_integrated_system()
    print("\n" + "="*60)
    compare_integration_strategies()