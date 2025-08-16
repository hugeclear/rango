#!/usr/bin/env python3
"""
CFS-Chameleon向けタスク適応化ピース統合システム
実際のタスク性能に基づく動的重み付けによる最適な方向ベクトル融合
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

# 既存の品質評価システム
from task_based_quality_evaluator import (
    TaskBasedQualityEvaluator,
    QualityEvaluationConfig,
    calculate_improved_quality_score
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
class AdaptiveFusionConfig:
    """タスク適応化統合設定"""
    # 重み計算方式
    weight_method: str = "softmax"  # softmax, linear, learned
    temperature: float = 1.0        # softmax温度パラメータ
    
    # 評価設定
    eval_sample_size: int = 20      # 評価サンプル数
    metrics: List[str] = None       # 使用メトリクス
    parallel_evaluation: bool = True # 並列評価フラグ
    max_workers: int = 4            # 並列ワーカー数
    
    # 統合設定
    min_weight_threshold: float = 0.01  # 最小重み閾値
    normalize_vectors: bool = True      # ベクトル正規化フラグ
    
    # キャッシュ設定
    cache_evaluations: bool = True      # 評価結果キャッシュ
    cache_max_size: int = 1000         # キャッシュ最大サイズ
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["rouge", "bertscore"]

class PiecePerformanceEvaluator:
    """ピース単体性能評価エンジン"""
    
    def __init__(self, config: AdaptiveFusionConfig = None):
        """
        初期化
        
        Args:
            config: 適応化統合設定
        """
        self.config = config or AdaptiveFusionConfig()
        
        # 品質評価器の初期化
        self.quality_evaluator = TaskBasedQualityEvaluator()
        
        # 評価キャッシュ
        self.evaluation_cache = {} if self.config.cache_evaluations else None
        
        logger.info("✅ PiecePerformanceEvaluator initialized")
        logger.info(f"   Weight method: {self.config.weight_method}")
        logger.info(f"   Metrics: {self.config.metrics}")
        logger.info(f"   Parallel evaluation: {self.config.parallel_evaluation}")
    
    def _get_cache_key(self, piece_id: str, eval_data_hash: str) -> str:
        """評価キャッシュキー生成"""
        return f"{piece_id}_{eval_data_hash}_{'-'.join(self.config.metrics)}"
    
    def _hash_eval_dataset(self, eval_dataset: List[Tuple[str, str]]) -> str:
        """評価データセットのハッシュ値計算"""
        import hashlib
        data_str = json.dumps(eval_dataset, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def evaluate_single_piece(self, 
                            piece: Any,
                            eval_dataset: List[Tuple[str, str]],
                            generate_with_piece: Callable[[str, Any], str]) -> float:
        """
        単一ピースの性能評価
        
        Args:
            piece: 評価対象のDirectionPiece
            eval_dataset: 評価データセット [(input, reference), ...]
            generate_with_piece: (prompt, piece) -> 生成文 の関数
            
        Returns:
            性能スコア (0.0-1.0)
        """
        piece_id = getattr(piece, 'piece_id', str(id(piece)))
        
        # キャッシュチェック
        if self.evaluation_cache is not None:
            data_hash = self._hash_eval_dataset(eval_dataset)
            cache_key = self._get_cache_key(piece_id, data_hash)
            
            if cache_key in self.evaluation_cache:
                logger.debug(f"Cache hit for piece {piece_id}")
                return self.evaluation_cache[cache_key]
        
        try:
            # サンプルサイズ制限
            eval_sample = eval_dataset[:self.config.eval_sample_size]
            
            logger.debug(f"🔍 Evaluating piece {piece_id} on {len(eval_sample)} samples")
            
            # メトリクス別スコア計算
            total_scores = []
            
            for input_text, reference_text in eval_sample:
                try:
                    # ピースを使用した生成
                    generated_text = generate_with_piece(input_text, piece)
                    
                    # メトリクス計算
                    sample_scores = []
                    
                    if "rouge" in self.config.metrics:
                        rouge_score = self.quality_evaluator.compute_rouge_score(
                            generated_text, reference_text
                        )
                        sample_scores.append(rouge_score)
                    
                    if "bleu" in self.config.metrics:
                        bleu_score = self.quality_evaluator.compute_bleu_score(
                            generated_text, reference_text
                        )
                        sample_scores.append(bleu_score)
                    
                    if "bertscore" in self.config.metrics:
                        bert_score = self.quality_evaluator.compute_bert_score(
                            generated_text, reference_text
                        )
                        sample_scores.append(bert_score)
                    
                    # サンプル平均スコア
                    if sample_scores:
                        sample_avg = np.mean(sample_scores)
                        total_scores.append(sample_avg)
                    
                except Exception as e:
                    logger.warning(f"Sample evaluation error: {e}")
                    continue
            
            # 全体平均スコア
            if total_scores:
                performance_score = np.mean(total_scores)
            else:
                logger.warning(f"No valid scores for piece {piece_id}, using default")
                performance_score = 0.1  # デフォルト低スコア
            
            # キャッシュに保存
            if self.evaluation_cache is not None:
                if len(self.evaluation_cache) >= self.config.cache_max_size:
                    # 古いエントリを削除（簡単な実装）
                    oldest_key = next(iter(self.evaluation_cache))
                    del self.evaluation_cache[oldest_key]
                
                self.evaluation_cache[cache_key] = performance_score
            
            logger.debug(f"✅ Piece {piece_id} performance: {performance_score:.4f}")
            return float(performance_score)
            
        except Exception as e:
            logger.error(f"❌ Piece evaluation error for {piece_id}: {e}")
            return 0.1
    
    def evaluate_pieces_batch(self,
                            pieces: List[Any],
                            eval_dataset: List[Tuple[str, str]],
                            generate_with_piece: Callable[[str, Any], str]) -> List[float]:
        """
        複数ピースのバッチ性能評価
        
        Args:
            pieces: 評価対象のDirectionPieceリスト
            eval_dataset: 評価データセット
            generate_with_piece: 生成関数
            
        Returns:
            各ピースの性能スコアリスト
        """
        logger.info(f"🚀 Batch evaluating {len(pieces)} pieces on {len(eval_dataset)} samples")
        
        if not self.config.parallel_evaluation or len(pieces) == 1:
            # シーケンシャル評価
            performance_scores = []
            for i, piece in enumerate(pieces):
                logger.info(f"   Evaluating piece {i+1}/{len(pieces)}")
                score = self.evaluate_single_piece(piece, eval_dataset, generate_with_piece)
                performance_scores.append(score)
            
        else:
            # 並列評価
            performance_scores = [0.0] * len(pieces)
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # タスク投入
                future_to_index = {}
                for i, piece in enumerate(pieces):
                    future = executor.submit(
                        self.evaluate_single_piece, piece, eval_dataset, generate_with_piece
                    )
                    future_to_index[future] = i
                
                # 結果収集
                completed = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        score = future.result()
                        performance_scores[index] = score
                        completed += 1
                        logger.info(f"   Completed {completed}/{len(pieces)} evaluations")
                    except Exception as e:
                        logger.error(f"Parallel evaluation error for piece {index}: {e}")
                        performance_scores[index] = 0.1
        
        logger.info(f"✅ Batch evaluation completed")
        logger.info(f"   Score range: {min(performance_scores):.4f} - {max(performance_scores):.4f}")
        logger.info(f"   Average score: {np.mean(performance_scores):.4f}")
        
        return performance_scores

class AdaptiveWeightCalculator:
    """適応的重み計算器"""
    
    def __init__(self, config: AdaptiveFusionConfig = None):
        """
        初期化
        
        Args:
            config: 適応化統合設定
        """
        self.config = config or AdaptiveFusionConfig()
        
        logger.info("✅ AdaptiveWeightCalculator initialized")
        logger.info(f"   Weight method: {self.config.weight_method}")
        logger.info(f"   Temperature: {self.config.temperature}")
    
    def compute_softmax_weights(self, performance_scores: List[float]) -> np.ndarray:
        """
        Softmax重み計算
        
        Args:
            performance_scores: 性能スコアリスト
            
        Returns:
            正規化された重み配列
        """
        scores = np.array(performance_scores)
        
        # 温度スケーリング
        scaled_scores = scores / self.config.temperature
        
        # Softmax計算
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # 数値安定化
        weights = exp_scores / np.sum(exp_scores)
        
        logger.debug(f"Softmax weights computed: {weights}")
        return weights
    
    def compute_linear_weights(self, performance_scores: List[float]) -> np.ndarray:
        """
        線形重み計算
        
        Args:
            performance_scores: 性能スコアリスト
            
        Returns:
            正規化された重み配列
        """
        scores = np.array(performance_scores)
        
        # 正の値に調整
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score
        
        # 正規化
        total_score = np.sum(scores)
        if total_score > 0:
            weights = scores / total_score
        else:
            # 全スコアが0の場合は均等重み
            weights = np.ones(len(scores)) / len(scores)
        
        logger.debug(f"Linear weights computed: {weights}")
        return weights
    
    def compute_learned_weights(self, 
                              performance_scores: List[float],
                              pieces: List[Any] = None) -> np.ndarray:
        """
        学習ベース重み計算（簡単な線形回帰）
        
        Args:
            performance_scores: 性能スコアリスト
            pieces: DirectionPieceリスト（特徴抽出用）
            
        Returns:
            学習された重み配列
        """
        # 簡単な実装: 性能スコア + ピース特徴の線形結合
        scores = np.array(performance_scores)
        
        if pieces is not None:
            # ピース特徴の抽出（重要度、品質スコア等）
            piece_features = []
            for piece in pieces:
                features = []
                
                # 重要度特徴
                if hasattr(piece, 'importance'):
                    features.append(piece.importance)
                elif isinstance(piece, dict) and 'importance' in piece:
                    features.append(piece['importance'])
                else:
                    features.append(0.5)
                
                # 品質スコア特徴  
                if hasattr(piece, 'quality_score'):
                    features.append(piece.quality_score)
                elif isinstance(piece, dict) and 'quality_score' in piece:
                    features.append(piece['quality_score'])
                else:
                    features.append(0.5)
                
                piece_features.append(features)
            
            piece_features = np.array(piece_features)
            
            # 簡単な特徴重み付け
            feature_weights = np.array([0.3, 0.2])  # importance, quality_score
            piece_scores = np.dot(piece_features, feature_weights)
            
            # 性能スコアと特徴スコアの結合
            combined_scores = 0.7 * scores + 0.3 * piece_scores
        else:
            combined_scores = scores
        
        # Softmax重み計算
        weights = self.compute_softmax_weights(combined_scores.tolist())
        
        logger.debug(f"Learned weights computed: {weights}")
        return weights
    
    def compute_weights(self, 
                       performance_scores: List[float],
                       pieces: List[Any] = None) -> np.ndarray:
        """
        設定に基づく重み計算
        
        Args:
            performance_scores: 性能スコアリスト
            pieces: DirectionPieceリスト
            
        Returns:
            計算された重み配列
        """
        if self.config.weight_method == "softmax":
            weights = self.compute_softmax_weights(performance_scores)
        elif self.config.weight_method == "linear":
            weights = self.compute_linear_weights(performance_scores)
        elif self.config.weight_method == "learned":
            weights = self.compute_learned_weights(performance_scores, pieces)
        else:
            logger.warning(f"Unknown weight method: {self.config.weight_method}, using softmax")
            weights = self.compute_softmax_weights(performance_scores)
        
        # 最小重み閾値の適用
        weights = np.maximum(weights, self.config.min_weight_threshold)
        weights = weights / np.sum(weights)  # 再正規化
        
        return weights

def fuse_pieces_adaptive(
    pieces: List[Any],
    eval_dataset: List[Tuple[str, str]],
    generate_with_piece: Callable[[str, Any], str],
    config: AdaptiveFusionConfig = None,
    metrics: List[str] = None
) -> np.ndarray:
    """
    タスク指標に基づいてピース統合重みを決定し、方向ベクトルを融合する。

    Args:
        pieces: 統合対象の DirectionPiece リスト
        eval_dataset: List of (input_text, reference_text) ペア
        generate_with_piece: (prompt, piece) -> 生成文 の関数
        config: 適応化統合設定
        metrics: 使用する性能指標リスト（後方互換用）

    Returns:
        np.ndarray: 融合後の方向ベクトル
    """
    logger.info(f"🦎 Adaptive piece fusion started")
    logger.info(f"   Pieces: {len(pieces)}")
    logger.info(f"   Eval samples: {len(eval_dataset)}")
    
    # 設定の初期化
    if config is None:
        config = AdaptiveFusionConfig()
        if metrics is not None:
            config.metrics = metrics
    
    if not pieces:
        logger.warning("No pieces provided for fusion")
        return np.array([])
    
    if len(pieces) == 1:
        logger.info("Single piece provided, returning its vector")
        piece = pieces[0]
        if hasattr(piece, 'u_component'):
            return np.array(piece.u_component)
        elif isinstance(piece, dict) and 'u_component' in piece:
            return np.array(piece['u_component'])
        else:
            logger.error("Invalid piece format")
            return np.array([])
    
    try:
        # 1. 各ピースの性能計測
        logger.info("📊 Step 1: Evaluating piece performances")
        performance_evaluator = PiecePerformanceEvaluator(config)
        performance_scores = performance_evaluator.evaluate_pieces_batch(
            pieces, eval_dataset, generate_with_piece
        )
        
        # 2. 重み計算
        logger.info("⚖️ Step 2: Computing adaptive weights")
        weight_calculator = AdaptiveWeightCalculator(config)
        weights = weight_calculator.compute_weights(performance_scores, pieces)
        
        logger.info(f"   Computed weights: {[f'{w:.4f}' for w in weights]}")
        
        # 3. 重み付き統合
        logger.info("🔗 Step 3: Fusing vectors with adaptive weights")
        
        # ベクトル抽出
        vectors = []
        for piece in pieces:
            if hasattr(piece, 'u_component'):
                vector = np.array(piece.u_component)
            elif isinstance(piece, dict) and 'u_component' in piece:
                vector = np.array(piece['u_component'])
            else:
                logger.warning(f"Invalid piece format, using zero vector")
                vector = np.zeros(384)  # デフォルト次元
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # 次元統一
        if len(vectors) > 1:
            target_dim = max(len(v) for v in vectors)
            aligned_vectors = []
            for vector in vectors:
                if len(vector) < target_dim:
                    # ゼロパディング
                    padded = np.zeros(target_dim)
                    padded[:len(vector)] = vector
                    aligned_vectors.append(padded)
                elif len(vector) > target_dim:
                    # 切り詰め
                    aligned_vectors.append(vector[:target_dim])
                else:
                    aligned_vectors.append(vector)
            vectors = np.array(aligned_vectors)
        
        # 重み付き和
        fused_vector = np.zeros(vectors.shape[1])
        for weight, vector in zip(weights, vectors):
            fused_vector += weight * vector
        
        # 4. 正規化
        if config.normalize_vectors:
            norm = np.linalg.norm(fused_vector)
            if norm > 0:
                fused_vector = fused_vector / norm
            else:
                logger.warning("Zero norm fused vector, skipping normalization")
        
        logger.info(f"✅ Adaptive fusion completed")
        logger.info(f"   Fused vector norm: {np.linalg.norm(fused_vector):.4f}")
        
        return fused_vector
        
    except Exception as e:
        logger.error(f"❌ Adaptive fusion error: {e}")
        # フォールバック: 均等重み統合
        logger.info("Falling back to uniform weight fusion")
        
        vectors = []
        for piece in pieces:
            if hasattr(piece, 'u_component'):
                vectors.append(np.array(piece.u_component))
            elif isinstance(piece, dict) and 'u_component' in piece:
                vectors.append(np.array(piece['u_component']))
        
        if vectors:
            fused_vector = np.mean(vectors, axis=0)
            if config and config.normalize_vectors:
                norm = np.linalg.norm(fused_vector)
                if norm > 0:
                    fused_vector = fused_vector / norm
            return fused_vector
        else:
            return np.array([])

class AdaptiveFusionChameleonEditor:
    """適応的統合対応CFS-Chameleonエディター"""
    
    def __init__(self, 
                 base_editor: Any = None,
                 fusion_config: AdaptiveFusionConfig = None):
        """
        初期化
        
        Args:
            base_editor: ベースCFS-Chameleonエディター
            fusion_config: 適応化統合設定
        """
        self.base_editor = base_editor
        self.fusion_config = fusion_config or AdaptiveFusionConfig()
        
        logger.info("✅ AdaptiveFusionChameleonEditor initialized")
    
    def generate_with_adaptive_fusion(self,
                                    prompt: str,
                                    pieces: List[Any],
                                    eval_dataset: List[Tuple[str, str]],
                                    max_length: int = 100) -> str:
        """
        適応的統合を使用した生成
        
        Args:
            prompt: 入力プロンプト
            pieces: 統合対象のDirectionPieceリスト
            eval_dataset: 評価データセット
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        logger.info(f"🦎 Generating with adaptive fusion")
        
        # モック生成関数
        def mock_generate_with_piece(input_text: str, piece: Any) -> str:
            piece_id = getattr(piece, 'piece_id', 'unknown')
            return f"Generated response for '{input_text[:30]}...' using piece {piece_id}"
        
        try:
            # 適応的統合実行
            fused_vector = fuse_pieces_adaptive(
                pieces, eval_dataset, mock_generate_with_piece, self.fusion_config
            )
            
            if self.base_editor and hasattr(self.base_editor, 'generate_with_chameleon'):
                # ベースエディターでの生成
                result = self.base_editor.generate_with_chameleon(
                    prompt=prompt,
                    max_length=max_length
                )
                return result
            else:
                # フォールバック生成
                return f"Adaptive fusion response to: {prompt[:50]}... (fusion vector norm: {np.linalg.norm(fused_vector):.4f})"
                
        except Exception as e:
            logger.error(f"❌ Adaptive fusion generation error: {e}")
            return f"Generation error: {prompt[:30]}..."

def demonstrate_adaptive_fusion():
    """適応的ピース統合のデモンストレーション"""
    print("🦎 タスク適応化ピース統合デモ")
    print("=" * 60)
    
    # 設定
    config = AdaptiveFusionConfig(
        weight_method="softmax",
        temperature=1.0,
        eval_sample_size=5,
        metrics=["rouge", "bertscore"],
        parallel_evaluation=False  # デモでは逐次実行
    )
    
    # サンプルピース作成
    sample_pieces = [
        {
            "piece_id": "movie_piece",
            "u_component": np.random.randn(384),
            "importance": 0.8,
            "quality_score": 0.7,
            "semantic_tags": ["movies", "entertainment"]
        },
        {
            "piece_id": "cooking_piece", 
            "u_component": np.random.randn(384),
            "importance": 0.6,
            "quality_score": 0.8,
            "semantic_tags": ["cooking", "recipes"]
        },
        {
            "piece_id": "tech_piece",
            "u_component": np.random.randn(384),
            "importance": 0.9,
            "quality_score": 0.6,
            "semantic_tags": ["technology", "programming"]
        }
    ]
    
    # 評価データセット
    eval_dataset = [
        ("映画の推薦をしてください", "面白い映画をお勧めします"),
        ("料理のレシピを教えて", "美味しい料理の作り方をご紹介します"),
        ("プログラミングについて教えて", "プログラミングの基礎を説明します"),
        ("今日の天気はどうですか？", "今日は晴れです"),
        ("好きな本は何ですか？", "様々な本をお勧めできます")
    ]
    
    # モック生成関数
    def mock_generate_with_piece(input_text: str, piece: Any) -> str:
        piece_id = piece.get('piece_id', 'unknown')
        tags = ', '.join(piece.get('semantic_tags', []))
        
        # ピースの意味タグに基づく簡単な生成シミュレーション
        if 'movies' in tags and '映画' in input_text:
            return "おすすめの映画を詳しく紹介いたします。最新作から名作まで幅広くご案内します。"
        elif 'cooking' in tags and '料理' in input_text:
            return "美味しい料理のレシピをステップバイステップで説明いたします。"
        elif 'technology' in tags and 'プログラミング' in input_text:
            return "プログラミングの基礎から応用まで丁寧に解説いたします。"
        else:
            return f"入力「{input_text[:20]}...」に対する{piece_id}からの一般的な回答です。"
    
    print(f"\n📊 Sample pieces:")
    for i, piece in enumerate(sample_pieces):
        print(f"   {i+1}. {piece['piece_id']} - importance: {piece['importance']}, quality: {piece['quality_score']}")
    
    print(f"\n📋 Evaluation dataset: {len(eval_dataset)} samples")
    for i, (inp, ref) in enumerate(eval_dataset):
        print(f"   {i+1}. '{inp}' -> '{ref}'")
    
    # 適応的統合実行
    print(f"\n🚀 Running adaptive piece fusion...")
    start_time = time.time()
    
    fused_vector = fuse_pieces_adaptive(
        pieces=sample_pieces,
        eval_dataset=eval_dataset,
        generate_with_piece=mock_generate_with_piece,
        config=config
    )
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ Adaptive fusion completed!")
    print(f"   Execution time: {execution_time:.2f}s")
    print(f"   Fused vector shape: {fused_vector.shape}")
    print(f"   Fused vector norm: {np.linalg.norm(fused_vector):.4f}")
    
    # 従来手法との比較
    print(f"\n🔄 Comparison with traditional fusion:")
    print("-" * 40)
    
    # 均等重み統合
    uniform_vectors = [np.array(piece['u_component']) for piece in sample_pieces]
    uniform_fused = np.mean(uniform_vectors, axis=0)
    uniform_norm = np.linalg.norm(uniform_fused)
    if uniform_norm > 0:
        uniform_fused = uniform_fused / uniform_norm
    
    # 重要度ベース統合
    importance_weights = [piece['importance'] for piece in sample_pieces]
    importance_weights = np.array(importance_weights) / np.sum(importance_weights)
    importance_fused = np.zeros_like(uniform_vectors[0])
    for w, vec in zip(importance_weights, uniform_vectors):
        importance_fused += w * vec
    importance_norm = np.linalg.norm(importance_fused)
    if importance_norm > 0:
        importance_fused = importance_fused / importance_norm
    
    print(f"Uniform weights fusion:     norm = {np.linalg.norm(uniform_fused):.4f}")
    print(f"Importance-based fusion:    norm = {np.linalg.norm(importance_fused):.4f}")
    print(f"Adaptive task-based fusion: norm = {np.linalg.norm(fused_vector):.4f}")
    
    # ベクトル類似度比較
    if fused_vector.shape == uniform_fused.shape:
        uniform_sim = np.dot(fused_vector, uniform_fused)
        importance_sim = np.dot(fused_vector, importance_fused)
        
        print(f"\nVector similarities to adaptive fusion:")
        print(f"   vs Uniform fusion:     {uniform_sim:.4f}")
        print(f"   vs Importance fusion:  {importance_sim:.4f}")
    
    # エディターのテスト
    print(f"\n🎯 Testing adaptive fusion editor:")
    print("-" * 40)
    
    editor = AdaptiveFusionChameleonEditor(fusion_config=config)
    
    test_prompts = [
        "映画について教えてください",
        "料理のコツを教えて", 
        "プログラミング学習方法"
    ]
    
    for prompt in test_prompts:
        result = editor.generate_with_adaptive_fusion(
            prompt=prompt,
            pieces=sample_pieces,
            eval_dataset=eval_dataset[:3],  # 小さなサブセット
            max_length=80
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Result: {result}")
    
    print("\n🎉 タスク適応化ピース統合デモ完了!")

if __name__ == "__main__":
    demonstrate_adaptive_fusion()