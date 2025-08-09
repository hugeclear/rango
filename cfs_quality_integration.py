#!/usr/bin/env python3
"""
CFS-Chameleon品質スコア統合モジュール
タスクベース品質評価をCFS-Chameleonシステムに統合
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
import json
from pathlib import Path

# タスクベース品質評価モジュール
from task_based_quality_evaluator import (
    calculate_improved_quality_score,
    QualityEvaluationConfig,
    create_sample_evaluation_datasets
)

# CFS-Chameleon関連モジュール
try:
    from cfs_chameleon_extension import DirectionPiece, CollaborativeDirectionPool
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    from cfs_improved_integration import ImprovedCFSChameleonEditor
    CFS_AVAILABLE = True
except ImportError:
    print("⚠️ CFS modules not available")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityAwareCFSEditor:
    """品質評価機能付きCFS-Chameleonエディター"""
    
    def __init__(self, 
                 base_config_path: str = "cfs_config.yaml",
                 quality_config: QualityEvaluationConfig = None,
                 enable_quality_evaluation: bool = True):
        """
        初期化
        
        Args:
            base_config_path: ベースCFS設定ファイル
            quality_config: 品質評価設定
            enable_quality_evaluation: 品質評価の有効化
        """
        self.base_config_path = base_config_path
        self.quality_config = quality_config or QualityEvaluationConfig(
            metrics=["rouge", "bleu", "bertscore"],
            metric_weights={"rouge": 0.4, "bleu": 0.3, "bertscore": 0.3},
            max_eval_samples=20
        )
        self.enable_quality_evaluation = enable_quality_evaluation
        
        # CFS-Chameleonエディターの初期化
        self._initialize_cfs_editor()
        
        # 評価データセットのセットアップ
        self.evaluation_datasets = self._setup_evaluation_data()
        
        logger.info("✅ QualityAwareCFSEditor initialized")
        logger.info(f"   Quality evaluation: {'enabled' if enable_quality_evaluation else 'disabled'}")
        logger.info(f"   Available datasets: {list(self.evaluation_datasets.keys())}")
    
    def _initialize_cfs_editor(self):
        """CFS-Chameleonエディターの初期化"""
        if CFS_AVAILABLE:
            try:
                self.cfs_editor = ImprovedCFSChameleonEditor(
                    use_collaboration=True,
                    config_path=self.base_config_path,
                    enable_improved_pieces=True
                )
                logger.info("✅ ImprovedCFSChameleonEditor loaded")
            except Exception as e:
                logger.warning(f"ImprovedCFSChameleonEditor failed, using basic editor: {e}")
                self.cfs_editor = CollaborativeChameleonEditor(
                    use_collaboration=True,
                    config_path=self.base_config_path
                )
        else:
            self.cfs_editor = None
            logger.warning("⚠️ CFS-Chameleon not available, using mock editor")
    
    def _setup_evaluation_data(self) -> Dict[str, List[Tuple[str, str]]]:
        """評価データのセットアップ"""
        # デフォルトデータセット
        datasets = create_sample_evaluation_datasets()
        
        # カスタムCFS特化データセットを追加
        datasets["cfs_personalization"] = [
            (
                "私の好みに合った映画を推薦してください",
                "あなたの好みを考慮した映画をおすすめします"
            ),
            (
                "個人的な質問に答えてください：今日の気分はどうですか？", 
                "個人的な状況を踏まえてお答えします"
            ),
            (
                "私の履歴に基づいてアドバイスをください",
                "あなたの過去の経験を考慮したアドバイスを提供します"
            ),
            (
                "パーソナライズされた学習計画を作成してください",
                "あなたに最適化された学習プランを提案します"
            ),
            (
                "私の興味に合った話題で会話しましょう",
                "あなたの関心事について楽しく話しましょう"
            )
        ]
        
        return datasets
    
    def generate_with_quality_piece(self, 
                                  input_text: str, 
                                  piece: Any,
                                  alpha_personal: float = 0.1,
                                  alpha_neutral: float = -0.05) -> str:
        """
        品質評価対応の生成関数
        
        Args:
            input_text: 入力テキスト
            piece: 方向ピース
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            
        Returns:
            生成されたテキスト
        """
        if self.cfs_editor is None:
            # モック生成
            return f"Mock quality-aware generation: {input_text[:50]}..."
        
        try:
            # CFS-Chameleonでの生成
            if hasattr(self.cfs_editor, 'generate_with_improved_collaboration'):
                result = self.cfs_editor.generate_with_improved_collaboration(
                    prompt=input_text,
                    user_id="quality_eval_user",
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    max_length=self.quality_config.generation_max_length
                )
            else:
                result = self.cfs_editor.generate_with_chameleon(
                    prompt=input_text,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    max_length=self.quality_config.generation_max_length
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Generation failed: {input_text[:30]}..."
    
    def evaluate_piece_quality(self, 
                             piece: Any,
                             dataset_name: str = "cfs_personalization") -> float:
        """
        方向ピースの品質評価
        
        Args:
            piece: 評価対象の方向ピース
            dataset_name: 使用する評価データセット名
            
        Returns:
            品質スコア
        """
        if not self.enable_quality_evaluation:
            logger.info("Quality evaluation disabled, returning default score")
            return 0.5
        
        if dataset_name not in self.evaluation_datasets:
            logger.warning(f"Dataset {dataset_name} not found, using default")
            dataset_name = "cfs_personalization"
        
        eval_dataset = self.evaluation_datasets[dataset_name]
        
        # 品質評価実行
        logger.info(f"🎯 Evaluating piece quality on {dataset_name} dataset")
        
        def generate_func(input_text: str, piece: Any) -> str:
            return self.generate_with_quality_piece(input_text, piece)
        
        try:
            quality_score = calculate_improved_quality_score(
                piece=piece,
                eval_dataset=eval_dataset,
                generate_with_piece=generate_func,
                config=self.quality_config
            )
            
            logger.info(f"✅ Piece quality evaluation completed: {quality_score:.4f}")
            return quality_score
            
        except Exception as e:
            logger.error(f"❌ Quality evaluation failed: {e}")
            return 0.0
    
    def batch_evaluate_pieces_quality(self, 
                                    pieces: List[Any],
                                    dataset_name: str = "cfs_personalization") -> List[float]:
        """
        複数方向ピースの一括品質評価
        
        Args:
            pieces: 評価対象の方向ピースリスト
            dataset_name: 使用する評価データセット名
            
        Returns:
            品質スコアのリスト
        """
        logger.info(f"🔄 Batch evaluating {len(pieces)} pieces")
        
        quality_scores = []
        for i, piece in enumerate(pieces):
            logger.info(f"   Evaluating piece {i+1}/{len(pieces)}")
            
            score = self.evaluate_piece_quality(piece, dataset_name)
            quality_scores.append(score)
        
        logger.info(f"✅ Batch evaluation completed")
        logger.info(f"   Average quality: {np.mean(quality_scores):.4f}")
        logger.info(f"   Score range: {np.min(quality_scores):.4f} - {np.max(quality_scores):.4f}")
        
        return quality_scores
    
    def update_pieces_with_quality_scores(self, 
                                        pieces: List[Any],
                                        dataset_name: str = "cfs_personalization") -> List[Any]:
        """
        方向ピースの品質スコアを更新
        
        Args:
            pieces: 方向ピースリスト
            dataset_name: 評価データセット名
            
        Returns:
            品質スコア更新済みピーステリスト
        """
        quality_scores = self.batch_evaluate_pieces_quality(pieces, dataset_name)
        
        updated_pieces = []
        for piece, quality_score in zip(pieces, quality_scores):
            # ピースの品質スコアを更新
            if hasattr(piece, 'quality_score'):
                piece.quality_score = quality_score
            elif isinstance(piece, dict):
                piece['quality_score'] = quality_score
            else:
                logger.warning(f"Cannot update quality score for piece type: {type(piece)}")
            
            updated_pieces.append(piece)
        
        return updated_pieces
    
    def generate_quality_report(self, 
                              pieces: List[Any],
                              dataset_name: str = "cfs_personalization") -> Dict[str, Any]:
        """
        品質評価レポートの生成
        
        Args:
            pieces: 方向ピースリスト
            dataset_name: 評価データセット名
            
        Returns:
            品質レポート
        """
        logger.info("📊 Generating quality evaluation report")
        
        quality_scores = self.batch_evaluate_pieces_quality(pieces, dataset_name)
        
        report = {
            "evaluation_timestamp": time.time(),
            "dataset_name": dataset_name,
            "total_pieces": len(pieces),
            "quality_scores": quality_scores,
            "statistics": {
                "mean": float(np.mean(quality_scores)),
                "std": float(np.std(quality_scores)),
                "min": float(np.min(quality_scores)),
                "max": float(np.max(quality_scores)),
                "median": float(np.median(quality_scores))
            },
            "quality_distribution": {
                "high_quality_count": sum(1 for s in quality_scores if s > 0.7),
                "medium_quality_count": sum(1 for s in quality_scores if 0.3 < s <= 0.7),
                "low_quality_count": sum(1 for s in quality_scores if s <= 0.3)
            },
            "evaluation_config": {
                "metrics": self.quality_config.metrics,
                "metric_weights": self.quality_config.metric_weights,
                "max_eval_samples": self.quality_config.max_eval_samples
            }
        }
        
        return report

def demonstrate_quality_aware_cfs():
    """品質評価機能付きCFS-Chameleonのデモンストレーション"""
    print("🦎 品質評価機能付きCFS-Chameleonデモ")
    print("=" * 60)
    
    # 品質評価設定
    quality_config = QualityEvaluationConfig(
        metrics=["rouge", "bleu", "bertscore"],
        metric_weights={"rouge": 0.4, "bleu": 0.3, "bertscore": 0.3},
        max_eval_samples=5,
        generation_max_length=80
    )
    
    # 品質評価機能付きエディターの初期化
    editor = QualityAwareCFSEditor(
        base_config_path="cfs_config.yaml",
        quality_config=quality_config,
        enable_quality_evaluation=True
    )
    
    # サンプル方向ピースの作成（実際にはDirectionPieceオブジェクト）
    sample_pieces = [
        {"id": "piece_1", "vector": np.random.randn(3072), "quality_score": 0.0},
        {"id": "piece_2", "vector": np.random.randn(3072), "quality_score": 0.0},
        {"id": "piece_3", "vector": np.random.randn(3072), "quality_score": 0.0}
    ]
    
    print(f"\n📝 Sample pieces created: {len(sample_pieces)}")
    
    # 個別ピース品質評価のテスト
    print("\n🎯 Individual piece quality evaluation:")
    for i, piece in enumerate(sample_pieces):
        print(f"\n🔸 Evaluating piece {i+1}:")
        
        score = editor.evaluate_piece_quality(piece, "cfs_personalization")
        print(f"   Quality Score: {score:.4f}")
    
    # 一括品質評価のテスト
    print("\n🔄 Batch quality evaluation:")
    updated_pieces = editor.update_pieces_with_quality_scores(
        sample_pieces, "cfs_personalization"
    )
    
    for i, piece in enumerate(updated_pieces):
        score = piece.get('quality_score', 'N/A')
        print(f"   Piece {i+1}: {score:.4f}")
    
    # 品質レポートの生成
    print("\n📊 Quality evaluation report:")
    report = editor.generate_quality_report(sample_pieces, "cfs_personalization")
    
    print(f"   Total pieces: {report['total_pieces']}")
    print(f"   Average quality: {report['statistics']['mean']:.4f}")
    print(f"   Quality range: {report['statistics']['min']:.4f} - {report['statistics']['max']:.4f}")
    print(f"   High quality pieces: {report['quality_distribution']['high_quality_count']}")
    print(f"   Medium quality pieces: {report['quality_distribution']['medium_quality_count']}")
    print(f"   Low quality pieces: {report['quality_distribution']['low_quality_count']}")
    
    # レポート保存
    report_file = "quality_evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Quality report saved: {report_file}")
    print("\n🎉 品質評価機能付きCFS-Chameleonデモ完了!")

if __name__ == "__main__":
    demonstrate_quality_aware_cfs()