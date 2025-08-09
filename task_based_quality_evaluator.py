#!/usr/bin/env python3
"""
CFS-Chameleon向けタスクベース品質スコア評価システム
実際の生成タスク性能（ROUGE, BLEU, BERTScore）に基づく品質スコア算出
"""

import numpy as np
import torch
from typing import List, Tuple, Callable, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path

# 評価ライブラリのインポート（フォールバック付き）
try:
    from rouge_score import rouge_scorer
    from sacrebleu import BLEU
    from bert_score import score as bert_score
    EVALUATION_LIBS_AVAILABLE = True
except ImportError:
    print("⚠️ Evaluation libraries not available. Using mock implementations.")
    EVALUATION_LIBS_AVAILABLE = False

# CFS-Chameleon関連モジュール
try:
    from cfs_chameleon_extension import DirectionPiece
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    CFS_AVAILABLE = True
except ImportError:
    print("⚠️ CFS modules not available. Using mock implementations.")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskPerformanceMetrics:
    """タスク性能指標の結果"""
    rouge_l: float
    bleu_score: float
    bert_score: float
    weighted_average: float
    sample_count: int

@dataclass
class QualityEvaluationConfig:
    """品質評価設定"""
    metrics: List[str] = None
    metric_weights: Dict[str, float] = None
    max_eval_samples: int = 50
    generation_max_length: int = 100
    normalize_scores: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["rouge", "bleu", "bertscore"]
        
        if self.metric_weights is None:
            self.metric_weights = {
                "rouge": 0.4,
                "bleu": 0.3, 
                "bertscore": 0.3
            }

class TaskBasedQualityEvaluator:
    """タスクベース品質評価器"""
    
    def __init__(self, config: QualityEvaluationConfig = None):
        """
        初期化
        
        Args:
            config: 評価設定
        """
        self.config = config or QualityEvaluationConfig()
        self._initialize_evaluators()
        
        logger.info("✅ TaskBasedQualityEvaluator initialized")
        logger.info(f"   Metrics: {self.config.metrics}")
        logger.info(f"   Weights: {self.config.metric_weights}")
    
    def _initialize_evaluators(self):
        """評価器の初期化"""
        if EVALUATION_LIBS_AVAILABLE:
            # ROUGE評価器
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
            
            # BLEU評価器
            self.bleu_scorer = BLEU()
            
            logger.info("✅ Real evaluation libraries loaded")
        else:
            # モック評価器
            self.rouge_scorer = None
            self.bleu_scorer = None
            logger.warning("⚠️ Using mock evaluation implementations")
    
    def compute_rouge_score(self, prediction: str, reference: str) -> float:
        """ROUGE-L スコア計算"""
        if self.rouge_scorer:
            try:
                scores = self.rouge_scorer.score(reference, prediction)
                return scores['rougeL'].fmeasure
            except Exception as e:
                logger.warning(f"ROUGE computation error: {e}")
                return 0.0
        else:
            # モック実装：簡単な重複率
            pred_tokens = set(prediction.lower().split())
            ref_tokens = set(reference.lower().split())
            if len(ref_tokens) == 0:
                return 0.0
            overlap = len(pred_tokens & ref_tokens)
            return overlap / len(ref_tokens)
    
    def compute_bleu_score(self, prediction: str, reference: str) -> float:
        """BLEU スコア計算"""
        if self.bleu_scorer:
            try:
                # SacreBLEUは複数参照をサポート
                score = self.bleu_scorer.sentence_score(prediction, [reference])
                return score.score / 100.0  # 0-1に正規化
            except Exception as e:
                logger.warning(f"BLEU computation error: {e}")
                return 0.0
        else:
            # モック実装：n-gram重複
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if len(pred_words) == 0 or len(ref_words) == 0:
                return 0.0
            
            # 単純な1-gram BLEU近似
            matches = 0
            for word in pred_words:
                if word in ref_words:
                    matches += 1
            
            precision = matches / len(pred_words)
            recall = matches / len(ref_words)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
    
    def compute_bert_score(self, prediction: str, reference: str) -> float:
        """BERTScore 計算"""
        if EVALUATION_LIBS_AVAILABLE:
            try:
                # BERTScoreは文のリストを期待
                P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
                return F1[0].item()
            except Exception as e:
                logger.warning(f"BERTScore computation error: {e}")
                return 0.0
        else:
            # モック実装：文字レベル類似度
            pred_chars = set(prediction.lower())
            ref_chars = set(reference.lower())
            
            if len(ref_chars) == 0:
                return 0.0
            
            overlap = len(pred_chars & ref_chars)
            return overlap / max(len(pred_chars), len(ref_chars))
    
    def evaluate_single_sample(self, 
                             prediction: str, 
                             reference: str) -> Dict[str, float]:
        """単一サンプルの評価"""
        metrics = {}
        
        if "rouge" in self.config.metrics:
            metrics["rouge"] = self.compute_rouge_score(prediction, reference)
        
        if "bleu" in self.config.metrics:
            metrics["bleu"] = self.compute_bleu_score(prediction, reference)
        
        if "bertscore" in self.config.metrics:
            metrics["bertscore"] = self.compute_bert_score(prediction, reference)
        
        return metrics
    
    def calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """加重平均スコア計算"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, score in metrics.items():
            if metric in self.config.metric_weights:
                weight = self.config.metric_weights[metric]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight

def calculate_improved_quality_score(
    piece: Any,  # DirectionPiece (CFS_AVAILABLEでない場合はAny)
    eval_dataset: List[Tuple[str, str]],
    generate_with_piece: Callable[[str, Any], str],
    metrics: List[str] = None,
    config: QualityEvaluationConfig = None
) -> float:
    """
    タスク性能指標に基づく品質スコアを計算
    
    Args:
        piece: 評価対象の方向ピース
        eval_dataset: List of (input_text, reference_text) ペア
        generate_with_piece: (prompt, piece) → LLM生成結果 を返す関数
        metrics: 使用する指標リスト（後方互換性）
        config: 評価設定
        
    Returns:
        float: 複数指標の加重平均による品質スコア
    """
    # 設定の初期化
    if config is None:
        config = QualityEvaluationConfig()
        if metrics:
            config.metrics = metrics
    
    # 評価器の初期化
    evaluator = TaskBasedQualityEvaluator(config)
    
    logger.info(f"🎯 Starting task-based quality evaluation for piece")
    logger.info(f"   Evaluation samples: {min(len(eval_dataset), config.max_eval_samples)}")
    logger.info(f"   Metrics: {config.metrics}")
    
    # 評価実行
    all_metrics = []
    successful_evaluations = 0
    
    # サンプル数制限
    eval_samples = eval_dataset[:config.max_eval_samples]
    
    for i, (input_text, reference_text) in enumerate(eval_samples):
        try:
            # ピースを適用した生成
            logger.debug(f"Evaluating sample {i+1}/{len(eval_samples)}")
            
            generated_text = generate_with_piece(input_text, piece)
            
            if not generated_text or len(generated_text.strip()) == 0:
                logger.warning(f"Empty generation for sample {i+1}")
                continue
            
            # 各指標の計算
            sample_metrics = evaluator.evaluate_single_sample(
                generated_text, reference_text
            )
            
            all_metrics.append(sample_metrics)
            successful_evaluations += 1
            
        except Exception as e:
            logger.warning(f"Evaluation error for sample {i+1}: {e}")
            continue
    
    if successful_evaluations == 0:
        logger.error("❌ No successful evaluations")
        return 0.0
    
    # 集約統計の計算
    aggregated_metrics = {}
    for metric in config.metrics:
        if metric in all_metrics[0]:  # 最初のサンプルにある指標のみ
            scores = [m[metric] for m in all_metrics if metric in m]
            aggregated_metrics[metric] = np.mean(scores)
    
    # 加重平均による最終品質スコア
    final_quality_score = evaluator.calculate_weighted_score(aggregated_metrics)
    
    # 正規化（オプション）
    if config.normalize_scores:
        # 0-1範囲にクリップ
        final_quality_score = max(0.0, min(1.0, final_quality_score))
    
    logger.info(f"✅ Quality evaluation completed")
    logger.info(f"   Successful evaluations: {successful_evaluations}/{len(eval_samples)}")
    logger.info(f"   Individual metrics: {aggregated_metrics}")
    logger.info(f"   Final quality score: {final_quality_score:.4f}")
    
    return final_quality_score

def create_mock_generation_function():
    """モック生成関数の作成（テスト用）"""
    def mock_generate_with_piece(input_text: str, piece: Any) -> str:
        """
        モック生成関数
        実際の実装では、CFS-Chameleonエディターを使用してpiece適用生成を行う
        """
        # 基本的な変換（実際にはLLMを使用）
        if "要約" in input_text:
            return f"要約結果：{input_text[:50]}の要点"
        elif "翻訳" in input_text:
            return f"Translation: {input_text[:30]} translated"
        else:
            return f"生成結果：{input_text[:40]}に対する応答"
    
    return mock_generate_with_piece

def create_sample_evaluation_datasets() -> Dict[str, List[Tuple[str, str]]]:
    """サンプル評価データセットの作成"""
    datasets = {
        "summarization": [
            (
                "今日は天気が良く、公園で友人と散歩を楽しみました。桜の花が咲いていて、とても美しい景色でした。多くの人が花見をしており、賑やかな雰囲気でした。", 
                "友人と公園で桜を見ながら散歩を楽しんだ。"
            ),
            (
                "新しいプロジェクトが始まり、チームメンバーとの会議を行いました。タスクの分担を決め、スケジュールを確認しました。来週までに各自の担当部分を完成させる予定です。",
                "新プロジェクトでチーム会議を行い、タスク分担とスケジュールを決めた。"
            ),
            (
                "最近読んだ本がとても面白く、主人公の成長物語に感動しました。困難を乗り越えて目標を達成する姿に勇気をもらいました。友人にも勧めたいと思います。",
                "面白い本を読み、主人公の成長物語に感動し友人に勧めたい。"
            )
        ],
        
        "qa": [
            (
                "日本の首都はどこですか？",
                "東京"
            ),
            (
                "機械学習とは何ですか？",
                "コンピューターがデータから自動的に学習して予測や判断を行う技術"
            ),
            (
                "健康的な食事のコツを教えてください。",
                "バランスの良い栄養摂取、適量の食事、野菜や果物を多く取ることが重要"
            )
        ],
        
        "dialogue": [
            (
                "今日はいい天気ですね。",
                "そうですね！散歩にはぴったりの日ですね。"
            ),
            (
                "週末は何をしますか？",
                "友人と映画を見に行く予定です。あなたはいかがですか？"
            ),
            (
                "最近忙しくて疲れています。",
                "お疲れ様です。しっかり休息を取ることも大切ですよ。"
            )
        ]
    }
    
    return datasets

def demonstrate_task_based_quality_evaluation():
    """タスクベース品質評価のデモンストレーション"""
    print("🎯 タスクベース品質スコア評価デモ")
    print("=" * 60)
    
    # サンプルデータセットの作成
    datasets = create_sample_evaluation_datasets()
    
    # モック生成関数の作成
    generate_func = create_mock_generation_function()
    
    # 異なる設定での評価
    configs = [
        QualityEvaluationConfig(
            metrics=["rouge", "bleu"],
            metric_weights={"rouge": 0.6, "bleu": 0.4},
            max_eval_samples=3
        ),
        QualityEvaluationConfig(
            metrics=["rouge", "bleu", "bertscore"],
            metric_weights={"rouge": 0.4, "bleu": 0.3, "bertscore": 0.3},
            max_eval_samples=3
        ),
        QualityEvaluationConfig(
            metrics=["bertscore"],
            metric_weights={"bertscore": 1.0},
            max_eval_samples=3
        )
    ]
    
    # 各データセットとコンフィグで評価
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 40)
        
        for i, config in enumerate(configs):
            print(f"\n🔸 Configuration {i+1}: {config.metrics}")
            
            # ダミーピース（実際にはDirectionPieceオブジェクト）
            mock_piece = {"id": f"piece_{i}", "vector": np.random.randn(768)}
            
            try:
                quality_score = calculate_improved_quality_score(
                    piece=mock_piece,
                    eval_dataset=dataset,
                    generate_with_piece=generate_func,
                    config=config
                )
                
                print(f"   Quality Score: {quality_score:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    print("\n🎉 タスクベース品質評価デモ完了!")

def create_real_cfs_quality_evaluator():
    """実際のCFS-Chameleonと統合した品質評価関数"""
    
    def cfs_generate_with_piece(input_text: str, piece: Any) -> str:
        """
        CFS-Chameleonを使用した実際の生成関数
        
        Args:
            input_text: 入力テキスト
            piece: DirectionPiece
            
        Returns:
            生成されたテキスト
        """
        if not CFS_AVAILABLE:
            # フォールバック：モック生成
            return f"Mock generation for: {input_text[:50]}..."
        
        try:
            # CFS-Chameleonエディターの初期化
            editor = CollaborativeChameleonEditor(
                use_collaboration=True,
                config_path="cfs_config.yaml"
            )
            
            # 方向ピースを一時的にプールに追加
            if hasattr(editor, 'direction_pool') and hasattr(editor.direction_pool, 'pieces'):
                editor.direction_pool.pieces.append(piece)
            
            # 生成実行
            result = editor.generate_with_chameleon(
                prompt=input_text,
                alpha_personal=0.1,
                alpha_neutral=-0.05,
                max_length=100
            )
            
            return result
            
        except Exception as e:
            logger.error(f"CFS generation error: {e}")
            return f"Generation error: {input_text[:30]}..."
    
    return cfs_generate_with_piece

if __name__ == "__main__":
    # デモンストレーション実行
    demonstrate_task_based_quality_evaluation()
    
    # 実際のCFS統合例
    print("\n" + "=" * 60)
    print("🦎 CFS-Chameleon統合例")
    
    cfs_generate_func = create_real_cfs_quality_evaluator()
    datasets = create_sample_evaluation_datasets()
    
    config = QualityEvaluationConfig(
        metrics=["rouge", "bleu", "bertscore"],
        max_eval_samples=2
    )
    
    # 要約タスクでの評価例
    mock_piece = {"vector": np.random.randn(3072), "quality": 0.5}
    
    try:
        quality_score = calculate_improved_quality_score(
            piece=mock_piece,
            eval_dataset=datasets["summarization"],
            generate_with_piece=cfs_generate_func,
            config=config
        )
        
        print(f"✅ CFS-Chameleon Quality Score: {quality_score:.4f}")
        
    except Exception as e:
        print(f"❌ CFS evaluation error: {e}")