#!/usr/bin/env python3
"""
LaMP-QA CFS-Chameleon統合ベンチマーク評価システム
質問応答タスクにおける協調的埋め込み編集システムの性能評価・比較フレームワーク

特徴:
- 従来版Chameleon vs CFS-Chameleon性能比較（QAタスク特化）
- QA評価指標: ROUGE-L, BLEU, BERTScore
- コールドスタート性能分析
- 協調学習効果の定量評価
- 統計的有意性検定
- 動的パラメータ調整機能
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# QA評価指標ライブラリ
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    QA_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ QA metrics libraries not available: {e}")
    print("Please install: pip install rouge-score nltk bert-score")
    QA_METRICS_AVAILABLE = False

# CFS-Chameleon統合モジュール
try:
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    from cfs_chameleon_extension import CollaborativeDirectionPool, UserContext
    from chameleon_evaluator import ChameleonEvaluator
    CFS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CFS-Chameleon modules not available: {e}")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class QAEvaluationResult:
    """QA評価結果格納クラス"""
    rouge_l: float
    bleu_score: float
    bert_score_f1: float
    inference_time: float
    cold_start_performance: float
    pool_utilization: float = 0.0
    user_coverage: int = 0
    user_scores: Dict[str, Dict[str, float]] = None

@dataclass
class QAComparisonResults:
    """QA比較評価結果"""
    legacy_chameleon: QAEvaluationResult
    cfs_chameleon: QAEvaluationResult
    improvement_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]

class LaMPQADataLoader:
    """LaMP-QA データローダー"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = "chameleon_prime_personalization/data/raw/LaMP-QA/merged.json"
        self.data_path = Path(data_path)
        self.merged_data = None
        self.ground_truth = None
        
    def load_merged_data(self) -> List[Dict]:
        """LaMP-QA merged.jsonからデータを読み込み"""
        possible_paths = [
            self.data_path,
            Path("chameleon_prime_personalization/data/raw/LaMP-QA/merged.json"),
            Path("data/raw/LaMP-QA/merged.json"),
            Path("LaMP-QA/merged.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading LaMP-QA data from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    self.merged_data = json.load(f)
                return self.merged_data
                
        # フォールバック: LaMP-2データで代用
        logger.warning("LaMP-QA data not found, using LaMP-2 as fallback")
        fallback_paths = [
            "chameleon_prime_personalization/data/raw/LaMP-2/merged.json",
            "data/raw/LaMP-2/merged.json"
        ]
        
        for path in fallback_paths:
            if Path(path).exists():
                logger.info(f"Loading fallback data from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    self.merged_data = json.load(f)
                return self.merged_data
        
        raise FileNotFoundError("Neither LaMP-QA nor LaMP-2 data found")
    
    def load_ground_truth(self) -> Dict[str, str]:
        """正解データ読み込み"""
        qa_answer_paths = [
            "chameleon_prime_personalization/data/raw/LaMP-QA/answers.json",
            "data/raw/LaMP-QA/answers.json"
        ]
        
        for path in qa_answer_paths:
            if Path(path).exists():
                logger.info(f"Loading QA answers from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    answers = json.load(f)
                return {str(item.get('id', item.get('question_id'))): 
                       item.get('answer', item.get('ground_truth', '')) 
                       for item in answers}
        
        # フォールバック
        logger.warning("LaMP-QA answers not found, using LaMP-2 as fallback")
        fallback_paths = [
            "chameleon_prime_personalization/data/raw/LaMP-2/answers.json",
            "data/raw/LaMP-2/answers.json"
        ]
        
        for path in fallback_paths:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'golds' in data:
                        # LaMP-2フォーマット: {"task": "LaMP_2", "golds": [...]}
                        answers = data['golds']
                        return {str(item.get('id')): item.get('output', item.get('answer', '')) 
                                for item in answers}
                    elif isinstance(data, list):
                        # 直接リストフォーマット
                        return {str(item.get('id')): item.get('answer', item.get('output', '')) 
                                for item in data}
                    else:
                        return {}
        
        return {}

class QAMetricsCalculator:
    """QA評価指標計算クラス"""
    
    def __init__(self):
        if QA_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
        
    def calculate_rouge_l(self, prediction: str, reference: str) -> float:
        """ROUGE-L計算"""
        if not QA_METRICS_AVAILABLE:
            return 0.0
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """BLEU計算"""
        if not QA_METRICS_AVAILABLE:
            return 0.0
        
        try:
            pred_tokens = prediction.lower().split()
            ref_tokens = [reference.lower().split()]
            return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> float:
        """BERTScore計算（バッチ処理）"""
        if not QA_METRICS_AVAILABLE or not predictions or not references:
            return 0.0
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return F1.mean().item()
        except:
            return 0.0

class LaMPQACFSBenchmark:
    """LaMP-QA CFS-Chameleonベンチマーク評価システム"""
    
    def __init__(self, use_collaboration: bool = True, config_path: str = None,
                 hook_layer: str = None, alpha_p: float = None, alpha_g: float = None,
                 rank_reduction: int = None):
        self.use_collaboration = use_collaboration
        self.config_path = config_path or "cfs_config.yaml"
        
        # 動的パラメータ設定
        self.dynamic_params = {
            'hook_layer': hook_layer,
            'alpha_p': alpha_p, 
            'alpha_g': alpha_g,
            'rank_reduction': rank_reduction
        }
        
        # 統計情報初期化
        self.evaluation_stats = {
            'total_users': 0,
            'cold_start_users': 0,
            'warm_start_users': 0,
            'avg_user_history_length': 0.0
        }
        
        # データとエディター初期化
        self.data_loader = LaMPQADataLoader()
        self.qa_calculator = QAMetricsCalculator()
        self.test_data = self.data_loader.load_merged_data()
        self.ground_truth = self.data_loader.load_ground_truth()
        
        # エディター初期化
        self._initialize_editors()
        
        # 結果保存ディレクトリ
        self.output_dir = Path("lampqa_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ LaMP-QA CFS評価システム初期化完了")
        logger.info(f"   協調機能: {'有効' if use_collaboration else '無効'}")
        logger.info(f"   テストサンプル数: {len(self.test_data)}")
        logger.info(f"   出力ディレクトリ: {self.output_dir}")

    def _initialize_editors(self):
        """エディター初期化"""
        if not CFS_AVAILABLE:
            logger.error("CFS-Chameleonモジュールが利用できません")
            self.cfs_editor = None
            self.legacy_editor = None
            return
        
        try:
            # 協調エディター
            if self.use_collaboration:
                collab_config = self._load_collaboration_config()
                self.cfs_editor = CollaborativeChameleonEditor(
                    use_collaboration=True,
                    collaboration_config=collab_config,
                    config_path=self.config_path
                )
                # 動的パラメータ適用
                self._apply_dynamic_params(self.cfs_editor)
                logger.info("✅ CFS-Chameleon協調エディター初期化完了")
            else:
                self.cfs_editor = None
            
            # レガシーエディター（比較用）
            self.legacy_editor = CollaborativeChameleonEditor(
                use_collaboration=False,
                config_path=self.config_path
            )
            # レガシーエディターにも動的パラメータ適用
            self._apply_dynamic_params(self.legacy_editor)
            logger.info("✅ レガシーChameleonエディター初期化完了")
            
            # Theta vectors読み込み
            self._load_theta_vectors()
            
        except Exception as e:
            logger.error(f"エディター初期化エラー: {e}")
            self.cfs_editor = None
            self.legacy_editor = None

    def _apply_dynamic_params(self, editor):
        """動的パラメータをエディターに適用"""
        if self.dynamic_params['hook_layer']:
            editor._config_target_layers = [self.dynamic_params['hook_layer']]
        if self.dynamic_params['alpha_p'] is not None:
            editor._config_alpha_personal = self.dynamic_params['alpha_p']
        if self.dynamic_params['alpha_g'] is not None:
            editor._config_alpha_general = self.dynamic_params['alpha_g']

    def _load_collaboration_config(self) -> Dict[str, Any]:
        """協調設定読み込み"""
        try:
            import yaml
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    collaboration_config = config.get('collaboration', {})
                    
                    # 動的rank_reduction適用
                    if self.dynamic_params['rank_reduction']:
                        collaboration_config['rank_reduction'] = self.dynamic_params['rank_reduction']
                    
                    return collaboration_config
        except Exception as e:
            logger.warning(f"協調設定読み込みエラー: {e}")
        
        return {
            'pool_size': 1000,
            'rank_reduction': self.dynamic_params.get('rank_reduction', 32),
            'privacy_noise_std': 0.01,
            'enable_learning': False
        }

    def _load_theta_vectors(self):
        """Theta vectors読み込み"""
        # CollaborativeChameleonEditorでは初期化時に自動読み込みされるため、
        # ここでは読み込み状況のチェックのみ実行
        logger.info("✅ Theta vectors reading handled by CollaborativeChameleonEditor")
        
        # 読み込み状況確認
        if hasattr(self.legacy_editor, 'theta_personal') and self.legacy_editor.theta_personal is not None:
            logger.info(f"Legacy editor theta vectors: {self.legacy_editor.theta_personal.shape}")
        else:
            logger.warning("Legacy editor theta vectors not loaded")
            
        if self.cfs_editor and hasattr(self.cfs_editor, 'theta_personal') and self.cfs_editor.theta_personal is not None:
            logger.info(f"CFS editor theta vectors: {self.cfs_editor.theta_personal.shape}")
        else:
            logger.warning("CFS editor theta vectors not loaded")

    def evaluate_qa_performance(self, editor, samples: List[Dict], 
                               system_name: str) -> QAEvaluationResult:
        """QA性能評価"""
        predictions = []
        references = []
        user_scores = {}
        inference_times = []
        
        logger.info(f"🔄 {system_name}評価開始")
        
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"   進捗: {i}/{len(samples)}")
            
            try:
                user_id = sample.get('user_id', 'unknown')
                question = sample.get('input', sample.get('question', ''))
                
                start_time = time.time()
                
                # 協調的生成またはレガシー生成
                if hasattr(editor, 'generate_with_chameleon'):
                    answer = editor.generate_with_chameleon(question, max_length=100)
                else:
                    answer = question  # フォールバック
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 正解取得
                sample_id = str(sample.get('id', i))
                reference = self.ground_truth.get(sample_id, '')
                
                predictions.append(answer)
                references.append(reference)
                
                # ユーザー別スコア計算
                if user_id not in user_scores:
                    user_scores[user_id] = {'rouge_l': [], 'bleu': []}
                
                rouge_l = self.qa_calculator.calculate_rouge_l(answer, reference)
                bleu = self.qa_calculator.calculate_bleu(answer, reference)
                
                user_scores[user_id]['rouge_l'].append(rouge_l)
                user_scores[user_id]['bleu'].append(bleu)
                
            except Exception as e:
                logger.warning(f"サンプル{i}評価エラー: {e}")
                predictions.append("")
                references.append("")
                inference_times.append(0.0)
        
        # 全体指標計算
        total_rouge_l = np.mean([self.qa_calculator.calculate_rouge_l(p, r) 
                                for p, r in zip(predictions, references)])
        total_bleu = np.mean([self.qa_calculator.calculate_bleu(p, r) 
                             for p, r in zip(predictions, references)])
        bert_f1 = self.qa_calculator.calculate_bert_score(predictions, references)
        
        # コールドスタート性能計算
        cold_start_performance = self._calculate_cold_start_performance(user_scores)
        
        # プール利用率（CFS-Chameleonの場合）
        pool_utilization = 0.0
        if hasattr(editor, 'direction_pool') and editor.direction_pool:
            pool_utilization = len(editor.direction_pool.pieces) / editor.direction_pool.pool_size
        
        # ユーザー別平均スコア計算
        user_avg_scores = {}
        for user_id, scores in user_scores.items():
            user_avg_scores[user_id] = {
                'rouge_l': np.mean(scores['rouge_l']) if scores['rouge_l'] else 0.0,
                'bleu': np.mean(scores['bleu']) if scores['bleu'] else 0.0
            }
        
        return QAEvaluationResult(
            rouge_l=total_rouge_l,
            bleu_score=total_bleu,
            bert_score_f1=bert_f1,
            inference_time=sum(inference_times),
            cold_start_performance=cold_start_performance,
            pool_utilization=pool_utilization,
            user_coverage=len(user_scores),
            user_scores=user_avg_scores
        )

    def _calculate_cold_start_performance(self, user_scores: Dict) -> float:
        """コールドスタートユーザーの性能計算"""
        cold_start_scores = []
        for user_id, scores in user_scores.items():
            if len(scores['rouge_l']) <= 3:  # 履歴3件以下をコールドスタートとする
                cold_start_scores.extend(scores['rouge_l'])
        
        return np.mean(cold_start_scores) if cold_start_scores else 0.0

    def run_comparison_evaluation(self) -> QAComparisonResults:
        """比較評価実行"""
        logger.info("🚀 LaMP-QA CFS-Chameleon比較評価開始")
        logger.info("=" * 60)
        
        # テストデータ準備（最大100サンプル）
        test_samples = self.test_data[:100]
        
        # レガシーChameleon評価
        legacy_result = self.evaluate_qa_performance(
            self.legacy_editor, test_samples, "従来版Chameleon"
        )
        
        # CFS-Chameleon評価（協調機能有効時）
        if self.use_collaboration and self.cfs_editor:
            cfs_result = self.evaluate_qa_performance(
                self.cfs_editor, test_samples, "CFS-Chameleon"
            )
        else:
            cfs_result = legacy_result  # 協調機能無効時は同じ結果
        
        # 改善指標計算
        improvement_metrics = {
            'rouge_l_improvement': ((cfs_result.rouge_l - legacy_result.rouge_l) / 
                                   legacy_result.rouge_l * 100) if legacy_result.rouge_l > 0 else 0.0,
            'bleu_improvement': ((cfs_result.bleu_score - legacy_result.bleu_score) / 
                                legacy_result.bleu_score * 100) if legacy_result.bleu_score > 0 else 0.0,
            'bert_improvement': ((cfs_result.bert_score_f1 - legacy_result.bert_score_f1) / 
                                legacy_result.bert_score_f1 * 100) if legacy_result.bert_score_f1 > 0 else 0.0,
            'speed_improvement': ((legacy_result.inference_time - cfs_result.inference_time) / 
                                 legacy_result.inference_time * 100) if legacy_result.inference_time > 0 else 0.0,
        }
        
        # 統計的有意性検定
        statistical_significance = self._calculate_statistical_significance(
            legacy_result, cfs_result
        )
        
        # 結果保存
        results = QAComparisonResults(
            legacy_chameleon=legacy_result,
            cfs_chameleon=cfs_result,
            improvement_metrics=improvement_metrics,
            statistical_significance=statistical_significance
        )
        
        self._save_results(results)
        self._display_results(results)
        
        return results

    def _calculate_statistical_significance(self, legacy: QAEvaluationResult, 
                                          cfs: QAEvaluationResult) -> Dict[str, float]:
        """統計的有意性計算"""
        try:
            # ユーザー別スコアで比較（サンプルサイズが小さい場合のフォールバック）
            legacy_scores = [scores['rouge_l'] for scores in legacy.user_scores.values()]
            cfs_scores = [scores['rouge_l'] for scores in cfs.user_scores.values()]
            
            if len(legacy_scores) > 1 and len(cfs_scores) > 1:
                t_stat, p_value = stats.ttest_rel(cfs_scores, legacy_scores)
            else:
                p_value = 1.0
                
            return {
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        except:
            return {'p_value': 1.0, 'is_significant': False}

    def _save_results(self, results: QAComparisonResults):
        """結果保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON保存
        result_dict = {
            'legacy_chameleon': {
                'rouge_l': results.legacy_chameleon.rouge_l,
                'bleu_score': results.legacy_chameleon.bleu_score,
                'bert_score_f1': results.legacy_chameleon.bert_score_f1,
                'inference_time': results.legacy_chameleon.inference_time,
                'cold_start_performance': results.legacy_chameleon.cold_start_performance
            },
            'cfs_chameleon': {
                'rouge_l': results.cfs_chameleon.rouge_l,
                'bleu_score': results.cfs_chameleon.bleu_score,
                'bert_score_f1': results.cfs_chameleon.bert_score_f1,
                'inference_time': results.cfs_chameleon.inference_time,
                'cold_start_performance': results.cfs_chameleon.cold_start_performance,
                'pool_utilization': results.cfs_chameleon.pool_utilization,
                'user_coverage': results.cfs_chameleon.user_coverage
            },
            'improvement_metrics': results.improvement_metrics,
            'statistical_significance': results.statistical_significance,
            'dynamic_params': self.dynamic_params
        }
        
        output_file = self.output_dir / f"lampqa_comparison_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"📁 結果保存: {output_file}")

    def _display_results(self, results: QAComparisonResults):
        """結果表示"""
        print("\n" + "=" * 60)
        print("📊 LaMP-QA CFS-Chameleon比較評価結果")
        print("=" * 60)
        
        # レガシー結果
        print(f"\n🔸 従来版Chameleon:")
        print(f"   ROUGE-L:      {results.legacy_chameleon.rouge_l:.4f}")
        print(f"   BLEU Score:   {results.legacy_chameleon.bleu_score:.4f}")
        print(f"   BERTScore:    {results.legacy_chameleon.bert_score_f1:.4f}")
        print(f"   推論時間:     {results.legacy_chameleon.inference_time:.2f}秒")
        print(f"   コールドスタート: {results.legacy_chameleon.cold_start_performance:.4f}")
        
        # CFS-Chameleon結果
        print(f"\n🦎 CFS-Chameleon:")
        print(f"   ROUGE-L:      {results.cfs_chameleon.rouge_l:.4f}")
        print(f"   BLEU Score:   {results.cfs_chameleon.bleu_score:.4f}")
        print(f"   BERTScore:    {results.cfs_chameleon.bert_score_f1:.4f}")
        print(f"   推論時間:     {results.cfs_chameleon.inference_time:.2f}秒")
        print(f"   コールドスタート: {results.cfs_chameleon.cold_start_performance:.4f}")
        print(f"   プール利用率: {results.cfs_chameleon.pool_utilization:.2%}")
        print(f"   ユーザー範囲: {results.cfs_chameleon.user_coverage}人")
        
        # 改善効果
        print(f"\n📈 改善効果:")
        print(f"   ROUGE-L改善:  {results.improvement_metrics['rouge_l_improvement']:+.1f}%")
        print(f"   BLEU改善:     {results.improvement_metrics['bleu_improvement']:+.1f}%")
        print(f"   BERTScore改善: {results.improvement_metrics['bert_improvement']:+.1f}%")
        print(f"   速度改善:     {results.improvement_metrics['speed_improvement']:+.1f}%")
        print(f"   統計的有意性: p = {results.statistical_significance['p_value']:.4f}")
        
        is_significant = results.statistical_significance['is_significant']
        print(f"   {'✅ 統計的に有意な改善が確認されました' if is_significant else '⚠️ 統計的有意性は検出されませんでした'}")
        
        # 動的パラメータ表示
        if any(v is not None for v in self.dynamic_params.values()):
            print(f"\n⚙️ 使用パラメータ:")
            for key, value in self.dynamic_params.items():
                if value is not None:
                    print(f"   {key}: {value}")
        
        print(f"\n✅ LaMP-QA CFS-Chameleon比較評価完了!")
        print(f"📁 結果保存先: {self.output_dir}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="LaMP-QA CFS-Chameleon Benchmark")
    parser.add_argument("--compare_modes", action="store_true", 
                       help="Compare legacy vs CFS-Chameleon")
    parser.add_argument("--use_collaboration", action="store_true",
                       help="Enable collaborative features")
    parser.add_argument("--config", default="cfs_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--hook_layer", type=str,
                       help="Target hook layer (e.g., model.layers.16.mlp)")
    parser.add_argument("--alpha_p", type=float,
                       help="Alpha personal parameter")
    parser.add_argument("--alpha_g", type=float,
                       help="Alpha general parameter")
    parser.add_argument("--rank_reduction", type=int,
                       help="Rank reduction for collaboration pool")
    
    args = parser.parse_args()
    
    if not QA_METRICS_AVAILABLE:
        logger.error("QA evaluation metrics not available. Please install required libraries.")
        return
    
    if not CFS_AVAILABLE:
        logger.error("CFS-Chameleon modules not available.")
        return
    
    # ベンチマーク実行
    benchmark = LaMPQACFSBenchmark(
        use_collaboration=args.use_collaboration,
        config_path=args.config,
        hook_layer=args.hook_layer,
        alpha_p=args.alpha_p,
        alpha_g=args.alpha_g,
        rank_reduction=args.rank_reduction
    )
    
    if args.compare_modes:
        results = benchmark.run_comparison_evaluation()
        
        # 簡単な影響分析コメント
        print(f"\n💡 パラメータ影響分析:")
        if args.hook_layer:
            print(f"   Hook層 {args.hook_layer}: {'性能向上' if any(v > 0 for v in results.improvement_metrics.values()) else '性能影響限定的'}")
        if args.alpha_p:
            print(f"   α_personal {args.alpha_p}: {'個性強化効果' if results.improvement_metrics['rouge_l_improvement'] > 0 else '過剰編集の可能性'}")
        if args.rank_reduction:
            print(f"   Rank reduction {args.rank_reduction}: {'協調精度向上' if results.cfs_chameleon.pool_utilization > 0.3 else '協調効果限定的'}")
        
        print(f"   総合評価: {'🏆 パラメータ最適化成功' if results.statistical_significance['is_significant'] else '🔧 更なる調整が必要'}")
    
    logger.info("🎉 すべての評価が完了しました!")

def test_theta_vector_loading():
    """Theta vector読み込みテスト関数"""
    print("\n🧪 Theta Vector読み込みテスト開始")
    print("=" * 50)
    
    try:
        # 設定ファイルから読み込み
        config_path = "cfs_config.yaml"
        editor = CollaborativeChameleonEditor(
            use_collaboration=False,
            config_path=config_path
        )
        
        # テスト実行
        print(f"📁 Config path: {config_path}")
        print(f"🔗 Theta P path: {getattr(editor, 'theta_p_path', 'Not set')}")
        print(f"🔗 Theta N path: {getattr(editor, 'theta_n_path', 'Not set')}")
        
        # Assertion checks
        assert hasattr(editor, "theta_personal"), "theta_personal attribute not found"
        assert hasattr(editor, "theta_neutral"), "theta_neutral attribute not found"
        
        if editor.theta_personal is not None and editor.theta_neutral is not None:
            print(f"✅ Theta vectors loaded successfully:")
            print(f"   Personal shape: {editor.theta_personal.shape}")
            print(f"   Neutral shape:  {editor.theta_neutral.shape}")
            return True
        else:
            print("❌ Theta vectors are None")
            return False
            
    except Exception as e:
        print(f"❌ Theta vector loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direction_vector_loading():
    """Direction vector読み込みテスト関数"""
    print("\n🧪 Direction Vector読み込みテスト開始")
    print("=" * 50)
    
    try:
        # 設定ファイルから読み込み
        config_path = "cfs_config.yaml"
        editor = CollaborativeChameleonEditor(
            use_collaboration=False,
            config_path=config_path
        )
        
        # テスト実行
        print(f"📁 Config path: {config_path}")
        print(f"🔗 Direction P path: {getattr(editor, 'direction_p_path', 'Not set')}")
        print(f"🔗 Direction N path: {getattr(editor, 'direction_n_path', 'Not set')}")
        
        # Assertion checks
        assert hasattr(editor, "direction_personal"), "direction_personal attribute not found"
        assert hasattr(editor, "direction_neutral"), "direction_neutral attribute not found"
        
        if editor.direction_personal is not None and editor.direction_neutral is not None:
            print(f"✅ Direction vectors loaded successfully:")
            print(f"   Personal shape: {editor.direction_personal.shape}")
            print(f"   Neutral shape:  {editor.direction_neutral.shape}")
            print(f"   Direction vectors ready for Chameleon editing!")
            return True
        else:
            print("❌ Direction vectors are None")
            return False
            
    except Exception as e:
        print(f"❌ Direction vector loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Theta vector読み込みテスト実行
    theta_test_success = test_theta_vector_loading()
    
    # Direction vector読み込みテスト実行
    direction_test_success = test_direction_vector_loading()
    
    if theta_test_success and direction_test_success:
        print("\n🚀 All vector loading tests passed, proceeding with main evaluation...")
        main()
    else:
        print("\n⚠️ Vector loading tests failed, please check configuration")
        if not theta_test_success:
            print("   ❌ Theta vector test failed")
        if not direction_test_success:
            print("   ❌ Direction vector test failed")
        exit(1)