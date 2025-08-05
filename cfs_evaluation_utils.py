#!/usr/bin/env python3
"""
CFS-Chameleon評価用ユーティリティ
性能指標計算、統計的有意性検定、結果可視化、レポート生成など

特徴:
- 包括的性能指標計算
- 高度な統計分析
- プロフェッショナル可視化
- 自動レポート生成
- 実験結果管理
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, cohen_kappa_score
)
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# カスタムカラーパレット
CFS_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#8B8B8B',
    'background': '#F5F5F5'
}

@dataclass
class DetailedMetrics:
    """詳細評価メトリクス"""
    # 基本メトリクス
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 拡張メトリクス
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    cohen_kappa: float
    
    # 信頼区間
    accuracy_ci: Tuple[float, float]
    f1_ci: Tuple[float, float]
    
    # サンプル情報
    total_samples: int
    correct_predictions: int
    
    # 分布情報
    class_distribution: Dict[str, int]
    confusion_matrix: List[List[int]]

@dataclass
class StatisticalTestResult:
    """統計検定結果"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significance_level: float = 0.05

@dataclass
class ComparisonAnalysis:
    """比較分析結果"""
    method1_name: str
    method2_name: str
    improvement_rate: float
    statistical_tests: List[StatisticalTestResult]
    effect_size_analysis: Dict[str, float]
    practical_significance: bool
    confidence_level: float

class CFSEvaluationMetrics:
    """CFS-Chameleon評価メトリクス計算器"""
    
    def __init__(self):
        self.metrics_history = []
        self.comparison_history = []
    
    def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str], 
                                      method_name: str = "Unknown") -> DetailedMetrics:
        """包括的メトリクス計算"""
        try:
            # 基本メトリクス
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            
            # 拡張メトリクス
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(y_true, y_pred)
            except:
                kappa = 0.0
            
            # 信頼区間計算
            accuracy_ci = self._calculate_confidence_interval(accuracy, len(y_true), 'proportion')
            f1_ci = self._calculate_confidence_interval(f1, len(y_true), 'proportion')
            
            # 分布情報
            unique_labels = list(set(y_true + y_pred))
            class_dist = {label: y_true.count(label) for label in unique_labels}
            
            # 混同行列
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            
            metrics = DetailedMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                macro_f1=macro_f1,
                micro_f1=micro_f1,
                weighted_f1=weighted_f1,
                cohen_kappa=kappa,
                accuracy_ci=accuracy_ci,
                f1_ci=f1_ci,
                total_samples=len(y_true),
                correct_predictions=sum(1 for t, p in zip(y_true, y_pred) if t == p),
                class_distribution=class_dist,
                confusion_matrix=cm.tolist()
            )
            
            # 履歴保存
            self.metrics_history.append({
                'method_name': method_name,
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            # フォールバック値
            return DetailedMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                macro_f1=0.0, micro_f1=0.0, weighted_f1=0.0, cohen_kappa=0.0,
                accuracy_ci=(0.0, 0.0), f1_ci=(0.0, 0.0),
                total_samples=len(y_true), correct_predictions=0,
                class_distribution={}, confusion_matrix=[]
            )
    
    def _calculate_confidence_interval(self, point_estimate: float, n: int, 
                                     dist_type: str = 'proportion', confidence: float = 0.95) -> Tuple[float, float]:
        """信頼区間計算"""
        try:
            alpha = 1 - confidence
            z_score = stats.norm.ppf(1 - alpha/2)
            
            if dist_type == 'proportion':
                # 比率の信頼区間
                p = point_estimate
                se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
                margin = z_score * se
                return (max(0, p - margin), min(1, p + margin))
            else:
                # 平均の信頼区間（簡易版）
                se = np.sqrt(point_estimate * (1 - point_estimate) / n) if n > 0 else 0
                margin = z_score * se
                return (point_estimate - margin, point_estimate + margin)
                
        except Exception as e:
            logger.warning(f"信頼区間計算エラー: {e}")
            return (point_estimate, point_estimate)
    
    def statistical_significance_test(self, results1: DetailedMetrics, results2: DetailedMetrics,
                                    test_type: str = "comprehensive") -> List[StatisticalTestResult]:
        """包括的統計的有意性検定"""
        test_results = []
        
        try:
            # 1. 精度差のz検定
            z_test_result = self._accuracy_z_test(results1, results2)
            test_results.append(z_test_result)
            
            # 2. 比率の差の検定
            proportion_test_result = self._proportion_difference_test(results1, results2)
            test_results.append(proportion_test_result)
            
            # 3. 効果サイズ計算
            effect_size_result = self._calculate_effect_size(results1, results2)
            test_results.append(effect_size_result)
            
            if test_type == "comprehensive":
                # 4. McNemar検定（対応サンプル用）
                mcnemar_result = self._mcnemar_test_approximation(results1, results2)
                test_results.append(mcnemar_result)
                
                # 5. Bootstrap信頼区間
                bootstrap_result = self._bootstrap_confidence_interval(results1, results2)
                test_results.append(bootstrap_result)
            
            return test_results
            
        except Exception as e:
            logger.error(f"統計検定エラー: {e}")
            return []
    
    def _accuracy_z_test(self, results1: DetailedMetrics, results2: DetailedMetrics) -> StatisticalTestResult:
        """精度差のz検定"""
        try:
            p1 = results1.accuracy
            p2 = results2.accuracy
            n1 = results1.total_samples
            n2 = results2.total_samples
            
            # プールされた比率
            p_pooled = (results1.correct_predictions + results2.correct_predictions) / (n1 + n2)
            
            # 標準誤差
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se > 0:
                z_stat = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                # 信頼区間
                diff = p2 - p1
                margin = 1.96 * se
                ci = (diff - margin, diff + margin)
                
                # 解釈
                interpretation = self._interpret_significance(p_value, 0.05)
            else:
                z_stat, p_value = 0.0, 1.0
                ci = (0.0, 0.0)
                interpretation = "計算不可"
            
            return StatisticalTestResult(
                test_name="Accuracy Z-test",
                statistic=z_stat,
                p_value=p_value,
                effect_size=abs(p2 - p1),
                confidence_interval=ci,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"Z検定エラー: {e}")
            return StatisticalTestResult("Accuracy Z-test", 0.0, 1.0, 0.0, (0.0, 0.0), "エラー")
    
    def _proportion_difference_test(self, results1: DetailedMetrics, results2: DetailedMetrics) -> StatisticalTestResult:
        """比率の差の検定"""
        try:
            # F1スコア差の検定
            f1_1 = results1.f1_score
            f1_2 = results2.f1_score
            n1 = results1.total_samples
            n2 = results2.total_samples
            
            # 近似的標準誤差
            se1 = np.sqrt(f1_1 * (1 - f1_1) / n1) if n1 > 0 else 0
            se2 = np.sqrt(f1_2 * (1 - f1_2) / n2) if n2 > 0 else 0
            se_diff = np.sqrt(se1**2 + se2**2)
            
            if se_diff > 0:
                t_stat = (f1_2 - f1_1) / se_diff
                df = min(n1, n2) - 1 if n1 > 1 and n2 > 1 else 1
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                # 信頼区間
                diff = f1_2 - f1_1
                margin = stats.t.ppf(0.975, df) * se_diff
                ci = (diff - margin, diff + margin)
                
                interpretation = self._interpret_significance(p_value, 0.05)
            else:
                t_stat, p_value = 0.0, 1.0
                ci = (0.0, 0.0)
                interpretation = "計算不可"
            
            return StatisticalTestResult(
                test_name="F1 Score Difference Test",
                statistic=t_stat,
                p_value=p_value,
                effect_size=abs(f1_2 - f1_1),
                confidence_interval=ci,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"比率差検定エラー: {e}")
            return StatisticalTestResult("F1 Difference Test", 0.0, 1.0, 0.0, (0.0, 0.0), "エラー")
    
    def _calculate_effect_size(self, results1: DetailedMetrics, results2: DetailedMetrics) -> StatisticalTestResult:
        """効果サイズ計算（Cohen's d）"""
        try:
            # 精度差のCohen's d計算
            p1 = results1.accuracy
            p2 = results2.accuracy
            
            # プールされた標準偏差（二項分布の近似）
            var1 = p1 * (1 - p1)
            var2 = p2 * (1 - p2)
            pooled_std = np.sqrt((var1 + var2) / 2)
            
            if pooled_std > 0:
                cohens_d = (p2 - p1) / pooled_std
                
                # 効果サイズの解釈
                if abs(cohens_d) < 0.2:
                    interpretation = "小さな効果"
                elif abs(cohens_d) < 0.5:
                    interpretation = "中程度の効果"
                elif abs(cohens_d) < 0.8:
                    interpretation = "大きな効果"
                else:
                    interpretation = "非常に大きな効果"
            else:
                cohens_d = 0.0
                interpretation = "効果なし"
            
            return StatisticalTestResult(
                test_name="Cohen's d Effect Size",
                statistic=cohens_d,
                p_value=0.0,  # 効果サイズにはp値なし
                effect_size=abs(cohens_d),
                confidence_interval=(0.0, 0.0),  # 簡略化
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"効果サイズ計算エラー: {e}")
            return StatisticalTestResult("Cohen's d", 0.0, 0.0, 0.0, (0.0, 0.0), "エラー")
    
    def _mcnemar_test_approximation(self, results1: DetailedMetrics, results2: DetailedMetrics) -> StatisticalTestResult:
        """McNemar検定の近似"""
        try:
            # 簡易的なMcNemar検定（正確な実装には対応サンプルが必要）
            n = min(results1.total_samples, results2.total_samples)
            c1 = results1.correct_predictions
            c2 = results2.correct_predictions
            
            # 近似的なb, c計算（不一致の推定）
            b_approx = max(1, int((results1.total_samples - c1 + results2.total_samples - c2) / 4))
            c_approx = max(1, int((results1.total_samples - c1 + results2.total_samples - c2) / 4))
            
            if b_approx + c_approx > 0:
                chi2_stat = (abs(b_approx - c_approx) - 1)**2 / (b_approx + c_approx)
                p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
                interpretation = self._interpret_significance(p_value, 0.05)
            else:
                chi2_stat, p_value = 0.0, 1.0
                interpretation = "計算不可"
            
            return StatisticalTestResult(
                test_name="McNemar Test (Approx)",
                statistic=chi2_stat,
                p_value=p_value,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"McNemar検定エラー: {e}")
            return StatisticalTestResult("McNemar Test", 0.0, 1.0, 0.0, (0.0, 0.0), "エラー")
    
    def _bootstrap_confidence_interval(self, results1: DetailedMetrics, results2: DetailedMetrics, 
                                     n_bootstrap: int = 1000) -> StatisticalTestResult:
        """Bootstrap信頼区間"""
        try:
            # 簡易的なBootstrap（正確な実装には元データが必要）
            diff_samples = []
            
            for _ in range(n_bootstrap):
                # ランダムサンプリングシミュレーション
                sim_acc1 = np.random.beta(results1.correct_predictions + 1, 
                                        results1.total_samples - results1.correct_predictions + 1)
                sim_acc2 = np.random.beta(results2.correct_predictions + 1,
                                        results2.total_samples - results2.correct_predictions + 1)
                diff_samples.append(sim_acc2 - sim_acc1)
            
            # 信頼区間計算
            ci_lower = np.percentile(diff_samples, 2.5)
            ci_upper = np.percentile(diff_samples, 97.5)
            
            # p値近似（0を含むかどうか）
            p_value_approx = 2 * min(np.mean(np.array(diff_samples) <= 0), 
                                   np.mean(np.array(diff_samples) >= 0))
            
            interpretation = self._interpret_significance(p_value_approx, 0.05)
            
            return StatisticalTestResult(
                test_name="Bootstrap CI",
                statistic=np.mean(diff_samples),
                p_value=p_value_approx,
                effect_size=abs(np.mean(diff_samples)),
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"Bootstrap信頼区間エラー: {e}")
            return StatisticalTestResult("Bootstrap CI", 0.0, 1.0, 0.0, (0.0, 0.0), "エラー")
    
    def _interpret_significance(self, p_value: float, alpha: float) -> str:
        """有意性の解釈"""
        if p_value < alpha:
            if p_value < 0.001:
                return "極めて有意 (p < 0.001)"
            elif p_value < 0.01:
                return "高度に有意 (p < 0.01)"
            else:
                return f"有意 (p = {p_value:.3f})"
        elif p_value < alpha * 2:
            return f"有意傾向 (p = {p_value:.3f})"
        else:
            return f"有意差なし (p = {p_value:.3f})"
    
    def generate_comparison_report(self, results1: DetailedMetrics, results2: DetailedMetrics,
                                 method1_name: str, method2_name: str, 
                                 output_path: str) -> ComparisonAnalysis:
        """包括的比較レポート生成"""
        try:
            # 統計検定実行
            statistical_tests = self.statistical_significance_test(results1, results2, "comprehensive")
            
            # 改善率計算
            improvement_rate = ((results2.accuracy - results1.accuracy) / results1.accuracy * 100) if results1.accuracy > 0 else 0.0
            
            # 効果サイズ分析
            effect_sizes = {}
            for test in statistical_tests:
                if "Effect Size" in test.test_name:
                    effect_sizes['cohen_d'] = test.effect_size
                elif "Accuracy" in test.test_name:
                    effect_sizes['accuracy_diff'] = test.effect_size
                elif "F1" in test.test_name:
                    effect_sizes['f1_diff'] = test.effect_size
            
            # 実用的有意性判定
            practical_significance = improvement_rate > 5.0 and any(test.p_value < 0.05 for test in statistical_tests)
            
            comparison_analysis = ComparisonAnalysis(
                method1_name=method1_name,
                method2_name=method2_name,
                improvement_rate=improvement_rate,
                statistical_tests=statistical_tests,
                effect_size_analysis=effect_sizes,
                practical_significance=practical_significance,
                confidence_level=0.95
            )
            
            # レポート生成
            self._write_comparison_report(comparison_analysis, results1, results2, output_path)
            
            # 比較履歴保存
            self.comparison_history.append({
                'timestamp': time.time(),
                'comparison': comparison_analysis
            })
            
            return comparison_analysis
            
        except Exception as e:
            logger.error(f"比較レポート生成エラー: {e}")
            return None
    
    def _write_comparison_report(self, analysis: ComparisonAnalysis, results1: DetailedMetrics, 
                               results2: DetailedMetrics, output_path: str):
        """比較レポート書き込み"""
        report_content = f"""
# CFS-Chameleon性能比較分析レポート

**分析日時**: {time.strftime("%Y年%m月%d日 %H:%M:%S")}  
**比較対象**: {analysis.method1_name} vs {analysis.method2_name}  
**信頼水準**: {analysis.confidence_level*100:.0f}%  

## 📊 性能比較サマリー

| 指標 | {analysis.method1_name} | {analysis.method2_name} | 改善 |
|------|-------------|-------------|------|
| 精度 | {results1.accuracy:.4f} | {results2.accuracy:.4f} | {results2.accuracy - results1.accuracy:+.4f} |
| F1スコア | {results1.f1_score:.4f} | {results2.f1_score:.4f} | {results2.f1_score - results1.f1_score:+.4f} |
| 適合率 | {results1.precision:.4f} | {results2.precision:.4f} | {results2.precision - results1.precision:+.4f} |
| 再現率 | {results1.recall:.4f} | {results2.recall:.4f} | {results2.recall - results1.recall:+.4f} |
| Cohen's κ | {results1.cohen_kappa:.4f} | {results2.cohen_kappa:.4f} | {results2.cohen_kappa - results1.cohen_kappa:+.4f} |

### 🎯 主要成果

- **全体改善率**: {analysis.improvement_rate:+.1f}%
- **実用的有意性**: {'✅ あり' if analysis.practical_significance else '❌ なし'}
- **サンプルサイズ**: {results1.method1_name}: {results1.total_samples}, {analysis.method2_name}: {results2.total_samples}

## 🧪 統計分析結果

"""
        
        for test in analysis.statistical_tests:
            report_content += f"""
### {test.test_name}
- **統計量**: {test.statistic:.4f}
- **p値**: {test.p_value:.6f}
- **効果サイズ**: {test.effect_size:.4f}
- **信頼区間**: [{test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f}]
- **解釈**: {test.interpretation}

"""
        
        report_content += f"""
## 📈 効果サイズ分析

"""
        
        for effect_name, effect_value in analysis.effect_size_analysis.items():
            magnitude = "大" if effect_value > 0.8 else "中" if effect_value > 0.2 else "小"
            report_content += f"- **{effect_name}**: {effect_value:.4f} ({magnitude})\n"
        
        report_content += f"""

## 💡 結論と推奨事項

### 統計的結論
{
    f"{analysis.method2_name}は{analysis.method1_name}と比較して統計的に有意な改善を示している。" 
    if analysis.practical_significance
    else f"{analysis.method2_name}は{analysis.method1_name}と比較して十分な改善を示していない。"
}

### 推奨事項
{
    "この改善は実用的価値があり、本番環境での採用を推奨する。" if analysis.practical_significance
    else "更なる改善が必要。パラメータ調整や手法の見直しを検討すべき。"
}

---

*本レポートはCFS-Chameleon評価ユーティリティにより自動生成されました。*
"""
        
        # ファイル保存
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 比較レポート生成: {output_file}")
    
    def create_performance_visualization(self, results_dict: Dict[str, DetailedMetrics], 
                                       output_path: str, title: str = "Performance Comparison"):
        """性能比較可視化"""
        try:
            # スタイル設定
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            methods = list(results_dict.keys())
            
            # 1. 主要メトリクス比較
            ax1 = axes[0, 0]
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            x = np.arange(len(metric_labels))
            width = 0.8 / len(methods)
            
            for i, method in enumerate(methods):
                values = [getattr(results_dict[method], metric) for metric in metrics]
                ax1.bar(x + i * width, values, width, label=method, alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Primary Metrics Comparison')
            ax1.set_xticks(x + width * (len(methods) - 1) / 2)
            ax1.set_xticklabels(metric_labels)
            ax1.legend()
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # 2. F1スコア詳細比較
            ax2 = axes[0, 1]
            f1_metrics = ['f1_score', 'macro_f1', 'micro_f1', 'weighted_f1']
            f1_labels = ['F1', 'Macro F1', 'Micro F1', 'Weighted F1']
            
            for i, method in enumerate(methods):
                values = [getattr(results_dict[method], metric) for metric in f1_metrics]
                ax2.bar(x + i * width, values, width, label=method, alpha=0.8)
            
            ax2.set_xlabel('F1 Variants')
            ax2.set_ylabel('Score')
            ax2.set_title('F1-Score Variants Comparison')
            ax2.set_xticks(x + width * (len(methods) - 1) / 2)
            ax2.set_xticklabels(f1_labels)
            ax2.legend()
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # 3. サンプル数とCorrect Predictions
            ax3 = axes[0, 2]
            total_samples = [results_dict[method].total_samples for method in methods]
            correct_predictions = [results_dict[method].correct_predictions for method in methods]
            
            x_methods = np.arange(len(methods))
            ax3.bar(x_methods - 0.2, total_samples, 0.4, label='Total Samples', alpha=0.7)
            ax3.bar(x_methods + 0.2, correct_predictions, 0.4, label='Correct', alpha=0.7)
            ax3.set_xlabel('Methods')
            ax3.set_ylabel('Count')
            ax3.set_title('Sample Distribution')
            ax3.set_xticks(x_methods)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 信頼区間表示
            ax4 = axes[1, 0]
            for i, method in enumerate(methods):
                accuracy = results_dict[method].accuracy
                ci = results_dict[method].accuracy_ci
                ax4.errorbar(i, accuracy, yerr=[[accuracy - ci[0]], [ci[1] - accuracy]], 
                           fmt='o', capsize=5, capthick=2, label=method)
            
            ax4.set_xlabel('Methods')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy with 95% Confidence Intervals')
            ax4.set_xticks(range(len(methods)))
            ax4.set_xticklabels(methods)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # 5. Cohen's Kappa比較
            ax5 = axes[1, 1]
            kappa_values = [results_dict[method].cohen_kappa for method in methods]
            colors = ['excellent' if k > 0.8 else 'good' if k > 0.6 else 'fair' if k > 0.4 else 'poor' for k in kappa_values]
            color_map = {'excellent': 'green', 'good': 'blue', 'fair': 'orange', 'poor': 'red'}
            bar_colors = [color_map.get(c, 'gray') for c in colors]
            
            ax5.bar(methods, kappa_values, color=bar_colors, alpha=0.7)
            ax5.set_xlabel('Methods')
            ax5.set_ylabel("Cohen's Kappa")
            ax5.set_title('Inter-rater Agreement (Cohen\'s Kappa)')
            ax5.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Good threshold')
            ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. クラス分布（最初の手法のみ）
            ax6 = axes[1, 2]
            if methods:
                first_method = methods[0]
                class_dist = results_dict[first_method].class_distribution
                if class_dist:
                    classes = list(class_dist.keys())
                    counts = list(class_dist.values())
                    ax6.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
                    ax6.set_title(f'Class Distribution ({first_method})')
                else:
                    ax6.text(0.5, 0.5, 'No class distribution\navailable', 
                           ha='center', va='center', transform=ax6.transAxes)
                    ax6.set_title('Class Distribution')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 性能可視化保存: {output_path}")
            
        except Exception as e:
            logger.error(f"可視化生成エラー: {e}")
    
    def export_results_to_csv(self, results_dict: Dict[str, DetailedMetrics], output_path: str):
        """結果をCSVエクスポート"""
        try:
            data = []
            for method_name, metrics in results_dict.items():
                row = {
                    'Method': method_name,
                    'Accuracy': metrics.accuracy,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1_Score': metrics.f1_score,
                    'Macro_F1': metrics.macro_f1,
                    'Micro_F1': metrics.micro_f1,
                    'Weighted_F1': metrics.weighted_f1,
                    'Cohen_Kappa': metrics.cohen_kappa,
                    'Total_Samples': metrics.total_samples,
                    'Correct_Predictions': metrics.correct_predictions,
                    'Accuracy_CI_Lower': metrics.accuracy_ci[0],
                    'Accuracy_CI_Upper': metrics.accuracy_ci[1],
                    'F1_CI_Lower': metrics.f1_ci[0],
                    'F1_CI_Upper': metrics.f1_ci[1]
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"📁 CSV結果エクスポート: {output_path}")
            
        except Exception as e:
            logger.error(f"CSVエクスポートエラー: {e}")
    
    def save_analysis_state(self, output_path: str):
        """分析状態保存"""
        try:
            state_data = {
                'metrics_history': [
                    {
                        'method_name': entry['method_name'],
                        'timestamp': entry['timestamp'],
                        'metrics': asdict(entry['metrics'])
                    }
                    for entry in self.metrics_history
                ],
                'comparison_history': [
                    {
                        'timestamp': entry['timestamp'],
                        'comparison': asdict(entry['comparison'])
                    }
                    for entry in self.comparison_history
                ],
                'save_timestamp': time.time()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 分析状態保存: {output_path}")
            
        except Exception as e:
            logger.error(f"状態保存エラー: {e}")

def demonstrate_evaluation_utils():
    """評価ユーティリティのデモンストレーション"""
    print("🔧 CFS-Chameleon評価ユーティリティ デモンストレーション")
    print("=" * 60)
    
    # サンプルデータ生成
    np.random.seed(42)
    
    # 模擬評価結果
    y_true = ['action', 'drama', 'comedy', 'horror', 'romance'] * 20
    y_pred_legacy = np.random.choice(['action', 'drama', 'comedy', 'horror', 'romance'], 100)
    y_pred_cfs = np.random.choice(['action', 'drama', 'comedy', 'horror', 'romance'], 100)
    
    # CFS-Chameleonの性能を向上させる（デモ用）
    improvement_indices = np.random.choice(100, 20, replace=False)
    for i in improvement_indices:
        y_pred_cfs[i] = y_true[i]
    
    # 評価実行
    evaluator = CFSEvaluationMetrics()
    
    # メトリクス計算
    legacy_metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred_legacy, "Legacy Chameleon")
    cfs_metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred_cfs, "CFS-Chameleon")
    
    print(f"\n📊 評価結果:")
    print(f"Legacy Chameleon - Accuracy: {legacy_metrics.accuracy:.4f}, F1: {legacy_metrics.f1_score:.4f}")
    print(f"CFS-Chameleon   - Accuracy: {cfs_metrics.accuracy:.4f}, F1: {cfs_metrics.f1_score:.4f}")
    
    # 統計検定
    statistical_tests = evaluator.statistical_significance_test(legacy_metrics, cfs_metrics)
    
    print(f"\n🧪 統計検定結果:")
    for test in statistical_tests:
        print(f"  {test.test_name}: p={test.p_value:.4f}, {test.interpretation}")
    
    # 可視化とレポート生成
    output_dir = Path("./demo_evaluation_output")
    output_dir.mkdir(exist_ok=True)
    
    # 比較レポート生成
    comparison_analysis = evaluator.generate_comparison_report(
        legacy_metrics, cfs_metrics, "Legacy Chameleon", "CFS-Chameleon",
        output_dir / "comparison_report.md"
    )
    
    # 可視化生成
    results_dict = {"Legacy Chameleon": legacy_metrics, "CFS-Chameleon": cfs_metrics}
    evaluator.create_performance_visualization(
        results_dict, output_dir / "performance_comparison.png",
        "CFS-Chameleon vs Legacy Chameleon Performance"
    )
    
    # CSV出力
    evaluator.export_results_to_csv(results_dict, output_dir / "evaluation_results.csv")
    
    # 状態保存
    evaluator.save_analysis_state(output_dir / "analysis_state.json")
    
    print(f"\n✅ デモンストレーション完了!")
    print(f"📁 結果出力先: {output_dir}")

if __name__ == "__main__":
    demonstrate_evaluation_utils()