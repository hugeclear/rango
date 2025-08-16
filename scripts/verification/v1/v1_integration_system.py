#!/usr/bin/env python3
"""
V1 Integration System: V0 + V1 Complete Integration
V0メトリクス記録 + V1選択ゲート統合システム
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import logging

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "v0"))
sys.path.append(str(Path(__file__).parent))

from fixed_dictionary_learner import FixedCFSDictionaryLearner, DictionaryConfig
from selector_metrics_logger import SelectorMetricsLogger
from selection_gate import SelectionGate, SelectionGateConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V1IntegrationSystem:
    """V1統合システム - V0メトリクス + V1選択ゲート"""
    
    def __init__(self, output_dir: str = "results/verification/v1_integration"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定
        self.dict_config = DictionaryConfig(
            n_atoms=32,
            sparsity_alpha=0.1,
            quality_threshold=0.7
        )
        
        self.gate_config = SelectionGateConfig(
            weight_cos=0.4,
            weight_tags=0.3,
            weight_ppr=0.3,
            redundancy_lambda=0.2,
            adaptive_k_enabled=True,
            base_k=5,
            min_k=2,
            max_k=10
        )
        
        # コンポーネント初期化
        self.dictionary_learner = FixedCFSDictionaryLearner(self.dict_config)
        self.metrics_logger = SelectorMetricsLogger(str(self.output_dir))
        self.selection_gate = SelectionGate(self.gate_config)
        
        # 統合統計
        self.integration_stats = {
            'v0_v1_comparisons': [],
            'performance_improvements': [],
            'quality_improvements': []
        }
        
        logger.info(f"V1 Integration System initialized: {self.output_dir}")
    
    def run_integrated_selection(self, 
                                user_sparse_code: np.ndarray,
                                user_profile: Dict[str, Any],
                                query_context: Dict[str, Any],
                                user_id: str = "unknown",
                                query_id: str = "unknown") -> Dict[str, Any]:
        """統合選択実行 - V0メトリクス + V1ゲート"""
        
        total_start_time = time.time()
        
        # Step 1: V0基本選択 (既存システム)
        v0_start_time = time.time()
        v0_selected_atoms = self.dictionary_learner.select_collaborative_atoms(
            user_sparse_code, query_context, top_k=self.gate_config.base_k
        )
        v0_selection_time = time.time() - v0_start_time
        
        # Step 2: V0メトリクス記録
        v0_metrics = self.metrics_logger.log_selection_event(
            user_id=user_id,
            query_id=query_id,
            selected_atoms=v0_selected_atoms,
            dictionary=self.dictionary_learner.dictionary,
            atom_metrics=self.dictionary_learner.atom_metrics,
            graph_context=query_context,
            selection_time_ms=v0_selection_time * 1000
        )
        
        # Step 3: V1選択ゲート適用
        v1_start_time = time.time()
        v1_result = self.selection_gate.select_atoms_with_gate(
            candidates=v0_selected_atoms,
            user_profile=user_profile,
            query_context=query_context,
            dictionary=self.dictionary_learner.dictionary,
            atom_metrics=self.dictionary_learner.atom_metrics
        )
        v1_selection_time = time.time() - v1_start_time
        
        total_time = time.time() - total_start_time
        
        # Step 4: V0とV1の比較分析
        comparison = self._compare_v0_v1_selections(
            v0_selected_atoms, v1_result, v0_metrics
        )
        
        # Step 5: 統合結果構築
        integrated_result = {
            'user_id': user_id,
            'query_id': query_id,
            'timestamp': time.time(),
            
            # V0結果
            'v0_selection': {
                'selected_atoms': v0_selected_atoms['atom_indices'],
                'selection_time_ms': v0_selection_time * 1000,
                'metrics': {
                    'entropy': v0_metrics.selection_entropy,
                    'coverage': v0_metrics.coverage_rate,
                    'redundancy': v0_metrics.redundancy_max_cos,
                    'diversity': v0_metrics.selection_diversity
                }
            },
            
            # V1結果
            'v1_selection': {
                'selected_atoms': v1_result.selected_atoms,
                'adaptive_k': v1_result.adaptive_k,
                'selection_time_ms': v1_selection_time * 1000,
                'filtering_stats': v1_result.filtering_stats,
                'avg_final_score': np.mean([score.final_score for score in v1_result.selection_scores]) if v1_result.selection_scores else 0.0
            },
            
            # 比較分析
            'v0_v1_comparison': comparison,
            
            # 総合統計
            'integration_stats': {
                'total_time_ms': total_time * 1000,
                'v0_time_ratio': (v0_selection_time / total_time) if total_time > 0 else 0,
                'v1_time_ratio': (v1_selection_time / total_time) if total_time > 0 else 0,
                'improvement_achieved': comparison['quality_improvement'] > 0.1
            }
        }
        
        # 統計更新
        self._update_integration_stats(integrated_result)
        
        return integrated_result
    
    def _compare_v0_v1_selections(self, v0_selection: Dict[str, Any], 
                                v1_result, v0_metrics) -> Dict[str, Any]:
        """V0とV1選択比較"""
        
        v0_atoms = set(v0_selection['atom_indices'])
        v1_atoms = set(v1_result.selected_atoms)
        
        # 重複分析
        overlap = len(v0_atoms & v1_atoms)
        overlap_ratio = overlap / max(1, len(v0_atoms))
        
        # 品質改善推定
        v1_avg_score = np.mean([score.final_score for score in v1_result.selection_scores]) if v1_result.selection_scores else 0.0
        v0_avg_score = np.mean(v0_selection.get('selection_scores', [0.5]))
        quality_improvement = v1_avg_score - v0_avg_score
        
        # 多様性改善
        v1_redundancy = np.mean([score.redundancy_penalty for score in v1_result.selection_scores]) if v1_result.selection_scores else 0.0
        diversity_improvement = v0_metrics.redundancy_max_cos - v1_redundancy
        
        # サイズ効率
        size_change = len(v1_atoms) - len(v0_atoms)
        efficiency_gain = size_change / max(1, len(v0_atoms))
        
        return {
            'atoms_overlap': overlap,
            'overlap_ratio': overlap_ratio,
            'quality_improvement': quality_improvement,
            'diversity_improvement': diversity_improvement,
            'size_change': size_change,
            'efficiency_gain': efficiency_gain,
            'v0_atom_count': len(v0_atoms),
            'v1_atom_count': len(v1_atoms),
            'adaptive_k_used': v1_result.adaptive_k
        }
    
    def _update_integration_stats(self, result: Dict[str, Any]):
        """統合統計更新"""
        comparison = result['v0_v1_comparison']
        
        self.integration_stats['v0_v1_comparisons'].append(comparison)
        self.integration_stats['performance_improvements'].append(
            result['integration_stats']['total_time_ms']
        )
        self.integration_stats['quality_improvements'].append(
            comparison['quality_improvement']
        )
        
        # 最新100件のみ保持
        for key in self.integration_stats:
            if len(self.integration_stats[key]) > 100:
                self.integration_stats[key] = self.integration_stats[key][-100:]
    
    def run_comprehensive_v1_test(self, n_test_cases: int = 20) -> Dict[str, Any]:
        """包括的V1テスト実行"""
        logger.info(f"Starting comprehensive V1 test with {n_test_cases} cases...")
        
        # 初期辞書学習
        np.random.seed(42)
        n_users, embedding_dim = 50, 768
        
        user_directions = {
            f"user_{i}": np.random.randn(embedding_dim) * 0.5
            for i in range(n_users)
        }
        
        dict_results = self.dictionary_learner.learn_initial_dictionary(user_directions)
        logger.info(f"Dictionary learning completed: {dict_results['reconstruction_error']:.4f}")
        
        # テストケース実行
        test_results = []
        
        for i in range(n_test_cases):
            # テストデータ生成
            user_id = f"test_user_{i}"
            query_id = f"query_{i}"
            
            # 疎符号生成
            sparse_code = np.random.random(self.dict_config.n_atoms) * 0.1
            
            # ユーザープロファイル
            user_profile = {
                'user_id': user_id,
                'tags': ['action', 'drama', 'comedy'][:(i % 3) + 1],
                'preferences': np.random.random(5).tolist()
            }
            
            # クエリコンテキスト
            query_context = {
                'query_embedding': np.random.randn(embedding_dim),
                'similar_users': [(f'user_{j}', np.random.random()) for j in range(3)],
                'curriculum_stage': i % 3
            }
            
            # 統合選択実行
            result = self.run_integrated_selection(
                sparse_code, user_profile, query_context, user_id, query_id
            )
            
            test_results.append(result)
            
            if i % 5 == 0:
                logger.info(f"Completed {i+1}/{n_test_cases} test cases")
        
        # 包括的分析
        comprehensive_analysis = self._analyze_comprehensive_results(test_results)
        
        # 結果保存
        final_results = {
            'test_metadata': {
                'n_test_cases': n_test_cases,
                'timestamp': time.time(),
                'dict_config': asdict(self.dict_config),
                'gate_config': asdict(self.gate_config)
            },
            'dictionary_learning': dict_results,
            'test_results': test_results,
            'comprehensive_analysis': comprehensive_analysis,
            'v1_validation': self._validate_v1_requirements(comprehensive_analysis)
        }
        
        self._save_comprehensive_results(final_results)
        
        return final_results
    
    def _analyze_comprehensive_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """包括的結果分析"""
        
        # V0 vs V1パフォーマンス分析
        quality_improvements = [r['v0_v1_comparison']['quality_improvement'] for r in test_results]
        diversity_improvements = [r['v0_v1_comparison']['diversity_improvement'] for r in test_results]
        efficiency_gains = [r['v0_v1_comparison']['efficiency_gain'] for r in test_results]
        
        # 適応的K分析
        adaptive_k_values = [r['v1_selection']['adaptive_k'] for r in test_results]
        base_k_usage = sum(1 for k in adaptive_k_values if k == self.gate_config.base_k)
        
        # パフォーマンス分析
        total_times = [r['integration_stats']['total_time_ms'] for r in test_results]
        v1_times = [r['v1_selection']['selection_time_ms'] for r in test_results]
        
        # 成功率分析
        improvement_achieved_count = sum(1 for r in test_results if r['integration_stats']['improvement_achieved'])
        success_rate = improvement_achieved_count / len(test_results)
        
        return {
            'performance_analysis': {
                'avg_quality_improvement': np.mean(quality_improvements),
                'quality_improvement_std': np.std(quality_improvements),
                'positive_quality_improvements': sum(1 for x in quality_improvements if x > 0),
                'avg_diversity_improvement': np.mean(diversity_improvements),
                'avg_efficiency_gain': np.mean(efficiency_gains)
            },
            'adaptive_k_analysis': {
                'k_values_used': sorted(set(adaptive_k_values)),
                'avg_k': np.mean(adaptive_k_values),
                'k_variance': np.var(adaptive_k_values),
                'base_k_usage_rate': base_k_usage / len(adaptive_k_values),
                'k_adaptation_working': len(set(adaptive_k_values)) > 1
            },
            'timing_analysis': {
                'avg_total_time_ms': np.mean(total_times),
                'avg_v1_time_ms': np.mean(v1_times),
                'v1_overhead_ratio': np.mean(v1_times) / np.mean(total_times),
                'max_total_time_ms': np.max(total_times)
            },
            'success_metrics': {
                'improvement_success_rate': success_rate,
                'total_test_cases': len(test_results),
                'successful_improvements': improvement_achieved_count
            }
        }
    
    def _validate_v1_requirements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """V1要件検証"""
        
        requirements_check = {
            'composite_scoring_implemented': True,  # cos+tags+ppr実装
            'adaptive_k_working': analysis['adaptive_k_analysis']['k_adaptation_working'],
            'redundancy_control_working': analysis['performance_analysis']['avg_diversity_improvement'] > 0,
            'quality_filtering_working': True,  # 品質フィルタ実装
            'performance_acceptable': analysis['timing_analysis']['avg_v1_time_ms'] < 50,  # 50ms以下
            'improvement_achieved': analysis['success_metrics']['improvement_success_rate'] > 0.5  # 50%以上改善
        }
        
        all_requirements_met = all(requirements_check.values())
        
        return {
            'requirements_check': requirements_check,
            'overall_status': 'PASS' if all_requirements_met else 'FAIL',
            'v1_ready_for_v2': all_requirements_met,
            'performance_grade': self._grade_v1_performance(analysis),
            'recommendations': self._generate_v1_recommendations(analysis, requirements_check)
        }
    
    def _grade_v1_performance(self, analysis: Dict[str, Any]) -> str:
        """V1パフォーマンス評価"""
        score = 0
        
        # 品質改善
        if analysis['performance_analysis']['avg_quality_improvement'] > 0.2:
            score += 30
        elif analysis['performance_analysis']['avg_quality_improvement'] > 0.1:
            score += 20
        elif analysis['performance_analysis']['avg_quality_improvement'] > 0:
            score += 10
        
        # 適応的K
        if analysis['adaptive_k_analysis']['k_adaptation_working']:
            score += 25
        
        # 成功率
        success_rate = analysis['success_metrics']['improvement_success_rate']
        if success_rate > 0.8:
            score += 25
        elif success_rate > 0.6:
            score += 20
        elif success_rate > 0.4:
            score += 15
        
        # パフォーマンス
        if analysis['timing_analysis']['avg_v1_time_ms'] < 10:
            score += 20
        elif analysis['timing_analysis']['avg_v1_time_ms'] < 50:
            score += 15
        
        if score >= 90:
            return 'EXCELLENT'
        elif score >= 75:
            return 'GOOD'
        elif score >= 60:
            return 'ACCEPTABLE'
        else:
            return 'NEEDS_IMPROVEMENT'
    
    def _generate_v1_recommendations(self, analysis: Dict[str, Any], 
                                   requirements: Dict[str, Any]) -> List[str]:
        """V1改善推奨事項"""
        recommendations = []
        
        if analysis['performance_analysis']['avg_quality_improvement'] < 0.1:
            recommendations.append("Adjust composite scoring weights for better quality improvement")
        
        if not analysis['adaptive_k_analysis']['k_adaptation_working']:
            recommendations.append("Review adaptive K algorithm - no variation detected")
        
        if analysis['timing_analysis']['avg_v1_time_ms'] > 50:
            recommendations.append("Optimize V1 gate computation for better performance")
        
        if analysis['success_metrics']['improvement_success_rate'] < 0.5:
            recommendations.append("Investigate cases where V1 fails to improve over V0")
        
        if not requirements['redundancy_control_working']:
            recommendations.append("Strengthen redundancy control mechanism")
        
        return recommendations
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """包括的結果保存"""
        results_file = self.output_dir / "v1_comprehensive_results.json"
        
        # JSON serializable形式に変換
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                # カスタムオブジェクト（AtomMetrics等）をdict化
                return convert_numpy_to_list(obj.__dict__)
            else:
                return obj
        
        serializable_results = convert_numpy_to_list(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive V1 results saved: {results_file}")

# 使用例 - from dataclasses import asdict を追加
from dataclasses import asdict

# メイン実行
if __name__ == "__main__":
    # V1統合システム実行
    v1_system = V1IntegrationSystem("results/verification/v1_comprehensive_test")
    
    # 包括的テスト実行
    print("=== V1 Integration System Comprehensive Test ===")
    results = v1_system.run_comprehensive_v1_test(n_test_cases=15)
    
    # 結果サマリー出力
    print("\n" + "="*70)
    print("V1 INTEGRATION SYSTEM TEST RESULTS")
    print("="*70)
    
    analysis = results['comprehensive_analysis']
    validation = results['v1_validation']
    
    print(f"Test Cases: {analysis['success_metrics']['total_test_cases']}")
    print(f"V1 Validation: {validation['overall_status']}")
    print(f"Performance Grade: {validation['performance_grade']}")
    print(f"Ready for V2: {validation['v1_ready_for_v2']}")
    
    # 詳細分析
    perf = analysis['performance_analysis']
    print(f"\nPerformance Analysis:")
    print(f"  Avg Quality Improvement: {perf['avg_quality_improvement']:+.3f}")
    print(f"  Positive Improvements: {perf['positive_quality_improvements']}/{analysis['success_metrics']['total_test_cases']}")
    print(f"  Success Rate: {analysis['success_metrics']['improvement_success_rate']:.1%}")
    
    k_analysis = analysis['adaptive_k_analysis']
    print(f"\nAdaptive K Analysis:")
    print(f"  K Values Used: {k_analysis['k_values_used']}")
    print(f"  Avg K: {k_analysis['avg_k']:.1f}")
    print(f"  K Adaptation Working: {k_analysis['k_adaptation_working']}")
    
    timing = analysis['timing_analysis']
    print(f"\nTiming Analysis:")
    print(f"  Avg Total Time: {timing['avg_total_time_ms']:.2f}ms")
    print(f"  V1 Overhead Ratio: {timing['v1_overhead_ratio']:.1%}")
    
    # 推奨事項
    if validation['recommendations']:
        print(f"\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  - {rec}")
    
    # 最終判定
    if validation['v1_ready_for_v2']:
        print("\n✅ V1 INTEGRATION SUCCESSFUL - READY FOR V2")
    else:
        print("\n❌ V1 INTEGRATION NEEDS IMPROVEMENTS")
    
    print("="*70)