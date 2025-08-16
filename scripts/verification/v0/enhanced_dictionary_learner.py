#!/usr/bin/env python3
"""
Enhanced Dictionary Learner with V0 Metrics Integration
V0メトリクス統合付き拡張辞書学習器
"""

import sys
import time
from pathlib import Path

# Add the parent directories to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from scripts.cfs_v2.dictionary_learner import CFSDictionaryLearner, DictionaryConfig, AtomMetrics
from selector_metrics_logger import SelectorMetricsLogger
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedCFSDictionaryLearner(CFSDictionaryLearner):
    """V0メトリクス統合拡張辞書学習器"""
    
    def __init__(self, config: DictionaryConfig, metrics_output_dir: str = None):
        super().__init__(config)
        
        # V0メトリクスロガー初期化
        self.metrics_logger = SelectorMetricsLogger(
            output_dir=metrics_output_dir or "results/verification/v0"
        )
        
        # 選択履歴とパフォーマンス追跡
        self.selection_history = []
        self.performance_feedback = {}
        
        logger.info("Enhanced CFS Dictionary Learner with V0 metrics initialized")
    
    def select_collaborative_atoms(self, 
                                 user_sparse_code: np.ndarray, 
                                 graph_context: Dict[str, Any],
                                 top_k: int = 5,
                                 user_id: str = "unknown",
                                 query_id: str = "unknown") -> Dict[str, Any]:
        """メトリクス記録付き原子選択"""
        
        selection_start_time = time.time()
        
        # 元の選択ロジック実行
        selected_atoms = super().select_collaborative_atoms(
            user_sparse_code, graph_context, top_k
        )
        
        selection_time_ms = (time.time() - selection_start_time) * 1000
        
        # V0メトリクス記録
        metrics = self.metrics_logger.log_selection_event(
            user_id=user_id,
            query_id=query_id,
            selected_atoms=selected_atoms,
            dictionary=self.dictionary,
            atom_metrics=self.atom_metrics,
            graph_context=graph_context,
            selection_time_ms=selection_time_ms
        )
        
        # 拡張結果に metrics 情報追加
        selected_atoms['v0_metrics'] = {
            'selection_entropy': metrics.selection_entropy,
            'coverage_rate': metrics.coverage_rate,
            'redundancy_max_cos': metrics.redundancy_max_cos,
            'selection_diversity': metrics.selection_diversity,
            'avg_leakage_risk': metrics.avg_leakage_risk,
            'selection_time_ms': selection_time_ms
        }
        
        # 選択履歴記録
        self.selection_history.append({
            'timestamp': metrics.timestamp,
            'user_id': user_id,
            'query_id': query_id,
            'metrics': metrics
        })
        
        return selected_atoms
    
    def get_selection_quality_summary(self) -> Dict[str, Any]:
        """選択品質サマリー取得"""
        return self.metrics_logger.generate_summary_report()
    
    def get_user_temporal_stability(self, user_id: str) -> Dict[str, float]:
        """ユーザー時系列安定性取得"""
        return self.metrics_logger.compute_temporal_stability(user_id)
    
    def analyze_selection_patterns(self) -> Dict[str, Any]:
        """選択パターン分析"""
        if not self.selection_history:
            return {'status': 'no_data'}
        
        # ユーザー別統計
        user_stats = {}
        for entry in self.selection_history:
            user_id = entry['user_id']
            metrics = entry['metrics']
            
            if user_id not in user_stats:
                user_stats[user_id] = {
                    'selection_count': 0,
                    'avg_entropy': 0.0,
                    'avg_coverage': 0.0,
                    'avg_diversity': 0.0,
                    'total_atoms_used': set()
                }
            
            stats = user_stats[user_id]
            stats['selection_count'] += 1
            stats['avg_entropy'] += metrics.selection_entropy
            stats['avg_coverage'] += metrics.coverage_rate
            stats['avg_diversity'] += metrics.selection_diversity
            stats['total_atoms_used'].update(metrics.selected_atoms)
        
        # 平均化
        for user_id, stats in user_stats.items():
            count = stats['selection_count']
            stats['avg_entropy'] /= count
            stats['avg_coverage'] /= count
            stats['avg_diversity'] /= count
            stats['unique_atoms_count'] = len(stats['total_atoms_used'])
            del stats['total_atoms_used']  # Set は JSON serializable ではないので削除
        
        # 全体統計
        total_selections = len(self.selection_history)
        unique_users = len(user_stats)
        
        overall_entropy = np.mean([entry['metrics'].selection_entropy for entry in self.selection_history])
        overall_coverage = np.mean([entry['metrics'].coverage_rate for entry in self.selection_history])
        overall_diversity = np.mean([entry['metrics'].selection_diversity for entry in self.selection_history])
        
        return {
            'overall_stats': {
                'total_selections': total_selections,
                'unique_users': unique_users,
                'avg_entropy': float(overall_entropy),
                'avg_coverage': float(overall_coverage),
                'avg_diversity': float(overall_diversity)
            },
            'user_stats': user_stats,
            'timestamp_range': {
                'start': min(entry['timestamp'] for entry in self.selection_history),
                'end': max(entry['timestamp'] for entry in self.selection_history)
            }
        }
    
    def export_metrics_for_analysis(self, output_path: str):
        """分析用メトリクスエクスポート"""
        import json
        
        # 分析用データ準備
        analysis_data = {
            'selection_quality_summary': self.get_selection_quality_summary(),
            'selection_patterns': self.analyze_selection_patterns(),
            'config': {
                'n_atoms': self.config.n_atoms,
                'sparsity_alpha': self.config.sparsity_alpha,
                'quality_threshold': self.config.quality_threshold
            },
            'export_timestamp': time.time()
        }
        
        # エクスポート
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"V0 metrics exported to {output_path}")

# テスト・デモ実行
if __name__ == "__main__":
    import json
    
    # テスト設定
    config = DictionaryConfig(
        n_atoms=32, 
        sparsity_alpha=0.1,
        quality_threshold=0.7
    )
    
    # 拡張学習器初期化
    enhanced_learner = EnhancedCFSDictionaryLearner(
        config=config, 
        metrics_output_dir="test_v0_output"
    )
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_users, embedding_dim = 50, 768
    
    # ダミーユーザー方向生成
    user_directions = {
        f"user_{i}": np.random.randn(embedding_dim) * 0.5
        for i in range(n_users)
    }
    
    # 辞書学習
    print("=== V0 Enhanced Dictionary Learning Test ===")
    results = enhanced_learner.learn_initial_dictionary(user_directions)
    print(f"✅ Dictionary learning completed: {results['reconstruction_error']:.4f} error")
    
    # テスト選択実行
    print("\n=== V0 Metrics Logging Test ===")
    for i in range(5):
        # 既存ユーザーの疎符号を使用
        user_id = f"user_{i % n_users}"
        if user_id in user_directions:
            test_direction = user_directions[user_id]
            sparse_code, _ = enhanced_learner.encode_user_direction(test_direction)
        else:
            # フォールバック: ダミー疎符号
            sparse_code = np.random.random(32) * 0.1
        
        graph_context = {
            'similar_users': [('user_1', 0.8), ('user_2', 0.6)],
            'curriculum_stage': 1
        }
        
        selected_atoms = enhanced_learner.select_collaborative_atoms(
            sparse_code, 
            graph_context, 
            top_k=5,
            user_id=f"test_user_{i}",
            query_id=f"query_{i}"
        )
        
        v0_metrics = selected_atoms['v0_metrics']
        print(f"Selection {i}: entropy={v0_metrics['selection_entropy']:.3f}, "
              f"coverage={v0_metrics['coverage_rate']:.3f}, "
              f"diversity={v0_metrics['selection_diversity']:.3f}")
    
    # 品質サマリー生成
    print("\n=== V0 Quality Summary ===")
    quality_summary = enhanced_learner.get_selection_quality_summary()
    print(f"Quality level: {quality_summary['quality_assessment']['quality_level']}")
    print(f"Quality score: {quality_summary['quality_assessment']['quality_score']}")
    
    # パターン分析
    print("\n=== V0 Pattern Analysis ===")
    patterns = enhanced_learner.analyze_selection_patterns()
    overall = patterns['overall_stats']
    print(f"Total selections: {overall['total_selections']}")
    print(f"Average entropy: {overall['avg_entropy']:.3f}")
    print(f"Average coverage: {overall['avg_coverage']:.3f}")
    print(f"Average diversity: {overall['avg_diversity']:.3f}")
    
    # メトリクスエクスポート
    enhanced_learner.export_metrics_for_analysis("v0_test_metrics_export.json")
    
    print("\n✅ V0 Enhanced Dictionary Learner Test Completed")
    print("   - Metrics logging: WORKING")
    print("   - Quality assessment: WORKING") 
    print("   - Pattern analysis: WORKING")
    print("   - Export functionality: WORKING")