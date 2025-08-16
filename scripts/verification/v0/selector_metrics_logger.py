#!/usr/bin/env python3
"""
V0: Selector Metrics Logger
選択器メトリクス記録システム - CFS-Chameleon検証フェーズV0

実装機能:
- selection_entropy: 原子選択エントロピー 
- coverage_rate: 辞書原子カバレッジ率
- redundancy_max_cos: 選択原子間最大コサイン類似度
- selection_diversity: 選択の多様性指標
- risk_profile: リスク分布統計
- temporal_stability: 時系列安定性

出力: JSONL形式でリアルタイム記録
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SelectorMetrics:
    """選択器メトリクス"""
    timestamp: float
    user_id: str
    query_id: str
    
    # 基本選択統計
    selected_atoms: List[int]
    atom_weights: List[float] 
    selection_scores: List[float]
    total_atoms_available: int
    
    # エントロピー・多様性
    selection_entropy: float
    coverage_rate: float
    redundancy_max_cos: float
    selection_diversity: float
    
    # リスク・品質指標
    avg_leakage_risk: float
    min_stability_score: float
    quality_variance: float
    
    # 選択効率
    selection_time_ms: float
    computation_complexity: int
    memory_usage_mb: float
    
    # グラフコンテキスト
    graph_influence_ratio: float
    similar_users_count: int
    curriculum_stage: int

class SelectorMetricsLogger:
    """選択器メトリクス記録システム"""
    
    def __init__(self, output_dir: str = "results/verification/v0"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 記録ファイルパス
        timestamp = int(time.time())
        self.metrics_file = self.output_dir / f"selector_metrics_{timestamp}.jsonl"
        self.summary_file = self.output_dir / f"selector_summary_{timestamp}.json"
        
        # 内部状態
        self.metrics_history = []
        self.selection_cache = defaultdict(list)
        self.temporal_stats = defaultdict(list)
        
        logger.info(f"SelectorMetricsLogger initialized: {self.metrics_file}")
    
    def log_selection_event(self, 
                          user_id: str, 
                          query_id: str,
                          selected_atoms: Dict[str, Any], 
                          dictionary: np.ndarray,
                          atom_metrics: Dict[int, Any],
                          graph_context: Dict[str, Any] = None,
                          selection_time_ms: float = 0.0) -> SelectorMetrics:
        """選択イベント記録"""
        
        start_time = time.time()
        
        # 基本情報抽出
        atom_indices = selected_atoms.get('atom_indices', [])
        atom_weights = selected_atoms.get('atom_weights', [])
        selection_scores = selected_atoms.get('selection_scores', [])
        
        # エントロピー計算
        selection_entropy = self._compute_selection_entropy(selection_scores)
        
        # カバレッジ率計算
        coverage_rate = len(atom_indices) / dictionary.shape[1] if dictionary.shape[1] > 0 else 0.0
        
        # 冗長性計算 (最大コサイン類似度)
        redundancy_max_cos = self._compute_redundancy(atom_indices, dictionary)
        
        # 多様性指標
        selection_diversity = self._compute_diversity_index(atom_indices, dictionary)
        
        # リスク・品質統計
        avg_leakage_risk, min_stability_score, quality_variance = self._compute_risk_quality_stats(
            atom_indices, atom_metrics
        )
        
        # グラフ影響分析
        graph_influence_ratio, similar_users_count, curriculum_stage = self._analyze_graph_influence(
            graph_context
        )
        
        # メモリ使用量推定
        memory_usage_mb = self._estimate_memory_usage(atom_indices, dictionary)
        
        # メトリクス構築
        metrics = SelectorMetrics(
            timestamp=time.time(),
            user_id=user_id,
            query_id=query_id,
            selected_atoms=atom_indices,
            atom_weights=atom_weights,
            selection_scores=selection_scores,
            total_atoms_available=dictionary.shape[1],
            selection_entropy=selection_entropy,
            coverage_rate=coverage_rate,
            redundancy_max_cos=redundancy_max_cos,
            selection_diversity=selection_diversity,
            avg_leakage_risk=avg_leakage_risk,
            min_stability_score=min_stability_score,
            quality_variance=quality_variance,
            selection_time_ms=selection_time_ms,
            computation_complexity=len(atom_indices) ** 2,  # 簡易複雑度
            memory_usage_mb=memory_usage_mb,
            graph_influence_ratio=graph_influence_ratio,
            similar_users_count=similar_users_count,
            curriculum_stage=curriculum_stage
        )
        
        # 記録・蓄積
        self._write_metrics_jsonl(metrics)
        self.metrics_history.append(metrics)
        self._update_temporal_stats(user_id, metrics)
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Metrics logged for {user_id}:{query_id} in {processing_time:.2f}ms")
        
        return metrics
    
    def _compute_selection_entropy(self, selection_scores: List[float]) -> float:
        """選択エントロピー計算"""
        if not selection_scores:
            return 0.0
        
        scores = np.array(selection_scores)
        scores = scores - np.min(scores)  # 非負化
        
        if np.sum(scores) == 0:
            return 0.0
        
        probabilities = scores / np.sum(scores)
        probabilities = probabilities[probabilities > 0]  # ゼロ除去
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _compute_redundancy(self, atom_indices: List[int], dictionary: np.ndarray) -> float:
        """冗長性計算 (原子間最大コサイン類似度)"""
        if len(atom_indices) < 2:
            return 0.0
        
        max_similarity = 0.0
        
        for i in range(len(atom_indices)):
            for j in range(i + 1, len(atom_indices)):
                atom_i = dictionary[:, atom_indices[i]]
                atom_j = dictionary[:, atom_indices[j]]
                
                norm_i = np.linalg.norm(atom_i)
                norm_j = np.linalg.norm(atom_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(atom_i, atom_j) / (norm_i * norm_j)
                    max_similarity = max(max_similarity, abs(similarity))
        
        return float(max_similarity)
    
    def _compute_diversity_index(self, atom_indices: List[int], dictionary: np.ndarray) -> float:
        """多様性指標計算"""
        if len(atom_indices) < 2:
            return 0.0 if len(atom_indices) == 0 else 1.0
        
        # 選択原子の平均ペアワイズ距離
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(atom_indices)):
            for j in range(i + 1, len(atom_indices)):
                atom_i = dictionary[:, atom_indices[i]]
                atom_j = dictionary[:, atom_indices[j]]
                
                distance = np.linalg.norm(atom_i - atom_j)
                total_distance += distance
                pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
        
        # 正規化 (最大可能距離で除算)
        max_possible_distance = 2.0  # 単位球上の最大距離
        diversity_index = min(1.0, avg_distance / max_possible_distance)
        
        return float(diversity_index)
    
    def _compute_risk_quality_stats(self, atom_indices: List[int], 
                                  atom_metrics: Dict[int, Any]) -> tuple:
        """リスク・品質統計計算"""
        if not atom_indices:
            return 0.0, 1.0, 0.0
        
        leakage_risks = []
        stability_scores = []
        contribution_scores = []
        
        for atom_idx in atom_indices:
            if atom_idx in atom_metrics:
                metrics = atom_metrics[atom_idx]
                leakage_risks.append(getattr(metrics, 'leakage_risk', 0.0))
                stability_scores.append(getattr(metrics, 'stability_score', 1.0))
                contribution_scores.append(getattr(metrics, 'contribution_score', 0.5))
        
        avg_leakage_risk = np.mean(leakage_risks) if leakage_risks else 0.0
        min_stability_score = np.min(stability_scores) if stability_scores else 1.0
        quality_variance = np.var(contribution_scores) if contribution_scores else 0.0
        
        return float(avg_leakage_risk), float(min_stability_score), float(quality_variance)
    
    def _analyze_graph_influence(self, graph_context: Dict[str, Any]) -> tuple:
        """グラフ影響分析"""
        if not graph_context:
            return 0.0, 0, 0
        
        similar_users = graph_context.get('similar_users', [])
        similar_users_count = len(similar_users)
        
        # グラフ影響比率 (類似ユーザー数に基づく)
        graph_influence_ratio = min(1.0, similar_users_count / 10.0)  # 最大10ユーザーで飽和
        
        curriculum_stage = graph_context.get('curriculum_stage', 0)
        
        return float(graph_influence_ratio), int(similar_users_count), int(curriculum_stage)
    
    def _estimate_memory_usage(self, atom_indices: List[int], dictionary: np.ndarray) -> float:
        """メモリ使用量推定"""
        # 選択原子のメモリ使用量推定
        selected_memory = len(atom_indices) * dictionary.shape[0] * 4  # float32 仮定
        
        # 計算バッファのメモリ使用量
        computation_memory = len(atom_indices) ** 2 * 4  # 類似度行列
        
        total_memory_bytes = selected_memory + computation_memory
        memory_mb = total_memory_bytes / (1024 * 1024)
        
        return float(memory_mb)
    
    def _write_metrics_jsonl(self, metrics: SelectorMetrics):
        """メトリクスJSONL書き込み"""
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, ensure_ascii=False)
            f.write('\n')
    
    def _update_temporal_stats(self, user_id: str, metrics: SelectorMetrics):
        """時系列統計更新"""
        self.temporal_stats[user_id].append({
            'timestamp': metrics.timestamp,
            'entropy': metrics.selection_entropy,
            'coverage': metrics.coverage_rate,
            'redundancy': metrics.redundancy_max_cos,
            'diversity': metrics.selection_diversity
        })
        
        # 最新10回のみ保持
        if len(self.temporal_stats[user_id]) > 10:
            self.temporal_stats[user_id] = self.temporal_stats[user_id][-10:]
    
    def compute_temporal_stability(self, user_id: str) -> Dict[str, float]:
        """時系列安定性計算"""
        if user_id not in self.temporal_stats or len(self.temporal_stats[user_id]) < 2:
            return {'entropy_stability': 1.0, 'coverage_stability': 1.0, 'diversity_stability': 1.0}
        
        history = self.temporal_stats[user_id]
        
        # 各メトリクスの標準偏差から安定性計算
        entropies = [h['entropy'] for h in history]
        coverages = [h['coverage'] for h in history]
        diversities = [h['diversity'] for h in history]
        
        entropy_stability = 1.0 - min(1.0, np.std(entropies))
        coverage_stability = 1.0 - min(1.0, np.std(coverages))
        diversity_stability = 1.0 - min(1.0, np.std(diversities))
        
        return {
            'entropy_stability': float(entropy_stability),
            'coverage_stability': float(coverage_stability),
            'diversity_stability': float(diversity_stability)
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """サマリーレポート生成"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        # 基本統計
        entropies = [m.selection_entropy for m in self.metrics_history]
        coverages = [m.coverage_rate for m in self.metrics_history]
        redundancies = [m.redundancy_max_cos for m in self.metrics_history]
        diversities = [m.selection_diversity for m in self.metrics_history]
        
        summary = {
            'total_selections': len(self.metrics_history),
            'time_range': {
                'start': min(m.timestamp for m in self.metrics_history),
                'end': max(m.timestamp for m in self.metrics_history)
            },
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies))
            },
            'coverage_stats': {
                'mean': float(np.mean(coverages)),
                'std': float(np.std(coverages)),
                'min': float(np.min(coverages)),
                'max': float(np.max(coverages))
            },
            'redundancy_stats': {
                'mean': float(np.mean(redundancies)),
                'std': float(np.std(redundancies)),
                'min': float(np.min(redundancies)),
                'max': float(np.max(redundancies))
            },
            'diversity_stats': {
                'mean': float(np.mean(diversities)),
                'std': float(np.std(diversities)),
                'min': float(np.min(diversities)),
                'max': float(np.max(diversities))
            },
            'quality_assessment': self._assess_overall_quality()
        }
        
        # サマリーファイル書き込み
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report generated: {self.summary_file}")
        return summary
    
    def _assess_overall_quality(self) -> Dict[str, Any]:
        """全体品質評価"""
        if not self.metrics_history:
            return {'status': 'insufficient_data'}
        
        # 品質基準
        avg_entropy = np.mean([m.selection_entropy for m in self.metrics_history])
        avg_coverage = np.mean([m.coverage_rate for m in self.metrics_history])
        avg_redundancy = np.mean([m.redundancy_max_cos for m in self.metrics_history])
        avg_diversity = np.mean([m.selection_diversity for m in self.metrics_history])
        
        # 品質判定
        quality_score = 0.0
        
        # 高エントロピー = 良い選択多様性
        if avg_entropy > 2.0:
            quality_score += 25
        elif avg_entropy > 1.0:
            quality_score += 15
        
        # 適度なカバレッジ
        if 0.05 <= avg_coverage <= 0.2:
            quality_score += 25
        elif 0.02 <= avg_coverage <= 0.3:
            quality_score += 15
        
        # 低冗長性
        if avg_redundancy < 0.3:
            quality_score += 25
        elif avg_redundancy < 0.5:
            quality_score += 15
        
        # 高多様性
        if avg_diversity > 0.7:
            quality_score += 25
        elif avg_diversity > 0.5:
            quality_score += 15
        
        # 品質レベル判定
        if quality_score >= 80:
            quality_level = 'excellent'
        elif quality_score >= 60:
            quality_level = 'good'
        elif quality_score >= 40:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'avg_entropy': float(avg_entropy),
            'avg_coverage': float(avg_coverage),
            'avg_redundancy': float(avg_redundancy),
            'avg_diversity': float(avg_diversity),
            'recommendations': self._generate_recommendations(quality_score, avg_entropy, avg_coverage, avg_redundancy, avg_diversity)
        }
    
    def _generate_recommendations(self, quality_score: float, avg_entropy: float, 
                                avg_coverage: float, avg_redundancy: float, 
                                avg_diversity: float) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        if avg_entropy < 1.0:
            recommendations.append("Increase selection entropy by adjusting scoring weights")
        
        if avg_coverage < 0.02:
            recommendations.append("Increase atom coverage by expanding top-K selection")
        elif avg_coverage > 0.3:
            recommendations.append("Reduce atom coverage to improve efficiency")
        
        if avg_redundancy > 0.5:
            recommendations.append("Reduce redundancy by improving diversity penalties")
        
        if avg_diversity < 0.5:
            recommendations.append("Improve selection diversity through better MMR implementation")
        
        if quality_score < 40:
            recommendations.append("Consider comprehensive selector redesign")
        
        return recommendations

# 統合用ヘルパー関数
def create_metrics_logger(output_dir: str = None) -> SelectorMetricsLogger:
    """メトリクスロガー作成ヘルパー"""
    if output_dir is None:
        output_dir = "results/verification/v0"
    
    return SelectorMetricsLogger(output_dir)

# 使用例
if __name__ == "__main__":
    # テスト用ダミーデータ
    np.random.seed(42)
    
    # ダミー辞書
    dictionary = np.random.randn(768, 64)
    
    # ダミー原子メトリクス
    atom_metrics = {
        i: type('AtomMetrics', (), {
            'leakage_risk': np.random.random(),
            'stability_score': np.random.random(),
            'contribution_score': np.random.random()
        })() for i in range(64)
    }
    
    # メトリクスロガーテスト
    logger_instance = create_metrics_logger("test_output")
    
    # テスト選択イベント
    for i in range(10):
        selected_atoms = {
            'atom_indices': np.random.choice(64, size=5, replace=False).tolist(),
            'atom_weights': np.random.random(5).tolist(),
            'selection_scores': np.random.random(5).tolist()
        }
        
        graph_context = {
            'similar_users': [('user_1', 0.8), ('user_2', 0.6)],
            'curriculum_stage': 1
        }
        
        metrics = logger_instance.log_selection_event(
            user_id=f"test_user_{i}",
            query_id=f"query_{i}",
            selected_atoms=selected_atoms,
            dictionary=dictionary,
            atom_metrics=atom_metrics,
            graph_context=graph_context,
            selection_time_ms=10.5
        )
    
    # サマリーレポート生成
    summary = logger_instance.generate_summary_report()
    
    print("✅ V0 Selector Metrics Logger Test Completed")
    print(f"   Metrics logged: {summary['total_selections']}")
    print(f"   Quality level: {summary['quality_assessment']['quality_level']}")
    print(f"   Avg entropy: {summary['entropy_stats']['mean']:.3f}")
    print(f"   Output files: {logger_instance.metrics_file}")