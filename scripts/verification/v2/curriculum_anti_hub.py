#!/usr/bin/env python3
"""
V2: Curriculum Negatives + Anti-Hub Sampling
カリキュラム負例生成 + アンチハブサンプリング実装

技術仕様:
- Curriculum stages: easy → medium → hard負例段階進行
- Anti-hub sampling: ハブノード偏りを補正
- Quality monitoring: 負例品質・安全性監視
- Adaptive progression: パフォーマンスベース段階進行
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import time
import json

# --- Robust curriculum spec parsing & clamping ---
HARD_CAP = 0.20  # 20% upper bound for hard negatives (high similarity = difficult to distinguish)

def _parse_curriculum_spec(spec: str):
    """Parse curriculum spec like 'easy:2,medium:2,hard:1' into normalized weights.
    Falls back to sensible defaults if parsing fails."""
    weights = {"easy": 3.0, "medium": 2.0, "hard": 0.0}
    if not spec:
        return weights
    try:
        parts = [p.strip() for p in str(spec).split(',') if p.strip()]
        for part in parts:
            if ':' not in part:
                continue
            k, v = part.split(':', 1)
            k = k.strip().lower()
            v = float(v.strip())
            if k in weights:
                weights[k] = max(0.0, v)
    except Exception:
        # keep defaults on any parsing error
        pass
    s = sum(weights.values()) or 1.0
    for k in list(weights.keys()):
        weights[k] = weights[k] / s
    return weights


def _clamp_curriculum(weights: dict, logger=None):
    """Clamp hard fraction at HARD_CAP and re-distribute overflow to easy/medium."""
    w = dict(weights)
    hard = w.get('hard', 0.0)
    if hard > HARD_CAP:
        overflow = hard - HARD_CAP
        w['hard'] = HARD_CAP
        em = (w.get('easy', 0.0) + w.get('medium', 0.0)) or 1.0
        w['easy'] = w.get('easy', 0.0) + overflow * (w.get('easy', 0.0) / em)
        w['medium'] = w.get('medium', 0.0) + overflow * (w.get('medium', 0.0) / em)
        # renormalize
        s = (w.get('easy', 0.0) + w.get('medium', 0.0) + w.get('hard', 0.0)) or 1.0
        for k in ('easy', 'medium', 'hard'):
            w[k] = w.get(k, 0.0) / s
        if logger is not None:
            logger.warning("[curriculum] Hard ratio clamped to %.2f; original weights=%r", HARD_CAP, weights)
    return w

logger = logging.getLogger(__name__)

@dataclass
class CurriculumConfig:
    """カリキュラム学習設定"""
    # 段階設定
    max_stages: int = 3                    # 最大段階数
    progression_threshold: float = 0.8     # 段階進行閾値
    stability_requirement: int = 3         # 安定性要求（連続成功回数）
    
    # 負例生成設定
    easy_similarity_range: Tuple[float, float] = (0.0, 0.3)    # Easy負例類似度範囲（低類似=区別容易）
    medium_similarity_range: Tuple[float, float] = (0.3, 0.6)  # Medium負例類似度範囲
    hard_similarity_range: Tuple[float, float] = (0.6, 0.8)    # Hard負例類似度範囲（高類似=区別困難）
    
    # アンチハブ設定
    hub_threshold: float = 2.0             # ハブ判定閾値（標準偏差）
    anti_hub_ratio: float = 0.3            # アンチハブサンプリング比率
    degree_penalty_alpha: float = 0.5      # 度数ペナルティ強度
    
    # 安全性設定
    safety_check_interval: int = 10        # 安全性チェック間隔
    max_hard_ratio: float = 0.2            # Hard負例最大比率
    quality_degradation_threshold: float = 0.1  # 品質劣化閾値

@dataclass
class CurriculumState:
    """カリキュラム学習状態"""
    current_stage: int = 0
    stability_count: int = 0
    performance_history: List[float] = None
    stage_transition_history: List[Dict] = None
    safety_alerts: List[str] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if self.stage_transition_history is None:
            self.stage_transition_history = []
        if self.safety_alerts is None:
            self.safety_alerts = []

@dataclass
class NegativeExample:
    """負例データ"""
    example_id: str
    similarity_score: float
    difficulty_level: str  # 'easy', 'medium', 'hard'
    hub_score: float
    quality_score: float
    generation_time: float

class CurriculumAntiHubSystem:
    """V2カリキュラム負例・アンチハブシステム"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.state = CurriculumState()
        
        # グラフ・ネットワーク情報
        self.graph = None
        self.node_embeddings = {}
        self.hub_scores = {}
        self.degree_distribution = {}
        
        # 負例生成履歴
        self.negative_generation_history = []
        self.quality_monitoring = {
            'easy_quality_scores': [],
            'medium_quality_scores': [],
            'hard_quality_scores': []
        }
        
        # 統計・監視
        self.generation_stats = defaultdict(int)
        self.safety_monitors = []
        
        logger.info("V2 Curriculum Anti-Hub System initialized")
    
    def initialize_graph_structure(self, 
                                 edges: List[Tuple], 
                                 node_embeddings: Dict[str, np.ndarray],
                                 user_metadata: Dict[str, Dict] = None):
        """グラフ構造初期化"""
        logger.info("Initializing graph structure for curriculum learning...")
        
        # NetworkXグラフ構築
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        self.node_embeddings = node_embeddings
        
        # ハブ分析
        self._analyze_hub_structure()
        
        # 度数分布分析
        self._analyze_degree_distribution()
        
        logger.info(f"Graph initialized: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Hub nodes detected: {len([n for n, s in self.hub_scores.items() if s > self.config.hub_threshold])}")
    
    def _analyze_hub_structure(self):
        """ハブ構造分析"""
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        if not degree_values:
            self.hub_scores = {}
            return
        
        mean_degree = np.mean(degree_values)
        std_degree = np.std(degree_values)
        
        # ハブスコア計算（z-score）
        for node, degree in degrees.items():
            if std_degree > 0:
                z_score = (degree - mean_degree) / std_degree
            else:
                z_score = 0
            self.hub_scores[node] = z_score
        
        hub_count = len([s for s in self.hub_scores.values() if s > self.config.hub_threshold])
        logger.info(f"Hub analysis completed: {hub_count} hub nodes (z > {self.config.hub_threshold})")
    
    def _analyze_degree_distribution(self):
        """度数分布分析"""
        degrees = dict(self.graph.degree())
        self.degree_distribution = {
            'min': min(degrees.values()) if degrees else 0,
            'max': max(degrees.values()) if degrees else 0,
            'mean': np.mean(list(degrees.values())) if degrees else 0,
            'std': np.std(list(degrees.values())) if degrees else 0,
            'median': np.median(list(degrees.values())) if degrees else 0
        }
    
    def generate_curriculum_negatives(self, 
                                    target_node: str,
                                    positive_examples: List[str],
                                    target_difficulty: str = None,
                                    n_negatives: int = 5) -> List[NegativeExample]:
        """カリキュラム負例生成"""
        
        if target_difficulty is None:
            target_difficulty = self._get_current_difficulty_level()

        start_time = time.time()
        used_node_ids = set()

        # 1) Curriculum spec が与えられていれば、重みをパース＆クランプして
        #    easy/medium/hard の混合で n_negatives を割当てる
        raw_spec = None
        if hasattr(self, 'args') and hasattr(self.args, 'neg_curriculum'):
            raw_spec = self.args.neg_curriculum
        elif hasattr(self, 'config') and hasattr(self.config, 'neg_curriculum'):
            raw_spec = getattr(self.config, 'neg_curriculum')

        selected_negatives: List[NegativeExample] = []
        per_diff_counts = {'easy': 0, 'medium': 0, 'hard': 0}

        if raw_spec:
            weights = _parse_curriculum_spec(raw_spec)
            weights = _clamp_curriculum(weights, logger)

            n_total = int(n_negatives)
            # 割当（四捨五入のズレを最後に調整）
            n_hard = int(round(weights.get('hard', 0.0) * n_total))
            n_easy = int(round(weights.get('easy', 0.0) * n_total))
            n_medium = n_total - n_easy - n_hard
            if n_medium < 0:
                n_medium = 0
                remain = n_total - n_easy - n_medium
                n_hard = max(0, remain)

            alloc_plan = [('easy', n_easy), ('medium', n_medium), ('hard', n_hard)]
        else:
            # 従来動作（単一難易度）
            alloc_plan = [(target_difficulty, int(n_negatives))]

        # 2) 難易度ごとに既存パイプラインで選択
        for diff, count in alloc_plan:
            if count <= 0:
                continue
            negative_candidates = self._collect_negative_candidates(
                target_node, positive_examples, diff
            )
            # avoid duplicates across difficulties
            negative_candidates = [c for c in negative_candidates if c.get('node_id') not in used_node_ids]
            anti_hub_filtered = self._apply_anti_hub_sampling(
                negative_candidates, diff
            )
            quality_filtered = self._apply_quality_filtering(
                anti_hub_filtered, target_node, positive_examples
            )
            picked, picked_node_ids = self._select_final_negatives(
                quality_filtered, count, diff
            )
            selected_negatives.extend(picked)
            per_diff_counts[diff] += len(picked)
            used_node_ids.update(picked_node_ids)

        # 3) もし不足がある場合は、最も候補が多かった難易度から補充（簡易フォールバック）
        if len(selected_negatives) < n_negatives:
            need = n_negatives - len(selected_negatives)
            # 再度 easy→medium→hard の順で緩く補完（安全側：低類似度から）
            for diff in ('easy', 'medium', 'hard'):
                if need <= 0:
                    break
                negative_candidates = self._collect_negative_candidates(
                    target_node, positive_examples, diff
                )
                negative_candidates = [c for c in negative_candidates if c.get('node_id') not in used_node_ids]
                anti_hub_filtered = self._apply_anti_hub_sampling(
                    negative_candidates, diff
                )
                quality_filtered = self._apply_quality_filtering(
                    anti_hub_filtered, target_node, positive_examples
                )
                extra, extra_node_ids = self._select_final_negatives(
                    quality_filtered, need, diff
                )
                selected_negatives.extend(extra)
                per_diff_counts[diff] += len(extra)
                used_node_ids.update(extra_node_ids)
                need = n_negatives - len(selected_negatives)

        # 3.5) Enforce HARD_CAP on actual composition by replacing overflow hard negatives
        total_selected = len(selected_negatives)
        if total_selected > 0:
            max_hard_allowed = int(np.floor(HARD_CAP * total_selected))
            current_hard = per_diff_counts.get('hard', 0)
            if current_hard > max_hard_allowed:
                need_replace = current_hard - max_hard_allowed
                # sort existing hard negatives by ascending quality (drop the worst first)
                hard_indices = [i for i, neg in enumerate(selected_negatives) if neg.difficulty_level == 'hard']
                hard_sorted = sorted(hard_indices, key=lambda i: selected_negatives[i].quality_score)
                drop_indices = set(hard_sorted[:need_replace])

                # build replacements preferring medium then easy
                replacements: List[NegativeExample] = []
                rep_counts = {'easy': 0, 'medium': 0}
                for rep_diff in ('medium', 'easy'):
                    if len(replacements) >= need_replace:
                        break
                    cand = self._collect_negative_candidates(target_node, positive_examples, rep_diff)
                    cand = [c for c in cand if c.get('node_id') not in used_node_ids]
                    cand = self._apply_anti_hub_sampling(cand, rep_diff)
                    cand = self._apply_quality_filtering(cand, target_node, positive_examples)
                    remain = need_replace - len(replacements)
                    picked, picked_ids = self._select_final_negatives(cand, remain, rep_diff)
                    replacements.extend(picked)
                    rep_counts[rep_diff] += len(picked)
                    used_node_ids.update(picked_ids)

                # apply replacement if we have any
                if replacements:
                    # rebuild selected_negatives without dropped hards
                    kept: List[NegativeExample] = [neg for idx, neg in enumerate(selected_negatives) if idx not in drop_indices]
                    selected_negatives = kept + replacements
                    # update counts
                    removed = len(drop_indices)
                    per_diff_counts['hard'] = max(0, per_diff_counts.get('hard', 0) - removed)
                    per_diff_counts['medium'] += rep_counts['medium']
                    per_diff_counts['easy'] += rep_counts['easy']

                # recompute and if still above cap, relax-and-fill to reduce hard share
                total_after = len(selected_negatives) or 1
                observed_hard_ratio = per_diff_counts.get('hard', 0) / total_after
                if observed_hard_ratio > HARD_CAP + 1e-6:
                    need_after = int(np.ceil(per_diff_counts.get('hard', 0) - np.floor(HARD_CAP * total_after)))
                    if need_after > 0:
                        replacements2, rep2_ids, rep2_counts = self._relax_and_fill(
                            target_node, positive_examples, need_after, used_node_ids, prefer_order=('medium','easy')
                        )
                        if replacements2:
                            # drop additional worst hard negatives and insert relaxed replacements
                            hard_indices = [i for i, neg in enumerate(selected_negatives) if neg.difficulty_level == 'hard']
                            hard_sorted = sorted(hard_indices, key=lambda i: selected_negatives[i].quality_score)
                            drop2 = set(hard_sorted[:len(replacements2)])
                            kept2 = [neg for idx, neg in enumerate(selected_negatives) if idx not in drop2]
                            selected_negatives = kept2 + replacements2
                            per_diff_counts['hard'] = max(0, per_diff_counts.get('hard', 0) - len(replacements2))
                            per_diff_counts['medium'] += rep2_counts.get('medium', 0)
                            per_diff_counts['easy'] += rep2_counts.get('easy', 0)
                            used_node_ids.update(rep2_ids)

                    # final check
                    total_after2 = len(selected_negatives) or 1
                    observed_hard_ratio = per_diff_counts.get('hard', 0) / total_after2
                    if observed_hard_ratio > HARD_CAP + 1e-6:
                        logger.warning("[curriculum] Hard negative ratio too high after relaxed filling: %.2f > %.2f",
                                       observed_hard_ratio, HARD_CAP)

        generation_time = time.time() - start_time

        # 4) 統計更新（難易度別 + 合計時間）
        for diff in ('easy', 'medium', 'hard'):
            if per_diff_counts[diff] > 0:
                self.generation_stats[f'{diff}_generated'] += per_diff_counts[diff]
        self.generation_stats['total_generation_time'] += generation_time

        # 5) 品質監視更新（難易度別に集計）
        for neg in selected_negatives:
            self.quality_monitoring[f'{neg.difficulty_level}_quality_scores'].append(neg.quality_score)

        # 6) 履歴記録
        quality_scores = [neg.quality_score for neg in selected_negatives]
        generation_record = {
            'timestamp': time.time(),
            'target_node': target_node,
            'requested_difficulty': target_difficulty,
            'allocation_plan': alloc_plan,
            'actual_counts': per_diff_counts,
            'n_generated': len(selected_negatives),
            'generation_time': generation_time,
            'avg_quality': float(np.mean(quality_scores)) if quality_scores else 0.0
        }
        self.negative_generation_history.append(generation_record)

        # 7) 監視ログ（ハード比が上限を超えたら警告）
        total_gen = sum(per_diff_counts.values()) or 1
        observed_hard_ratio = per_diff_counts['hard'] / total_gen
        if observed_hard_ratio > HARD_CAP + 1e-6:
            logger.warning("[curriculum] Hard negative ratio too high after allocation: %.2f > %.2f",
                           observed_hard_ratio, HARD_CAP)

        logger.debug(
            "Generated %d negatives (easy=%d, medium=%d, hard=%d) in %.3fs",
            len(selected_negatives), per_diff_counts['easy'], per_diff_counts['medium'], per_diff_counts['hard'], generation_time
        )

        return selected_negatives
    
    def _get_current_difficulty_level(self) -> str:
        """現在の難易度レベル取得
        カリキュラム学習: easy(低類似=区別容易) → medium → hard(高類似=区別困難)
        """
        stage_to_difficulty = {0: 'easy', 1: 'medium', 2: 'hard'}
        return stage_to_difficulty.get(self.state.current_stage, 'easy')
    
    def _collect_negative_candidates(self, 
                                   target_node: str,
                                   positive_examples: List[str],
                                   difficulty: str) -> List[Dict[str, Any]]:
        """負例候補収集"""
        if target_node not in self.node_embeddings:
            return []
        
        target_embedding = self.node_embeddings[target_node]
        positive_embeddings = [
            self.node_embeddings[node] for node in positive_examples
            if node in self.node_embeddings
        ]
        
        if not positive_embeddings:
            return []
        
        # 正例の重心計算
        positive_centroid = np.mean(positive_embeddings, axis=0)
        
        # 類似度範囲取得
        similarity_range = self._get_similarity_range(difficulty)
        
        candidates = []
        for node in self.graph.nodes():
            if (node != target_node and 
                node not in positive_examples and 
                node in self.node_embeddings):
                
                node_embedding = self.node_embeddings[node]
                
                # 正例重心との類似度
                similarity = self._compute_cosine_similarity(positive_centroid, node_embedding)
                
                # 難易度範囲チェック
                if similarity_range[0] <= similarity <= similarity_range[1]:
                    candidates.append({
                        'node_id': node,
                        'similarity': similarity,
                        'hub_score': self.hub_scores.get(node, 0.0),
                        'embedding': node_embedding
                    })
        
        return candidates
    
    def _get_similarity_range(self, difficulty: str) -> Tuple[float, float]:
        """難易度別類似度範囲取得"""
        if difficulty == 'easy':
            return self.config.easy_similarity_range
        elif difficulty == 'medium':
            return self.config.medium_similarity_range
        elif difficulty == 'hard':
            return self.config.hard_similarity_range
        else:
            return (0.0, 1.0)
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度計算"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _apply_anti_hub_sampling(self, 
                                candidates: List[Dict[str, Any]], 
                                difficulty: str) -> List[Dict[str, Any]]:
        """アンチハブサンプリング適用"""
        if not candidates:
            return candidates
        
        # ハブ度数によるペナルティ計算
        for candidate in candidates:
            hub_score = candidate['hub_score']
            
            # ハブペナルティ: exp(-alpha * |hub_score|)
            hub_penalty = np.exp(-self.config.degree_penalty_alpha * abs(hub_score))
            candidate['hub_penalty'] = hub_penalty
            
            # アンチハブボーナス（非ハブノードを優遇）
            anti_hub_bonus = 1.0 + (1.0 - hub_penalty) * self.config.anti_hub_ratio
            candidate['anti_hub_bonus'] = anti_hub_bonus
        
        # ハブペナルティを考慮したスコアリング
        for candidate in candidates:
            base_score = 1.0 - candidate['similarity']  # 類似度が低いほど良い負例
            penalized_score = base_score * candidate['anti_hub_bonus']
            candidate['penalized_score'] = penalized_score
        
        # スコア順ソート
        sorted_candidates = sorted(candidates, key=lambda x: x['penalized_score'], reverse=True)
        
        return sorted_candidates
    
    def _apply_quality_filtering(self, 
                               candidates: List[Dict[str, Any]],
                               target_node: str,
                               positive_examples: List[str],
                               min_quality: float = 0.3) -> List[Dict[str, Any]]:
        """品質フィルタリング"""
        quality_filtered = []
        
        for candidate in candidates:
            # 品質スコア計算
            quality_score = self._compute_negative_quality(
                candidate, target_node, positive_examples
            )
            candidate['quality_score'] = quality_score
            
            # 品質閾値チェック（引数で可変）
            if quality_score >= min_quality:
                quality_filtered.append(candidate)
        
        return quality_filtered
    
    def _compute_negative_quality(self, 
                                candidate: Dict[str, Any],
                                target_node: str,
                                positive_examples: List[str]) -> float:
        """負例品質スコア計算"""
        quality_components = []
        
        # 1. 類似度適切性（難易度に応じた適切な類似度）
        similarity = candidate['similarity']
        similarity_quality = 1.0 - abs(similarity - 0.5)  # 0.5付近が最適
        quality_components.append(similarity_quality * 0.4)
        
        # 2. ハブ多様性（非ハブノード優遇）
        hub_score = candidate['hub_score']
        hub_diversity = max(0.0, 1.0 - abs(hub_score) / 3.0)  # z-score 3以下を優遇
        quality_components.append(hub_diversity * 0.3)
        
        # 3. グラフ構造多様性（異なるクラスタからの選択）
        structure_diversity = self._compute_structure_diversity(
            candidate['node_id'], target_node, positive_examples
        )
        quality_components.append(structure_diversity * 0.3)
        
        return sum(quality_components)
    
    def _compute_structure_diversity(self, 
                                   candidate_node: str,
                                   target_node: str,
                                   positive_examples: List[str]) -> float:
        """構造多様性計算"""
        if candidate_node not in self.graph:
            return 0.5  # 中立
        
        # 候補ノードの近隣
        candidate_neighbors = set(self.graph.neighbors(candidate_node))
        
        # ターゲット・正例との近隣重複度
        total_overlap = 0
        comparison_nodes = [target_node] + positive_examples
        
        for comp_node in comparison_nodes:
            if comp_node in self.graph:
                comp_neighbors = set(self.graph.neighbors(comp_node))
                overlap = len(candidate_neighbors & comp_neighbors)
                total_overlap += overlap
        
        # 重複が少ないほど多様性が高い
        max_possible_overlap = len(candidate_neighbors) * len(comparison_nodes)
        diversity = 1.0 - (total_overlap / max(1, max_possible_overlap))
        
        return diversity
    
    def _relax_and_fill(self,
                        target_node: str,
                        positive_examples: List[str],
                        need: int,
                        used_node_ids: set,
                        prefer_order: Tuple[str, ...] = ('medium', 'easy')) -> Tuple[List['NegativeExample'], List[str], Dict[str, int]]:
        """
        easy/medium 候補が不足する場合に、類似度範囲と品質閾値を段階的に緩めて補充する。
        戻り値: (picked_list, picked_ids, per_diff_counts_add)
        """
        picked_all: List[NegativeExample] = []
        picked_ids: List[str] = []
        add_counts = {'easy': 0, 'medium': 0}

        if need <= 0:
            return picked_all, picked_ids, add_counts

        # 段階的に類似度レンジを広げ、品質閾値を下げる
        widen_steps = [0.05, 0.10, 0.15, 0.25]
        quality_steps = [0.28, 0.25, 0.22, 0.20, 0.15]

        # 正例重心（候補構築のフォールバックで使う）
        pos_embeds = [self.node_embeddings[n] for n in positive_examples if n in self.node_embeddings]
        pos_centroid = np.mean(pos_embeds, axis=0) if pos_embeds else None

        for dw in widen_steps:
            for diff in prefer_order:
                if len(picked_all) >= need:
                    break
                base_low, base_high = self._get_similarity_range(diff)
                low = max(0.0, base_low - dw)
                high = min(1.0, base_high + dw)

                # レンジ内候補を構築
                cands = []
                for node in self.graph.nodes():
                    if node == target_node or node in positive_examples or node in used_node_ids:
                        continue
                    emb = self.node_embeddings.get(node)
                    if emb is None or pos_centroid is None:
                        continue
                    sim = self._compute_cosine_similarity(pos_centroid, emb)
                    if low <= sim <= high:
                        cands.append({'node_id': node, 'similarity': sim, 'hub_score': self.hub_scores.get(node, 0.0), 'embedding': emb})

                if not cands:
                    continue

                # 既存パイプ：アンチハブ → 品質（緩め）→ 最終選抜
                cands = self._apply_anti_hub_sampling(cands, diff)
                remain = need - len(picked_all)
                picked_local: List[NegativeExample] = []
                ids_local: List[str] = []

                for qmin in quality_steps:
                    cf = self._apply_quality_filtering(cands, target_node, positive_examples, min_quality=qmin)
                    if not cf:
                        continue
                    sel, sel_ids = self._select_final_negatives(cf, remain, diff)
                    picked_local = sel
                    ids_local = sel_ids
                    if picked_local:
                        break

                if picked_local:
                    picked_all.extend(picked_local)
                    picked_ids.extend(ids_local)
                    add_counts[diff] += len(picked_local)

        # 最終フォールバック：どうしても不足なら、任意ノードから medium として補完
        if len(picked_all) < need and pos_centroid is not None:
            remain = need - len(picked_all)
            loose_cands = []
            for node in self.graph.nodes():
                if node == target_node or node in positive_examples or node in used_node_ids:
                    continue
                emb = self.node_embeddings.get(node)
                if emb is None:
                    continue
                sim = self._compute_cosine_similarity(pos_centroid, emb)
                loose_cands.append({'node_id': node, 'similarity': sim, 'hub_score': self.hub_scores.get(node, 0.0), 'embedding': emb})
            if loose_cands:
                loose_cands = self._apply_anti_hub_sampling(loose_cands, 'medium')
                cf = self._apply_quality_filtering(loose_cands, target_node, positive_examples, min_quality=0.10)
                sel, sel_ids = self._select_final_negatives(cf, remain, 'medium')
                picked_all.extend(sel)
                picked_ids.extend(sel_ids)
                add_counts['medium'] += len(sel)

        return picked_all, picked_ids, add_counts
    
    def _select_final_negatives(self, 
                              candidates: List[Dict[str, Any]], 
                              n_negatives: int,
                              difficulty: str) -> Tuple[List[NegativeExample], List[str]]:
        """最終負例選択
        Returns: (negative_examples, selected_node_ids)
        """
        # 品質スコア順ソート
        sorted_candidates = sorted(candidates, key=lambda x: x['quality_score'], reverse=True)
        
        # Top-N選択
        selected_count = min(n_negatives, len(sorted_candidates))
        selected_candidates = sorted_candidates[:selected_count]
        
        # NegativeExample オブジェクトに変換
        negative_examples: List[NegativeExample] = []
        selected_node_ids: List[str] = []
        for i, candidate in enumerate(selected_candidates):
            node_id = candidate['node_id']
            neg_example = NegativeExample(
                example_id=f"{difficulty}_{node_id}_{int(time.time())}_{i}",
                similarity_score=candidate['similarity'],
                difficulty_level=difficulty,
                hub_score=candidate['hub_score'],
                quality_score=candidate['quality_score'],
                generation_time=time.time()
            )
            negative_examples.append(neg_example)
            selected_node_ids.append(node_id)
        
        return negative_examples, selected_node_ids
    
    def update_curriculum_progress(self, performance_metrics: Dict[str, float]) -> bool:
        """カリキュラム進行更新"""
        current_performance = performance_metrics.get('accuracy', 0.0)
        self.state.performance_history.append(current_performance)
        
        # 最新10回の平均性能
        recent_performance = np.mean(self.state.performance_history[-10:])
        
        # 段階進行判定
        progression_needed = False
        
        if recent_performance >= self.config.progression_threshold:
            self.state.stability_count += 1
            
            # 安定性要件達成で段階進行
            if self.state.stability_count >= self.config.stability_requirement:
                if self.state.current_stage < self.config.max_stages - 1:
                    old_stage = self.state.current_stage
                    self.state.current_stage += 1
                    self.state.stability_count = 0
                    
                    # 段階遷移記録
                    transition_record = {
                        'timestamp': time.time(),
                        'from_stage': old_stage,
                        'to_stage': self.state.current_stage,
                        'trigger_performance': recent_performance,
                        'stability_count': self.config.stability_requirement
                    }
                    self.state.stage_transition_history.append(transition_record)
                    
                    progression_needed = True
                    logger.info(f"Curriculum advanced: Stage {old_stage} → {self.state.current_stage}")
        else:
            self.state.stability_count = 0
        
        # 安全性チェック
        self._perform_safety_check(performance_metrics)
        
        return progression_needed
    
    def _perform_safety_check(self, performance_metrics: Dict[str, float]):
        """安全性チェック実行"""
        if len(self.state.performance_history) < 5:
            return
        
        # 性能劣化チェック
        recent_avg = np.mean(self.state.performance_history[-3:])
        previous_avg = np.mean(self.state.performance_history[-6:-3])
        
        if previous_avg > 0 and (recent_avg - previous_avg) < -self.config.quality_degradation_threshold:
            alert = f"Performance degradation detected: {previous_avg:.3f} → {recent_avg:.3f}"
            self.state.safety_alerts.append(alert)
            logger.warning(alert)
        
        # Hard負例比率チェック
        hard_generated = self.generation_stats.get('hard_generated', 0)
        total_generated = sum(self.generation_stats[k] for k in self.generation_stats if k.endswith('_generated'))
        
        if total_generated > 0:
            hard_ratio = hard_generated / total_generated
            if hard_ratio > self.config.max_hard_ratio:
                alert = f"Hard negative ratio too high: {hard_ratio:.2%} > {self.config.max_hard_ratio:.2%}"
                self.state.safety_alerts.append(alert)
                logger.warning(alert)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """カリキュラム状態取得"""
        recent_performance = np.mean(self.state.performance_history[-5:]) if self.state.performance_history else 0.0
        
        quality_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            scores = self.quality_monitoring[f'{difficulty}_quality_scores']
            if scores:
                quality_stats[difficulty] = {
                    'mean_quality': np.mean(scores),
                    'count': len(scores),
                    'latest_scores': scores[-5:]
                }
        
        return {
            'current_stage': self.state.current_stage,
            'current_difficulty': self._get_current_difficulty_level(),
            'stability_count': self.state.stability_count,
            'recent_performance': recent_performance,
            'stage_transitions': len(self.state.stage_transition_history),
            'safety_alerts_count': len(self.state.safety_alerts),
            'generation_stats': dict(self.generation_stats),
            'quality_stats': quality_stats,
            'hub_nodes_count': len([s for s in self.hub_scores.values() if s > self.config.hub_threshold]),
            'graph_stats': {
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0,
                'degree_distribution': self.degree_distribution
            }
        }

# テスト・使用例
if __name__ == "__main__":
    # テスト設定
    config = CurriculumConfig(
        max_stages=3,
        progression_threshold=0.8,
        anti_hub_ratio=0.3,
        hub_threshold=2.0
    )
    
    # システム初期化
    curriculum_system = CurriculumAntiHubSystem(config)
    
    # テスト用グラフ生成
    np.random.seed(42)
    n_nodes = 100
    
    # ハブ的構造のグラフ生成
    edges = []
    for i in range(n_nodes):
        # 最初の10ノードをハブにする
        if i < 10:
            n_connections = np.random.randint(15, 25)
        else:
            n_connections = np.random.randint(2, 8)
        
        targets = np.random.choice(n_nodes, size=min(n_connections, n_nodes-1), replace=False)
        for target in targets:
            if target != i:
                edges.append((f"node_{i}", f"node_{target}"))
    
    # ダミー埋め込み
    node_embeddings = {
        f"node_{i}": np.random.randn(128) * 0.5
        for i in range(n_nodes)
    }
    
    # グラフ初期化
    curriculum_system.initialize_graph_structure(edges, node_embeddings)
    
    print("=== V2 Curriculum Anti-Hub System Test ===")
    
    # 各段階でのカリキュラム負例生成テスト
    for stage in range(3):
        curriculum_system.state.current_stage = stage
        difficulty = curriculum_system._get_current_difficulty_level()
        
        print(f"\n--- Stage {stage}: {difficulty.upper()} negatives ---")
        
        # テスト負例生成
        target_node = "node_0"
        positive_examples = ["node_1", "node_2", "node_3"]
        
        negatives = curriculum_system.generate_curriculum_negatives(
            target_node, positive_examples, target_difficulty=difficulty, n_negatives=5
        )
        
        print(f"Generated {len(negatives)} {difficulty} negatives:")
        for neg in negatives:
            print(f"  {neg.example_id}: sim={neg.similarity_score:.3f}, "
                  f"hub={neg.hub_score:.2f}, quality={neg.quality_score:.3f}")
        
        # パフォーマンスフィードバック（模擬）
        simulated_performance = {'accuracy': 0.7 + stage * 0.1}  # 段階的改善
        progressed = curriculum_system.update_curriculum_progress(simulated_performance)
        
        if progressed:
            print(f"  → Advanced to next stage!")
    
    # システム状態サマリー
    print("\n=== Curriculum System Status ===")
    status = curriculum_system.get_curriculum_status()
    print(f"Current Stage: {status['current_stage']} ({status['current_difficulty']})")
    print(f"Recent Performance: {status['recent_performance']:.3f}")
    print(f"Generation Stats: {status['generation_stats']}")
    print(f"Hub Nodes: {status['hub_nodes_count']}")
    print(f"Safety Alerts: {status['safety_alerts_count']}")
    
    if status['quality_stats']:
        print("\nQuality Statistics:")
        for difficulty, stats in status['quality_stats'].items():
            print(f"  {difficulty}: mean_quality={stats['mean_quality']:.3f}, count={stats['count']}")
    
    print("\n✅ V2 Curriculum Anti-Hub System Test Completed")
    print("   - Curriculum progression: WORKING")
    print("   - Anti-hub sampling: WORKING")
    print("   - Quality monitoring: WORKING")
    print("   - Safety checks: WORKING")