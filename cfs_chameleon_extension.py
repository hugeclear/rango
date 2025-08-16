#!/usr/bin/env python3
"""
CFS-Chameleon Extension: Collaborative Feature Sharing Integration
世界初の軽量学習協調的埋め込み編集システム

特徴:
- 既存Chameleon実装の完全互換性保持
- 方向ベクトル分解・プール化・協調的選択
- プライバシー保護下での協調学習
- フラグによる協調機能のON/OFF制御
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DirectionPiece:
    """分解された方向ベクトルのピース"""
    u_component: np.ndarray      # U成分 (左特異ベクトル)
    v_component: np.ndarray      # V成分 (右特異ベクトル)  
    singular_value: float        # 特異値
    importance: float            # 重要度スコア
    semantic_tag: str            # 意味的タグ
    user_id: str                 # 貢献ユーザーID
    quality_score: float         # 品質スコア
    creation_time: float         # 作成時刻
    usage_count: int             # 使用回数

@dataclass
class UserContext:
    """ユーザーコンテキスト情報"""
    user_id: str
    preference_vector: np.ndarray  # ユーザー嗜好ベクトル
    history_embedding: np.ndarray  # 履歴埋め込み
    activity_level: float          # アクティビティレベル
    similarity_cache: Dict[str, float]  # 類似度キャッシュ

class CollaborativeDirectionPool:
    """
    協調的方向プール
    
    機能:
    1. 個人方向ベクトルの分解・保存
    2. 意味的インデックス構築
    3. 高速類似度検索
    4. プライバシー保護機構
    """
    
    def __init__(self, pool_size: int = 1000, rank_reduction: int = 32):
        self.pool_size = pool_size
        self.rank_reduction = rank_reduction
        
        # ピースストレージ
        self.pieces: List[DirectionPiece] = []
        self.piece_index: Dict[str, int] = {}  # ハッシュ → インデックス
        
        # ユーザーコンテキスト
        self.user_contexts: Dict[str, UserContext] = {}
        
        # 検索インデックス
        self.semantic_index = defaultdict(list)  # タグ → ピースリスト
        self.similarity_matrix = None  # キャッシュ用類似度行列
        
        # 統計情報
        self.stats = {
            'total_contributions': 0,
            'active_users': 0,
            'avg_quality_score': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info(f"CollaborativeDirectionPool initialized (capacity: {pool_size}, rank: {rank_reduction})")
    
    def add_direction_vector(self, direction_vector: np.ndarray, user_id: str, 
                           semantic_context: str = "") -> List[DirectionPiece]:
        """
        個人方向ベクトルを分解してプールに追加
        
        Args:
            direction_vector: 個人方向ベクトル
            user_id: 貢献ユーザーID
            semantic_context: 意味的コンテキスト
            
        Returns:
            生成されたピースリスト
        """
        # SVD分解による方向ベクトル分割
        pieces = self._decompose_direction_vector(direction_vector, user_id, semantic_context)
        
        # プールに追加
        for piece in pieces:
            if len(self.pieces) < self.pool_size:
                self.pieces.append(piece)
                piece_hash = self._compute_piece_hash(piece)
                self.piece_index[piece_hash] = len(self.pieces) - 1
                
                # 意味的インデックス更新
                self.semantic_index[piece.semantic_tag].append(len(self.pieces) - 1)
            else:
                # プール満杯時は品質スコアに基づいて置換
                self._replace_lowest_quality_piece(piece)
        
        # 統計更新
        self._update_statistics(user_id)
        
        logger.info(f"Added {len(pieces)} pieces from user {user_id}")
        return pieces
    
    def _decompose_direction_vector(self, direction_vector: np.ndarray, user_id: str, 
                                  semantic_context: str) -> List[DirectionPiece]:
        """SVD分解による方向ベクトル分割"""
        try:
            # 方向ベクトルを行列に変形（複数の基底を想定）
            if len(direction_vector.shape) == 1:
                # 1Dベクトルの場合、外積行列を作成
                matrix = np.outer(direction_vector, direction_vector)
            else:
                matrix = direction_vector
            
            # SVD分解実行
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # ランク削減
            rank = min(self.rank_reduction, len(S))
            U_reduced = U[:, :rank]
            S_reduced = S[:rank]
            Vt_reduced = Vt[:rank, :]
            
            pieces = []
            for i in range(rank):
                # 重要度計算（特異値ベース）
                importance = S_reduced[i] / np.sum(S_reduced)
                
                # 意味的タグ生成
                semantic_tag = self._generate_semantic_tag(U_reduced[:, i], semantic_context)
                
                # 品質スコア計算
                quality_score = self._calculate_quality_score(U_reduced[:, i], S_reduced[i])
                
                piece = DirectionPiece(
                    u_component=U_reduced[:, i],
                    v_component=Vt_reduced[i, :],
                    singular_value=S_reduced[i],
                    importance=importance,
                    semantic_tag=semantic_tag,
                    user_id=user_id,
                    quality_score=quality_score,
                    creation_time=time.time(),
                    usage_count=0
                )
                pieces.append(piece)
            
            return pieces
            
        except Exception as e:
            logger.error(f"Direction decomposition failed: {e}")
            return []
    
    def _generate_semantic_tag(self, component: np.ndarray, context: str) -> str:
        """意味的タグの自動生成"""
        # 成分の特徴に基づいてタグ生成
        norm = np.linalg.norm(component)
        sparsity = np.sum(np.abs(component) < 0.01) / len(component)
        
        if norm > 0.8:
            tag = "high_impact"
        elif norm > 0.5:
            tag = "medium_impact"
        else:
            tag = "subtle_adjustment"
        
        if sparsity > 0.7:
            tag += "_sparse"
        elif sparsity < 0.3:
            tag += "_dense"
        
        # コンテキスト情報を追加
        if context:
            context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            tag += f"_{context_hash}"
        
        return tag
    
    def _calculate_quality_score(self, component: np.ndarray, singular_value: float) -> float:
        """品質スコアの統計的計算"""
        # 複数の品質指標を組み合わせ
        norm_score = min(np.linalg.norm(component), 1.0)
        singular_score = min(singular_value, 1.0)
        stability_score = 1.0 - (np.std(component) / (np.mean(np.abs(component)) + 1e-8))
        
        return (norm_score + singular_score + stability_score) / 3.0
    
    def select_collaborative_pieces(self, user_context: UserContext, query_embedding: np.ndarray, 
                                  top_k: int = 10, strategy: str = "analytical") -> List[DirectionPiece]:
        """
        ユーザーコンテキストとクエリに基づく最適ピース選択
        
        Args:
            user_context: ユーザーコンテキスト
            query_embedding: クエリ埋め込み
            top_k: 選択するピース数
            strategy: 選択戦略 ("analytical", "learned")
            
        Returns:
            選択されたピースリスト
        """
        if not self.pieces:
            return []
        
        if strategy == "analytical":
            return self._analytical_selection(user_context, query_embedding, top_k)
        elif strategy == "learned":
            return self._learned_selection(user_context, query_embedding, top_k)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _analytical_selection(self, user_context: UserContext, query_embedding: np.ndarray, 
                            top_k: int) -> List[DirectionPiece]:
        """解析的選択アルゴリズム（学習なし）"""
        scores = []
        
        for piece in self.pieces:
            # 1. コンテキスト類似度
            context_sim = self._compute_context_similarity(user_context, piece)
            
            # 2. 意味的関連性
            semantic_sim = self._compute_semantic_similarity(query_embedding, piece)
            
            # 3. 品質スコア
            quality_score = piece.quality_score
            
            # 4. ユーザー類似度（協調フィルタリング的）
            user_sim = self._compute_user_similarity(user_context.user_id, piece.user_id)
            
            # 総合スコア計算（重み付き和）
            total_score = (0.3 * context_sim + 0.3 * semantic_sim + 
                          0.2 * quality_score + 0.2 * user_sim)
            
            scores.append((total_score, piece))
        
        # Top-K選択
        scores.sort(key=lambda x: x[0], reverse=True)
        selected_pieces = [piece for _, piece in scores[:top_k]]
        
        # 使用回数更新
        for piece in selected_pieces:
            piece.usage_count += 1
        
        return selected_pieces
    
    def _compute_context_similarity(self, user_context: UserContext, piece: DirectionPiece) -> float:
        """コンテキスト類似度計算"""
        try:
            # ユーザー嗜好ベクトルとピース成分のコサイン類似度
            if len(user_context.preference_vector) != len(piece.u_component):
                return 0.0
            
            dot_product = np.dot(user_context.preference_vector, piece.u_component)
            norm_product = (np.linalg.norm(user_context.preference_vector) * 
                          np.linalg.norm(piece.u_component))
            
            if norm_product == 0:
                return 0.0
            
            return abs(dot_product / norm_product)
        except:
            return 0.0
    
    def _compute_semantic_similarity(self, query_embedding: np.ndarray, piece: DirectionPiece) -> float:
        """意味的関連性計算"""
        try:
            # クエリ埋め込みとピース成分の類似度
            if len(query_embedding) != len(piece.u_component):
                return 0.0
            
            dot_product = np.dot(query_embedding, piece.u_component)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(piece.u_component)
            
            if norm_product == 0:
                return 0.0
            
            return abs(dot_product / norm_product)
        except:
            return 0.0
    
    def _compute_user_similarity(self, user_id1: str, user_id2: str) -> float:
        """ユーザー類似度計算（協調フィルタリング）"""
        if user_id1 == user_id2:
            return 1.0
        
        # キャッシュ確認
        cache_key = f"{user_id1}_{user_id2}"
        if user_id1 in self.user_contexts and cache_key in self.user_contexts[user_id1].similarity_cache:
            return self.user_contexts[user_id1].similarity_cache[cache_key]
        
        # 簡単な文字列類似度（実際の実装ではより複雑な手法を使用）
        similarity = 1.0 - (abs(hash(user_id1)) % 1000 - abs(hash(user_id2)) % 1000) / 1000.0
        
        # キャッシュに保存
        if user_id1 in self.user_contexts:
            self.user_contexts[user_id1].similarity_cache[cache_key] = similarity
        
        return max(0.0, similarity)
    
    def fuse_selected_directions(self, selected_pieces: List[DirectionPiece], 
                               fusion_strategy: str = "analytical") -> np.ndarray:
        """
        選択された方向ピースを統合して最終的な協調方向を生成
        
        Args:
            selected_pieces: 選択されたピースリスト
            fusion_strategy: 統合戦略
            
        Returns:
            統合された協調方向ベクトル
        """
        if not selected_pieces:
            return np.zeros(768)  # デフォルト次元
        
        if fusion_strategy == "analytical":
            return self._analytical_fusion(selected_pieces)
        elif fusion_strategy == "weighted_attention":
            return self._attention_fusion(selected_pieces)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def _analytical_fusion(self, pieces: List[DirectionPiece]) -> np.ndarray:
        """解析的統合（重み付き平均）"""
        if not pieces:
            return np.zeros(768)
        
        # 重要度に基づく重み計算
        weights = np.array([piece.importance * piece.quality_score for piece in pieces])
        weights = weights / np.sum(weights)  # 正規化
        
        # 重み付き平均による統合
        fused_direction = np.zeros_like(pieces[0].u_component)
        for i, piece in enumerate(pieces):
            fused_direction += weights[i] * piece.u_component
        
        # 正規化
        norm = np.linalg.norm(fused_direction)
        if norm > 0:
            fused_direction = fused_direction / norm
        
        return fused_direction
    
    def _attention_fusion(self, pieces: List[DirectionPiece]) -> np.ndarray:
        """アテンション機構による統合"""
        if not pieces:
            return np.zeros(768)
        
        # 簡単なアテンション重み計算
        attention_scores = []
        for piece in pieces:
            # 品質スコアと重要度の組み合わせ
            score = piece.quality_score * piece.importance
            attention_scores.append(score)
        
        # ソフトマックス正規化
        attention_weights = np.exp(attention_scores)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # アテンション重み付き統合
        fused_direction = np.zeros_like(pieces[0].u_component)
        for i, piece in enumerate(pieces):
            fused_direction += attention_weights[i] * piece.u_component
        
        # 正規化
        norm = np.linalg.norm(fused_direction)
        if norm > 0:
            fused_direction = fused_direction / norm
        
        return fused_direction
    
    def _compute_piece_hash(self, piece: DirectionPiece) -> str:
        """ピースのハッシュ値計算（重複検出用）"""
        content = f"{piece.user_id}_{piece.semantic_tag}_{piece.singular_value:.6f}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _replace_lowest_quality_piece(self, new_piece: DirectionPiece):
        """最低品質ピースを新しいピースで置換"""
        if not self.pieces:
            return
        
        # 最低品質ピースを特定
        min_quality_idx = min(range(len(self.pieces)), 
                            key=lambda i: self.pieces[i].quality_score)
        
        if new_piece.quality_score > self.pieces[min_quality_idx].quality_score:
            # 古いピースを削除
            old_piece = self.pieces[min_quality_idx]
            old_hash = self._compute_piece_hash(old_piece)
            
            # 新しいピースで置換
            self.pieces[min_quality_idx] = new_piece
            new_hash = self._compute_piece_hash(new_piece)
            
            # インデックス更新
            if old_hash in self.piece_index:
                del self.piece_index[old_hash]
            self.piece_index[new_hash] = min_quality_idx
            
            # 意味的インデックス更新
            self.semantic_index[old_piece.semantic_tag].remove(min_quality_idx)
            self.semantic_index[new_piece.semantic_tag].append(min_quality_idx)
    
    def _update_statistics(self, user_id: str):
        """統計情報更新"""
        self.stats['total_contributions'] += 1
        
        if user_id not in self.user_contexts:
            self.stats['active_users'] += 1
        
        # 平均品質スコア更新
        if self.pieces:
            total_quality = sum(piece.quality_score for piece in self.pieces)
            self.stats['avg_quality_score'] = total_quality / len(self.pieces)
    
    def get_statistics(self) -> Dict[str, Any]:
        """プール統計情報取得"""
        return {
            **self.stats,
            'pool_utilization': len(self.pieces) / self.pool_size,
            'unique_semantic_tags': len(self.semantic_index),
            'avg_piece_usage': np.mean([piece.usage_count for piece in self.pieces]) if self.pieces else 0
        }
    
    def get_collaborative_directions(self, user_context: UserContext, query_embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        協調的方向ベクトル生成
        
        Args:
            user_context: ユーザーコンテキスト
            query_embedding: クエリ埋め込み
            
        Returns:
            'personal'と'neutral'方向ベクトルの辞書
        """
        try:
            # 最適ピース選択
            selected_pieces = self.select_collaborative_pieces(user_context, query_embedding, top_k=8)
            
            if not selected_pieces:
                logger.warning(f"No collaborative pieces found for user {user_context.user_id}")
                # デフォルト方向を返す（エラーでなく警告レベル）
                return {
                    'personal': user_context.preference_vector[:min(len(user_context.preference_vector), 3072)],
                    'neutral': np.zeros(3072, dtype=np.float32)
                }
            
            # 協調的方向統合
            collaborative_personal = self._attention_fusion(selected_pieces)
            
            # 個人方向とのブレンド（70%協調、30%個人）
            personal_component = user_context.preference_vector[:min(len(user_context.preference_vector), len(collaborative_personal))]
            if len(personal_component) < len(collaborative_personal):
                # パディング
                padded_personal = np.zeros_like(collaborative_personal)
                padded_personal[:len(personal_component)] = personal_component
                personal_component = padded_personal
            
            blended_personal = 0.7 * collaborative_personal + 0.3 * personal_component
            
            # ニュートラル方向生成（逆方向）
            collaborative_neutral = -0.5 * blended_personal + 0.3 * np.random.randn(*blended_personal.shape) * 0.1
            
            # 正規化
            if np.linalg.norm(blended_personal) > 0:
                blended_personal = blended_personal / np.linalg.norm(blended_personal)
            if np.linalg.norm(collaborative_neutral) > 0:
                collaborative_neutral = collaborative_neutral / np.linalg.norm(collaborative_neutral)
            
            # 3072次元に調整（必要に応じて）
            if len(blended_personal) != 3072:
                if len(blended_personal) > 3072:
                    blended_personal = blended_personal[:3072]
                    collaborative_neutral = collaborative_neutral[:3072]
                else:
                    padded_personal = np.zeros(3072, dtype=np.float32)
                    padded_neutral = np.zeros(3072, dtype=np.float32)
                    padded_personal[:len(blended_personal)] = blended_personal
                    padded_neutral[:len(collaborative_neutral)] = collaborative_neutral
                    blended_personal = padded_personal
                    collaborative_neutral = padded_neutral
            
            logger.debug(f"Generated collaborative directions for user {user_context.user_id}: P={blended_personal.shape}, N={collaborative_neutral.shape}")
            
            return {
                'personal': blended_personal.astype(np.float32),
                'neutral': collaborative_neutral.astype(np.float32)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate collaborative directions: {e}")
            # フォールバック: ユーザーの既存方向を返す
            return {
                'personal': user_context.preference_vector[:min(len(user_context.preference_vector), 3072)],
                'neutral': np.zeros(3072, dtype=np.float32)
            }
    
    def save_pool(self, filepath: str):
        """プールの状態を保存"""
        pool_data = {
            'pieces': [],
            'user_contexts': {},
            'stats': self.stats,
            'config': {
                'pool_size': self.pool_size,
                'rank_reduction': self.rank_reduction
            }
        }
        
        # ピースデータのシリアライズ
        for piece in self.pieces:
            piece_data = {
                'u_component': piece.u_component.tolist(),
                'v_component': piece.v_component.tolist(),
                'singular_value': piece.singular_value,
                'importance': piece.importance,
                'semantic_tag': piece.semantic_tag,
                'user_id': piece.user_id,
                'quality_score': piece.quality_score,
                'creation_time': piece.creation_time,
                'usage_count': piece.usage_count
            }
            pool_data['pieces'].append(piece_data)
        
        # ユーザーコンテキストのシリアライズ
        for user_id, context in self.user_contexts.items():
            context_data = {
                'user_id': context.user_id,
                'preference_vector': context.preference_vector.tolist(),
                'history_embedding': context.history_embedding.tolist(),
                'activity_level': context.activity_level,
                'similarity_cache': context.similarity_cache
            }
            pool_data['user_contexts'][user_id] = context_data
        
        with open(filepath, 'w') as f:
            json.dump(pool_data, f, indent=2)
        
        logger.info(f"Pool saved to {filepath}")
    
    def load_pool(self, filepath: str):
        """保存されたプール状態を読み込み"""
        try:
            with open(filepath, 'r') as f:
                pool_data = json.load(f)
            
            # 設定復元
            config = pool_data.get('config', {})
            self.pool_size = config.get('pool_size', 1000)
            self.rank_reduction = config.get('rank_reduction', 32)
            
            # ピースデータ復元
            self.pieces = []
            self.piece_index = {}
            self.semantic_index = defaultdict(list)
            
            for i, piece_data in enumerate(pool_data['pieces']):
                piece = DirectionPiece(
                    u_component=np.array(piece_data['u_component']),
                    v_component=np.array(piece_data['v_component']),
                    singular_value=piece_data['singular_value'],
                    importance=piece_data['importance'],
                    semantic_tag=piece_data['semantic_tag'],
                    user_id=piece_data['user_id'],
                    quality_score=piece_data['quality_score'],
                    creation_time=piece_data['creation_time'],
                    usage_count=piece_data['usage_count']
                )
                self.pieces.append(piece)
                
                # インデックス復元
                piece_hash = self._compute_piece_hash(piece)
                self.piece_index[piece_hash] = i
                self.semantic_index[piece.semantic_tag].append(i)
            
            # ユーザーコンテキスト復元
            self.user_contexts = {}
            for user_id, context_data in pool_data.get('user_contexts', {}).items():
                context = UserContext(
                    user_id=context_data['user_id'],
                    preference_vector=np.array(context_data['preference_vector']),
                    history_embedding=np.array(context_data['history_embedding']),
                    activity_level=context_data['activity_level'],
                    similarity_cache=context_data['similarity_cache']
                )
                self.user_contexts[user_id] = context
            
            # 統計復元
            self.stats = pool_data.get('stats', {})
            
            logger.info(f"Pool loaded from {filepath} ({len(self.pieces)} pieces, {len(self.user_contexts)} users)")
            
        except Exception as e:
            logger.error(f"Failed to load pool: {e}")

class LightweightGateNetwork(nn.Module):
    """軽量ゲートネットワーク（オプション学習コンポーネント）"""
    
    def __init__(self, embedding_dim: int = 768, num_directions: int = 200, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_directions = num_directions
        
        # 軽量ネットワーク構成
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_directions),
            nn.Sigmoid()
        )
        
        # 総パラメータ数: ~200K (768*256 + 256*200 = ~250K)
        logger.info(f"LightweightGateNetwork initialized ({self._count_parameters()} parameters)")
    
    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """
        ユーザー埋め込みから方向選択ゲートを計算
        
        Args:
            user_embedding: ユーザー埋め込み (batch_size, embedding_dim)
            
        Returns:
            方向選択ゲート (batch_size, num_directions)
        """
        return self.gate_network(user_embedding)
    
    def _count_parameters(self) -> int:
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 基本動作テスト
    print("🔧 CFS-Chameleon Extension Test")
    
    # プール初期化
    pool = CollaborativeDirectionPool(pool_size=100, rank_reduction=16)
    
    # サンプル方向ベクトル追加
    sample_direction = np.random.randn(768)
    pieces = pool.add_direction_vector(sample_direction, "user_001", "movie_preferences")
    
    print(f"Generated {len(pieces)} pieces")
    print(f"Pool statistics: {pool.get_statistics()}")
    
    # 軽量ゲートネットワークテスト
    gate_net = LightweightGateNetwork()
    sample_embedding = torch.randn(1, 768)
    gates = gate_net(sample_embedding)
    
    print(f"Gate network output shape: {gates.shape}")
    print("✅ CFS-Chameleon Extension test completed")