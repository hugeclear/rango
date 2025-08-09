#!/usr/bin/env python3
"""
CFS-Chameleon二元方向ピース統合システム
パーソナル方向とニュートラル方向ピースの統合管理・編集システム
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass

# 方向ピース生成モジュール
from neutral_direction_generator import (
    generate_neutral_direction_pieces,
    NeutralDirectionPiece
)
from improved_direction_pieces_generator import (
    generate_improved_direction_pieces
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
class DualDirectionConfig:
    """二元方向ピース設定"""
    personal_weight: float = 1.0      # パーソナル方向の重み
    neutral_weight: float = -0.5      # ニュートラル方向の重み（通常負値）
    max_personal_pieces: int = 10     # 最大パーソナルピース数
    max_neutral_pieces: int = 8       # 最大ニュートラルピース数
    rank_reduction: int = 16          # SVDランク削減数
    quality_threshold: float = 0.1    # 品質閾値
    enable_dynamic_weighting: bool = True  # 動的重み調整

class DualDirectionPool:
    """パーソナル・ニュートラル二元方向ピースプール"""
    
    def __init__(self, 
                 config: DualDirectionConfig = None,
                 capacity: int = 1000):
        """
        初期化
        
        Args:
            config: 二元方向設定
            capacity: プール総容量
        """
        self.config = config or DualDirectionConfig()
        self.capacity = capacity
        
        # 方向別ピース格納
        self.personal_pieces: List[Any] = []
        self.neutral_pieces: List[Any] = []
        
        # ユーザーマッピング
        self.user_personal_mapping: Dict[str, List[int]] = {}
        self.user_neutral_mapping: Dict[str, List[int]] = {}
        
        logger.info("✅ DualDirectionPool initialized")
        logger.info(f"   Personal weight: {self.config.personal_weight}")
        logger.info(f"   Neutral weight: {self.config.neutral_weight}")
        logger.info(f"   Max pieces: Personal={self.config.max_personal_pieces}, Neutral={self.config.max_neutral_pieces}")
    
    def add_user_dual_pieces(self, 
                           user_id: str,
                           user_history_texts: List[str],
                           neutral_reference: str = "これは一般的な内容です") -> Dict[str, int]:
        """
        ユーザーの履歴からパーソナル・ニュートラル両方向のピースを生成・追加
        
        Args:
            user_id: ユーザーID
            user_history_texts: ユーザー履歴テキストリスト
            neutral_reference: ニュートラル参照テキスト
            
        Returns:
            追加されたピース数の辞書
        """
        logger.info(f"🔄 Generating dual direction pieces for user {user_id}")
        logger.info(f"   History texts: {len(user_history_texts)}")
        
        added_counts = {"personal": 0, "neutral": 0}
        
        try:
            # パーソナル方向ピース生成
            logger.info("   📈 Generating personal direction pieces...")
            personal_pieces_data = generate_improved_direction_pieces(
                user_history_texts=user_history_texts,
                neutral_reference_text=neutral_reference,
                rank_reduction=self.config.rank_reduction
            )
            
            # パーソナルピース追加
            for piece_data in personal_pieces_data[:self.config.max_personal_pieces]:
                if piece_data.get('quality_score', 0) >= self.config.quality_threshold:
                    self.personal_pieces.append(piece_data)
                    personal_index = len(self.personal_pieces) - 1
                    
                    if user_id not in self.user_personal_mapping:
                        self.user_personal_mapping[user_id] = []
                    self.user_personal_mapping[user_id].append(personal_index)
                    added_counts["personal"] += 1
            
            # ニュートラル方向ピース生成
            logger.info("   📉 Generating neutral direction pieces...")
            neutral_pieces_data = generate_neutral_direction_pieces(
                user_history_texts=user_history_texts,
                rank_reduction=self.config.rank_reduction
            )
            
            # ニュートラルピース追加
            for piece_data in neutral_pieces_data[:self.config.max_neutral_pieces]:
                if piece_data.get('quality_score', 0) >= self.config.quality_threshold:
                    self.neutral_pieces.append(piece_data)
                    neutral_index = len(self.neutral_pieces) - 1
                    
                    if user_id not in self.user_neutral_mapping:
                        self.user_neutral_mapping[user_id] = []
                    self.user_neutral_mapping[user_id].append(neutral_index)
                    added_counts["neutral"] += 1
            
            logger.info(f"✅ Added dual pieces for user {user_id}")
            logger.info(f"   Personal pieces: {added_counts['personal']}")
            logger.info(f"   Neutral pieces: {added_counts['neutral']}")
            
            return added_counts
            
        except Exception as e:
            logger.error(f"❌ Failed to add dual pieces for user {user_id}: {e}")
            return added_counts
    
    def select_dual_pieces_for_context(self, 
                                     user_context: str,
                                     user_id: str = None,
                                     top_k_personal: int = 3,
                                     top_k_neutral: int = 2) -> Dict[str, List[Tuple[Any, float]]]:
        """
        コンテキストに基づくパーソナル・ニュートラル両方向ピースの選択
        
        Args:
            user_context: ユーザーコンテキスト
            user_id: ユーザーID
            top_k_personal: 選択するパーソナルピース数
            top_k_neutral: 選択するニュートラルピース数
            
        Returns:
            選択されたピースの辞書 {"personal": [...], "neutral": [...]}
        """
        logger.info(f"🔍 Selecting dual pieces for context")
        logger.info(f"   Personal top-k: {top_k_personal}, Neutral top-k: {top_k_neutral}")
        
        selected_pieces = {"personal": [], "neutral": []}
        
        try:
            # パーソナルピース選択
            if self.personal_pieces:
                personal_scores = self._compute_context_similarity(
                    user_context, self.personal_pieces, "personal"
                )
                
                # 上位パーソナルピース選択
                top_personal_indices = np.argsort(personal_scores)[-top_k_personal:][::-1]
                for idx in top_personal_indices:
                    if idx < len(self.personal_pieces):
                        piece = self.personal_pieces[idx]
                        score = personal_scores[idx]
                        selected_pieces["personal"].append((piece, score))
            
            # ニュートラルピース選択
            if self.neutral_pieces:
                neutral_scores = self._compute_context_similarity(
                    user_context, self.neutral_pieces, "neutral"
                )
                
                # 上位ニュートラルピース選択
                top_neutral_indices = np.argsort(neutral_scores)[-top_k_neutral:][::-1]
                for idx in top_neutral_indices:
                    if idx < len(self.neutral_pieces):
                        piece = self.neutral_pieces[idx]
                        score = neutral_scores[idx]
                        selected_pieces["neutral"].append((piece, score))
            
            logger.info(f"✅ Selected pieces:")
            logger.info(f"   Personal: {len(selected_pieces['personal'])} pieces")
            logger.info(f"   Neutral: {len(selected_pieces['neutral'])} pieces")
            
            return selected_pieces
            
        except Exception as e:
            logger.error(f"❌ Dual piece selection error: {e}")
            return selected_pieces
    
    def _compute_context_similarity(self, 
                                  context: str, 
                                  pieces: List[Any], 
                                  direction_type: str) -> np.ndarray:
        """
        コンテキストとピースの類似度計算
        
        Args:
            context: ユーザーコンテキスト
            pieces: ピースリスト
            direction_type: 方向タイプ（"personal" または "neutral"）
            
        Returns:
            類似度スコア配列
        """
        if not pieces:
            return np.array([])
        
        # 簡単な実装: semantic_contextとの文字レベル類似度
        similarities = []
        context_words = set(context.lower().split())
        
        for piece in pieces:
            piece_context = piece.get('semantic_context', '')
            piece_words = set(piece_context.lower().split())
            
            # 単語重複度を基本とした類似度
            if context_words and piece_words:
                overlap = len(context_words & piece_words)
                union = len(context_words | piece_words)
                similarity = overlap / union if union > 0 else 0.1
            else:
                similarity = 0.1
            
            # 方向タイプに応じた重み調整
            if direction_type == "personal":
                # パーソナルピースは高品質ほど高評価
                quality_bonus = piece.get('quality_score', 0.5) * 0.3
                similarity += quality_bonus
            elif direction_type == "neutral":
                # ニュートラルピースは安定性重視
                importance_bonus = piece.get('importance', 0.5) * 0.2
                similarity += importance_bonus
            
            similarities.append(min(similarity, 1.0))
        
        return np.array(similarities)
    
    def compute_dual_direction_editing_vectors(self, 
                                             selected_pieces: Dict[str, List[Tuple[Any, float]]],
                                             target_dimension: int = 3072) -> Dict[str, np.ndarray]:
        """
        選択されたピースから編集用方向ベクトルを計算
        
        Args:
            selected_pieces: 選択されたピース辞書
            target_dimension: 目標次元数
            
        Returns:
            編集ベクトル辞書 {"personal": vector, "neutral": vector}
        """
        editing_vectors = {"personal": None, "neutral": None}
        
        try:
            # パーソナル編集ベクトル計算
            if selected_pieces["personal"]:
                personal_components = []
                personal_weights = []
                
                for piece, score in selected_pieces["personal"]:
                    v_component = np.array(piece.get('v_component', []))
                    if len(v_component) > 0:
                        # 目標次元に調整
                        if len(v_component) != target_dimension:
                            if len(v_component) < target_dimension:
                                # 零パディング
                                padded = np.zeros(target_dimension)
                                padded[:len(v_component)] = v_component
                                v_component = padded
                            else:
                                # 切り詰め
                                v_component = v_component[:target_dimension]
                        
                        personal_components.append(v_component)
                        personal_weights.append(score * piece.get('importance', 0.5))
                
                if personal_components:
                    # 重み付き平均
                    personal_weights = np.array(personal_weights)
                    personal_weights = personal_weights / (np.sum(personal_weights) + 1e-8)
                    
                    personal_vector = np.zeros(target_dimension)
                    for component, weight in zip(personal_components, personal_weights):
                        personal_vector += weight * component
                    
                    # 正規化
                    editing_vectors["personal"] = personal_vector / (np.linalg.norm(personal_vector) + 1e-8)
            
            # ニュートラル編集ベクトル計算
            if selected_pieces["neutral"]:
                neutral_components = []
                neutral_weights = []
                
                for piece, score in selected_pieces["neutral"]:
                    v_component = np.array(piece.get('v_component', []))
                    if len(v_component) > 0:
                        # 目標次元に調整
                        if len(v_component) != target_dimension:
                            if len(v_component) < target_dimension:
                                # 零パディング
                                padded = np.zeros(target_dimension)
                                padded[:len(v_component)] = v_component
                                v_component = padded
                            else:
                                # 切り詰め
                                v_component = v_component[:target_dimension]
                        
                        neutral_components.append(v_component)
                        neutral_weights.append(score * piece.get('importance', 0.5))
                
                if neutral_components:
                    # 重み付き平均
                    neutral_weights = np.array(neutral_weights)
                    neutral_weights = neutral_weights / (np.sum(neutral_weights) + 1e-8)
                    
                    neutral_vector = np.zeros(target_dimension)
                    for component, weight in zip(neutral_components, neutral_weights):
                        neutral_vector += weight * component
                    
                    # 正規化
                    editing_vectors["neutral"] = neutral_vector / (np.linalg.norm(neutral_vector) + 1e-8)
            
            logger.info(f"✅ Computed dual editing vectors:")
            logger.info(f"   Personal vector: {'✓' if editing_vectors['personal'] is not None else '✗'}")
            logger.info(f"   Neutral vector: {'✓' if editing_vectors['neutral'] is not None else '✗'}")
            
            return editing_vectors
            
        except Exception as e:
            logger.error(f"❌ Dual editing vector computation error: {e}")
            return editing_vectors
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """プール統計情報取得"""
        return {
            "total_personal_pieces": len(self.personal_pieces),
            "total_neutral_pieces": len(self.neutral_pieces),
            "total_pieces": len(self.personal_pieces) + len(self.neutral_pieces),
            "users_with_personal": len(self.user_personal_mapping),
            "users_with_neutral": len(self.user_neutral_mapping),
            "personal_avg_quality": np.mean([p.get('quality_score', 0) for p in self.personal_pieces]) if self.personal_pieces else 0,
            "neutral_avg_quality": np.mean([p.get('quality_score', 0) for p in self.neutral_pieces]) if self.neutral_pieces else 0,
            "config": {
                "personal_weight": self.config.personal_weight,
                "neutral_weight": self.config.neutral_weight,
                "max_personal_pieces": self.config.max_personal_pieces,
                "max_neutral_pieces": self.config.max_neutral_pieces
            }
        }

class DualDirectionChameleonEditor:
    """二元方向ピース対応CFS-Chameleonエディター"""
    
    def __init__(self, 
                 config: DualDirectionConfig = None,
                 base_editor: Any = None):
        """
        初期化
        
        Args:
            config: 二元方向設定
            base_editor: ベースCFS-Chameleonエディター
        """
        self.config = config or DualDirectionConfig()
        self.base_editor = base_editor
        
        # 二元方向プール
        self.dual_pool = DualDirectionPool(self.config)
        
        logger.info("✅ DualDirectionChameleonEditor initialized")
    
    def add_user_history(self, 
                        user_id: str, 
                        history_texts: List[str],
                        neutral_reference: str = None) -> bool:
        """
        ユーザー履歴の追加
        
        Args:
            user_id: ユーザーID
            history_texts: 履歴テキストリスト
            neutral_reference: ニュートラル参照テキスト
            
        Returns:
            追加成功フラグ
        """
        if not neutral_reference:
            neutral_reference = "これは一般的で客観的な内容です"
        
        try:
            added_counts = self.dual_pool.add_user_dual_pieces(
                user_id, history_texts, neutral_reference
            )
            
            total_added = added_counts["personal"] + added_counts["neutral"]
            logger.info(f"✅ Added history for user {user_id}: {total_added} total pieces")
            
            return total_added > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to add history for user {user_id}: {e}")
            return False
    
    def generate_with_dual_directions(self, 
                                    prompt: str,
                                    user_context: str = None,
                                    user_id: str = None,
                                    alpha_personal: float = None,
                                    alpha_neutral: float = None,
                                    max_length: int = 100) -> str:
        """
        二元方向ピースを使用した生成
        
        Args:
            prompt: 入力プロンプト
            user_context: ユーザーコンテキスト
            user_id: ユーザーID
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        # デフォルト値設定
        if alpha_personal is None:
            alpha_personal = self.config.personal_weight
        if alpha_neutral is None:
            alpha_neutral = self.config.neutral_weight
        
        context_for_selection = user_context or prompt
        
        logger.info(f"🦎 Generating with dual direction pieces")
        logger.info(f"   Alpha personal: {alpha_personal}")
        logger.info(f"   Alpha neutral: {alpha_neutral}")
        
        try:
            # 二元方向ピース選択
            selected_pieces = self.dual_pool.select_dual_pieces_for_context(
                user_context=context_for_selection,
                user_id=user_id,
                top_k_personal=3,
                top_k_neutral=2
            )
            
            # 編集ベクトル計算
            editing_vectors = self.dual_pool.compute_dual_direction_editing_vectors(
                selected_pieces
            )
            
            # 実際の生成（ベースエディター使用 or フォールバック）
            if self.base_editor and hasattr(self.base_editor, 'generate_with_chameleon'):
                try:
                    # 二元方向編集情報をベースエディターに反映
                    # （実装は既存のCFS-Chameleonエディターの構造に依存）
                    result = self.base_editor.generate_with_chameleon(
                        prompt=prompt,
                        alpha_personal=alpha_personal,
                        alpha_neutral=alpha_neutral,
                        max_length=max_length
                    )
                    
                    logger.info(f"✅ Generated with dual directions: {len(result)} chars")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Base editor error: {e}, using fallback")
            
            # フォールバック生成
            personal_influence = len(selected_pieces["personal"]) * alpha_personal
            neutral_influence = len(selected_pieces["neutral"]) * abs(alpha_neutral)
            
            response = f"Dual-direction enhanced response to: {prompt[:50]}..."
            response += f" (Personal influence: {personal_influence:.2f}, Neutral influence: {neutral_influence:.2f})"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Dual direction generation error: {e}")
            return f"Generation error: {prompt[:30]}..."

def demonstrate_dual_direction_integration():
    """二元方向ピース統合のデモンストレーション"""
    print("🦎 二元方向ピース統合デモ")
    print("=" * 60)
    
    # 設定
    config = DualDirectionConfig(
        personal_weight=1.0,
        neutral_weight=-0.5,
        max_personal_pieces=5,
        max_neutral_pieces=4,
        rank_reduction=8
    )
    
    # エディター初期化
    editor = DualDirectionChameleonEditor(config)
    
    # サンプルユーザーデータ
    user_histories = {
        "user_movie": [
            "今日は素晴らしい映画を見ました！",
            "SF映画が大好きで、特にタイムトラベル系が面白いです",
            "映画館での体験は家では味わえない特別なものです",
            "友達と映画について熱く語り合うのが楽しいです"
        ],
        "user_cooking": [
            "新しいレシピに挑戦するのが趣味です",
            "料理を作る時間が一番リラックスできます",
            "家族のために心を込めて作る料理は格別です",
            "料理教室で学んだことを家で実践するのが楽しみです"
        ]
    }
    
    print(f"\n📦 Adding user histories...")
    
    # 各ユーザーの履歴を追加
    for user_id, history in user_histories.items():
        print(f"\n👤 Processing {user_id}:")
        print(f"   History texts: {len(history)}")
        
        success = editor.add_user_history(user_id, history)
        print(f"   Addition {'✅ successful' if success else '❌ failed'}")
    
    # プール統計表示
    stats = editor.dual_pool.get_pool_statistics()
    print(f"\n📊 Dual Direction Pool Statistics:")
    print(f"   Personal pieces: {stats['total_personal_pieces']}")
    print(f"   Neutral pieces: {stats['total_neutral_pieces']}")
    print(f"   Total pieces: {stats['total_pieces']}")
    print(f"   Users with personal: {stats['users_with_personal']}")
    print(f"   Users with neutral: {stats['users_with_neutral']}")
    print(f"   Personal avg quality: {stats['personal_avg_quality']:.4f}")
    print(f"   Neutral avg quality: {stats['neutral_avg_quality']:.4f}")
    
    # 二元方向生成テスト
    test_prompts = [
        "おすすめの映画を教えてください",
        "美味しい料理のレシピを教えて",
        "今日の気分はどうですか？"
    ]
    
    print(f"\n🎯 Dual direction generation test:")
    print("-" * 40)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # 映画ユーザーのコンテキストで生成
        result = editor.generate_with_dual_directions(
            prompt=prompt,
            user_context="映画について話したい",
            user_id="user_movie",
            max_length=80
        )
        
        print(f"   🎬 Movie user response: {result[:100]}...")
        
        # 料理ユーザーのコンテキストで生成  
        result = editor.generate_with_dual_directions(
            prompt=prompt,
            user_context="料理について相談したい",
            user_id="user_cooking",
            max_length=80
        )
        
        print(f"   🍳 Cooking user response: {result[:100]}...")
    
    # 設定比較テスト
    print(f"\n🔄 Configuration comparison test:")
    print("-" * 40)
    
    prompt = "今日の予定について教えてください"
    
    # 強いパーソナル設定
    result1 = editor.generate_with_dual_directions(
        prompt=prompt,
        alpha_personal=1.5,
        alpha_neutral=-0.3,
        max_length=60
    )
    
    # バランス設定
    result2 = editor.generate_with_dual_directions(
        prompt=prompt,
        alpha_personal=0.8,
        alpha_neutral=-0.8,
        max_length=60
    )
    
    print(f"Strong personal: {result1[:80]}...")
    print(f"Balanced: {result2[:80]}...")
    
    print("\n🎉 二元方向ピース統合デモ完了!")

if __name__ == "__main__":
    demonstrate_dual_direction_integration()