#!/usr/bin/env python3
"""
改善版方向ピース生成のCFS-Chameleon統合モジュール
履歴ベースの意味的方向ベクトルをCFSシステムに統合
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from improved_direction_pieces_generator import generate_improved_direction_pieces, DirectionPiece

# 既存CFS-Chameleonモジュールをインポート
try:
    from chameleon_cfs_integrator import CollaborativeChameleonEditor
    from cfs_chameleon_extension import CollaborativeDirectionPool, DirectionPiece as CFSDirectionPiece
    CFS_AVAILABLE = True
except ImportError:
    logger.warning("CFS-Chameleon modules not available, using mock implementations")
    CFS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImprovedCFSChameleonEditor(CollaborativeChameleonEditor if CFS_AVAILABLE else object):
    """
    改善版方向ピース生成を統合したCFS-Chameleonエディター
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "cuda",
                 use_collaboration: bool = True,
                 config_path: str = None,
                 enable_improved_pieces: bool = True):
        """
        改善版CFS-Chameleonエディター初期化
        
        Args:
            model_name: ベースモデル名
            device: 計算デバイス
            use_collaboration: 協調機能使用フラグ
            config_path: 設定ファイルパス
            enable_improved_pieces: 改善版ピース生成使用フラグ
        """
        
        if CFS_AVAILABLE:
            # 既存のCFS-Chameleonエディターを初期化
            super().__init__(
                model_name=model_name,
                device=device,
                use_collaboration=use_collaboration,
                config_path=config_path
            )
        
        self.enable_improved_pieces = enable_improved_pieces
        self.improved_pieces_cache = {}  # ユーザー別改善ピースキャッシュ
        
        logger.info(f"✅ ImprovedCFSChameleonEditor initialized")
        logger.info(f"   Improved pieces: {'enabled' if enable_improved_pieces else 'disabled'}")
        
    def add_user_history_to_pool(self, 
                                user_id: str, 
                                history_texts: List[str],
                                neutral_reference: str = "これは一般的な文章です",
                                rank_reduction: int = 16) -> bool:
        """
        改善版方向ピース生成でユーザー履歴をプールに追加
        
        Args:
            user_id: ユーザーID
            history_texts: ユーザー履歴テキストリスト
            neutral_reference: ニュートラル参照テキスト
            rank_reduction: SVD分解ランク削減数
            
        Returns:
            追加成功かどうか
        """
        if not self.enable_improved_pieces or not self.use_collaboration:
            logger.warning("Improved pieces or collaboration not enabled")
            return False
        
        try:
            logger.info(f"🔄 Generating improved direction pieces for user {user_id}")
            
            # 改善版方向ピース生成
            pieces_data = generate_improved_direction_pieces(
                user_history_texts=history_texts,
                neutral_reference_text=neutral_reference,
                rank_reduction=rank_reduction
            )
            
            if not pieces_data:
                logger.error(f"❌ No pieces generated for user {user_id}")
                return False
            
            # CFS方向プールに追加
            added_count = 0
            for piece_data in pieces_data:
                if self._add_improved_piece_to_pool(user_id, piece_data):
                    added_count += 1
            
            # キャッシュに保存
            self.improved_pieces_cache[user_id] = pieces_data
            
            logger.info(f"✅ Added {added_count}/{len(pieces_data)} improved pieces for user {user_id}")
            return added_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to add improved pieces for user {user_id}: {e}")
            return False
    
    def _add_improved_piece_to_pool(self, user_id: str, piece_data: Dict[str, Any]) -> bool:
        """
        改善版ピースデータをCFS方向プールに追加
        
        Args:
            user_id: ユーザーID
            piece_data: 改善版ピースデータ
            
        Returns:
            追加成功かどうか
        """
        if not CFS_AVAILABLE or not hasattr(self, 'direction_pool'):
            return False
        
        try:
            # V成分（右特異ベクトル）を方向ベクトルとして使用
            direction_vector = np.array(piece_data['v_component'])
            
            # CFSDirectionPieceに変換
            cfs_piece = CFSDirectionPiece(
                vector=direction_vector,
                user_id=user_id,
                semantic_tags=[piece_data['semantic_context']],
                usage_count=0,
                quality_score=piece_data['quality_score'],
                creation_time=piece_data['creation_timestamp'],
                context_embedding=direction_vector[:min(len(direction_vector), 768)]  # 適切な次元に調整
            )
            
            # プライバシー保護ノイズ適用
            if hasattr(self, 'collaboration_config'):
                noise_std = self.collaboration_config.get('privacy_noise_std', 0.01)
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, cfs_piece.vector.shape)
                    cfs_piece.vector = cfs_piece.vector + noise
                    cfs_piece.vector = cfs_piece.vector / (np.linalg.norm(cfs_piece.vector) + 1e-8)
            
            # プールに追加
            self.direction_pool.pieces.append(cfs_piece)
            
            # ユーザーマッピング更新
            if user_id not in self.direction_pool.user_mapping:
                self.direction_pool.user_mapping[user_id] = []
            self.direction_pool.user_mapping[user_id].append(len(self.direction_pool.pieces) - 1)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add piece to pool: {e}")
            return False
    
    def generate_with_improved_collaboration(self, 
                                           prompt: str, 
                                           user_id: str,
                                           user_history: Optional[List[str]] = None,
                                           alpha_personal: float = 0.1,
                                           alpha_neutral: float = -0.05,
                                           target_layers: List[str] = None,
                                           max_length: int = 100) -> str:
        """
        改善版協調機能を使用した生成
        
        Args:
            prompt: 入力プロンプト
            user_id: ユーザーID
            user_history: ユーザー履歴（指定時は動的ピース生成）
            alpha_personal: パーソナル方向強度
            alpha_neutral: ニュートラル方向強度
            target_layers: 編集対象層
            max_length: 最大生成長
            
        Returns:
            生成されたテキスト
        """
        try:
            # ユーザー履歴が提供された場合、動的にピース生成
            if user_history and self.enable_improved_pieces:
                logger.info(f"🔄 Dynamic improved pieces generation for user {user_id}")
                self.add_user_history_to_pool(user_id, user_history)
            
            # 既存のCFS-Chameleon生成を使用
            if CFS_AVAILABLE and hasattr(super(), 'generate_with_collaborative_chameleon'):
                return super().generate_with_collaborative_chameleon(
                    prompt=prompt,
                    user_id=user_id,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    target_layers=target_layers or ["model.layers.16.mlp"],
                    max_length=max_length
                )
            else:
                # フォールバック: 基本生成
                return self.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    target_layers=target_layers or ["model.layers.16.mlp"],
                    max_length=max_length
                )
                
        except Exception as e:
            logger.error(f"❌ Improved collaboration generation error: {e}")
            return f"Error in generation: {e}"
    
    def analyze_improved_pieces_quality(self, user_id: str) -> Dict[str, Any]:
        """
        改善版ピースの品質分析
        
        Args:
            user_id: ユーザーID
            
        Returns:
            品質分析結果
        """
        if user_id not in self.improved_pieces_cache:
            return {"error": "No improved pieces found for user"}
        
        pieces = self.improved_pieces_cache[user_id]
        
        # 統計計算
        singular_values = [p['singular_value'] for p in pieces]
        importance_scores = [p['importance'] for p in pieces]
        quality_scores = [p['quality_score'] for p in pieces]
        
        analysis = {
            "user_id": user_id,
            "total_pieces": len(pieces),
            "singular_values": {
                "mean": np.mean(singular_values),
                "std": np.std(singular_values),
                "max": np.max(singular_values),
                "min": np.min(singular_values)
            },
            "importance_distribution": {
                "mean": np.mean(importance_scores),
                "std": np.std(importance_scores),
                "cumulative_top3": sum(sorted(importance_scores, reverse=True)[:3])
            },
            "quality_metrics": {
                "average_quality": np.mean(quality_scores),
                "high_quality_count": sum(1 for q in quality_scores if q > 0.1),
                "quality_variance": np.var(quality_scores)
            },
            "semantic_diversity": len(set(p['semantic_context'] for p in pieces))
        }
        
        return analysis
    
    def save_improved_pieces_cache(self, filepath: str):
        """改善版ピースキャッシュ保存"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.improved_pieces_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Improved pieces cache saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def load_improved_pieces_cache(self, filepath: str):
        """改善版ピースキャッシュ読み込み"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.improved_pieces_cache = json.load(f)
                logger.info(f"✅ Improved pieces cache loaded from {filepath}")
            else:
                logger.warning(f"⚠️ Cache file not found: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")

def demo_improved_cfs_integration():
    """改善版CFS統合のデモンストレーション"""
    print("🦎 改善版CFS-Chameleon統合デモ")
    print("="*60)
    
    # サンプルユーザーデータ
    user_histories = {
        "user_1": [
            "今日は素晴らしい映画を見ました",
            "SF映画が大好きで、特にタイムトラベル系が興味深いです",
            "映画館での体験は家で見るのとは全く違います",
            "友達と映画について語り合うのが楽しいです"
        ],
        "user_2": [
            "料理をするのが趣味で、新しいレシピに挑戦しています",
            "特に和食の奥深さに魅力を感じています",
            "季節の食材を使った料理が好きです",
            "料理を通じて家族とのコミュニケーションが深まります"
        ]
    }
    
    try:
        # 改善版エディター初期化
        editor = ImprovedCFSChameleonEditor(
            use_collaboration=True,
            enable_improved_pieces=True
        )
        
        # 各ユーザーの履歴を処理
        for user_id, history in user_histories.items():
            print(f"\n👤 処理中: {user_id}")
            
            # 改善版ピース生成とプール追加
            success = editor.add_user_history_to_pool(
                user_id=user_id,
                history_texts=history,
                neutral_reference="これは一般的な内容です",
                rank_reduction=8
            )
            
            if success:
                print(f"✅ {user_id}の改善版ピース生成完了")
                
                # 品質分析
                analysis = editor.analyze_improved_pieces_quality(user_id)
                print(f"📊 品質分析:")
                print(f"   総ピース数: {analysis['total_pieces']}")
                print(f"   平均重要度: {analysis['importance_distribution']['mean']:.4f}")
                print(f"   平均品質: {analysis['quality_metrics']['average_quality']:.4f}")
                print(f"   意味的多様性: {analysis['semantic_diversity']}")
                
                # 改善版協調生成テスト
                test_prompt = "おすすめを教えてください"
                result = editor.generate_with_improved_collaboration(
                    prompt=test_prompt,
                    user_id=user_id,
                    alpha_personal=0.1,
                    max_length=50
                )
                print(f"🔮 生成結果: {result[:100]}...")
            else:
                print(f"❌ {user_id}のピース生成に失敗")
        
        # キャッシュ保存
        cache_path = "improved_cfs_pieces_cache.json"
        editor.save_improved_pieces_cache(cache_path)
        print(f"\n💾 キャッシュ保存: {cache_path}")
        
    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")
        logger.error(f"Demo execution error: {e}")
    
    print("\n🎉 改善版CFS統合デモ完了!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_improved_cfs_integration()