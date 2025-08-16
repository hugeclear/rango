#!/usr/bin/env python3
"""
CFS-Chameleon向けニュートラル方向ピース生成システム
パーソナル方向と対照的なニュートラル方向ベクトルのピース化実装
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path

# 必要なライブラリのインポート
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers/SentenceTransformers not available. Using mock implementations.")
    TRANSFORMERS_AVAILABLE = False

# CFS-Chameleon関連モジュール
try:
    from cfs_chameleon_extension import DirectionPiece
    CFS_AVAILABLE = True
except ImportError:
    print("⚠️ CFS modules not available. Using mock implementations.")
    CFS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeutralDirectionPiece:
    """ニュートラル方向ピース構造"""
    u_component: np.ndarray      # 左特異ベクトル成分
    singular_value: float        # 特異値
    v_component: np.ndarray      # 右特異ベクトル成分
    importance: float            # 重要度（特異値の寄与率）
    quality_score: float         # 品質スコア
    semantic_context: str        # 意味的コンテキスト（neutral_タグ付き）
    source_history_indices: List[int]  # 元履歴のインデックス
    creation_timestamp: float    # 作成時刻
    direction_type: str = "neutral"  # 方向タイプ識別

class NeutralDirectionGenerator:
    """ニュートラル方向ピース生成器"""
    
    def __init__(self, 
                 llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cuda"):
        """
        初期化
        
        Args:
            llm_model_name: ニュートラル言い換え生成用LLMモデル名
            embedding_model_name: 埋め込み用モデル名
            device: 計算デバイス
        """
        self.device = device
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        # モデル初期化
        self._initialize_models()
        
        logger.info(f"✅ NeutralDirectionGenerator initialized")
        logger.info(f"   LLM: {llm_model_name}")
        logger.info(f"   Embedding: {embedding_model_name}")
        logger.info(f"   Device: {device}")
    
    def _initialize_models(self):
        """モデル初期化"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # 埋め込みモデル
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"✅ Embedding model loaded: {self.embedding_model_name}")
                
                # LLMモデル（ニュートラル言い換え用）
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device == "cuda" else "cpu"
                )
                
                # パディングトークン設定
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info(f"✅ LLM model loaded: {self.llm_model_name}")
                
            except Exception as e:
                logger.error(f"❌ Model initialization error: {e}")
                self.embedding_model = None
                self.llm_model = None
                self.tokenizer = None
        else:
            # モックモデル
            self.embedding_model = None
            self.llm_model = None
            self.tokenizer = None
            logger.warning("⚠️ Using mock models for demonstration")
    
    def generate_neutral_paraphrases(self, 
                                   original_text: str, 
                                   num_variants: int = 3) -> List[str]:
        """
        LLMを使用してニュートラル・汎用的な言い換えを生成
        
        Args:
            original_text: 元テキスト
            num_variants: 生成する言い換え数
            
        Returns:
            ニュートラル言い換えリスト
        """
        if self.llm_model is None or self.tokenizer is None:
            # モック実装
            return self._mock_neutral_paraphrase_generation(original_text, num_variants)
        
        try:
            neutral_paraphrases = []
            
            # ニュートラル方向の言い換え生成プロンプト
            neutral_prompt = f"""以下のテキストを、より中立的で客観的な表現に言い換えてください。個人的な感情や主観を取り除き、一般的で汎用的な表現にしてください。{num_variants}つの異なる言い換えを生成してください。

元テキスト: {original_text}

中立的な言い換え:"""
            
            neutral_variants = self._generate_text_variants(neutral_prompt, num_variants)
            neutral_paraphrases.extend(neutral_variants)
            
            logger.debug(f"Generated {len(neutral_paraphrases)} neutral paraphrases")
            
            return neutral_paraphrases
            
        except Exception as e:
            logger.error(f"❌ Neutral paraphrase generation error: {e}")
            return self._mock_neutral_paraphrase_generation(original_text, num_variants)
    
    def _generate_text_variants(self, prompt: str, num_variants: int) -> List[str]:
        """テキスト変種生成"""
        variants = []
        
        for i in range(num_variants):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 60,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                variant = generated[len(prompt):].strip()
                
                if variant and len(variant) > 10:  # 有効な生成結果のみ追加
                    variants.append(variant)
                    
            except Exception as e:
                logger.warning(f"Generation attempt {i+1} failed: {e}")
                continue
        
        return variants
    
    def _mock_neutral_paraphrase_generation(self, original_text: str, num_variants: int) -> List[str]:
        """モックニュートラル言い換え生成（デモ用）"""
        neutral_variants = [
            f"{original_text}という状況が一般的に観察されます",
            f"一般的に{original_text}ということが言えます",
            f"客観的に見ると{original_text}です"
        ][:num_variants]
        
        return neutral_variants
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """テキストリストの埋め込み計算"""
        if self.embedding_model is None:
            # モック埋め込み
            return np.random.randn(len(texts), 384)  # MiniLMの次元数
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Embedding computation error: {e}")
            return np.random.randn(len(texts), 384)
    
    def compute_neutral_direction_vectors(self, 
                                        original_texts: List[str], 
                                        neutral_paraphrases_list: List[List[str]]) -> np.ndarray:
        """
        元テキスト・ニュートラル言い換えからニュートラル方向ベクトルを計算
        
        Args:
            original_texts: 元テキストリスト
            neutral_paraphrases_list: 各テキストに対するニュートラル言い換えリスト
            
        Returns:
            ニュートラル方向ベクトルの行列 (num_pairs, embedding_dim)
        """
        neutral_direction_vectors = []
        
        # 各テキストとそのニュートラル言い換えから方向ベクトルを計算
        for i, (original_text, neutral_paraphrases) in enumerate(zip(original_texts, neutral_paraphrases_list)):
            
            # 元テキストの埋め込み
            original_emb = self.compute_embeddings([original_text])[0]
            
            for neutral_text in neutral_paraphrases:
                # ニュートラル言い換えの埋め込み
                neutral_emb = self.compute_embeddings([neutral_text])[0]
                
                # ニュートラル方向ベクトル = neutral - original
                # （ニュートラル方向への移動を表す）
                neutral_direction_vec = neutral_emb - original_emb
                
                # 正規化
                neutral_direction_vec = neutral_direction_vec / (np.linalg.norm(neutral_direction_vec) + 1e-8)
                
                neutral_direction_vectors.append(neutral_direction_vec)
        
        return np.array(neutral_direction_vectors)
    
    def perform_neutral_svd_decomposition(self, 
                                        neutral_direction_matrix: np.ndarray, 
                                        rank_reduction: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ニュートラル方向ベクトル行列のSVD分解
        
        Args:
            neutral_direction_matrix: ニュートラル方向ベクトル行列
            rank_reduction: ランク削減数
            
        Returns:
            (U, S, Vt) - SVD分解結果
        """
        try:
            # SVD分解実行
            U, S, Vt = np.linalg.svd(neutral_direction_matrix, full_matrices=False)
            
            # ランク削減
            rank = min(rank_reduction, len(S))
            U_reduced = U[:, :rank]
            S_reduced = S[:rank]
            Vt_reduced = Vt[:rank, :]
            
            logger.info(f"✅ Neutral SVD decomposition completed")
            logger.info(f"   Original shape: {neutral_direction_matrix.shape}")
            logger.info(f"   Reduced rank: {rank}")
            logger.info(f"   Variance explained: {np.sum(S_reduced**2) / np.sum(S**2):.4f}")
            
            return U_reduced, S_reduced, Vt_reduced
            
        except Exception as e:
            logger.error(f"❌ Neutral SVD decomposition error: {e}")
            # フォールバック: 単位行列
            dim = neutral_direction_matrix.shape[1]
            rank = min(rank_reduction, dim)
            U_reduced = np.eye(neutral_direction_matrix.shape[0], rank)
            S_reduced = np.ones(rank)
            Vt_reduced = np.eye(rank, dim)
            
            return U_reduced, S_reduced, Vt_reduced
    
    def create_neutral_direction_pieces(self, 
                                      U: np.ndarray, 
                                      S: np.ndarray, 
                                      Vt: np.ndarray,
                                      history_texts: List[str]) -> List[NeutralDirectionPiece]:
        """
        SVD結果からニュートラル方向ピースを作成
        
        Args:
            U, S, Vt: SVD分解結果
            history_texts: 元履歴テキスト
            
        Returns:
            ニュートラル方向ピースのリスト
        """
        pieces = []
        total_variance = np.sum(S**2)
        
        for i in range(len(S)):
            # 重要度計算（特異値の寄与率）
            importance = (S[i]**2) / total_variance
            
            # 品質スコア（ニュートラル方向では特異値ベース）
            quality_score = importance * 0.8  # ニュートラルは若干控えめに設定
            
            # 意味的コンテキスト（neutral_タグ付き）
            base_context = f"neutral_component_{i+1}"
            if i < len(history_texts):
                # 元履歴から推定される中立化カテゴリ
                if "映画" in history_texts[i] or "娯楽" in history_texts[i]:
                    base_context = "neutral_entertainment"
                elif "料理" in history_texts[i] or "食事" in history_texts[i]:
                    base_context = "neutral_cooking"
                elif "技術" in history_texts[i] or "プログラミング" in history_texts[i]:
                    base_context = "neutral_technology"
                else:
                    base_context = "neutral_general"
            
            # 元履歴インデックス（この成分に最も寄与する履歴を特定）
            source_indices = []
            if i < U.shape[1]:
                # U成分で最も大きな値を持つ履歴を特定
                u_component = U[:, i]
                top_indices = np.argsort(np.abs(u_component))[-3:]  # 上位3つ
                source_indices = top_indices.tolist()
            
            piece = NeutralDirectionPiece(
                u_component=U[:, i] if i < U.shape[1] else np.zeros(U.shape[0]),
                singular_value=S[i],
                v_component=Vt[i, :] if i < Vt.shape[0] else np.zeros(Vt.shape[1]),
                importance=importance,
                quality_score=quality_score,
                semantic_context=base_context,
                source_history_indices=source_indices,
                creation_timestamp=time.time(),
                direction_type="neutral"
            )
            
            pieces.append(piece)
        
        # 重要度順にソート
        pieces.sort(key=lambda x: x.importance, reverse=True)
        
        logger.info(f"✅ Created {len(pieces)} neutral direction pieces")
        logger.info(f"   Top piece importance: {pieces[0].importance:.4f}")
        logger.info(f"   Total variance covered: {sum(p.importance for p in pieces):.4f}")
        
        return pieces

def generate_neutral_direction_pieces(
    user_history_texts: List[str],
    llm_model: Any = None,
    embedding_model: Any = None,
    rank_reduction: int = 16
) -> List[Dict[str, Any]]:
    """
    ニュートラル方向ベクトルをピース化して返す関数
    
    Args:
        user_history_texts: ユーザ履歴のテキストリスト
        llm_model: neutral 言い換え生成用のモデル（未使用、互換性のため）
        embedding_model: テキスト埋め込み用モデル（未使用、互換性のため）
        rank_reduction: SVD のランク削減数
        
    Returns:
        List[Dict]: ニュートラル方向のピース情報辞書リスト
    """
    logger.info(f"🚀 Starting neutral direction pieces generation")
    logger.info(f"   History texts: {len(user_history_texts)}")
    logger.info(f"   Rank reduction: {rank_reduction}")
    
    # ジェネレーター初期化
    generator = NeutralDirectionGenerator()
    
    all_neutral_paraphrases = []
    
    # 各履歴テキストからニュートラル言い換えを生成
    for i, history_text in enumerate(user_history_texts):
        logger.info(f"📝 Processing history {i+1}/{len(user_history_texts)}: {history_text[:50]}...")
        
        try:
            # ニュートラル言い換え生成
            neutral_paraphrases = generator.generate_neutral_paraphrases(
                history_text, num_variants=3
            )
            
            all_neutral_paraphrases.append(neutral_paraphrases)
            
        except Exception as e:
            logger.error(f"❌ Error processing history {i+1}: {e}")
            # フォールバック: モック言い換え
            all_neutral_paraphrases.append([f"中立的な表現: {history_text}"])
            continue
    
    if not all_neutral_paraphrases or all(not paraphrases for paraphrases in all_neutral_paraphrases):
        logger.error("❌ No neutral paraphrases generated")
        return []
    
    # ニュートラル方向ベクトル計算
    neutral_direction_matrix = generator.compute_neutral_direction_vectors(
        user_history_texts, all_neutral_paraphrases
    )
    
    logger.info(f"📊 Neutral direction matrix shape: {neutral_direction_matrix.shape}")
    
    # SVD分解実行
    U, S, Vt = generator.perform_neutral_svd_decomposition(neutral_direction_matrix, rank_reduction)
    
    # ニュートラル方向ピース作成
    neutral_pieces = generator.create_neutral_direction_pieces(U, S, Vt, user_history_texts)
    
    # 辞書形式に変換
    pieces_dict = []
    for piece in neutral_pieces:
        piece_dict = {
            "u_component": piece.u_component.tolist(),
            "singular_value": float(piece.singular_value),
            "v_component": piece.v_component.tolist(),
            "importance": float(piece.importance),
            "quality_score": float(piece.quality_score),
            "semantic_context": piece.semantic_context,
            "source_history_indices": piece.source_history_indices,
            "creation_timestamp": piece.creation_timestamp,
            "direction_type": piece.direction_type
        }
        pieces_dict.append(piece_dict)
    
    logger.info(f"🎉 Neutral direction pieces generation completed!")
    logger.info(f"   Generated {len(pieces_dict)} neutral pieces")
    
    return pieces_dict

def demonstrate_neutral_direction_generation():
    """ニュートラル方向ピース生成のデモンストレーション"""
    print("🔄 ニュートラル方向ピース生成デモ")
    print("=" * 60)
    
    # サンプルデータ
    user_history = [
        "今日は映画を見に行きたい気分です",
        "面白いSF映画が大好きです", 
        "最近見た映画は本当に素晴らしかった",
        "映画館で友達と楽しい時間を過ごしました",
        "新しい映画の予告編を見てワクワクしています"
    ]
    
    # ニュートラル方向ピース生成実行
    start_time = time.time()
    neutral_pieces = generate_neutral_direction_pieces(
        user_history_texts=user_history,
        rank_reduction=8
    )
    execution_time = time.time() - start_time
    
    # 結果表示
    print(f"\n📊 ニュートラルピース生成結果:")
    print(f"   実行時間: {execution_time:.2f}秒")
    print(f"   生成ピース数: {len(neutral_pieces)}")
    
    for i, piece in enumerate(neutral_pieces[:3]):  # 上位3つのみ表示
        print(f"\n🔸 ニュートラルピース {i+1}:")
        print(f"   特異値: {piece['singular_value']:.4f}")
        print(f"   重要度: {piece['importance']:.4f}")
        print(f"   品質スコア: {piece['quality_score']:.4f}")
        print(f"   意味的コンテキスト: {piece['semantic_context']}")
        print(f"   方向タイプ: {piece['direction_type']}")
        print(f"   U成分次元: {len(piece['u_component'])}")
        print(f"   V成分次元: {len(piece['v_component'])}")
    
    # パーソナル vs ニュートラルの比較説明
    print(f"\n🔄 パーソナル vs ニュートラル方向ピース比較:")
    print(f"   パーソナル方向: 個人的・感情的な表現への方向")
    print(f"   ニュートラル方向: 中立的・客観的な表現への方向")
    print(f"   CFS適用時: パーソナル強調(+α) + ニュートラル抑制(-β)")
    
    print("\n🎉 ニュートラル方向ピース生成デモ完了!")

if __name__ == "__main__":
    demonstrate_neutral_direction_generation()