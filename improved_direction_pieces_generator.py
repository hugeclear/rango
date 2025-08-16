#!/usr/bin/env python3
"""
改善版方向ベクトルピース生成システム
履歴ベースの意味的方向ベクトル生成によりCFS精度を向上

🎯 改善ポイント:
- 外積による方向情報消失を回避
- 履歴データから多様な意味的方向を抽出
- SVD分解で意味的に豊かなピースを生成
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time

# 必要なライブラリのインポート
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers/SentenceTransformers not available. Using mock implementations.")
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DirectionPiece:
    """改善版方向ピース構造"""
    u_component: np.ndarray      # 左特異ベクトル成分
    singular_value: float        # 特異値
    v_component: np.ndarray      # 右特異ベクトル成分
    importance: float            # 重要度（特異値の寄与率）
    quality_score: float         # 品質スコア
    semantic_context: str        # 意味的コンテキスト
    source_history_indices: List[int]  # 元履歴のインデックス
    creation_timestamp: float    # 作成時刻

class ImprovedDirectionPiecesGenerator:
    """改善版方向ベクトルピース生成器"""
    
    def __init__(self, 
                 llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cuda"):
        """
        初期化
        
        Args:
            llm_model_name: 言い換え生成用LLMモデル名
            embedding_model_name: 埋め込み用モデル名
            device: 計算デバイス
        """
        self.device = device
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        # モデル初期化
        self._initialize_models()
        
        logger.info(f"✅ ImprovedDirectionPiecesGenerator initialized")
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
                
                # LLMモデル（言い換え用）
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
    
    def generate_paraphrases(self, 
                           original_text: str, 
                           neutral_reference: str,
                           num_variants: int = 3) -> Tuple[List[str], List[str]]:
        """
        LLMを使用してパーソナル・ニュートラル方向の言い換えを生成
        
        Args:
            original_text: 元テキスト
            neutral_reference: ニュートラル参照テキスト
            num_variants: 生成する言い換え数
            
        Returns:
            (personal_paraphrases, neutral_paraphrases)
        """
        if self.llm_model is None or self.tokenizer is None:
            # モック実装
            return self._mock_paraphrase_generation(original_text, neutral_reference, num_variants)
        
        try:
            personal_paraphrases = []
            neutral_paraphrases = []
            
            # パーソナル方向の言い換え生成
            personal_prompt = f"""以下のテキストを、より個人的で感情的な表現に言い換えてください。{num_variants}つの異なる言い換えを生成してください。

元テキスト: {original_text}

言い換え:"""
            
            personal_variants = self._generate_text_variants(personal_prompt, num_variants)
            personal_paraphrases.extend(personal_variants)
            
            # ニュートラル方向の言い換え生成
            neutral_prompt = f"""以下のテキストを、より中立的で客観的な表現に言い換えてください。参照テキストのような中立的なトーンに合わせてください。{num_variants}つの異なる言い換えを生成してください。

元テキスト: {original_text}
参照テキスト: {neutral_reference}

言い換え:"""
            
            neutral_variants = self._generate_text_variants(neutral_prompt, num_variants)
            neutral_paraphrases.extend(neutral_variants)
            
            logger.debug(f"Generated {len(personal_paraphrases)} personal and {len(neutral_paraphrases)} neutral paraphrases")
            
            return personal_paraphrases, neutral_paraphrases
            
        except Exception as e:
            logger.error(f"❌ Paraphrase generation error: {e}")
            return self._mock_paraphrase_generation(original_text, neutral_reference, num_variants)
    
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
                        max_length=inputs['input_ids'].shape[1] + 50,
                        do_sample=True,
                        temperature=0.8,
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
    
    def _mock_paraphrase_generation(self, original_text: str, neutral_reference: str, num_variants: int) -> Tuple[List[str], List[str]]:
        """モック言い換え生成（デモ用）"""
        personal_variants = [
            f"私は本当に{original_text}と感じています",
            f"{original_text}ということに深く共感します",
            f"個人的には{original_text}という経験をしました"
        ][:num_variants]
        
        neutral_variants = [
            f"{original_text}という状況が観察されます",
            f"一般的に{original_text}ということが言えます",
            f"客観的に見ると{original_text}です"
        ][:num_variants]
        
        return personal_variants, neutral_variants
    
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
    
    def compute_direction_vectors(self, 
                                personal_paraphrases: List[str], 
                                neutral_paraphrases: List[str]) -> np.ndarray:
        """
        パーソナル・ニュートラル言い換えから方向ベクトルを計算
        
        Args:
            personal_paraphrases: パーソナル方向の言い換えリスト
            neutral_paraphrases: ニュートラル方向の言い換えリスト
            
        Returns:
            方向ベクトルの行列 (num_pairs, embedding_dim)
        """
        direction_vectors = []
        
        # 各ペアから方向ベクトルを計算
        min_length = min(len(personal_paraphrases), len(neutral_paraphrases))
        
        for i in range(min_length):
            personal_emb = self.compute_embeddings([personal_paraphrases[i]])[0]
            neutral_emb = self.compute_embeddings([neutral_paraphrases[i]])[0]
            
            # 方向ベクトル = personal - neutral
            direction_vec = personal_emb - neutral_emb
            
            # 正規化
            direction_vec = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)
            
            direction_vectors.append(direction_vec)
        
        return np.array(direction_vectors)
    
    def perform_svd_decomposition(self, 
                                direction_matrix: np.ndarray, 
                                rank_reduction: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        方向ベクトル行列のSVD分解
        
        Args:
            direction_matrix: 方向ベクトル行列 (num_history, embedding_dim)
            rank_reduction: ランク削減数
            
        Returns:
            (U, S, Vt) - SVD分解結果
        """
        try:
            # SVD分解実行
            U, S, Vt = np.linalg.svd(direction_matrix, full_matrices=False)
            
            # ランク削減
            rank = min(rank_reduction, len(S))
            U_reduced = U[:, :rank]
            S_reduced = S[:rank]
            Vt_reduced = Vt[:rank, :]
            
            logger.info(f"✅ SVD decomposition completed")
            logger.info(f"   Original shape: {direction_matrix.shape}")
            logger.info(f"   Reduced rank: {rank}")
            logger.info(f"   Variance explained: {np.sum(S_reduced**2) / np.sum(S**2):.4f}")
            
            return U_reduced, S_reduced, Vt_reduced
            
        except Exception as e:
            logger.error(f"❌ SVD decomposition error: {e}")
            # フォールバック: 単位行列
            dim = direction_matrix.shape[1]
            rank = min(rank_reduction, dim)
            U_reduced = np.eye(direction_matrix.shape[0], rank)
            S_reduced = np.ones(rank)
            Vt_reduced = np.eye(rank, dim)
            
            return U_reduced, S_reduced, Vt_reduced
    
    def create_direction_pieces(self, 
                              U: np.ndarray, 
                              S: np.ndarray, 
                              Vt: np.ndarray,
                              history_texts: List[str],
                              semantic_contexts: List[str]) -> List[DirectionPiece]:
        """
        SVD結果から方向ピースを作成
        
        Args:
            U, S, Vt: SVD分解結果
            history_texts: 元履歴テキスト
            semantic_contexts: 意味的コンテキスト
            
        Returns:
            方向ピースのリスト
        """
        pieces = []
        total_variance = np.sum(S**2)
        
        for i in range(len(S)):
            # 重要度計算（特異値の寄与率）
            importance = (S[i]**2) / total_variance
            
            # 品質スコア（暫定的に重要度を使用）
            quality_score = importance
            
            # 意味的コンテキスト決定
            if i < len(semantic_contexts):
                semantic_context = semantic_contexts[i]
            else:
                semantic_context = f"Direction component {i+1}"
            
            # 元履歴インデックス（この成分に最も寄与する履歴を特定）
            source_indices = []
            if i < U.shape[1]:
                # U成分で最も大きな値を持つ履歴を特定
                u_component = U[:, i]
                top_indices = np.argsort(np.abs(u_component))[-3:]  # 上位3つ
                source_indices = top_indices.tolist()
            
            piece = DirectionPiece(
                u_component=U[:, i] if i < U.shape[1] else np.zeros(U.shape[0]),
                singular_value=S[i],
                v_component=Vt[i, :] if i < Vt.shape[0] else np.zeros(Vt.shape[1]),
                importance=importance,
                quality_score=quality_score,
                semantic_context=semantic_context,
                source_history_indices=source_indices,
                creation_timestamp=time.time()
            )
            
            pieces.append(piece)
        
        # 重要度順にソート
        pieces.sort(key=lambda x: x.importance, reverse=True)
        
        logger.info(f"✅ Created {len(pieces)} direction pieces")
        logger.info(f"   Top piece importance: {pieces[0].importance:.4f}")
        logger.info(f"   Total variance covered: {sum(p.importance for p in pieces):.4f}")
        
        return pieces

def generate_improved_direction_pieces(
    user_history_texts: List[str],
    neutral_reference_text: str,
    llm_model: Any = None,
    embedding_model: Any = None,
    rank_reduction: int = 16
) -> List[Dict[str, Any]]:
    """
    改善版の方向ベクトルピース生成関数
    
    Args:
        user_history_texts: ユーザ履歴のテキストリスト
        neutral_reference_text: ニュートラル基準テキスト
        llm_model: 言い換え生成用のLLMモデル（未使用、互換性のため）
        embedding_model: テキスト埋め込み用モデル（未使用、互換性のため）
        rank_reduction: SVD分解時のランク削減数
        
    Returns:
        ピース情報を含む辞書のリスト
    """
    logger.info(f"🚀 Starting improved direction pieces generation")
    logger.info(f"   History texts: {len(user_history_texts)}")
    logger.info(f"   Rank reduction: {rank_reduction}")
    
    # ジェネレーター初期化
    generator = ImprovedDirectionPiecesGenerator()
    
    all_direction_vectors = []
    semantic_contexts = []
    
    # 各履歴テキストを処理
    for i, history_text in enumerate(user_history_texts):
        logger.info(f"📝 Processing history {i+1}/{len(user_history_texts)}: {history_text[:50]}...")
        
        try:
            # 言い換え生成
            personal_paraphrases, neutral_paraphrases = generator.generate_paraphrases(
                history_text, neutral_reference_text, num_variants=3
            )
            
            # 方向ベクトル計算
            direction_vectors = generator.compute_direction_vectors(
                personal_paraphrases, neutral_paraphrases
            )
            
            # 結果を集積
            all_direction_vectors.append(direction_vectors)
            semantic_contexts.extend([f"History {i+1} - variant {j+1}" 
                                   for j in range(len(direction_vectors))])
            
        except Exception as e:
            logger.error(f"❌ Error processing history {i+1}: {e}")
            continue
    
    if not all_direction_vectors:
        logger.error("❌ No direction vectors generated")
        return []
    
    # 全方向ベクトルを行列に結合
    direction_matrix = np.vstack(all_direction_vectors)
    logger.info(f"📊 Combined direction matrix shape: {direction_matrix.shape}")
    
    # SVD分解実行
    U, S, Vt = generator.perform_svd_decomposition(direction_matrix, rank_reduction)
    
    # 方向ピース作成
    pieces = generator.create_direction_pieces(U, S, Vt, user_history_texts, semantic_contexts)
    
    # 辞書形式に変換
    pieces_dict = []
    for piece in pieces:
        piece_dict = {
            "u_component": piece.u_component.tolist(),
            "singular_value": float(piece.singular_value),
            "v_component": piece.v_component.tolist(),
            "importance": float(piece.importance),
            "quality_score": float(piece.quality_score),
            "semantic_context": piece.semantic_context,
            "source_history_indices": piece.source_history_indices,
            "creation_timestamp": piece.creation_timestamp
        }
        pieces_dict.append(piece_dict)
    
    logger.info(f"🎉 Direction pieces generation completed!")
    logger.info(f"   Generated {len(pieces_dict)} pieces")
    
    return pieces_dict

# デモンストレーション用メイン関数
def main():
    """デモンストレーション実行"""
    print("🦎 改善版方向ベクトルピース生成デモ")
    print("="*60)
    
    # サンプルデータ
    user_history = [
        "今日は映画を見に行きたい",
        "面白いSF映画が好きです", 
        "最近見た映画は良かった",
        "映画館で友達と楽しい時間を過ごした",
        "新しい映画の予告編を見てワクワクした"
    ]
    
    neutral_reference = "映画を見ることは楽しいです"
    
    # 改善版ピース生成実行
    start_time = time.time()
    pieces = generate_improved_direction_pieces(
        user_history_texts=user_history,
        neutral_reference_text=neutral_reference,
        rank_reduction=8
    )
    execution_time = time.time() - start_time
    
    # 結果表示
    print(f"\n📊 生成結果:")
    print(f"   実行時間: {execution_time:.2f}秒")
    print(f"   生成ピース数: {len(pieces)}")
    
    for i, piece in enumerate(pieces[:3]):  # 上位3つのみ表示
        print(f"\n🔸 ピース {i+1}:")
        print(f"   特異値: {piece['singular_value']:.4f}")
        print(f"   重要度: {piece['importance']:.4f}")
        print(f"   品質スコア: {piece['quality_score']:.4f}")
        print(f"   意味的コンテキスト: {piece['semantic_context']}")
        print(f"   U成分次元: {len(piece['u_component'])}")
        print(f"   V成分次元: {len(piece['v_component'])}")
    
    print("\n🎉 デモ完了!")

if __name__ == "__main__":
    main()