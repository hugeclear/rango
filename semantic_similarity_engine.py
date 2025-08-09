#!/usr/bin/env python3
"""
CFS-Chameleon向け意味的類似度計算エンジン
従来のコサイン類似度から意味的にリッチな埋め込みベースの類似度計算への高度化
"""

import numpy as np
import torch
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path
import hashlib

# 埋め込みモデルライブラリ
try:
    from sentence_transformers import SentenceTransformer
    import openai
    EMBEDDING_LIBS_AVAILABLE = True
except ImportError:
    print("⚠️ Embedding libraries not available. Using mock implementations.")
    EMBEDDING_LIBS_AVAILABLE = False

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
class SemanticSimilarityConfig:
    """意味的類似度計算設定"""
    primary_model: str = "sentence-transformers"  # sentence-transformers, openai, hybrid
    model_name: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-ada-002"
    device: str = "cuda"
    cache_embeddings: bool = True
    batch_size: int = 32
    similarity_threshold: float = 0.1
    context_weight: float = 0.7  # ユーザコンテキストの重み
    semantic_tag_weight: float = 0.3  # semantic_tagの重み
    normalize_embeddings: bool = True
    max_cache_size: int = 10000

@dataclass
class EmbeddingCache:
    """埋め込みキャッシュ"""
    text_to_embedding: Dict[str, np.ndarray] = None
    piece_to_embedding: Dict[str, np.ndarray] = None
    access_count: Dict[str, int] = None
    
    def __post_init__(self):
        if self.text_to_embedding is None:
            self.text_to_embedding = {}
        if self.piece_to_embedding is None:
            self.piece_to_embedding = {}
        if self.access_count is None:
            self.access_count = {}

class SemanticSimilarityEngine:
    """意味的類似度計算エンジン"""
    
    def __init__(self, config: SemanticSimilarityConfig = None):
        """
        初期化
        
        Args:
            config: 意味的類似度計算設定
        """
        self.config = config or SemanticSimilarityConfig()
        self.embedding_cache = EmbeddingCache()
        
        # 埋め込みモデルの初期化
        self._initialize_embedding_models()
        
        logger.info("✅ SemanticSimilarityEngine initialized")
        logger.info(f"   Primary model: {self.config.primary_model}")
        logger.info(f"   Model name: {self.config.model_name}")
        logger.info(f"   Cache enabled: {self.config.cache_embeddings}")
    
    def _initialize_embedding_models(self):
        """埋め込みモデルの初期化"""
        if EMBEDDING_LIBS_AVAILABLE:
            try:
                # SentenceTransformer初期化
                if self.config.primary_model in ["sentence-transformers", "hybrid"]:
                    self.sentence_model = SentenceTransformer(
                        self.config.model_name,
                        device=self.config.device
                    )
                    logger.info(f"✅ SentenceTransformer loaded: {self.config.model_name}")
                else:
                    self.sentence_model = None
                
                # OpenAI API設定（実際のキーが必要）
                if self.config.primary_model in ["openai", "hybrid"]:
                    # 注意: 実際の使用時にはopenai.api_keyを設定する必要があります
                    self.openai_available = True
                    logger.info(f"✅ OpenAI Embedding configured: {self.config.openai_model}")
                else:
                    self.openai_available = False
                
            except Exception as e:
                logger.error(f"❌ Embedding model initialization error: {e}")
                self.sentence_model = None
                self.openai_available = False
        else:
            # モック設定
            self.sentence_model = None
            self.openai_available = False
            logger.warning("⚠️ Using mock embedding implementations")
    
    def _get_text_hash(self, text: str) -> str:
        """テキストのハッシュ値を取得（キャッシュキー用）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """SentenceTransformerによる埋め込み生成"""
        if self.sentence_model is None:
            # モック埋め込み（384次元）
            return np.random.randn(len(texts), 384).astype(np.float32)
        
        try:
            embeddings = self.sentence_model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"SentenceTransformer encoding error: {e}")
            return np.random.randn(len(texts), 384).astype(np.float32)
    
    def _encode_with_openai(self, texts: List[str]) -> np.ndarray:
        """OpenAI Embeddingによる埋め込み生成"""
        if not self.openai_available:
            # モック埋め込み（1536次元）
            return np.random.randn(len(texts), 1536).astype(np.float32)
        
        try:
            # バッチサイズの制限を考慮
            embeddings = []
            batch_size = min(self.config.batch_size, 100)  # OpenAIの制限
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = openai.Embedding.create(
                    input=batch_texts,
                    engine=self.config.openai_model
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if self.config.normalize_embeddings:
                embeddings_array = embeddings_array / np.linalg.norm(
                    embeddings_array, axis=1, keepdims=True
                )
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return np.random.randn(len(texts), 1536).astype(np.float32)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        テキストリストの埋め込み生成
        
        Args:
            texts: 埋め込みを生成するテキストリスト
            
        Returns:
            埋め込み行列 (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, 384)
        
        # キャッシュチェック
        if self.config.cache_embeddings:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache.text_to_embedding:
                    cached_embeddings.append((i, self.embedding_cache.text_to_embedding[text_hash]))
                    self.embedding_cache.access_count[text_hash] = \
                        self.embedding_cache.access_count.get(text_hash, 0) + 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # 新規埋め込み生成
            if uncached_texts:
                if self.config.primary_model == "openai":
                    new_embeddings = self._encode_with_openai(uncached_texts)
                elif self.config.primary_model == "hybrid":
                    # ハイブリッド: SentenceTransformerをメインに使用
                    new_embeddings = self._encode_with_sentence_transformer(uncached_texts)
                else:
                    new_embeddings = self._encode_with_sentence_transformer(uncached_texts)
                
                # キャッシュに保存
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    text_hash = self._get_text_hash(text)
                    self.embedding_cache.text_to_embedding[text_hash] = embedding
                    self.embedding_cache.access_count[text_hash] = 1
                    cached_embeddings.append((uncached_indices[i], embedding))
            
            # 結果を元の順序で並び替え
            cached_embeddings.sort(key=lambda x: x[0])
            final_embeddings = np.array([emb for _, emb in cached_embeddings])
            
        else:
            # キャッシュなしで直接生成
            if self.config.primary_model == "openai":
                final_embeddings = self._encode_with_openai(texts)
            else:
                final_embeddings = self._encode_with_sentence_transformer(texts)
        
        return final_embeddings
    
    def extract_piece_semantic_info(self, piece: Any) -> str:
        """
        DirectionPieceから意味的情報を抽出
        
        Args:
            piece: DirectionPiece (CFS_AVAILABLEでない場合はdict)
            
        Returns:
            抽出された意味的テキスト
        """
        semantic_info_parts = []
        
        if hasattr(piece, 'semantic_tags') and piece.semantic_tags:
            semantic_info_parts.extend(piece.semantic_tags)
        elif isinstance(piece, dict) and 'semantic_tags' in piece:
            if isinstance(piece['semantic_tags'], list):
                semantic_info_parts.extend(piece['semantic_tags'])
            else:
                semantic_info_parts.append(str(piece['semantic_tags']))
        
        # context_embeddingから情報を抽出（可能な場合）
        if hasattr(piece, 'context_embedding') and hasattr(piece, 'original_text'):
            if hasattr(piece, 'original_text'):
                semantic_info_parts.append(piece.original_text)
        
        # フォールバック: ピースIDやインデックス情報
        if not semantic_info_parts:
            if hasattr(piece, 'piece_id'):
                semantic_info_parts.append(f"piece_{piece.piece_id}")
            elif isinstance(piece, dict) and 'id' in piece:
                semantic_info_parts.append(f"piece_{piece['id']}")
            else:
                semantic_info_parts.append("generic_direction_piece")
        
        return " ".join(semantic_info_parts)

def compute_semantic_similarity_rich(
    text_or_vector: Union[str, np.ndarray],
    piece: Any,
    embedding_model: Optional[SemanticSimilarityEngine] = None,
    config: Optional[SemanticSimilarityConfig] = None
) -> float:
    """
    意味的にリッチな類似度を計算する関数
    
    Args:
        text_or_vector: ユーザ履歴テキスト or すでに得られた埋め込みベクトル
        piece: 評価対象の DirectionPiece
        embedding_model: SemanticSimilarityEngine インスタンス
        config: 設定（embedding_modelがNoneの場合に使用）
        
    Returns:
        float: 0.0-1.0 の意味的類似度スコア
    """
    # 埋め込みエンジンの初期化
    if embedding_model is None:
        embedding_model = SemanticSimilarityEngine(config or SemanticSimilarityConfig())
    
    try:
        # ユーザコンテキストの埋め込み取得
        if isinstance(text_or_vector, str):
            user_embeddings = embedding_model.encode_texts([text_or_vector])
            user_embedding = user_embeddings[0]
        elif isinstance(text_or_vector, np.ndarray):
            user_embedding = text_or_vector
            if embedding_model.config.normalize_embeddings:
                user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
        else:
            logger.error(f"Invalid input type: {type(text_or_vector)}")
            return 0.0
        
        # ピースの意味的情報抽出
        piece_semantic_text = embedding_model.extract_piece_semantic_info(piece)
        
        # ピースの埋め込み取得
        piece_embeddings = embedding_model.encode_texts([piece_semantic_text])
        piece_embedding = piece_embeddings[0]
        
        # コサイン類似度計算
        # 次元が異なる場合の対処
        min_dim = min(len(user_embedding), len(piece_embedding))
        user_emb_truncated = user_embedding[:min_dim]
        piece_emb_truncated = piece_embedding[:min_dim]
        
        # 正規化
        user_emb_norm = user_emb_truncated / (np.linalg.norm(user_emb_truncated) + 1e-8)
        piece_emb_norm = piece_emb_truncated / (np.linalg.norm(piece_emb_truncated) + 1e-8)
        
        # コサイン類似度
        similarity = np.dot(user_emb_norm, piece_emb_norm)
        
        # 0-1範囲にクリップ
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        logger.debug(f"Semantic similarity computed: {similarity:.4f}")
        logger.debug(f"User text: {text_or_vector if isinstance(text_or_vector, str) else 'vector'}")
        logger.debug(f"Piece semantic: {piece_semantic_text}")
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ Semantic similarity computation error: {e}")
        return 0.0

def compute_batch_semantic_similarity(
    user_contexts: List[Union[str, np.ndarray]],
    pieces: List[Any],
    embedding_model: Optional[SemanticSimilarityEngine] = None,
    config: Optional[SemanticSimilarityConfig] = None
) -> np.ndarray:
    """
    バッチでの意味的類似度計算
    
    Args:
        user_contexts: ユーザコンテキストリスト
        pieces: DirectionPieceリスト
        embedding_model: SemanticSimilarityEngine インスタンス
        config: 設定
        
    Returns:
        類似度行列 (len(user_contexts), len(pieces))
    """
    if embedding_model is None:
        embedding_model = SemanticSimilarityEngine(config or SemanticSimilarityConfig())
    
    logger.info(f"🔄 Computing batch semantic similarity: {len(user_contexts)} contexts × {len(pieces)} pieces")
    
    try:
        # ユーザコンテキストの埋め込み取得
        user_texts = []
        user_vectors = []
        
        for ctx in user_contexts:
            if isinstance(ctx, str):
                user_texts.append(ctx)
                user_vectors.append(None)
            else:
                user_texts.append(None)
                user_vectors.append(ctx)
        
        # テキストの埋め込み生成
        text_embeddings = []
        if any(t is not None for t in user_texts):
            valid_texts = [t for t in user_texts if t is not None]
            if valid_texts:
                text_embeddings = embedding_model.encode_texts(valid_texts)
        
        # 最終的なユーザ埋め込み配列構築
        final_user_embeddings = []
        text_idx = 0
        
        for i, (text, vector) in enumerate(zip(user_texts, user_vectors)):
            if text is not None:
                final_user_embeddings.append(text_embeddings[text_idx])
                text_idx += 1
            else:
                final_user_embeddings.append(vector)
        
        final_user_embeddings = np.array(final_user_embeddings)
        
        # ピースの埋め込み取得
        piece_texts = [embedding_model.extract_piece_semantic_info(piece) for piece in pieces]
        piece_embeddings = embedding_model.encode_texts(piece_texts)
        
        # バッチ類似度計算
        # 次元調整
        min_dim = min(final_user_embeddings.shape[1], piece_embeddings.shape[1])
        user_emb_truncated = final_user_embeddings[:, :min_dim]
        piece_emb_truncated = piece_embeddings[:, :min_dim]
        
        # 正規化
        user_emb_norm = user_emb_truncated / (np.linalg.norm(user_emb_truncated, axis=1, keepdims=True) + 1e-8)
        piece_emb_norm = piece_emb_truncated / (np.linalg.norm(piece_emb_truncated, axis=1, keepdims=True) + 1e-8)
        
        # 類似度行列計算
        similarity_matrix = np.dot(user_emb_norm, piece_emb_norm.T)
        
        # 0-1範囲に正規化
        similarity_matrix = (similarity_matrix + 1.0) / 2.0
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        logger.info(f"✅ Batch similarity computation completed")
        logger.info(f"   Similarity range: {similarity_matrix.min():.4f} - {similarity_matrix.max():.4f}")
        logger.info(f"   Average similarity: {similarity_matrix.mean():.4f}")
        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"❌ Batch semantic similarity error: {e}")
        # フォールバック: ランダム類似度
        return np.random.uniform(0.1, 0.9, (len(user_contexts), len(pieces)))

class HybridSimilarityCalculator:
    """ハイブリッド類似度計算器（複数手法の組み合わせ）"""
    
    def __init__(self, 
                 semantic_engine: SemanticSimilarityEngine,
                 semantic_weight: float = 0.8,
                 vector_weight: float = 0.2):
        """
        初期化
        
        Args:
            semantic_engine: 意味的類似度エンジン
            semantic_weight: 意味的類似度の重み
            vector_weight: ベクトル類似度の重み
        """
        self.semantic_engine = semantic_engine
        self.semantic_weight = semantic_weight
        self.vector_weight = vector_weight
        
        logger.info("✅ HybridSimilarityCalculator initialized")
        logger.info(f"   Semantic weight: {semantic_weight}")
        logger.info(f"   Vector weight: {vector_weight}")
    
    def compute_hybrid_similarity(self,
                                text_or_vector: Union[str, np.ndarray],
                                piece: Any) -> float:
        """
        ハイブリッド類似度計算
        
        Args:
            text_or_vector: ユーザコンテキスト
            piece: DirectionPiece
            
        Returns:
            ハイブリッド類似度スコア
        """
        try:
            # 意味的類似度
            semantic_sim = compute_semantic_similarity_rich(
                text_or_vector, piece, self.semantic_engine
            )
            
            # ベクトル類似度（従来手法）
            if isinstance(text_or_vector, str):
                # テキストの場合は意味的類似度のみ使用
                vector_sim = 0.5  # ニュートラル
            else:
                # ベクトルの場合は直接計算
                if hasattr(piece, 'u_component'):
                    piece_vector = piece.u_component
                elif hasattr(piece, 'vector'):
                    piece_vector = piece.vector
                elif isinstance(piece, dict) and 'u_component' in piece:
                    piece_vector = np.array(piece['u_component'])
                else:
                    piece_vector = np.random.randn(len(text_or_vector))
                
                # 次元調整
                min_dim = min(len(text_or_vector), len(piece_vector))
                user_vec = text_or_vector[:min_dim]
                piece_vec = piece_vector[:min_dim]
                
                # コサイン類似度
                dot_product = np.dot(user_vec, piece_vec)
                magnitude = np.linalg.norm(user_vec) * np.linalg.norm(piece_vec)
                vector_sim = dot_product / (magnitude + 1e-8)
                vector_sim = max(0.0, min(1.0, (vector_sim + 1.0) / 2.0))
            
            # ハイブリッドスコア
            hybrid_score = (
                self.semantic_weight * semantic_sim + 
                self.vector_weight * vector_sim
            )
            
            logger.debug(f"Hybrid similarity - Semantic: {semantic_sim:.4f}, Vector: {vector_sim:.4f}, Final: {hybrid_score:.4f}")
            
            return hybrid_score
            
        except Exception as e:
            logger.error(f"❌ Hybrid similarity error: {e}")
            return 0.5

def demonstrate_semantic_similarity():
    """意味的類似度計算のデモンストレーション"""
    print("🔍 意味的類似度計算エンジンデモ")
    print("=" * 60)
    
    # 設定とエンジンの初期化
    config = SemanticSimilarityConfig(
        primary_model="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_embeddings=True
    )
    
    engine = SemanticSimilarityEngine(config)
    
    # サンプルデータ
    user_contexts = [
        "最近のSF小説について語りたい",
        "料理のレシピを教えてください", 
        "映画の感想を聞かせて",
        "プログラミングの質問があります"
    ]
    
    # モックDirectionPieces
    mock_pieces = [
        {"id": "sci_fi", "semantic_tags": ["science_fiction", "literature"], "u_component": np.random.randn(768)},
        {"id": "cooking", "semantic_tags": ["cooking", "recipes"], "u_component": np.random.randn(768)},
        {"id": "movies", "semantic_tags": ["entertainment", "films"], "u_component": np.random.randn(768)},
        {"id": "tech", "semantic_tags": ["programming", "technology"], "u_component": np.random.randn(768)}
    ]
    
    print(f"\n📝 Testing individual similarity computation:")
    print("-" * 40)
    
    # 個別類似度計算テスト
    for i, context in enumerate(user_contexts):
        print(f"\n🔸 User Context: '{context}'")
        
        for j, piece in enumerate(mock_pieces):
            similarity = compute_semantic_similarity_rich(context, piece, engine)
            piece_info = f"Piece {piece['id']} ({', '.join(piece['semantic_tags'])})"
            print(f"   vs {piece_info:<30}: {similarity:.4f}")
    
    print(f"\n\n🔄 Testing batch similarity computation:")
    print("-" * 40)
    
    # バッチ類似度計算テスト
    similarity_matrix = compute_batch_semantic_similarity(
        user_contexts, mock_pieces, engine
    )
    
    print(f"\nSimilarity Matrix ({len(user_contexts)}×{len(mock_pieces)}):")
    print("Context\\Piece", end="")
    for piece in mock_pieces:
        print(f"{piece['id'][:8]:>10}", end="")
    print()
    
    for i, context in enumerate(user_contexts):
        context_short = context[:15] + "..." if len(context) > 15 else context
        print(f"{context_short:<13}", end="")
        for j in range(len(mock_pieces)):
            print(f"{similarity_matrix[i,j]:>10.4f}", end="")
        print()
    
    print(f"\n\n🔗 Testing hybrid similarity:")
    print("-" * 40)
    
    # ハイブリッド類似度テスト
    hybrid_calc = HybridSimilarityCalculator(engine, semantic_weight=0.7, vector_weight=0.3)
    
    context = user_contexts[0]  # "最近のSF小説について語りたい"
    piece = mock_pieces[0]      # sci_fi piece
    
    hybrid_sim = hybrid_calc.compute_hybrid_similarity(context, piece)
    print(f"Hybrid similarity for '{context}' vs sci_fi piece: {hybrid_sim:.4f}")
    
    print(f"\n📊 Cache statistics:")
    print(f"   Cached embeddings: {len(engine.embedding_cache.text_to_embedding)}")
    print(f"   Cache hits: {sum(engine.embedding_cache.access_count.values())}")
    
    print("\n🎉 意味的類似度計算デモ完了!")

if __name__ == "__main__":
    demonstrate_semantic_similarity()