#!/usr/bin/env python3
"""
GraphRAG動的CFS-Chameleon向けFAISS索引構築システム (Step-2 完全版)
Step-1の共通埋め込みからFAISS索引を構築し、Top-k近傍検索で疎グラフエッジを生成

## 機能:
- 入力: NPY+JSON / Parquet形式対応
- 索引: IVF+PQ (大規模向け) / HNSW (高精度向け)
- メトリック: cosine (L2正規化自動) / l2距離
- GPU/CPU自動選択、OOM回避、バッチ処理
- 疎グラフエッジリスト出力（PPR用）
- 統計情報ログ（ゲート設計用）
- 単体テスト内蔵

## 対象ベンチマーク:
- LaMP-2（生成タスク）
- Tenrec（ランキングタスク）
"""

import os
import sys
import json
import argparse
import random
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from scipy import stats

# 依存関係チェック
try:
    import faiss
    import pyarrow.parquet as pq
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ 依存パッケージが不足: {e}")
    print("pip install faiss-gpu pandas pyarrow tqdm numpy scipy を実行してください")
    print("CPU環境の場合は faiss-gpu → faiss-cpu に変更")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)

class FAISSIndexBuilder:
    """FAISS索引構築・近傍検索システム"""
    
    def __init__(self,
                 index_type: str = "ivf_pq",
                 metric: str = "cosine",
                 normalize: bool = None,
                 use_gpu: Union[str, bool] = "auto",
                 nlist: int = None,
                 seed: int = 42):
        """
        初期化
        
        Args:
            index_type: 索引タイプ ("ivf_pq", "hnsw_flat")
            metric: 距離メトリック ("cosine", "l2")  
            normalize: L2正規化フラグ（None=自動、cosineなら True）
            use_gpu: GPU使用 ("auto", True, False)
            nlist: IVFクラスタ数 (None=自動最適化)
            seed: 乱数シード
        """
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist  # 後で自動補正の可能性あり
        
        # 正規化設定: cosineなら自動でTrue、それ以外はFalseまたはユーザー指定
        if normalize is None:
            self.normalize = (metric == "cosine")
        else:
            self.normalize = normalize
            
        self.seed = seed
        
        # 再現性確保
        self._set_seed(seed)
        
        # GPU設定
        if use_gpu == "auto":
            self.use_gpu = faiss.get_num_gpus() > 0
        else:
            self.use_gpu = bool(use_gpu)
            
        # GPU利用可能性チェック
        if self.use_gpu and faiss.get_num_gpus() == 0:
            logger.warning("GPU指定されましたがFAISS GPUが利用不可、CPUで実行")
            self.use_gpu = False
            
        logger.info(f"🚀 FAISSIndexBuilder初期化")
        logger.info(f"   索引タイプ: {index_type}")
        logger.info(f"   距離メトリック: {metric}")
        logger.info(f"   L2正規化: {self.normalize}")
        logger.info(f"   GPU使用: {self.use_gpu} (利用可能GPU数: {faiss.get_num_gpus()})")
        
        # 内部状態
        self.index = None
        self.ids = None
        self.embeddings = None
        self.dim = None
        
    def _set_seed(self, seed: int):
        """乱数シード固定"""
        random.seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def _round_to_multiple(x, base=8):
        """数値を指定の倍数に丸める"""
        return int(base * max(1, round(x / base)))
    
    def _choose_nlist(self, n_samples: int) -> int:
        """
        IVF nlist パラメータの自動最適化
        
        Args:
            n_samples: サンプル数
            
        Returns:
            最適化されたnlist値
        """
        # ユーザ指定が妥当ならそのまま使う
        if isinstance(self.nlist, int) and 64 <= self.nlist <= 16384:
            return self.nlist
        
        # 自動推奨: 4*sqrt(N) を 8の倍数へ丸め、境界をクリップ
        import math
        suggested = self._round_to_multiple(4.0 * math.sqrt(max(1, n_samples)), 8)
        chosen = int(min(16384, max(64, suggested)))
        
        if self.nlist is None:
            logger.info(f"IVF nlist auto-selected: N={n_samples:,}, suggested={suggested}, used={chosen}")
        else:
            logger.info(f"IVF nlist auto-corrected: user nlist={self.nlist} is out of range, used={chosen}")
        
        return chosen
        
    def load_embeddings(self, 
                       emb_npy: Optional[str] = None,
                       ids_json: Optional[str] = None,
                       emb_parquet: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        埋め込みデータ読み込み
        
        Args:
            emb_npy: NPYファイルパス
            ids_json: IDリストJSONファイルパス
            emb_parquet: Parquetファイルパス（列: id, embedding）
            
        Returns:
            (埋め込み配列, IDリスト)
        """
        logger.info("📂 埋め込みデータ読み込み開始")
        
        if emb_parquet:
            # Parquet形式読み込み
            logger.info(f"   Parquet形式: {emb_parquet}")
            try:
                if not Path(emb_parquet).exists():
                    raise FileNotFoundError(f"Parquetファイルが見つかりません: {emb_parquet}")
                    
                df = pd.read_parquet(emb_parquet)
                
                # 必要列チェック
                required_cols = ['id', 'embedding']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"必要な列が不足: {missing_cols}, 利用可能: {list(df.columns)}")
                
                # IDリスト抽出
                ids = [str(id_val) for id_val in df['id'].tolist()]
                
                # 埋め込み配列変換
                embeddings_list = df['embedding'].tolist()
                
                # list[float] → numpy配列変換
                try:
                    embeddings = np.array(embeddings_list, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    # 入れ子リスト処理
                    embeddings = np.vstack([np.array(emb, dtype=np.float32) for emb in embeddings_list])
                
                logger.info(f"✅ Parquet読み込み完了: {len(ids):,} 埋め込み")
                
            except Exception as e:
                raise RuntimeError(f"❌ CRITICAL: Parquet読み込み失敗: {e}")
                
        elif emb_npy and ids_json:
            # NPY + JSON形式読み込み
            logger.info(f"   NPY + JSON形式: {emb_npy}, {ids_json}")
            try:
                # ファイル存在確認
                if not Path(emb_npy).exists():
                    raise FileNotFoundError(f"NPYファイルが見つかりません: {emb_npy}")
                if not Path(ids_json).exists():
                    raise FileNotFoundError(f"JSONファイルが見つかりません: {ids_json}")
                
                # 埋め込み読み込み
                embeddings = np.load(emb_npy).astype(np.float32)
                
                # ID読み込み
                with open(ids_json, 'r', encoding='utf-8') as f:
                    ids_raw = json.load(f)
                ids = [str(id_val) for id_val in ids_raw]
                
                logger.info(f"✅ NPY+JSON読み込み完了: {len(ids):,} 埋め込み")
                
            except Exception as e:
                raise RuntimeError(f"❌ CRITICAL: NPY+JSON読み込み失敗: {e}")
                
        else:
            raise ValueError(
                "❌ CRITICAL: --emb_parquet または (--emb_npy + --ids_json) のいずれかを指定してください"
            )
            
        # データ整合性チェック
        if len(ids) != len(embeddings):
            raise RuntimeError(
                f"❌ CRITICAL: ID数({len(ids)})と埋め込み数({len(embeddings)})が不一致"
            )
            
        # 次元チェック
        if len(embeddings.shape) != 2:
            raise RuntimeError(
                f"❌ CRITICAL: 埋め込み形状が不正: {embeddings.shape} (期待: (N, dim))"
            )
            
        # 次元一貫性チェック
        self.dim = embeddings.shape[1]
        if self.dim not in [384, 768, 1024, 1536]:  # 一般的な埋め込み次元
            logger.warning(f"⚠️ 非標準的な埋め込み次元: {self.dim}")
            
        self.embeddings = embeddings
        self.ids = ids
        
        # 統計情報ログ出力（ゲート設計用）
        self._log_embedding_statistics()
        
        # L2正規化処理
        if self.normalize:
            logger.info("🔄 L2正規化チェック・実行中...")
            self._apply_l2_normalization()
        
        logger.info(f"📊 データ準備完了: {len(ids):,} samples × {self.dim} dims")
        return embeddings, ids
        
    def _log_embedding_statistics(self):
        """埋め込み統計情報をログ出力（ゲート設計用）"""
        if self.embeddings is None:
            return
            
        # 基本統計
        emb_flat = self.embeddings.flatten()
        mean_val = np.mean(emb_flat)
        std_val = np.std(emb_flat)
        
        # ノルム統計
        norms = np.linalg.norm(self.embeddings, axis=1)
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        norm_min = np.min(norms)
        norm_max = np.max(norms)
        
        # 高次統計（ゲート設計用）
        try:
            skewness = stats.skew(emb_flat)
            kurt = stats.kurtosis(emb_flat)
        except:
            skewness, kurt = 0.0, 0.0
            
        logger.info(f"📊 埋め込み統計情報（ゲート設計用）:")
        logger.info(f"   値統計: mean={mean_val:.6f}, std={std_val:.6f}")
        logger.info(f"   ノルム: mean={norm_mean:.6f}, std={norm_std:.6f}, min={norm_min:.6f}, max={norm_max:.6f}")
        logger.info(f"   分布形状: skewness={skewness:.6f}, kurtosis={kurt:.6f}")
        
    def _apply_l2_normalization(self):
        """L2正規化実行（重複正規化を回避）"""
        # 既に正規化済みかチェック
        norms = np.linalg.norm(self.embeddings, axis=1)
        is_already_normalized = np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4)
        
        if is_already_normalized:
            logger.info("✅ 埋め込みは既にL2正規化済み（重複正規化を回避）")
            return
            
        # L2正規化実行
        logger.info("   L2正規化を実行中...")
        faiss.normalize_L2(self.embeddings)
        
        # 正規化後確認
        norms_after = np.linalg.norm(self.embeddings, axis=1)
        logger.info(f"✅ L2正規化完了: ノルム範囲 [{norms_after.min():.6f}, {norms_after.max():.6f}]")
        
    def build_index(self,
                   pq_m: int = 16,
                   pq_bits: int = 8,
                   hnsw_m: int = 32,
                   efc: int = 200,
                   train_size: int = 200000,
                   add_batch_size: int = 10000) -> faiss.Index:
        """
        FAISS索引構築
        
        Args:
            pq_m: PQ分割数
            pq_bits: PQビット数
            hnsw_m: HNSW接続数
            efc: HNSW構築efConstruction
            train_size: 訓練サンプル数上限
            add_batch_size: 追加バッチサイズ
            
        Returns:
            構築済みFAISSインデックス
        """
        if self.embeddings is None:
            raise RuntimeError("❌ CRITICAL: 埋め込みが読み込まれていません")
            
        logger.info(f"🔧 FAISS索引構築開始: {self.index_type}")
        start_time = time.time()
        
        # 索引作成
        if self.index_type == "ivf_pq":
            # IVF+PQ索引
            nlist = self._choose_nlist(len(self.embeddings))
            logger.info(f"   IVFパラメータ: nlist={nlist}, pq_m={pq_m}, pq_bits={pq_bits}")
            
            # 次元チェック（PQ分割可能性）
            if self.dim % pq_m != 0:
                logger.warning(f"⚠️ 次元({self.dim})がPQ分割数({pq_m})で割り切れません、調整を推奨")
                
            # quantizer作成（cosine/L2共にL2距離で近似可能）
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, pq_m, pq_bits)
            
        elif self.index_type == "hnsw_flat":
            # HNSW索引
            logger.info(f"   HNSWパラメータ: M={hnsw_m}, efConstruction={efc}")
            
            # HNSWはL2距離ベース
            index = faiss.IndexHNSWFlat(self.dim, hnsw_m)
            index.hnsw.efConstruction = efc
            
        else:
            raise ValueError(f"❌ 未対応索引タイプ: {self.index_type}")
            
        # GPU化
        if self.use_gpu:
            try:
                logger.info("🚀 GPU索引に変換中...")
                # GPU化前の索引タイプを保存
                cpu_index = index
                index = faiss.index_cpu_to_all_gpus(index)
                logger.info("✅ GPU索引化完了")
            except Exception as e:
                logger.warning(f"⚠️ GPU化失敗、CPUで続行: {e}")
                self.use_gpu = False
                index = cpu_index
                
        # 訓練（IVF+PQのみ）
        if self.index_type == "ivf_pq":
            logger.info("🎓 IVF+PQ索引訓練開始")
            
            # 訓練サンプル準備
            n_samples = len(self.embeddings)
            actual_train_size = min(train_size, n_samples)
            
            if actual_train_size < n_samples:
                # ランダムサンプリング
                train_indices = np.random.choice(n_samples, actual_train_size, replace=False)
                train_data = self.embeddings[train_indices]
                logger.info(f"   訓練データ: {actual_train_size:,} / {n_samples:,} samples (ランダム抽出)")
            else:
                train_data = self.embeddings
                logger.info(f"   訓練データ: {actual_train_size:,} samples (全データ)")
                
            # 訓練実行
            logger.info("   訓練実行中...")
            try:
                index.train(train_data)
                logger.info("✅ IVF+PQ訓練完了")
            except Exception as e:
                raise RuntimeError(f"❌ CRITICAL: IVF+PQ訓練失敗: {e}")
                
        # データ追加（バッチ処理でOOM回避）
        logger.info(f"📥 索引へのデータ追加開始（バッチサイズ: {add_batch_size:,}）")
        
        with tqdm(total=len(self.embeddings), desc="データ追加", unit="vectors") as pbar:
            for i in range(0, len(self.embeddings), add_batch_size):
                end_i = min(i + add_batch_size, len(self.embeddings))
                batch_data = self.embeddings[i:end_i]
                
                try:
                    index.add(batch_data)
                    pbar.update(len(batch_data))
                    
                    # GPUメモリ管理
                    if self.use_gpu and hasattr(faiss, 'gpu'):
                        faiss.gpu.synchronize_all_devices()
                        
                except Exception as e:
                    raise RuntimeError(f"❌ CRITICAL: バッチ {i//add_batch_size + 1} 追加失敗: {e}")
                    
        elapsed_time = time.time() - start_time
        vectors_per_sec = len(self.embeddings) / elapsed_time
        
        logger.info(f"✅ 索引構築完了:")
        logger.info(f"   処理時間: {elapsed_time:.2f}s")
        logger.info(f"   処理速度: {vectors_per_sec:.1f} vectors/sec")
        logger.info(f"   索引サイズ: {index.ntotal:,} vectors")
        
        self.index = index
        return index
        
    def search_neighbors(self,
                        topk: int = 10,
                        batch_size: int = 8192,
                        efs: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-k近傍検索（全ベクトル対象）
        
        Args:
            topk: 近傍数
            batch_size: 検索バッチサイズ
            efs: HNSW検索efSearch
            
        Returns:
            (距離配列, インデックス配列) - shape: (n_queries, topk)
        """
        if self.index is None or self.embeddings is None:
            raise RuntimeError("❌ CRITICAL: 索引または埋め込みが準備されていません")
            
        logger.info(f"🔍 Top-{topk} 近傍検索開始")
        logger.info(f"   クエリ数: {len(self.embeddings):,}")
        logger.info(f"   バッチサイズ: {batch_size:,}")
        
        # HNSW検索パラメータ設定
        if self.index_type == "hnsw_flat":
            try:
                if hasattr(self.index, 'hnsw'):
                    self.index.hnsw.efSearch = efs
                elif hasattr(self.index, 'index') and hasattr(self.index.index, 'hnsw'):
                    # GPU版の場合
                    self.index.index.hnsw.efSearch = efs
                logger.info(f"   HNSW efSearch: {efs}")
            except Exception as e:
                logger.warning(f"⚠️ HNSW efSearch設定失敗: {e}")
                
        start_time = time.time()
        
        # バッチ検索でOOM回避
        all_distances = []
        all_indices = []
        
        with tqdm(total=len(self.embeddings), desc="近傍検索", unit="queries") as pbar:
            for i in range(0, len(self.embeddings), batch_size):
                end_i = min(i + batch_size, len(self.embeddings))
                batch_queries = self.embeddings[i:end_i]
                
                try:
                    # バッチ検索実行
                    batch_distances, batch_indices = self.index.search(batch_queries, topk)
                    
                    all_distances.append(batch_distances)
                    all_indices.append(batch_indices)
                    
                    pbar.update(len(batch_queries))
                    
                    # GPU メモリ同期
                    if self.use_gpu and hasattr(faiss, 'gpu'):
                        faiss.gpu.synchronize_all_devices()
                        
                except Exception as e:
                    raise RuntimeError(f"❌ CRITICAL: バッチ {i//batch_size + 1} 検索失敗: {e}")
                    
        # 結果結合
        distances = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        
        elapsed_time = time.time() - start_time
        queries_per_sec = len(self.embeddings) / elapsed_time
        
        logger.info(f"✅ 近傍検索完了:")
        logger.info(f"   処理時間: {elapsed_time:.2f}s")
        logger.info(f"   検索速度: {queries_per_sec:.1f} queries/sec")
        logger.info(f"   結果形状: {distances.shape}")
        
        return distances, indices
        
    def convert_to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """
        距離を類似度に変換（cosineメトリック用）
        
        Args:
            distances: FAISS L2距離配列
            
        Returns:
            類似度配列 (cosine時は変換、l2時はそのまま)
        """
        if self.metric != "cosine":
            # L2距離はそのまま返却（距離として使用）
            return distances
            
        # L2距離 → cosine類似度変換（安全クランプ適用）
        # 正規化済みベクトルなら L2距離 d∈[0,2]、変換: sim = 1 - 0.5 * d^2
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            distances = np.clip(distances, 0.0, 2.0, out=distances)
            similarities = 1.0 - 0.5 * (distances * distances)
        
        similarities = np.clip(similarities, -1.0, 1.0)  # [-1, 1]範囲にクリップ
        
        return similarities
        
    def create_graph_edges(self,
                          distances: np.ndarray,
                          indices: np.ndarray,
                          undirected: bool = False,
                          remove_self_loops: bool = True) -> pd.DataFrame:
        """
        近傍検索結果から疎グラフエッジリスト生成
        
        Args:
            distances: 距離配列 (n_queries, topk)
            indices: インデックス配列 (n_queries, topk)
            undirected: 無向グラフにするか
            remove_self_loops: 自己ループ除去するか
            
        Returns:
            エッジリストDataFrame (src_id, dst_id, score)
        """
        logger.info("🕸️ 疎グラフエッジ生成開始")
        logger.info(f"   入力形状: {distances.shape}")
        logger.info(f"   無向グラフ: {undirected}")
        logger.info(f"   自己ループ除去: {remove_self_loops}")
        
        # 距離→類似度変換（必要に応じて）
        scores = self.convert_to_similarities(distances)
        
        # エッジリスト構築
        edges = []
        n_queries, topk = indices.shape
        
        logger.info("   エッジリスト構築中...")
        for i in tqdm(range(n_queries), desc="エッジ生成", unit="nodes", leave=False):
            src_id = self.ids[i]
            
            for j in range(topk):
                dst_idx = indices[i, j]
                
                # 無効インデックスをスキップ（FAISS が -1 を返すことがある）
                if dst_idx < 0 or dst_idx >= len(self.ids):
                    continue
                    
                dst_id = self.ids[dst_idx]
                score = float(scores[i, j])
                
                # 自己ループ除去
                if remove_self_loops and src_id == dst_id:
                    continue
                    
                edges.append({
                    'src_id': src_id,
                    'dst_id': dst_id,
                    'score': score
                })
                
        # DataFrame作成
        if not edges:
            logger.warning("⚠️ エッジが生成されませんでした")
            return pd.DataFrame(columns=['src_id', 'dst_id', 'score'])
            
        df_edges = pd.DataFrame(edges)
        logger.info(f"   初期エッジ数: {len(df_edges):,}")
        
        # 無向グラフ化
        if undirected:
            logger.info("   無向グラフ化実行中...")
            
            # 逆向きエッジ生成
            reverse_edges = df_edges.copy()
            reverse_edges = reverse_edges.rename(columns={'src_id': 'dst_id', 'dst_id': 'src_id'})
            
            # 結合・重複除去
            df_combined = pd.concat([df_edges, reverse_edges], ignore_index=True)
            df_edges = df_combined.drop_duplicates(subset=['src_id', 'dst_id'], keep='first')
            
            logger.info(f"   無向グラフ化後エッジ数: {len(df_edges):,}")
            
        # グラフ統計情報
        unique_nodes = len(set(df_edges['src_id'].tolist() + df_edges['dst_id'].tolist()))
        avg_degree = len(df_edges) / unique_nodes if unique_nodes > 0 else 0
        score_stats = df_edges['score'].describe()
        
        logger.info(f"✅ グラフエッジ生成完了:")
        logger.info(f"   エッジ数: {len(df_edges):,}")
        logger.info(f"   ノード数: {unique_nodes:,}")
        logger.info(f"   平均次数: {avg_degree:.2f}")
        logger.info(f"   スコア統計: min={score_stats['min']:.6f}, max={score_stats['max']:.6f}, mean={score_stats['mean']:.6f}")
        
        return df_edges
        
    def save_results(self,
                    distances: np.ndarray,
                    indices: np.ndarray,
                    df_edges: pd.DataFrame,
                    output_dir: str,
                    **index_params) -> Dict[str, str]:
        """
        結果保存
        
        Args:
            distances: 距離配列
            indices: インデックス配列
            df_edges: エッジDataFrame
            output_dir: 出力ディレクトリ
            **index_params: 索引パラメータ（メタデータ用）
            
        Returns:
            保存ファイルパス辞書
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 結果保存開始: {output_dir}")
        saved_files = {}
        
        # 1. FAISS索引保存
        try:
            index_path = output_path / "index.faiss"
            
            # GPU索引をCPUに戻してから保存
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
                
            saved_files['index'] = str(index_path)
            logger.info(f"✅ FAISS索引保存完了: {index_path}")
            
        except Exception as e:
            logger.error(f"❌ FAISS索引保存失敗: {e}")
            
        # 2. 近傍結果Parquet保存（長い形式）
        try:
            neighbors_data = []
            scores = self.convert_to_similarities(distances)
            
            for i in range(len(indices)):
                src_id = self.ids[i]
                for j, (dst_idx, score, distance) in enumerate(zip(indices[i], scores[i], distances[i])):
                    if dst_idx >= 0 and dst_idx < len(self.ids):
                        neighbors_data.append({
                            'src_id': src_id,
                            'dst_id': self.ids[dst_idx],
                            'rank': j + 1,
                            'score': float(score),
                            'distance': float(distance)
                        })
                        
            df_neighbors = pd.DataFrame(neighbors_data)
            neighbors_path = output_path / "neighbors.parquet"
            df_neighbors.to_parquet(neighbors_path, index=False)
            saved_files['neighbors'] = str(neighbors_path)
            logger.info(f"✅ 近傍結果保存完了: {neighbors_path}")
            
        except Exception as e:
            logger.error(f"❌ 近傍結果保存失敗: {e}")
            
        # 3. グラフエッジ保存（PPR用）
        try:
            edges_path = output_path / "graph_edges.parquet"
            df_edges.to_parquet(edges_path, index=False)
            saved_files['edges'] = str(edges_path)
            logger.info(f"✅ グラフエッジ保存完了: {edges_path}")
            
        except Exception as e:
            logger.error(f"❌ グラフエッジ保存失敗: {e}")
            
        # 4. メタデータ保存
        try:
            metadata = {
                # 索引設定
                'index_type': self.index_type,
                'metric': self.metric,
                'normalize': self.normalize,
                'use_gpu': self.use_gpu,
                
                # データ情報
                'embedding_dim': self.dim,
                'total_vectors': len(self.embeddings),
                'topk': distances.shape[1] if distances.size > 0 else 0,
                'total_edges': len(df_edges),
                
                # 実行設定
                'seed': self.seed,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                
                # 索引パラメータ
                **index_params
            }
            
            metadata_path = output_path / "meta.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            saved_files['metadata'] = str(metadata_path)
            logger.info(f"✅ メタデータ保存完了: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ メタデータ保存失敗: {e}")
            
        logger.info(f"🎯 保存完了: {len(saved_files)} ファイル")
        return saved_files


def run_self_test():
    """単体テスト（疑似データで動作検証）"""
    logger.info("🧪 単体テスト実行開始")
    
    try:
        # 疑似データ生成
        np.random.seed(42)
        n_samples = 1000
        dim = 128
        
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        ids = [f"test_id_{i:04d}" for i in range(n_samples)]
        
        logger.info(f"   疑似データ生成: {n_samples} samples × {dim} dims")
        
        # IVF+PQ テスト
        logger.info("   IVF+PQ索引テスト...")
        builder_ivf = FAISSIndexBuilder(
            index_type="ivf_pq",
            metric="cosine",
            use_gpu=False,  # テストはCPUで
            nlist=16,  # 小さな値でテスト
            seed=42
        )
        
        builder_ivf.embeddings = embeddings
        builder_ivf.ids = ids
        builder_ivf.dim = dim
        
        if builder_ivf.normalize:
            faiss.normalize_L2(builder_ivf.embeddings)
            
        # 小規模パラメータで索引構築
        index_ivf = builder_ivf.build_index(
            pq_m=8,
            pq_bits=8,
            train_size=500,
            add_batch_size=200
        )
        
        # 近傍検索
        distances_ivf, indices_ivf = builder_ivf.search_neighbors(topk=5, batch_size=100)
        
        # エッジ生成
        df_edges_ivf = builder_ivf.create_graph_edges(
            distances_ivf, indices_ivf, undirected=True, remove_self_loops=True
        )
        
        # HNSW テスト
        logger.info("   HNSW索引テスト...")
        builder_hnsw = FAISSIndexBuilder(
            index_type="hnsw_flat",
            metric="l2",
            use_gpu=False,
            nlist=None,  # HNSWでは使わない
            seed=42
        )
        
        builder_hnsw.embeddings = embeddings.copy()
        builder_hnsw.ids = ids.copy()
        builder_hnsw.dim = dim
        
        index_hnsw = builder_hnsw.build_index(hnsw_m=16, efc=100, add_batch_size=200)
        distances_hnsw, indices_hnsw = builder_hnsw.search_neighbors(topk=5, batch_size=100)
        df_edges_hnsw = builder_hnsw.create_graph_edges(distances_hnsw, indices_hnsw)
        
        # 結果検証
        assert distances_ivf.shape == (n_samples, 5), f"IVF距離形状エラー: {distances_ivf.shape}"
        assert indices_ivf.shape == (n_samples, 5), f"IVFインデックス形状エラー: {indices_ivf.shape}"
        assert len(df_edges_ivf) > 0, "IVFエッジ生成失敗"
        
        assert distances_hnsw.shape == (n_samples, 5), f"HNSW距離形状エラー: {distances_hnsw.shape}"
        assert indices_hnsw.shape == (n_samples, 5), f"HNSWインデックス形状エラー: {indices_hnsw.shape}"
        assert len(df_edges_hnsw) > 0, "HNSWエッジ生成失敗"
        
        logger.info("✅ 単体テスト成功")
        logger.info(f"   IVF結果: 距離範囲 [{distances_ivf.min():.6f}, {distances_ivf.max():.6f}], エッジ数 {len(df_edges_ivf):,}")
        logger.info(f"   HNSW結果: 距離範囲 [{distances_hnsw.min():.6f}, {distances_hnsw.max():.6f}], エッジ数 {len(df_edges_hnsw):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 単体テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="GraphRAG動的CFS-Chameleon向けFAISS索引構築 (Step-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  # LaMP-2ユーザー (cosine, IVF+PQ, GPU)
  python build_faiss_index.py \\
    --emb_npy ./embeddings/lamp2_user_embeddings.npy \\
    --ids_json ./embeddings/lamp2_user_embeddings.index.json \\
    --metric cosine --normalize \\
    --index_type ivf_pq --nlist 4096 --pq_m 16 --pq_bits 8 \\
    --train_size 200000 --topk 10 \\
    --out_dir ./faiss/lamp2_users --use_gpu \\
    --batch_size 8192 --seed 42

  # Tenrecユーザー (l2, HNSW, CPU)
  python build_faiss_index.py \\
    --emb_parquet ./embeddings/tenrec_user_embeddings.parquet \\
    --metric l2 \\
    --index_type hnsw_flat --hnsw_m 32 --efc 200 --efs 128 \\
    --topk 20 \\
    --out_dir ./faiss/tenrec_users --batch_size 4096 --seed 42

  # Tenrecアイテム (cosine, IVF+PQ, 無向グラフ)
  python build_faiss_index.py \\
    --emb_npy ./embeddings/tenrec_item_embeddings.npy \\
    --ids_json ./embeddings/tenrec_item_embeddings.index.json \\
    --metric cosine --normalize \\
    --index_type ivf_pq --nlist 8192 --pq_m 32 --pq_bits 8 \\
    --train_size 400000 --topk 15 --undirected \\
    --out_dir ./faiss/tenrec_items --use_gpu \\
    --batch_size 4096 --seed 42

  # 単体テスト実行
  python build_faiss_index.py --self_test
        """
    )
    
    # 入力データ（排他的）
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--emb_parquet', type=str,
                            help='Parquet埋め込みファイル（列: id, embedding）')
    
    # NPY+JSON入力グループ
    npy_group = parser.add_argument_group('NPY+JSON入力')
    npy_group.add_argument('--emb_npy', type=str, help='NPY埋め込みファイル')
    npy_group.add_argument('--ids_json', type=str, help='JSON IDリストファイル')
    
    # 必須引数（テスト時は不要）
    parser.add_argument('--out_dir', type=str, required='--self_test' not in sys.argv,
                        help='出力ディレクトリ')
    
    # 索引設定
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'],
                        help='距離メトリック (default: cosine)')
    parser.add_argument('--normalize', action='store_true',
                        help='明示的L2正規化フラグ（cosineは自動でTrue）')
    parser.add_argument('--index_type', type=str, default='ivf_pq',
                        choices=['ivf_pq', 'hnsw_flat'],
                        help='索引タイプ (default: ivf_pq)')
    
    # 近傍検索
    parser.add_argument('--topk', type=int, default=10,
                        help='近傍数 (default: 10)')
    parser.add_argument('--undirected', action='store_true',
                        help='無向グラフエッジ生成')
    
    # IVF+PQパラメータ
    ivf_group = parser.add_argument_group('IVF+PQ設定')
    ivf_group.add_argument('--nlist', type=int, default=None,
                          help='IVFのクラスタ数。未指定なら自動最適化(≈4*sqrt(N), 64〜16384)')
    ivf_group.add_argument('--pq_m', type=int, default=16,
                          help='PQ分割数 (default: 16)')
    ivf_group.add_argument('--pq_bits', type=int, default=8,
                          help='PQビット数 (default: 8)')
    ivf_group.add_argument('--train_size', type=int, default=200000,
                          help='訓練サンプル数上限 (default: 200000)')
    
    # HNSWパラメータ
    hnsw_group = parser.add_argument_group('HNSW設定')
    hnsw_group.add_argument('--hnsw_m', type=int, default=32,
                           help='HNSW接続数 (default: 32)')
    hnsw_group.add_argument('--efc', type=int, default=200,
                           help='HNSW構築efConstruction (default: 200)')
    hnsw_group.add_argument('--efs', type=int, default=128,
                           help='HNSW検索efSearch (default: 128)')
    
    # 実行環境
    parser.add_argument('--use_gpu', type=str, default='auto',
                        help='GPU使用 (auto/true/false, default: auto)')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='検索バッチサイズ (default: 8192)')
    parser.add_argument('--add_batch_size', type=int, default=10000,
                        help='索引追加バッチサイズ (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (default: 42)')
    
    # テスト・デバッグ
    parser.add_argument('--self_test', action='store_true',
                        help='単体テスト実行')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細ログ表示')
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # 単体テスト実行
    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)
        
    # 引数検証
    if not args.emb_parquet and not (args.emb_npy and args.ids_json):
        parser.error("--emb_parquet または (--emb_npy + --ids_json) のいずれかを指定してください")
        
    if not args.out_dir:
        parser.error("--out_dir は必須です")
        
    try:
        # GPU設定変換
        if args.use_gpu.lower() == 'auto':
            use_gpu = 'auto'
        elif args.use_gpu.lower() in ['true', '1', 'yes']:
            use_gpu = True
        else:
            use_gpu = False
            
        # FAISS索引ビルダー初期化
        builder = FAISSIndexBuilder(
            index_type=args.index_type,
            metric=args.metric,
            normalize=args.normalize,  # Noneの場合、cosineなら自動でTrue
            use_gpu=use_gpu,
            nlist=args.nlist,
            seed=args.seed
        )
        
        # 埋め込みデータ読み込み
        embeddings, ids = builder.load_embeddings(
            emb_npy=args.emb_npy,
            ids_json=args.ids_json,
            emb_parquet=args.emb_parquet
        )
        
        # 索引構築
        index = builder.build_index(
            pq_m=args.pq_m,
            pq_bits=args.pq_bits,
            hnsw_m=args.hnsw_m,
            efc=args.efc,
            train_size=args.train_size,
            add_batch_size=args.add_batch_size
        )
        
        # 近傍検索
        distances, indices = builder.search_neighbors(
            topk=args.topk,
            batch_size=args.batch_size,
            efs=args.efs
        )
        
        # グラフエッジ生成
        df_edges = builder.create_graph_edges(
            distances=distances,
            indices=indices,
            undirected=args.undirected,
            remove_self_loops=True
        )
        
        # 結果保存
        saved_files = builder.save_results(
            distances=distances,
            indices=indices,
            df_edges=df_edges,
            output_dir=args.out_dir,
            # メタデータ用パラメータ
            nlist=args.nlist,
            pq_m=args.pq_m,
            pq_bits=args.pq_bits,
            hnsw_m=args.hnsw_m,
            efc=args.efc,
            efs=args.efs,
            train_size=args.train_size,
            add_batch_size=args.add_batch_size,
            topk=args.topk,
            batch_size=args.batch_size,
            undirected=args.undirected
        )
        
        # 完了レポート
        print(f"\n🎉 FAISS索引構築完了!")
        print(f"📊 処理統計:")
        print(f"   索引タイプ: {args.index_type}")
        print(f"   メトリック: {args.metric}")
        print(f"   L2正規化: {builder.normalize}")
        print(f"   ベクトル数: {len(embeddings):,}")
        print(f"   次元数: {builder.dim}")
        print(f"   Top-k: {args.topk}")
        print(f"   エッジ数: {len(df_edges):,}")
        print(f"   GPU使用: {builder.use_gpu}")
        print(f"   出力先: {args.out_dir}")
        
        print(f"\n📁 保存ファイル:")
        for file_type, file_path in saved_files.items():
            print(f"   {file_type}: {file_path}")
            
    except Exception as e:
        logger.error(f"❌ 実行失敗: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()