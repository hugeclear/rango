#!/usr/bin/env python3
"""
GraphRAG動的CFS-Chameleon向けPersonalized PageRank (PPR) システム (Step-3)
Step-2の疎グラフエッジからCSR行列を構築し、Power Iterationで近似PPRを計算

## 機能:
- 入力: Step-2のgraph_edges.parquet（src_id, dst_id, score）
- 前処理: ID再マッピング、重複エッジ統合、自己ループ除去
- CSR疎隣接行列構築、行正規化で確率遷移行列化
- Power IterationによるPPR近似（p = α·e + (1-α)·Pᵀ·p）
- 収束判定、上位L保持、複数シード対応

## 対象ベンチマーク:
- LaMP-2（生成タスク）+ Tenrec（ランキングタスク）
- GraphRAG動的CFS-Chameleonの協調ユーザー選択
"""

import os
import sys
import json
import argparse
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

# 依存関係チェック
try:
    from scipy.sparse import csr_matrix, save_npz, load_npz
    from scipy.sparse.csgraph import connected_components
    import pyarrow.parquet as pq
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ 依存パッケージが不足: {e}")
    print("pip install numpy scipy pandas pyarrow tqdm を実行してください")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)

class PPRCalculator:
    """Personalized PageRank 計算システム"""
    
    def __init__(self, 
                 alpha: float = 0.15,
                 eps: float = 1e-6,
                 max_iter: int = 50,
                 normalize_edge: bool = True,
                 undirected: bool = False,
                 seed: int = 42):
        """
        初期化
        
        Args:
            alpha: 再始動確率（典型的に0.1-0.2）
            eps: 収束判定閾値
            max_iter: 最大反復回数
            normalize_edge: エッジ重み行正規化するか
            undirected: 無向グラフ化するか
            seed: 乱数シード
        """
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        self.normalize_edge = normalize_edge
        self.undirected = undirected
        self.seed = seed
        
        # 乱数シード固定
        np.random.seed(seed)
        
        # 内部状態
        self.adj_matrix = None
        self.id_to_idx = None
        self.idx_to_id = None
        self.n_nodes = 0
        
        logger.info(f"🎯 PPRCalculator初期化")
        logger.info(f"   再始動確率: {alpha}")
        logger.info(f"   収束閾値: {eps}")
        logger.info(f"   最大反復: {max_iter}")
        logger.info(f"   エッジ正規化: {normalize_edge}")
        logger.info(f"   無向グラフ: {undirected}")
        
    def load_graph_edges(self, edges_path: str) -> pd.DataFrame:
        """
        グラフエッジ読み込み
        
        Args:
            edges_path: graph_edges.parquetのパス
            
        Returns:
            エッジDataFrame
        """
        logger.info(f"📂 グラフエッジ読み込み: {edges_path}")
        
        if not Path(edges_path).exists():
            raise RuntimeError(f"❌ CRITICAL: エッジファイルが見つかりません: {edges_path}")
            
        try:
            df_edges = pd.read_parquet(edges_path)
            
            # 必要列チェック
            required_cols = ['src_id', 'dst_id', 'score']
            missing_cols = [col for col in required_cols if col not in df_edges.columns]
            if missing_cols:
                raise RuntimeError(f"❌ CRITICAL: 必要な列が不足: {missing_cols}")
            
            # データ型確保
            df_edges['src_id'] = df_edges['src_id'].astype(str)
            df_edges['dst_id'] = df_edges['dst_id'].astype(str)
            df_edges['score'] = df_edges['score'].astype(float)
            
            logger.info(f"✅ エッジ読み込み完了: {len(df_edges):,} エッジ")
            
            # 基本統計
            unique_nodes = len(set(df_edges['src_id'].tolist() + df_edges['dst_id'].tolist()))
            score_stats = df_edges['score'].describe()
            
            logger.info(f"📊 グラフ統計:")
            logger.info(f"   ユニークノード数: {unique_nodes:,}")
            logger.info(f"   スコア統計: min={score_stats['min']:.6f}, max={score_stats['max']:.6f}, mean={score_stats['mean']:.6f}")
            
            return df_edges
            
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: エッジ読み込み失敗: {e}")
            
    def load_user_ids(self, ids_path: str) -> List[str]:
        """
        ユーザーIDリスト読み込み
        
        Args:
            ids_path: Step-1のindex.jsonパス
            
        Returns:
            IDリスト
        """
        logger.info(f"📂 ユーザーID読み込み: {ids_path}")
        
        if not Path(ids_path).exists():
            raise RuntimeError(f"❌ CRITICAL: IDファイルが見つかりません: {ids_path}")
            
        try:
            with open(ids_path, 'r', encoding='utf-8') as f:
                ids_data = json.load(f)
            
            # リスト形式か確認
            if isinstance(ids_data, list):
                # 単純なリスト形式 (例: ["100", "101", ...])
                if len(ids_data) > 0 and isinstance(ids_data[0], (str, int)):
                    ids = [str(id_val) for id_val in ids_data]
                # オブジェクトリスト形式 (例: [{"user_id": "100", "n_docs": 1}, ...])
                elif len(ids_data) > 0 and isinstance(ids_data[0], dict):
                    if 'user_id' in ids_data[0]:
                        ids = [str(item['user_id']) for item in ids_data]
                    else:
                        raise RuntimeError(f"❌ CRITICAL: オブジェクト形式の場合、'user_id'フィールドが必要です")
                else:
                    raise RuntimeError(f"❌ CRITICAL: 空のIDリストまたは未知の形式です")
            else:
                raise RuntimeError(f"❌ CRITICAL: IDファイルはリスト形式である必要があります")
                
            logger.info(f"✅ ユーザーID読み込み完了: {len(ids):,} ユーザー")
            return ids
            
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: ユーザーID読み込み失敗: {e}")
            
    def preprocess_edges(self, df_edges: pd.DataFrame) -> pd.DataFrame:
        """
        エッジ前処理（重複統合、自己ループ除去、検証）
        
        Args:
            df_edges: 生エッジDataFrame
            
        Returns:
            前処理済みエッジDataFrame
        """
        logger.info("🔄 エッジ前処理開始")
        
        # 元の統計
        original_edges = len(df_edges)
        
        # 1. スコア負値チェック
        negative_scores = df_edges['score'] < 0
        if negative_scores.any():
            n_negative = negative_scores.sum()
            logger.warning(f"⚠️ 負のスコアを検出: {n_negative} エッジ、0にクリップ")
            df_edges.loc[negative_scores, 'score'] = 0.0
            
        # 2. 自己ループ除去
        self_loops = df_edges['src_id'] == df_edges['dst_id']
        if self_loops.any():
            n_self_loops = self_loops.sum()
            logger.info(f"   自己ループ除去: {n_self_loops} エッジ")
            df_edges = df_edges[~self_loops]
            
        # 3. 重複エッジ統合（最大スコア採用）
        logger.info("   重複エッジ統合中...")
        df_edges = df_edges.groupby(['src_id', 'dst_id'], as_index=False)['score'].max()
        
        # 4. 無向グラフ化（指定時）
        if self.undirected:
            logger.info("   無向グラフ化実行中...")
            
            # 逆向きエッジ生成
            df_reverse = df_edges.copy()
            df_reverse = df_reverse.rename(columns={'src_id': 'dst_id', 'dst_id': 'src_id'})
            
            # 結合・重複処理（平均スコア採用）
            df_combined = pd.concat([df_edges, df_reverse], ignore_index=True)
            df_edges = df_combined.groupby(['src_id', 'dst_id'], as_index=False)['score'].mean()
            
        # 5. ゼロスコアエッジ除去
        zero_scores = df_edges['score'] == 0.0
        if zero_scores.any():
            n_zero = zero_scores.sum()
            logger.info(f"   ゼロスコアエッジ除去: {n_zero} エッジ")
            df_edges = df_edges[~zero_scores]
            
        # 前処理後統計
        processed_edges = len(df_edges)
        unique_nodes = len(set(df_edges['src_id'].tolist() + df_edges['dst_id'].tolist()))
        
        logger.info(f"✅ エッジ前処理完了:")
        logger.info(f"   元エッジ数: {original_edges:,}")
        logger.info(f"   処理後エッジ数: {processed_edges:,}")
        logger.info(f"   ユニークノード数: {unique_nodes:,}")
        
        return df_edges
        
    def build_csr_matrix(self, df_edges: pd.DataFrame, user_ids: List[str]) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
        """
        CSR疎隣接行列構築
        
        Args:
            df_edges: 前処理済みエッジDataFrame
            user_ids: 全ユーザーIDリスト
            
        Returns:
            (CSR隣接行列, ID→インデックスマップ, インデックス→IDマップ)
        """
        logger.info("🔧 CSR疎隣接行列構築開始")
        
        # ID→インデックスマッピング構築
        all_node_ids = set(user_ids)
        edge_node_ids = set(df_edges['src_id'].tolist() + df_edges['dst_id'].tolist())
        
        # エッジに存在しないユーザーも含める（孤立ノード対応）
        missing_in_edges = all_node_ids - edge_node_ids
        if missing_in_edges:
            logger.info(f"   エッジに存在しない孤立ノード: {len(missing_in_edges):,}")
            
        # 統合ノードリスト作成
        final_node_ids = sorted(list(all_node_ids))
        self.n_nodes = len(final_node_ids)
        
        # ID←→インデックスマッピング
        self.id_to_idx = {node_id: i for i, node_id in enumerate(final_node_ids)}
        self.idx_to_id = {i: node_id for node_id, i in self.id_to_idx.items()}
        
        logger.info(f"   ノード総数: {self.n_nodes:,}")
        logger.info(f"   エッジ数: {len(df_edges):,}")
        
        # CSR行列データ準備
        rows = []
        cols = []
        data = []
        
        for _, edge in tqdm(df_edges.iterrows(), total=len(df_edges), desc="CSR構築", unit="edges", leave=False):
            src_idx = self.id_to_idx.get(edge['src_id'])
            dst_idx = self.id_to_idx.get(edge['dst_id'])
            
            if src_idx is not None and dst_idx is not None:
                rows.append(src_idx)
                cols.append(dst_idx)
                data.append(edge['score'])
            else:
                logger.debug(f"未知ノードをスキップ: {edge['src_id']} -> {edge['dst_id']}")
                
        # CSR行列構築
        try:
            adj_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32)
            
            # 行正規化（確率遷移行列化）
            if self.normalize_edge:
                logger.info("   行正規化実行中...")
                row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
                
                # ゼロ行（出次数なし）の処理
                zero_rows = row_sums == 0
                n_zero_rows = zero_rows.sum()
                if n_zero_rows > 0:
                    logger.info(f"   出次数ゼロノード: {n_zero_rows} (全ノードに等確率遷移)")
                    # ゼロ行は均等分布で置き換え
                    row_sums[zero_rows] = 1.0
                    
                # 行正規化
                row_sums_inv = 1.0 / row_sums
                row_sums_inv[~np.isfinite(row_sums_inv)] = 0.0
                
                # 行正規化適用
                adj_matrix = adj_matrix.multiply(row_sums_inv[:, np.newaxis])
                
                # ゼロ行の均等分布設定
                if n_zero_rows > 0:
                    uniform_prob = 1.0 / self.n_nodes
                    # CSR形式で直接設定はできないため、COO形式を使用
                    adj_matrix = adj_matrix.tocoo()
                    
                    # ゼロ行のインデックスを取得
                    zero_row_indices = np.where(zero_rows)[0]
                    
                    # 既存のデータを保持
                    rows = adj_matrix.row.tolist()
                    cols = adj_matrix.col.tolist()
                    data = adj_matrix.data.tolist()
                    
                    # ゼロ行に均等分布を追加
                    for row_idx in zero_row_indices:
                        for col_idx in range(self.n_nodes):
                            rows.append(row_idx)
                            cols.append(col_idx)
                            data.append(uniform_prob)
                    
                    # 新しいCSR行列を構築
                    adj_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32)
                        
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: CSR行列構築失敗: {e}")
            
        # 行列統計
        density = adj_matrix.nnz / (self.n_nodes * self.n_nodes)
        
        logger.info(f"✅ CSR行列構築完了:")
        logger.info(f"   行列サイズ: {adj_matrix.shape}")
        logger.info(f"   非ゼロ要素: {adj_matrix.nnz:,}")
        logger.info(f"   密度: {density:.8f}")
        logger.info(f"   メモリ使用量: {adj_matrix.data.nbytes / 1024**2:.1f} MB")
        
        self.adj_matrix = adj_matrix
        return adj_matrix, self.id_to_idx, self.idx_to_id
        
    def compute_ppr(self, 
                   seed_nodes: Union[int, List[int]], 
                   topL: int = 50) -> Dict[int, np.ndarray]:
        """
        Power IterationによるPPR計算
        
        Args:
            seed_nodes: シードノードインデックス（単一またはリスト）
            topL: 上位L個を保持
            
        Returns:
            {seed_idx: ppr_vector} の辞書
        """
        if self.adj_matrix is None:
            raise RuntimeError("❌ CRITICAL: CSR行列が構築されていません")
            
        # シードノードリスト化
        if isinstance(seed_nodes, int):
            seed_nodes = [seed_nodes]
            
        logger.info(f"🧮 PPR計算開始: {len(seed_nodes)} シードノード")
        logger.info(f"   上位保持数: {topL}")
        
        results = {}
        
        for seed_idx in tqdm(seed_nodes, desc="PPR計算", unit="seeds"):
            if seed_idx < 0 or seed_idx >= self.n_nodes:
                logger.warning(f"⚠️ 無効シードインデックス: {seed_idx}")
                continue
                
            # 初期化ベクトル（シードノードで1、他は0）
            e = np.zeros(self.n_nodes, dtype=np.float32)
            e[seed_idx] = 1.0
            
            # PPRベクトル初期化（均等分布）
            p = np.ones(self.n_nodes, dtype=np.float32) / self.n_nodes
            
            # Power Iteration
            for iteration in range(self.max_iter):
                p_prev = p.copy()
                
                # p = α * e + (1 - α) * P^T * p
                # P^T * p は転置行列との積（CSRでは .T.dot() を使用）
                p = self.alpha * e + (1 - self.alpha) * self.adj_matrix.T.dot(p)
                
                # L1正規化（確率分布として維持）
                p = p / p.sum()
                
                # 収束判定
                l1_diff = np.sum(np.abs(p - p_prev))
                if l1_diff < self.eps:
                    logger.debug(f"  シード {seed_idx}: 反復 {iteration+1} で収束 (L1差分: {l1_diff:.2e})")
                    break
            else:
                logger.warning(f"⚠️ シード {seed_idx}: 最大反復数 {self.max_iter} で未収束")
                
            # 上位L個保持（メモリ効率化）
            if topL < self.n_nodes:
                top_indices = np.argpartition(p, -topL)[-topL:]
                top_indices = top_indices[np.argsort(p[top_indices])[::-1]]
                
                # スパース化
                p_sparse = np.zeros_like(p)
                p_sparse[top_indices] = p[top_indices]
                p = p_sparse
                
            results[seed_idx] = p
            
        logger.info(f"✅ PPR計算完了: {len(results)} シード処理済み")
        return results
        
    def compute_graph_statistics(self) -> Dict[str, Any]:
        """
        グラフ統計情報計算
        
        Returns:
            統計情報辞書
        """
        if self.adj_matrix is None:
            raise RuntimeError("❌ CRITICAL: CSR行列が構築されていません")
            
        logger.info("📊 グラフ統計計算中...")
        
        try:
            # 基本統計
            n_nodes = self.adj_matrix.shape[0]
            n_edges = self.adj_matrix.nnz
            density = n_edges / (n_nodes * n_nodes) if n_nodes > 0 else 0.0
            
            # 次数統計
            out_degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()
            in_degrees = np.array(self.adj_matrix.sum(axis=0)).flatten()
            
            # 連結成分数
            n_components, labels = connected_components(
                self.adj_matrix, directed=not self.undirected, return_labels=True
            )
            
            # 最大連結成分サイズ
            component_sizes = np.bincount(labels)
            largest_component_size = component_sizes.max() if len(component_sizes) > 0 else 0
            
            stats = {
                'n_nodes': int(n_nodes),
                'n_edges': int(n_edges),
                'density': float(density),
                'avg_out_degree': float(out_degrees.mean()),
                'avg_in_degree': float(in_degrees.mean()),
                'max_out_degree': int(out_degrees.max()),
                'max_in_degree': int(in_degrees.max()),
                'n_connected_components': int(n_components),
                'largest_component_size': int(largest_component_size),
                'largest_component_ratio': float(largest_component_size / n_nodes) if n_nodes > 0 else 0.0,
                'is_undirected': self.undirected,
                'is_normalized': self.normalize_edge,
            }
            
            logger.info(f"✅ グラフ統計完了:")
            logger.info(f"   ノード数: {stats['n_nodes']:,}")
            logger.info(f"   エッジ数: {stats['n_edges']:,}")
            logger.info(f"   平均出次数: {stats['avg_out_degree']:.2f}")
            logger.info(f"   連結成分数: {stats['n_connected_components']}")
            logger.info(f"   最大成分比率: {stats['largest_component_ratio']:.1%}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ グラフ統計計算失敗: {e}")
            return {}
            
    def save_results(self, 
                    ppr_results: Dict[int, np.ndarray],
                    output_dir: str,
                    topL: int = 50,
                    graph_stats: Dict[str, Any] = None) -> Dict[str, str]:
        """
        結果保存
        
        Args:
            ppr_results: PPR計算結果
            output_dir: 出力ディレクトリ
            topL: 上位保持数
            graph_stats: グラフ統計（オプション）
            
        Returns:
            保存ファイルパス辞書
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 結果保存開始: {output_dir}")
        saved_files = {}
        
        # 1. CSR行列保存
        try:
            csr_path = output_path / "graph_csr.npz"
            save_npz(csr_path, self.adj_matrix)
            saved_files['csr'] = str(csr_path)
            logger.info(f"✅ CSR行列保存完了: {csr_path}")
        except Exception as e:
            logger.error(f"❌ CSR行列保存失敗: {e}")
            
        # 2. IDマッピング保存
        try:
            id_map_path = output_path / "id_map.json"
            id_mapping = {
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'n_nodes': self.n_nodes
            }
            with open(id_map_path, 'w', encoding='utf-8') as f:
                json.dump(id_mapping, f, ensure_ascii=False, indent=2)
            saved_files['id_map'] = str(id_map_path)
            logger.info(f"✅ IDマッピング保存完了: {id_map_path}")
        except Exception as e:
            logger.error(f"❌ IDマッピング保存失敗: {e}")
            
        # 3. PPR結果Parquet保存
        try:
            ppr_data = []
            
            for seed_idx, ppr_vector in ppr_results.items():
                seed_id = self.idx_to_id[seed_idx]
                
                # 非ゼロ要素のみ抽出（上位L個保持のため）
                nonzero_indices = np.nonzero(ppr_vector)[0]
                
                if len(nonzero_indices) == 0:
                    continue
                    
                # スコア順ソート
                scores = ppr_vector[nonzero_indices]
                sorted_indices = np.argsort(scores)[::-1]
                
                for rank, idx in enumerate(sorted_indices[:topL]):
                    node_idx = nonzero_indices[idx]
                    node_id = self.idx_to_id[node_idx]
                    score = scores[idx]
                    
                    ppr_data.append({
                        'seed_id': seed_id,
                        'rank': rank + 1,
                        'node_id': node_id,
                        'score': float(score)
                    })
                    
            if ppr_data:
                df_ppr = pd.DataFrame(ppr_data)
                ppr_path = output_path / "ppr_topk.parquet"
                df_ppr.to_parquet(ppr_path, index=False)
                saved_files['ppr'] = str(ppr_path)
                logger.info(f"✅ PPR結果保存完了: {ppr_path} ({len(ppr_data):,} レコード)")
            else:
                logger.warning("⚠️ PPR結果が空のため保存をスキップ")
                
        except Exception as e:
            logger.error(f"❌ PPR結果保存失敗: {e}")
            
        # 4. グラフ統計保存
        if graph_stats:
            try:
                stats_path = output_path / "graph_stats.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_stats, f, ensure_ascii=False, indent=2)
                saved_files['stats'] = str(stats_path)
                logger.info(f"✅ グラフ統計保存完了: {stats_path}")
            except Exception as e:
                logger.error(f"❌ グラフ統計保存失敗: {e}")
                
        logger.info(f"🎯 保存完了: {len(saved_files)} ファイル")
        return saved_files


def parse_seed_nodes(seed_id: Optional[str] = None, 
                    seed_file: Optional[str] = None,
                    id_to_idx: Dict[str, int] = None) -> List[int]:
    """
    シードノード解析
    
    Args:
        seed_id: 単一シードID
        seed_file: シードIDリストファイル
        id_to_idx: ID→インデックスマッピング
        
    Returns:
        シードインデックスリスト
    """
    if seed_id:
        # 単一シード
        if seed_id not in id_to_idx:
            raise RuntimeError(f"❌ CRITICAL: シードID '{seed_id}' が見つかりません")
        return [id_to_idx[seed_id]]
        
    elif seed_file:
        # ファイルから複数シード
        if not Path(seed_file).exists():
            raise RuntimeError(f"❌ CRITICAL: シードファイルが見つかりません: {seed_file}")
            
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                seed_ids = [line.strip() for line in f if line.strip()]
                
            seed_indices = []
            missing_ids = []
            
            for sid in seed_ids:
                if sid in id_to_idx:
                    seed_indices.append(id_to_idx[sid])
                else:
                    missing_ids.append(sid)
                    
            if missing_ids:
                logger.warning(f"⚠️ 見つからないシードID: {len(missing_ids)} 個")
                for mid in missing_ids[:5]:  # 最初の5個表示
                    logger.warning(f"   - {mid}")
                    
            if not seed_indices:
                raise RuntimeError("❌ CRITICAL: 有効なシードIDが見つかりません")
                
            return seed_indices
            
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: シードファイル読み込み失敗: {e}")
    else:
        # 全ノードをシードとする
        return list(range(len(id_to_idx)))


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="GraphRAG動的CFS-Chameleon向けPersonalized PageRank計算 (Step-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 無向化して全体性を安定化（推奨）
  python run_ppr.py \\
    --edges ./faiss/lamp2_users/graph_edges.parquet \\
    --ids ./embeddings/lamp2_users/embeddings_index.json \\
    --alpha 0.15 --eps 1e-6 --max_iter 50 --topL 50 \\
    --undirected --normalize_edge \\
    --output ./ppr_results

  # 特定ユーザーをシードに個別PPR
  python run_ppr.py \\
    --edges ./faiss/lamp2_users/graph_edges.parquet \\
    --ids ./embeddings/lamp2_users/embeddings_index.json \\
    --alpha 0.2 --eps 1e-6 --max_iter 50 --topL 50 \\
    --undirected --normalize_edge \\
    --seed_id user_123 \\
    --output ./ppr_results_user123

  # 複数シードファイル指定
  python run_ppr.py \\
    --edges ./faiss/tenrec_users/graph_edges.parquet \\
    --ids ./embeddings/tenrec_users/embeddings_index.json \\
    --alpha 0.15 --eps 1e-6 --max_iter 50 --topL 100 \\
    --undirected --normalize_edge \\
    --seed_file ./seed_users.txt \\
    --output ./ppr_results_multi
        """
    )
    
    # 必須引数
    parser.add_argument('--edges', type=str, required=True,
                        help='Step-2のgraph_edges.parquetパス')
    parser.add_argument('--ids', type=str, required=True,
                        help='Step-1のID index.jsonパス')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')
    
    # PPRパラメータ
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='再始動確率 (default: 0.15)')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='収束判定閾値 (default: 1e-6)')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='最大反復回数 (default: 50)')
    parser.add_argument('--topL', type=int, default=50,
                        help='出力上位L個保持 (default: 50)')
    
    # グラフ処理オプション
    parser.add_argument('--normalize_edge', action='store_true', default=True,
                        help='エッジ重み行正規化（確率遷移行列化）')
    parser.add_argument('--no_normalize_edge', action='store_false', dest='normalize_edge',
                        help='エッジ重み正規化を無効化')
    parser.add_argument('--undirected', action='store_true',
                        help='無向グラフ化（双方向エッジ、重み平均）')
    
    # シード指定
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed_id', type=str,
                           help='単一シードノードID')
    seed_group.add_argument('--seed_file', type=str,
                           help='シードIDリストファイル（1行1ID）')
    
    # その他
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細ログ表示')
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # PPR計算機初期化
        ppr_calc = PPRCalculator(
            alpha=args.alpha,
            eps=args.eps,
            max_iter=args.max_iter,
            normalize_edge=args.normalize_edge,
            undirected=args.undirected,
            seed=args.seed
        )
        
        # データ読み込み
        logger.info("🔄 データ読み込み開始")
        df_edges = ppr_calc.load_graph_edges(args.edges)
        user_ids = ppr_calc.load_user_ids(args.ids)
        
        # エッジ前処理
        df_edges = ppr_calc.preprocess_edges(df_edges)
        
        # CSR行列構築
        adj_matrix, id_to_idx, idx_to_id = ppr_calc.build_csr_matrix(df_edges, user_ids)
        
        # シードノード解析
        seed_indices = parse_seed_nodes(args.seed_id, args.seed_file, id_to_idx)
        logger.info(f"🎯 シードノード: {len(seed_indices)} 個")
        
        # グラフ統計計算
        graph_stats = ppr_calc.compute_graph_statistics()
        
        # PPR計算実行
        ppr_results = ppr_calc.compute_ppr(seed_indices, args.topL)
        
        # 結果保存
        saved_files = ppr_calc.save_results(
            ppr_results=ppr_results,
            output_dir=args.output,
            topL=args.topL,
            graph_stats=graph_stats
        )
        
        # 完了レポート
        print(f"\n🎉 PPR計算完了!")
        print(f"📊 処理統計:")
        print(f"   グラフ: {graph_stats.get('n_nodes', 0):,} ノード, {graph_stats.get('n_edges', 0):,} エッジ")
        print(f"   再始動確率: {args.alpha}")
        print(f"   シード数: {len(seed_indices):,}")
        print(f"   上位保持: {args.topL}")
        print(f"   無向グラフ: {args.undirected}")
        print(f"   正規化: {args.normalize_edge}")
        print(f"   出力先: {args.output}")
        
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