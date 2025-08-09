#!/usr/bin/env python3
"""
integrate_ppr_cfs.py

GraphRAG Step-4: PPR出力をCFS-Chameleon協調重み付けに統合

PPRスコア (ppr_topk.parquet) とユーザ埋め込み (theta_p.npy/theta_n.npy) から
協調ユーザプールの重み付きリストを生成する。

Usage:
    python integrate_ppr_cfs.py \
      --ppr ./ppr_results/ppr_topk.parquet \
      --id_map ./ppr_results/id_map.json \
      --theta_p ./chameleon_prime_personalization/processed/LaMP-2/theta_p.npy \
      --theta_n ./chameleon_prime_personalization/processed/LaMP-2/theta_n.npy \
      --out ./graphrag_cfs_weights \
      --gamma 1.0 --beta 0.3 --min_ppr 1e-4 --min_cos 0.2 --topM 50

Input:
    - ppr_topk.parquet: PPR結果 (seed_id, node_id, score)
    - id_map.json: 外部ID↔内部ID マッピング  
    - theta_p.npy/theta_n.npy: ユーザ埋め込み
    
Output:
    - cfs_pool.parquet: (user_id, neighbor_id, weight)
    - meta.json: ハイパーパラメータ設定

Algorithm:
    weight = (ppr_score^gamma) * (cos_sim(u, v)^beta)
    フィルタ: ppr_score >= min_ppr, cos_sim >= min_cos
    制限: 各ユーザ最大 topM 近傍

Author: Claude Code Assistant
Date: 2025-08-10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PPRCFSIntegrator:
    """PPR出力をCFS協調重み付けに統合するクラス"""
    
    def __init__(self, 
                 gamma: float = 1.0,
                 beta: float = 0.3,
                 min_ppr: float = 1e-4,
                 min_cos: float = 0.2,
                 top_m: int = 50,
                 diversity_enabled: bool = False,
                 max_per_cluster: int = 10,
                 random_seed: int = 42):
        """
        Args:
            gamma: PPRスコアの指数 weight = ppr_score^gamma
            beta: コサイン類似度の指数  
            min_ppr: PPRスコア最小閾値
            min_cos: コサイン類似度最小閾値
            top_m: 各ユーザの最大近傍数
            diversity_enabled: 多様性正則化の有効化
            max_per_cluster: クラスタごとの最大近傍数
            random_seed: 乱数シード
        """
        self.gamma = gamma
        self.beta = beta
        self.min_ppr = min_ppr
        self.min_cos = min_cos
        self.top_m = top_m
        self.diversity_enabled = diversity_enabled
        self.max_per_cluster = max_per_cluster
        self.random_seed = random_seed
        
        # 乱数シード設定
        np.random.seed(random_seed)
        
        logger.info(f"🔧 PPRCFSIntegrator初期化完了")
        logger.info(f"   gamma={gamma}, beta={beta}")
        logger.info(f"   min_ppr={min_ppr}, min_cos={min_cos}")
        logger.info(f"   top_m={top_m}, diversity={diversity_enabled}")
        
    def load_ppr_data(self, ppr_path: str, id_map: Dict[int, str]) -> pd.DataFrame:
        """PPRデータを読み込み、文字列IDを内部インデックスにマッピング"""
        logger.info(f"📂 PPRデータ読み込み: {ppr_path}")
        
        if ppr_path.endswith('.parquet'):
            ppr_df = pd.read_parquet(ppr_path)
        else:
            # ディレクトリの場合、ppr_topk.parquetを探す
            ppr_file = Path(ppr_path) / "ppr_topk.parquet"
            if not ppr_file.exists():
                raise FileNotFoundError(f"PPRファイルが見つかりません: {ppr_file}")
            ppr_df = pd.read_parquet(ppr_file)
        
        # 必須カラムチェック
        required_cols = ['seed_id', 'node_id', 'score']
        missing_cols = [col for col in required_cols if col not in ppr_df.columns]
        if missing_cols:
            raise ValueError(f"必須カラムが不足: {missing_cols}")
        
        # ID変換マッピング作成（外部ID文字列 → 内部インデックス）
        reverse_id_map = {v: k for k, v in id_map.items()} if id_map else {}
        
        def extract_user_id(id_str):
            """文字列IDからuser_idを抽出"""
            try:
                id_dict = eval(id_str)
                if isinstance(id_dict, dict) and 'user_id' in id_dict:
                    return id_dict['user_id']
                return None
            except:
                return None
        
        def map_to_internal_id(id_str):
            """外部ID文字列を内部インデックスにマッピング"""
            user_id = extract_user_id(id_str)
            if user_id is None:
                return None
            return reverse_id_map.get(user_id, None)
        
        logger.info("🔄 PPR ID変換中...")
        
        # ID変換
        ppr_df['seed_internal_id'] = ppr_df['seed_id'].apply(map_to_internal_id)
        ppr_df['node_internal_id'] = ppr_df['node_id'].apply(map_to_internal_id)
        
        # 変換できなかったエントリを除去
        original_size = len(ppr_df)
        ppr_df = ppr_df.dropna(subset=['seed_internal_id', 'node_internal_id'])
        ppr_df['seed_internal_id'] = ppr_df['seed_internal_id'].astype(int)
        ppr_df['node_internal_id'] = ppr_df['node_internal_id'].astype(int)
        id_mapped_size = len(ppr_df)
        
        # PPRスコアフィルタ
        ppr_df = ppr_df[ppr_df['score'] >= self.min_ppr]
        filtered_size = len(ppr_df)
        
        logger.info(f"✅ PPRデータ読み込み完了")
        logger.info(f"   総エントリ数: {original_size:,}")
        logger.info(f"   ID変換後: {id_mapped_size:,}")
        logger.info(f"   フィルタ後: {filtered_size:,} (min_ppr={self.min_ppr})")
        logger.info(f"   ユニークseed: {ppr_df['seed_internal_id'].nunique():,}")
        logger.info(f"   ユニークnode: {ppr_df['node_internal_id'].nunique():,}")
        logger.info(f"   スコア範囲: [{ppr_df['score'].min():.6f}, {ppr_df['score'].max():.6f}]")
        
        return ppr_df
    
    def load_id_mapping(self, id_map_path: Optional[str]) -> Dict[int, str]:
        """ID マッピングを読み込み（内部ID → 外部ID）"""
        if id_map_path and os.path.exists(id_map_path):
            logger.info(f"📂 IDマッピング読み込み: {id_map_path}")
            with open(id_map_path, 'r') as f:
                id_map_data = json.load(f)
            
            # 形式に応じて変換
            if isinstance(id_map_data, dict):
                if 'id_to_idx' in id_map_data:
                    # LaMP形式: {"id_to_idx": {"{'user_id': '100', 'n_docs': 1}": 0, ...}}
                    # 外部ID文字列から実際のuser_idを抽出
                    id_map = {}
                    for external_id_str, internal_idx in id_map_data['id_to_idx'].items():
                        try:
                            # 文字列を評価してuser_idを抽出
                            id_dict = eval(external_id_str)
                            if isinstance(id_dict, dict) and 'user_id' in id_dict:
                                user_id = id_dict['user_id']
                                id_map[internal_idx] = user_id
                            else:
                                logger.warning(f"⚠️ IDフォーマット不正: {external_id_str}")
                        except Exception as e:
                            logger.warning(f"⚠️ ID解析エラー: {external_id_str} - {e}")
                            continue
                elif 'id_to_index' in id_map_data:
                    # 標準形式: {external_id: internal_index} → {internal_index: external_id}
                    id_map = {v: k for k, v in id_map_data['id_to_index'].items()}
                else:
                    # 直接辞書形式（数値キーをintに変換）
                    id_map = {}
                    for k, v in id_map_data.items():
                        try:
                            id_map[int(k)] = str(v)
                        except ValueError:
                            logger.warning(f"⚠️ 数値変換不可: {k}")
                            continue
            else:
                raise ValueError(f"IDマッピング形式が不正: {type(id_map_data)}")
                
            logger.info(f"✅ IDマッピング読み込み完了: {len(id_map):,}エントリ")
        else:
            logger.warning("⚠️ IDマッピングファイルなし - 内部IDをそのまま使用")
            id_map = {}
        
        return id_map
    
    def load_user_embeddings(self, user_embeddings_path: str) -> np.ndarray:
        """ユーザ埋め込みを読み込み"""
        if not os.path.exists(user_embeddings_path):
            raise FileNotFoundError(f"ユーザ埋め込みファイルが見つかりません: {user_embeddings_path}")
        
        logger.info(f"📂 ユーザ埋め込み読み込み: {user_embeddings_path}")
        embeddings = np.load(user_embeddings_path).astype(np.float32)
        
        logger.info(f"✅ ユーザ埋め込み読み込み完了")
        logger.info(f"   形状: {embeddings.shape}")
        logger.info(f"   データ型: {embeddings.dtype}")
        logger.info(f"   値範囲: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        
        # 2次元チェック
        if len(embeddings.shape) != 2:
            raise ValueError(f"埋め込みは2次元である必要があります: {embeddings.shape}")
        
        logger.info(f"   ノルム統計: mean={np.linalg.norm(embeddings, axis=1).mean():.4f}")
        
        return embeddings
    
    def compute_collaborative_weights(self, 
                                    ppr_df: pd.DataFrame,
                                    embeddings: np.ndarray,
                                    id_map: Dict[int, str]) -> pd.DataFrame:
        """協調重み計算のメイン処理"""
        logger.info("🧮 協調重み計算開始")
        
        # ユーザリスト取得（内部IDを使用）
        unique_users = ppr_df['seed_internal_id'].unique()
        total_users = len(unique_users)
        
        logger.info(f"   処理ユーザ数: {total_users:,}")
        logger.info(f"   埋め込み数: {len(embeddings):,}")
        
        # 多様性クラスタリング（オプション）
        cluster_labels = None
        if self.diversity_enabled:
            logger.info("🔄 多様性正則化用クラスタリング実行")
            n_clusters = min(50, len(embeddings) // 10)  # 適応的クラスタ数
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            logger.info(f"   クラスタ数: {n_clusters}, max_per_cluster: {self.max_per_cluster}")
        
        # 結果蓄積用リスト
        results = []
        
        # 各ユーザに対する処理
        for user_internal_id in tqdm(unique_users, desc="協調重み計算"):
            # 該当ユーザのPPRエントリ取得
            user_ppr = ppr_df[ppr_df['seed_internal_id'] == user_internal_id].copy()
            
            if len(user_ppr) == 0:
                logger.warning(f"⚠️ User {user_internal_id}: PPRエントリなし")
                continue
            
            # ユーザ埋め込み取得（境界チェック）
            if user_internal_id >= len(embeddings):
                logger.warning(f"⚠️ User {user_internal_id}: 埋め込み範囲外 ({len(embeddings)})")
                continue
            
            user_embedding = embeddings[user_internal_id].reshape(1, -1)
            
            # 近傍埋め込み取得
            neighbor_internal_ids = user_ppr['node_internal_id'].values
            valid_neighbors = neighbor_internal_ids[neighbor_internal_ids < len(embeddings)]
            
            if len(valid_neighbors) == 0:
                logger.warning(f"⚠️ User {user_internal_id}: 有効な近傍なし")
                continue
            
            neighbor_embeddings = embeddings[valid_neighbors]
            
            # コサイン類似度計算
            cos_similarities = cosine_similarity(user_embedding, neighbor_embeddings).flatten()
            
            # フィルタ適用
            valid_mask = (cos_similarities >= self.min_cos) & \
                        (user_ppr['node_internal_id'].isin(valid_neighbors))
            
            if not valid_mask.any():
                logger.warning(f"⚠️ User {user_internal_id}: フィルタ後に近傍なし")
                continue
            
            # フィルタ後データ
            filtered_ppr = user_ppr[valid_mask].copy()
            
            # 対応するコサイン類似度の取得
            neighbor_to_cos = dict(zip(valid_neighbors, cos_similarities))
            filtered_cos = np.array([neighbor_to_cos[nid] for nid in filtered_ppr['node_internal_id']])
            
            # 重み計算: weight = (ppr_score^gamma) * (cos_sim^beta)
            ppr_weights = np.power(filtered_ppr['score'].values, self.gamma)
            cos_weights = np.power(filtered_cos, self.beta)
            final_weights = ppr_weights * cos_weights
            
            # データフレームに追加
            filtered_ppr = filtered_ppr.copy()
            filtered_ppr['cos_sim'] = filtered_cos
            filtered_ppr['weight'] = final_weights
            
            # 多様性制約（オプション）
            if self.diversity_enabled and cluster_labels is not None:
                filtered_ppr = self._apply_diversity_constraint(
                    filtered_ppr, cluster_labels
                )
            
            # Top-M選択
            top_neighbors = filtered_ppr.nlargest(self.top_m, 'weight')
            
            # 結果に追加（外部ID変換）
            for _, row in top_neighbors.iterrows():
                neighbor_internal_id = row['node_internal_id']
                external_user_id = id_map.get(user_internal_id, str(user_internal_id))
                external_neighbor_id = id_map.get(neighbor_internal_id, str(neighbor_internal_id))
                
                results.append({
                    'user_id': external_user_id,
                    'neighbor_id': external_neighbor_id,
                    'weight': row['weight'],
                    'ppr_score': row['score'],
                    'cos_sim': row['cos_sim']
                })
        
        # データフレーム化
        result_df = pd.DataFrame(results)
        
        # データ型最適化
        if len(result_df) > 0:
            result_df['weight'] = result_df['weight'].astype(np.float32)
            result_df['ppr_score'] = result_df['ppr_score'].astype(np.float32)
            result_df['cos_sim'] = result_df['cos_sim'].astype(np.float32)
        
        logger.info(f"✅ 協調重み計算完了")
        logger.info(f"   処理済みユーザ: {result_df['user_id'].nunique():,}/{total_users:,}")
        logger.info(f"   総エッジ数: {len(result_df):,}")
        logger.info(f"   平均近傍数/ユーザ: {len(result_df)/result_df['user_id'].nunique():.1f}")
        
        if len(result_df) > 0:
            logger.info(f"   重み統計: [{result_df['weight'].min():.6f}, {result_df['weight'].max():.6f}]")
            logger.info(f"   重み分布: median={result_df['weight'].median():.6f}, 95%={result_df['weight'].quantile(0.95):.6f}")
        
        return result_df
    
    def _apply_diversity_constraint(self, 
                                  neighbor_df: pd.DataFrame,
                                  cluster_labels: np.ndarray) -> pd.DataFrame:
        """多様性制約を適用（クラスタごとにmax_per_cluster制限）"""
        if len(neighbor_df) <= self.max_per_cluster:
            return neighbor_df
        
        # クラスタラベル追加
        neighbor_df = neighbor_df.copy()
        neighbor_df['cluster'] = cluster_labels[neighbor_df['node_internal_id'].values]
        
        # クラスタごとにTop-N選択
        diverse_results = []
        for cluster_id in neighbor_df['cluster'].unique():
            cluster_neighbors = neighbor_df[neighbor_df['cluster'] == cluster_id]
            top_in_cluster = cluster_neighbors.nlargest(self.max_per_cluster, 'weight')
            diverse_results.append(top_in_cluster)
        
        result = pd.concat(diverse_results, ignore_index=True)
        return result.drop('cluster', axis=1)
    
    def save_results(self, 
                    result_df: pd.DataFrame,
                    output_dir: str,
                    meta_info: Dict) -> None:
        """結果を保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 結果保存開始: {output_path}")
        
        # メインデータ保存
        cfs_pool_path = output_path / "cfs_pool.parquet"
        result_df.to_parquet(cfs_pool_path, index=False)
        logger.info(f"✅ 協調プール保存: {cfs_pool_path}")
        
        # メタデータ保存
        meta_path = output_path / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ メタデータ保存: {meta_path}")
        
        logger.info(f"🎯 保存完了: {len(result_df):,} エントリ")
    
    def run_integration(self,
                       ppr_path: str,
                       id_map_path: Optional[str],
                       user_embeddings_path: str,
                       output_dir: str) -> None:
        """統合処理のメインエントリーポイント"""
        start_time = time.time()
        
        logger.info("🚀 PPR-CFS統合処理開始")
        logger.info(f"   PPRデータ: {ppr_path}")
        logger.info(f"   IDマップ: {id_map_path}")
        logger.info(f"   ユーザ埋め込み: {user_embeddings_path}")
        logger.info(f"   出力先: {output_dir}")
        
        try:
            # データ読み込み（順序重要：ID mapを先に読み込んでからPPRデータを処理）
            id_map = self.load_id_mapping(id_map_path)
            embeddings = self.load_user_embeddings(user_embeddings_path)
            ppr_df = self.load_ppr_data(ppr_path, id_map)
            
            # ID整合性チェック
            max_ppr_internal_id = max(ppr_df['seed_internal_id'].max(), ppr_df['node_internal_id'].max())
            if max_ppr_internal_id >= len(embeddings):
                logger.warning(f"⚠️ ID範囲不整合: PPR最大内部ID={max_ppr_internal_id}, 埋め込み数={len(embeddings)}")
            
            # 協調重み計算
            result_df = self.compute_collaborative_weights(ppr_df, embeddings, id_map)
            
            # 空結果チェック
            if len(result_df) == 0:
                raise RuntimeError("協調プール生成失敗: 結果が空です")
            
            # メタデータ準備
            meta_info = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_sec': time.time() - start_time,
                'parameters': {
                    'gamma': self.gamma,
                    'beta': self.beta,
                    'min_ppr': self.min_ppr,
                    'min_cos': self.min_cos,
                    'top_m': self.top_m,
                    'diversity_enabled': self.diversity_enabled,
                    'max_per_cluster': self.max_per_cluster,
                    'random_seed': self.random_seed
                },
                'statistics': {
                    'total_users': int(result_df['user_id'].nunique()),
                    'total_edges': int(len(result_df)),
                    'avg_neighbors_per_user': float(len(result_df) / result_df['user_id'].nunique()),
                    'weight_min': float(result_df['weight'].min()),
                    'weight_max': float(result_df['weight'].max()),
                    'weight_median': float(result_df['weight'].median()),
                    'weight_95pct': float(result_df['weight'].quantile(0.95))
                }
            }
            
            # 結果保存
            self.save_results(result_df, output_dir, meta_info)
            
            execution_time = time.time() - start_time
            logger.info(f"🎉 PPR-CFS統合完了!")
            logger.info(f"   実行時間: {execution_time:.1f}s")
            logger.info(f"   処理速度: {len(result_df)/execution_time:.0f} edges/sec")
            
        except Exception as e:
            logger.error(f"❌ PPR-CFS統合失敗: {e}")
            raise


def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="PPR出力をCFS-Chameleon協調重み付けに統合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
    python integrate_ppr_cfs.py \\
      --ppr ./ppr_results/ppr_topk.parquet \\
      --id_map ./ppr_results/id_map.json \\
      --user_embeddings ./embeddings/lamp2_user_embeddings.npy \\
      --out ./graphrag_cfs_weights \\
      --gamma 1.0 --beta 0.3 --min_ppr 1e-4 --min_cos 0.2 --topM 50

出力:
    ./graphrag_cfs_weights/
    ├── cfs_pool.parquet    # (user_id, neighbor_id, weight)
    └── meta.json           # ハイパーパラメータとメタデータ
        """
    )
    
    # 必須引数
    parser.add_argument('--ppr', required=True,
                       help='PPRデータ (ppr_topk.parquet またはそのディレクトリ)')
    parser.add_argument('--user_embeddings', required=True,
                       help='ユーザ埋め込み (*.npy)')
    parser.add_argument('--out', required=True,
                       help='出力ディレクトリ')
    
    # オプション引数
    parser.add_argument('--id_map', default=None,
                       help='IDマッピング (id_map.json)')
    
    # ハイパーパラメータ
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='PPRスコア指数 (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='コサイン類似度指数 (default: 0.3)')
    parser.add_argument('--min_ppr', type=float, default=1e-4,
                       help='PPRスコア最小閾値 (default: 1e-4)')
    parser.add_argument('--min_cos', type=float, default=0.2,
                       help='コサイン類似度最小閾値 (default: 0.2)')
    parser.add_argument('--topM', type=int, default=50,
                       help='各ユーザの最大近傍数 (default: 50)')
    
    # 多様性制約
    parser.add_argument('--diversity', action='store_true',
                       help='多様性正則化を有効にする')
    parser.add_argument('--max_per_cluster', type=int, default=10,
                       help='クラスタごとの最大近傍数 (default: 10)')
    
    # その他
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細ログ出力')
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # ログレベル調整
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 引数検証
    if not os.path.exists(args.ppr):
        raise FileNotFoundError(f"PPRデータが見つかりません: {args.ppr}")
    
    if not os.path.exists(args.user_embeddings):
        raise FileNotFoundError(f"ユーザ埋め込みファイルが見つかりません: {args.user_embeddings}")
    
    # 統合処理実行
    integrator = PPRCFSIntegrator(
        gamma=args.gamma,
        beta=args.beta,
        min_ppr=args.min_ppr,
        min_cos=args.min_cos,
        top_m=args.topM,
        diversity_enabled=args.diversity,
        max_per_cluster=args.max_per_cluster,
        random_seed=args.seed
    )
    
    integrator.run_integration(
        ppr_path=args.ppr,
        id_map_path=args.id_map,
        user_embeddings_path=args.user_embeddings,
        output_dir=args.out
    )


if __name__ == "__main__":
    main()