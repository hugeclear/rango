#!/usr/bin/env python3
"""
Production Pipeline: Graph-based Chameleon Reranking
Tenrec推薦システム + グラフ協調フィルタリング + Chameleon再ランキング
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import time
from datetime import datetime
from collections import defaultdict

def setup_production_logging():
    """本番CLI経路ログセットアップ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # ログ先頭: which python / pandas.__file__
    try:
        python_path = subprocess.check_output(['which', 'python'], text=True).strip()
        logger.info(f"Python executable: {python_path}")
    except:
        logger.info(f"Python executable: {sys.executable}")
    
    import pandas
    logger.info(f"Pandas location: {pandas.__file__}")
    
    return logger

def validate_config(args) -> Dict[str, Any]:
    """設定検証（空/同一/PLACEHOLDER検出で即エラー）"""
    logger = logging.getLogger(__name__)
    
    # 必須引数検証
    if not args.tenrec_data or args.tenrec_data.strip() == "" or "PLACEHOLDER" in args.tenrec_data:
        logger.error("ERROR: --tenrec-data is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.ppr_scores or args.ppr_scores.strip() == "" or "PLACEHOLDER" in args.ppr_scores:
        logger.error("ERROR: --ppr-scores is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.item_embeddings or args.item_embeddings.strip() == "" or "PLACEHOLDER" in args.item_embeddings:
        logger.error("ERROR: --item-embeddings is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.out_dir or args.out_dir.strip() == "" or "PLACEHOLDER" in args.out_dir:
        logger.error("ERROR: --out-dir is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # パラメータ検証
    if args.top_items <= 0:
        logger.error(f"ERROR: --top-items must be positive, got {args.top_items}")
        sys.exit(1)
    
    if args.rerank_size <= 0:
        logger.error(f"ERROR: --rerank-size must be positive, got {args.rerank_size}")
        sys.exit(1)
    
    if args.limit <= 0:
        logger.error(f"ERROR: --limit must be positive, got {args.limit}")
        sys.exit(1)
    
    if args.seed < 0:
        logger.error(f"ERROR: --seed must be non-negative, got {args.seed}")
        sys.exit(1)
    
    if not 0.0 <= args.alpha <= 1.0:
        logger.error(f"ERROR: --alpha must be in [0,1], got {args.alpha}")
        sys.exit(1)
    
    if args.collab_weight < 0.0 or args.collab_weight > 1.0:
        logger.error(f"ERROR: --collab-weight must be in [0,1], got {args.collab_weight}")
        sys.exit(1)
    
    config = {
        'tenrec_data_path': Path(args.tenrec_data.strip()),
        'ppr_scores_path': Path(args.ppr_scores.strip()),
        'item_embeddings_path': Path(args.item_embeddings.strip()),
        'out_dir': Path(args.out_dir.strip()),
        'top_items': args.top_items,
        'rerank_size': args.rerank_size,
        'limit': args.limit,
        'seed': args.seed,
        'alpha': args.alpha,
        'beta': getattr(args, 'beta', -0.1),
        'collab_weight': args.collab_weight,
        'temperature': getattr(args, 'temperature', 0.3),
        'model_path': getattr(args, 'model_path', './chameleon_prime_personalization/models/base_model')
    }
    
    # ファイル存在確認
    required_files = [
        ('tenrec_data_path', 'Tenrec data file'),
        ('ppr_scores_path', 'PPR scores file'),
        ('item_embeddings_path', 'Item embeddings file')
    ]
    
    for path_key, description in required_files:
        if not config[path_key].exists():
            logger.error(f"ERROR: {description} not found: {config[path_key]}")
            sys.exit(1)
    
    return config

def load_tenrec_dataset(tenrec_path: Path, limit: int) -> Tuple[List[Dict], Dict[str, Any]]:
    """Tenrecデータセット読み込み"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading Tenrec dataset: {tenrec_path}")
    
    try:
        if tenrec_path.suffix == '.json':
            with open(tenrec_path, 'r') as f:
                data = json.load(f)
        elif tenrec_path.suffix == '.jsonl':
            data = []
            with open(tenrec_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif tenrec_path.suffix == '.csv':
            df = pd.read_csv(tenrec_path)
            data = df.to_dict('records')
        else:
            logger.error(f"ERROR: Unsupported Tenrec file format: {tenrec_path.suffix}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(data)} Tenrec interactions")
        
    except Exception as e:
        logger.error(f"Failed to load Tenrec data: {e}")
        sys.exit(1)
    
    # データ形式の標準化
    standardized_data = []
    user_item_interactions = defaultdict(list)
    
    for i, item in enumerate(data):
        if i >= limit:
            break
        
        # 標準フィールド抽出
        user_id = item.get('user_id') or item.get('userId') or item.get('user')
        item_id = item.get('item_id') or item.get('itemId') or item.get('item')
        rating = item.get('rating', 1.0) or item.get('score', 1.0)
        timestamp = item.get('timestamp') or item.get('time', 0)
        
        if user_id is not None and item_id is not None:
            interaction = {
                'user_id': str(user_id),
                'item_id': str(item_id),
                'rating': float(rating),
                'timestamp': int(timestamp),
                'original_data': item
            }
            
            standardized_data.append(interaction)
            user_item_interactions[str(user_id)].append(str(item_id))
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1} interactions...")
    
    # 統計情報
    unique_users = len(user_item_interactions)
    unique_items = len(set(item['item_id'] for item in standardized_data))
    avg_interactions_per_user = np.mean([len(items) for items in user_item_interactions.values()])
    
    stats = {
        'total_interactions': len(standardized_data),
        'unique_users': unique_users,
        'unique_items': unique_items,
        'avg_interactions_per_user': avg_interactions_per_user,
        'sparsity': len(standardized_data) / (unique_users * unique_items) if unique_users > 0 and unique_items > 0 else 0
    }
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Interactions: {stats['total_interactions']:,}")
    logger.info(f"  Users: {stats['unique_users']:,}")
    logger.info(f"  Items: {stats['unique_items']:,}")
    logger.info(f"  Avg interactions/user: {stats['avg_interactions_per_user']:.1f}")
    logger.info(f"  Sparsity: {stats['sparsity']:.6f}")
    
    return standardized_data, stats

def load_embeddings_and_ppr(embeddings_path: Path, ppr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """アイテム埋め込み+PPRスコア読み込み"""
    logger = logging.getLogger(__name__)
    
    # アイテム埋め込み読み込み
    logger.info(f"Loading item embeddings: {embeddings_path}")
    try:
        item_embeddings = np.load(embeddings_path)
        logger.info(f"Loaded item embeddings: {item_embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to load item embeddings: {e}")
        sys.exit(1)
    
    # PPRスコア読み込み
    logger.info(f"Loading PPR scores: {ppr_path}")
    try:
        if ppr_path.suffix == '.npz':
            data = np.load(ppr_path)
            if 'indices' in data and 'scores' in data:
                ppr_indices = data['indices']
                ppr_scores = data['scores']
                
                # スパース→デンス変換
                n_users = ppr_indices.shape[0]
                max_idx = ppr_indices.max() + 1
                full_ppr = np.zeros((n_users, max_idx), dtype=np.float32)
                
                for i in range(n_users):
                    full_ppr[i, ppr_indices[i]] = ppr_scores[i]
                
                ppr_matrix = full_ppr
            else:
                ppr_matrix = data['arr_0'] if 'arr_0' in data else np.array([])
        else:
            ppr_matrix = np.load(ppr_path)
        
        logger.info(f"Loaded PPR scores: {ppr_matrix.shape}")
    except Exception as e:
        logger.error(f"Failed to load PPR scores: {e}")
        sys.exit(1)
    
    return item_embeddings, ppr_matrix

def create_collaborative_recommender(interactions: List[Dict], ppr_matrix: np.ndarray, 
                                   item_embeddings: np.ndarray, config: Dict[str, Any]) -> Any:
    """協調フィルタリング推薦システム作成"""
    logger = logging.getLogger(__name__)
    
    class GraphCollaborativeRecommender:
        def __init__(self, interactions: List[Dict], ppr_matrix: np.ndarray, 
                     item_embeddings: np.ndarray, config: Dict[str, Any]):
            self.interactions = interactions
            self.ppr_matrix = ppr_matrix
            self.item_embeddings = item_embeddings
            self.config = config
            
            # ユーザー-アイテム辞書構築
            self.user_items = defaultdict(set)
            self.item_users = defaultdict(set)
            self.user_id_map = {}
            self.item_id_map = {}
            
            # ID→インデックスマッピング
            unique_users = sorted(set(item['user_id'] for item in interactions))
            unique_items = sorted(set(item['item_id'] for item in interactions))
            
            self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
            self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
            
            # インタラクション構築
            for interaction in interactions:
                user_id = interaction['user_id']
                item_id = interaction['item_id']
                
                self.user_items[user_id].add(item_id)
                self.item_users[item_id].add(user_id)
            
            logger.info(f"CollaborativeRecommender initialized: {len(unique_users)} users, {len(unique_items)} items")
        
        def get_candidate_items(self, user_id: str, top_k: int) -> List[Tuple[str, float]]:
            """候補アイテム取得（協調フィルタリング）"""
            if user_id not in self.user_id_map:
                # 新規ユーザー: 人気アイテム返却
                item_popularity = defaultdict(int)
                for interaction in self.interactions:
                    item_popularity[interaction['item_id']] += 1
                
                popular_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
                return [(item_id, score) for item_id, score in popular_items[:top_k]]
            
            user_idx = self.user_id_map[user_id]
            user_interacted_items = self.user_items[user_id]
            
            # PPRベース協調ユーザー取得
            if user_idx < len(self.ppr_matrix):
                user_ppr = self.ppr_matrix[user_idx]
                similar_user_indices = np.argsort(user_ppr)[-20:][::-1]  # Top-20類似ユーザー
            else:
                similar_user_indices = []
            
            # 協調アイテムスコア計算
            item_scores = defaultdict(float)
            
            for similar_user_idx in similar_user_indices:
                if similar_user_idx >= len(self.user_id_map) or user_ppr[similar_user_idx] < 0.01:
                    continue
                
                # 類似ユーザーのアイテム取得
                similar_user_id = list(self.user_id_map.keys())[similar_user_idx]
                similar_user_items = self.user_items.get(similar_user_id, set())
                
                # スコア加算（PPR重み付け）
                ppr_weight = user_ppr[similar_user_idx]
                for item_id in similar_user_items:
                    if item_id not in user_interacted_items:  # 既交流アイテム除外
                        item_scores[item_id] += ppr_weight
            
            # コンテンツベース類似度追加（アイテム埋め込み）
            if len(user_interacted_items) > 0 and len(self.item_embeddings) > 0:
                user_profile = np.zeros(self.item_embeddings.shape[1])
                profile_count = 0
                
                for item_id in user_interacted_items:
                    if item_id in self.item_id_map:
                        item_idx = self.item_id_map[item_id]
                        if item_idx < len(self.item_embeddings):
                            user_profile += self.item_embeddings[item_idx]
                            profile_count += 1
                
                if profile_count > 0:
                    user_profile /= profile_count
                    
                    # 全アイテムとの類似度計算
                    similarities = np.dot(self.item_embeddings, user_profile)
                    
                    for item_id, item_idx in self.item_id_map.items():
                        if item_id not in user_interacted_items and item_idx < len(similarities):
                            content_score = similarities[item_idx]
                            item_scores[item_id] += self.config['collab_weight'] * content_score
            
            # トップKアイテム返却
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_k]
        
        def get_item_features(self, item_id: str) -> Dict[str, Any]:
            """アイテム特徴量取得"""
            features = {'item_id': item_id}
            
            # 埋め込みベクトル
            if item_id in self.item_id_map:
                item_idx = self.item_id_map[item_id]
                if item_idx < len(self.item_embeddings):
                    features['embedding'] = self.item_embeddings[item_idx]
            
            # 人気度
            popularity = sum(1 for interaction in self.interactions if interaction['item_id'] == item_id)
            features['popularity'] = popularity
            
            return features
    
    return GraphCollaborativeRecommender(interactions, ppr_matrix, item_embeddings, config)

def create_chameleon_reranker(model_path: str, config: Dict[str, Any]) -> Any:
    """Chameleon再ランキングシステム作成"""
    logger = logging.getLogger(__name__)
    
    try:
        sys.path.append('.')
        from chameleon_editor import ChameleonEditor
    except ImportError:
        logger.error("ERROR: ChameleonEditor not found, check import path")
        sys.exit(1)
    
    class ChameleonReranker(ChameleonEditor):
        def __init__(self, model_path: str, config: Dict[str, Any]):
            super().__init__(model_path)
            self.config = config
            
            logger.info(f"ChameleonReranker initialized: alpha={config['alpha']}, beta={config.get('beta', 0)}")
        
        def rerank_items(self, user_id: str, candidate_items: List[Tuple[str, float]], 
                        user_history: List[str], rerank_size: int) -> List[Tuple[str, float]]:
            """Chameleon再ランキング"""
            if not candidate_items:
                return []
            
            # 再ランキング対象選択
            top_candidates = candidate_items[:rerank_size]
            remaining_candidates = candidate_items[rerank_size:]
            
            # 各アイテムのChameleonスコア計算
            reranked_scores = []
            
            for item_id, cf_score in top_candidates:
                try:
                    # プロンプト構築
                    prompt = self._build_rerank_prompt(user_id, item_id, user_history)
                    
                    # Chameleonスコア生成
                    chameleon_score = self._compute_chameleon_preference_score(prompt)
                    
                    # 協調フィルタリングスコアと組み合わせ
                    combined_score = (1 - self.config['alpha']) * cf_score + self.config['alpha'] * chameleon_score
                    
                    reranked_scores.append((item_id, combined_score))
                    
                except Exception as e:
                    logger.warning(f"Chameleon reranking failed for item {item_id}: {e}")
                    reranked_scores.append((item_id, cf_score))  # フォールバック
            
            # 再ランキング結果 + 残り候補結合
            reranked_scores.sort(key=lambda x: x[1], reverse=True)
            final_ranking = reranked_scores + remaining_candidates
            
            return final_ranking
        
        def _build_rerank_prompt(self, user_id: str, item_id: str, user_history: List[str]) -> str:
            """再ランキングプロンプト構築"""
            # ユーザー履歴サマリー
            if user_history:
                history_text = f"User {user_id} previously interacted with: {', '.join(user_history[-5:])}"
            else:
                history_text = f"User {user_id} is a new user"
            
            # 推薦プロンプト
            prompt = f"{history_text}\n\nWould user {user_id} be interested in item {item_id}?\nAnswer with a score from 0-10:"
            
            return prompt
        
        def _compute_chameleon_preference_score(self, prompt: str) -> float:
            """Chameleon好み度スコア計算"""
            try:
                # Chameleon生成
                response = self.generate_with_chameleon(prompt, max_new_tokens=10, temperature=self.config['temperature'])
                
                # スコア抽出
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', response)
                if score_match:
                    score = float(score_match.group(1))
                    # 0-10スケールを0-1に正規化
                    normalized_score = min(max(score / 10.0, 0.0), 1.0)
                    return normalized_score
                else:
                    return 0.5  # デフォルト中性スコア
                    
            except Exception as e:
                logger.warning(f"Chameleon score computation failed: {e}")
                return 0.5
    
    return ChameleonReranker(model_path, config)

def run_reranking_evaluation(dataset: List[Dict], recommender: Any, reranker: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """再ランキング評価実行"""
    logger = logging.getLogger(__name__)
    
    # ユーザー別にデータ分割（学習/テスト）
    user_interactions = defaultdict(list)
    for interaction in dataset:
        user_interactions[interaction['user_id']].append(interaction)
    
    # 十分なインタラクションがあるユーザーのみ評価
    eval_users = [user_id for user_id, interactions in user_interactions.items() 
                  if len(interactions) >= 5]
    
    logger.info(f"Evaluating on {len(eval_users)} users with >= 5 interactions")
    
    results = []
    total_time = 0
    
    for i, user_id in enumerate(eval_users):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing user {i + 1}/{len(eval_users)}")
        
        try:
            user_interactions_list = user_interactions[user_id]
            
            # 時系列ソート
            user_interactions_list.sort(key=lambda x: x['timestamp'])
            
            # 学習/テストスプリット（最後の20%をテスト）
            split_idx = int(0.8 * len(user_interactions_list))
            train_interactions = user_interactions_list[:split_idx]
            test_interactions = user_interactions_list[split_idx:]
            
            if len(test_interactions) == 0:
                continue
            
            # ユーザー履歴（学習用）
            user_history = [item['item_id'] for item in train_interactions]
            
            start_time = time.time()
            
            # 候補アイテム取得
            candidates = recommender.get_candidate_items(user_id, config['top_items'])
            
            if not candidates:
                continue
            
            # Chameleon再ランキング
            reranked_items = reranker.rerank_items(
                user_id, candidates, user_history, config['rerank_size']
            )
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # 評価メトリクス計算
            test_items = set(item['item_id'] for item in test_interactions)
            
            # Hit Rate@K
            recommended_items = [item_id for item_id, score in reranked_items[:10]]
            hits = len(set(recommended_items) & test_items)
            hit_rate = hits / min(len(test_items), 10)
            
            # NDCG@K
            ndcg_score = compute_ndcg(recommended_items, test_items, k=10)
            
            user_result = {
                'user_id': user_id,
                'hit_rate_10': hit_rate,
                'ndcg_10': ndcg_score,
                'num_candidates': len(candidates),
                'num_test_items': len(test_items),
                'processing_time': end_time - start_time
            }
            
            results.append(user_result)
            
        except Exception as e:
            logger.warning(f"Evaluation failed for user {user_id}: {e}")
    
    # 全体統計
    if results:
        avg_hit_rate = np.mean([r['hit_rate_10'] for r in results])
        avg_ndcg = np.mean([r['ndcg_10'] for r in results])
        avg_time = total_time / len(results)
    else:
        avg_hit_rate = avg_ndcg = avg_time = 0.0
    
    evaluation_results = {
        'user_results': results,
        'overall_metrics': {
            'avg_hit_rate_10': avg_hit_rate,
            'avg_ndcg_10': avg_ndcg,
            'avg_processing_time': avg_time,
            'num_evaluated_users': len(results)
        },
        'config': config
    }
    
    logger.info(f"Evaluation completed: Hit@10={avg_hit_rate:.4f}, NDCG@10={avg_ndcg:.4f}")
    
    return evaluation_results

def compute_ndcg(recommended: List[str], relevant: set, k: int) -> float:
    """NDCG@K計算"""
    if not recommended or not relevant:
        return 0.0
    
    # DCG計算
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # IDCG計算
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def save_reranking_results(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """再ランキング結果保存"""
    logger = logging.getLogger(__name__)
    
    # 実行ID生成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"graph_rerank_tenrec_{timestamp}"
    
    # 出力ディレクトリ作成
    run_dir = config['out_dir'] / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # メイン結果保存
    results_file = run_dir / 'reranking_results.json'
    
    save_data = {
        'run_id': run_id,
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    logger.info(f"Saving reranking results: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    # サマリー保存
    summary_file = run_dir / 'summary.json'
    
    summary = {
        'run_id': run_id,
        'overall_metrics': results['overall_metrics'],
        'dataset_stats': {
            'total_users_evaluated': len(results['user_results']),
            'avg_hit_rate_10': results['overall_metrics']['avg_hit_rate_10'],
            'avg_ndcg_10': results['overall_metrics']['avg_ndcg_10'],
            'avg_processing_time': results['overall_metrics']['avg_processing_time']
        }
    }
    
    logger.info(f"Saving summary: {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return str(run_dir)

def main():
    """メイン処理"""
    logger = setup_production_logging()
    
    parser = argparse.ArgumentParser(description="Graph-based Chameleon Reranking for Tenrec")
    parser.add_argument('--tenrec-data', required=True,
                       help='Path to Tenrec dataset file')
    parser.add_argument('--ppr-scores', required=True,
                       help='Path to PPR scores (.npy or .npz)')
    parser.add_argument('--item-embeddings', required=True,
                       help='Path to item embeddings (.npy)')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--top-items', type=int, default=100,
                       help='Number of candidate items (default: 100)')
    parser.add_argument('--rerank-size', type=int, default=20,
                       help='Number of items to rerank (default: 20)')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Max interactions to process (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Chameleon weight in reranking (default: 0.3)')
    parser.add_argument('--beta', type=float, default=-0.1,
                       help='Chameleon beta parameter (default: -0.1)')
    parser.add_argument('--collab-weight', type=float, default=0.7,
                       help='Collaborative filtering weight (default: 0.7)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Generation temperature (default: 0.3)')
    parser.add_argument('--model-path', default='./chameleon_prime_personalization/models/base_model',
                       help='Path to base model')
    
    args = parser.parse_args()
    
    # 設定検証
    config = validate_config(args)
    
    # Effective-config 1行出力
    effective_config = f"tenrec_data={config['tenrec_data_path'].name}, top_items={config['top_items']}, rerank_size={config['rerank_size']}, limit={config['limit']}, alpha={config['alpha']}, collab_weight={config['collab_weight']}, seed={config['seed']}"
    logger.info(f"Effective-config: {effective_config}")
    
    try:
        # データ読み込み
        logger.info("=== STEP 1: Loading Tenrec dataset ===")
        dataset, dataset_stats = load_tenrec_dataset(config['tenrec_data_path'], config['limit'])
        
        # 埋め込み+PPR読み込み
        logger.info("=== STEP 2: Loading embeddings and PPR scores ===")
        item_embeddings, ppr_matrix = load_embeddings_and_ppr(
            config['item_embeddings_path'], config['ppr_scores_path']
        )
        
        # 推薦システム作成
        logger.info("=== STEP 3: Creating collaborative recommender ===")
        recommender = create_collaborative_recommender(dataset, ppr_matrix, item_embeddings, config)
        
        # 再ランキングシステム作成
        logger.info("=== STEP 4: Creating Chameleon reranker ===")
        reranker = create_chameleon_reranker(config['model_path'], config)
        
        # 評価実行
        logger.info("=== STEP 5: Running reranking evaluation ===")
        results = run_reranking_evaluation(dataset, recommender, reranker, config)
        
        # 結果保存
        logger.info("=== STEP 6: Saving results ===")
        run_dir = save_reranking_results(results, config)
        
        logger.info(f"[OK] Graph-based Chameleon reranking completed")
        logger.info(f"Results saved in: {run_dir}")
        logger.info(f"Hit@10: {results['overall_metrics']['avg_hit_rate_10']:.4f}")
        logger.info(f"NDCG@10: {results['overall_metrics']['avg_ndcg_10']:.4f}")
        logger.info(f"Avg time/user: {results['overall_metrics']['avg_processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to run graph Chameleon reranking: {e}")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check Tenrec dataset file format and content")
        logger.error("  - Verify PPR scores and embeddings files are compatible")
        logger.error("  - Ensure model path exists and is accessible")
        logger.error("  - Check parameter ranges are reasonable")
        logger.error("  - Install required packages: torch, transformers, numpy, pandas")
        sys.exit(1)
    
    logger.info("Graph-based Chameleon reranking completed successfully")

if __name__ == "__main__":
    main()