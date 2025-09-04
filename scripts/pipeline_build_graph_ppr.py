#!/usr/bin/env python3
"""
Production Pipeline: Build Graph + PPR
余弦近傍CSRグラフ構築 + Personalized PageRank計算
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import numpy.linalg as nla
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import time

def _to_csr(m):
    """疎行列をCSR形式に統一"""
    try:
        from scipy.sparse import csr_matrix, isspmatrix_csr
        if isspmatrix_csr(m):
            return m
        return m.tocsr() if hasattr(m, 'tocsr') else csr_matrix(m)
    except ImportError:
        return m

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
    if not args.embeddings or args.embeddings.strip() == "" or "PLACEHOLDER" in args.embeddings:
        logger.error("ERROR: --embeddings is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.graph_out or args.graph_out.strip() == "" or "PLACEHOLDER" in args.graph_out:
        logger.error("ERROR: --graph-out is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.ppr_out or args.ppr_out.strip() == "" or "PLACEHOLDER" in args.ppr_out:
        logger.error("ERROR: --ppr-out is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # パラメータ検証
    if args.knn <= 0:
        logger.error(f"ERROR: --knn must be positive, got {args.knn}")
        sys.exit(1)
    
    if not 0.0 <= args.alpha <= 1.0:
        logger.error(f"ERROR: --alpha must be in [0,1], got {args.alpha}")
        sys.exit(1)
    
    if args.eps <= 0:
        logger.error(f"ERROR: --eps must be positive, got {args.eps}")
        sys.exit(1)
    
    if args.max_iter <= 0:
        logger.error(f"ERROR: --max-iter must be positive, got {args.max_iter}")
        sys.exit(1)
    
    if args.top_ppr <= 0:
        logger.error(f"ERROR: --top-ppr must be positive, got {args.top_ppr}")
        sys.exit(1)
    
    if args.sim_thresh < 0.0 or args.sim_thresh >= 1.0:
        logger.error(f"ERROR: --sim-thresh must be in [0,1), got {args.sim_thresh}")
        sys.exit(1)
    
    if args.seed < 0:
        logger.error(f"ERROR: --seed must be non-negative, got {args.seed}")
        sys.exit(1)
    
    config = {
        'embeddings_path': Path(args.embeddings.strip()),
        'graph_out_path': Path(args.graph_out.strip()),
        'ppr_out_path': Path(args.ppr_out.strip()),
        'knn': args.knn,
        'alpha': args.alpha,
        'eps': args.eps,
        'max_iter': args.max_iter,
        'top_ppr': args.top_ppr,
        'sim_thresh': args.sim_thresh,
        'seed': args.seed,
        'overwrite': args.overwrite,
        'normalize_embeddings': getattr(args, 'normalize', True),
        'undirected': getattr(args, 'undirected', True)
    }
    
    # 埋め込みファイル存在確認
    if not config['embeddings_path'].exists():
        logger.error(f"ERROR: Embeddings file not found: {config['embeddings_path']}")
        sys.exit(1)
    
    return config

def load_embeddings_and_index(embeddings_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """埋め込み+インデックス読み込み"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading embeddings: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        sys.exit(1)
    
    # インデックスファイル検索
    index_path = None
    for candidate in [
        embeddings_path.parent / (embeddings_path.stem + "_index.json"),
        embeddings_path.with_suffix(".json"),
        embeddings_path.parent / "index.json"
    ]:
        if candidate.exists():
            index_path = candidate
            break
    
    index_data = {}
    if index_path:
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            logger.info(f"Loaded index metadata: {len(index_data)} fields")
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
    else:
        logger.warning("No index metadata found")
    
    # 基本検証
    if embeddings.size == 0:
        logger.error("ERROR: Empty embeddings array")
        sys.exit(1)
    
    if len(embeddings.shape) != 2:
        logger.error(f"ERROR: Embeddings must be 2D, got shape {embeddings.shape}")
        sys.exit(1)
    
    return embeddings, index_data

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """埋め込み正規化（L2ノルム）"""
    logger = logging.getLogger(__name__)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_norm_count = np.sum(norms == 0)
    
    if zero_norm_count > 0:
        logger.warning(f"Found {zero_norm_count} zero-norm embeddings, replacing with epsilon")
        norms = np.where(norms == 0, 1e-8, norms)
    
    normalized = embeddings / norms
    logger.info(f"Normalized embeddings: mean_norm={np.linalg.norm(normalized, axis=1).mean():.6f}")
    
    return normalized

def build_cosine_similarity_graph(embeddings: np.ndarray, config: Dict[str, Any]) -> Tuple[Any, Any]:
    """余弦類似度グラフ構築（sparse CSR）"""
    logger = logging.getLogger(__name__)
    
    try:
        from scipy.sparse import csr_matrix
        from scipy.spatial.distance import cdist
        import faiss
    except ImportError as e:
        logger.error(f"ERROR: Required package not available: {e}")
        logger.error("Install with: pip install scipy faiss-gpu")
        sys.exit(1)
    
    n_vectors, dim = embeddings.shape
    knn = min(config['knn'], n_vectors - 1)  # 自己除外
    
    logger.info(f"Building cosine similarity graph: {n_vectors} nodes, k={knn}")
    
    # Seed固定
    np.random.seed(config['seed'])
    faiss.omp_set_num_threads(4)  # 並列制御
    
    # 正規化（余弦類似度のため）
    if config['normalize_embeddings']:
        embeddings = normalize_embeddings(embeddings)
    
    # FAISS内積インデックス（正規化済み→余弦類似度）
    logger.info("Building FAISS inner product index for cosine similarity")
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine for normalized vectors
    
    # GPU使用（可能であれば）
    if faiss.get_num_gpus() > 0:
        try:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
            logger.info("Using GPU acceleration for similarity computation")
        except:
            logger.info("GPU acceleration failed, using CPU")
    
    index.add(embeddings.astype(np.float32))
    
    # k-NN検索
    logger.info(f"Performing k-NN search (k={knn+1})...")  # +1 for self-exclusion
    start_time = time.time()
    
    similarities, indices = index.search(embeddings.astype(np.float32), knn + 1)
    
    search_time = time.time() - start_time
    logger.info(f"k-NN search completed in {search_time:.1f}s")
    
    # GPU→CPU移行
    if faiss.get_num_gpus() > 0:
        try:
            index = faiss.index_gpu_to_cpu(index)
        except:
            pass
    
    # 自己参照除去 + 閾値フィルタリング
    logger.info("Building sparse adjacency matrix...")
    
    row_indices = []
    col_indices = []
    data_values = []
    
    for i in range(n_vectors):
        # 自己参照除去
        mask = indices[i] != i
        neighbor_indices = indices[i][mask]
        neighbor_sims = similarities[i][mask]
        
        # 類似度閾値フィルタ
        valid_mask = neighbor_sims >= config['sim_thresh']
        
        if np.any(valid_mask):
            valid_neighbors = neighbor_indices[valid_mask]
            valid_sims = neighbor_sims[valid_mask]
            
            row_indices.extend([i] * len(valid_neighbors))
            col_indices.extend(valid_neighbors.tolist())
            data_values.extend(valid_sims.tolist())
    
    # CSRマトリックス構築
    adjacency_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)), 
        shape=(n_vectors, n_vectors)
    )
    
    # 無向グラフ化（対称化）
    if config['undirected']:
        logger.info("Making graph undirected (symmetrizing adjacency matrix)")
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
    
    edge_count = adjacency_matrix.nnz
    avg_degree = edge_count / n_vectors
    
    logger.info(f"Graph constructed: {edge_count:,} edges, avg_degree={avg_degree:.1f}")
    logger.info(f"Sparsity: {edge_count / (n_vectors ** 2):.6f}")
    
    return adjacency_matrix, {
        'num_nodes': n_vectors,
        'num_edges': edge_count,
        'avg_degree': avg_degree,
        'knn': knn,
        'sim_thresh': config['sim_thresh'],
        'undirected': config['undirected']
    }

def compute_personalized_pagerank(adj_matrix: Any, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Personalized PageRank計算（全ノード）"""
    logger = logging.getLogger(__name__)
    
    try:
        from scipy.sparse import diags
        # scipy.sparse.linalg.normは使用せず、numpy.linalg.normのみ使用
    except ImportError:
        logger.error("ERROR: scipy not available for PPR computation")
        sys.exit(1)
    
    n_nodes = adj_matrix.shape[0]
    alpha = config['alpha']
    eps = config['eps']
    max_iter = config['max_iter']
    
    logger.info(f"Computing Personalized PageRank: alpha={alpha}, eps={eps}, max_iter={max_iter}")
    
    # 遷移行列構築（行正規化）
    logger.info("Building transition matrix...")
    
    # adj_matrixをCSRに統一
    adj_matrix = _to_csr(adj_matrix)
    
    row_sums = np.asarray(adj_matrix.sum(axis=1)).flatten()
    # 0除算回避: 最小値をクリップ
    row_sums = np.maximum(row_sums, 1e-12)
    
    # ゼロ次数ノード処理（self-loop追加）
    zero_degree_mask = row_sums <= 1e-12
    if np.any(zero_degree_mask):
        zero_count = np.sum(zero_degree_mask)
        logger.warning(f"Found {zero_count} zero-degree nodes, adding self-loops")
        
        # 対角要素に1追加
        from scipy.sparse import eye
        adj_matrix = adj_matrix + eye(n_nodes, format='csr') * zero_degree_mask.astype(float)
        row_sums = np.asarray(adj_matrix.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-12)
    
    # 行正規化
    from scipy.sparse import diags
    row_sums_inv = 1.0 / row_sums
    D_inv = diags(row_sums_inv, format='csr')
    P = D_inv @ adj_matrix  # 遷移行列
    P = _to_csr(P)
    
    logger.info("Starting PPR power iteration...")
    
    # 全ノードのPPR計算（並列化）
    all_ppr_scores = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    converged_count = 0
    total_iterations = 0
    
    for seed_node in range(n_nodes):
        if (seed_node + 1) % 1000 == 0 or seed_node < 10:
            logger.info(f"Computing PPR for node {seed_node + 1}/{n_nodes}")
        
        # 初期化（seed nodeに集中）
        personalization = np.zeros(n_nodes, dtype=np.float32)
        personalization[seed_node] = 1.0
        
        ppr_scores = personalization.copy()
        
        # Power iteration
        for iteration in range(max_iter):
            ppr_new = alpha * personalization + (1 - alpha) * (P.T @ ppr_scores)
            
            # 収束判定（numpy.linalg.normを使用）
            diff = nla.norm(ppr_new - ppr_scores)
            if diff < eps:
                converged_count += 1
                total_iterations += iteration + 1
                break
            
            ppr_scores = ppr_new
        
        all_ppr_scores[seed_node] = ppr_scores
    
    avg_iterations = total_iterations / n_nodes if n_nodes > 0 else 0
    convergence_rate = converged_count / n_nodes if n_nodes > 0 else 0
    
    logger.info(f"PPR computation completed:")
    logger.info(f"  Converged: {converged_count}/{n_nodes} ({convergence_rate:.1%})")
    logger.info(f"  Avg iterations: {avg_iterations:.1f}")
    
    ppr_stats = {
        'num_nodes': n_nodes,
        'alpha': alpha,
        'eps': eps,
        'max_iter': max_iter,
        'converged_nodes': converged_count,
        'convergence_rate': convergence_rate,
        'avg_iterations': avg_iterations,
        'total_computation_time': time.time()
    }
    
    return all_ppr_scores, ppr_stats

def save_graph_and_ppr(adj_matrix: Any, ppr_scores: np.ndarray, config: Dict[str, Any], 
                       graph_stats: Dict[str, Any], ppr_stats: Dict[str, Any], 
                       index_data: Dict[str, Any]) -> Dict[str, str]:
    """グラフ+PPR保存"""
    logger = logging.getLogger(__name__)
    
    try:
        from scipy.sparse import save_npz
    except ImportError:
        logger.error("ERROR: scipy not available for sparse matrix saving")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    config['graph_out_path'].parent.mkdir(parents=True, exist_ok=True)
    config['ppr_out_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # グラフ保存（sparse npz）
    graph_file = config['graph_out_path']
    if not graph_file.suffix:
        graph_file = graph_file.with_suffix('.npz')
    
    logger.info(f"Saving graph adjacency matrix: {graph_file}")
    save_npz(graph_file, adj_matrix)
    
    # PPR保存（dense numpy）
    ppr_file = config['ppr_out_path']
    if not ppr_file.suffix:
        ppr_file = ppr_file.with_suffix('.npy')
    
    logger.info(f"Saving PPR scores: {ppr_file}")
    np.save(ppr_file, ppr_scores)
    
    # Top-K PPR保存（メモリ効率のため）
    top_ppr_file = ppr_file.with_name(ppr_file.stem + f"_top{config['top_ppr']}.npz")
    
    logger.info(f"Computing and saving top-{config['top_ppr']} PPR...")
    
    # 各ノードのTop-K PPRインデックス・スコア
    top_k_indices = np.argpartition(ppr_scores, -config['top_ppr'], axis=1)[:, -config['top_ppr']:]
    top_k_scores = np.take_along_axis(ppr_scores, top_k_indices, axis=1)
    
    # ソート（降順）
    sort_indices = np.argsort(top_k_scores, axis=1)[:, ::-1]
    top_k_indices = np.take_along_axis(top_k_indices, sort_indices, axis=1)
    top_k_scores = np.take_along_axis(top_k_scores, sort_indices, axis=1)
    
    np.savez_compressed(
        top_ppr_file,
        indices=top_k_indices.astype(np.int32),
        scores=top_k_scores.astype(np.float32)
    )
    
    # メタデータ保存
    graph_meta_file = graph_file.with_suffix('.json')
    ppr_meta_file = ppr_file.with_suffix('.json')
    
    graph_metadata = {
        'graph_file': str(graph_file.name),
        'adjacency_format': 'scipy_csr_sparse',
        'creation_time': pd.Timestamp.now().isoformat(),
        'source_embeddings': str(config['embeddings_path']),
        'parameters': {
            'knn': config['knn'],
            'sim_thresh': config['sim_thresh'],
            'undirected': config['undirected'],
            'normalize_embeddings': config['normalize_embeddings'],
            'seed': config['seed']
        },
        'statistics': graph_stats,
        'source_index': index_data
    }
    
    ppr_metadata = {
        'ppr_file': str(ppr_file.name),
        'ppr_top_file': str(top_ppr_file.name),
        'creation_time': pd.Timestamp.now().isoformat(),
        'source_graph': str(graph_file),
        'parameters': {
            'alpha': config['alpha'],
            'eps': config['eps'],
            'max_iter': config['max_iter'],
            'top_ppr': config['top_ppr'],
            'seed': config['seed']
        },
        'statistics': ppr_stats,
        'source_index': index_data
    }
    
    logger.info(f"Saving graph metadata: {graph_meta_file}")
    with open(graph_meta_file, 'w') as f:
        json.dump(graph_metadata, f, indent=2)
    
    logger.info(f"Saving PPR metadata: {ppr_meta_file}")
    with open(ppr_meta_file, 'w') as f:
        json.dump(ppr_metadata, f, indent=2)
    
    return {
        'graph': str(graph_file),
        'graph_metadata': str(graph_meta_file),
        'ppr_full': str(ppr_file),
        'ppr_top': str(top_ppr_file),
        'ppr_metadata': str(ppr_meta_file)
    }

def main():
    """メイン処理"""
    logger = setup_production_logging()
    
    parser = argparse.ArgumentParser(description="Build Cosine Similarity Graph + Personalized PageRank")
    parser.add_argument('--embeddings', required=True,
                       help='Path to embeddings .npy file')
    parser.add_argument('--graph-out', required=True,
                       help='Output path for adjacency matrix')
    parser.add_argument('--ppr-out', required=True,
                       help='Output path for PPR scores')
    parser.add_argument('--knn', type=int, default=50,
                       help='Number of nearest neighbors (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.15,
                       help='PPR damping factor (default: 0.15)')
    parser.add_argument('--eps', type=float, default=1e-6,
                       help='PPR convergence threshold (default: 1e-6)')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='PPR max iterations (default: 100)')
    parser.add_argument('--top-ppr', type=int, default=100,
                       help='Top-K PPR scores to save (default: 100)')
    parser.add_argument('--sim-thresh', type=float, default=0.1,
                       help='Cosine similarity threshold (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize embeddings for cosine similarity (default: True)')
    parser.add_argument('--undirected', action='store_true', default=True,
                       help='Make graph undirected (default: True)')
    
    args = parser.parse_args()
    
    # 設定検証
    config = validate_config(args)
    
    # Effective-config 1行出力
    effective_config = f"embeddings={config['embeddings_path'].name}, knn={config['knn']}, alpha={config['alpha']}, eps={config['eps']}, sim_thresh={config['sim_thresh']}, top_ppr={config['top_ppr']}, seed={config['seed']}, undirected={config['undirected']}"
    logger.info(f"Effective-config: {effective_config}")
    
    # 出力ファイル存在チェック
    graph_file = config['graph_out_path']
    if not graph_file.suffix:
        graph_file = graph_file.with_suffix('.npz')
    
    ppr_file = config['ppr_out_path']
    if not ppr_file.suffix:
        ppr_file = ppr_file.with_suffix('.npy')
    
    if (graph_file.exists() or ppr_file.exists()) and not config['overwrite']:
        logger.info(f"Files already exist (use --overwrite to regenerate):")
        if graph_file.exists():
            logger.info(f"  Graph: {graph_file}")
        if ppr_file.exists():
            logger.info(f"  PPR: {ppr_file}")
        sys.exit(0)
    
    try:
        # 埋め込み読み込み
        embeddings, index_data = load_embeddings_and_index(config['embeddings_path'])
        
        # グラフ構築
        logger.info("=== STEP 1: Building cosine similarity graph ===")
        adj_matrix, graph_stats = build_cosine_similarity_graph(embeddings, config)
        
        # PPR計算
        logger.info("=== STEP 2: Computing Personalized PageRank ===")
        ppr_scores, ppr_stats = compute_personalized_pagerank(adj_matrix, config)
        
        # 保存
        logger.info("=== STEP 3: Saving graph and PPR results ===")
        paths = save_graph_and_ppr(adj_matrix, ppr_scores, config, graph_stats, ppr_stats, index_data)
        
        logger.info(f"[OK] Graph + PPR saved successfully")
        logger.info(f"Graph summary:")
        logger.info(f"  Nodes: {graph_stats['num_nodes']:,}")
        logger.info(f"  Edges: {graph_stats['num_edges']:,}")
        logger.info(f"  Avg degree: {graph_stats['avg_degree']:.1f}")
        logger.info(f"  Undirected: {graph_stats['undirected']}")
        
        logger.info(f"PPR summary:")
        logger.info(f"  Alpha: {ppr_stats['alpha']}")
        logger.info(f"  Converged: {ppr_stats['converged_nodes']}/{ppr_stats['num_nodes']} ({ppr_stats['convergence_rate']:.1%})")
        logger.info(f"  Avg iterations: {ppr_stats['avg_iterations']:.1f}")
        
        logger.info(f"Files created:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
            
    except Exception as e:
        logger.error(f"Failed to build graph + PPR: {e}")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check embeddings file format and content")
        logger.error("  - Verify sufficient memory for graph construction")
        logger.error("  - Try smaller knn or higher sim_thresh values")
        logger.error("  - Install required packages: pip install scipy faiss-gpu")
        logger.error("  - Check PPR parameters: alpha in [0,1], eps > 0")
        sys.exit(1)
    
    logger.info("Graph + PPR construction completed successfully")

if __name__ == "__main__":
    main()