#!/usr/bin/env python3
"""
Production Pipeline: Build FAISS Index
IVF+PQインデックス構築 with seed固定
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
from datetime import datetime
# import pandas as pd  # Removed to avoid GLIBCXX compatibility issues
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess

def setup_production_logging():
    """本番CLI経路ログセットアップ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # ログ先頭: which python / numpy.__file__
    try:
        python_path = subprocess.check_output(['which', 'python'], text=True).strip()
        logger.info(f"Python executable: {python_path}")
    except:
        logger.info(f"Python executable: {sys.executable}")
    
    logger.info(f"NumPy location: {np.__file__}")
    
    return logger

def validate_config(args) -> Dict[str, Any]:
    """設定検証（空/同一/PLACEHOLDER検出で即エラー）"""
    logger = logging.getLogger(__name__)
    
    # 必須引数検証
    if not args.embeddings or args.embeddings.strip() == "" or "PLACEHOLDER" in args.embeddings:
        logger.error("ERROR: --embeddings is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.index_out or args.index_out.strip() == "" or "PLACEHOLDER" in args.index_out:
        logger.error("ERROR: --index-out is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.index_type or args.index_type.strip() == "" or "PLACEHOLDER" in args.index_type:
        logger.error("ERROR: --index-type is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # インデックスタイプ検証
    valid_types = ['ivf_pq', 'flat']
    if args.index_type not in valid_types:
        logger.error(f"ERROR: --index-type must be one of {valid_types}, got '{args.index_type}'")
        sys.exit(1)
    
    # パラメータ検証
    if args.index_type == 'ivf_pq':
        if args.nlists <= 0:
            logger.error(f"ERROR: --nlists must be positive for IVF, got {args.nlists}")
            sys.exit(1)
        if args.pq_m <= 0:
            logger.error(f"ERROR: --pq-m must be positive for PQ, got {args.pq_m}")
            sys.exit(1)
        if args.pq_bits not in [4, 8]:
            logger.error(f"ERROR: --pq-bits must be 4 or 8, got {args.pq_bits}")
            sys.exit(1)
    
    if args.seed < 0:
        logger.error(f"ERROR: --seed must be non-negative, got {args.seed}")
        sys.exit(1)
    
    config = {
        'embeddings_path': Path(args.embeddings.strip()),
        'index_out_path': Path(args.index_out.strip()),
        'index_type': args.index_type.strip(),
        'nlists': args.nlists,
        'pq_m': args.pq_m,
        'pq_bits': args.pq_bits,
        'seed': args.seed,
        'overwrite': args.overwrite,
        'use_gpu': getattr(args, 'use_gpu', False)
    }
    
    # 埋め込みファイル存在確認
    if not config['embeddings_path'].exists():
        logger.error(f"ERROR: Embeddings file not found: {config['embeddings_path']}")
        sys.exit(1)
    
    return config

def load_embeddings_with_metadata(embeddings_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """埋め込み+メタデータ読み込み"""
    logger = logging.getLogger(__name__)
    
    # 埋め込み読み込み
    logger.info(f"Loading embeddings: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        sys.exit(1)
    
    # メタデータ読み込み（あれば）
    metadata_path = embeddings_path.parent / (embeddings_path.stem + "_metadata.json")
    metadata = {}
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {len(metadata)} fields")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    
    # 基本検証
    if embeddings.size == 0:
        logger.error("ERROR: Empty embeddings array")
        sys.exit(1)
    
    if len(embeddings.shape) != 2:
        logger.error(f"ERROR: Embeddings must be 2D, got shape {embeddings.shape}")
        sys.exit(1)
    
    return embeddings, metadata

def build_faiss_index(embeddings: np.ndarray, config: Dict[str, Any]) -> Any:
    """FAISSインデックス構築"""
    logger = logging.getLogger(__name__)
    
    try:
        import faiss
    except ImportError:
        logger.error("ERROR: faiss-gpu or faiss-cpu not installed")
        logger.error("Install with: pip install faiss-gpu  # or faiss-cpu")
        sys.exit(1)
    
    # Seed固定
    np.random.seed(config['seed'])
    logger.info(f"Set random seed: {config['seed']}")
    
    n_vectors, dim = embeddings.shape
    logger.info(f"Building {config['index_type']} index for {n_vectors} vectors (dim={dim})")
    
    # データ型変換 (FAISS requires float32)
    if embeddings.dtype != np.float32:
        logger.info(f"Converting {embeddings.dtype} -> float32")
        embeddings = embeddings.astype(np.float32)
    
    # インデックス構築
    if config['index_type'] == 'flat':
        # Flat L2インデックス（完全検索）
        logger.info("Building Flat L2 index")
        index = faiss.IndexFlatL2(dim)
        
    elif config['index_type'] == 'ivf_pq':
        # IVF+PQインデックス
        nlists = config['nlists']
        pq_m = config['pq_m']
        pq_bits = config['pq_bits']
        
        logger.info(f"Building IVF+PQ index: nlists={nlists}, pq_m={pq_m}, pq_bits={pq_bits}")
        
        # PQパラメータ検証
        if dim % pq_m != 0:
            logger.error(f"ERROR: dim ({dim}) must be divisible by pq_m ({pq_m})")
            sys.exit(1)
        
        # nlists自動調整（推奨: min(max(32, int(sqrt(N)*2)), 128)）
        recommended_nlists = min(max(32, int(np.sqrt(n_vectors) * 2)), 128)
        if nlists > n_vectors // 8:
            logger.warning(f"nlists ({nlists}) might be too large for {n_vectors} vectors (recommended: ~{recommended_nlists})")
            # 自動調整
            if nlists > recommended_nlists * 2:
                old_nlists = nlists
                nlists = recommended_nlists
                logger.info(f"Auto-adjusted nlists: {old_nlists} -> {nlists} for N={n_vectors}")
                config['nlists'] = nlists
        
        # IVF+PQ index作成
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlists, pq_m, pq_bits)
        
        # GPU使用設定
        if config['use_gpu'] and faiss.get_num_gpus() > 0:
            logger.info(f"Using GPU acceleration ({faiss.get_num_gpus()} GPUs available)")
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        
        # トレーニング
        logger.info("Training IVF+PQ index...")
        
        # トレーニングサンプル選択（最大100K）
        if n_vectors > 100000:
            train_indices = np.random.choice(n_vectors, 100000, replace=False)
            train_data = embeddings[train_indices]
            logger.info(f"Using {len(train_indices)} samples for training")
        else:
            train_data = embeddings
        
        index.train(train_data)
        logger.info("IVF+PQ training completed")
        
        # GPU→CPU移行（保存用）
        if config['use_gpu'] and faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)
    
    else:
        logger.error(f"Unsupported index type: {config['index_type']}")
        sys.exit(1)
    
    # データ追加
    logger.info("Adding vectors to index...")
    index.add(embeddings)
    logger.info(f"Index populated: {index.ntotal} vectors")
    
    return index

def save_faiss_index(index: Any, config: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, str]:
    """FAISSインデックス保存"""
    logger = logging.getLogger(__name__)
    
    try:
        import faiss
    except ImportError:
        logger.error("ERROR: faiss not available for saving")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    config['index_out_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # インデックス保存
    index_file = config['index_out_path']
    if not index_file.suffix:
        index_file = index_file.with_suffix('.faiss')
    
    logger.info(f"Saving FAISS index: {index_file}")
    try:
        faiss.write_index(index, str(index_file))
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        sys.exit(1)
    
    # メタデータ保存
    meta_file = index_file.with_suffix('.json')
    
    index_metadata = {
        'index_type': config['index_type'],
        'index_file': str(index_file.name),
        'total_vectors': int(index.ntotal),
        'dimension': int(index.d),
        'seed': config['seed'],
        'creation_time': datetime.now().isoformat(),
        'source_embeddings': str(config['embeddings_path']),
        'parameters': {}
    }
    
    # タイプ別パラメータ
    if config['index_type'] == 'ivf_pq':
        index_metadata['parameters'] = {
            'nlists': config['nlists'],
            'pq_m': config['pq_m'],
            'pq_bits': config['pq_bits'],
            'is_trained': bool(index.is_trained)
        }
    
    # 元メタデータマージ
    if metadata:
        index_metadata['source_metadata'] = metadata
    
    logger.info(f"Saving metadata: {meta_file}")
    with open(meta_file, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    
    return {
        'index': str(index_file),
        'metadata': str(meta_file)
    }

def test_index_basic_search(index: Any, embeddings: np.ndarray, k: int = 5) -> Dict[str, Any]:
    """インデックス基本検索テスト"""
    logger = logging.getLogger(__name__)
    
    try:
        import faiss
    except ImportError:
        return {'status': 'skip', 'reason': 'faiss_not_available'}
    
    logger.info("Testing index with basic search...")
    
    # テストクエリ（最初の5ベクトル）
    test_queries = embeddings[:min(5, len(embeddings))]
    
    try:
        # 検索実行
        distances, indices = index.search(test_queries, k)
        
        # 結果検証
        test_results = {
            'status': 'success',
            'num_queries': len(test_queries),
            'k': k,
            'avg_distance': float(distances.mean()),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'valid_indices': bool(np.all(indices >= 0))
        }
        
        logger.info(f"Search test successful: avg_dist={test_results['avg_distance']:.4f}")
        return test_results
        
    except Exception as e:
        logger.warning(f"Index test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def main():
    """メイン処理"""
    logger = setup_production_logging()
    
    parser = argparse.ArgumentParser(description="Build FAISS Index from Embeddings")
    parser.add_argument('--embeddings', required=True,
                       help='Path to embeddings .npy file')
    parser.add_argument('--index-out', required=True,
                       help='Output path for FAISS index')
    parser.add_argument('--index-type', choices=['ivf_pq', 'flat'], required=True,
                       help='Index type: ivf_pq or flat')
    parser.add_argument('--nlists', type=int, default=1024,
                       help='Number of clusters for IVF (default: 1024)')
    parser.add_argument('--pq-m', type=int, default=16,
                       help='PQ segments (default: 16)')
    parser.add_argument('--pq-bits', type=int, choices=[4, 8], default=8,
                       help='PQ bits per segment (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing index')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration (if available)')
    
    args = parser.parse_args()
    
    # 設定検証
    config = validate_config(args)
    
    # Effective-config 1行出力
    if config['index_type'] == 'ivf_pq':
        params_str = f"nlists={config['nlists']}, pq_m={config['pq_m']}, pq_bits={config['pq_bits']}"
    else:
        params_str = "none"
    
    effective_config = f"embeddings={config['embeddings_path'].name}, index_type={config['index_type']}, {params_str}, seed={config['seed']}, overwrite={config['overwrite']}"
    logger.info(f"Effective-config: {effective_config}")
    
    # 出力ファイル存在チェック
    index_file = config['index_out_path']
    if not index_file.suffix:
        index_file = index_file.with_suffix('.faiss')
    
    if index_file.exists() and not config['overwrite']:
        logger.info(f"Index already exists: {index_file} (use --overwrite to regenerate)")
        sys.exit(0)
    
    try:
        # 埋め込み読み込み
        embeddings, metadata = load_embeddings_with_metadata(config['embeddings_path'])
        
        # インデックス構築
        index = build_faiss_index(embeddings, config)
        
        # インデックステスト
        test_results = test_index_basic_search(index, embeddings)
        logger.info(f"Index test: {test_results['status']}")
        
        # 保存
        paths = save_faiss_index(index, config, metadata)
        
        logger.info(f"[OK] faiss index saved: {paths['index']}")
        logger.info(f"Index summary:")
        logger.info(f"  Type: {config['index_type']}")
        logger.info(f"  Vectors: {index.ntotal:,}")
        logger.info(f"  Dimension: {index.d}")
        if hasattr(index, 'is_trained'):
            logger.info(f"  Trained: {index.is_trained}")
        logger.info(f"  Files: {paths['index']}, {paths['metadata']}")
        
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check embeddings file format (.npy)")
        logger.error("  - Verify sufficient memory for index building")
        logger.error("  - Try smaller nlists or pq_m values")
        logger.error("  - Check PQ parameters: dim must be divisible by pq_m")
        logger.error("  - Install faiss: pip install faiss-gpu")
        sys.exit(1)
    
    logger.info("FAISS index building completed successfully")

if __name__ == "__main__":
    main()