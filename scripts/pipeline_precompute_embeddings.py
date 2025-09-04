#!/usr/bin/env python3
"""
Production Pipeline: Precompute Embeddings
LaMP-2/Tenrec共通埋め込み計算 (例: all-MiniLM-L6-v2, dim=384)
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
# from sentence_transformers import SentenceTransformer  # 互換性問題のためコメントアウト
import subprocess

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
    if not args.datasets or args.datasets.strip() == "" or "PLACEHOLDER" in args.datasets:
        logger.error("ERROR: --datasets is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.model or args.model.strip() == "" or "PLACEHOLDER" in args.model:
        logger.error("ERROR: --model is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.out_dir or args.out_dir.strip() == "" or "PLACEHOLDER" in args.out_dir:
        logger.error("ERROR: --out-dir is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # 次元数検証
    if args.dim <= 0:
        logger.error(f"ERROR: --dim must be positive, got {args.dim}")
        sys.exit(1)
    
    # データセット重複検証
    dataset_list = [d.strip() for d in args.datasets.split(',')]
    if len(dataset_list) != len(set(dataset_list)):
        logger.error(f"ERROR: Duplicate datasets found in {dataset_list}")
        sys.exit(1)
    
    config = {
        'datasets': dataset_list,
        'model': args.model.strip(),
        'dim': args.dim,
        'out_dir': Path(args.out_dir.strip()),
        'limit': args.limit,
        'overwrite': args.overwrite,
        'batch_size': getattr(args, 'batch_size', 32)
    }
    
    return config

def load_dataset_texts(dataset_name: str, limit: Optional[int] = None) -> List[str]:
    """データセット読み込み"""
    logger = logging.getLogger(__name__)
    
    if dataset_name.lower() == 'lamp2':
        # LaMP-2データ読み込み
        data_path = Path('./chameleon_prime_personalization/data/raw/LaMP-2')
        
        texts = []
        for split_file in ['train_questions.json', 'dev_questions.json']:
            file_path = data_path / split_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        # プロフィール文書
                        profile = item.get('profile', [])
                        for profile_item in profile:
                            desc = profile_item.get('description', '').strip()
                            if desc:
                                texts.append(desc)
                        
                        # 入力クエリ
                        input_text = item.get('input', '').strip()
                        if input_text:
                            texts.append(input_text)
        
        logger.info(f"Loaded {len(texts)} texts from LaMP-2")
        
    elif dataset_name.lower() == 'tenrec':
        # Tenrecデータ読み込み
        data_path = Path('./data/tenrec')  # 仮想パス
        
        texts = []
        # Tenrec用のダミーデータ（実装時には実データを使用）
        if data_path.exists():
            # 実データ読み込みロジック
            pass
        else:
            # ダミーデータ生成
            texts = [
                f"Tenrec item {i}: Movie recommendation content"
                for i in range(1000)
            ]
            logger.warning(f"Using dummy Tenrec data: {len(texts)} items")
    
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    
    # 制限適用
    if limit and limit > 0:
        texts = texts[:limit]
        logger.info(f"Limited to {len(texts)} texts (limit={limit})")
    
    # 空文字列・重複除去
    texts = list(set([t.strip() for t in texts if t.strip()]))
    logger.info(f"After deduplication: {len(texts)} unique texts")
    
    return texts

def compute_embeddings(texts: List[str], model_name: str, dim: int, batch_size: int = 32) -> np.ndarray:
    """埋め込み計算（transformers直接利用でcached_download問題回避）"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model with transformers library: {model_name}")
    try:
        # SentenceTransformerの代わりにtransformersライブラリを直接使用
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # モデル次元確認
        with torch.no_grad():
            sample_input = tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            sample_output = model(**sample_input)
            model_dim = sample_output.last_hidden_state.shape[-1]
        
        if model_dim != dim:
            logger.warning(f"Model dimension ({model_dim}) != specified dim ({dim})")
            logger.info(f"Using actual model dimension: {model_dim}")
            dim = model_dim
        
        # GPU使用可能であれば移動
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Using device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.error("Note: Using transformers library directly to avoid sentence-transformers compatibility issues")
        sys.exit(1)
    
    logger.info(f"Computing embeddings for {len(texts)} texts (batch_size={batch_size})")
    
    try:
        embeddings_list = []
        model.eval()
        
        # バッチ処理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # トークナイズ
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # GPU移動
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling for sentence embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        # 結果結合
        final_embeddings = np.vstack(embeddings_list)
        logger.info(f"Computed embeddings: shape={final_embeddings.shape}")
        return final_embeddings
        
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        sys.exit(1)

def save_embeddings(embeddings: np.ndarray, texts: List[str], dataset_name: str, 
                   out_dir: Path, model_name: str) -> Dict[str, str]:
    """埋め込み保存 (npy + idx)"""
    logger = logging.getLogger(__name__)
    
    # 出力ディレクトリ作成
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイル名生成
    safe_model = model_name.replace('/', '_').replace('-', '_')
    prefix = f"{dataset_name}_{safe_model}"
    
    npy_path = out_dir / f"{prefix}_embeddings.npy"
    idx_path = out_dir / f"{prefix}_index.json"
    meta_path = out_dir / f"{prefix}_metadata.json"
    
    # 埋め込み保存
    logger.info(f"Saving embeddings: {npy_path}")
    np.save(npy_path, embeddings)
    
    # インデックス保存（テキストID→インデックス）
    index_data = {
        'texts': texts,
        'embeddings_shape': embeddings.shape,
        'model_name': model_name,
        'dataset': dataset_name
    }
    
    logger.info(f"Saving index: {idx_path}")
    with open(idx_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # メタデータ保存
    metadata = {
        'dataset': dataset_name,
        'model': model_name,
        'embedding_dim': embeddings.shape[1],
        'num_texts': len(texts),
        'embeddings_file': str(npy_path.name),
        'index_file': str(idx_path.name),
        'creation_time': pd.Timestamp.now().isoformat()
    }
    
    logger.info(f"Saving metadata: {meta_path}")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'embeddings': str(npy_path),
        'index': str(idx_path),
        'metadata': str(meta_path)
    }

def main():
    """メイン処理"""
    logger = setup_production_logging()
    
    parser = argparse.ArgumentParser(description="Precompute Embeddings for LaMP-2/Tenrec")
    parser.add_argument('--datasets', required=True, 
                       help='Datasets to process (CSV): lamp2,tenrec')
    parser.add_argument('--model', required=True,
                       help='Sentence transformer model name')
    parser.add_argument('--dim', type=int, required=True,
                       help='Expected embedding dimension')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of texts (0=no limit)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Encoding batch size')
    
    args = parser.parse_args()
    
    # 設定検証
    config = validate_config(args)
    
    # Effective-config 1行出力
    effective_config = f"datasets={config['datasets']}, model={config['model']}, dim={config['dim']}, out_dir={config['out_dir']}, limit={config['limit']}, overwrite={config['overwrite']}"
    logger.info(f"Effective-config: {effective_config}")
    
    # 各データセット処理
    results = {}
    
    for dataset in config['datasets']:
        logger.info(f"Processing dataset: {dataset}")
        
        try:
            # 出力ファイル存在チェック
            safe_model = config['model'].replace('/', '_').replace('-', '_')
            prefix = f"{dataset}_{safe_model}"
            npy_path = config['out_dir'] / f"{prefix}_embeddings.npy"
            
            if npy_path.exists() and not config['overwrite']:
                logger.info(f"Embeddings already exist: {npy_path} (use --overwrite to regenerate)")
                results[dataset] = {'status': 'skipped', 'path': str(npy_path)}
                continue
            
            # データセット読み込み
            texts = load_dataset_texts(dataset, config['limit'])
            
            if not texts:
                logger.error(f"No texts found for dataset: {dataset}")
                results[dataset] = {'status': 'error', 'reason': 'no_texts'}
                continue
            
            # 埋め込み計算
            embeddings = compute_embeddings(
                texts, config['model'], config['dim'], config['batch_size']
            )
            
            # 保存
            paths = save_embeddings(
                embeddings, texts, dataset, config['out_dir'], config['model']
            )
            
            results[dataset] = {
                'status': 'success',
                'paths': paths,
                'num_texts': len(texts),
                'embedding_shape': embeddings.shape
            }
            
            logger.info(f"[OK] embeddings saved: {dataset} -> {paths['embeddings']}")
            
        except Exception as e:
            logger.error(f"Failed to process {dataset}: {e}")
            results[dataset] = {'status': 'error', 'reason': str(e)}
    
    # 結果サマリー
    logger.info("Processing summary:")
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    for dataset, result in results.items():
        status_emoji = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'error' else "⏭️"
        logger.info(f"  {status_emoji} {dataset}: {result['status']}")
    
    logger.info(f"Completed: {success_count}/{total_count} datasets successful")
    
    # エラー時はexit!=0
    if success_count != total_count:
        logger.error("Some datasets failed to process")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check dataset paths and formats")
        logger.error("  - Verify model name is correct")
        logger.error("  - Ensure sufficient disk space")
        logger.error("  - Check network connection for model download")
        sys.exit(1)
    
    logger.info("All datasets processed successfully")

if __name__ == "__main__":
    main()