#!/usr/bin/env python3
"""
GraphRAG × Chameleon ワンコマンド統合パイプライン
Steps: embeddings -> FAISS(IVF+PQ) -> Graph(PPR) -> Eval
"""

import os
import sys
import json
import time
import math
import subprocess
import argparse
import logging
import shutil
from pathlib import Path
from typing import Set, List

def sh(cmd: list, check=True):
    """サブプロセス実行"""
    logger = logging.getLogger(__name__)
    logger.info(f"Executing: {' '.join(cmd)}")
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.stdout:
        logger.info(f"STDOUT: {p.stdout.strip()}")
    if p.stderr:
        logger.info(f"STDERR: {p.stderr.strip()}")
    if p.returncode != 0 and check:
        logger.error(f"Command failed with exit code {p.returncode}")
        sys.exit(p.returncode)
    return p

def setup_logging():
    """本番CLI経路ログセットアップ"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # ログ先頭: which python / pandas.__file__
    try:
        which = shutil.which("python") or sys.executable
        logger.info(f"Python executable: {which}")
    except Exception:
        logger.info(f"Python executable: {sys.executable}")
    
    try:
        import pandas
        logger.info(f"Pandas location: {pandas.__file__}")
    except Exception as e:
        logger.warning(f"Pandas not available: {e}")
    
    return logger

def check_theta_cache_coverage(lamp2_file: str, theta_cache_dir: str, logger) -> Set[str]:
    """θ キャッシュ検証: 未処理ユーザー特定"""
    missing_users = set()
    
    if not os.path.exists(theta_cache_dir):
        logger.info(f"Theta cache directory not found: {theta_cache_dir}")
        return missing_users
    
    cache_path = Path(theta_cache_dir)
    
    # キャッシュされているユーザー取得
    cached_users = set()
    for theta_p_file in cache_path.glob("*_theta_p.npy"):
        user_id = theta_p_file.stem.replace("_theta_p", "")
        theta_n_file = cache_path / f"{user_id}_theta_n.npy"
        if theta_n_file.exists():
            cached_users.add(user_id)
    
    logger.info(f"Found {len(cached_users)} users with complete θ cache")
    
    # LaMP-2データから必要ユーザー特定
    required_users = set()
    if os.path.exists(lamp2_file):
        try:
            with open(lamp2_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # LaMP-2のIDから最初の桁でユーザーID抽出
                        question_id = str(data.get('id', ''))
                        if question_id:
                            user_id = question_id[0] if question_id else 'unknown'
                            required_users.add(user_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read LaMP-2 file: {e}")
    
    # 未キャッシュユーザー特定
    missing_users = required_users - cached_users
    
    logger.info(f"Cache coverage: {len(cached_users)}/{len(required_users)} users")
    if missing_users:
        logger.warning(f"Missing θ cache for {len(missing_users)} users: {sorted(list(missing_users))[:10]}...")
    
    return missing_users

def run_fakeit_pipeline(missing_users: Set[str], args, logger) -> bool:
    """未処理ユーザーに対してFake itパイプライン実行"""
    if not missing_users:
        logger.info("All users have θ cache - skipping Fake it pipeline")
        return True
    
    logger.info(f"=== STEP 0.5: Running Fake it Pipeline for {len(missing_users)} missing users ===")
    
    # ユーザー数制限（デモ用）
    max_fakeit_users = min(len(missing_users), args.limit)
    
    try:
        fakeit_cmd = [
            "python", "scripts/pipeline_fakeit_build_directions.py",
            "--data-dir", "chameleon_prime_personalization/data/raw/LaMP-2",
            "--model-path", args.model_path,
            "--output-dir", f"{args.out}/personalization",
            "--max-users", str(max_fakeit_users),
            "--seed", str(args.seed),
            "--personal-insights", "2",  # 少なめにして高速化
            "--neutral-insights", "1",
            "--pairs-per-insight", "2",
            "--max-new-tokens", "30",
            "--temperature", str(args.temperature)
        ]
        
        if hasattr(args, 'verbose') and args.verbose:
            fakeit_cmd.append("--verbose")
        
        sh(fakeit_cmd)
        
        logger.info("✅ Fake it pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fake it pipeline failed: {e}")
        return False

def validate_config(args):
    """設定検証（空/同一/PLACEHOLDER検出で即エラー）"""
    logger = logging.getLogger(__name__)
    
    # 必須引数検証
    if not args.dataset or args.dataset.strip() == "" or "PLACEHOLDER" in args.dataset:
        logger.error("ERROR: --dataset is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.out or args.out.strip() == "" or "PLACEHOLDER" in args.out:
        logger.error("ERROR: --out is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.model_path or args.model_path.strip() == "" or "PLACEHOLDER" in args.model_path:
        logger.error("ERROR: --model-path is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # 数値パラメータ検証
    if args.limit <= 0:
        logger.error(f"ERROR: --limit must be positive, got {args.limit}")
        sys.exit(1)
    
    if args.seed < 0:
        logger.error(f"ERROR: --seed must be non-negative, got {args.seed}")
        sys.exit(1)
    
    if args.nlists <= 0:
        logger.error(f"ERROR: --nlists must be positive, got {args.nlists}")
        sys.exit(1)

def preview_dataset_head(lamp2_file: str, logger):
    """先頭3件のQ/A抜粋"""
    if not os.path.exists(lamp2_file):
        logger.warning(f"Dataset file not found: {lamp2_file}")
        return
    
    try:
        head = []
        with open(lamp2_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    data = json.loads(line)
                    sample = {
                        "id": data.get("id", f"line_{i}"),
                        "question": data.get("question", data.get("input", ""))[:100] + "...",
                        "reference": data.get("reference", data.get("output", ""))
                    }
                    head.append(sample)
                except json.JSONDecodeError:
                    head.append({"raw": line.strip()[:100] + "..."})
        
        logger.info(f"Head-3 samples: {head}")
    except Exception as e:
        logger.warning(f"Failed to preview dataset head: {e}")

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="One-shot Graph Chameleon pipeline")
    parser.add_argument("--dataset", choices=["lamp2"], required=True,
                       help="Dataset to process")
    parser.add_argument("--lamp2-file", default="data/evaluation/lamp2_backup_eval.jsonl",
                       help="LaMP-2 dataset file path")
    parser.add_argument("--out", default="runs",
                       help="Output directory for results")
    parser.add_argument("--outdir", dest="out", 
                       help="Output directory for results (alias for --out)")
    parser.add_argument("--emb-out", default="assets/embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--faiss-out", default="assets/faiss/lamp2_ivfpq.faiss",
                       help="Output path for FAISS index")
    parser.add_argument("--ppr-out", default="assets/ppr/lamp2_ppr_top100.npz",
                       help="Output path for PPR scores")
    parser.add_argument("--limit", type=int, default=20000,
                       help="Max texts to process (default: 20000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--alpha-grid", default="0.2,0.3,0.4",
                       help="Alpha values for evaluation grid search")
    parser.add_argument("--beta-grid", default="0.0",
                       help="Beta values for evaluation grid search (can be negative)")
    parser.add_argument("--topk-grid", default="10,20",
                       help="Top-K values for evaluation grid search")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature (default: 0.0)")
    parser.add_argument("--max-gen", type=int, default=8,
                       help="Max generation tokens (default: 8)")
    parser.add_argument("--model-path", default="./chameleon_prime_personalization/models/base_model",
                       help="Path to base model")
    parser.add_argument("--nlists", type=int, default=128,
                       help="FAISS IVF clusters (default: 128)")
    parser.add_argument("--pq-m", type=int, default=32,
                       help="FAISS PQ segments (default: 32)")
    parser.add_argument("--pq-bits", type=int, default=8,
                       help="FAISS PQ bits (default: 8)")
    parser.add_argument("--knn", type=int, default=50,
                       help="Graph k-NN neighbors (default: 50)")
    parser.add_argument("--sim-thresh", type=float, default=0.15,
                       help="Graph similarity threshold (default: 0.15)")
    parser.add_argument("--top-ppr", type=int, default=100,
                       help="Top PPR scores to save (default: 100)")
    parser.add_argument("--max-iter", type=int, default=10,
                       help="PPR max iterations (default: 10)")
    
    # Fake it / θ cache related arguments
    parser.add_argument("--build-theta-if-missing", action="store_true",
                       help="Automatically run Fake it pipeline for users without θ cache")
    parser.add_argument("--theta-cache-dir", default=None,
                       help="Directory to check for existing θ vectors (default: {out}/personalization/theta_cache)")
    parser.add_argument("--force-fakeit", action="store_true", 
                       help="Force run Fake it pipeline even if θ cache exists")
    parser.add_argument("--skip-fakeit", action="store_true",
                       help="Skip Fake it pipeline completely (use existing θ only)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging including Fake it pipeline")
    
    args = parser.parse_args()
    
    # 設定検証
    validate_config(args)
    
    # 出力ディレクトリ作成
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.emb_out).mkdir(parents=True, exist_ok=True)
    Path(args.faiss_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ppr_out).parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Effective-config 1行出力
    effective_config = (
        f"dataset={args.dataset}, limit={args.limit}, seed={args.seed}, "
        f"nlists={args.nlists}, pq_m={args.pq_m}, pq_bits={args.pq_bits}, "
        f"knn={args.knn}, sim_thresh={args.sim_thresh}, "
        f"top_ppr={args.top_ppr}, max_iter={args.max_iter}, "
        f"alpha_grid={args.alpha_grid}, beta_grid={args.beta_grid}, topk_grid={args.topk_grid}"
    )
    logger.info(f"Effective-config: {effective_config}")
    
    try:
        # 先頭3件のQ/A抜粋
        preview_dataset_head(args.lamp2_file, logger)
        
        # STEP 0: θ キャッシュ検証と Fake it パイプライン実行
        theta_cache_dir = args.theta_cache_dir or f"{args.out}/personalization/theta_cache"
        
        if args.skip_fakeit:
            logger.info("=== STEP 0: Skipping Fake it pipeline (--skip-fakeit) ===")
        elif args.force_fakeit:
            logger.info("=== STEP 0: Force running Fake it pipeline (--force-fakeit) ===")
            run_fakeit_pipeline(set(['force']), args, logger)
        elif args.build_theta_if_missing:
            logger.info("=== STEP 0: θ Cache validation and Fake it pipeline ===")
            missing_users = check_theta_cache_coverage(args.lamp2_file, theta_cache_dir, logger)
            
            if missing_users:
                success = run_fakeit_pipeline(missing_users, args, logger)
                if not success:
                    logger.error("Fake it pipeline failed - continuing with existing θ cache only")
            else:
                logger.info("✅ All users have θ cache - proceeding to evaluation")
        else:
            logger.info("=== STEP 0: Using existing θ vectors (add --build-theta-if-missing to auto-generate) ===")
            if os.path.exists(theta_cache_dir):
                cache_count = len(list(Path(theta_cache_dir).glob("*_theta_p.npy")))
                logger.info(f"Found {cache_count} θ_P vectors in cache")
        
        # STEP 1: 埋め込み計算
        logger.info("=== STEP 1: Computing embeddings ===")
        emb_npy = f"{args.emb_out}/lamp2_sentence_transformers_all_MiniLM_L6_v2_embeddings.npy"
        
        if not Path(emb_npy).exists():
            sh([
                "python", "scripts/pipeline_precompute_embeddings.py",
                "--datasets", "lamp2",
                "--model", "sentence-transformers/all-MiniLM-L6-v2",
                "--dim", "384",
                "--out-dir", args.emb_out,
                "--limit", str(args.limit),
                "--overwrite"
            ])
        else:
            logger.info(f"Using existing embeddings: {emb_npy}")
        
        # STEP 2: FAISS インデックス構築（nlists 自動調整）
        logger.info("=== STEP 2: Building FAISS index ===")
        
        import numpy as np
        arr = np.load(emb_npy, mmap_mode="r")
        N = arr.shape[0]
        safe_nlists = args.nlists
        
        if N < args.nlists * 8:
            safe_nlists = max(32, 2 ** int(math.log2(max(2, N // 8))))
            logger.info(f"Auto-adjusted nlists: {args.nlists} -> {safe_nlists} for N={N}")
        
        sh([
            "python", "scripts/pipeline_build_faiss.py",
            "--embeddings", emb_npy,
            "--index-out", args.faiss_out,
            "--index-type", "ivf_pq",
            "--nlists", str(safe_nlists),
            "--pq-m", str(args.pq_m),
            "--pq-bits", str(args.pq_bits),
            "--seed", str(args.seed),
            "--overwrite"
        ])
        
        # STEP 3: グラフ構築 + PPR計算
        logger.info("=== STEP 3: Building graph and computing PPR ===")
        
        sh([
            "python", "scripts/pipeline_build_graph_ppr.py",
            "--embeddings", emb_npy,
            "--graph-out", str(Path(args.ppr_out).parent / "graph.npz"),
            "--ppr-out", args.ppr_out,
            "--knn", str(args.knn),
            "--alpha", "0.85",  # PPR damping factor
            "--max-iter", str(args.max_iter),
            "--top-ppr", str(args.top_ppr),
            "--sim-thresh", str(args.sim_thresh),
            "--normalize", "--undirected", "--overwrite"
        ])
        
        # STEP 4: グラフ強化Chameleon評価
        logger.info("=== STEP 4: Running graph Chameleon evaluation ===")
        
        # PPR出力ファイルの実際のパスを確認
        actual_ppr_file = args.ppr_out
        top_ppr_file = Path(args.ppr_out).with_name(Path(args.ppr_out).stem + f"_top{args.top_ppr}.npz")
        
        if top_ppr_file.exists():
            actual_ppr_file = str(top_ppr_file)
            logger.info(f"Using top-K PPR file: {actual_ppr_file}")
        
        sh([
            "python", "scripts/run_graph_chameleon_eval.py",
            "--dataset", "lamp2",
            "--ppr-scores", actual_ppr_file,
            "--out-dir", args.out,
            "--limit", str(args.limit),
            "--seed", str(args.seed),
            f"--alpha-grid={args.alpha_grid}",
            f"--beta-grid={args.beta_grid}",
            f"--topk-grid={args.topk_grid}",
            "--model-path", args.model_path,
            "--max-gen", str(args.max_gen),
            "--temperature", str(args.temperature)
        ])
        
        execution_time = time.time() - start_time
        
        logger.info(f"[OK] One-shot pipeline finished in {execution_time:.1f}s ({execution_time/60:.1f}min)")
        logger.info(f"Results saved under: {args.out}")
        
        # 結果ディレクトリの内容表示
        results_dirs = list(Path(args.out).glob("runs/*"))
        if results_dirs:
            latest_run = max(results_dirs, key=lambda p: p.stat().st_mtime)
            logger.info(f"Latest run directory: {latest_run}")
            
            # サマリーファイルがあれば内容表示
            summary_file = latest_run / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    logger.info(f"Best accuracy: {summary.get('best_accuracy', 'N/A')}")
                    logger.info(f"Best parameters: {summary.get('best_parameters', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Failed to read summary: {e}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check dataset files are accessible")
        logger.error("  - Verify model path exists and is valid")
        logger.error("  - Ensure sufficient disk space and memory")
        logger.error("  - Install required packages: torch, transformers, scipy, faiss")
        logger.error("  - Check GPU availability for CUDA operations")
        sys.exit(1)
    
    logger.info("Graph Chameleon pipeline completed successfully")

if __name__ == "__main__":
    main()