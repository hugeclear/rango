#!/usr/bin/env python3
"""
Production Pipeline: Graph-based Chameleon Evaluation
グラフ強化Chameleon評価 + グリッドサーチ
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
import itertools

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

def parse_grid(grid_str: str, param_name: str) -> List[float]:
    """グリッド文字列をfloatリストに解析（負数対応）"""
    logger = logging.getLogger(__name__)
    
    try:
        values = []
        for x in grid_str.split(','):
            x = x.strip()
            if x:
                values.append(float(x))
        
        if not values:
            logger.error(f"ERROR: {param_name} grid is empty: '{grid_str}'")
            sys.exit(1)
        
        return values
        
    except ValueError as e:
        logger.error(f"ERROR: Failed to parse {param_name} grid '{grid_str}': {e}")
        logger.error("Troubleshooting: Use comma-separated numbers, e.g., '0.1,0.2,-0.1'")
        sys.exit(1)

def parse_int_grid(grid_str: str, param_name: str) -> List[int]:
    """グリッド文字列をintリストに解析"""
    logger = logging.getLogger(__name__)
    
    try:
        values = []
        for x in grid_str.split(','):
            x = x.strip()
            if x:
                values.append(int(x))
        
        if not values:
            logger.error(f"ERROR: {param_name} grid is empty: '{grid_str}'")
            sys.exit(1)
        
        return values
        
    except ValueError as e:
        logger.error(f"ERROR: Failed to parse {param_name} grid '{grid_str}': {e}")
        logger.error("Troubleshooting: Use comma-separated integers, e.g., '5,10,20'")
        sys.exit(1)

def validate_config(args) -> Dict[str, Any]:
    """設定検証（空/同一/PLACEHOLDER検出で即エラー）"""
    logger = logging.getLogger(__name__)
    
    # 必須引数検証
    if not args.dataset or args.dataset.strip() == "" or "PLACEHOLDER" in args.dataset:
        logger.error("ERROR: --dataset is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.ppr_scores or args.ppr_scores.strip() == "" or "PLACEHOLDER" in args.ppr_scores:
        logger.error("ERROR: --ppr-scores is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    if not args.out_dir or args.out_dir.strip() == "" or "PLACEHOLDER" in args.out_dir:
        logger.error("ERROR: --out-dir is empty, blank, or contains PLACEHOLDER")
        sys.exit(1)
    
    # パラメータ検証
    if args.limit <= 0:
        logger.error(f"ERROR: --limit must be positive, got {args.limit}")
        sys.exit(1)
    
    if args.seed < 0:
        logger.error(f"ERROR: --seed must be non-negative, got {args.seed}")
        sys.exit(1)
    
    # グリッドサーチパラメータ検証（負数対応）
    alpha_values = parse_grid(args.alpha_grid, "alpha")
    beta_values = parse_grid(args.beta_grid, "beta")
    topk_values = parse_int_grid(args.topk_grid, "topk")
    
    # Alpha値検証（正数のみ）
    for alpha in alpha_values:
        if alpha <= 0:
            logger.error(f"ERROR: alpha values must be positive, got {alpha}")
            sys.exit(1)
    
    # Beta値検証（負数も許可、ただし-1.0未満は警告）
    for beta in beta_values:
        if beta < -1.0:
            logger.warning(f"WARNING: beta value {beta} is very negative (< -1.0), may cause instability")
    
    # TopK値検証（正整数のみ）
    for topk in topk_values:
        if topk <= 0:
            logger.error(f"ERROR: topk values must be positive, got {topk}")
            sys.exit(1)
    
    config = {
        'dataset': args.dataset.strip().lower(),
        'ppr_scores_path': Path(args.ppr_scores.strip()),
        'out_dir': Path(args.out_dir.strip()),
        'limit': args.limit,
        'seed': args.seed,
        'alpha_grid': alpha_values,
        'beta_grid': beta_values,
        'topk_grid': topk_values,
        'max_generations': getattr(args, 'max_gen', 50),
        'temperature': getattr(args, 'temperature', 0.3),
        'model_path': getattr(args, 'model_path', './chameleon_prime_personalization/models/base_model')
    }
    
    # ファイル存在確認
    if not config['ppr_scores_path'].exists():
        logger.error(f"ERROR: PPR scores file not found: {config['ppr_scores_path']}")
        sys.exit(1)
    
    return config

def load_ppr_scores(ppr_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PPRスコア読み込み"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading PPR scores: {ppr_path}")
    
    try:
        if ppr_path.suffix == '.npz':
            # Top-K PPR形式
            data = np.load(ppr_path)
            if 'indices' in data and 'scores' in data:
                ppr_indices = data['indices']
                ppr_scores = data['scores']
                logger.info(f"Loaded top-K PPR: {ppr_indices.shape} (sparse format)")
                
                # スパース→デンス変換（必要に応じて）
                n_nodes = ppr_indices.shape[0]
                max_idx = ppr_indices.max() + 1
                full_ppr = np.zeros((n_nodes, max_idx), dtype=np.float32)
                
                for i in range(n_nodes):
                    full_ppr[i, ppr_indices[i]] = ppr_scores[i]
                
                return full_ppr, {'format': 'top_k_sparse', 'k': ppr_indices.shape[1]}
            else:
                logger.error("ERROR: Invalid npz format, expected 'indices' and 'scores'")
                sys.exit(1)
        
        elif ppr_path.suffix == '.npy':
            # Full PPR形式
            ppr_scores = np.load(ppr_path)
            logger.info(f"Loaded full PPR: {ppr_scores.shape}")
            return ppr_scores, {'format': 'full_dense'}
        
        else:
            logger.error(f"ERROR: Unsupported PPR file format: {ppr_path.suffix}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load PPR scores: {e}")
        sys.exit(1)

def load_lamp2_dataset(limit: int) -> List[Dict[str, Any]]:
    """LaMP-2データセット読み込み"""
    logger = logging.getLogger(__name__)
    
    data_path = Path('./chameleon_prime_personalization/data/raw/LaMP-2')
    
    # dev_questionsとdev_outputs読み込み
    questions_file = data_path / 'dev_questions.json'
    outputs_file = data_path / 'dev_outputs.json'
    
    if not questions_file.exists() or not outputs_file.exists():
        logger.error(f"ERROR: LaMP-2 files not found: {questions_file}, {outputs_file}")
        sys.exit(1)
    
    logger.info(f"Loading LaMP-2 dataset: {questions_file}")
    
    try:
        with open(questions_file, 'r') as f:
            questions = json.load(f)
        
        with open(outputs_file, 'r') as f:
            outputs_data = json.load(f)
            outputs = {str(item['id']): item['output'] for item in outputs_data['golds']}
        
        logger.info(f"Loaded {len(questions)} questions, {len(outputs)} outputs")
        
    except Exception as e:
        logger.error(f"Failed to load LaMP-2 data: {e}")
        sys.exit(1)
    
    # データ結合
    dataset = []
    for item in questions:
        item_id = str(item['id'])
        if item_id in outputs:
            dataset.append({
                'id': item['id'],
                'user_id': str(item['id'])[0],  # LaMP-2 convention
                'question': item['input'],
                'profile': item.get('profile', []),
                'reference': outputs[item_id]
            })
    
    logger.info(f"Merged dataset: {len(dataset)} items")
    
    # 制限適用
    if limit > 0:
        dataset = dataset[:limit]
        logger.info(f"Limited to {len(dataset)} items")
    
    return dataset

def create_graph_enhanced_chameleon_editor(ppr_scores: np.ndarray, alpha: float, beta: float, 
                                          top_k: int, model_path: str) -> Any:
    """グラフ強化Chameleonエディタ作成"""
    logger = logging.getLogger(__name__)
    
    try:
        # Chameleonエディタのベースクラスをインポート
        sys.path.append('.')
        from chameleon_evaluator import ChameleonEditor
    except ImportError:
        logger.error("ERROR: ChameleonEditor not found, check import path")
        sys.exit(1)
    
    class GraphEnhancedChameleonEditor(ChameleonEditor):
        def __init__(self, model_path: str, ppr_scores: np.ndarray, alpha: float, beta: float, top_k: int, 
                     diag_token_level: bool = False, diag_kl: bool = False):
            super().__init__(model_path)
            self.ppr_scores = ppr_scores
            self.alpha = alpha
            self.beta = beta
            self.top_k = top_k
            
            # Configure optimized diagnostics (default: minimal logging)
            self.set_diagnostics_config(token_level=diag_token_level, kl_computation=diag_kl)
            
            # Load direction vectors (theta vectors)
            theta_p_path = "chameleon_prime_personalization/processed/LaMP-2/theta_p.json"
            theta_n_path = "chameleon_prime_personalization/processed/LaMP-2/theta_n.json"
            try:
                self.load_theta_vectors(theta_p_path, theta_n_path)
                logger.info("Direction vectors loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load direction vectors: {e}")
                raise
            
            logger.info(f"GraphEnhancedChameleonEditor initialized: alpha={alpha}, beta={beta}, top_k={top_k}")
        
        def get_collaborative_context(self, user_id: int) -> List[int]:
            """PPRベース協調ユーザー取得"""
            if user_id >= len(self.ppr_scores):
                return []
            
            user_ppr = self.ppr_scores[user_id]
            top_indices = np.argsort(user_ppr)[-self.top_k-1:-1][::-1]  # 自己除外
            
            # 最小PPRスコア閾値
            min_score = 0.01
            valid_indices = [int(idx) for idx in top_indices if user_ppr[idx] >= min_score]
            
            return valid_indices
        
        def generate_with_graph_context(self, prompt: str, user_id: int, user_profile: List[Dict], 
                                      max_new_tokens: int = 20, temperature: float = 0.0) -> str:
            """グラフコンテキスト付き生成 (要件準拠完全版)"""
            # 協調ユーザー取得
            collab_users = self.get_collaborative_context(user_id)
            
            # グラフ強化プロンプト構築
            enhanced_prompt = self._build_graph_enhanced_prompt(
                prompt, user_profile, collab_users, user_id
            )
            
            # Required generation settings per specification (warning-free)
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': temperature > 0.0,
                'use_cache': True
            }
            
            # Only include temperature if sampling is enabled
            if temperature > 0.0:
                gen_kwargs['temperature'] = temperature
            
            return self.generate_with_chameleon(
                enhanced_prompt, 
                alpha_personal=self.alpha,
                alpha_neutral=self.beta,
                gen_kwargs=gen_kwargs,
                last_k_tokens=16  # Required: default 16
            )
        
        def _build_graph_enhanced_prompt(self, prompt: str, user_profile: List[Dict], 
                                       collab_users: List[int], user_id: int) -> str:
            """グラフ強化プロンプト構築"""
            # ユーザープロファイル
            profile_text = ""
            if user_profile:
                profile_tags = [item.get('tag', '') for item in user_profile[:5]]
                profile_text = f"User {user_id} preferences: {', '.join(profile_tags)}"
            
            # 協調フィルタリング情報
            collab_text = ""
            if collab_users:
                collab_text = f"Similar users: {collab_users[:3]}"
            
            # 拡張プロンプト
            if profile_text or collab_text:
                context_parts = [p for p in [profile_text, collab_text] if p]
                enhanced_prompt = f"{' | '.join(context_parts)}\n\n{prompt}"
            else:
                enhanced_prompt = prompt
            
            return enhanced_prompt
    
    return GraphEnhancedChameleonEditor(model_path, ppr_scores, alpha, beta, top_k, 
                                       diag_token_level=False, diag_kl=False)

def run_single_evaluation(dataset: List[Dict], editor: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """単一設定での評価実行"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running evaluation on {len(dataset)} samples...")
    
    predictions = []
    references = []
    generation_times = []
    
    for i, sample in enumerate(dataset):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing {i + 1}/{len(dataset)}")
        
        try:
            # プロンプト構築
            prompt = f"Given this movie description, classify it with a single tag:\n\n{sample['question']}\n\nAnswer:"
            
            # グラフ強化生成 (最適化版)
            start_time = time.time()
            prediction = editor.generate_with_graph_context(
                prompt, 
                int(sample['user_id']), 
                sample['profile'],
                max_new_tokens=config['max_generations'],
                temperature=config.get('temperature', 0.0)
            )
            generation_time = time.time() - start_time
            
            # 予測クリーンアップ
            cleaned_prediction = prediction.strip().lower().split()[0] if prediction.strip() else "unknown"
            
            predictions.append(cleaned_prediction)
            references.append(sample['reference'].lower())
            generation_times.append(generation_time)
            
        except Exception as e:
            logger.warning(f"Generation failed for sample {i}: {e}")
            predictions.append("error")
            references.append(sample['reference'].lower())
            generation_times.append(0.0)
    
    # 評価メトリクス計算
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions) if predictions else 0.0
    
    avg_gen_time = np.mean(generation_times) if generation_times else 0.0
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(predictions),
        'avg_generation_time': avg_gen_time,
        'predictions': predictions,
        'references': references,
        'generation_times': generation_times
    }
    
    logger.info(f"Evaluation results: accuracy={accuracy:.4f} ({correct}/{len(predictions)})")
    
    return results

def run_grid_search(dataset: List[Dict], ppr_scores: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """グリッドサーチ実行"""
    logger = logging.getLogger(__name__)
    
    # パラメータ組み合わせ生成
    param_combinations = list(itertools.product(
        config['alpha_grid'],
        config['beta_grid'],
        config['topk_grid']
    ))
    
    logger.info(f"Starting grid search: {len(param_combinations)} combinations")
    logger.info(f"  Alpha: {config['alpha_grid']}")
    logger.info(f"  Beta: {config['beta_grid']}")
    logger.info(f"  Top-K: {config['topk_grid']}")
    
    results = []
    best_accuracy = 0.0
    best_params = None
    
    for i, (alpha, beta, topk) in enumerate(param_combinations):
        logger.info(f"\n=== Combination {i+1}/{len(param_combinations)}: alpha={alpha}, beta={beta}, topk={topk} ===")
        
        try:
            # エディタ作成
            editor = create_graph_enhanced_chameleon_editor(
                ppr_scores, alpha, beta, topk, config['model_path']
            )
            
            # Reset alpha reduction factor at start of each grid search combination
            if hasattr(editor, 'chameleon_editor') and hasattr(editor.chameleon_editor, '_alpha_reduction_factor'):
                editor.chameleon_editor._alpha_reduction_factor = 1.0
            
            # 評価実行
            eval_results = run_single_evaluation(dataset, editor, config)
            
            # パラメータ追加
            eval_results['parameters'] = {
                'alpha': alpha,
                'beta': beta,
                'topk': topk
            }
            
            results.append(eval_results)
            
            # 最良結果更新
            if eval_results['accuracy'] > best_accuracy:
                best_accuracy = eval_results['accuracy']
                best_params = (alpha, beta, topk)
            
            logger.info(f"Result: accuracy={eval_results['accuracy']:.4f}, time={eval_results['avg_generation_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed combination {i+1}: {e}")
            # エラー結果追加
            results.append({
                'parameters': {'alpha': alpha, 'beta': beta, 'topk': topk},
                'accuracy': 0.0,
                'error': str(e)
            })
    
    logger.info(f"\n=== Grid Search Completed ===")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    if best_params:
        logger.info(f"Best parameters: alpha={best_params[0]}, beta={best_params[1]}, topk={best_params[2]}")
    
    return {
        'results': results,
        'best_accuracy': best_accuracy,
        'best_parameters': best_params,
        'total_combinations': len(param_combinations)
    }

def save_evaluation_results(grid_results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """評価結果保存"""
    logger = logging.getLogger(__name__)
    
    # 実行ID生成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"graph_chameleon_{config['dataset']}_{timestamp}"
    
    # 出力ディレクトリ作成
    run_dir = config['out_dir'] / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # メイン結果保存
    results_file = run_dir / 'evaluation_results.json'
    
    save_data = {
        'run_id': run_id,
        'timestamp': timestamp,
        'dataset': config['dataset'],
        'ppr_source': str(config['ppr_scores_path']),
        'config': {
            'limit': config['limit'],
            'seed': config['seed'],
            'alpha_grid': config['alpha_grid'],
            'beta_grid': config['beta_grid'],
            'topk_grid': config['topk_grid'],
            'max_generations': config['max_generations'],
            'model_path': config['model_path']
        },
        'grid_search': grid_results
    }
    
    logger.info(f"Saving evaluation results: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # サマリー保存
    summary_file = run_dir / 'summary.json'
    
    summary = {
        'run_id': run_id,
        'best_accuracy': grid_results['best_accuracy'],
        'best_parameters': grid_results['best_parameters'],
        'total_combinations': grid_results['total_combinations'],
        'dataset_size': config['limit'],
        'top_3_results': sorted(
            [r for r in grid_results['results'] if 'accuracy' in r],
            key=lambda x: x['accuracy'],
            reverse=True
        )[:3]
    }
    
    logger.info(f"Saving summary: {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return str(run_dir)

def main():
    """メイン処理"""
    logger = setup_production_logging()
    
    parser = argparse.ArgumentParser(description="Graph-based Chameleon Evaluation with Grid Search")
    parser.add_argument('--dataset', choices=['lamp2'], required=True,
                       help='Dataset to evaluate on')
    parser.add_argument('--ppr-scores', required=True,
                       help='Path to PPR scores (.npy or .npz)')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=100,
                       help='Max samples to evaluate (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--alpha-grid', default='0.1,0.3,0.5',
                       help='Alpha values for grid search (CSV, default: 0.1,0.3,0.5)')
    parser.add_argument('--beta-grid', default='-0.05,-0.1,-0.2',
                       help='Beta values for grid search (CSV, default: -0.05,-0.1,-0.2)')
    parser.add_argument('--topk-grid', default='5,10,20',
                       help='Top-K values for grid search (CSV, default: 5,10,20)')
    parser.add_argument('--model-path', default='./chameleon_prime_personalization/models/base_model',
                       help='Path to base model')
    parser.add_argument('--max-gen', type=int, default=50,
                       help='Max generation tokens (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Generation temperature (default: 0.3)')
    
    args = parser.parse_args()
    
    # 設定検証
    config = validate_config(args)
    
    # Effective-config 1行出力
    effective_config = f"dataset={config['dataset']}, ppr={config['ppr_scores_path'].name}, limit={config['limit']}, alpha_grid={config['alpha_grid']}, beta_grid={config['beta_grid']}, topk_grid={config['topk_grid']}, seed={config['seed']}"
    logger.info(f"Effective-config: {effective_config}")
    
    try:
        # PPRスコア読み込み
        logger.info("=== STEP 1: Loading PPR scores ===")
        ppr_scores, ppr_metadata = load_ppr_scores(config['ppr_scores_path'])
        
        # データセット読み込み
        logger.info("=== STEP 2: Loading dataset ===")
        if config['dataset'] == 'lamp2':
            dataset = load_lamp2_dataset(config['limit'])
        else:
            logger.error(f"Unsupported dataset: {config['dataset']}")
            sys.exit(1)
        
        # グリッドサーチ実行
        logger.info("=== STEP 3: Running grid search evaluation ===")
        grid_results = run_grid_search(dataset, ppr_scores, config)
        
        # 結果保存
        logger.info("=== STEP 4: Saving results ===")
        run_dir = save_evaluation_results(grid_results, config)
        
        logger.info(f"[OK] Graph Chameleon evaluation completed")
        logger.info(f"Results saved in: {run_dir}")
        logger.info(f"Best accuracy: {grid_results['best_accuracy']:.4f}")
        if grid_results['best_parameters']:
            alpha, beta, topk = grid_results['best_parameters']
            logger.info(f"Best parameters: alpha={alpha}, beta={beta}, topk={topk}")
        
    except Exception as e:
        logger.error(f"Failed to run graph Chameleon evaluation: {e}")
        logger.error("Troubleshooting hints:")
        logger.error("  - Check PPR scores file format and content")
        logger.error("  - Verify dataset files are accessible")
        logger.error("  - Ensure model path exists and is valid")
        logger.error("  - Check grid search parameters are reasonable")
        logger.error("  - Install required packages: torch, transformers, sentence-transformers")
        sys.exit(1)
    
    logger.info("Graph-based Chameleon evaluation completed successfully")

if __name__ == "__main__":
    main()