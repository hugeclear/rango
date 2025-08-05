#!/usr/bin/env python3
"""
CFS-Chameleon重要パラメータ組み合わせ評価
効率的な評価のため条件を絞り込み
"""

import subprocess
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging
import os

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_evaluation(hook_layer: str, alpha_p: float, rank_reduction: int) -> Dict[str, Any]:
    """単一条件での評価実行"""
    logger.info(f"🔄 実験: hook={hook_layer}, α_p={alpha_p}, rank={rank_reduction}")
    
    # コマンド構築
    cmd = [
        "python", "lampqa_cfs_benchmark.py",
        "--compare_modes",
        "--use_collaboration", 
        f"--config=cfs_config.yaml",
        f"--hook_layer={hook_layer}",
        f"--alpha_p={alpha_p}",
        f"--alpha_g=-0.05",
        f"--rank_reduction={rank_reduction}"
    ]
    
    try:
        # 評価実行
        start_time = time.time()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分タイムアウト
            shell=False,
            env=env
        )
        execution_time = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"❌ Command failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        # 最新結果ファイル取得
        results_dir = Path("lampqa_evaluation_results")
        if not results_dir.exists():
            raise FileNotFoundError("Results directory not found")
        
        latest_result_file = max(results_dir.glob("lampqa_comparison_*.json"), key=lambda x: x.stat().st_mtime)
        
        with open(latest_result_file, 'r') as f:
            evaluation_data = json.load(f)
        
        # 結果抽出
        legacy = evaluation_data.get("legacy_chameleon", {})
        cfs = evaluation_data.get("cfs_chameleon", {})
        improvements = evaluation_data.get("improvement_metrics", {})
        significance = evaluation_data.get("statistical_significance", {})
        
        result_data = {
            # 実験条件
            "hook_layer": hook_layer,
            "alpha_p": alpha_p,
            "rank_reduction": rank_reduction,
            "execution_time_total": round(execution_time, 2),
            
            # Legacy Chameleon結果
            "legacy_rouge_l": round(legacy.get("rouge_l", 0.0), 4),
            "legacy_bleu": round(legacy.get("bleu_score", 0.0), 4),
            "legacy_bert_score": round(legacy.get("bert_score_f1", 0.0), 4),
            "legacy_inference_time": round(legacy.get("inference_time", 0.0), 4),
            
            # CFS-Chameleon結果
            "cfs_rouge_l": round(cfs.get("rouge_l", 0.0), 4),
            "cfs_bleu": round(cfs.get("bleu_score", 0.0), 4),
            "cfs_bert_score": round(cfs.get("bert_score_f1", 0.0), 4),
            "cfs_inference_time": round(cfs.get("inference_time", 0.0), 4),
            "cfs_pool_utilization": round(cfs.get("pool_utilization", 0.0) * 100, 2),  # %表示
            
            # 改善指標
            "rouge_l_improvement": round(improvements.get("rouge_l_improvement", 0.0), 4),
            "bleu_improvement": round(improvements.get("bleu_improvement", 0.0), 4),
            "bert_improvement": round(improvements.get("bert_improvement", 0.0), 4),
            "speed_improvement": round(improvements.get("speed_improvement", 0.0), 4),
            
            # 統計的有意性
            "p_value": round(significance.get("p_value", 1.0), 6),
            "is_significant": significance.get("is_significant", False),
            
            # ステータス
            "status": "SUCCESS"
        }
        
        logger.info(f"✅ 実験完了: ROUGE-L={result_data['cfs_rouge_l']}, BLEU={result_data['cfs_bleu']}, BERTScore={result_data['cfs_bert_score']}")
        return result_data
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 実験タイムアウト: {hook_layer}, α_p={alpha_p}, rank={rank_reduction}")
        return {"status": "TIMEOUT", "hook_layer": hook_layer, "alpha_p": alpha_p, "rank_reduction": rank_reduction}
        
    except Exception as e:
        logger.error(f"❌ 実験エラー: {hook_layer}, α_p={alpha_p}, rank={rank_reduction} - {e}")
        return {"status": "ERROR", "hook_layer": hook_layer, "alpha_p": alpha_p, "rank_reduction": rank_reduction}

def main():
    """メイン実行"""
    # 重要な条件のみに絞り込み（16条件）
    conditions = [
        # Hook Layer 14 (config default)
        ("model.layers.14.mlp", 0.05, 16),
        ("model.layers.14.mlp", 0.1, 16),
        ("model.layers.14.mlp", 0.2, 16),
        ("model.layers.14.mlp", 0.3, 16),
        
        # Hook Layer 16 (中間層)
        ("model.layers.16.mlp", 0.05, 16),
        ("model.layers.16.mlp", 0.1, 16),
        ("model.layers.16.mlp", 0.2, 16),
        ("model.layers.16.mlp", 0.3, 16),
        
        # Hook Layer 18 (深い層)
        ("model.layers.18.mlp", 0.05, 16),
        ("model.layers.18.mlp", 0.1, 16),
        ("model.layers.18.mlp", 0.2, 16),
        ("model.layers.18.mlp", 0.3, 16),
        
        # Rank Reduction の影響確認（Layer 16, Alpha 0.1固定）
        ("model.layers.16.mlp", 0.1, 8),
        ("model.layers.16.mlp", 0.1, 32),
        
        # 極値確認
        ("model.layers.16.mlp", 0.01, 16),  # 非常に小さいAlpha
        ("model.layers.16.mlp", 0.5, 16),   # 大きなAlpha
    ]
    
    logger.info(f"🧪 CFS-Chameleon重要パラメータ評価開始")
    logger.info(f"📊 総実験条件数: {len(conditions)}")
    
    results = []
    total_start_time = time.time()
    
    for i, (hook_layer, alpha_p, rank_reduction) in enumerate(conditions, 1):
        logger.info(f"📈 進捗: {i}/{len(conditions)} ({i/len(conditions)*100:.1f}%)")
        
        result = run_single_evaluation(hook_layer, alpha_p, rank_reduction)
        results.append(result)
        
        # 途中経過表示
        if i % 4 == 0:
            elapsed = time.time() - total_start_time
            remaining = (elapsed / i) * (len(conditions) - i)
            logger.info(f"⏱️  経過時間: {elapsed/60:.1f}分, 推定残り時間: {remaining/60:.1f}分")
    
    total_execution_time = time.time() - total_start_time
    logger.info(f"🎉 全実験完了! 総実行時間: {total_execution_time/60:.1f}分")
    
    # 結果保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # DataFrame作成（成功した実験のみ）
    success_results = [r for r in results if r.get("status") == "SUCCESS"]
    
    if success_results:
        df = pd.DataFrame(success_results)
        
        # CSV保存
        csv_path = f"cfs_quick_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📁 結果保存: {csv_path}")
        
        # 結果分析
        print("\n" + "="*80)
        print("🎯 CFS-CHAMELEON重要パラメータ評価結果")
        print("="*80)
        
        # 各指標での最良条件
        best_rouge_l = df.loc[df['cfs_rouge_l'].idxmax()]
        best_bleu = df.loc[df['cfs_bleu'].idxmax()]
        best_bert_score = df.loc[df['cfs_bert_score'].idxmax()]
        
        print(f"\n🏆 各指標での最良条件:")
        print(f"ROUGE-L最高値: {best_rouge_l['cfs_rouge_l']:.4f}")
        print(f"  条件: hook={best_rouge_l['hook_layer']}, α_p={best_rouge_l['alpha_p']}, rank={best_rouge_l['rank_reduction']}")
        
        print(f"BLEU最高値: {best_bleu['cfs_bleu']:.4f}")
        print(f"  条件: hook={best_bleu['hook_layer']}, α_p={best_bleu['alpha_p']}, rank={best_bleu['rank_reduction']}")
        
        print(f"BERTScore最高値: {best_bert_score['cfs_bert_score']:.4f}")
        print(f"  条件: hook={best_bert_score['hook_layer']}, α_p={best_bert_score['alpha_p']}, rank={best_bert_score['rank_reduction']}")
        
        # 総合スコア
        df['total_improvement'] = (df['rouge_l_improvement'] + df['bleu_improvement'] + df['bert_improvement']) / 3
        best_overall = df.loc[df['total_improvement'].idxmax()]
        
        print(f"\n🎯 総合最適条件:")
        print(f"  Hook Layer: {best_overall['hook_layer']}")
        print(f"  Alpha Personal: {best_overall['alpha_p']}")
        print(f"  Rank Reduction: {best_overall['rank_reduction']}")
        print(f"  平均改善率: {best_overall['total_improvement']:.4f}%")
        print(f"  ROUGE-L: {best_overall['cfs_rouge_l']:.4f}")
        print(f"  BLEU: {best_overall['cfs_bleu']:.4f}") 
        print(f"  BERTScore: {best_overall['cfs_bert_score']:.4f}")
        print(f"  推論時間: {best_overall['cfs_inference_time']:.4f}秒")
        print(f"  プール利用率: {best_overall['cfs_pool_utilization']:.2f}%")
        
        # 詳細表
        print(f"\n📊 詳細結果表:")
        display_columns = ['hook_layer', 'alpha_p', 'rank_reduction', 'cfs_rouge_l', 'cfs_bleu', 'cfs_bert_score', 'cfs_inference_time', 'cfs_pool_utilization']
        print(df[display_columns].to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
        
        print("="*80)
        
    else:
        logger.error("❌ 成功した実験がありません")
        # エラー結果も保存
        error_df = pd.DataFrame(results)
        error_csv = f"cfs_evaluation_errors_{timestamp}.csv"
        error_df.to_csv(error_csv, index=False)
        logger.info(f"📁 エラー結果保存: {error_csv}")

if __name__ == "__main__":
    main()