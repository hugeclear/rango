#!/usr/bin/env python3
"""
CFS-Chameleon総合評価実験スクリプト
Direction vectors適用済みシステムの全パラメータ組み合わせ評価
"""

import subprocess
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CFSComprehensiveEvaluator:
    def __init__(self):
        # 実験パラメータ設定
        self.hook_layers = ["model.layers.14.mlp", "model.layers.16.mlp", "model.layers.18.mlp"]
        self.alpha_personal_values = [0.05, 0.1, 0.2, 0.3]
        self.rank_reduction_values = [8, 16, 32]
        self.alpha_general = -0.05  # 固定値
        self.config_path = "cfs_config.yaml"
        
        # 結果保存用
        self.results = []
        self.evaluation_count = 0
        self.total_conditions = len(self.hook_layers) * len(self.alpha_personal_values) * len(self.rank_reduction_values)
        
        logger.info(f"🧪 CFS-Chameleon総合評価開始")
        logger.info(f"📊 総実験条件数: {self.total_conditions}")
        logger.info(f"🔧 Hook layers: {self.hook_layers}")
        logger.info(f"📈 Alpha personal: {self.alpha_personal_values}")
        logger.info(f"🎯 Rank reduction: {self.rank_reduction_values}")
    
    def run_single_evaluation(self, hook_layer: str, alpha_p: float, rank_reduction: int) -> Dict[str, Any]:
        """単一条件での評価実行"""
        self.evaluation_count += 1
        logger.info(f"🔄 実験 {self.evaluation_count}/{self.total_conditions}: hook={hook_layer}, α_p={alpha_p}, rank={rank_reduction}")
        
        # コマンド構築
        cmd = [
            "python", "lampqa_cfs_benchmark.py",
            "--compare_modes",
            "--use_collaboration", 
            f"--config={self.config_path}",
            f"--hook_layer={hook_layer}",
            f"--alpha_p={alpha_p}",
            f"--alpha_g={self.alpha_general}",
            f"--rank_reduction={rank_reduction}"
        ]
        
        try:
            # 評価実行
            start_time = time.time()
            env = dict(subprocess.os.environ)
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
            return self._create_error_result(hook_layer, alpha_p, rank_reduction, "TIMEOUT")
            
        except Exception as e:
            logger.error(f"❌ 実験エラー: {hook_layer}, α_p={alpha_p}, rank={rank_reduction} - {e}")
            return self._create_error_result(hook_layer, alpha_p, rank_reduction, "ERROR")
    
    def _create_error_result(self, hook_layer: str, alpha_p: float, rank_reduction: int, status: str) -> Dict[str, Any]:
        """エラー時の結果データ作成"""
        return {
            "hook_layer": hook_layer,
            "alpha_p": alpha_p,
            "rank_reduction": rank_reduction,
            "execution_time_total": 0.0,
            "legacy_rouge_l": 0.0,
            "legacy_bleu": 0.0,
            "legacy_bert_score": 0.0,
            "legacy_inference_time": 0.0,
            "cfs_rouge_l": 0.0,
            "cfs_bleu": 0.0,
            "cfs_bert_score": 0.0,
            "cfs_inference_time": 0.0,
            "cfs_pool_utilization": 0.0,
            "rouge_l_improvement": 0.0,
            "bleu_improvement": 0.0,
            "bert_improvement": 0.0,
            "speed_improvement": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "status": status
        }
    
    def run_comprehensive_evaluation(self):
        """総合評価実行"""
        logger.info("🚀 総当り実験開始")
        
        total_start_time = time.time()
        
        for hook_layer in self.hook_layers:
            for alpha_p in self.alpha_personal_values:
                for rank_reduction in self.rank_reduction_values:
                    result = self.run_single_evaluation(hook_layer, alpha_p, rank_reduction)
                    self.results.append(result)
                    
                    # 途中経過表示
                    if self.evaluation_count % 5 == 0:
                        elapsed = time.time() - total_start_time
                        remaining = (elapsed / self.evaluation_count) * (self.total_conditions - self.evaluation_count)
                        logger.info(f"📊 進捗: {self.evaluation_count}/{self.total_conditions} ({self.evaluation_count/self.total_conditions*100:.1f}%)")
                        logger.info(f"⏱️  経過時間: {elapsed/60:.1f}分, 推定残り時間: {remaining/60:.1f}分")
        
        total_execution_time = time.time() - total_start_time
        logger.info(f"🎉 全実験完了! 総実行時間: {total_execution_time/60:.1f}分")
        
        # 結果保存と分析
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """結果保存"""
        # DataFrame作成
        df = pd.DataFrame(self.results)
        
        # CSV保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"cfs_comprehensive_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📁 結果保存: {csv_path}")
        
        # JSON保存
        json_path = f"cfs_comprehensive_evaluation_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"📁 結果保存: {json_path}")
        
        self.csv_path = csv_path
        self.json_path = json_path
    
    def analyze_results(self):
        """結果分析とレポート生成"""
        df = pd.DataFrame(self.results)
        
        # 成功した実験のみ分析
        success_df = df[df['status'] == 'SUCCESS'].copy()
        
        if len(success_df) == 0:
            logger.error("❌ 成功した実験がありません")
            return
        
        logger.info("📊 結果分析開始")
        
        # 各指標での最良条件
        best_rouge_l = success_df.loc[success_df['cfs_rouge_l'].idxmax()]
        best_bleu = success_df.loc[success_df['cfs_bleu'].idxmax()]
        best_bert_score = success_df.loc[success_df['cfs_bert_score'].idxmax()]
        
        # 総合スコア計算（3指標の平均改善率）
        success_df['total_improvement'] = (
            success_df['rouge_l_improvement'] + 
            success_df['bleu_improvement'] + 
            success_df['bert_improvement']
        ) / 3
        
        best_overall = success_df.loc[success_df['total_improvement'].idxmax()]
        
        # レポート出力
        print("\n" + "="*80)
        print("🎯 CFS-CHAMELEON 総合評価結果レポート")
        print("="*80)
        
        print("\n📊 各指標での最良条件:")
        print(f"🏆 ROUGE-L最高値: {best_rouge_l['cfs_rouge_l']:.4f}")
        print(f"   条件: hook={best_rouge_l['hook_layer']}, α_p={best_rouge_l['alpha_p']}, rank={best_rouge_l['rank_reduction']}")
        
        print(f"🏆 BLEU最高値: {best_bleu['cfs_bleu']:.4f}")
        print(f"   条件: hook={best_bleu['hook_layer']}, α_p={best_bleu['alpha_p']}, rank={best_bleu['rank_reduction']}")
        
        print(f"🏆 BERTScore最高値: {best_bert_score['cfs_bert_score']:.4f}")
        print(f"   条件: hook={best_bert_score['hook_layer']}, α_p={best_bert_score['alpha_p']}, rank={best_bert_score['rank_reduction']}")
        
        print(f"\n🎯 総合最適条件（平均改善率最大）:")
        print(f"   Hook Layer: {best_overall['hook_layer']}")
        print(f"   Alpha Personal: {best_overall['alpha_p']}")
        print(f"   Rank Reduction: {best_overall['rank_reduction']}")
        print(f"   平均改善率: {best_overall['total_improvement']:.4f}%")
        print(f"   ROUGE-L: {best_overall['cfs_rouge_l']:.4f}")
        print(f"   BLEU: {best_overall['cfs_bleu']:.4f}")
        print(f"   BERTScore: {best_overall['cfs_bert_score']:.4f}")
        print(f"   推論時間: {best_overall['cfs_inference_time']:.4f}秒")
        print(f"   プール利用率: {best_overall['cfs_pool_utilization']:.2f}%")
        
        # 統計サマリー
        print(f"\n📈 統計サマリー:")
        print(f"   成功実験数: {len(success_df)}/{len(df)}")
        print(f"   ROUGE-L平均: {success_df['cfs_rouge_l'].mean():.4f} (σ={success_df['cfs_rouge_l'].std():.4f})")
        print(f"   BLEU平均: {success_df['cfs_bleu'].mean():.4f} (σ={success_df['cfs_bleu'].std():.4f})")
        print(f"   BERTScore平均: {success_df['cfs_bert_score'].mean():.4f} (σ={success_df['cfs_bert_score'].std():.4f})")
        
        # 考察
        print(f"\n💡 考察:")
        hook_analysis = success_df.groupby('hook_layer')['total_improvement'].mean().sort_values(ascending=False)
        alpha_analysis = success_df.groupby('alpha_p')['total_improvement'].mean().sort_values(ascending=False)
        rank_analysis = success_df.groupby('rank_reduction')['total_improvement'].mean().sort_values(ascending=False)
        
        print(f"   最良Hook層: {hook_analysis.index[0]} (平均改善率: {hook_analysis.iloc[0]:.4f}%)")
        print(f"   最良Alpha値: {alpha_analysis.index[0]} (平均改善率: {alpha_analysis.iloc[0]:.4f}%)")
        print(f"   最良Rank値: {rank_analysis.index[0]} (平均改善率: {rank_analysis.iloc[0]:.4f}%)")
        
        print("="*80)

if __name__ == "__main__":
    evaluator = CFSComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()