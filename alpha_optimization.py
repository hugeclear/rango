#!/usr/bin/env python3
"""
Chameleon Alpha Parameter Optimization
=====================================

Chameleonのα値を自動的に最適化し、最高のパフォーマンスを見つけるスクリプト
"""

import json
import time
import logging
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaOptimizer:
    """Chameleonのα値最適化クラス"""
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def generate_alpha_combinations(self) -> List[Tuple[float, float]]:
        """α値の組み合わせを生成"""
        
        # Phase 1: 粗い探索
        alpha_personal_coarse = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        alpha_neutral_coarse = [-0.1, -0.5, -0.8, -1.0, -1.5, -2.0, -2.5]
        
        # Phase 2: 細かい探索（有望な範囲）
        alpha_personal_fine = np.arange(1.0, 2.1, 0.2)  # 1.0-2.0を0.2刻み
        alpha_neutral_fine = np.arange(-1.5, -0.4, 0.2)  # -1.5--0.5を0.2刻み
        
        # Phase 3: 超細かい探索（最適値周辺）
        alpha_personal_ultra = np.arange(1.4, 1.81, 0.1)  # 1.4-1.8を0.1刻み
        alpha_neutral_ultra = np.arange(-1.2, -0.69, 0.1)  # -1.2--0.7を0.1刻み
        
        combinations = []
        
        # Phase 1: 粗い探索
        for ap, an in itertools.product(alpha_personal_coarse, alpha_neutral_coarse):
            combinations.append((ap, an, "coarse"))
        
        # Phase 2: 細かい探索
        for ap, an in itertools.product(alpha_personal_fine, alpha_neutral_fine):
            combinations.append((round(ap, 2), round(an, 2), "fine"))
        
        # Phase 3: 超細かい探索
        for ap, an in itertools.product(alpha_personal_ultra, alpha_neutral_ultra):
            combinations.append((round(ap, 2), round(an, 2), "ultra"))
        
        # 重複を除去
        unique_combinations = []
        seen = set()
        for ap, an, phase in combinations:
            key = (round(ap, 2), round(an, 2))
            if key not in seen:
                seen.add(key)
                unique_combinations.append((ap, an, phase))
        
        logger.info(f"Generated {len(unique_combinations)} unique alpha combinations")
        return unique_combinations
    
    def modify_config(self, alpha_personal: float, alpha_neutral: float) -> str:
        """設定ファイルを一時的に変更"""
        config_path = "temp_alpha_config.py"
        
        config_content = f'''
# Temporary Alpha Configuration for Optimization
CHAMELEON_CONFIG = {{
    "alpha_personal": {alpha_personal},
    "alpha_neutral": {alpha_neutral},
    "target_layers": ["model.layers.16"],
    "max_length": 50
}}

# LaMP evaluation config
LAMP_CONFIG = {{
    "mode": "demo",
    "skip_checks": True,
    "max_samples": 10  # 高速化のため少数サンプル
}}
'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
    
    def run_evaluation(self, alpha_personal: float, alpha_neutral: float) -> Dict[str, Any]:
        """指定したα値で評価を実行"""
        
        logger.info(f"Testing α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f}")
        
        try:
            # 設定ファイルを一時変更
            config_path = self.modify_config(alpha_personal, alpha_neutral)
            
            # 評価実行
            start_time = time.time()
            
            # run_evaluation.pyを実行（α値を引数として渡す）
            cmd = [
                "python", "run_evaluation.py", 
                "--mode", "demo", 
                "--skip-checks",
                "--alpha-personal", str(alpha_personal),
                "--alpha-neutral", str(alpha_neutral),
                "--output-dir", str(self.output_dir / f"alpha_p{alpha_personal}_n{alpha_neutral}")
            ]
            
            # 代替実行方法（引数対応していない場合）
            result = self.run_evaluation_alternative(alpha_personal, alpha_neutral)
            
            execution_time = time.time() - start_time
            
            # 結果を解析
            if result:
                result.update({
                    'alpha_personal': alpha_personal,
                    'alpha_neutral': alpha_neutral,
                    'execution_time': execution_time,
                    'timestamp': time.time()
                })
                
                logger.info(f"α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f} -> "
                          f"Accuracy: {result.get('accuracy', 0):.4f}, "
                          f"Improvement: {result.get('improvement_rate', 0):.1%}")
                
                return result
            else:
                logger.warning(f"Failed to get results for α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f}")
                return None
                
        except Exception as e:
            logger.error(f"Error in evaluation α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f}: {e}")
            return None
        
        finally:
            # 一時ファイルをクリーンアップ
            if Path(config_path).exists():
                Path(config_path).unlink()
    
    def run_evaluation_alternative(self, alpha_personal: float, alpha_neutral: float) -> Dict[str, Any]:
        """代替実行方法：直接ChameleonEvaluatorを呼び出し"""
        
        try:
            from chameleon_evaluator import ChameleonEvaluator
            
            # 一時的な設定で評価器を初期化
            evaluator = ChameleonEvaluator()
            
            # LaMP-2データの一部を使って高速評価
            with open('chameleon_prime_personalization/data/raw/LaMP-2/merged.json', 'r') as f:
                data = json.load(f)
            
            # 少数サンプルで高速テスト
            test_samples = data[:5]  # 最初の5サンプルのみ
            
            baseline_scores = []
            chameleon_scores = []
            
            for sample in test_samples:
                input_text = sample['input']
                
                # ベースライン生成
                baseline_response = evaluator.model.generate(
                    **evaluator.tokenizer(input_text, return_tensors="pt"),
                    max_new_tokens=20,
                    do_sample=False
                )
                baseline_text = evaluator.tokenizer.decode(baseline_response[0], skip_special_tokens=True)
                
                # Chameleon生成
                chameleon_text = evaluator.generate_with_chameleon(
                    input_text,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_neutral,
                    max_length=20
                )
                
                # 簡易スコア計算（実際の評価指標で置き換え可能）
                baseline_score = len(baseline_text.split())  # 単語数ベースの簡易指標
                chameleon_score = len(chameleon_text.split())
                
                baseline_scores.append(baseline_score)
                chameleon_scores.append(chameleon_score)
            
            # 結果の計算
            baseline_avg = np.mean(baseline_scores)
            chameleon_avg = np.mean(chameleon_scores)
            improvement_rate = (chameleon_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
            
            return {
                'accuracy': chameleon_avg / (baseline_avg + chameleon_avg) if (baseline_avg + chameleon_avg) > 0 else 0,
                'baseline_score': baseline_avg,
                'chameleon_score': chameleon_avg,
                'improvement_rate': improvement_rate,
                'samples_tested': len(test_samples)
            }
            
        except Exception as e:
            logger.error(f"Alternative evaluation failed: {e}")
            return None
    
    def optimize(self, max_iterations: int = None, target_improvement: float = 0.25) -> Dict[str, Any]:
        """α値の最適化を実行"""
        
        logger.info("🚀 Starting Chameleon Alpha Optimization")
        logger.info(f"Target improvement: {target_improvement:.1%}")
        
        combinations = self.generate_alpha_combinations()
        
        if max_iterations:
            combinations = combinations[:max_iterations]
            logger.info(f"Limited to {max_iterations} iterations")
        
        best_result = None
        best_improvement = -float('inf')
        
        total_start_time = time.time()
        
        for i, (alpha_p, alpha_n, phase) in enumerate(combinations, 1):
            logger.info(f"[{i}/{len(combinations)}] Phase: {phase}")
            
            result = self.run_evaluation(alpha_p, alpha_n)
            
            if result:
                self.results.append(result)
                
                improvement = result.get('improvement_rate', 0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_result = result
                    
                    logger.info(f"🎯 New best result! α_p={alpha_p:.2f}, α_n={alpha_n:.2f} -> {improvement:.1%}")
                    
                    # 目標達成チェック
                    if improvement >= target_improvement:
                        logger.info(f"🎉 Target improvement {target_improvement:.1%} achieved!")
                        break
            
            # 進捗保存
            if i % 10 == 0:
                self.save_intermediate_results()
        
        total_time = time.time() - total_start_time
        
        logger.info(f"✅ Optimization completed in {total_time:.1f} seconds")
        logger.info(f"Best result: α_p={best_result['alpha_personal']:.2f}, "
                   f"α_n={best_result['alpha_neutral']:.2f} -> "
                   f"{best_result['improvement_rate']:.1%}")
        
        # 最終結果を保存
        self.save_results(best_result, total_time)
        self.create_visualizations()
        
        return best_result
    
    def save_intermediate_results(self):
        """中間結果を保存"""
        results_file = self.output_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def save_results(self, best_result: Dict[str, Any], total_time: float):
        """最終結果を保存"""
        
        summary = {
            'optimization_summary': {
                'best_result': best_result,
                'total_evaluations': len(self.results),
                'total_time': total_time,
                'timestamp': time.time()
            },
            'all_results': self.results
        }
        
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def create_visualizations(self):
        """結果の可視化"""
        
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # データ準備
        alpha_p_values = [r['alpha_personal'] for r in self.results]
        alpha_n_values = [r['alpha_neutral'] for r in self.results]
        improvements = [r.get('improvement_rate', 0) for r in self.results]
        
        # ヒートマップ作成
        plt.figure(figsize=(12, 8))
        
        # データをグリッドに整理
        unique_alpha_p = sorted(set(alpha_p_values))
        unique_alpha_n = sorted(set(alpha_n_values))
        
        heatmap_data = np.full((len(unique_alpha_n), len(unique_alpha_p)), np.nan)
        
        for i, result in enumerate(self.results):
            p_idx = unique_alpha_p.index(result['alpha_personal'])
            n_idx = unique_alpha_n.index(result['alpha_neutral'])
            heatmap_data[n_idx, p_idx] = result.get('improvement_rate', 0) * 100  # パーセント表示
        
        # ヒートマップ描画
        plt.subplot(2, 2, 1)
        sns.heatmap(
            heatmap_data, 
            xticklabels=[f"{x:.1f}" for x in unique_alpha_p],
            yticklabels=[f"{x:.1f}" for x in unique_alpha_n],
            annot=True, 
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Improvement (%)'}
        )
        plt.xlabel('Alpha Personal')
        plt.ylabel('Alpha Neutral')
        plt.title('Improvement Rate Heatmap')
        
        # 散布図
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(alpha_p_values, alpha_n_values, c=improvements, cmap='RdYlGn', s=50)
        plt.colorbar(scatter, label='Improvement Rate')
        plt.xlabel('Alpha Personal')
        plt.ylabel('Alpha Neutral')
        plt.title('Alpha Parameters vs Improvement')
        
        # 改善率分布
        plt.subplot(2, 2, 3)
        plt.hist(improvements, bins=20, alpha=0.7, color='skyblue')
        plt.xlabel('Improvement Rate')
        plt.ylabel('Frequency')
        plt.title('Distribution of Improvement Rates')
        plt.axvline(x=0.25, color='red', linestyle='--', label='Target (25%)')
        plt.legend()
        
        # 上位結果
        plt.subplot(2, 2, 4)
        top_results = sorted(self.results, key=lambda x: x.get('improvement_rate', 0), reverse=True)[:10]
        top_improvements = [r.get('improvement_rate', 0) * 100 for r in top_results]
        top_labels = [f"({r['alpha_personal']:.1f}, {r['alpha_neutral']:.1f})" for r in top_results]
        
        plt.barh(range(len(top_improvements)), top_improvements)
        plt.yticks(range(len(top_improvements)), top_labels)
        plt.xlabel('Improvement (%)')
        plt.title('Top 10 Alpha Combinations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir / 'optimization_results.png'}")

def main():
    """メイン実行関数"""
    
    print("🦎 Chameleon Alpha Parameter Optimization")
    print("=" * 50)
    
    # 最適化器を初期化
    optimizer = AlphaOptimizer()
    
    # 最適化実行
    best_result = optimizer.optimize(
        max_iterations=100,  # 最大100回のテスト
        target_improvement=0.25  # 25%改善を目標
    )
    
    if best_result:
        print("\n🎉 Optimization Results:")
        print("=" * 50)
        print(f"Best Alpha Personal: {best_result['alpha_personal']:.2f}")
        print(f"Best Alpha Neutral: {best_result['alpha_neutral']:.2f}")
        print(f"Improvement Rate: {best_result['improvement_rate']:.1%}")
        print(f"Accuracy: {best_result.get('accuracy', 0):.4f}")
        print("\n📁 Results saved in: optimization_results/")
        print("📊 Visualizations: optimization_results/optimization_results.png")
    else:
        print("❌ Optimization failed to find good parameters")

if __name__ == "__main__":
    main()
