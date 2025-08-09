#!/usr/bin/env python3
"""
CFS-Chameleon ハイパーパラメータチューニング機能
グリッドサーチによる最適α値探索
"""

import sys
import os
import argparse
import json
import time
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append('/home/nakata/master_thesis/rango')
os.chdir('/home/nakata/master_thesis/rango')

print('🎯 CFS-CHAMELEON HYPERPARAMETER TUNING')
print('='*80)

from chameleon_cfs_integrator import CollaborativeChameleonEditor

class HyperparameterTuner:
    """ハイパーパラメータチューニングクラス"""
    
    def __init__(self, config_path: str = 'cfs_config.yaml'):
        self.config_path = config_path
        self.results = []
        self.best_params = None
        self.best_score = -float('inf')
    
    def load_test_data(self, max_samples: int = 6) -> List[Dict[str, Any]]:
        """テストデータ読み込み（チューニング用少数サンプル）"""
        try:
            data_path = Path("chameleon_prime_personalization/data/raw/LaMP-2/merged.json")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # チューニング用に少数サンプル使用
            test_samples = data[:max_samples]
            
            print(f"✅ Test data loaded: {len(test_samples)} samples for tuning")
            return test_samples
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: Failed to load test data: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def evaluate_parameter_combination(self, alpha_p: float, alpha_n: float, 
                                     test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """パラメータ組み合わせの評価"""
        print(f"\n🔧 Testing α_p={alpha_p}, α_n={alpha_n}")
        
        try:
            # CFS-Chameleonエディター作成
            editor = CollaborativeChameleonEditor(
                use_collaboration=True,
                config_path=self.config_path
            )
            
            results = []
            total_start = time.time()
            
            for i, sample in enumerate(test_samples):
                input_text = sample.get('input', '')[:120]
                expected = sample.get('output', 'N/A')
                user_id = f"tune_user_{i % 2}"  # 2ユーザーでテスト
                
                sample_start = time.time()
                
                try:
                    # 生成実行
                    generated = editor.generate_with_collaborative_chameleon(
                        prompt=input_text,
                        user_id=user_id,
                        alpha_personal=alpha_p,
                        alpha_neutral=alpha_n,
                        max_length=100  # チューニング用短い生成
                    )
                    
                    generation_time = time.time() - sample_start
                    
                    # 簡単な品質メトリクス
                    quality_score = self._compute_quality_score(generated, expected, input_text)
                    
                    results.append({
                        'sample_id': i,
                        'user_id': user_id,
                        'input': input_text,
                        'expected': expected,
                        'generated': generated,
                        'generation_time': generation_time,
                        'quality_score': quality_score,
                        'status': 'success'
                    })
                    
                    print(f"     Sample {i+1}: Quality={quality_score:.3f}, Time={generation_time:.2f}s")
                    
                except Exception as e:
                    results.append({
                        'sample_id': i,
                        'user_id': user_id,
                        'input': input_text,
                        'expected': expected,
                        'generated': f'ERROR: {e}',
                        'generation_time': time.time() - sample_start,
                        'quality_score': 0.0,
                        'status': 'failed'
                    })
                    
                    print(f"     Sample {i+1}: FAILED - {e}")
            
            total_time = time.time() - total_start
            
            # 統計計算
            successful_results = [r for r in results if r['status'] == 'success']
            success_rate = len(successful_results) / len(results) if results else 0
            
            if successful_results:
                avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
                avg_time = sum(r['generation_time'] for r in successful_results) / len(successful_results)
                
                # 総合スコア計算 (品質重視、時間効率も考慮)
                overall_score = (0.7 * avg_quality + 0.2 * success_rate - 0.1 * (avg_time / 10.0))
            else:
                avg_quality = 0.0
                avg_time = 0.0
                overall_score = 0.0
            
            evaluation = {
                'alpha_p': alpha_p,
                'alpha_n': alpha_n,
                'success_rate': success_rate,
                'avg_quality_score': avg_quality,
                'avg_generation_time': avg_time,
                'overall_score': overall_score,
                'total_evaluation_time': total_time,
                'sample_results': results
            }
            
            print(f"   📊 Success Rate: {success_rate*100:.1f}%")
            print(f"   📊 Avg Quality: {avg_quality:.3f}")
            print(f"   📊 Avg Time: {avg_time:.2f}s")
            print(f"   🎯 Overall Score: {overall_score:.4f}")
            
            return evaluation
            
        except Exception as e:
            error_msg = f"❌ Parameter evaluation failed: {e}"
            print(f"   {error_msg}")
            
            return {
                'alpha_p': alpha_p,
                'alpha_n': alpha_n,
                'success_rate': 0.0,
                'avg_quality_score': 0.0,
                'avg_generation_time': 0.0,
                'overall_score': -1.0,  # ペナルティ
                'total_evaluation_time': 0.0,
                'error': str(e),
                'sample_results': []
            }
    
    def _compute_quality_score(self, generated: str, expected: str, input_text: str) -> float:
        """簡易品質スコア計算"""
        if not generated or len(generated.strip()) == 0:
            return 0.0
        
        # 基本品質チェック
        quality = 0.0
        
        # 1. 長さチェック (10-300文字が適正)
        length_score = 1.0
        if len(generated) < 10:
            length_score = len(generated) / 10.0
        elif len(generated) > 300:
            length_score = max(0.0, 1.0 - (len(generated) - 300) / 300.0)
        
        quality += 0.3 * length_score
        
        # 2. 反復パターン検出 (品質低下指標)
        repetition_penalty = 0.0
        words = generated.split()
        if len(words) > 5:
            unique_words = set(words)
            repetition_rate = 1.0 - (len(unique_words) / len(words))
            if repetition_rate > 0.5:  # 50%以上の反復はペナルティ
                repetition_penalty = repetition_rate * 0.5
        
        quality += 0.3 * (1.0 - repetition_penalty)
        
        # 3. 入力との関連性（簡易）
        input_words = set(input_text.lower().split())
        generated_words = set(generated.lower().split())
        if input_words and generated_words:
            relevance_score = len(input_words.intersection(generated_words)) / len(input_words.union(generated_words))
        else:
            relevance_score = 0.0
        
        quality += 0.2 * relevance_score
        
        # 4. 文章として成立しているか
        structure_score = 0.5  # デフォルト
        if any(char.isalpha() for char in generated):  # アルファベット含有
            structure_score += 0.2
        if any(char in '.!?' for char in generated):  # 句読点含有
            structure_score += 0.2
        if not any(char in r'/\*[]{}()' for char in generated[:50]):  # 記号の過度含有なし
            structure_score += 0.1
        
        quality += 0.2 * min(structure_score, 1.0)
        
        return max(0.0, min(1.0, quality))
    
    def run_grid_search(self, alpha_p_values: List[float], alpha_n_values: List[float]) -> Dict[str, Any]:
        """グリッドサーチ実行"""
        print(f"\n🔍 Starting Grid Search...")
        print(f"   α_p values: {alpha_p_values}")
        print(f"   α_n values: {alpha_n_values}")
        print(f"   Total combinations: {len(alpha_p_values) * len(alpha_n_values)}")
        
        # テストデータ読み込み
        test_samples = self.load_test_data(max_samples=6)
        
        grid_start = time.time()
        
        # 全組み合わせを評価
        for i, (alpha_p, alpha_n) in enumerate(itertools.product(alpha_p_values, alpha_n_values)):
            print(f"\n📋 Combination {i+1}/{len(alpha_p_values) * len(alpha_n_values)}")
            
            evaluation = self.evaluate_parameter_combination(alpha_p, alpha_n, test_samples)
            self.results.append(evaluation)
            
            # 最良スコア更新
            if evaluation['overall_score'] > self.best_score:
                self.best_score = evaluation['overall_score']
                self.best_params = {
                    'alpha_p': alpha_p,
                    'alpha_n': alpha_n
                }
                print(f"   🏆 NEW BEST: Score={self.best_score:.4f}")
        
        grid_time = time.time() - grid_start
        
        # 結果分析
        analysis = self._analyze_grid_results(grid_time)
        
        return analysis
    
    def _analyze_grid_results(self, total_time: float) -> Dict[str, Any]:
        """グリッドサーチ結果分析"""
        print(f"\n" + "="*80)
        print("📊 GRID SEARCH RESULTS ANALYSIS")
        print("="*80)
        
        if not self.results:
            print("❌ No results to analyze")
            return {'error': 'No results available'}
        
        # 結果ソート
        sorted_results = sorted(self.results, key=lambda x: x['overall_score'], reverse=True)
        
        print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS:")
        print("─" * 80)
        print(f"{'Rank':<4} {'α_p':<8} {'α_n':<8} {'Score':<8} {'Success%':<9} {'Quality':<8} {'AvgTime':<8}")
        print("─" * 80)
        
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1:<4} {result['alpha_p']:<8.2f} {result['alpha_n']:<8.3f} "
                  f"{result['overall_score']:<8.4f} {result['success_rate']*100:<9.1f} "
                  f"{result['avg_quality_score']:<8.3f} {result['avg_generation_time']:<8.2f}")
        
        # 統計分析
        successful_evals = [r for r in self.results if r['overall_score'] >= 0]
        success_rate = len(successful_evals) / len(self.results) * 100
        
        print(f"\n📈 GRID SEARCH STATISTICS:")
        print(f"   • Total Evaluations: {len(self.results)}")
        print(f"   • Successful Evaluations: {len(successful_evals)} ({success_rate:.1f}%)")
        print(f"   • Best Overall Score: {self.best_score:.4f}")
        print(f"   • Total Search Time: {total_time:.1f}s ({total_time/60:.1f}min)")
        
        if self.best_params:
            print(f"\n🎯 RECOMMENDED PARAMETERS:")
            print(f"   • α_p (alpha_personal): {self.best_params['alpha_p']}")
            print(f"   • α_n (alpha_neutral): {self.best_params['alpha_n']}")
            
            # 最良結果詳細
            best_result = sorted_results[0]
            print(f"\n📋 BEST CONFIGURATION DETAILS:")
            print(f"   • Success Rate: {best_result['success_rate']*100:.1f}%")
            print(f"   • Average Quality Score: {best_result['avg_quality_score']:.3f}")
            print(f"   • Average Generation Time: {best_result['avg_generation_time']:.2f}s")
            print(f"   • Overall Performance Score: {best_result['overall_score']:.4f}")
        
        # パフォーマンス分析
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        
        # α_p の影響分析
        alpha_p_groups = {}
        for result in successful_evals:
            alpha_p = result['alpha_p']
            if alpha_p not in alpha_p_groups:
                alpha_p_groups[alpha_p] = []
            alpha_p_groups[alpha_p].append(result['overall_score'])
        
        if alpha_p_groups:
            best_alpha_p = max(alpha_p_groups.keys(), key=lambda x: sum(alpha_p_groups[x])/len(alpha_p_groups[x]))
            print(f"   • Best α_p value: {best_alpha_p} (avg score: {sum(alpha_p_groups[best_alpha_p])/len(alpha_p_groups[best_alpha_p]):.3f})")
        
        # α_n の影響分析
        alpha_n_groups = {}
        for result in successful_evals:
            alpha_n = result['alpha_n']
            if alpha_n not in alpha_n_groups:
                alpha_n_groups[alpha_n] = []
            alpha_n_groups[alpha_n].append(result['overall_score'])
        
        if alpha_n_groups:
            best_alpha_n = max(alpha_n_groups.keys(), key=lambda x: sum(alpha_n_groups[x])/len(alpha_n_groups[x]))
            print(f"   • Best α_n value: {best_alpha_n} (avg score: {sum(alpha_n_groups[best_alpha_n])/len(alpha_n_groups[best_alpha_n]):.3f})")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': len(self.results),
            'successful_evaluations': len(successful_evals),
            'total_time': total_time,
            'top_results': sorted_results[:5],
            'all_results': self.results
        }
    
    def save_results(self, output_file: str):
        """結果をJSONファイルに保存"""
        try:
            analysis = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'timestamp': time.time(),
                'all_results': self.results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='CFS-Chameleon Hyperparameter Tuning')
    
    # CLI引数定義
    parser.add_argument('--alpha_p', type=float, nargs='+', 
                       default=[0.01, 0.1, 0.2, 0.4], 
                       help='Alpha personal values to test')
    parser.add_argument('--alpha_n', type=float, nargs='+', 
                       default=[-0.02, -0.05, -0.1], 
                       help='Alpha neutral values to test')
    parser.add_argument('--grid-search', action='store_true', 
                       help='Enable grid search mode')
    parser.add_argument('--output', type=str, 
                       default='hyperparameter_results.json', 
                       help='Output JSON file')
    parser.add_argument('--config', type=str, 
                       default='cfs_config.yaml', 
                       help='CFS-Chameleon config file')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Hyperparameter Tuning...")
    print(f"   Alpha Personal Values: {args.alpha_p}")
    print(f"   Alpha Neutral Values: {args.alpha_n}")
    print(f"   Grid Search: {'✅ Enabled' if args.grid_search else '❌ Disabled'}")
    
    # GPU確認
    import torch
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   GPU: Not available (using CPU)")
    
    if args.grid_search:
        # グリッドサーチモード
        tuner = HyperparameterTuner(args.config)
        
        try:
            analysis = tuner.run_grid_search(args.alpha_p, args.alpha_n)
            
            # 結果保存
            tuner.save_results(args.output)
            
            print(f"\n" + "="*80)
            print("✨ HYPERPARAMETER TUNING COMPLETED!")
            print("="*80)
            
            if tuner.best_params:
                print(f"🏆 Recommended Configuration:")
                print(f"   --alpha_p {tuner.best_params['alpha_p']}")
                print(f"   --alpha_n {tuner.best_params['alpha_n']}")
                print(f"🎯 Expected Performance Score: {tuner.best_score:.4f}")
            else:
                print("❌ No optimal parameters found")
            
            return True
            
        except Exception as e:
            print(f"❌ Grid search failed: {e}")
            return False
    
    else:
        # 単一パラメータテスト
        print("\n⚠️ Grid search not enabled. Use --grid-search for full optimization.")
        print("   Example: python hyperparameter_tuner.py --grid-search")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)