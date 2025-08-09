#!/usr/bin/env python3
"""
CFS-Chameleon クイックハイパーパラメータチューニングデモ
短時間でグリッドサーチ機能を検証
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

sys.path.append('/home/nakata/master_thesis/rango')
os.chdir('/home/nakata/master_thesis/rango')

from chameleon_cfs_integrator import CollaborativeChameleonEditor

def quick_hyperparameter_demo():
    """クイックハイパーパラメータデモ"""
    print('🎯 CFS-CHAMELEON QUICK HYPERPARAMETER DEMO')
    print('='*70)
    print('🔧 デモ設計: 2サンプル × 3組み合わせ = 6テスト (高速)')
    print('='*70)
    
    # テストパラメータ組み合わせ
    param_combinations = [
        {'alpha_p': 0.1, 'alpha_n': -0.02, 'name': 'Conservative'},
        {'alpha_p': 0.2, 'alpha_n': -0.05, 'name': 'Moderate'},
        {'alpha_p': 0.3, 'alpha_n': -0.1, 'name': 'Aggressive'}
    ]
    
    # テストデータ（2サンプルのみ）
    test_samples = [
        {
            'input': 'What is your favorite movie genre?',
            'output': 'Action movies',
            'user': 'demo_user_1'
        },
        {
            'input': 'Recommend a good restaurant.',
            'output': 'Italian cuisine',
            'user': 'demo_user_2'
        }
    ]
    
    results = []
    overall_start = time.time()
    
    print(f"\n🚀 Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"\n📋 Configuration {i+1}/{len(param_combinations)}: {params['name']}")
        print(f"   α_p = {params['alpha_p']}, α_n = {params['alpha_n']}")
        
        config_start = time.time()
        
        try:
            # CFS-Chameleonエディター作成
            editor = CollaborativeChameleonEditor(
                use_collaboration=True,
                config_path='cfs_config.yaml'
            )
            
            sample_results = []
            
            for j, sample in enumerate(test_samples):
                sample_start = time.time()
                
                try:
                    # 生成実行
                    generated = editor.generate_with_collaborative_chameleon(
                        prompt=sample['input'],
                        user_id=sample['user'],
                        alpha_personal=params['alpha_p'],
                        alpha_neutral=params['alpha_n'],
                        max_length=80  # 短い生成
                    )
                    
                    generation_time = time.time() - sample_start
                    quality_score = compute_quick_quality(generated, sample['input'])
                    
                    sample_results.append({
                        'sample_id': j + 1,
                        'user_id': sample['user'],
                        'generated': generated,
                        'generation_time': generation_time,
                        'quality_score': quality_score,
                        'status': 'success'
                    })
                    
                    print(f"     Sample {j+1}: Quality={quality_score:.3f}, Time={generation_time:.2f}s")
                    
                except Exception as e:
                    sample_results.append({
                        'sample_id': j + 1,
                        'user_id': sample['user'],
                        'generated': f'ERROR: {e}',
                        'generation_time': time.time() - sample_start,
                        'quality_score': 0.0,
                        'status': 'failed'
                    })
                    print(f"     Sample {j+1}: FAILED - {e}")
            
            config_time = time.time() - config_start
            
            # 統計計算
            successful = [r for r in sample_results if r['status'] == 'success']
            success_rate = len(successful) / len(sample_results) if sample_results else 0
            
            if successful:
                avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
                avg_time = sum(r['generation_time'] for r in successful) / len(successful)
                
                # 総合スコア (品質75% + 成功率20% - 時間ペナルティ5%)
                overall_score = (0.75 * avg_quality + 0.2 * success_rate - 0.05 * (avg_time / 5.0))
            else:
                avg_quality = 0.0
                avg_time = 0.0
                overall_score = 0.0
            
            config_result = {
                'name': params['name'],
                'alpha_p': params['alpha_p'],
                'alpha_n': params['alpha_n'],
                'success_rate': success_rate,
                'avg_quality_score': avg_quality,
                'avg_generation_time': avg_time,
                'overall_score': overall_score,
                'config_time': config_time,
                'sample_results': sample_results
            }
            
            results.append(config_result)
            
            print(f"   📊 Success: {success_rate*100:.1f}%, Quality: {avg_quality:.3f}, Score: {overall_score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            results.append({
                'name': params['name'],
                'alpha_p': params['alpha_p'],
                'alpha_n': params['alpha_n'],
                'success_rate': 0.0,
                'avg_quality_score': 0.0,
                'avg_generation_time': 0.0,
                'overall_score': -1.0,
                'config_time': time.time() - config_start,
                'error': str(e),
                'sample_results': []
            })
    
    total_time = time.time() - overall_start
    
    # 結果分析
    print(f"\n" + "="*70)
    print("📊 HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    
    # 結果ソート
    valid_results = [r for r in results if r['overall_score'] >= 0]
    sorted_results = sorted(valid_results, key=lambda x: x['overall_score'], reverse=True)
    
    if sorted_results:
        print(f"\n🏆 RANKING:")
        print("─" * 70)
        print(f"{'Rank':<4} {'Config':<12} {'α_p':<6} {'α_n':<7} {'Score':<7} {'Success%':<9}")
        print("─" * 70)
        
        for i, result in enumerate(sorted_results):
            print(f"{i+1:<4} {result['name']:<12} {result['alpha_p']:<6.1f} "
                  f"{result['alpha_n']:<7.2f} {result['overall_score']:<7.4f} "
                  f"{result['success_rate']*100:<9.1f}")
        
        # 最良設定
        best = sorted_results[0]
        print(f"\n🎯 RECOMMENDED HYPERPARAMETERS:")
        print(f"   • Configuration: {best['name']}")
        print(f"   • α_p (alpha_personal): {best['alpha_p']}")
        print(f"   • α_n (alpha_neutral): {best['alpha_n']}")
        print(f"   • Expected Performance Score: {best['overall_score']:.4f}")
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   • Best Success Rate: {best['success_rate']*100:.1f}%")
        print(f"   • Best Quality Score: {best['avg_quality_score']:.3f}")
        print(f"   • Best Avg Time: {best['avg_generation_time']:.2f}s")
        
        # パラメータの影響分析
        alpha_p_impact = {}
        for result in valid_results:
            alpha_p = result['alpha_p']
            if alpha_p not in alpha_p_impact:
                alpha_p_impact[alpha_p] = []
            alpha_p_impact[alpha_p].append(result['overall_score'])
        
        print(f"\n💡 PARAMETER INSIGHTS:")
        for alpha_p, scores in alpha_p_impact.items():
            avg_score = sum(scores) / len(scores)
            print(f"   • α_p = {alpha_p}: Avg Score = {avg_score:.4f}")
        
    else:
        print("❌ No successful configurations found")
    
    print(f"\n⏱️  Total Demo Time: {total_time:.1f}s")
    
    # 結果をJSONで出力
    output_data = {
        'demo_type': 'quick_hyperparameter_tuning',
        'timestamp': time.time(),
        'total_time': total_time,
        'results': results,
        'best_params': sorted_results[0] if sorted_results else None
    }
    
    with open('quick_tuning_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: quick_tuning_results.json")
    
    success = len(valid_results) > 0 and max(r['overall_score'] for r in valid_results) > 0.5
    
    if success:
        print(f"\n🎉 HYPERPARAMETER TUNING DEMO: SUCCESSFUL!")
        print(f"✅ Grid search functionality validated")
    else:
        print(f"\n⚠️ HYPERPARAMETER TUNING DEMO: NEEDS IMPROVEMENT")
    
    return success

def compute_quick_quality(generated: str, input_text: str) -> float:
    """簡易品質スコア計算"""
    if not generated or len(generated.strip()) == 0:
        return 0.0
    
    quality = 0.0
    
    # 長さチェック
    if 10 <= len(generated) <= 200:
        quality += 0.4
    elif len(generated) > 5:
        quality += 0.2
    
    # 反復チェック
    words = generated.split()
    if len(words) > 2:
        unique_ratio = len(set(words)) / len(words)
        quality += 0.3 * unique_ratio
    
    # 基本文構造
    if any(char.isalpha() for char in generated):
        quality += 0.2
    if not any(char in r'\/*[]{}()' for char in generated[:30]):
        quality += 0.1
    
    return min(1.0, quality)

def main():
    """メイン実行"""
    import torch
    print(f"🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    success = quick_hyperparameter_demo()
    
    print("\n✨ CFS-Chameleon Hyperparameter Demo Completed!")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)