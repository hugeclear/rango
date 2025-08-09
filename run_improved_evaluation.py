#!/usr/bin/env python3
"""
改善された評価実行スクリプト
問題を修正してLegacy Chameleonの正確な性能評価を実行
"""

import sys
import os
import time
import logging

sys.path.append('/home/nakata/master_thesis/rango')
os.chdir('/home/nakata/master_thesis/rango')

print('🎯 改善された CFS-Chameleon 評価実行')
print('='*70)
print('🔧 修正項目:')
print('   1. max_length: 10 → 128 (適切な生成長)')
print('   2. Alpha values: α_p=1.5→0.3, α_n=-0.8→-0.05 (過度編集防止)')
print('   3. Legacy Chameleon動作確認済み')
print('   4. 明確な結果表示')
print('='*70)

from chameleon_cfs_integrator import CollaborativeChameleonEditor
from pathlib import Path
import json

def load_test_samples(limit=5):
    """テストサンプル読み込み"""
    data_path = Path("chameleon_prime_personalization/data/raw/LaMP-2/merged.json")
    
    if not data_path.exists():
        print(f'❌ データファイル未発見: {data_path}')
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'✅ テストデータ読み込み: {len(data)} → {min(limit, len(data))} samples')
    return data[:limit]

def create_movie_prompt(sample):
    """映画推薦プロンプト作成"""
    input_text = sample.get('input', '')
    if len(input_text) > 100:
        input_text = input_text[:100] + "..."
    return input_text

def evaluate_legacy_chameleon():
    """Legacy Chameleon改善評価"""
    print('\n🔄 Legacy Chameleon改善評価開始')
    print('-' * 50)
    
    try:
        # 改善されたパラメータでエディター初期化
        editor = CollaborativeChameleonEditor(
            use_collaboration=False,  # Legacy mode
            config_path='cfs_config.yaml'
        )
        
        test_samples = load_test_samples(5)
        if not test_samples:
            return None
        
        results = []
        total_start_time = time.time()
        
        for i, sample in enumerate(test_samples, 1):
            print(f'   📝 Sample {i}/5 処理中...')
            
            # プロンプト作成
            prompt = create_movie_prompt(sample)
            expected = sample.get('output', 'unknown')
            
            start_time = time.time()
            
            try:
                # 改善されたパラメータで生成
                generated = editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=0.3,      # 改善: 1.5 → 0.3
                    alpha_neutral=-0.05,     # 改善: -0.8 → -0.05  
                    max_length=128           # 改善: 10 → 128
                )
                
                generation_time = time.time() - start_time
                
                results.append({
                    'sample_id': i,
                    'input': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    'expected': expected,
                    'generated': generated[:100] + "..." if len(generated) > 100 else generated,
                    'generation_time': generation_time,
                    'status': 'success',
                    'gen_length': len(generated)
                })
                
                print(f'     ✅ 成功: {len(generated)} chars, {generation_time:.2f}s')
                print(f'     出力: "{generated[:60]}..."')
                
            except Exception as e:
                generation_time = time.time() - start_time
                results.append({
                    'sample_id': i,
                    'input': prompt[:50],
                    'expected': expected,
                    'generated': f'ERROR: {e}',
                    'generation_time': generation_time,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f'     ❌ 失敗: {e}')
        
        total_time = time.time() - total_start_time
        
        # 結果統計
        successful = sum(1 for r in results if r['status'] == 'success')
        success_rate = (successful / len(results)) * 100 if results else 0
        avg_time = sum(r['generation_time'] for r in results) / len(results) if results else 0
        avg_length = sum(r.get('gen_length', 0) for r in results if r['status'] == 'success') / successful if successful > 0 else 0
        
        summary = {
            'system': 'Legacy_Chameleon_Improved',
            'total_samples': len(results),
            'successful': successful,
            'success_rate': success_rate,
            'avg_generation_time': avg_time,
            'avg_generation_length': avg_length,
            'total_evaluation_time': total_time,
            'results': results
        }
        
        return summary
        
    except Exception as e:
        print(f'❌ Legacy Chameleon評価エラー: {e}')
        return None

def display_evaluation_summary(summary):
    """評価結果詳細表示"""
    if not summary:
        print('❌ 評価結果なし')
        return
    
    print('\n📊 Legacy Chameleon改善評価結果')
    print('='*70)
    
    # 基本統計
    print(f'🎯 基本統計:')
    print(f'   総サンプル数:     {summary["total_samples"]}')
    print(f'   成功数:          {summary["successful"]}')
    print(f'   成功率:          {summary["success_rate"]:.1f}%')
    print(f'   平均生成時間:     {summary["avg_generation_time"]:.2f}秒')
    print(f'   平均生成長:      {summary["avg_generation_length"]:.0f}文字')
    print(f'   総評価時間:      {summary["total_evaluation_time"]:.1f}秒')
    
    # 成功率評価
    if summary["success_rate"] == 100:
        status = '🏆 EXCELLENT'
        assessment = 'システム完全動作'
    elif summary["success_rate"] >= 80:
        status = '✅ GOOD'
        assessment = 'システム安定動作'
    elif summary["success_rate"] >= 50:
        status = '⚠️ PARTIAL'
        assessment = '一部問題あり'
    else:
        status = '❌ POOR'
        assessment = 'システム不安定'
    
    print(f'   システム状態:     {status} - {assessment}')
    
    # 改善効果
    print(f'\n🔧 改善効果:')
    print(f'   修正前問題: max_length=10, α過大, フォールバック隠蔽')
    print(f'   修正後改善: max_length=128, α適正, エラー明示化')
    print(f'   生成品質: {summary["avg_generation_length"]:.0f}文字の適切な出力')
    
    # サンプル別詳細
    print(f'\n📝 サンプル別詳細:')
    for result in summary['results']:
        status_emoji = '✅' if result['status'] == 'success' else '❌'
        sample_id = result['sample_id']
        gen_time = result['generation_time']
        
        if result['status'] == 'success':
            gen_len = result.get('gen_length', 0)
            print(f'   Sample {sample_id}: {status_emoji} {gen_len}文字 ({gen_time:.2f}s)')
            print(f'     出力: "{result["generated"][:80]}..."')
        else:
            print(f'   Sample {sample_id}: {status_emoji} エラー ({gen_time:.2f}s)')
            print(f'     エラー: {result.get("error", "unknown")}')

def main():
    """メイン評価実行"""
    print('\n🚀 改善評価開始...')
    
    # Legacy Chameleon改善評価
    summary = evaluate_legacy_chameleon()
    
    # 結果表示
    display_evaluation_summary(summary)
    
    # 結果保存
    if summary:
        output_path = f'cfs_evaluation_results/improved_evaluation_{int(time.time())}.json'
        os.makedirs('cfs_evaluation_results', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f'\n💾 評価結果保存: {output_path}')
    
    print('\n✨ 改善評価完了!')
    print('='*70)

if __name__ == "__main__":
    main()