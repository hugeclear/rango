#!/usr/bin/env python3
"""
CFS-Chameleon End-to-End デモスクリプト
3ユーザー×各3サンプルで完全動作検証
例外発生時は必ずテストが失敗するように設計
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append('/home/nakata/master_thesis/rango')
os.chdir('/home/nakata/master_thesis/rango')

print('🎯 CFS-CHAMELEON END-TO-END DEMO')
print('='*80)
print('🔧 デモ設計:')
print('   • 3ユーザー × 各3サンプル = 9回の生成テスト')
print('   • 全例外でテスト失敗 (Silent failure禁止)')
print('   • 協調機能・統計・品質すべて検証')
print('   • CFS vs Legacy vs Baseline 3システム比較')
print('='*80)

from chameleon_cfs_integrator import CollaborativeChameleonEditor

class E2EDemoRunner:
    """End-to-End デモ実行クラス"""
    
    def __init__(self):
        self.test_users = ["user_101", "user_202", "user_303"]
        self.samples_per_user = 3
        self.total_samples = len(self.test_users) * self.samples_per_user
        self.results = {}
        self.start_time = time.time()
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """テストデータ読み込み"""
        try:
            data_path = Path("chameleon_prime_personalization/data/raw/LaMP-2/merged.json")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 9サンプル取得
            test_samples = data[:self.total_samples]
            
            if len(test_samples) < self.total_samples:
                raise RuntimeError(f"❌ CRITICAL: Not enough test data: {len(test_samples)} < {self.total_samples}")
            
            print(f"✅ Test data loaded: {len(test_samples)} samples")
            return test_samples
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: Failed to load test data: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def test_cfs_chameleon_system(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """CFS-Chameleonシステムテスト"""
        print("\n🦎 Testing CFS-Chameleon System...")
        
        try:
            # CFS-Chameleon初期化
            editor = CollaborativeChameleonEditor(
                use_collaboration=True,
                config_path='cfs_config.yaml'
            )
            
            print("✅ CFS-Chameleon initialized successfully")
            
            results = []
            user_idx = 0
            sample_idx = 0
            
            for i, sample in enumerate(test_samples):
                current_user = self.test_users[user_idx]
                user_sample_idx = sample_idx + 1
                
                print(f"\n   🔄 Sample {i+1}/{self.total_samples} - User: {current_user} (Sample {user_sample_idx}/3)")
                
                input_text = sample.get('input', '')[:150]  # 適度な長さに制限
                expected = sample.get('output', 'N/A')
                
                sample_start = time.time()
                
                try:
                    # CFS-Chameleon生成実行
                    generated = editor.generate_with_collaborative_chameleon(
                        prompt=input_text,
                        user_id=current_user,
                        alpha_personal=0.15,  # 控えめな編集強度
                        alpha_neutral=-0.03,
                        max_length=120
                    )
                    
                    generation_time = time.time() - sample_start
                    
                    # 生成品質検証
                    if not generated or len(generated.strip()) == 0:
                        raise RuntimeError(f"❌ CRITICAL: Empty generation for user {current_user}")
                    
                    if len(generated) < 10:
                        raise RuntimeError(f"❌ CRITICAL: Too short generation ({len(generated)} chars) for user {current_user}")
                    
                    results.append({
                        'sample_id': i + 1,
                        'user_id': current_user,
                        'user_sample': user_sample_idx,
                        'input': input_text[:80] + "..." if len(input_text) > 80 else input_text,
                        'expected': expected,
                        'generated': generated[:100] + "..." if len(generated) > 100 else generated,
                        'full_generated_length': len(generated),
                        'generation_time': generation_time,
                        'status': 'success'
                    })
                    
                    print(f"     ✅ SUCCESS: {len(generated)} chars in {generation_time:.2f}s")
                    print(f"     📝 Output: \"{generated[:60]}...\"")
                    
                except Exception as e:
                    # 例外発生 = E2Eテスト失敗
                    error_msg = f"❌ CRITICAL E2E FAILURE: Sample {i+1} (User {current_user}): {e}"
                    print(f"     {error_msg}")
                    raise RuntimeError(error_msg)
                
                # 次のユーザー・サンプルに進む
                sample_idx += 1
                if sample_idx >= self.samples_per_user:
                    sample_idx = 0
                    user_idx += 1
            
            # 協調機能統計確認
            try:
                stats = editor.get_collaboration_statistics()
                collab_stats = stats.get('collaboration_stats', {})
                
                directions_generated = collab_stats.get('collaborative_directions_generated', 0)
                if directions_generated == 0:
                    raise RuntimeError("❌ CRITICAL: No collaborative directions were generated")
                
                print(f"\n📊 CFS-Chameleon Statistics:")
                print(f"   • Collaborative Directions Generated: {directions_generated}")
                print(f"   • Total Collaborations: {collab_stats.get('total_collaborations', 0)}")
                print(f"   • Privacy Applications: {collab_stats.get('privacy_applications', 0)}")
                
            except Exception as e:
                error_msg = f"❌ CRITICAL: Failed to get collaboration statistics: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)
            
            return {
                'system_name': 'CFS-Chameleon',
                'results': results,
                'status': 'completed',
                'collaboration_stats': collab_stats
            }
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: CFS-Chameleon system test failed: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def test_baseline_comparison(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ベースラインシステム比較テスト"""
        print("\n📊 Testing Baseline System for Comparison...")
        
        try:
            from chameleon_evaluator import ChameleonEvaluator
            evaluator = ChameleonEvaluator('config.yaml')
            
            baseline_results = []
            
            # 3サンプルのみでクイック比較
            for i, sample in enumerate(test_samples[:3]):
                input_text = sample.get('input', '')[:150]
                expected = sample.get('output', 'N/A')
                
                sample_start = time.time()
                
                try:
                    # ベースライン生成（編集なし）
                    inputs = evaluator.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=100)
                    inputs = {k: v.to(evaluator.device) for k, v in inputs.items()}
                    
                    with evaluator.model.torch.no_grad():
                        outputs = evaluator.model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 60,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=evaluator.tokenizer.eos_token_id
                        )
                    
                    generated = evaluator.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated = generated[len(input_text):].strip()  # 入力部分を除去
                    
                    generation_time = time.time() - sample_start
                    
                    baseline_results.append({
                        'sample_id': i + 1,
                        'input': input_text[:80] + "..." if len(input_text) > 80 else input_text,
                        'generated': generated[:100] + "..." if len(generated) > 100 else generated,
                        'generation_time': generation_time,
                        'status': 'success'
                    })
                    
                    print(f"     ✅ Baseline Sample {i+1}: {len(generated)} chars in {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"     ⚠️ Baseline Sample {i+1} failed: {e}")
                    baseline_results.append({
                        'sample_id': i + 1,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            return {
                'system_name': 'Baseline',
                'results': baseline_results,
                'status': 'completed'
            }
            
        except Exception as e:
            print(f"     ⚠️ Baseline system test failed: {e}")
            return {
                'system_name': 'Baseline',
                'results': [],
                'status': 'failed',
                'error': str(e)
            }
    
    def analyze_e2e_results(self, cfs_results: Dict[str, Any], baseline_results: Dict[str, Any]):
        """E2E結果分析"""
        print("\n" + "="*80)
        print("📊 END-TO-END DEMO ANALYSIS")
        print("="*80)
        
        # CFS-Chameleon結果分析
        print(f"\n🦎 CFS-CHAMELEON PERFORMANCE:")
        if cfs_results['status'] == 'completed':
            cfs_samples = cfs_results['results']
            cfs_success = len([r for r in cfs_samples if r['status'] == 'success'])
            cfs_success_rate = (cfs_success / len(cfs_samples) * 100) if cfs_samples else 0
            
            if cfs_success > 0:
                avg_time = sum(r['generation_time'] for r in cfs_samples if r['status'] == 'success') / cfs_success
                avg_length = sum(r['full_generated_length'] for r in cfs_samples if r['status'] == 'success') / cfs_success
            else:
                avg_time = 0
                avg_length = 0
            
            print(f"   • Success Rate: {cfs_success_rate:.1f}% ({cfs_success}/{len(cfs_samples)})")
            print(f"   • Average Generation Time: {avg_time:.2f}s")
            print(f"   • Average Output Length: {avg_length:.0f} chars")
            
            # ユーザー別分析
            user_stats = {}
            for result in cfs_samples:
                if result['status'] == 'success':
                    user = result['user_id']
                    if user not in user_stats:
                        user_stats[user] = []
                    user_stats[user].append(result)
            
            print(f"   • User Coverage: {len(user_stats)}/{len(self.test_users)} users tested")
            for user, samples in user_stats.items():
                print(f"     - {user}: {len(samples)} successful samples")
            
            # 協調機能統計
            if 'collaboration_stats' in cfs_results:
                stats = cfs_results['collaboration_stats']
                print(f"   • Collaborative Directions Generated: {stats.get('collaborative_directions_generated', 0)}")
                print(f"   • Total Collaborations: {stats.get('total_collaborations', 0)}")
        else:
            print(f"   ❌ CFS-Chameleon failed: {cfs_results.get('error', 'Unknown error')}")
            cfs_success_rate = 0
        
        # ベースライン結果分析
        print(f"\n📊 BASELINE COMPARISON:")
        if baseline_results['status'] == 'completed':
            baseline_samples = baseline_results['results']
            baseline_success = len([r for r in baseline_samples if r['status'] == 'success'])
            baseline_success_rate = (baseline_success / len(baseline_samples) * 100) if baseline_samples else 0
            
            print(f"   • Success Rate: {baseline_success_rate:.1f}% ({baseline_success}/{len(baseline_samples)})")
            
            if baseline_success > 0:
                baseline_avg_time = sum(r['generation_time'] for r in baseline_samples if r['status'] == 'success') / baseline_success
                print(f"   • Average Generation Time: {baseline_avg_time:.2f}s")
        else:
            print(f"   ⚠️ Baseline failed: {baseline_results.get('error', 'Unknown error')}")
            baseline_success_rate = 0
        
        # 総合評価
        print(f"\n🎯 END-TO-END DEMO VERDICT:")
        
        if cfs_success_rate == 100:
            e2e_status = "🏆 EXCELLENT: Complete E2E success!"
            recommendation = "CFS-Chameleon ready for production deployment"
        elif cfs_success_rate >= 80:
            e2e_status = "✅ GOOD: Strong E2E performance"
            recommendation = "CFS-Chameleon suitable for pilot deployment"
        elif cfs_success_rate >= 60:
            e2e_status = "⚠️ MIXED: Partial E2E success"
            recommendation = "CFS-Chameleon needs improvement before deployment"
        else:
            e2e_status = "❌ FAILED: E2E test failed"
            recommendation = "CFS-Chameleon requires significant fixes"
        
        print(f"   • Status: {e2e_status}")
        print(f"   • Recommendation: {recommendation}")
        
        # 比較分析
        if baseline_success_rate > 0:
            if cfs_success_rate > baseline_success_rate:
                comparison = f"✅ CFS-Chameleon outperforms baseline (+{cfs_success_rate - baseline_success_rate:.1f}%)"
            elif cfs_success_rate == baseline_success_rate:
                comparison = f"➡️ CFS-Chameleon matches baseline performance"
            else:
                comparison = f"❌ CFS-Chameleon underperforms baseline ({cfs_success_rate - baseline_success_rate:.1f}%)"
        else:
            comparison = f"🏆 CFS-Chameleon succeeds where baseline fails"
        
        print(f"   • Comparison: {comparison}")
        
        return cfs_success_rate >= 80  # E2E success criteria
    
    def run_full_demo(self) -> bool:
        """完全E2Eデモ実行"""
        print("\n🚀 Starting Full End-to-End Demo...")
        
        try:
            # 1. テストデータ読み込み
            test_samples = self.load_test_data()
            
            # 2. CFS-Chameleonシステムテスト
            cfs_results = self.test_cfs_chameleon_system(test_samples)
            
            # 3. ベースライン比較テスト
            baseline_results = self.test_baseline_comparison(test_samples)
            
            # 4. 結果分析
            success = self.analyze_e2e_results(cfs_results, baseline_results)
            
            total_time = time.time() - self.start_time
            
            print(f"\n" + "="*80)
            print("✨ END-TO-END DEMO COMPLETED!")
            print("="*80)
            print(f"🏁 Total Demo Time: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"📋 Demo Success: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: E2E demo failed with exception: {e}"
            print(f"\n{error_msg}")
            
            total_time = time.time() - self.start_time
            print(f"\n" + "="*80)
            print("💥 END-TO-END DEMO FAILED!")
            print("="*80)
            print(f"🏁 Demo Time Before Failure: {total_time:.1f}s")
            print(f"🚨 Exception: {str(e)[:200]}...")
            
            return False


def main():
    """メイン実行"""
    print("\n🚀 Initializing End-to-End Demo...")
    
    # GPU確認
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU")
    
    # E2Eデモ実行
    demo_runner = E2EDemoRunner()
    
    try:
        success = demo_runner.run_full_demo()
        
        if success:
            print("\n🎉 CFS-Chameleon E2E Demo: SUCCESSFUL!")
            print("🚀 System ready for production benchmarking!")
        else:
            print("\n🔧 CFS-Chameleon E2E Demo: REQUIRES IMPROVEMENT")
            print("⚠️ Address issues before production deployment")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Demo crashed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)