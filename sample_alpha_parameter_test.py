#!/usr/bin/env python3
"""
Simple Alpha Parameter Testing
==============================

既存のrun_evaluation.pyを使用してα値を段階的にテストする簡単なスクリプト
"""

import subprocess
import time
import json
import re
from pathlib import Path
import shutil

class SimpleAlphaTester:
    """シンプルなα値テスター"""
    
    def __init__(self):
        self.results = []
        self.backup_created = False
    
    def create_backup(self):
        """バックアップを作成"""
        if not self.backup_created and Path("chameleon_evaluator.py").exists():
            shutil.copy("chameleon_evaluator.py", "chameleon_evaluator.py.backup")
            self.backup_created = True
            print("✅ Backup created: chameleon_evaluator.py.backup")
    
    def modify_alpha_values(self, alpha_personal: float, alpha_neutral: float):
        """α値を一時的に変更"""
        
        self.create_backup()
        
        # ファイルを読み込み
        with open("chameleon_evaluator.py", "r") as f:
            content = f.read()
        
        # Starting Chameleon evaluation の行を探して値を変更
        content = re.sub(
            r'Starting Chameleon evaluation \(α_p=[0-9.-]+, α_n=[0-9.-]+\)',
            f'Starting Chameleon evaluation (α_p={alpha_personal}, α_n={alpha_neutral})',
            content
        )
        
        # register_editing_hooks の呼び出し部分を変更
        content = re.sub(
            r'self\.register_editing_hooks\([^,]+,\s*[0-9.-]+,\s*[0-9.-]+\)',
            f'self.register_editing_hooks(target_layers, {alpha_personal}, {alpha_neutral})',
            content
        )
        
        # generate_with_chameleon のデフォルト値を変更
        content = re.sub(
            r'alpha_personal: float = [0-9.-]+',
            f'alpha_personal: float = {alpha_personal}',
            content
        )
        content = re.sub(
            r'alpha_neutral: float = [0-9.-]+',
            f'alpha_neutral: float = {alpha_neutral}',
            content
        )
        
        # ファイルに書き戻し
        with open("chameleon_evaluator.py", "w") as f:
            f.write(content)
    
    def restore_backup(self):
        """バックアップから復元"""
        if Path("chameleon_evaluator.py.backup").exists():
            shutil.copy("chameleon_evaluator.py.backup", "chameleon_evaluator.py")
            print("✅ Restored from backup")
    
    def run_single_test(self, alpha_personal: float, alpha_neutral: float) -> dict:
        """単一のα値組み合わせをテスト"""
        
        print(f"\n🧪 Testing α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f}")
        
        try:
            # α値を変更
            self.modify_alpha_values(alpha_personal, alpha_neutral)
            
            # 評価実行
            start_time = time.time()
            result = subprocess.run(
                ["python", "run_evaluation.py", "--mode", "demo", "--skip-checks"],
                capture_output=True,
                text=True,
                timeout=180  # 3分でタイムアウト
            )
            end_time = time.time()
            
            if result.returncode == 0:
                # 結果を解析
                parsed_result = self.parse_output(result.stderr)
                if parsed_result:
                    parsed_result.update({
                        'alpha_personal': alpha_personal,
                        'alpha_neutral': alpha_neutral,
                        'execution_time': end_time - start_time,
                        'timestamp': time.time()
                    })
                    
                    improvement = parsed_result.get('improvement_rate', 0) * 100
                    print(f"   ✅ Success! Improvement: {improvement:+.1f}%")
                    return parsed_result
                else:
                    print("   ❌ Failed to parse results")
                    return None
            else:
                print(f"   ❌ Execution failed (code: {result.returncode})")
                return None
                
        except subprocess.TimeoutExpired:
            print("   ⏰ Timed out")
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
        finally:
            # バックアップから復元
            self.restore_backup()
    
    def parse_output(self, output: str) -> dict:
        """実行結果から数値を抽出"""
        
        try:
            lines = output.split('\n')
            
            baseline_accuracy = None
            chameleon_accuracy = None
            improvement_rate = None
            
            # ベースライン結果を探す
            for i, line in enumerate(lines):
                if "Baseline Performance:" in line:
                    # 次の数行でAccuracyを探す
                    for j in range(i+1, min(i+10, len(lines))):
                        if "Accuracy:" in lines[j]:
                            match = re.search(r'Accuracy:\s*([0-9.]+)', lines[j])
                            if match:
                                baseline_accuracy = float(match.group(1))
                                break
                    break
            
            # Chameleon結果を探す
            for i, line in enumerate(lines):
                if "Chameleon Performance:" in line:
                    # 次の数行でAccuracyを探す
                    for j in range(i+1, min(i+10, len(lines))):
                        if "Accuracy:" in lines[j]:
                            match = re.search(r'Accuracy:\s*([0-9.]+)', lines[j])
                            if match:
                                chameleon_accuracy = float(match.group(1))
                                break
                    break
            
            # 改善率を探す
            for line in lines:
                if "Improvement Rate:" in line:
                    match = re.search(r'Improvement Rate:\s*([+-]?[0-9.]+)%', line)
                    if match:
                        improvement_rate = float(match.group(1)) / 100.0
                        break
            
            if baseline_accuracy is not None and chameleon_accuracy is not None:
                # 改善率が見つからない場合は計算
                if improvement_rate is None and baseline_accuracy > 0:
                    improvement_rate = (chameleon_accuracy - baseline_accuracy) / baseline_accuracy
                
                return {
                    'baseline_accuracy': baseline_accuracy,
                    'chameleon_accuracy': chameleon_accuracy,
                    'improvement_rate': improvement_rate or 0,
                    'accuracy': chameleon_accuracy
                }
            
            return None
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def test_promising_ranges(self):
        """有望なα値範囲をテスト"""
        
        print("🦎 Simple Alpha Parameter Testing")
        print("=" * 50)
        
        # テスト対象のα値組み合わせ
        test_combinations = [
            # 基本テスト
            (0.1, -0.05),
            (0.5, -0.2),
            (1.0, -0.5),
            
            # 中程度
            (1.2, -0.6),
            (1.5, -0.8),
            (1.8, -1.0),
            
            # 強め
            (2.0, -1.2),
            (2.5, -1.5),
            (3.0, -2.0),
            
            # 非対称パターン
            (1.0, -1.5),
            (2.0, -0.5),
            (1.5, -2.0),
            
            # 細かい調整
            (1.4, -0.7),
            (1.6, -0.9),
            (1.7, -1.1),
        ]
        
        best_result = None
        best_improvement = -float('inf')
        
        print(f"Testing {len(test_combinations)} combinations...")
        
        for i, (alpha_p, alpha_n) in enumerate(test_combinations, 1):
            print(f"\n[{i}/{len(test_combinations)}]", end=" ")
            
            result = self.run_single_test(alpha_p, alpha_n)
            
            if result:
                self.results.append(result)
                
                improvement = result.get('improvement_rate', 0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_result = result
                    print(f"   🎯 New best: {improvement:+.1%}")
                
                # 25%達成チェック
                if improvement >= 0.25:
                    print(f"   🎉 Target achieved! {improvement:.1%}")
                    break
        
        # 結果まとめ
        self.show_results(best_result)
        self.save_results()
    
    def show_results(self, best_result):
        """結果を表示"""
        
        print("\n" + "=" * 50)
        print("📊 RESULTS SUMMARY")
        print("=" * 50)
        
        if best_result:
            print(f"🏆 Best Result:")
            print(f"   α_personal: {best_result['alpha_personal']:.2f}")
            print(f"   α_neutral:  {best_result['alpha_neutral']:.2f}")
            print(f"   Improvement: {best_result['improvement_rate']:+.1%}")
            print(f"   Baseline:    {best_result['baseline_accuracy']:.4f}")
            print(f"   Chameleon:   {best_result['chameleon_accuracy']:.4f}")
        else:
            print("❌ No successful results found")
        
        if self.results:
            print(f"\n📈 All Results ({len(self.results)} total):")
            sorted_results = sorted(self.results, key=lambda x: x.get('improvement_rate', 0), reverse=True)
            
            for i, result in enumerate(sorted_results[:5], 1):
                improvement = result['improvement_rate']
                print(f"   {i}. α_p={result['alpha_personal']:5.2f}, α_n={result['alpha_neutral']:6.2f} → {improvement:+6.1%}")
    
    def save_results(self):
        """結果をファイルに保存"""
        
        if self.results:
            results_file = "alpha_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'total_tests': len(self.results),
                    'results': self.results
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")

def main():
    """メイン実行"""
    tester = SimpleAlphaTester()
    tester.test_promising_ranges()

if __name__ == "__main__":
    main()
