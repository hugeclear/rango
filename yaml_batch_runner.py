#!/usr/bin/env python3
"""
YAML Batch Alpha Parameter Runner
=================================

alpha_batch_config.yamlに基づいてα値を一括テストする効率的なランナー
"""

import yaml
import subprocess
import time
import json
import re
from pathlib import Path
import shutil
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class YAMLBatchRunner:
    """YAML設定に基づくバッチα値テスター"""
    
    def __init__(self, config_file="alpha_batch_config.yaml", main_config="config.yaml"):
        self.config_file = config_file
        self.main_config = main_config
        self.batch_config = None
        self.results = []
        self.best_result = None
        self.best_improvement = -float('inf')
        
    def load_batch_config(self):
        """バッチ設定を読み込み"""
        try:
            with open(self.config_file, 'r') as f:
                self.batch_config = yaml.safe_load(f)
            print(f"✅ Loaded batch config: {self.config_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Config file not found: {self.config_file}")
            return False
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    
    def backup_main_config(self):
        """メイン設定のバックアップ"""
        if Path(self.main_config).exists():
            backup_file = f"{self.main_config}.batch_backup"
            shutil.copy(self.main_config, backup_file)
            print(f"✅ Main config backed up: {backup_file}")
    
    def restore_main_config(self):
        """メイン設定を復元"""
        backup_file = f"{self.main_config}.batch_backup"
        if Path(backup_file).exists():
            shutil.copy(backup_file, self.main_config)
    
    def update_main_config(self, alpha_personal: float, alpha_neutral: float):
        """メイン設定ファイルを更新"""
        try:
            # 既存設定を読み込み
            if Path(self.main_config).exists():
                with open(self.main_config, 'r') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # Chameleon設定を更新
            if 'chameleon' not in config:
                config['chameleon'] = {}
            
            config['chameleon']['alpha_personal'] = alpha_personal
            config['chameleon']['alpha_neutral'] = alpha_neutral
            
            # target_layersが設定されていない場合は追加
            if 'target_layers' not in config['chameleon']:
                config['chameleon']['target_layers'] = self.batch_config['chameleon']['target_layers']
            
            # ファイルに書き戻し
            with open(self.main_config, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"   ⚠️  Config update failed: {e}")
    
    def run_single_evaluation(self, alpha_personal: float, alpha_neutral: float, priority: str = "medium") -> Dict[str, Any]:
        """単一の評価を実行"""
        
        print(f"   🧪 α_p={alpha_personal:5.2f}, α_n={alpha_neutral:6.2f} [{priority:6s}]", end=" ")
        
        try:
            # メイン設定を更新
            self.update_main_config(alpha_personal, alpha_neutral)
            
            # 評価実行
            start_time = time.time()
            result = subprocess.run(
                ["python", "run_evaluation.py", "--mode", "demo", "--skip-checks"],
                capture_output=True,
                text=True,
                timeout=self.batch_config['evaluation']['timeout']
            )
            end_time = time.time()
            
            if result.returncode == 0:
                # 結果解析
                parsed_result = self.parse_evaluation_output(result.stdout + "\n" + result.stderr)
                if parsed_result:
                    parsed_result.update({
                        'alpha_personal': alpha_personal,
                        'alpha_neutral': alpha_neutral,
                        'priority': priority,
                        'execution_time': end_time - start_time,
                        'timestamp': time.time()
                    })
                    
                    improvement = parsed_result.get('improvement_rate', 0)
                    print(f"→ {improvement:+6.1%}", end="")
                    
                    # 改善度による評価
                    criteria = self.batch_config['success_criteria']
                    if improvement >= criteria['excellent_improvement']:
                        print(" 🎉")
                    elif improvement >= criteria['good_improvement']:
                        print(" ✨")
                    elif improvement >= criteria['minimum_improvement']:
                        print(" 📈")
                    else:
                        print(" 📊")
                    
                    return parsed_result
                else:
                    print("❌ Parse failed")
                    return None
            else:
                print("❌ Execution failed")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def parse_evaluation_output(self, output: str) -> Dict[str, Any]:
        """評価出力を解析"""
        
        # ベースライン精度
        baseline_match = re.search(r'📊 Baseline Performance:.*?Accuracy:\s*([0-9.]+)', output, re.DOTALL)
        baseline_accuracy = float(baseline_match.group(1)) if baseline_match else None
        
        # Chameleon精度
        chameleon_match = re.search(r'🦎 Chameleon Performance:.*?Accuracy:\s*([0-9.]+)', output, re.DOTALL)
        chameleon_accuracy = float(chameleon_match.group(1)) if chameleon_match else None
        
        # 改善率
        improvement_match = re.search(r'Improvement Rate:\s*([+-]?[0-9.]+)%', output)
        improvement_rate = float(improvement_match.group(1)) / 100.0 if improvement_match else None
        
        if baseline_accuracy is not None and chameleon_accuracy is not None:
            if improvement_rate is None and baseline_accuracy > 0:
                improvement_rate = (chameleon_accuracy - baseline_accuracy) / baseline_accuracy
            
            return {
                'baseline_accuracy': baseline_accuracy,
                'chameleon_accuracy': chameleon_accuracy,
                'improvement_rate': improvement_rate or 0,
                'accuracy': chameleon_accuracy
            }
        
        return None
    
    def run_batch_optimization(self):
        """バッチ最適化を実行"""
        
        if not self.load_batch_config():
            return False
        
        print("🦎 YAML Batch Alpha Parameter Optimization")
        print("=" * 70)
        
        experiment = self.batch_config['experiment']
        print(f"Experiment: {experiment['name']}")
        print(f"Target: {experiment['target_improvement']:.1%} improvement")
        print(f"Early stop: {'Yes' if experiment['early_stop'] else 'No'}")
        
        # バックアップ作成
        self.backup_main_config()
        
        try:
            # 全ての組み合わせを収集
            all_combinations = []
            
            # フェーズ順に実行
            for phase_name in self.batch_config['execution']['order']:
                if phase_name in self.batch_config['alpha_test_patterns']:
                    phase = self.batch_config['alpha_test_patterns'][phase_name]
                    print(f"\n📋 {phase_name.upper()}: {phase['description']}")
                    
                    # 優先度順にソート
                    combinations = phase['combinations']
                    priority_order = self.batch_config['execution']['priority_order']
                    combinations.sort(key=lambda x: priority_order.index(x['priority']))
                    
                    phase_results = []
                    
                    for combo in combinations:
                        result = self.run_single_evaluation(
                            combo['alpha_personal'],
                            combo['alpha_neutral'], 
                            combo['priority']
                        )
                        
                        if result:
                            self.results.append(result)
                            phase_results.append(result)
                            
                            # ベスト更新チェック
                            improvement = result['improvement_rate']
                            if improvement > self.best_improvement:
                                self.best_improvement = improvement
                                self.best_result = result
                                print(f"      🎯 NEW BEST: {improvement:+.1%}")
                            
                            # 目標達成で早期終了
                            if experiment['early_stop'] and improvement >= experiment['target_improvement']:
                                print(f"      🎉 TARGET ACHIEVED! {improvement:.1%}")
                                self.show_final_results()
                                return True
                    
                    # フェーズ結果サマリー
                    if phase_results:
                        best_phase = max(phase_results, key=lambda x: x['improvement_rate'])
                        print(f"   📊 Phase best: α_p={best_phase['alpha_personal']:.2f}, "
                              f"α_n={best_phase['alpha_neutral']:.2f} → {best_phase['improvement_rate']:+.1%}")
            
            # 最終結果表示
            self.show_final_results()
            return True
            
        finally:
            # 設定復元
            self.restore_main_config()
    
    def show_final_results(self):
        """最終結果を表示"""
        
        print("\n" + "=" * 70)
        print("📊 BATCH OPTIMIZATION RESULTS")
        print("=" * 70)
        
        if self.best_result:
            print(f"🏆 BEST RESULT:")
            print(f"   α_personal: {self.best_result['alpha_personal']:.2f}")
            print(f"   α_neutral:  {self.best_result['alpha_neutral']:.2f}")
            print(f"   Improvement: {self.best_result['improvement_rate']:+.1%}")
            print(f"   Baseline:    {self.best_result['baseline_accuracy']:.4f}")
            print(f"   Chameleon:   {self.best_result['chameleon_accuracy']:.4f}")
            print(f"   Priority:    {self.best_result['priority']}")
            
            # 成功レベル判定
            criteria = self.batch_config['success_criteria']
            improvement = self.best_result['improvement_rate']
            
            if improvement >= criteria['excellent_improvement']:
                print("   🎉 EXCELLENT! Target exceeded!")
            elif improvement >= criteria['good_improvement']:
                print("   ✨ VERY GOOD! Strong improvement!")
            elif improvement >= criteria['minimum_improvement']:
                print("   📈 GOOD! Notable improvement!")
            else:
                print("   📊 Some improvement detected")
            
            # 推奨設定
            print(f"\n💡 RECOMMENDED CONFIG:")
            print(f"   # Add to your config.yaml:")
            print(f"   chameleon:")
            print(f"     alpha_personal: {self.best_result['alpha_personal']:.2f}")
            print(f"     alpha_neutral: {self.best_result['alpha_neutral']:.2f}")
            
        else:
            print("❌ No successful results found")
        
        # 統計情報
        if self.results:
            improvements = [r['improvement_rate'] for r in self.results]
            print(f"\n📈 STATISTICS:")
            print(f"   Total tests: {len(self.results)}")
            print(f"   Best:        {max(improvements):+.1%}")
            print(f"   Average:     {np.mean(improvements):+.1%}")
            print(f"   Std dev:     {np.std(improvements):.3f}")
            
            # Top 5結果
            top_results = sorted(self.results, key=lambda x: x['improvement_rate'], reverse=True)[:5]
            print(f"\n🏆 TOP 5 RESULTS:")
            for i, result in enumerate(top_results, 1):
                print(f"   {i}. α_p={result['alpha_personal']:5.2f}, α_n={result['alpha_neutral']:6.2f} "
                      f"→ {result['improvement_rate']:+6.1%} [{result['priority']}]")
        
        # 結果保存
        self.save_batch_results()
        self.create_visualizations()
    
    def save_batch_results(self):
        """バッチ結果を保存"""
        
        output_dir = Path(self.batch_config['output']['base_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # 結果データ
        batch_results = {
            'experiment': self.batch_config['experiment'],
            'timestamp': time.time(),
            'total_tests': len(self.results),
            'best_result': self.best_result,
            'all_results': self.results,
            'statistics': {
                'improvements': [r['improvement_rate'] for r in self.results],
                'best_improvement': self.best_improvement,
                'average_improvement': np.mean([r['improvement_rate'] for r in self.results]) if self.results else 0,
                'std_improvement': np.std([r['improvement_rate'] for r in self.results]) if self.results else 0
            }
        }
        
        # JSON保存
        results_file = output_dir / "batch_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def create_visualizations(self):
        """結果の可視化"""
        
        if not self.results:
            return
        
        output_dir = Path(self.batch_config['output']['base_dir'])
        
        # データ準備
        alpha_p_values = [r['alpha_personal'] for r in self.results]
        alpha_n_values = [r['alpha_neutral'] for r in self.results]
        improvements = [r['improvement_rate'] * 100 for r in self.results]  # パーセント
        
        # プロット作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 散布図
        scatter = ax1.scatter(alpha_p_values, alpha_n_values, c=improvements, 
                            cmap='RdYlGn', s=60, alpha=0.7)
        ax1.set_xlabel('Alpha Personal')
        ax1.set_ylabel('Alpha Neutral')
        ax1.set_title('Alpha Parameters vs Improvement Rate')
        plt.colorbar(scatter, ax=ax1, label='Improvement (%)')
        
        # 改善率分布
        ax2.hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Improvement Rate (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Improvement Rates')
        ax2.axvline(x=25, color='red', linestyle='--', label='Target (25%)')
        ax2.legend()
        
        # Alpha Personal vs Improvement
        ax3.scatter(alpha_p_values, improvements, alpha=0.6, color='orange')
        ax3.set_xlabel('Alpha Personal')
        ax3.set_ylabel('Improvement Rate (%)')
        ax3.set_title('Alpha Personal vs Improvement')
        
        # Alpha Neutral vs Improvement
        ax4.scatter(alpha_n_values, improvements, alpha=0.6, color='green')
        ax4.set_xlabel('Alpha Neutral')
        ax4.set_ylabel('Improvement Rate (%)')
        ax4.set_title('Alpha Neutral vs Improvement')
        
        plt.tight_layout()
        
        # 保存
        plot_file = output_dir / "batch_optimization_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {plot_file}")

def main():
    """メイン実行"""
    runner = YAMLBatchRunner()
    runner.run_batch_optimization()

if __name__ == "__main__":
    main()
