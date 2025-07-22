#!/usr/bin/env python3
"""
Chameleon LaMP自動評価システム - 実行スクリプト

Usage:
    python run_evaluation.py --mode demo      # デモ実行 (3ユーザー)
    python run_evaluation.py --mode full      # 本格評価 (10ユーザー)
    python run_evaluation.py --mode ablation  # アブレーション研究
    python run_evaluation.py --mode results   # 結果確認
"""

import argparse
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

class EnvironmentChecker:
    """実行環境チェッククラス"""
    
    @staticmethod
    def check_python_version():
        """Python バージョンチェック"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
        logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    @staticmethod
    def check_gpu():
        """CUDA/GPU 利用可能性チェック"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(f"✅ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
                return True
            else:
                logger.warning("⚠️  CUDA not available - using CPU (slower)")
                return False
        except ImportError:
            logger.error("❌ PyTorch not installed")
            return False
    
    @staticmethod
    def check_memory():
        """メモリ使用量チェック"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            logger.info(f"💾 Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
            
            if available_gb < 8:
                logger.warning("⚠️  Low memory available - consider closing other applications")
            
            return available_gb >= 4  # 最低4GB必要
        except ImportError:
            logger.warning("Cannot check memory - psutil not installed")
            return True
    
    @staticmethod
    def check_dependencies():
        """依存関係チェック"""
        required_packages = [
            'torch', 'transformers', 'numpy', 'pandas', 'scikit-learn',
            'matplotlib', 'seaborn', 'scipy', 'yaml', 'nltk'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} installed")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} not installed")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.error("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    @staticmethod
    def check_data_files():
        """必要データファイルの存在確認"""
        required_files = [
            "chameleon_prime_personalization/data/raw/LaMP-2/merged.json",
            "processed/LaMP-2/theta_p.json",
            "processed/LaMP-2/theta_n.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} not found")
            else:
                logger.info(f"✅ {file_path} found")
        
        if missing_files:
            logger.error("Missing required data files:")
            for file_path in missing_files:
                logger.error(f"  - {file_path}")
            return False
        
        return True
    
    @classmethod
    def run_all_checks(cls):
        """全チェック実行"""
        logger.info("🔍 Running environment checks...")
        
        checks = [
            ("Python version", cls.check_python_version),
            ("Dependencies", cls.check_dependencies),
            ("GPU/CUDA", cls.check_gpu),
            ("Memory", cls.check_memory),
            ("Data files", cls.check_data_files)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                if result is False:
                    logger.error(f"❌ {check_name} check failed")
                    return False
            except Exception as e:
                logger.error(f"❌ {check_name} check failed: {e}")
                return False
        
        logger.info("✅ All environment checks passed!")
        return True

class EvaluationRunner:
    """評価実行管理クラス"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.config_path = self.script_dir / "config.yaml"
        self.results_dir = self.script_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_demo_evaluation(self) -> Dict[str, Any]:
        """デモ評価実行 (3ユーザー、約5分)"""
        logger.info("🚀 Starting demo evaluation...")
        logger.info("📋 Demo parameters:")
        logger.info("   - Users: 3")
        logger.info("   - Estimated time: 5-10 minutes")
        logger.info("   - Purpose: Quick system verification")
        
        return self._run_chameleon_evaluation(mode="demo")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """本格評価実行 (10ユーザー、約30-60分)"""
        logger.info("🚀 Starting full evaluation...")
        logger.info("📋 Full evaluation parameters:")
        logger.info("   - Users: 10")
        logger.info("   - Estimated time: 30-60 minutes")
        logger.info("   - Purpose: Complete research evaluation")
        
        # 確認プロンプト
        response = input("This will take 30-60 minutes. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Evaluation cancelled by user")
            return {}
        
        return self._run_chameleon_evaluation(mode="full")
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """アブレーション研究実行"""
        logger.info("🔬 Starting ablation study...")
        
        alpha_values = [0.5, 1.0, 1.5, 2.0]
        results = {}
        
        for alpha in alpha_values:
            logger.info(f"Running ablation with alpha_personal={alpha}")
            
            # パラメータを一時的に変更
            config = self._load_config()
            config['chameleon']['alpha_personal'] = alpha
            self._save_temp_config(config)
            
            result = self._run_chameleon_evaluation(mode="ablation", config_override=True)
            results[f"alpha_{alpha}"] = result
        
        logger.info("✅ Ablation study completed")
        return results
    
    def show_recent_results(self):
        """最近の評価結果を表示"""
        logger.info("📊 Recent evaluation results:")
        
        result_dirs = list(self.results_dir.glob("evaluation_*"))
        result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not result_dirs:
            logger.info("No previous results found")
            return
        
        # 最新の結果を表示
        latest_result = result_dirs[0]
        result_file = latest_result / "results.json"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            self._display_results_summary(results)
        else:
            logger.info(f"Result file not found in {latest_result}")
    
    def _run_chameleon_evaluation(self, mode: str, config_override: bool = False) -> Dict[str, Any]:
        """Chameleon評価実行"""
        try:
            from chameleon_evaluator import ChameleonEvaluator
            
            config_path = "temp_config.yaml" if config_override else str(self.config_path)
            evaluator = ChameleonEvaluator(config_path=config_path if Path(config_path).exists() else None)
            
            start_time = time.time()
            results = evaluator.run_evaluation(mode=mode)
            end_time = time.time()
            
            logger.info(f"✅ Evaluation completed in {end_time - start_time:.1f} seconds")
            
            # 結果サマリー表示
            self._display_results_summary(results)
            
            return results
            
        except ImportError as e:
            logger.error(f"Failed to import chameleon_evaluator: {e}")
            return {}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
        finally:
            # 一時設定ファイルを削除
            if config_override and Path("temp_config.yaml").exists():
                os.remove("temp_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if self.config_path.exists():
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # デフォルト設定
            return {
                'model': {
                    'name': 'meta-llama/Llama-3.2-3B-Instruct',
                    'device': 'auto'
                },
                'chameleon': {
                    'alpha_personal': 1.5,
                    'alpha_general': -0.8
                },
                'evaluation': {
                    'max_users': 10
                }
            }
    
    def _save_temp_config(self, config: Dict[str, Any]):
        """一時設定ファイル保存"""
        import yaml
        with open("temp_config.yaml", 'w') as f:
            yaml.dump(config, f)
    
    def _display_results_summary(self, results: Dict[str, Any]):
        """結果サマリー表示"""
        print("\n" + "=" * 60)
        print("🎯 Evaluation Results Summary")
        print("=" * 60)
        
        baseline = results.get('baseline', {})
        chameleon = results.get('chameleon', {})
        significance = results.get('significance', {})
        
        if baseline:
            print(f"\n📊 Baseline:")
            print(f"   Accuracy: {baseline.get('accuracy', 0):.3f}")
            print(f"   Exact Match: {baseline.get('exact_match', 0):.3f}")
            print(f"   BLEU Score: {baseline.get('bleu_score', 0):.3f}")
        
        if chameleon:
            print(f"\n🦎 Chameleon:")
            print(f"   Accuracy: {chameleon.get('accuracy', 0):.3f}")
            print(f"   Exact Match: {chameleon.get('exact_match', 0):.3f}")
            print(f"   BLEU Score: {chameleon.get('bleu_score', 0):.3f}")
        
        if baseline and chameleon and significance:
            improvement = significance.get('improvement_rate', 0) * 100
            p_value = significance.get('p_value', 1.0)
            
            print(f"\n📈 Improvement:")
            print(f"   Rate: {improvement:+.1f}%")
            print(f"   p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("   ✅ Statistically significant!")
            else:
                print("   ⚠️  Not statistically significant")
        
        print("\n" + "=" * 60)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Chameleon LaMP自動評価システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  python run_evaluation.py --mode demo      # 3ユーザー、5分のデモ実行
  python run_evaluation.py --mode full      # 10ユーザー、60分の本格評価
  python run_evaluation.py --mode ablation  # パラメータ感度分析
  python run_evaluation.py --mode results   # 過去の結果表示
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["demo", "full", "ablation", "results"],
        default="demo",
        help="実行モード (default: demo)"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="環境チェックをスキップ"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="結果出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    print("🦎 Chameleon LaMP自動評価システム")
    print("=" * 50)
    
    # 環境チェック
    if not args.skip_checks:
        if not EnvironmentChecker.run_all_checks():
            logger.error("❌ Environment checks failed. Fix issues before proceeding.")
            sys.exit(1)
    
    # 評価実行
    runner = EvaluationRunner()
    
    try:
        if args.mode == "demo":
            results = runner.run_demo_evaluation()
        elif args.mode == "full":
            results = runner.run_full_evaluation()
        elif args.mode == "ablation":
            results = runner.run_ablation_study()
        elif args.mode == "results":
            runner.show_recent_results()
            return
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        if results:
            logger.info("🎉 Evaluation completed successfully!")
            logger.info(f"📁 Results saved in: {runner.results_dir}")
        else:
            logger.error("❌ Evaluation failed or was cancelled")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()