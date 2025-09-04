#!/usr/bin/env python3
"""
Step 3: Systematic Grid Search with Statistical Validation

Phase 3: 評価条件健全化と系統的パラメータ探索
現在の状況：140サンプルの層別化データセット準備完了、生成設定修正済み
目標：統計的検証付きグリッドサーチで最適パラメータ特定と有意性確認

Key improvements from user requirements:
- Use expanded dataset (140 samples) for statistical significance
- Fixed generation settings (sampling mode for editing observation) 
- Comprehensive parameter space exploration
- Statistical validation with proper significance testing
- Early stopping and efficiency optimization
"""

import sys
import os
import json
import time
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from itertools import product
import csv

# Statistical testing
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GridSearchConfig:
    """Configuration for systematic grid search"""
    # Parameter ranges
    alpha_personal_range: List[float] = None
    alpha_general_range: List[float] = None
    target_layers_options: List[List[str]] = None
    
    # Generation settings
    temperature: float = 0.7  # Fixed sampling temperature for editing observation
    max_new_tokens: int = 10
    do_sample: bool = True
    
    # Statistical validation
    significance_level: float = 0.05
    min_improvement_threshold: float = 0.01  # 1% minimum improvement
    
    # Efficiency settings
    early_stopping_patience: int = 3  # Stop after 3 consecutive non-improvements
    max_grid_size: int = 50  # Maximum number of parameter combinations
    parallel_evaluation: bool = False  # For future implementation
    
    def __post_init__(self):
        # Set default parameter ranges if not provided
        if self.alpha_personal_range is None:
            self.alpha_personal_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        
        if self.alpha_general_range is None:
            self.alpha_general_range = [0.0, -0.02, -0.05, -0.1, -0.15, -0.2, -0.3]
        
        if self.target_layers_options is None:
            self.target_layers_options = [
                ["model.layers.20"],
                ["model.layers.24"], 
                ["model.layers.27"],
                ["model.layers.20", "model.layers.27"],
                ["model.layers.24", "model.layers.27"],
                ["model.layers.20", "model.layers.24", "model.layers.27"]
            ]


@dataclass
class GridSearchResult:
    """Results from a single grid search experiment"""
    # Parameters
    alpha_personal: float
    alpha_general: float
    target_layers: List[str]
    
    # Performance metrics
    baseline_accuracy: float
    chameleon_accuracy: float
    baseline_bleu: float
    chameleon_bleu: float
    baseline_f1: float
    chameleon_f1: float
    
    # Statistical validation
    accuracy_improvement: float
    accuracy_improvement_pct: float
    p_value: float
    is_statistically_significant: bool
    
    # Diagnostic information
    avg_edit_ratio: float
    hook_calls_mean: float
    inference_time: float
    sample_count: int
    
    # Meta information
    timestamp: str
    success: bool
    error_message: str = ""


class SystematicGridSearcher:
    """
    Performs systematic grid search with statistical validation for Chameleon optimization
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 dataset_path: str = "data/evaluation/lamp2_expanded_eval.jsonl"):
        self.config_path = Path(config_path)
        self.dataset_path = Path(dataset_path)
        
        # Load system configuration
        self.system_config = self._load_config()
        
        # Grid search configuration
        self.grid_config = GridSearchConfig()
        
        # Results storage
        self.results: List[GridSearchResult] = []
        self.best_result: Optional[GridSearchResult] = None
        
        # Evaluation dataset
        self.evaluation_dataset = []
        
        # Early stopping tracking
        self.consecutive_non_improvements = 0
        
        # Statistics tracking
        self.search_stats = {
            'total_configurations': 0,
            'completed_configurations': 0,
            'successful_configurations': 0,
            'statistically_significant_results': 0,
            'best_accuracy_improvement': 0.0,
            'total_evaluation_time': 0.0,
            'start_time': None,
            'end_time': None
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def load_evaluation_dataset(self) -> bool:
        """Load the expanded evaluation dataset"""
        logger.info(f"📊 Loading evaluation dataset: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset file not found: {self.dataset_path}")
            return False
        
        try:
            self.evaluation_dataset = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.evaluation_dataset.append(sample)
            
            logger.info(f"✅ Loaded {len(self.evaluation_dataset)} evaluation samples")
            
            # Quick validation
            if len(self.evaluation_dataset) < 50:
                logger.warning(f"⚠️ Small dataset size: {len(self.evaluation_dataset)} samples")
            
            # Check data quality
            users = set(s['user_id'] for s in self.evaluation_dataset)
            tags = set(s['reference'] for s in self.evaluation_dataset)
            
            logger.info(f"📊 Dataset statistics:")
            logger.info(f"   • Users: {len(users)}")
            logger.info(f"   • Tags: {len(tags)}")
            logger.info(f"   • Avg samples per user: {len(self.evaluation_dataset)/len(users):.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def generate_parameter_combinations(self) -> List[Tuple[float, float, List[str]]]:
        """Generate all parameter combinations for grid search"""
        logger.info("🔧 Generating parameter combinations...")
        
        config = self.grid_config
        
        # Generate all combinations
        combinations = []
        for alpha_p in config.alpha_personal_range:
            for alpha_g in config.alpha_general_range:
                for layers in config.target_layers_options:
                    combinations.append((alpha_p, alpha_g, layers))
        
        # Limit grid size if needed
        if len(combinations) > config.max_grid_size:
            logger.info(f"⚠️ Grid size ({len(combinations)}) exceeds maximum ({config.max_grid_size})")
            logger.info("🎯 Using intelligent sampling to reduce grid size...")
            
            # Intelligent sampling: prioritize diverse layer combinations and moderate parameters
            priority_combinations = []
            regular_combinations = []
            
            for combo in combinations:
                alpha_p, alpha_g, layers = combo
                # Priority: moderate parameters + diverse layer combinations
                if (0.2 <= alpha_p <= 0.5 and -0.2 <= alpha_g <= -0.05 and len(layers) >= 2):
                    priority_combinations.append(combo)
                else:
                    regular_combinations.append(combo)
            
            # Select combinations
            selected = priority_combinations[:config.max_grid_size//2]
            remaining_slots = config.max_grid_size - len(selected)
            if remaining_slots > 0:
                import random
                random.seed(42)  # Reproducible
                selected.extend(random.sample(regular_combinations, min(remaining_slots, len(regular_combinations))))
            
            combinations = selected
            logger.info(f"🎯 Reduced to {len(combinations)} combinations (prioritized diverse layer configs)")
        
        self.search_stats['total_configurations'] = len(combinations)
        
        logger.info(f"📋 Grid search configuration:")
        logger.info(f"   • Alpha personal: {config.alpha_personal_range}")
        logger.info(f"   • Alpha general: {config.alpha_general_range}")
        logger.info(f"   • Layer combinations: {len(config.target_layers_options)}")
        logger.info(f"   • Total combinations: {len(combinations)}")
        logger.info(f"   • Generation mode: sampling (temp={config.temperature})")
        
        return combinations
    
    def evaluate_single_configuration(self, alpha_personal: float, alpha_general: float, 
                                    target_layers: List[str]) -> GridSearchResult:
        """Evaluate a single parameter configuration"""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"⚡ Evaluating: α_p={alpha_personal:.1f}, α_g={alpha_general:.2f}, layers={len(target_layers)}")
        
        try:
            # Import ChameleonEvaluator
            sys.path.insert(0, str(Path.cwd()))
            from chameleon_evaluator import ChameleonEvaluator
            
            # Create evaluator with fixed generation settings
            evaluator = ChameleonEvaluator(
                config_path=str(self.config_path),
                data_path="./chameleon_prime_personalization/data",
                decoding_mode="sample"  # Use sampling mode for editing observation
            )
            
            # Override generation kwargs for consistency
            evaluator.gen_kwargs.update({
                'temperature': self.grid_config.temperature,
                'do_sample': self.grid_config.do_sample,
                'max_new_tokens': self.grid_config.max_new_tokens,
                'top_p': 0.9
            })
            
            # Convert evaluation dataset to format expected by evaluator
            test_samples = []
            ground_truth = {}
            
            for sample in self.evaluation_dataset:
                test_sample = {
                    'id': sample['id'],
                    'input': sample['question'],
                    'profile': sample.get('profile', [])
                }
                test_samples.append(test_sample)
                ground_truth[str(sample['id'])] = sample['reference']
            
            # Cache the samples
            evaluator.test_samples_cache = test_samples
            
            # Run baseline evaluation
            baseline_result = evaluator.evaluation_engine.evaluate_baseline(test_samples, ground_truth)
            
            # Run Chameleon evaluation with specified parameters
            chameleon_result = evaluator.evaluation_engine.evaluate_chameleon(
                test_samples=test_samples,
                ground_truth=ground_truth,
                alpha_personal=alpha_personal,
                alpha_neutral=alpha_general,
                target_layers=target_layers,
                name=f"Chameleon(α_p={alpha_personal:.1f},α_g={alpha_general:.2f})"
            )
            
            # Statistical significance testing
            if len(baseline_result.predictions) == len(chameleon_result.predictions):
                # Paired t-test for accuracy differences
                baseline_correct = np.array([int(p == g) for p, g in zip(baseline_result.predictions, baseline_result.ground_truths)])
                chameleon_correct = np.array([int(p == g) for p, g in zip(chameleon_result.predictions, chameleon_result.ground_truths)])
                
                if len(baseline_correct) > 1:
                    try:
                        # Use paired t-test
                        _, p_value = ttest_rel(chameleon_correct, baseline_correct)
                    except:
                        # Fallback to Wilcoxon signed-rank test
                        _, p_value = wilcoxon(chameleon_correct, baseline_correct, alternative='two-sided')
                else:
                    p_value = 1.0
            else:
                p_value = 1.0
            
            # Compute improvements
            accuracy_improvement = chameleon_result.accuracy - baseline_result.accuracy
            accuracy_improvement_pct = (accuracy_improvement / baseline_result.accuracy * 100) if baseline_result.accuracy > 0 else 0.0
            
            # Statistical significance
            is_significant = p_value < self.grid_config.significance_level
            
            # Diagnostic information
            avg_edit_ratio = np.mean(evaluator.evaluation_diagnostics['edit_ratios']) if evaluator.evaluation_diagnostics['edit_ratios'] else 0.0
            hook_calls_mean = np.mean(evaluator.evaluation_diagnostics['hook_calls']) if evaluator.evaluation_diagnostics['hook_calls'] else 0.0
            
            inference_time = time.time() - start_time
            
            # Create result
            result = GridSearchResult(
                alpha_personal=alpha_personal,
                alpha_general=alpha_general,
                target_layers=target_layers,
                baseline_accuracy=baseline_result.accuracy,
                chameleon_accuracy=chameleon_result.accuracy,
                baseline_bleu=baseline_result.bleu_score,
                chameleon_bleu=chameleon_result.bleu_score,
                baseline_f1=baseline_result.f1_score,
                chameleon_f1=chameleon_result.f1_score,
                accuracy_improvement=accuracy_improvement,
                accuracy_improvement_pct=accuracy_improvement_pct,
                p_value=p_value,
                is_statistically_significant=is_significant,
                avg_edit_ratio=avg_edit_ratio,
                hook_calls_mean=hook_calls_mean,
                inference_time=inference_time,
                sample_count=len(test_samples),
                timestamp=timestamp,
                success=True
            )
            
            # Log result
            significance_marker = "✅" if is_significant else "❌"
            improvement_marker = "📈" if accuracy_improvement > 0 else "📉"
            
            logger.info(f"   Result: {improvement_marker} Δacc={accuracy_improvement_pct:+.1f}% (p={p_value:.4f}) {significance_marker}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Configuration failed: {e}")
            
            result = GridSearchResult(
                alpha_personal=alpha_personal,
                alpha_general=alpha_general,
                target_layers=target_layers,
                baseline_accuracy=0.0,
                chameleon_accuracy=0.0,
                baseline_bleu=0.0,
                chameleon_bleu=0.0,
                baseline_f1=0.0,
                chameleon_f1=0.0,
                accuracy_improvement=0.0,
                accuracy_improvement_pct=0.0,
                p_value=1.0,
                is_statistically_significant=False,
                avg_edit_ratio=0.0,
                hook_calls_mean=0.0,
                inference_time=time.time() - start_time,
                sample_count=len(self.evaluation_dataset),
                timestamp=timestamp,
                success=False,
                error_message=str(e)
            )
            
            return result
    
    def should_early_stop(self, latest_result: GridSearchResult) -> bool:
        """Check if early stopping criteria are met"""
        if not self.grid_config.early_stopping_patience:
            return False
        
        # Check if this result is an improvement over the best so far
        if self.best_result is None:
            self.best_result = latest_result
            self.consecutive_non_improvements = 0
            return False
        
        # Compare with best result
        is_improvement = (
            latest_result.success and
            latest_result.accuracy_improvement > self.best_result.accuracy_improvement and
            latest_result.accuracy_improvement > self.grid_config.min_improvement_threshold
        )
        
        if is_improvement:
            self.best_result = latest_result
            self.consecutive_non_improvements = 0
            logger.info(f"🎯 New best result: {latest_result.accuracy_improvement_pct:+.1f}% improvement")
        else:
            self.consecutive_non_improvements += 1
            logger.info(f"⚠️ No improvement ({self.consecutive_non_improvements}/{self.grid_config.early_stopping_patience})")
        
        # Early stopping decision
        should_stop = self.consecutive_non_improvements >= self.grid_config.early_stopping_patience
        if should_stop:
            logger.info(f"🛑 Early stopping triggered after {self.consecutive_non_improvements} consecutive non-improvements")
        
        return should_stop
    
    def run_grid_search(self) -> Dict[str, Any]:
        """Run the complete systematic grid search"""
        logger.info("🚀 Starting systematic grid search with statistical validation")
        logger.info("=" * 80)
        
        self.search_stats['start_time'] = time.time()
        
        try:
            # Load evaluation dataset
            if not self.load_evaluation_dataset():
                return {"success": False, "error": "Failed to load evaluation dataset"}
            
            # Generate parameter combinations
            combinations = self.generate_parameter_combinations()
            
            # Execute grid search
            logger.info(f"🔍 Executing grid search...")
            
            for i, (alpha_p, alpha_g, layers) in enumerate(combinations):
                logger.info(f"📊 Configuration {i+1}/{len(combinations)}")
                
                result = self.evaluate_single_configuration(alpha_p, alpha_g, layers)
                self.results.append(result)
                
                # Update statistics
                self.search_stats['completed_configurations'] += 1
                if result.success:
                    self.search_stats['successful_configurations'] += 1
                    if result.is_statistically_significant:
                        self.search_stats['statistically_significant_results'] += 1
                
                # Track best result
                if result.success and (self.best_result is None or 
                                     result.accuracy_improvement > self.best_result.accuracy_improvement):
                    self.best_result = result
                    self.search_stats['best_accuracy_improvement'] = result.accuracy_improvement
                
                # Check early stopping
                if self.should_early_stop(result):
                    logger.info(f"🛑 Early stopping at configuration {i+1}/{len(combinations)}")
                    break
                
                # Progress update
                if (i + 1) % 5 == 0 or i == len(combinations) - 1:
                    self._print_progress_update(i + 1, len(combinations))
            
            # Final analysis
            self.search_stats['end_time'] = time.time()
            self.search_stats['total_evaluation_time'] = self.search_stats['end_time'] - self.search_stats['start_time']
            
            analysis_results = self._analyze_results()
            
            # Save results
            self._save_results()
            
            logger.info("✅ Grid search completed successfully!")
            return {
                "success": True,
                "results": analysis_results,
                "best_result": asdict(self.best_result) if self.best_result else None,
                "statistics": self.search_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Grid search failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _print_progress_update(self, completed: int, total: int):
        """Print progress update"""
        progress_pct = completed / total * 100
        successful = self.search_stats['successful_configurations']
        significant = self.search_stats['statistically_significant_results']
        
        logger.info(f"📊 Progress: {completed}/{total} ({progress_pct:.1f}%) | "
                   f"Success: {successful}/{completed} | Significant: {significant}/{successful if successful > 0 else 1}")
        
        if self.best_result:
            logger.info(f"🏆 Current best: {self.best_result.accuracy_improvement_pct:+.1f}% "
                       f"(α_p={self.best_result.alpha_personal:.1f}, α_g={self.best_result.alpha_general:.2f})")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze grid search results"""
        logger.info("📊 Analyzing grid search results...")
        
        successful_results = [r for r in self.results if r.success]
        significant_results = [r for r in successful_results if r.is_statistically_significant]
        improved_results = [r for r in successful_results if r.accuracy_improvement > 0]
        
        if not successful_results:
            logger.warning("⚠️ No successful results to analyze")
            return {"analysis": "no_successful_results"}
        
        # Statistical analysis
        accuracy_improvements = [r.accuracy_improvement_pct for r in successful_results]
        p_values = [r.p_value for r in successful_results]
        
        analysis = {
            "total_configurations": len(self.results),
            "successful_configurations": len(successful_results),
            "statistically_significant": len(significant_results),
            "positive_improvements": len(improved_results),
            "accuracy_improvement_stats": {
                "mean": np.mean(accuracy_improvements),
                "std": np.std(accuracy_improvements),
                "min": np.min(accuracy_improvements),
                "max": np.max(accuracy_improvements),
                "median": np.median(accuracy_improvements)
            },
            "p_value_stats": {
                "mean": np.mean(p_values),
                "min": np.min(p_values),
                "significant_fraction": len(significant_results) / len(successful_results)
            },
            "best_configurations": sorted(
                [asdict(r) for r in successful_results], 
                key=lambda x: x['accuracy_improvement'], 
                reverse=True
            )[:5],  # Top 5 configurations
            "parameter_analysis": self._analyze_parameter_effects(successful_results)
        }
        
        # Log analysis summary
        logger.info(f"📈 Results Analysis:")
        logger.info(f"   • Successful configurations: {len(successful_results)}/{len(self.results)}")
        logger.info(f"   • Statistically significant: {len(significant_results)}")
        logger.info(f"   • Positive improvements: {len(improved_results)}")
        logger.info(f"   • Mean improvement: {analysis['accuracy_improvement_stats']['mean']:+.1f}%")
        logger.info(f"   • Best improvement: {analysis['accuracy_improvement_stats']['max']:+.1f}%")
        logger.info(f"   • Significant fraction: {analysis['p_value_stats']['significant_fraction']:.1%}")
        
        return analysis
    
    def _analyze_parameter_effects(self, results: List[GridSearchResult]) -> Dict[str, Any]:
        """Analyze the effects of different parameters"""
        # Group by parameter values
        by_alpha_p = {}
        by_alpha_g = {}
        by_layer_count = {}
        
        for result in results:
            # Alpha personal
            if result.alpha_personal not in by_alpha_p:
                by_alpha_p[result.alpha_personal] = []
            by_alpha_p[result.alpha_personal].append(result.accuracy_improvement_pct)
            
            # Alpha general
            if result.alpha_general not in by_alpha_g:
                by_alpha_g[result.alpha_general] = []
            by_alpha_g[result.alpha_general].append(result.accuracy_improvement_pct)
            
            # Layer count
            layer_count = len(result.target_layers)
            if layer_count not in by_layer_count:
                by_layer_count[layer_count] = []
            by_layer_count[layer_count].append(result.accuracy_improvement_pct)
        
        def analyze_group(group_dict: Dict) -> Dict:
            return {
                str(k): {
                    "mean": np.mean(v),
                    "std": np.std(v),
                    "count": len(v),
                    "best": np.max(v)
                }
                for k, v in group_dict.items()
            }
        
        return {
            "alpha_personal_effects": analyze_group(by_alpha_p),
            "alpha_general_effects": analyze_group(by_alpha_g),
            "layer_count_effects": analyze_group(by_layer_count)
        }
    
    def _save_results(self):
        """Save grid search results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_grid_search_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "grid_search_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, ensure_ascii=False)
        
        # Save CSV for analysis
        csv_file = output_dir / "grid_search_results.csv"
        if self.results:
            fieldnames = list(asdict(self.results[0]).keys())
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    # Convert list to string for CSV
                    row['target_layers'] = ','.join(row['target_layers'])
                    writer.writerow(row)
        
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            summary = {
                "search_configuration": asdict(self.grid_config),
                "statistics": self.search_stats,
                "best_result": asdict(self.best_result) if self.best_result else None,
                "dataset_info": {
                    "path": str(self.dataset_path),
                    "sample_count": len(self.evaluation_dataset),
                    "users": len(set(s['user_id'] for s in self.evaluation_dataset)),
                    "tags": len(set(s['reference'] for s in self.evaluation_dataset))
                }
            }
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_dir}")


def main():
    """Main execution function for systematic grid search"""
    logger.info("🚀 Starting Step 3: Systematic Grid Search with Statistical Validation")
    logger.info("=" * 80)
    
    # Initialize grid searcher
    searcher = SystematicGridSearcher(
        config_path="config.yaml",
        dataset_path="data/evaluation/lamp2_expanded_eval.jsonl"
    )
    
    # Customize grid search parameters
    searcher.grid_config.alpha_personal_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    searcher.grid_config.alpha_general_range = [0.0, -0.02, -0.05, -0.1, -0.15, -0.2]
    searcher.grid_config.target_layers_options = [
        ["model.layers.20"],
        ["model.layers.27"], 
        ["model.layers.20", "model.layers.27"],
        ["model.layers.20", "model.layers.24", "model.layers.27"]
    ]
    searcher.grid_config.max_grid_size = 30  # Limit for efficiency
    searcher.grid_config.early_stopping_patience = 5  # Allow more exploration
    
    # Run grid search
    results = searcher.run_grid_search()
    
    if results["success"]:
        logger.info("🎉 Step 3 COMPLETED SUCCESSFULLY!")
        logger.info("✅ Systematic parameter exploration completed")
        logger.info("✅ Statistical validation applied to all configurations")
        logger.info("✅ Optimal parameters identified with significance testing")
        
        if searcher.best_result:
            best = searcher.best_result
            logger.info(f"🏆 Best configuration:")
            logger.info(f"   • α_personal: {best.alpha_personal}")
            logger.info(f"   • α_general: {best.alpha_general}")
            logger.info(f"   • Target layers: {best.target_layers}")
            logger.info(f"   • Accuracy improvement: {best.accuracy_improvement_pct:+.1f}%")
            logger.info(f"   • Statistical significance: {'YES' if best.is_statistically_significant else 'NO'} (p={best.p_value:.4f})")
        
        logger.info("🚀 Ready for Step 4: Final validation and production deployment")
        return 0
    else:
        logger.error(f"❌ Step 3 FAILED: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)