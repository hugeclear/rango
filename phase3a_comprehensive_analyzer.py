#!/usr/bin/env python3
"""
Phase 3-A: Comprehensive Evaluation & Analysis
Systematic ablation study and deep performance analysis of 3-layer architecture
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# System path setup
sys.path.append('/home/nakata/master_thesis/rango')

# Import all evaluator layers
from chameleon_evaluator import ChameleonEvaluator
from causal_chameleon_evaluator import CausalConstrainedChameleon  
from manifold_chameleon_evaluator import ManifoldChameleonEvaluator

@dataclass
class AblationResults:
    """Results container for each ablation configuration"""
    config_name: str
    accuracy: float
    bleu_score: float
    f1_score: float
    exact_match: float
    inference_time_ms: float
    memory_usage_mb: float
    convergence_steps: int
    statistical_significance: Dict[str, float]
    detailed_metrics: Dict[str, Any]

@dataclass
class CausalAnalysisResults:
    """Results container for causal inference analysis"""
    ate_distribution: np.ndarray
    causal_graph: Dict[str, Any]
    temporal_sensitivity: Dict[str, float]
    do_calculus_efficiency: Dict[str, float]
    statistical_tests: Dict[str, float]

@dataclass 
class StiefelAnalysisResults:
    """Results container for Stiefel manifold analysis"""
    convergence_curve: np.ndarray
    orthogonality_evolution: np.ndarray
    gradient_norms: np.ndarray
    geodesic_distances: np.ndarray
    numerical_stability: Dict[str, float]

class Phase3AAnalyzer:
    """
    Comprehensive analyzer for Phase 3-A evaluation
    Conducts systematic ablation study and deep performance analysis
    """
    
    def __init__(self, output_dir: str = "results/phase3a_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_path = './chameleon_prime_personalization/data'
        self.config_path = 'config.yaml'
        
        # Setup logging
        self.setup_logging()
        
        # Initialize configuration matrix
        self.configurations = {
            'Config_A': {
                'name': 'Chameleon Only (Baseline)',
                'evaluator_class': ChameleonEvaluator,
                'use_causal': False,
                'use_manifold': False,
                'description': 'Original Chameleon baseline without enhancements'
            },
            'Config_B': {
                'name': 'Chameleon + Causal Inference',
                'evaluator_class': CausalConstrainedChameleon,
                'use_causal': True,
                'use_manifold': False,
                'description': 'Chameleon enhanced with PC algorithm and temporal constraints'
            },
            'Config_C': {
                'name': 'Chameleon + Stiefel Manifold',
                'evaluator_class': ManifoldChameleonEvaluator,
                'use_causal': False,
                'use_manifold': True,
                'description': 'Chameleon with Stiefel manifold optimization'
            },
            'Config_D': {
                'name': 'Full System (All Layers)',
                'evaluator_class': ManifoldChameleonEvaluator,
                'use_causal': True,
                'use_manifold': True,
                'description': 'Complete 3-layer architecture with all enhancements'
            }
        }
        
        self.ablation_results: Dict[str, AblationResults] = {}
        self.causal_analysis: Optional[CausalAnalysisResults] = None
        self.stiefel_analysis: Optional[StiefelAnalysisResults] = None
        
        self.logger.info("Phase3AAnalyzer initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / f"phase3a_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Phase3AAnalyzer')
    
    def run_ablation_study(self, mode: str = 'demo') -> Dict[str, AblationResults]:
        """
        Execute systematic ablation study across all 4 configurations
        
        Args:
            mode: Evaluation mode ('demo' for quick testing, 'full' for complete analysis)
        
        Returns:
            Dictionary of configuration results
        """
        self.logger.info("=== PHASE 3-A ABLATION STUDY STARTED ===")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Configurations: {list(self.configurations.keys())}")
        
        ablation_start_time = time.time()
        
        for config_key, config in self.configurations.items():
            self.logger.info(f"\n🧪 Evaluating {config_key}: {config['name']}")
            print(f"\n{'='*60}")
            print(f"🧪 CONFIGURATION {config_key}: {config['name']}")
            print(f"{'='*60}")
            
            try:
                config_start_time = time.time()
                
                # Initialize evaluator based on configuration
                evaluator = self._initialize_evaluator(config)
                
                # Run evaluation
                results = evaluator.run_evaluation(mode=mode)
                config_end_time = time.time()
                
                if results and isinstance(results, dict):
                    # Extract and process results
                    ablation_result = self._process_ablation_result(
                        config_key, config, results, 
                        config_end_time - config_start_time
                    )
                    
                    self.ablation_results[config_key] = ablation_result
                    
                    # Log results summary
                    self._log_config_summary(config_key, ablation_result)
                    
                else:
                    self.logger.error(f"❌ {config_key} failed: No valid results")
                    
            except Exception as e:
                self.logger.error(f"❌ {config_key} failed with error: {e}")
                import traceback
                traceback.print_exc()
        
        ablation_end_time = time.time()
        total_time = ablation_end_time - ablation_start_time
        
        self.logger.info(f"\n=== ABLATION STUDY COMPLETED ===")
        self.logger.info(f"Total execution time: {total_time:.1f}s")
        self.logger.info(f"Successful configurations: {len(self.ablation_results)}/4")
        
        # Generate comparative analysis
        if len(self.ablation_results) >= 2:
            self._generate_ablation_comparison()
        
        return self.ablation_results
    
    def _initialize_evaluator(self, config: Dict[str, Any]):
        """Initialize evaluator based on configuration specifications"""
        evaluator_class = config['evaluator_class']
        
        if evaluator_class == ChameleonEvaluator:
            return evaluator_class(self.config_path, self.data_path)
        elif evaluator_class == CausalConstrainedChameleon:
            return evaluator_class(self.config_path, self.data_path)
        elif evaluator_class == ManifoldChameleonEvaluator:
            evaluator = evaluator_class(self.config_path, self.data_path)
            # Configure manifold and causal settings
            if hasattr(evaluator, 'use_manifold'):
                evaluator.use_manifold = config['use_manifold']
            if hasattr(evaluator, 'use_causal_constraints'):
                evaluator.use_causal_constraints = config['use_causal']
            return evaluator
        else:
            raise ValueError(f"Unknown evaluator class: {evaluator_class}")
    
    def _process_ablation_result(self, config_key: str, config: Dict[str, Any], 
                               results: Dict[str, Any], execution_time: float) -> AblationResults:
        """Process raw evaluation results into structured ablation results"""
        
        # CRITICAL FIX: Use correct keys returned by ChameleonEvaluator
        baseline = results.get('baseline')  # NOT 'baseline_performance'
        enhanced = results.get('chameleon')  # NOT 'chameleon_performance'
        
        if enhanced and baseline:
            accuracy = enhanced.accuracy
            bleu_score = getattr(enhanced, 'bleu_score', 0.0)
            f1_score = getattr(enhanced, 'f1_score', 0.0)
            exact_match = getattr(enhanced, 'exact_match', 0.0)
        else:
            # Fallback to baseline if enhanced results not available
            accuracy = baseline.accuracy if baseline else 0.0
            bleu_score = getattr(baseline, 'bleu_score', 0.0) if baseline else 0.0
            f1_score = getattr(baseline, 'f1_score', 0.0) if baseline else 0.0
            exact_match = getattr(baseline, 'exact_match', 0.0) if baseline else 0.0
        
        # Extract statistical significance
        significance = results.get('significance', {})
        statistical_significance = {
            'p_value': significance.get('p_value', 1.0),
            'confidence_interval_lower': significance.get('ci_lower', 0.0),
            'confidence_interval_upper': significance.get('ci_upper', 0.0),
            'effect_size': significance.get('effect_size', 0.0)
        }
        
        # Detailed metrics
        detailed_metrics = {
            'baseline_accuracy': baseline.accuracy if baseline else 0.0,
            'enhanced_accuracy': enhanced.accuracy if enhanced else 0.0,
            'improvement_rate': ((enhanced.accuracy - baseline.accuracy) / baseline.accuracy * 100) 
                               if baseline and enhanced and baseline.accuracy > 0 else 0.0,
            'sample_count': baseline.total_samples if baseline else 0,  # Use total_samples not len(predictions)
            'execution_mode': results.get('mode', 'demo')  # Default to 'demo'
        }
        
        return AblationResults(
            config_name=config['name'],
            accuracy=accuracy,
            bleu_score=bleu_score,
            f1_score=f1_score,
            exact_match=exact_match,
            inference_time_ms=execution_time * 1000,  # Convert to milliseconds
            memory_usage_mb=0.0,  # To be implemented with memory profiling
            convergence_steps=0,   # To be extracted from optimization logs
            statistical_significance=statistical_significance,
            detailed_metrics=detailed_metrics
        )
    
    def _log_config_summary(self, config_key: str, result: AblationResults):
        """Log summary of configuration results"""
        self.logger.info(f"📊 {config_key} Results:")
        self.logger.info(f"   Accuracy: {result.accuracy:.4f}")
        self.logger.info(f"   BLEU Score: {result.bleu_score:.4f}")
        self.logger.info(f"   F1 Score: {result.f1_score:.4f}")
        self.logger.info(f"   Inference Time: {result.inference_time_ms:.1f}ms")
        self.logger.info(f"   Improvement Rate: {result.detailed_metrics['improvement_rate']:+.1f}%")
        self.logger.info(f"   Statistical Significance: p={result.statistical_significance['p_value']:.6f}")
    
    def _generate_ablation_comparison(self):
        """Generate comparative analysis of ablation results"""
        self.logger.info("\n📈 GENERATING ABLATION COMPARISON")
        
        # Create comparison DataFrame
        comparison_data = []
        for config_key, result in self.ablation_results.items():
            comparison_data.append({
                'Configuration': config_key,
                'Name': result.config_name,
                'Accuracy': result.accuracy,
                'BLEU': result.bleu_score,
                'F1': result.f1_score,
                'Exact_Match': result.exact_match,
                'Time_ms': result.inference_time_ms,
                'Improvement_%': result.detailed_metrics['improvement_rate'],
                'P_Value': result.statistical_significance['p_value'],
                'Significant': result.statistical_significance['p_value'] < 0.05
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = self.output_dir / 'ablation_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        self.logger.info(f"Comparison table saved: {comparison_file}")
        
        # Generate summary statistics
        self._generate_summary_statistics(comparison_df)
        
        return comparison_df
    
    def _generate_summary_statistics(self, comparison_df: pd.DataFrame):
        """Generate summary statistics from ablation results"""
        
        summary_stats = {
            'best_accuracy': {
                'config': comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Configuration'],
                'value': comparison_df['Accuracy'].max(),
                'improvement': comparison_df['Improvement_%'].max()
            },
            'fastest_inference': {
                'config': comparison_df.loc[comparison_df['Time_ms'].idxmin(), 'Configuration'],
                'value': comparison_df['Time_ms'].min()
            },
            'significant_improvements': int(comparison_df['Significant'].sum()),
            'total_configurations': len(comparison_df),
            'accuracy_range': {
                'min': comparison_df['Accuracy'].min(),
                'max': comparison_df['Accuracy'].max(),
                'std': comparison_df['Accuracy'].std()
            }
        }
        
        # Log summary
        self.logger.info("\n🏆 ABLATION SUMMARY STATISTICS:")
        self.logger.info(f"Best Accuracy: {summary_stats['best_accuracy']['config']} "
                        f"({summary_stats['best_accuracy']['value']:.4f}, "
                        f"{summary_stats['best_accuracy']['improvement']:+.1f}%)")
        self.logger.info(f"Fastest Inference: {summary_stats['fastest_inference']['config']} "
                        f"({summary_stats['fastest_inference']['value']:.1f}ms)")
        self.logger.info(f"Significant Improvements: {summary_stats['significant_improvements']}"
                        f"/{summary_stats['total_configurations']}")
        self.logger.info(f"Accuracy Range: {summary_stats['accuracy_range']['min']:.4f} - "
                        f"{summary_stats['accuracy_range']['max']:.4f} "
                        f"(σ={summary_stats['accuracy_range']['std']:.4f})")
        
        # Save summary statistics
        summary_file = self.output_dir / 'ablation_summary_statistics.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary statistics saved: {summary_file}")
    
    def analyze_causal_effects(self) -> CausalAnalysisResults:
        """
        Quantify causal inference effects with ATE distribution and PC algorithm analysis
        """
        self.logger.info("\n🕸️ ANALYZING CAUSAL EFFECTS")
        
        # This is a placeholder for causal analysis implementation
        # In a real implementation, this would:
        # 1. Extract causal graphs from CausalConstrainedChameleon
        # 2. Compute ATE distributions with bootstrap
        # 3. Analyze temporal constraint sensitivity
        # 4. Evaluate do-calculus computation efficiency
        
        # Placeholder results
        causal_results = CausalAnalysisResults(
            ate_distribution=np.random.normal(0.1, 0.02, 1000),  # Placeholder
            causal_graph={'nodes': [], 'edges': []},  # To be implemented
            temporal_sensitivity={'12h': 0.8, '24h': 0.9, '48h': 0.85},  # Placeholder
            do_calculus_efficiency={'computation_time': 0.5, 'convergence_rate': 0.95},
            statistical_tests={'ate_significance': 0.01, 'causal_strength': 0.75}
        )
        
        self.causal_analysis = causal_results
        self.logger.info("Causal effects analysis completed (placeholder)")
        
        return causal_results
    
    def analyze_stiefel_optimization(self) -> StiefelAnalysisResults:
        """
        Analyze Stiefel manifold optimization characteristics
        """
        self.logger.info("\n🧮 ANALYZING STIEFEL MANIFOLD OPTIMIZATION")
        
        # Placeholder for Stiefel analysis implementation
        # In a real implementation, this would:
        # 1. Extract optimization trajectories
        # 2. Compute orthogonality constraint evolution
        # 3. Analyze gradient norms and convergence
        # 4. Calculate geodesic distances
        # 5. Assess numerical stability
        
        # Placeholder results
        stiefel_results = StiefelAnalysisResults(
            convergence_curve=np.exp(-np.linspace(0, 5, 100)),  # Exponential decay
            orthogonality_evolution=np.random.normal(1e-6, 1e-7, 100),  # Near-zero error
            gradient_norms=np.exp(-np.linspace(0, 3, 100)) * 0.1,  # Decreasing norms
            geodesic_distances=np.linspace(0, 2, 100),  # Increasing distance
            numerical_stability={'condition_number': 12.5, 'spectral_gap': 0.85}
        )
        
        self.stiefel_analysis = stiefel_results
        self.logger.info("Stiefel optimization analysis completed (placeholder)")
        
        return stiefel_results
    
    def run_comprehensive_analysis(self, mode: str = 'demo') -> Dict[str, Any]:
        """
        Execute complete Phase 3-A analysis pipeline
        
        Args:
            mode: Evaluation mode for ablation study
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info("🎯 STARTING COMPREHENSIVE PHASE 3-A ANALYSIS")
        
        analysis_start_time = time.time()
        
        # Step 1: Ablation Study
        self.logger.info("\n📊 Step 1: Ablation Study")
        ablation_results = self.run_ablation_study(mode=mode)
        
        # Step 2: Causal Analysis (if causal configurations were successful)
        causal_configs = ['Config_B', 'Config_D']
        if any(config in ablation_results for config in causal_configs):
            self.logger.info("\n🕸️ Step 2: Causal Effects Analysis")
            causal_analysis = self.analyze_causal_effects()
        else:
            causal_analysis = None
            self.logger.warning("Skipping causal analysis - no causal configurations successful")
        
        # Step 3: Stiefel Analysis (if manifold configurations were successful)
        manifold_configs = ['Config_C', 'Config_D']
        if any(config in ablation_results for config in manifold_configs):
            self.logger.info("\n🧮 Step 3: Stiefel Optimization Analysis")
            stiefel_analysis = self.analyze_stiefel_optimization()
        else:
            stiefel_analysis = None
            self.logger.warning("Skipping Stiefel analysis - no manifold configurations successful")
        
        analysis_end_time = time.time()
        total_analysis_time = analysis_end_time - analysis_start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            'ablation_study': ablation_results,
            'causal_analysis': causal_analysis,
            'stiefel_analysis': stiefel_analysis,
            'analysis_metadata': {
                'total_time_seconds': total_analysis_time,
                'timestamp': datetime.now().isoformat(),
                'mode': mode,
                'successful_configurations': len(ablation_results),
                'output_directory': str(self.output_dir)
            }
        }
        
        # Save comprehensive results
        results_file = self.output_dir / 'comprehensive_analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert results to JSON-serializable format
            json_results = self._serialize_results_for_json(comprehensive_results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n✅ PHASE 3-A ANALYSIS COMPLETED")
        self.logger.info(f"Total analysis time: {total_analysis_time:.1f}s")
        self.logger.info(f"Results saved: {results_file}")
        
        return comprehensive_results
    
    def _serialize_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex objects to JSON-serializable format"""
        
        def convert_value(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: convert_value(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        return convert_value(results)

def main():
    """Main execution function for Phase 3-A analysis"""
    print("🎯 PHASE 3-A: Comprehensive Evaluation & Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = Phase3AAnalyzer()
    
    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_analysis(mode='demo')
        
        print("\n✅ Phase 3-A Analysis Completed Successfully!")
        print(f"Results directory: {analyzer.output_dir}")
        
        # Print summary
        if results['ablation_study']:
            print(f"\nSuccessful configurations: {len(results['ablation_study'])}/4")
            for config, result in results['ablation_study'].items():
                print(f"  {config}: {result.accuracy:.4f} accuracy")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()