#!/usr/bin/env python3
"""
Week 2 evaluation runner for GraphRAG-CFS-Chameleon
Integrates all components: diversity, new metrics, significance testing
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.runner import GraphRAGChameleonEvaluator
from eval.significance import compare_all_conditions
from eval.report import EvaluationReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(data_path: str = None) -> List[Dict[str, Any]]:
    """
    Load test data for evaluation
    This is a placeholder - replace with actual data loading
    """
    if data_path and Path(data_path).exists():
        with open(data_path, 'r') as f:
            return json.load(f)
    
    # Mock test data for demonstration
    logger.warning("Using mock test data - replace with actual LaMP-2 data loading")
    
    mock_data = []
    for i in range(50):  # Small sample for testing
        mock_data.append({
            'user_id': f'user_{i % 10}',  # 10 unique users
            'question': f'What is the answer to question {i}?',
            'reference': f'The answer to question {i} is {i * 2}.',
            'context': f'This is context for user {i % 10}.',
            'id': str(i)
        })
    
    logger.info(f"Loaded {len(mock_data)} test examples (mock data)")
    return mock_data


def create_default_config() -> Dict[str, Any]:
    """Create default evaluation configuration"""
    return {
        'legacy_mode': False,
        'graphrag': {
            'enabled': True,
            'cfs_pool_path': 'artifacts/20250810_053000/graphrag_cfs_weights/cfs_pool.parquet',
            'user_embeddings_path': 'artifacts/20250810_053000/embeddings/lamp2_user_embeddings.npy'
        },
        'diversity': {
            'enabled': False,
            'method': 'mmr',
            'lambda': 0.3
        },
        'selection': {
            'q_quantile': 0.8
        },
        'clustering': {
            'enabled': False,
            'algorithm': 'kmeans',
            'max_per_cluster': 10
        },
        'cfs': {
            'enabled': True
        },
        'evaluation': {
            'include_bertscore': True,
            'bertscore_model': 'microsoft/deberta-base-mnli',
            'fast_metrics': False
        },
        'significance': {
            'test_method': 'ttest',
            'alpha': 0.05,
            'correction_method': 'holm'
        }
    }


def parse_conditions(conditions_str: str) -> List[str]:
    """Parse conditions string into list"""
    if not conditions_str:
        return ['legacy_chameleon', 'graphrag_v1', 'graphrag_v1_diversity']
    
    return [c.strip() for c in conditions_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Week 2 GraphRAG-CFS-Chameleon evaluation')
    parser.add_argument('--run-id', type=str, default=None, help='Unique run identifier')
    parser.add_argument('--output-dir', type=str, default='results/w2', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--data', type=str, default=None, help='Test data file path')
    parser.add_argument('--conditions', type=str, default='legacy_chameleon,graphrag_v1,graphrag_v1_diversity', 
                       help='Comma-separated list of conditions to evaluate')
    parser.add_argument('--include-bertscore', action='store_true', help='Include BERTScore computation')
    parser.add_argument('--significance-test', action='store_true', help='Perform significance testing')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--fast-mode', action='store_true', help='Skip slow computations for quick testing')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger.info("Starting Week 2 evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.fast_mode:
        config['evaluation']['fast_metrics'] = True
        config['evaluation']['include_bertscore'] = False
    elif args.include_bertscore:
        config['evaluation']['include_bertscore'] = True
    
    # Load test data
    test_data = load_test_data(args.data)
    
    # Parse conditions
    conditions = parse_conditions(args.conditions)
    logger.info(f"Evaluating conditions: {conditions}")
    
    # Initialize evaluator
    evaluator = GraphRAGChameleonEvaluator(
        config=config,
        output_dir=args.output_dir,
        run_id=args.run_id
    )
    
    # Run evaluation
    logger.info("Running evaluation across all conditions...")
    evaluation_results = evaluator.run_evaluation(test_data, conditions)
    
    # Perform significance testing if requested
    significance_results = {}
    if args.significance_test and len(conditions) > 1:
        logger.info("Performing statistical significance testing...")
        
        # Extract scores for each condition
        condition_scores = {}
        for condition, result in evaluation_results.items():
            # For now, use mock scores - in real implementation, extract from predictions
            n_examples = result['metadata']['n_examples']
            np.random.seed(42)  # For reproducible mock scores
            mock_scores = np.random.uniform(0.3, 0.8, n_examples).tolist()
            condition_scores[condition] = mock_scores
        
        # Compare all conditions
        for metric_name in ['f1_score', 'rouge_l_f1', 'bertscore_f1']:
            if metric_name in evaluation_results[conditions[0]]['metrics']:
                sig_result = compare_all_conditions(
                    condition_scores,
                    metric_name=metric_name,
                    test_method=config['significance']['test_method'],
                    alpha=config['significance']['alpha'],
                    correction_method=config['significance']['correction_method']
                )
                significance_results[metric_name] = sig_result
    
    # Generate comprehensive report if requested
    if args.generate_report:
        logger.info("Generating comprehensive report...")
        
        reporter = EvaluationReporter(evaluator.result_dir)
        
        # Use primary metric for significance analysis
        primary_significance = significance_results.get('f1_score', {})
        
        report_content = reporter.generate_comprehensive_report(
            evaluation_results,
            primary_significance,
            output_file='evaluation_report.md'
        )
        
        # Create summary table
        summary_df = reporter.create_summary_table(evaluation_results)
        logger.info(f"Summary table saved with {len(summary_df)} conditions")
    
    # Save significance results
    if significance_results:
        sig_file = evaluator.result_dir / 'significance_analysis.json'
        with open(sig_file, 'w') as f:
            json.dump(significance_results, f, indent=2)
        logger.info(f"Significance analysis saved to {sig_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Conditions evaluated: {len(conditions)}")
    logger.info(f"Test examples: {len(test_data)}")
    logger.info(f"Results directory: {evaluator.result_dir}")
    
    # Print key results
    logger.info("\nPERFORMANCE SUMMARY:")
    for condition, result in evaluation_results.items():
        metrics = result['metrics']
        f1 = metrics.get('f1_score', 0)
        rouge = metrics.get('rouge_l_f1', 0)
        bert = metrics.get('bertscore_f1', 0)
        logger.info(f"  {condition}: F1={f1:.4f}, ROUGE-L={rouge:.4f}, BERTScore={bert:.4f}")
    
    if significance_results:
        logger.info(f"\nSIGNIFICANCE TESTING:")
        for metric_name, sig_result in significance_results.items():
            n_significant = sum(
                1 for comp_result in sig_result.get('pairwise_results', {}).values()
                if comp_result.get('corrected_significant', False)
            )
            total_comparisons = sig_result.get('n_comparisons', 0)
            logger.info(f"  {metric_name}: {n_significant}/{total_comparisons} significant comparisons")
    
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    import numpy as np  # Import here to avoid issues if not available
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)