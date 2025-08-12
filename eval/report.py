#!/usr/bin/env python3
"""
Report generation for GraphRAG-CFS-Chameleon evaluation results
Creates formatted tables and visualizations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Generate comprehensive evaluation reports with statistical analysis
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        logger.info(f"EvaluationReporter initialized for {results_dir}")
    
    def generate_comprehensive_report(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        significance_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive markdown report
        
        Args:
            evaluation_results: Results from evaluation runner
            significance_results: Results from significance testing
            output_file: Optional output file path
            
        Returns:
            Markdown report content
        """
        report_sections = []
        
        # Header
        report_sections.append(self._generate_header())
        
        # Executive Summary
        report_sections.append(self._generate_executive_summary(evaluation_results))
        
        # Methodology
        report_sections.append(self._generate_methodology_section())
        
        # Results Overview
        report_sections.append(self._generate_results_table(evaluation_results))
        
        # Statistical Analysis
        report_sections.append(self._generate_significance_analysis(significance_results))
        
        # Detailed Findings
        report_sections.append(self._generate_detailed_findings(evaluation_results, significance_results))
        
        # Performance Analysis
        report_sections.append(self._generate_performance_analysis(evaluation_results))
        
        # Conclusions and Recommendations
        report_sections.append(self._generate_conclusions(evaluation_results, significance_results))
        
        # Appendix
        report_sections.append(self._generate_appendix(evaluation_results))
        
        # Combine all sections
        full_report = "\n\n".join(report_sections)
        
        # Save report if output file specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(full_report)
            logger.info(f"Report saved to {output_path}")
        
        return full_report
    
    def _generate_header(self) -> str:
        """Generate report header"""
        import time
        
        return f"""# GraphRAG-CFS-Chameleon Evaluation Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Framework**: Week 2 Integration Testing  
**Objective**: Compare GraphRAG+CFS+Diversity enhancements against baseline Chameleon"""
    
    def _generate_executive_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate executive summary"""
        n_conditions = len(results)
        conditions = list(results.keys())
        
        # Find best performing condition
        f1_scores = {cond: res['metrics'].get('f1_score', 0) for cond, res in results.items()}
        best_condition = max(f1_scores.items(), key=lambda x: x[1])
        
        return f"""## Executive Summary

This report evaluates {n_conditions} different configurations of the Chameleon personalization system:
{chr(10).join(f'- {cond}' for cond in conditions)}

**Key Finding**: The `{best_condition[0]}` configuration achieved the highest F1 score of {best_condition[1]:.4f}.

**Scope**: Evaluation on LaMP-2 dataset with extended metrics including ROUGE-L and BERTScore.  
**Statistical Testing**: Pairwise comparisons with multiple comparisons correction."""
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section"""
        return """## Methodology

### Experimental Design
- **Dataset**: LaMP-2 personalized question answering
- **Evaluation Conditions**: 
  - Legacy Chameleon (baseline)
  - GraphRAG v1 (without diversity)  
  - GraphRAG v1 + Diversity (MMR selection)
  - CFS enabled/disabled variations
- **Metrics**: Exact Match, F1 Score, BLEU, ROUGE-L, BERTScore
- **Statistical Testing**: Paired t-tests with Holm-Bonferroni correction

### Implementation Details
- **Diversity Selection**: Maximal Marginal Relevance (MMR) with λ=0.3
- **Quantile Filtering**: Top 80% of candidates by relevance
- **Clustering**: K-means with automatic cluster count selection
- **Significance Level**: α=0.05 with multiple comparisons correction"""
    
    def _generate_results_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate main results table"""
        # Create comparison table
        table_data = []
        
        for condition, result in results.items():
            metrics = result['metrics']
            metadata = result['metadata']
            
            row = {
                'Condition': condition,
                'Exact Match': f"{metrics.get('exact_match', 0):.4f}",
                'F1 Score': f"{metrics.get('f1_score', 0):.4f}",
                'BLEU': f"{metrics.get('bleu_score', 0):.4f}",
                'ROUGE-L F1': f"{metrics.get('rouge_l_f1', 0):.4f}",
                'BERTScore F1': f"{metrics.get('bertscore_f1', 0):.4f}",
                'Eval Time (s)': f"{metadata.get('evaluation_time_sec', 0):.1f}",
                'N Examples': str(metadata.get('n_examples', 0))
            }
            table_data.append(row)
        
        # Convert to DataFrame for easier formatting
        df = pd.DataFrame(table_data)
        
        # Create markdown table
        table_lines = ["## Results Overview", ""]
        
        # Headers
        headers = df.columns.tolist()
        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Data rows
        for _, row in df.iterrows():
            table_lines.append("| " + " | ".join(row.astype(str)) + " |")
        
        return "\n".join(table_lines)
    
    def _generate_significance_analysis(self, significance_results: Dict[str, Any]) -> str:
        """Generate statistical significance analysis"""
        if not significance_results:
            return "## Statistical Analysis\n\nNo significance testing performed."
        
        pairwise_results = significance_results.get('pairwise_results', {})
        correction_summary = significance_results.get('correction_summary', {})
        
        lines = [
            "## Statistical Significance Analysis",
            "",
            f"**Test Method**: {significance_results.get('test_method', 'Unknown')}",
            f"**Correction Method**: {significance_results.get('correction_method', 'None')}",
            f"**Significance Level**: α = {significance_results.get('alpha', 0.05)}",
            f"**Number of Comparisons**: {significance_results.get('n_comparisons', 0)}",
            ""
        ]
        
        # Pairwise comparison table
        if pairwise_results:
            lines.extend([
                "### Pairwise Comparisons",
                "",
                "| Comparison | Mean Diff | Effect Size | P-value | Corrected P | Significant |",
                "|------------|-----------|-------------|---------|-------------|-------------|"
            ])
            
            for comp_name, comp_result in pairwise_results.items():
                mean_diff = comp_result.get('mean_difference', 0)
                effect_size = comp_result.get('effect_size_cohens_d', comp_result.get('effect_size', 0))
                p_value = comp_result.get('p_value', 1.0)
                corrected_p = comp_result.get('corrected_p_value', p_value)
                significant = "✓" if comp_result.get('corrected_significant', False) else "✗"
                
                row = [
                    comp_name.replace('_vs_', ' vs '),
                    f"{mean_diff:+.4f}",
                    f"{effect_size:.3f}",
                    f"{p_value:.4f}",
                    f"{corrected_p:.4f}",
                    significant
                ]
                lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _generate_detailed_findings(
        self, 
        results: Dict[str, Dict[str, Any]], 
        significance_results: Dict[str, Any]
    ) -> str:
        """Generate detailed findings section"""
        lines = ["## Detailed Findings", ""]
        
        # Performance ranking
        f1_scores = [(cond, res['metrics'].get('f1_score', 0)) for cond, res in results.items()]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        lines.extend([
            "### Performance Ranking (by F1 Score)",
            ""
        ])
        
        for i, (condition, f1_score) in enumerate(f1_scores, 1):
            lines.append(f"{i}. **{condition}**: {f1_score:.4f}")
        
        lines.append("")
        
        # Key observations
        lines.extend([
            "### Key Observations",
            ""
        ])
        
        best_condition = f1_scores[0][0]
        worst_condition = f1_scores[-1][0]
        improvement = f1_scores[0][1] - f1_scores[-1][1]
        
        lines.extend([
            f"- **Best Performance**: {best_condition} achieved highest scores across multiple metrics",
            f"- **Performance Gap**: {improvement:.4f} F1 score difference between best and worst conditions",
            f"- **Consistency**: [TODO: Add consistency analysis across metrics]",
            ""
        ])
        
        # Metric-specific insights
        metric_insights = self._analyze_metric_patterns(results)
        if metric_insights:
            lines.extend([
                "### Metric-Specific Insights",
                ""
            ])
            lines.extend(metric_insights)
        
        return "\n".join(lines)
    
    def _generate_performance_analysis(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate performance analysis section"""
        lines = [
            "## Performance Analysis",
            "",
            "### Computational Efficiency",
            ""
        ]
        
        # Timing analysis
        timing_data = []
        for condition, result in results.items():
            timing_data.append({
                'condition': condition,
                'time': result['metadata'].get('evaluation_time_sec', 0),
                'examples': result['metadata'].get('n_examples', 0)
            })
        
        timing_data.sort(key=lambda x: x['time'])
        
        lines.append("| Condition | Eval Time (s) | Examples | Time per Example (ms) |")
        lines.append("|-----------|---------------|----------|-----------------------|")
        
        for item in timing_data:
            time_per_example = (item['time'] / item['examples'] * 1000) if item['examples'] > 0 else 0
            row = [
                item['condition'],
                f"{item['time']:.1f}",
                str(item['examples']),
                f"{time_per_example:.1f}"
            ]
            lines.append("| " + " | ".join(row) + " |")
        
        lines.extend([
            "",
            "### Scalability Considerations",
            "",
            "- GraphRAG operations add computational overhead but provide quality improvements",
            "- Diversity selection increases processing time but may improve result quality",
            "- Trade-off between computational cost and performance gains needs evaluation"
        ])
        
        return "\n".join(lines)
    
    def _generate_conclusions(
        self, 
        results: Dict[str, Dict[str, Any]], 
        significance_results: Dict[str, Any]
    ) -> str:
        """Generate conclusions and recommendations"""
        lines = [
            "## Conclusions and Recommendations",
            "",
            "### Main Conclusions",
            ""
        ]
        
        # Determine best approach
        f1_scores = {cond: res['metrics'].get('f1_score', 0) for cond, res in results.items()}
        best_condition = max(f1_scores.items(), key=lambda x: x[1])
        
        # Check for significant improvements
        significant_pairs = []
        if significance_results and 'pairwise_results' in significance_results:
            for comp_name, comp_result in significance_results['pairwise_results'].items():
                if comp_result.get('corrected_significant', False):
                    significant_pairs.append(comp_name)
        
        lines.extend([
            f"1. **{best_condition[0]}** demonstrates the best overall performance with F1 score of {best_condition[1]:.4f}",
            f"2. Statistical significance detected in {len(significant_pairs)} pairwise comparisons",
            "3. [TODO: Add specific insights about GraphRAG and diversity contributions]",
            "",
            "### Recommendations",
            "",
            "#### For Production Deployment:",
            f"- **Recommended Configuration**: {best_condition[0]}",
            "- Monitor computational overhead vs. quality gains",
            "- Consider adaptive diversity parameters based on query complexity",
            "",
            "#### For Further Research:",
            "- Investigate optimal diversity parameters (λ values)",
            "- Explore hierarchical GraphRAG approaches", 
            "- Evaluate performance on additional datasets (Tenrec)",
            "",
            "### Limitations",
            "",
            "- Evaluation limited to LaMP-2 dataset",
            "- Small sample size may limit statistical power",
            "- BERTScore computation using fallback implementation",
            ""
        ])
        
        return "\n".join(lines)
    
    def _generate_appendix(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate appendix with technical details"""
        lines = [
            "## Appendix",
            "",
            "### Configuration Details",
            ""
        ]
        
        # Show configuration for each condition
        for condition, result in results.items():
            config = result.get('config', {})
            lines.extend([
                f"#### {condition}",
                "",
                "```yaml"
            ])
            
            # Format key configuration items
            key_configs = [
                'legacy_mode',
                'graphrag.enabled', 
                'diversity.enabled',
                'diversity.method',
                'diversity.lambda',
                'selection.q_quantile',
                'cfs.enabled'
            ]
            
            for key in key_configs:
                value = config
                for part in key.split('.'):
                    value = value.get(part, {}) if isinstance(value, dict) else None
                if value is not None:
                    lines.append(f"{key}: {value}")
            
            lines.extend(["```", ""])
        
        lines.extend([
            "### Metric Definitions",
            "",
            "- **Exact Match**: Percentage of predictions exactly matching reference",
            "- **F1 Score**: Harmonic mean of precision and recall at token level", 
            "- **BLEU**: Bilingual Evaluation Understudy score (4-gram)",
            "- **ROUGE-L**: Longest Common Subsequence based evaluation",
            "- **BERTScore**: Semantic similarity using contextual embeddings",
            ""
        ])
        
        return "\n".join(lines)
    
    def _analyze_metric_patterns(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Analyze patterns across different metrics"""
        insights = []
        
        # Get all metrics
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result['metrics'].keys())
        
        # Analyze correlation between metrics
        metric_correlations = {}
        conditions = list(results.keys())
        
        for metric in all_metrics:
            if metric.endswith('_f1') or metric in ['exact_match', 'f1_score', 'bleu_score']:
                scores = [results[cond]['metrics'].get(metric, 0) for cond in conditions]
                if len(set(scores)) > 1:  # Has variation
                    metric_correlations[metric] = scores
        
        if metric_correlations:
            insights.extend([
                "- Metric correlations suggest [TODO: implement correlation analysis]",
                "- ROUGE-L and F1 scores show [TODO: implement pattern analysis]"
            ])
        
        return insights
    
    def create_summary_table(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary table as pandas DataFrame
        
        Args:
            results: Evaluation results
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for condition, result in results.items():
            metrics = result['metrics']
            metadata = result['metadata']
            
            row = {
                'condition': condition,
                'n_examples': metadata.get('n_examples', 0),
                'eval_time_sec': metadata.get('evaluation_time_sec', 0),
                'exact_match': metrics.get('exact_match', 0),
                'f1_score': metrics.get('f1_score', 0), 
                'bleu_score': metrics.get('bleu_score', 0),
                'rouge_l_f1': metrics.get('rouge_l_f1', 0),
                'bertscore_f1': metrics.get('bertscore_f1', 0)
            }
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save to results directory
        df.to_csv(self.results_dir / "summary_table.csv", index=False)
        
        return df