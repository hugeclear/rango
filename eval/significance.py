#!/usr/bin/env python3
"""
Statistical significance testing for GraphRAG-CFS-Chameleon evaluation
Implements t-tests and permutation tests for metric comparisons
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


def two_sided_ttest(
    scores_a: List[float], 
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform two-sided t-test between two score distributions
    
    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B  
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length for paired t-test")
    
    # Paired t-test (same test examples)
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Effect size (Cohen's d for paired samples)
    diff = scores_a - scores_b
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    
    # Confidence interval for mean difference
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    ci_lower, ci_upper = stats.t.interval(
        1 - alpha, 
        len(diff) - 1, 
        loc=mean_diff, 
        scale=se_diff
    )
    
    result = {
        'test_type': 'paired_ttest',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'mean_difference': mean_diff,
        'effect_size_cohens_d': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': len(scores_a)
    }
    
    logger.debug(f"T-test: p={p_value:.6f}, effect_size={effect_size:.4f}")
    
    return result


def permutation_test(
    scores_a: List[float],
    scores_b: List[float], 
    n_permutations: int = 10000,
    metric: str = "mean_difference",
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform permutation test for comparing two score distributions
    
    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B
        n_permutations: Number of permutation samples
        metric: Test statistic ("mean_difference", "median_difference")
        alpha: Significance level
        random_state: Random seed
        
    Returns:
        Dictionary with test results
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length for paired permutation test")
    
    np.random.seed(random_state)
    
    # Observed test statistic
    if metric == "mean_difference":
        observed_stat = np.mean(scores_a) - np.mean(scores_b)
    elif metric == "median_difference":
        observed_stat = np.median(scores_a) - np.median(scores_b)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Generate permutation distribution
    perm_stats = []
    
    for _ in range(n_permutations):
        # For paired samples, randomly flip signs of differences
        diff = scores_a - scores_b
        signs = np.random.choice([-1, 1], size=len(diff))
        perm_diff = diff * signs
        
        # Reconstruct permuted scores
        perm_a = scores_b + perm_diff
        perm_b = scores_b
        
        # Compute permuted test statistic
        if metric == "mean_difference":
            perm_stat = np.mean(perm_a) - np.mean(perm_b)
        elif metric == "median_difference":
            perm_stat = np.median(perm_a) - np.median(perm_b)
        
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    
    # Effect size (standardized difference)
    pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
    effect_size = observed_stat / pooled_std if pooled_std > 0 else 0.0
    
    result = {
        'test_type': 'permutation_test',
        'metric': metric,
        'observed_statistic': float(observed_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_permutations': n_permutations,
        'effect_size': effect_size,
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'n_samples': len(scores_a),
        'random_state': random_state
    }
    
    logger.debug(f"Permutation test: p={p_value:.6f}, observed_stat={observed_stat:.4f}")
    
    return result


def bootstrap_confidence_interval(
    scores: List[float],
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000, 
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic
    
    Args:
        scores: Input scores
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed
        
    Returns:
        Lower and upper confidence bounds
    """
    scores = np.array(scores)
    np.random.seed(random_state)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper


def multiple_comparisons_correction(
    p_values: List[float], 
    method: str = "holm",
    alpha: float = 0.05
) -> Dict[str, Union[List[float], List[bool]]]:
    """
    Apply multiple comparisons correction to p-values
    
    Args:
        p_values: List of p-values
        method: Correction method ("holm", "bonferroni", "fdr_bh")
        alpha: Overall significance level
        
    Returns:
        Dictionary with corrected p-values and significance flags
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)
    
    if method == "bonferroni":
        corrected_alpha = alpha / n_tests
        corrected_p_values = p_values * n_tests
        significant = corrected_p_values < alpha
        
    elif method == "holm":
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        corrected_p_values = np.zeros_like(p_values)
        significant = np.zeros(n_tests, dtype=bool)
        
        for i, idx in enumerate(sorted_indices):
            corrected_p = p_values[idx] * (n_tests - i)
            corrected_p_values[idx] = min(corrected_p, 1.0)
            
            # Check significance (sequential)
            if corrected_p < alpha:
                significant[idx] = True
            else:
                # Once we fail to reject, all subsequent tests are non-significant
                break
                
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        sorted_indices = np.argsort(p_values)
        corrected_p_values = np.zeros_like(p_values)
        significant = np.zeros(n_tests, dtype=bool)
        
        for i in reversed(range(n_tests)):
            idx = sorted_indices[i]
            corrected_p = p_values[idx] * n_tests / (i + 1)
            corrected_p_values[idx] = min(corrected_p, 1.0)
            
            if i == n_tests - 1:
                # Highest p-value
                significant[idx] = corrected_p < alpha
            else:
                # Use previous decision
                next_idx = sorted_indices[i + 1]
                significant[idx] = corrected_p < alpha and significant[next_idx]
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    result = {
        'method': method,
        'original_p_values': p_values.tolist(),
        'corrected_p_values': corrected_p_values.tolist(),
        'significant': significant.tolist(),
        'alpha': alpha,
        'n_tests': n_tests
    }
    
    logger.debug(f"Multiple comparisons ({method}): {np.sum(significant)}/{n_tests} significant")
    
    return result


def compare_all_conditions(
    condition_results: Dict[str, List[float]],
    metric_name: str = "f1_score",
    test_method: str = "ttest",
    alpha: float = 0.05,
    correction_method: str = "holm"
) -> Dict[str, Any]:
    """
    Compare all pairs of conditions for statistical significance
    
    Args:
        condition_results: Dictionary mapping condition names to score lists
        metric_name: Name of metric being compared
        test_method: Statistical test ("ttest" or "permutation")
        alpha: Significance level
        correction_method: Multiple comparisons correction method
        
    Returns:
        Dictionary with pairwise comparison results
    """
    conditions = list(condition_results.keys())
    n_conditions = len(conditions)
    
    # Perform all pairwise tests
    pairwise_results = {}
    p_values = []
    comparison_names = []
    
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            cond_a = conditions[i]
            cond_b = conditions[j]
            scores_a = condition_results[cond_a]
            scores_b = condition_results[cond_b]
            
            comparison_name = f"{cond_a}_vs_{cond_b}"
            comparison_names.append(comparison_name)
            
            if test_method == "ttest":
                test_result = two_sided_ttest(scores_a, scores_b, alpha)
            elif test_method == "permutation":
                test_result = permutation_test(scores_a, scores_b, alpha=alpha)
            else:
                raise ValueError(f"Unknown test method: {test_method}")
            
            pairwise_results[comparison_name] = test_result
            p_values.append(test_result['p_value'])
    
    # Apply multiple comparisons correction
    correction_result = multiple_comparisons_correction(
        p_values, 
        method=correction_method, 
        alpha=alpha
    )
    
    # Update significance flags with corrected results
    for i, comparison_name in enumerate(comparison_names):
        pairwise_results[comparison_name]['corrected_significant'] = correction_result['significant'][i]
        pairwise_results[comparison_name]['corrected_p_value'] = correction_result['corrected_p_values'][i]
    
    result = {
        'metric_name': metric_name,
        'test_method': test_method,
        'correction_method': correction_method,
        'alpha': alpha,
        'n_conditions': n_conditions,
        'n_comparisons': len(comparison_names),
        'pairwise_results': pairwise_results,
        'correction_summary': correction_result
    }
    
    # Summary statistics
    n_significant_uncorrected = sum(1 for r in pairwise_results.values() if r['significant'])
    n_significant_corrected = sum(correction_result['significant'])
    
    logger.info(f"Pairwise comparisons for {metric_name}: "
               f"{n_significant_uncorrected}/{len(comparison_names)} significant (uncorrected), "
               f"{n_significant_corrected}/{len(comparison_names)} significant (corrected)")
    
    return result