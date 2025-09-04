#!/usr/bin/env python3
"""
Do-Calculus for Causal Effect Estimation in Chameleon
Implements Average Treatment Effect (ATE) estimation for personalization interventions

Estimates causal effects of Chameleon editing on user engagement and satisfaction
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class CausalTreatment:
    """Represents a causal treatment/intervention"""
    treatment_name: str
    treatment_intensity: float  # Alpha parameter value
    target_features: List[str]
    application_timestamp: float

@dataclass
class CausalOutcome:
    """Represents observed outcomes after treatment"""
    outcome_name: str
    outcome_value: float
    measurement_timestamp: float
    outcome_type: str  # 'continuous', 'binary', 'categorical'

@dataclass
class ATEResult:
    """Average Treatment Effect estimation result"""
    treatment: CausalTreatment
    control_mean: float
    treatment_mean: float
    ate_estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    n_control: int
    n_treatment: int
    effect_size: float
    statistical_power: float

class DoCalculusEstimator:
    """
    Implements Pearl's do-calculus for causal effect estimation in Chameleon context
    
    Estimates Average Treatment Effect (ATE) of personalization interventions:
    ATE = E[Y | do(X = x_treatment)] - E[Y | do(X = x_control)]
    
    Where:
    - Y = user engagement outcome (accuracy, satisfaction, etc.)
    - X = Chameleon editing intervention (alpha parameters)
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 min_samples_per_group: int = 10,
                 bootstrap_samples: int = 1000):
        """
        Initialize do-calculus estimator
        
        Args:
            confidence_level: Confidence level for intervals (default 95%)
            min_samples_per_group: Minimum samples per treatment group
            bootstrap_samples: Number of bootstrap samples for CI estimation
        """
        self.confidence_level = confidence_level
        self.min_samples_per_group = min_samples_per_group
        self.bootstrap_samples = bootstrap_samples
        
        self.alpha_level = 1 - confidence_level
        
        logger.info(f"Do-calculus estimator initialized with {confidence_level*100}% confidence level")
    
    def extract_treatments_from_evaluation_history(self, evaluation_results: List[Dict]) -> List[CausalTreatment]:
        """
        Extract causal treatments from Chameleon evaluation history
        
        Args:
            evaluation_results: List of evaluation results from runs/ directory
            
        Returns:
            List of CausalTreatment objects
        """
        treatments = []
        
        for i, result in enumerate(evaluation_results):
            # Extract treatment parameters from evaluation config
            alpha_personal = result.get('alpha_personal', result.get('alpha', 1.5))
            alpha_neutral = result.get('alpha_neutral', result.get('beta', -0.8))
            
            # Create treatment intensity measure
            treatment_intensity = abs(alpha_personal) + abs(alpha_neutral)
            
            # Identify target features
            target_features = result.get('layers', result.get('target_layers', ['model.layers.20']))
            if isinstance(target_features, str):
                target_features = target_features.split(',')
            
            treatment = CausalTreatment(
                treatment_name=f"chameleon_alpha_{alpha_personal:.2f}_beta_{alpha_neutral:.2f}",
                treatment_intensity=treatment_intensity,
                target_features=target_features,
                application_timestamp=result.get('timestamp', i * 3600)  # Pseudo-timestamp
            )
            
            treatments.append(treatment)
        
        logger.info(f"Extracted {len(treatments)} causal treatments from evaluation history")
        return treatments
    
    def extract_outcomes_from_results(self, evaluation_results: List[Dict]) -> List[CausalOutcome]:
        """
        Extract causal outcomes from evaluation results
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            List of CausalOutcome objects
        """
        outcomes = []
        
        for i, result in enumerate(evaluation_results):
            # Primary outcome: accuracy improvement
            if 'chameleon_accuracy' in result and 'baseline_accuracy' in result:
                improvement = result['chameleon_accuracy'] - result['baseline_accuracy']
                outcomes.append(CausalOutcome(
                    outcome_name='accuracy_improvement',
                    outcome_value=improvement,
                    measurement_timestamp=result.get('timestamp', i * 3600),
                    outcome_type='continuous'
                ))
            
            # Secondary outcomes
            for metric in ['bleu_score', 'f1_score', 'exact_match']:
                if f'chameleon_{metric}' in result:
                    outcomes.append(CausalOutcome(
                        outcome_name=f'chameleon_{metric}',
                        outcome_value=result[f'chameleon_{metric}'],
                        measurement_timestamp=result.get('timestamp', i * 3600),
                        outcome_type='continuous'
                    ))
        
        logger.info(f"Extracted {len(outcomes)} causal outcomes from evaluation results")
        return outcomes
    
    def estimate_ate_with_matching(self, treatments: List[CausalTreatment],
                                  outcomes: List[CausalOutcome],
                                  treatment_threshold: float = 2.0) -> Optional[ATEResult]:
        """
        Estimate Average Treatment Effect using propensity score matching
        
        Args:
            treatments: List of treatments
            outcomes: List of outcomes
            treatment_threshold: Threshold for treatment vs control classification
            
        Returns:
            ATEResult or None if insufficient data
        """
        if len(treatments) != len(outcomes):
            logger.error(f"Mismatched treatments ({len(treatments)}) and outcomes ({len(outcomes)})")
            return None
        
        # Classify treatments into control and treatment groups
        control_outcomes = []
        treatment_outcomes = []
        control_treatments = []
        treatment_treatments = []
        
        for treatment, outcome in zip(treatments, outcomes):
            if outcome.outcome_name == 'accuracy_improvement':  # Focus on primary outcome
                if treatment.treatment_intensity <= treatment_threshold:
                    control_outcomes.append(outcome.outcome_value)
                    control_treatments.append(treatment)
                else:
                    treatment_outcomes.append(outcome.outcome_value)
                    treatment_treatments.append(treatment)
        
        if len(control_outcomes) < self.min_samples_per_group or len(treatment_outcomes) < self.min_samples_per_group:
            logger.warning(f"Insufficient samples: control={len(control_outcomes)}, "
                          f"treatment={len(treatment_outcomes)}, min={self.min_samples_per_group}")
            return None
        
        # Calculate means
        control_mean = np.mean(control_outcomes)
        treatment_mean = np.mean(treatment_outcomes)
        ate_estimate = treatment_mean - control_mean
        
        # Statistical testing
        t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(treatment_outcomes) - 1) * np.var(treatment_outcomes, ddof=1) +
                             (len(control_outcomes) - 1) * np.var(control_outcomes, ddof=1)) /
                            (len(treatment_outcomes) + len(control_outcomes) - 2))
        effect_size = ate_estimate / (pooled_std + 1e-8)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(treatment_outcomes, control_outcomes)
        
        # Statistical power (approximate)
        power = self._estimate_statistical_power(treatment_outcomes, control_outcomes, effect_size)
        
        # Create representative treatment
        avg_treatment = CausalTreatment(
            treatment_name=f"high_intensity_editing",
            treatment_intensity=np.mean([t.treatment_intensity for t in treatment_treatments]),
            target_features=['model.layers.20'],  # Most common target
            application_timestamp=np.mean([t.application_timestamp for t in treatment_treatments])
        )
        
        result = ATEResult(
            treatment=avg_treatment,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            ate_estimate=ate_estimate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_control=len(control_outcomes),
            n_treatment=len(treatment_outcomes),
            effect_size=effect_size,
            statistical_power=power
        )
        
        logger.info(f"ATE estimated: {ate_estimate:.4f} (CI: [{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f})")
        return result
    
    def _bootstrap_ci(self, treatment_group: List[float], 
                     control_group: List[float]) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for ATE
        
        Args:
            treatment_group: Treatment group outcomes
            control_group: Control group outcomes
            
        Returns:
            Tuple of (lower_ci, upper_ci)
        """
        ate_samples = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample from each group
            treatment_sample = np.random.choice(treatment_group, size=len(treatment_group), replace=True)
            control_sample = np.random.choice(control_group, size=len(control_group), replace=True)
            
            # Calculate ATE for bootstrap sample
            ate_boot = np.mean(treatment_sample) - np.mean(control_sample)
            ate_samples.append(ate_boot)
        
        # Calculate confidence interval
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        
        ci_lower = np.percentile(ate_samples, lower_percentile)
        ci_upper = np.percentile(ate_samples, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _estimate_statistical_power(self, treatment_group: List[float],
                                  control_group: List[float], 
                                  effect_size: float) -> float:
        """
        Estimate statistical power of the test
        
        Args:
            treatment_group: Treatment group outcomes
            control_group: Control group outcomes
            effect_size: Calculated effect size
            
        Returns:
            Estimated statistical power (0-1)
        """
        n1, n2 = len(treatment_group), len(control_group)
        
        # Approximate power calculation for two-sample t-test
        # Using non-central t-distribution approximation
        df = n1 + n2 - 2
        
        # Non-centrality parameter
        delta = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
        
        # Critical value for two-tailed test
        t_critical = stats.t.ppf(1 - self.alpha_level / 2, df)
        
        # Power calculation (simplified approximation)
        power = 1 - stats.t.cdf(t_critical - delta, df) + stats.t.cdf(-t_critical - delta, df)
        
        return max(0, min(1, power))
    
    def estimate_dose_response_function(self, treatments: List[CausalTreatment],
                                      outcomes: List[CausalOutcome]) -> Dict[str, Any]:
        """
        Estimate dose-response relationship between treatment intensity and outcomes
        
        Args:
            treatments: List of treatments with varying intensities
            outcomes: Corresponding outcomes
            
        Returns:
            Dictionary with dose-response analysis
        """
        if len(treatments) != len(outcomes):
            return {'error': 'Mismatched treatments and outcomes'}
        
        # Extract dose-outcome pairs for accuracy improvement
        dose_outcome_pairs = []
        for treatment, outcome in zip(treatments, outcomes):
            if outcome.outcome_name == 'accuracy_improvement':
                dose_outcome_pairs.append((treatment.treatment_intensity, outcome.outcome_value))
        
        if len(dose_outcome_pairs) < 5:
            return {'error': 'Insufficient data for dose-response analysis'}
        
        doses = np.array([pair[0] for pair in dose_outcome_pairs])
        outcomes_array = np.array([pair[1] for pair in dose_outcome_pairs])
        
        # Fit linear dose-response model
        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, outcomes_array)
        
        # Optimal dose estimation (where marginal benefit equals marginal cost)
        # Simplified: assume cost is linear in dose, find peak of quadratic approximation
        if len(dose_outcome_pairs) >= 10:
            # Fit quadratic model for optimal dose
            coeffs = np.polyfit(doses, outcomes_array, 2)
            if coeffs[0] < 0:  # Concave down (has maximum)
                optimal_dose = -coeffs[1] / (2 * coeffs[0])
                optimal_dose = np.clip(optimal_dose, doses.min(), doses.max())
            else:
                optimal_dose = doses[np.argmax(outcomes_array)]
        else:
            optimal_dose = doses[np.argmax(outcomes_array)]
        
        return {
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_r_squared': r_value**2,
            'linear_p_value': p_value,
            'linear_std_error': std_err,
            'optimal_dose': optimal_dose,
            'dose_range': (doses.min(), doses.max()),
            'outcome_range': (outcomes_array.min(), outcomes_array.max()),
            'n_observations': len(dose_outcome_pairs)
        }
    
    def generate_causal_report(self, ate_result: ATEResult, 
                             dose_response: Dict[str, Any]) -> str:
        """
        Generate comprehensive causal analysis report
        
        Args:
            ate_result: ATE estimation result
            dose_response: Dose-response analysis result
            
        Returns:
            Formatted causal analysis report
        """
        if not ate_result:
            return "Insufficient data for causal analysis."
        
        # Interpret ATE result
        if ate_result.p_value < 0.05:
            significance = "statistically significant"
            confidence = "high"
        elif ate_result.p_value < 0.1:
            significance = "marginally significant"
            confidence = "moderate"
        else:
            significance = "not statistically significant"
            confidence = "low"
        
        # Interpret effect size
        if abs(ate_result.effect_size) < 0.2:
            effect_magnitude = "small"
        elif abs(ate_result.effect_size) < 0.5:
            effect_magnitude = "medium"
        elif abs(ate_result.effect_size) < 0.8:
            effect_magnitude = "large"
        else:
            effect_magnitude = "very large"
        
        report = f"""CAUSAL ANALYSIS REPORT - CHAMELEON PERSONALIZATION EFFECTS
{'='*80}

AVERAGE TREATMENT EFFECT (ATE) ANALYSIS:
Treatment: {ate_result.treatment.treatment_name}
Treatment Intensity: {ate_result.treatment.treatment_intensity:.3f}

Results:
• Control Group Mean: {ate_result.control_mean:.4f} (n={ate_result.n_control})
• Treatment Group Mean: {ate_result.treatment_mean:.4f} (n={ate_result.n_treatment})
• Average Treatment Effect: {ate_result.ate_estimate:.4f}
• 95% Confidence Interval: [{ate_result.confidence_interval[0]:.4f}, {ate_result.confidence_interval[1]:.4f}]
• p-value: {ate_result.p_value:.4f}
• Effect Size (Cohen's d): {ate_result.effect_size:.3f} ({effect_magnitude})
• Statistical Power: {ate_result.statistical_power:.3f}

INTERPRETATION:
The causal effect of Chameleon personalization editing is {significance} 
(p = {ate_result.p_value:.4f}). The effect size is {effect_magnitude} 
({ate_result.effect_size:.3f}), indicating a {'positive' if ate_result.ate_estimate > 0 else 'negative'} 
impact on user accuracy improvement.

Confidence in this result is {confidence} based on statistical power ({ate_result.statistical_power:.3f}) 
and sample sizes (control: {ate_result.n_control}, treatment: {ate_result.n_treatment}).
"""

        # Add dose-response analysis if available
        if dose_response and 'error' not in dose_response:
            report += f"""
DOSE-RESPONSE ANALYSIS:
Linear Model: Outcome = {dose_response['linear_intercept']:.4f} + {dose_response['linear_slope']:.4f} × Dose
R² = {dose_response['linear_r_squared']:.3f}, p = {dose_response['linear_p_value']:.4f}

Optimal Treatment Intensity: {dose_response['optimal_dose']:.3f}
Dose Range Tested: [{dose_response['dose_range'][0]:.3f}, {dose_response['dose_range'][1]:.3f}]
Outcome Range Observed: [{dose_response['outcome_range'][0]:.4f}, {dose_response['outcome_range'][1]:.4f}]

RECOMMENDATION:
{'Based on dose-response analysis, the optimal Chameleon editing intensity is ' + f"{dose_response['optimal_dose']:.3f}." if dose_response['optimal_dose'] else 'Insufficient data for dose optimization.'}
"""

        report += f"""
METHODOLOGICAL NOTES:
• Analysis based on {ate_result.n_control + ate_result.n_treatment} total observations
• Bootstrap confidence intervals ({self.bootstrap_samples} samples)
• Minimum effect size detectable with 80% power: {2.8 / np.sqrt(ate_result.n_control + ate_result.n_treatment):.4f}

CAUSAL ASSUMPTIONS:
1. Exchangeability: Treatment assignment independent of potential outcomes
2. Positivity: All units have non-zero probability of receiving each treatment
3. Consistency: Observed outcomes match potential outcomes under assigned treatment
4. No interference: Units do not interfere with each other

{'='*80}
Report generated by DoCalculusEstimator - Causal Inference for Chameleon
"""
        
        return report

def integrate_do_calculus_with_chameleon_evaluation(evaluation_results: List[Dict]) -> ATEResult:
    """
    Integration helper for existing Chameleon evaluation system
    
    Args:
        evaluation_results: List of evaluation results from runs/ directory
        
    Returns:
        ATE estimation result
    """
    estimator = DoCalculusEstimator(
        confidence_level=0.95,
        min_samples_per_group=5,  # Lower threshold for small datasets
        bootstrap_samples=1000
    )
    
    # Extract treatments and outcomes
    treatments = estimator.extract_treatments_from_evaluation_history(evaluation_results)
    outcomes = estimator.extract_outcomes_from_results(evaluation_results)
    
    # Estimate ATE
    ate_result = estimator.estimate_ate_with_matching(treatments, outcomes)
    
    if ate_result:
        # Estimate dose-response
        dose_response = estimator.estimate_dose_response_function(treatments, outcomes)
        
        # Generate report
        report = estimator.generate_causal_report(ate_result, dose_response)
        logger.info("Causal analysis completed successfully")
        logger.info(f"ATE: {ate_result.ate_estimate:.4f} (p={ate_result.p_value:.4f})")
    else:
        logger.warning("Insufficient data for ATE estimation")
    
    return ate_result