#!/usr/bin/env python3
"""
Gate threshold sweep tool for systematic tuning of the Chameleon edit gate.

Usage:
python tools/sweep_gate.py --split test --N 100 \
  --alphas "ap=2.75,ag=-1.0" \
  --taus 0.0:0.1:0.005 \
  --out results/analysis/gate_curve.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from chameleon_evaluator import ChameleonEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_alphas(alpha_str: str) -> Dict[str, float]:
    """Parse alpha parameter string like 'ap=2.75,ag=-1.0' into dict."""
    alphas = {}
    for pair in alpha_str.split(','):
        if '=' not in pair:
            continue
        key, val = pair.strip().split('=', 1)
        alphas[key.strip()] = float(val.strip())
    return alphas


def parse_tau_range(tau_str: str) -> List[float]:
    """Parse tau range string like '0.0:0.1:0.005' into list of thresholds."""
    parts = tau_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid tau range format: {tau_str}. Expected 'start:end:step'")
    
    start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
    taus = []
    current = start
    while current <= end:
        taus.append(current)
        current += step
    return taus


def compute_gate_stats(evaluator: ChameleonEvaluator, samples: List[dict], 
                      alpha_personal: float, alpha_general: float, 
                      thresholds: List[float]) -> Dict:
    """Compute gate application rates across different thresholds."""
    
    logger.info(f"Computing gate stats for {len(samples)} samples with ap={alpha_personal}, ag={alpha_general}")
    
    # Compute direction vectors and gate values for all samples
    sample_data = []
    for i, sample in enumerate(tqdm(samples, desc="Computing direction vectors")):
        try:
            prompt = sample.get("prompt") or sample.get("text") or ""
            if not prompt:
                logger.warning(f"Sample {i} has no prompt, skipping")
                continue
                
            # Compute direction vectors
            dv = evaluator._compute_direction_vectors_strict({"prompt": prompt})
            hidden_norm = max(1e-8, float(dv.get('l2_general', 0.0)))
            
            # Compute gate info for this sample
            gate_info = evaluator.compute_gate(
                hidden_norm,
                float(dv.get('l2_personal', 0.0)),
                float(dv.get('l2_general', 0.0)),
                alpha_personal,
                alpha_general,
                0.0  # We'll test different thresholds
            )
            
            sample_data.append({
                "sample_id": i,
                "gate_value": gate_info.get("gate", 0.0),
                "l2_personal": float(dv.get('l2_personal', 0.0)),
                "l2_general": float(dv.get('l2_general', 0.0)),
                "hidden_norm": hidden_norm,
                "cos_theta": float(dv.get('cos_theta', 0.0)),
            })
            
        except Exception as e:
            logger.warning(f"Failed to process sample {i}: {e}")
            continue
    
    if not sample_data:
        raise RuntimeError("No valid samples found")
    
    logger.info(f"Successfully processed {len(sample_data)} samples")
    
    # Compute application rates for different thresholds
    gate_values = np.array([s["gate_value"] for s in sample_data])
    threshold_stats = []
    
    for tau in thresholds:
        applied = (gate_values >= tau).sum()
        rate = applied / len(gate_values)
        
        threshold_stats.append({
            "threshold": tau,
            "applied_count": int(applied),
            "total_count": len(gate_values),
            "application_rate": float(rate),
        })
    
    return {
        "alpha_personal": alpha_personal,
        "alpha_general": alpha_general,
        "sample_count": len(sample_data),
        "gate_stats": {
            "min": float(gate_values.min()),
            "max": float(gate_values.max()),
            "mean": float(gate_values.mean()),
            "std": float(gate_values.std()),
            "median": float(np.median(gate_values)),
            "q25": float(np.percentile(gate_values, 25)),
            "q75": float(np.percentile(gate_values, 75)),
        },
        "threshold_sweep": threshold_stats,
        "sample_data": sample_data,
    }


def find_optimal_threshold(threshold_stats: List[dict], target_rate: float = 0.3) -> Tuple[float, float]:
    """Find threshold closest to target application rate."""
    best_tau = None
    best_rate = None
    min_diff = float('inf')
    
    for stat in threshold_stats:
        diff = abs(stat["application_rate"] - target_rate)
        if diff < min_diff:
            min_diff = diff
            best_tau = stat["threshold"]
            best_rate = stat["application_rate"]
    
    return best_tau, best_rate


def pick_tau_for_rate(curve: List[Dict], target: float = 0.3) -> float:
    """
    Automated τ selection for target application rate.
    
    Args:
        curve: List of {"threshold": float, "application_rate": float, ...} dicts
        target: Target application rate (default: 0.3 for 30%)
        
    Returns:
        Optimal threshold value
    """
    if not curve:
        return 0.0
        
    return min(curve, key=lambda x: abs(x["application_rate"] - target))["threshold"]


def generate_experiment_report(results: Dict, target_rate: float = 0.3) -> str:
    """
    Generate one-line experiment report for reproducible tuning.
    Format: (ap, ag, τ, r, Δacc, p値)
    """
    ap = results["alpha_personal"]
    ag = results["alpha_general"]
    tau = results["optimal_threshold"]
    r = results["optimal_rate"]
    
    # Placeholder for accuracy and p-value (would come from actual evaluation)
    delta_acc = "N/A"  # Would be calculated from actual LaMP-2 evaluation
    p_value = "N/A"    # Would come from statistical significance test
    
    return f"({ap:.2f}, {ag:.2f}, {tau:.4f}, {r:.3f}, {delta_acc}, {p_value})"


def main():
    parser = argparse.ArgumentParser(description="Gate threshold sweep for Chameleon editing")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--N", type=int, default=100, help="Number of samples to analyze")
    parser.add_argument("--alphas", required=True, help="Alpha parameters like 'ap=2.75,ag=-1.0'")
    parser.add_argument("--taus", required=True, help="Threshold range like '0.0:0.1:0.005'")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--config", help="Optional config file path")
    parser.add_argument("--target-rate", type=float, default=0.3, help="Target application rate for optimal threshold")
    
    args = parser.parse_args()
    
    # Parse parameters
    try:
        alphas = parse_alphas(args.alphas)
        alpha_personal = alphas.get("ap", alphas.get("alpha_personal", 2.75))
        alpha_general = alphas.get("ag", alphas.get("alpha_general", -1.0))
        thresholds = parse_tau_range(args.taus)
    except Exception as e:
        logger.error(f"Parameter parsing error: {e}")
        return 1
    
    logger.info(f"Alpha parameters: ap={alpha_personal}, ag={alpha_general}")
    logger.info(f"Testing {len(thresholds)} thresholds from {thresholds[0]} to {thresholds[-1]}")
    
    # Initialize evaluator
    try:
        evaluator = ChameleonEvaluator()
        logger.info("ChameleonEvaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Load test samples (mock data for now - replace with actual dataset loading)
    # This should be replaced with actual LaMP-2 dataset loading
    samples = []
    for i in range(args.N):
        # Mock sample - replace with real LaMP-2 loading
        samples.append({
            "prompt": f"User's movie preferences: action, thriller\nMovie: Sample movie description {i}\n\nClassify this movie:",
            "id": i,
        })
    
    logger.info(f"Loaded {len(samples)} test samples")
    
    # Compute gate statistics
    try:
        start_time = time.time()
        results = compute_gate_stats(evaluator, samples, alpha_personal, alpha_general, thresholds)
        elapsed = time.time() - start_time
        
        # Find optimal threshold
        optimal_tau, optimal_rate = find_optimal_threshold(results["threshold_sweep"], args.target_rate)
        
        results.update({
            "optimal_threshold": optimal_tau,
            "optimal_rate": optimal_rate,
            "target_rate": args.target_rate,
            "computation_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Generate experiment report
        experiment_report = generate_experiment_report(results, args.target_rate)
        results["experiment_report"] = experiment_report
        
        logger.info(f"Optimal threshold: {optimal_tau:.4f} (rate: {optimal_rate:.3f}, target: {args.target_rate:.3f})")
        logger.info(f"Experiment report: {experiment_report}")
        
    except Exception as e:
        logger.error(f"Gate analysis failed: {e}")
        return 1
    
    # Save results
    try:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print(f"\n=== Gate Threshold Sweep Results ===")
        print(f"Samples analyzed: {results['sample_count']}")
        print(f"Alpha parameters: ap={alpha_personal}, ag={alpha_general}")
        print(f"Gate value range: {results['gate_stats']['min']:.4f} - {results['gate_stats']['max']:.4f}")
        print(f"Gate value mean±std: {results['gate_stats']['mean']:.4f}±{results['gate_stats']['std']:.4f}")
        print(f"Optimal threshold: {optimal_tau:.4f} (rate: {optimal_rate:.3f})")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())