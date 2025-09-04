#!/usr/bin/env python3
"""
Parameter grid search for Chameleon effectiveness.

Usage:
    python tools/parameter_grid_search.py --data_path data --limit 50 --strict --output results/grid_search_results.csv

Exit codes:
    0: Grid search completed successfully
    1: Grid search failed
    2: Invalid arguments
"""

import argparse
import subprocess
import json
import csv
import sys
from pathlib import Path
from itertools import product

def run_benchmark(args, alpha_personal, gate_threshold, output_dir):
    """Run benchmark with specific parameters"""
    cmd = [
        "python", "tools/run_benchmark_lamp2.py",
        "--data_path", args.data_path,
        "--split", "test", 
        "--limit", str(args.limit),
        "--seed", "42",
        "--alpha_personal", str(alpha_personal),
        "--alpha_general", "-1.0",
        "--norm_scale", "0.9",
        "--edit_gate_threshold", str(gate_threshold),
        "--mode", "id",
        "--calibrate",
        "--out_dir", str(output_dir)
    ]
    
    if args.strict:
        cmd.extend(["--strict", "--prior_mode", "user", "--user_prior_path", "data/user_priors.jsonl"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def analyze_predictions(predictions_file):
    """Analyze predictions and return metrics"""
    try:
        b = c = total = correct_baseline = correct_chameleon = 0
        
        with open(predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    gold = data["gold"]
                    baseline = data["baseline"]
                    chameleon = data["chameleon"]
                    
                    if baseline == gold:
                        correct_baseline += 1
                        if chameleon != gold:
                            b += 1  # baseline correct, chameleon wrong
                    else:
                        if chameleon == gold:
                            c += 1  # baseline wrong, chameleon correct
                    
                    if chameleon == gold:
                        correct_chameleon += 1
                    
                    total += 1
        
        if total == 0:
            return None
            
        return {
            "total": total,
            "b": b, 
            "c": c,
            "b_plus_c": b + c,
            "c_minus_b": c - b,
            "baseline_acc": correct_baseline / total,
            "chameleon_acc": correct_chameleon / total,
            "improvement": (correct_chameleon - correct_baseline) / total,
            "effect_rate": (b + c) / total
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Parameter grid search")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--limit", type=int, default=100, help="Sample limit")
    parser.add_argument("--strict", action="store_true", help="Use strict mode")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run (seconds)")
    parser.add_argument("--output", default="results/grid_search_results.csv", help="Output CSV file")
    parser.add_argument("--alpha-range", default="1.5,2.0,2.5,3.0", help="Alpha personal values (comma-separated)")
    parser.add_argument("--gate-range", default="0.0,0.01,0.02,0.03", help="Gate threshold values (comma-separated)")
    
    args = parser.parse_args()
    
    # Parse parameter ranges
    try:
        alpha_values = [float(x) for x in args.alpha_range.split(',')]
        gate_values = [float(x) for x in args.gate_range.split(',')]
    except ValueError:
        print("❌ Invalid parameter ranges")
        sys.exit(2)
    
    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Grid search
    total_runs = len(alpha_values) * len(gate_values)
    completed_runs = 0
    results = []
    
    print(f"🔍 Starting parameter grid search")
    print(f"   Alpha values: {alpha_values}")
    print(f"   Gate values: {gate_values}")
    print(f"   Total combinations: {total_runs}")
    print(f"   Sample limit: {args.limit}")
    print()
    
    for alpha_personal, gate_threshold in product(alpha_values, gate_values):
        completed_runs += 1
        print(f"🧪 Run {completed_runs}/{total_runs}: α={alpha_personal}, gate={gate_threshold}")
        
        output_dir = Path("results/grid_search") / f"ap{alpha_personal}_g{gate_threshold}"
        
        # Run benchmark
        success = run_benchmark(args, alpha_personal, gate_threshold, output_dir)
        
        if success:
            # Analyze results
            pred_file = output_dir / "predictions.jsonl"
            metrics = analyze_predictions(pred_file) if pred_file.exists() else None
            
            if metrics:
                result = {
                    "alpha_personal": alpha_personal,
                    "gate_threshold": gate_threshold,
                    "status": "success",
                    **metrics
                }
                print(f"   ✅ b+c={metrics['b_plus_c']}, c-b={metrics['c_minus_b']}, improvement={metrics['improvement']:.3f}")
            else:
                result = {
                    "alpha_personal": alpha_personal,
                    "gate_threshold": gate_threshold,
                    "status": "analysis_failed"
                }
                print(f"   ❌ Analysis failed")
        else:
            result = {
                "alpha_personal": alpha_personal,
                "gate_threshold": gate_threshold,
                "status": "run_failed"
            }
            print(f"   ❌ Run failed")
        
        results.append(result)
    
    # Save results to CSV
    if results:
        fieldnames = ["alpha_personal", "gate_threshold", "status", "total", "b", "c", "b_plus_c", "c_minus_b", 
                     "baseline_acc", "chameleon_acc", "improvement", "effect_rate"]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    # Summary analysis
    successful_results = [r for r in results if r.get("b_plus_c") is not None]
    
    print()
    print("📊 Grid Search Summary")
    print("=" * 40)
    print(f"Completed runs: {completed_runs}/{total_runs}")
    print(f"Successful analyses: {len(successful_results)}")
    print(f"Results saved to: {output_path}")
    
    if successful_results:
        # Find best b+c
        best_bc = max(successful_results, key=lambda x: x["b_plus_c"])
        print(f"Best b+c: {best_bc['b_plus_c']} (α={best_bc['alpha_personal']}, gate={best_bc['gate_threshold']})")
        
        # Find best improvement
        best_imp = max(successful_results, key=lambda x: x["improvement"])
        print(f"Best improvement: {best_imp['improvement']:.3f} (α={best_imp['alpha_personal']}, gate={best_imp['gate_threshold']})")
        
        # Count effective configurations
        effective_configs = len([r for r in successful_results if r["b_plus_c"] > 0])
        print(f"Effective configurations: {effective_configs}/{len(successful_results)}")
        
        sys.exit(0)
    else:
        print("❌ No successful results - check setup")
        sys.exit(1)

if __name__ == "__main__":
    main()