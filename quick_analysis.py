#!/usr/bin/env python3
"""Quick analysis script for breakthrough experiments"""

import json
import collections
import pathlib
import sys

def analyze_experiment(out_dir):
    """Analyze a single experiment's results"""
    out = pathlib.Path(out_dir)
    
    if not (out / "predictions.jsonl").exists():
        print(f"❌ {out_dir}: No predictions found")
        return None
    
    # Load effects if available
    effects = {}
    if (out / "effects.json").exists():
        effects = json.load(open(out / "effects.json"))
    
    # Analyze predictions
    bc = collections.Counter()
    cc = collections.Counter() 
    same = 0
    total = 0
    
    with open(out / "predictions.jsonl") as f:
        for line in f:
            o = json.loads(line)
            total += 1
            if o["baseline"] == o["chameleon"]:
                same += 1
            bc[o["baseline"]] += 1
            cc[o["chameleon"]] += 1
    
    # Calculate metrics
    n = effects.get("n", total)
    b = effects.get("b", 0) 
    c = effects.get("c", 0)
    bc_total = b + c
    net = c - b
    delta_acc = effects.get("delta_acc", 0)
    p_value = effects.get("p_value", 1.0)
    
    identical_rate = same / total if total > 0 else 0
    baseline_top1_share = bc.most_common(1)[0][1] / total if bc else 0
    cham_top1_share = cc.most_common(1)[0][1] / total if cc else 0
    
    # Success evaluation
    if bc_total >= 25:
        status = "🎉 BREAKTHROUGH"
        color = "\033[92m"  # Green
    elif bc_total >= 15:
        status = "✅ STRONG"
        color = "\033[93m"  # Yellow
    elif bc_total >= 10:
        status = "⚠️  MODERATE"
        color = "\033[94m"  # Blue
    else:
        status = "❌ WEAK"
        color = "\033[91m"  # Red
    
    reset_color = "\033[0m"
    
    print(f"{color}{status}{reset_color} {out_dir}:")
    print(f"  📊 Effects: n={n}, b={b}, c={c}, b+c={bc_total}, net={net}")
    print(f"  📈 Quality: Δacc={delta_acc:.4f}, p={p_value:.3f}")
    print(f"  🎯 Distribution: identical={identical_rate:.3f}, psych_drop={baseline_top1_share:.3f}→{cham_top1_share:.3f}")
    print(f"  🔝 Baseline: {bc.most_common(3)}")
    print(f"  🔝 Chameleon: {cc.most_common(3)}")
    
    return {
        'name': out_dir,
        'b_plus_c': bc_total,
        'net_improvement': net,
        'delta_acc': delta_acc,
        'p_value': p_value,
        'identical_rate': identical_rate,
        'psychology_drop': baseline_top1_share - cham_top1_share,
        'status': status.split()[1] if len(status.split()) > 1 else 'UNKNOWN'
    }

def main():
    if len(sys.argv) < 2:
        # Analyze all A experiments
        experiments = [
            "results/bench/a1_no_pmi_alpha10",
            "results/bench/a2_pmi_alpha10", 
            "results/bench/a3_pmi_calib_entropy"
        ]
    else:
        experiments = sys.argv[1:]
    
    print("🔍 BREAKTHROUGH EXPERIMENT ANALYSIS")
    print("=" * 50)
    
    results = []
    for exp in experiments:
        if pathlib.Path(exp).exists():
            result = analyze_experiment(exp)
            if result:
                results.append(result)
            print()
    
    # Summary comparison
    if len(results) > 1:
        print("📋 SUMMARY COMPARISON:")
        print("-" * 30)
        results.sort(key=lambda x: x['b_plus_c'], reverse=True)
        
        for r in results:
            print(f"{r['status']:<12} {r['name']:<25} b+c={r['b_plus_c']:<3} net={r['net_improvement']:<3} drop={r['psychology_drop']:.3f}")
        
        best = results[0]
        print(f"\n🏆 BEST: {best['name']} with b+c={best['b_plus_c']}")
        
        if best['b_plus_c'] >= 25:
            print("🎉 BREAKTHROUGH ACHIEVED! Ready for Phase B optimization!")
        elif best['b_plus_c'] >= 15:
            print("✅ STRONG PROGRESS! Consider α=12.0 or --prior_beta 0.5 for final push")
        else:
            print("⚠️  Need more aggressive intervention. Try α=12.0-15.0 or different approaches")

if __name__ == "__main__":
    main()