#!/usr/bin/env python3
"""
Focused STRICT mode experiment runner - key conditions only.
"""
import subprocess
import sys
import json
import time
from pathlib import Path

def run_experiment(name, alpha, gate, limit=50, seed=42):
    """Run a single STRICT experiment with validation."""
    out_dir = f"results/bench/strict_{name}_n{limit}"
    
    cmd = [
        "python", "tools/run_benchmark_lamp2.py",
        "--data_path", "data", "--split", "test", 
        "--limit", str(limit), "--seed", str(seed),
        "--alpha_personal", str(alpha),
        "--alpha_general", "-1.0",
        "--norm_scale", "0.9",
        "--edit_gate_threshold", str(gate),
        "--target_layers", "-4", "-3", "-2", "-1",
        "--mode", "id", "--calibrate",
        "--strict", "--prior_mode", "user",
        "--user_prior_path", "data/user_priors.jsonl",
        "--out_dir", out_dir
    ]
    
    print(f"🚀 Running {name}: α={alpha}, gate={gate}, n={limit}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ {name} failed after {duration:.1f}s")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return None
            
        print(f"✅ {name} completed in {duration:.1f}s")
        return out_dir
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  {name} timed out after 600s")
        return None

def validate_strict_compliance(predictions_file):
    """Quick STRICT compliance check."""
    try:
        violations = 0
        total = 0
        
        with open(predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prior = data.get('prior', {})
                    source = prior.get('source', 'missing')
                    total += 1
                    if source != 'user':
                        violations += 1
        
        if violations == 0:
            print(f"  ✅ STRICT OK: {total} predictions, all user priors")
            return True
        else:
            print(f"  ❌ STRICT FAIL: {violations}/{total} non-user priors")
            return False
            
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False

def detect_effects(predictions_file):
    """Quick editing effects detection."""
    try:
        result = subprocess.run([
            "python", "tools/detect_editing_effects.py", predictions_file
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse the output for key metrics
            for line in result.stdout.split('\n'):
                if 'b+c' in line or 'Effect rate' in line or 'Net improvement' in line:
                    print(f"  📊 {line.strip()}")
            return True
        else:
            print(f"  ❌ Effects detection failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Effects error: {e}")
        return False

def main():
    """Run focused STRICT experiments."""
    print("🎯 STRICT Mode Focused Experiments")
    
    # Key experimental conditions
    experiments = [
        ("baseline", 2.75, 0.022),  # Conservative
        ("moderate", 4.0, 0.0),     # Moderate  
        ("aggressive", 6.0, 0.0),   # Aggressive
    ]
    
    results = []
    
    for name, alpha, gate in experiments:
        out_dir = run_experiment(name, alpha, gate, limit=30, seed=42)
        
        if out_dir and Path(f"{out_dir}/predictions.jsonl").exists():
            predictions_file = f"{out_dir}/predictions.jsonl"
            
            # Validate STRICT compliance
            strict_ok = validate_strict_compliance(predictions_file)
            
            # Detect editing effects
            effects_ok = detect_effects(predictions_file)
            
            results.append({
                'name': name,
                'alpha': alpha,
                'gate': gate,
                'out_dir': out_dir,
                'strict_compliant': strict_ok,
                'effects_detected': effects_ok
            })
        else:
            results.append({
                'name': name,
                'alpha': alpha,
                'gate': gate,
                'out_dir': None,
                'strict_compliant': False,
                'effects_detected': False
            })
    
    # Summary report
    print(f"\n📋 EXPERIMENT SUMMARY")
    for r in results:
        status = "✅" if r['strict_compliant'] and r['effects_detected'] else "❌"
        print(f"  {status} {r['name']}: α={r['alpha']} gate={r['gate']} dir={r['out_dir']}")
    
    # Write summary JSON
    with open("results/bench/strict_focused_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Summary saved to results/bench/strict_focused_summary.json")

if __name__ == "__main__":
    main()