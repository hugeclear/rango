#!/usr/bin/env python3
"""
Quick ablation smoke tests for debugging Chameleon effects.

Usage:
    python tools/run_ablation_smoke.py --data_path data --limit 10 --strict

Exit codes:
    0: Smoke tests completed successfully
    1: Smoke tests failed
    2: Invalid arguments or file errors
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🧪 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed (exit {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ Timeout (5min limit)")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def check_effects(predictions_file, test_name):
    """Check for editing effects in predictions"""
    try:
        b = c = total = 0
        with open(predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    gold = data["gold"]
                    baseline = data["baseline"] 
                    chameleon = data["chameleon"]
                    
                    if baseline == gold and chameleon != gold:
                        b += 1
                    elif baseline != gold and chameleon == gold:
                        c += 1
                    total += 1
        
        print(f"   📊 {test_name}: b={b} c={c} b+c={b+c} (total={total})")
        return b + c > 0
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run ablation smoke tests")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--limit", type=int, default=10, help="Sample limit for speed")
    parser.add_argument("--strict", action="store_true", help="Use strict mode")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test (seconds)")
    
    args = parser.parse_args()
    
    # Prepare base command
    base_cmd = [
        "python", "tools/run_benchmark_lamp2.py",
        "--data_path", args.data_path,
        "--split", "test",
        "--limit", str(args.limit),
        "--seed", "42",
        "--alpha_personal", "2.75",
        "--alpha_general", "-1.0", 
        "--norm_scale", "0.9",
        "--edit_gate_threshold", "0.022",
        "--mode", "id"
    ]
    
    if args.strict:
        base_cmd.extend(["--strict", "--prior_mode", "user", "--user_prior_path", "data/user_priors.jsonl"])
    
    # Test configurations
    tests = [
        {
            "name": "baseline_normal",
            "desc": "Normal calibration ON",
            "extra_args": ["--calibrate"],
            "out_dir": "results/ablation/smoke_normal"
        },
        {
            "name": "calibration_off", 
            "desc": "Calibration OFF (prior influence isolation)",
            "extra_args": ["--calibrate", "false"],
            "out_dir": "results/ablation/smoke_no_calibrate"
        },
        {
            "name": "forced_gate",
            "desc": "Forced gate application (threshold=-1e6)",
            "extra_args": ["--calibrate", "--edit_gate_threshold", "-1e6"],
            "out_dir": "results/ablation/smoke_forced_gate"
        }
    ]
    
    # Run tests
    results = {}
    print(f"🚀 Running {len(tests)} ablation smoke tests (limit={args.limit})")
    print()
    
    for test in tests:
        cmd = base_cmd + test["extra_args"] + ["--out_dir", test["out_dir"]]
        success = run_command(cmd, f"Test: {test['desc']}")
        
        effects_detected = False
        if success:
            pred_file = Path(test["out_dir"]) / "predictions.jsonl"
            if pred_file.exists():
                effects_detected = check_effects(pred_file, test["name"])
        
        results[test["name"]] = {
            "success": success,
            "effects": effects_detected
        }
        print()
    
    # Summary
    print("📋 Ablation Smoke Test Summary")
    print("=" * 40)
    
    total_success = 0
    total_effects = 0
    
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        effects = "🎯" if result["effects"] else "🔳"
        print(f"{status} {effects} {test_name}: run={result['success']} effects={result['effects']}")
        
        if result["success"]:
            total_success += 1
        if result["effects"]:
            total_effects += 1
    
    print()
    print(f"Success rate: {total_success}/{len(tests)}")
    print(f"Effects detected: {total_effects}/{len(tests)}")
    
    # Analysis
    if total_success == 0:
        print("🔴 All tests failed - check basic setup")
        sys.exit(1)
    elif total_effects == 0:
        print("🟡 No effects detected - check calibration, gate, or direction vectors")
        print("   • If calibration_off shows effects: prior影響が強すぎる可能性")
        print("   • If forced_gate shows effects: ゲート閾値・DVスケール問題")
        print("   • If no effects anywhere: DV生成・hook登録を確認")
        sys.exit(1)
    else:
        print("✅ Smoke tests passed - effects detected, system appears functional")
        sys.exit(0)

if __name__ == "__main__":
    main()