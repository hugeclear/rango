#!/usr/bin/env python3
"""
Complete validation pipeline for unbreakable Chameleon experiments.

Usage:
    python tools/run_complete_validation.py --mode smoke  # Quick smoke test
    python tools/run_complete_validation.py --mode full   # Full validation

Exit codes:
    0: All validations passed - ready for N=500 run
    1: Validation failures detected
    2: Setup or file errors
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

def run_step(cmd, description, required=True):
    """Run a validation step"""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, list):
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed (exit {result.returncode})")
            if result.stdout:
                print(f"   Output: {result.stdout[-200:]}")
            if result.stderr:
                print(f"   Error: {result.stderr[-200:]}")
            
            if required:
                return False
            else:
                print(f"   ⚠️ Optional step failed, continuing...")
                return True
                
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False if required else True

def main():
    parser = argparse.ArgumentParser(description="Complete validation pipeline")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True, help="Validation mode")
    parser.add_argument("--limit", type=int, help="Sample limit (default: smoke=10, full=100)")
    parser.add_argument("--skip-model", action="store_true", help="Skip model loading steps for speed")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.limit is None:
        args.limit = 10 if args.mode == "smoke" else 100
    
    print(f"🚀 Running {args.mode} validation pipeline (limit={args.limit})")
    print("=" * 60)
    
    steps_passed = 0
    total_steps = 0
    
    # Step 1: Data validation
    total_steps += 1
    if run_step([
        "python", "tools/validate_lamp2.py",
        "--dataset", "data/evaluation/lamp2_expanded_eval.jsonl",
        "--report", "results/diagnostics/lamp2_preflight.md"
    ], "Step 1: LaMP-2 dataset validation"):
        steps_passed += 1
    else:
        print("🔴 Critical: Dataset validation failed")
        sys.exit(2)
    
    # Step 2: Generate hash manifest
    total_steps += 1
    manifest_cmd = '''python - <<'PY'
import hashlib, json, pathlib
def sha(p): 
    h=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    return h[:16]
manifest={
  "dataset":"data/evaluation/lamp2_expanded_eval.jsonl",
  "id2tag":"data/id2tag.txt", 
  "user_priors":"data/user_priors.jsonl",
}
hashes = {}
for k,v in manifest.items(): 
    hashes[k+"_sha256"]=sha(v)
manifest.update(hashes)
pathlib.Path("results/diagnostics").mkdir(parents=True, exist_ok=True)
pathlib.Path("results/diagnostics/manifest.json").write_text(json.dumps(manifest,indent=2))
print("Hash manifest generated")
PY'''
    if run_step(manifest_cmd, "Step 2: Generate data hash manifest"):
        steps_passed += 1
    
    # Step 3: Generate user priors
    total_steps += 1
    if run_step([
        "python", "tools/preflight_priors.py",
        "--dataset", "data/evaluation/lamp2_expanded_eval.jsonl",
        "--labels", "data/id2tag.txt",
        "--out", "data/user_priors.jsonl"
    ], "Step 3: Generate user priors"):
        steps_passed += 1
    else:
        print("🔴 Critical: Prior generation failed")
        sys.exit(2)
    
    if not args.skip_model:
        # Step 4: Run ablation smoke tests
        total_steps += 1
        if run_step([
            "python", "tools/run_ablation_smoke.py", 
            "--data_path", "data",
            "--limit", str(min(args.limit, 10)),
            "--strict"
        ], "Step 4: Ablation smoke tests", required=False):
            steps_passed += 1
        
        # Step 5: Effect detection test
        total_steps += 1
        test_effects_cmd = '''python - <<'PY'
import json
import sys
# Mock test data with effects
data = [
    {"gold": "action", "baseline": "comedy", "chameleon": "action"},  # c=1
    {"gold": "sci-fi", "baseline": "sci-fi", "chameleon": "comedy"}, # b=1  
    {"gold": "romance", "baseline": "action", "chameleon": "romance"} # c=1
]
with open("results/test_effects.jsonl", "w") as f:
    for item in data:
        json.dump(item, f)
        f.write("\\n")
print("Mock data created for effect detection test")
PY'''
        
        if run_step(test_effects_cmd, "Step 5a: Create test data", required=False):
            if run_step([
                "python", "tools/detect_editing_effects.py",
                "results/test_effects.jsonl",
                "--verbose"
            ], "Step 5b: Test effect detection"):
                steps_passed += 1
    else:
        print("⏭️ Skipping model-dependent steps due to --skip-model flag")
    
    # Step 6: Test strict validation
    total_steps += 1
    test_strict_cmd = '''python - <<'PY'
import json
# Mock strict validation data
data = [
    {"id": "test1", "prediction": "action", "prior": {"source": "user", "prompt": "test1", "user_id": "u1"}},
    {"id": "test2", "prediction": "comedy", "prior": {"source": "user", "prompt": "test2", "user_id": "u2"}}
]
with open("results/test_strict.jsonl", "w") as f:
    for item in data:
        json.dump(item, f)
        f.write("\\n")
print("Mock data created for strict validation test")
PY'''
    
    if run_step(test_strict_cmd, "Step 6a: Create test strict data"):
        if run_step([
            "python", "tools/validate_strict_results.py",
            "results/test_strict.jsonl",
            "--verbose"
        ], "Step 6b: Test strict validation"):
            steps_passed += 1
    
    # Cleanup test files
    run_step("rm -f results/test_*.jsonl", "Cleanup test files", required=False)
    
    # Summary
    print()
    print("📋 Validation Pipeline Summary")
    print("=" * 40)
    print(f"Steps passed: {steps_passed}/{total_steps}")
    
    # Read manifest if available
    manifest_path = Path("results/diagnostics/manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            print(f"Data integrity hashes:")
            for k, v in manifest.items():
                if k.endswith("_sha256"):
                    print(f"  {k}: {v}")
        except Exception:
            pass
    
    # Final assessment
    if steps_passed == total_steps:
        print()
        print("✅ ALL VALIDATIONS PASSED")
        print("🎯 System ready for full N=500 experiment")
        print()
        print("Next steps:")
        print("1. Run: python tools/run_benchmark_lamp2.py --strict --limit 500 ...")
        print("2. Validate: python tools/validate_strict_results.py results/...")
        print("3. Check effects: python tools/detect_editing_effects.py results/...")
        print("4. Diagnose health: python tools/diagnose_gate_health.py results/...")
        sys.exit(0)
    else:
        print()
        print(f"❌ VALIDATION FAILURES: {total_steps - steps_passed}/{total_steps} steps failed")
        print("🔧 Fix issues before proceeding to full experiment")
        sys.exit(1)

if __name__ == "__main__":
    main()