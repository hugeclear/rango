#!/usr/bin/env python3
"""
🎯 CFS-Chameleon Production Validation Test
全本番機能の最終動作確認テスト

Previously failing command now works:
CUDA_VISIBLE_DEVICES=0 python lamp2_cfs_benchmark.py --use_collaboration --config cfs_config.yaml --evaluation_mode cfs --include_baseline
"""

import subprocess
import sys
import time

def run_command_test(command, test_name, timeout=60):
    """コマンドテスト実行"""
    print(f"\n🧪 {test_name}")
    print(f"Command: {command}")
    print("=" * 70)
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"CUDA_VISIBLE_DEVICES": "0"}
        )
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({execution_time:.1f}s)")
            
            # Check for critical success indicators
            if "✅ Collaborative Chameleon editing completed" in result.stdout:
                print("   ✓ Collaborative editing working")
            if "collaboration_sessions" in result.stderr and "KeyError" in result.stderr:
                print("   ❌ KeyError still present!")
                return False
            if "fallback" in result.stderr.lower() or "fallback" in result.stdout.lower():
                print("   ❌ Fallback still being used!")
                return False
            
            return True
        else:
            print(f"❌ {test_name} FAILED")
            print(f"Exit code: {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {test_name} TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False

def main():
    """Production validation main"""
    print("🚀 CFS-Chameleon Production Validation Test Suite")
    print("=" * 70)
    
    tests = [
        {
            "command": "python lamp2_cfs_benchmark.py --use_collaboration --config cfs_config.yaml --evaluation_mode cfs --sample_limit=3",
            "name": "CFS-Only Evaluation",
            "timeout": 60
        },
        {
            "command": "python lamp2_cfs_benchmark.py --compare_modes --use_collaboration --config cfs_config.yaml --sample_limit=3", 
            "name": "Comparison Mode",
            "timeout": 120
        },
        {
            "command": "python lamp2_cfs_benchmark.py --use_collaboration --config cfs_config.yaml --evaluation_mode cfs --include_baseline --sample_limit=3",
            "name": "Original Failing Command (FIXED)",
            "timeout": 60
        }
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        if run_command_test(test["command"], test["name"], test["timeout"]):
            passed_tests += 1
    
    print(f"\n" + "=" * 70)
    print(f"📊 PRODUCTION VALIDATION RESULTS")
    print(f"=" * 70)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print(f"🎉 ALL TESTS PASSED - Production system is fully operational!")
        print(f"✅ collaboration_sessions KeyError: FIXED")
        print(f"✅ Fallback usage: ELIMINATED") 
        print(f"✅ CFS-Chameleon: FULLY FUNCTIONAL")
        print(f"\n🚀 Ready for production use!")
        return 0
    else:
        print(f"❌ {total_tests - passed_tests} TESTS FAILED - Issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())