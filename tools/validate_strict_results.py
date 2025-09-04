#!/usr/bin/env python3
"""
Post-run validation for strict mode: Ensure zero fallbacks occurred.

Usage:
    python tools/validate_strict_results.py results/bench/strict_n140/predictions.jsonl
    
Exit codes:
    0: All predictions have prior.source=='user' (SUCCESS)
    1: Found non-user prior sources (FAILURE)
    2: File not found or invalid
"""

import json
import sys
import argparse
import hashlib
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser(description="Validate strict mode results")
    parser.add_argument("predictions_file", help="Path to predictions.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")
    
    args = parser.parse_args()
    
    try:
        sources = []
        per_user_hash = defaultdict(set)
        total_lines = 0
        
        with open(args.predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        prior_info = data.get("prior", {})
                        source = prior_info.get("source", "missing")
                        sources.append(source)
                        
                        # Check prior hash consistency per user
                        user_id = prior_info.get("user_id", "")
                        prompt = prior_info.get("prompt") or prior_info.get("prior_prompt") or ""
                        if prompt:
                            h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
                            per_user_hash[user_id].add(h)
                        
                        total_lines += 1
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON on line {total_lines + 1}: {e}")
                        sys.exit(2)
        
        if not sources:
            print(f"❌ No predictions found in {args.predictions_file}")
            sys.exit(2)
            
        # Count sources
        source_counts = Counter(sources)
        
        print(f"📊 Validation Results for {args.predictions_file}")
        print(f"   Total predictions: {total_lines}")
        
        if args.verbose or len(source_counts) > 1:
            print(f"   Prior source breakdown:")
            for source, count in sorted(source_counts.items()):
                percentage = (count / total_lines) * 100
                print(f"     {source}: {count} ({percentage:.1f}%)")
        
        # Check strict compliance
        non_user_sources = {k: v for k, v in source_counts.items() if k != "user"}
        
        # Check prior hash consistency
        bad_hash_users = {u for u, s in per_user_hash.items() if len(s) > 1}
        
        violations = []
        if non_user_sources:
            violations.append(f"Non-user prior sources: {dict(non_user_sources)}")
        if bad_hash_users:
            violations.append(f"Users with inconsistent priors: {len(bad_hash_users)}")
        
        if violations:
            print(f"❌ STRICT MODE VIOLATION: {'; '.join(violations)}")
            if non_user_sources:
                for source, count in non_user_sources.items():
                    print(f"   {source}: {count} violations")
            if bad_hash_users:
                print(f"   Inconsistent prior users: {list(bad_hash_users)[:5]}{'...' if len(bad_hash_users) > 5 else ''}")
            print(f"   Expected: All predictions must have prior.source=='user' with consistent hashes")
            sys.exit(1)
        else:
            print(f"✅ STRICT MODE COMPLIANCE: All {total_lines} predictions use user priors")
            print(f"   Zero fallbacks detected - experiment is valid")
            print(f"   Prior hash consistency: ✅ ({len(per_user_hash)} users)")
            sys.exit(0)
            
    except FileNotFoundError:
        print(f"❌ File not found: {args.predictions_file}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()