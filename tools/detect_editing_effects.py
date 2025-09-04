#!/usr/bin/env python3
"""
Detect editing effects: Check if Chameleon modifications have observable impact.

Usage:
    python tools/detect_editing_effects.py results/bench/strict_n140/predictions.jsonl

Exit codes:
    0: Editing effects detected (b+c > 0)
    1: No editing effects (b+c = 0 - potential issue)
    2: File not found or invalid
"""

import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Detect editing effects in predictions")
    parser.add_argument("predictions_file", help="Path to predictions.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")
    
    args = parser.parse_args()
    
    try:
        b = c = total = 0
        
        with open(args.predictions_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        gold = data["gold"]
                        baseline = data["baseline"]
                        chameleon = data["chameleon"]
                        
                        # b: baseline correct, chameleon wrong
                        if baseline == gold and chameleon != gold:
                            b += 1
                        # c: baseline wrong, chameleon correct  
                        elif baseline != gold and chameleon == gold:
                            c += 1
                            
                        total += 1
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"❌ Error on line {line_num}: {e}")
                        sys.exit(2)
        
        if total == 0:
            print(f"❌ No predictions found in {args.predictions_file}")
            sys.exit(2)
        
        print(f"📊 Editing Effects Analysis for {args.predictions_file}")
        print(f"   Total predictions: {total}")
        print(f"   b (baseline→chameleon worse): {b}")
        print(f"   c (baseline→chameleon better): {c}")
        print(f"   b+c (total editing effects): {b+c}")
        print(f"   c-b (net improvement): {c-b}")
        
        if args.verbose:
            print(f"   Effect rate: {(b+c)/total:.1%}")
            print(f"   Net improvement rate: {(c-b)/total:.1%}")
        
        if b + c > 0:
            print(f"✅ EDITING EFFECTS DETECTED: b+c = {b+c} > 0")
            print(f"   Chameleon modifications are having observable impact")
            sys.exit(0)
        else:
            print(f"❌ NO EDITING EFFECTS: b+c = {b+c} = 0")
            print(f"   Potential issues: calibration off, gate threshold, or direction vectors")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"❌ File not found: {args.predictions_file}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()