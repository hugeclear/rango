#!/bin/bash
# Phase A validation script

set -euo pipefail

OUT="results/bench/strict_try_alpha6"

echo "🔍 Phase A Validation Results"
echo "============================="

if [[ ! -f "$OUT/predictions.jsonl" ]]; then
    echo "❌ Predictions file not found: $OUT/predictions.jsonl"
    exit 1
fi

echo ""
echo "📊 EDITING EFFECTS ANALYSIS"
echo "---------------------------"
python tools/detect_editing_effects.py "$OUT/predictions.jsonl" --verbose | head -20

echo ""
echo "🔒 STRICT COMPLIANCE CHECK" 
echo "-------------------------"
python tools/validate_strict_results.py "$OUT/predictions.jsonl" | head -15

echo ""
echo "🚪 GATE HEALTH DIAGNOSIS"
echo "-----------------------"
python tools/diagnose_gate_health.py "$OUT/predictions.jsonl" | head -15

echo ""
echo "📈 PHASE A SUCCESS CRITERIA"
echo "==========================="
echo "Target: b+c ≥ 30 (sufficient differences)"
echo "Target: c > b (net improvement positive)"  
echo "Target: 100% user priors (STRICT compliance)"
echo "Target: gate_rate ≈ 1.0 (healthy gating)"

# Extract key metrics for decision making
python - <<'PY'
import json, pathlib
try:
    effects = json.load(open("results/bench/strict_try_alpha6/effects.json"))
    b, c = effects.get("b", 0), effects.get("c", 0)
    bc = b + c
    net = c - b
    
    print(f"📊 RESULTS: b={b}, c={c}, b+c={bc}, c-b={net}")
    print(f"📊 Effect rate: {bc/effects.get('n',1)*100:.1f}%")
    print(f"📊 Net improvement: {net/effects.get('n',1)*100:.1f}%")
    
    # Decision criteria
    if bc >= 30 and net > 0:
        print("✅ PHASE A SUCCESS: Proceed to Phase B parameter optimization")
    elif bc >= 15:
        print("⚠️  MODERATE EFFECTS: Consider increasing α or sample size")
    else:
        print("❌ INSUFFICIENT EFFECTS: Need parameter adjustment")
        
except Exception as e:
    print(f"❌ Analysis error: {e}")
PY

echo ""
echo "Next: Run Phase B mini grid search for optimal parameters"