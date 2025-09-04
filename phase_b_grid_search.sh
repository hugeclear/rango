#!/bin/bash
# Phase B: Mini grid search for optimal parameter range
set -euo pipefail

echo "🔍 Phase B: Parameter Grid Search"
echo "================================="

BASE="results/bench/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE"
echo "📁 Output directory: $BASE"

# Grid search parameters
ALPHAS=(4.0 5.0 6.0)
GATES=(0.00 0.01 0.02)  
LAYERS=("-4 -3 -2 -1" "-8 -7 -6 -5")

echo "🔧 Parameter combinations:"
echo "  α: ${ALPHAS[@]}"
echo "  gate: ${GATES[@]}"  
echo "  layers: top4 vs mid4"
echo ""

total=$((${#ALPHAS[@]} * ${#GATES[@]} * ${#LAYERS[@]}))
current=0

for A in "${ALPHAS[@]}"; do
  for G in "${GATES[@]}"; do
    for L in "${LAYERS[@]}"; do
      current=$((current + 1))
      layer_name=$(echo "$L" | tr ' ' '_')
      OUT="$BASE/a${A}_g${G}_lay${layer_name}"
      
      echo "[$current/$total] Running α=$A gate=$G layers=$L"
      
      timeout 1200 python tools/run_benchmark_lamp2.py \
        --data_path data --split test --limit 100 --seed 42 \
        --alpha_personal "$A" --alpha_general -1.0 \
        --norm_scale 0.9 --edit_gate_threshold "$G" \
        --target_layers $L \
        --mode id --calibrate \
        --strict --prior_mode user \
        --user_prior_path data/user_priors.jsonl \
        --out_dir "$OUT" || {
        echo "  ❌ Failed or timed out"
        continue
      }
      
      # Generate analysis files immediately
      if [[ -f "$OUT/predictions.jsonl" ]]; then
        python tools/detect_editing_effects.py "$OUT/predictions.jsonl" > "$OUT/effects.json" 2>/dev/null || echo "  ⚠️ Effects analysis failed"
        python tools/diagnose_gate_health.py "$OUT/predictions.jsonl" > "$OUT/gate_diag.json" 2>/dev/null || echo "  ⚠️ Gate diagnosis failed"
        echo "  ✅ Completed"
      else
        echo "  ❌ No predictions file generated"
      fi
    done
  done
done

echo ""
echo "📊 Collecting Results"
echo "===================="

# Results analysis
python - <<PY "$BASE"
import json, sys, glob, os

base = sys.argv[1]
rows = []

for d in sorted(glob.glob(os.path.join(base, "*"))):
    try:
        effects_file = os.path.join(d, "effects.json")
        gate_file = os.path.join(d, "gate_diag.json")
        
        if os.path.exists(effects_file):
            E = json.load(open(effects_file))
            try:
                G = json.load(open(gate_file))
            except:
                G = {}
                
            dirname = os.path.basename(d)
            n = E.get("n", 0)
            b = E.get("b", 0) 
            c = E.get("c", 0)
            bc = b + c
            net = c - b
            delta_acc = E.get("delta_acc", 0)
            p_value = E.get("p_value", 1.0)
            gate_rate = G.get("gate_rate", 0)
            mean_cos = G.get("mean_cos_theta", 0)
            
            rows.append((dirname, n, b, c, bc, net, delta_acc, p_value, gate_rate, mean_cos))
    except Exception as e:
        print(f"Error processing {d}: {e}")

# Sort by net improvement (c-b), then by total effects (b+c), then by gate_rate
rows.sort(key=lambda r: (-r[5], -r[4], -r[8]))

print("🏆 PHASE B RESULTS (sorted by c-b, b+c, gate_rate)")
print("=" * 80)
print(f"{'config':<20} {'n':<3} {'b':<3} {'c':<3} {'b+c':<4} {'c-b':<4} {'Δacc':<6} {'p':<8} {'gate':<5} {'cosθ':<5}")
print("-" * 80)

for i, (config, n, b, c, bc, net, delta_acc, p_val, gate_rate, cos_theta) in enumerate(rows):
    status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    print(f"{status}{config:<18} {n:<3} {b:<3} {c:<3} {bc:<4} {net:<4} {delta_acc:<6.3f} {p_val:<8.3f} {gate_rate:<5.2f} {cos_theta:<5.2f}")

# Identify best candidates
print("\n📋 ANALYSIS")
print("=" * 40)
best_configs = rows[:3]
for i, (config, n, b, c, bc, net, delta_acc, p_val, gate_rate, cos_theta) in enumerate(best_configs):
    rank = ["🥇 BEST", "🥈 RUNNER-UP", "🥉 THIRD"][i]
    print(f"{rank}: {config}")
    print(f"   Effects: b={b}, c={c}, net improvement={net} ({net/n*100:.1f}%)")
    print(f"   Health: gate_rate={gate_rate:.2f}, cos_theta={cos_theta:.2f}")
    
    # Quality assessment
    if net > 0 and bc >= 15 and gate_rate > 0.8:
        print(f"   ✅ EXCELLENT: Strong effects with healthy gating")
    elif net > 0 and bc >= 10:
        print(f"   ✅ GOOD: Positive net improvement")
    elif bc >= 15:
        print(f"   ⚠️  MIXED: High effects but unclear net benefit") 
    else:
        print(f"   ❌ WEAK: Insufficient effects")
    print()

print(f"💾 Results saved in: {base}")
print(f"📊 Total configurations tested: {len(rows)}")

PY

echo ""
echo "Next: Use best configuration for Phase C sample size calculation"