#!/bin/bash
set -u

echo "🎯 SEQUENTIAL BREAKTHROUGH EXPERIMENTS"
echo "====================================="

# Function to wait for experiment completion and analyze
wait_and_analyze() {
    local name="$1"
    local out_dir="$2"
    
    echo "⏳ Waiting for $name to complete..."
    while [ ! -f "$out_dir/predictions.jsonl" ]; do
        sleep 30
        echo "   Still waiting for $name..."
    done
    
    echo "✅ $name completed! Analyzing results..."
    
    # Generate effects analysis
    python tools/detect_editing_effects.py "$out_dir/predictions.jsonl" > "$out_dir/effects.json" 2>/dev/null || echo "   Effects analysis failed"
    
    # Immediate analysis
    python - <<PY
import json, collections, pathlib
out = pathlib.Path("$out_dir")
if (out/"effects.json").exists():
    E = json.load(open(out/"effects.json"))
    n,b,c = E.get("n",0),E.get("b",0),E.get("c",0)
    print(f"📊 RESULTS: n={n}  b={b}  c={c}  b+c={b+c}  Δacc={E.get('delta_acc'):.4f}  p={E.get('p_value'):.3f}")
    
    # Distribution analysis
    if (out/"predictions.jsonl").exists():
        bc=collections.Counter(); cc=collections.Counter()
        same=0; total=0
        for line in open(out/"predictions.jsonl"):
            o=json.loads(line); total+=1
            if o["baseline"]==o["chameleon"]: same+=1
            bc[o["baseline"]]+=1; cc[o["chameleon"]]+=1
        
        print(f"📈 DISTRIBUTION:")
        print(f"   Identical rate: {same/total:.3f} (target: ≤0.70)")
        print(f"   Baseline top3: {bc.most_common(3)}")
        print(f"   Chameleon top3: {cc.most_common(3)}")
        
        baseline_top1 = bc.most_common(1)[0][1]/total if bc else 0
        cham_top1 = cc.most_common(1)[0][1]/total if cc else 0
        print(f"   Psychology dominance: baseline={baseline_top1:.3f} → chameleon={cham_top1:.3f}")
        
        # Success evaluation
        if b+c >= 25:
            print("🎉 SUCCESS: b+c ≥ 25 (breakthrough achieved!)")
        elif b+c >= 15:
            print("✅ PROGRESS: b+c ≥ 15 (significant improvement)")
        elif b+c >= 10:
            print("⚠️  MODERATE: b+c ≥ 10 (some improvement)")
        else:
            print("❌ LOW: b+c < 10 (insufficient improvement)")
else:
    print("❌ No effects analysis available")
PY
    echo ""
}

# Experiment 1: Enhanced baseline
OUT1=results/bench/a1_no_pmi_alpha10
if [ ! -f "$OUT1/predictions.jsonl" ]; then
    echo "🚀 Starting A1: Enhanced baseline (no PMI, α=10, temp↑)"
    python tools/run_benchmark_lamp2.py \
      --data_path data --split test --limit 100 --seed 42 \
      --alpha_personal 10.0 --alpha_general -1.0 \
      --norm_scale 0.9 --edit_gate_threshold 0.0 \
      --target_layers -4 -3 -2 -1 \
      --mode id --calibrate \
      --score_temp 1.5 \
      --strict --prior_mode user --user_prior_path data/user_priors.jsonl \
      --out_dir "$OUT1" || echo "A1 failed"
fi
wait_and_analyze "A1-Enhanced-Baseline" "$OUT1"

# Experiment 2: PMI correction  
OUT2=results/bench/a2_pmi_alpha10
if [ ! -f "$OUT2/predictions.jsonl" ]; then
    echo "🚀 Starting A2: PMI correction (α=10, temp↑)"
    python tools/run_benchmark_lamp2.py \
      --data_path data --split test --limit 100 --seed 42 \
      --alpha_personal 10.0 --alpha_general -1.0 \
      --norm_scale 0.9 --edit_gate_threshold 0.0 \
      --target_layers -4 -3 -2 -1 \
      --mode id --calibrate \
      --use_pmi --score_temp 1.5 \
      --strict --prior_mode user --user_prior_path data/user_priors.jsonl \
      --out_dir "$OUT2" || echo "A2 failed"
fi
wait_and_analyze "A2-PMI-Correction" "$OUT2"

# Experiment 3: PMI + entropy calibration
OUT3=results/bench/a3_pmi_calib_entropy
if [ ! -f "$OUT3/predictions.jsonl" ]; then
    echo "🚀 Starting A3: PMI + entropy calibration"
    python tools/run_benchmark_lamp2.py \
      --data_path data --split test --limit 100 --seed 42 \
      --alpha_personal 10.0 --alpha_general -1.0 \
      --norm_scale 0.9 --edit_gate_threshold 0.0 \
      --target_layers -4 -3 -2 -1 \
      --mode id --calibrate \
      --use_pmi --score_temp 1.5 \
      --target_entropy 2.2 --max_top1_share 0.40 \
      --strict --prior_mode user --user_prior_path data/user_priors.jsonl \
      --out_dir "$OUT3" || echo "A3 failed"
fi  
wait_and_analyze "A3-PMI-Entropy" "$OUT3"

# Final summary
echo "🏁 ALL EXPERIMENTS COMPLETED"
echo "============================"
echo ""
echo "📋 FINAL COMPARISON:"

for exp in "a1_no_pmi_alpha10:A1-Enhanced-Baseline" "a2_pmi_alpha10:A2-PMI-Correction" "a3_pmi_calib_entropy:A3-PMI-Entropy"; do
    dir=$(echo $exp | cut -d: -f1)
    name=$(echo $exp | cut -d: -f2)
    out="results/bench/$dir"
    
    if [ -f "$out/effects.json" ]; then
        python - <<PY
import json, collections
E = json.load(open("$out/effects.json"))
b,c = E.get("b",0), E.get("c",0)
print(f"$name: b+c={b+c}, net={c-b}, Δacc={E.get('delta_acc',0):.4f}")
PY
    else
        echo "$name: No results"
    fi
done

echo ""
echo "🎯 Next steps: If best result has b+c ≥ 25, proceed to Phase B!"
echo "   If b+c < 25, try α=12.0 or add --prior_beta 0.5"