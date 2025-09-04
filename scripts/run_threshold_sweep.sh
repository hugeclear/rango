#!/bin/bash
# Automated threshold sweep for Chameleon gate optimization
set -e

# Configuration
DATA_PATH=${DATA_PATH:-"data"}
SPLIT=${SPLIT:-"test"}
LIMIT=${LIMIT:-500}
SEED=${SEED:-42}
ALPHA_PERSONAL=${ALPHA_PERSONAL:-2.75}
ALPHA_GENERAL=${ALPHA_GENERAL:-"-1.0"}
NORM_SCALE=${NORM_SCALE:-0.9}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-1}

echo "🔬 Chameleon Gate Threshold Sweep"
echo "================================="
echo "Configuration:"
echo "  Data: $DATA_PATH/$SPLIT (limit=$LIMIT)"
echo "  Alpha: personal=$ALPHA_PERSONAL, general=$ALPHA_GENERAL"
echo "  Seed: $SEED"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="results/sweep/gate_sweep_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

# Log configuration
cat > "$SWEEP_DIR/config.txt" << EOF
Gate Threshold Sweep Configuration
==================================
Timestamp: $(date)
Data Path: $DATA_PATH
Split: $SPLIT
Limit: $LIMIT
Seed: $SEED
Alpha Personal: $ALPHA_PERSONAL
Alpha General: $ALPHA_GENERAL
Norm Scale: $NORM_SCALE
Max New Tokens: $MAX_NEW_TOKENS
Min New Tokens: $MIN_NEW_TOKENS
EOF

# Threshold values to test
THRESHOLDS=(0.018 0.022 0.026 0.030 0.040 5.0)

echo "Testing ${#THRESHOLDS[@]} thresholds: ${THRESHOLDS[*]}"
echo ""

# Run benchmarks for each threshold
for tau in "${THRESHOLDS[@]}"; do
    echo "🧪 Testing threshold τ = $tau"
    
    # Create output directory for this threshold
    TAU_DIR="$SWEEP_DIR/tau_${tau}"
    
    # Run benchmark
    python tools/run_benchmark_lamp2.py \
        --data_path "$DATA_PATH" \
        --split "$SPLIT" \
        --limit "$LIMIT" \
        --seed "$SEED" \
        --alpha_personal "$ALPHA_PERSONAL" \
        --alpha_general "$ALPHA_GENERAL" \
        --norm_scale "$NORM_SCALE" \
        --edit_gate_threshold "$tau" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --min_new_tokens "$MIN_NEW_TOKENS" \
        --repetition_penalty 1.0 \
        --out_dir "$TAU_DIR" || {
            echo "❌ Failed to run benchmark for τ = $tau"
            continue
        }
    
    echo "✅ Completed τ = $tau"
    echo ""
done

echo "📊 Generating sweep summary..."

# Create sweep summary
python - << EOF
import os
import csv
import json
from pathlib import Path

sweep_dir = Path("$SWEEP_DIR")
results = []

# Collect results from each threshold
for tau_dir in sorted(sweep_dir.glob("tau_*")):
    tau = tau_dir.name.replace("tau_", "")
    summary_csv = tau_dir / "summary.csv"
    
    if summary_csv.exists():
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            row["threshold"] = float(tau)
            results.append(row)
        print(f"✅ Loaded results for τ = {tau}")
    else:
        print(f"❌ No results found for τ = {tau}")

# Write combined summary
if results:
    # Sort by threshold
    results.sort(key=lambda x: x["threshold"])
    
    # CSV summary
    summary_path = sweep_dir / "sweep_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ["threshold", "n", "baseline_acc", "chameleon_acc", "delta_acc", 
                     "mcnemar_b", "mcnemar_c", "p_value", "valid_bl_rate", "valid_ch_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"📁 Summary written to: {summary_path}")
    
    # Find best threshold
    best_result = max(results, key=lambda x: float(x.get("delta_acc", 0)))
    best_tau = best_result["threshold"]
    best_delta = float(best_result["delta_acc"])
    best_p = float(best_result["p_value"])
    
    # Markdown report
    md_path = sweep_dir / "sweep_report.md" 
    with open(md_path, 'w') as f:
        f.write("# Gate Threshold Sweep Report\\n\\n")
        f.write(f"**Best Result:** τ = {best_tau}, Δacc = {best_delta:+.4f}, p = {best_p:.3g}\\n\\n")
        
        f.write("## All Results\\n\\n")
        f.write("| Threshold | Δ Accuracy | P-Value | Status |\\n")
        f.write("|-----------|------------|---------|---------|\\n")
        
        for r in results:
            tau = r["threshold"]
            delta = float(r["delta_acc"])
            p = float(r["p_value"])
            
            if p < 0.05:
                status = "✅ Significant"
            elif delta > 0:
                status = "📈 Positive"
            else:
                status = "📊 Neutral"
                
            f.write(f"| {tau} | {delta:+.4f} | {p:.3g} | {status} |\\n")
    
    print(f"📊 Report written to: {md_path}")
    print(f"🎯 Best threshold: τ = {best_tau} (Δacc = {best_delta:+.4f}, p = {best_p:.3g})")
    
else:
    print("❌ No results collected")
EOF

echo ""
echo "🎉 Threshold sweep complete!"
echo "📁 Results saved to: $SWEEP_DIR"
echo ""
echo "Next steps:"
echo "1. Review sweep_report.md for best threshold"
echo "2. Use best threshold for production evaluation"
echo "3. Consider running larger sample size for final validation"