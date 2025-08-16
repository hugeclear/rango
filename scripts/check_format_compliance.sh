#!/usr/bin/env bash
# Format Compliance Check Script
# Purpose: Run ablation study and verify format compliance rate meets threshold

set -euo pipefail

echo "🔍 Format Compliance Verification"
echo "=================================="

# Configuration
THRESHOLD=0.98
SAMPLE_SIZE=8
DATA_PATH="/home/nakata/master_thesis/rango/data/evaluation/lamp2_backup_eval.jsonl"
RUNS_DIR="runs/format_compliance_check"
PATTERN="regex:^Answer:\s*([A-Za-z0-9_\- ]+)\s*$"

echo "📊 Configuration:"
echo "   Threshold: ${THRESHOLD} (98%)"
echo "   Sample Size: ${SAMPLE_SIZE}"
echo "   Data: ${DATA_PATH}"
echo "   Pattern: ${PATTERN}"

# Ensure data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "❌ Data file not found: $DATA_PATH"
    echo "   Please ensure the evaluation data is available"
    exit 1
fi

# Create test data subset
TEST_DATA="/tmp/lamp2_compliance_test.jsonl"
head -n "$SAMPLE_SIZE" "$DATA_PATH" > "$TEST_DATA"

echo ""
echo "🧪 Running Format Compliance Test..."
echo "   Command: conda run -n faiss310 python scripts/verification/ablation_study.py"

# Run ablation study with strict format checking
if conda run -n faiss310 python scripts/verification/ablation_study.py \
    --data "$TEST_DATA" \
    --runs-dir "$RUNS_DIR" \
    --treatments gate_curriculum \
    --seed 42 \
    --strict-output "$PATTERN" \
    --reask-on-format-fail \
    --reask-max-retries 2 \
    --reask-temperature 0.0 \
    --decoding-temperature 0.0 \
    --decoding-top-p 0.0 \
    --decoding-max-tokens 8 \
    --decoding-stop-tokens "\\n" \
    --selector "cos+tags+ppr" \
    --selector-weights "alpha=1.0,beta=0.4,gamma=0.6,lambda=0.3" \
    --mmr-lambda 0.3 \
    --adaptive-k "min=1,max=5,tau=0.05" \
    --neg-curriculum "easy:1,medium:0,hard:0" \
    --anti-hub on \
    --ppr-restart 0.15 \
    --hub-degree-cap 200 \
    --generate-report > /tmp/compliance_output.log 2>&1; then
    
    echo "✅ Ablation study completed successfully"
else
    echo "❌ Ablation study failed"
    echo "   Check log: /tmp/compliance_output.log"
    cat /tmp/compliance_output.log
    exit 1
fi

echo ""
echo "📈 Extracting Format Compliance Rate..."

# Extract compliance rate from output
COMPLIANCE_RATE=""
if grep -q "Format Compliance Rate:" /tmp/compliance_output.log; then
    COMPLIANCE_RATE=$(grep "Format Compliance Rate:" /tmp/compliance_output.log | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    echo "   Found compliance rate: ${COMPLIANCE_RATE}"
elif [[ -f "${RUNS_DIR}/ablation_study_report.json" ]]; then
    # Try to extract from JSON report
    COMPLIANCE_RATE=$(python3 -c "
import json
try:
    with open('${RUNS_DIR}/ablation_study_report.json', 'r') as f:
        data = json.load(f)
    rate = data.get('format_compliance_rate', 0.0)
    print(f'{rate:.3f}')
except Exception as e:
    print('0.000')
    ")
    echo "   Extracted from report: ${COMPLIANCE_RATE}"
else
    echo "⚠️  Could not extract compliance rate from output"
    COMPLIANCE_RATE="0.000"
fi

# Validate compliance rate
if [[ -z "$COMPLIANCE_RATE" ]] || [[ "$COMPLIANCE_RATE" == "0.000" ]]; then
    echo "❌ Failed to extract valid compliance rate"
    echo "   This may indicate format compliance tracking is not working"
    exit 1
fi

echo ""
echo "🎯 Format Compliance Assessment:"
echo "   Measured Rate: ${COMPLIANCE_RATE} ($(python3 -c "print(f'{float(\"$COMPLIANCE_RATE\")*100:.1f}%')"))"
echo "   Required Rate: ${THRESHOLD} ($(python3 -c "print(f'{float(\"$THRESHOLD\")*100:.1f}%')"))"

# Compare with threshold
MEETS_THRESHOLD=$(python3 -c "
rate = float('$COMPLIANCE_RATE')
threshold = float('$THRESHOLD')
print('true' if rate >= threshold else 'false')
")

if [[ "$MEETS_THRESHOLD" == "true" ]]; then
    echo "   Result: ✅ PASS - Format compliance meets threshold"
    echo ""
    echo "🎉 FORMAT COMPLIANCE VERIFICATION SUCCESSFUL!"
    echo "   The system generates outputs in the required 'Answer: <tag>' format"
    echo "   with sufficient reliability for production use."
    exit 0
else
    echo "   Result: ❌ FAIL - Format compliance below threshold"
    SHORTFALL=$(python3 -c "
rate = float('$COMPLIANCE_RATE')
threshold = float('$THRESHOLD')
shortfall = threshold - rate
print(f'{shortfall:.3f}')
")
    SHORTFALL_PCT=$(python3 -c "
rate = float('$COMPLIANCE_RATE')
threshold = float('$THRESHOLD')
shortfall = threshold - rate
print(f'{shortfall*100:.1f}%')
")
    
    echo ""
    echo "⚠️  FORMAT COMPLIANCE INSUFFICIENT"
    echo "   Shortfall: ${SHORTFALL} (${SHORTFALL_PCT})"
    echo ""
    echo "💡 RECOMMENDATIONS:"
    echo "   1. Strengthen prompt instructions for format adherence"
    echo "   2. Increase reask-max-retries for more aggressive correction"
    echo "   3. Lower reask-temperature for more deterministic retry generation"
    echo "   4. Review and optimize regex pattern for format detection"
    echo "   5. Consider model fine-tuning for better instruction following"
    echo ""
    echo "❌ FORMAT COMPLIANCE CHECK FAILED"
    exit 1
fi