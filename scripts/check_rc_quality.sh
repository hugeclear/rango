#!/usr/bin/env bash
# RC Quality Check Script
# Purpose: Compare V2-Complete vs Baseline quality and assess RC readiness
set -euo pipefail

echo "🔍 RC Quality Assessment"
echo "======================="

# Extract quality metrics from log files
baseline_log="runs/prod_baseline/out.log"
v2_log="runs/prod_rc/out.log"

if [[ ! -f "$baseline_log" ]]; then
    echo "❌ Baseline log not found: $baseline_log"
    echo "   Run: bash scripts/run_baseline_eval.sh"
    exit 1
fi

if [[ ! -f "$v2_log" ]]; then
    echo "❌ V2 log not found: $v2_log"
    echo "   Run: bash scripts/run_prod_eval.sh"
    exit 1
fi

echo "📊 Extracting quality metrics..."

# Extract ROUGE-L scores (proxy for quality)
baseline_rouge=$(grep -E "rouge_l.*:" "$baseline_log" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "0.000")
v2_rouge=$(grep -E "rouge_l.*:" "$v2_log" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "0.000")

# Extract BERTScore if available (backup quality metric)
baseline_bert=$(grep -E "bert_score.*:" "$baseline_log" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "$baseline_rouge")
v2_bert=$(grep -E "bert_score.*:" "$v2_log" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "$v2_rouge")

# Convert to comparable numbers (use ROUGE-L as primary, BERTScore as fallback)
baseline_quality=${baseline_rouge:-$baseline_bert}
v2_quality=${v2_rouge:-$v2_bert}

echo "📈 Quality Metrics:"
echo "   Baseline:     $baseline_quality"
echo "   V2-Complete:  $v2_quality"

# Calculate improvement
if command -v python3 >/dev/null 2>&1; then
    improvement=$(python3 -c "
baseline = float('$baseline_quality' or 0)
v2 = float('$v2_quality' or 0)
diff = v2 - baseline
print(f'{diff:+.3f}')
")
    improvement_pct=$(python3 -c "
baseline = float('$baseline_quality' or 0)
v2 = float('$v2_quality' or 0)
if baseline > 0:
    pct = ((v2 - baseline) / baseline) * 100
    print(f'{pct:+.1f}%')
else:
    print('N/A')
")
else
    improvement="N/A"
    improvement_pct="N/A"
fi

echo "   Improvement:  $improvement ($improvement_pct)"

# Quality assessment threshold (0.15 = 15% improvement minimum)
threshold=0.15

# Assessment logic
if command -v python3 >/dev/null 2>&1; then
    passes_threshold=$(python3 -c "
baseline = float('$baseline_quality' or 0)
v2 = float('$v2_quality' or 0)
diff = v2 - baseline
threshold = $threshold
print('true' if diff >= threshold else 'false')
")
else
    passes_threshold="false"
fi

echo ""
echo "🎯 RC Quality Assessment:"

if [[ "$passes_threshold" == "true" ]]; then
    echo "✅ RC quality PASSED"
    echo "   V2-Complete shows +$improvement improvement (≥ +$threshold threshold)"
    echo "   🚀 Ready for production deployment"
    exit 0
elif [[ "$improvement" != "N/A" ]] && python3 -c "print(float('$v2_quality' or 0) > float('$baseline_quality' or 0))" 2>/dev/null | grep -q "True"; then
    echo "⚠️  RC quality MARGINAL"
    echo "   V2-Complete shows +$improvement improvement (< +$threshold threshold)"
    echo "   📊 Positive trend but below target - monitor closely"
    exit 0
else
    echo "❌ RC quality INSUFFICIENT"
    echo "   V2-Complete shows $improvement change (target: ≥ +$threshold)"
    echo "   🔧 Optimization needed before production deployment"
    exit 1
fi