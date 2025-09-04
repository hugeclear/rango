# LaMP-2 Benchmark Usage Guide

## 🎯 Quick Start

### Single Benchmark Run (500 samples)
```bash
python tools/run_benchmark_lamp2.py \
  --data_path data \
  --split test --limit 500 --seed 42 \
  --alpha_personal 2.75 --alpha_general -1.0 \
  --norm_scale 0.9 --edit_gate_threshold 0.022 \
  --max_new_tokens 2 --min_new_tokens 1 --repetition_penalty 1.0 \
  --out_dir results/bench/lamp2_smoke_ap275_ag-10_tau0022
```

### Automated Threshold Sweep
```bash
# Test multiple gate thresholds automatically
bash scripts/run_threshold_sweep.sh

# Or with custom parameters:
LIMIT=500 ALPHA_PERSONAL=2.75 ALPHA_GENERAL=-1.0 \
  bash scripts/run_threshold_sweep.sh
```

## 📊 Output Files

Each benchmark run creates:

### `predictions.jsonl`
Per-sample results in JSON Lines format:
```json
{"idx": 0, "gold": "drama", "baseline": "drama", "chameleon": "drama", 
 "baseline_ok": true, "chameleon_ok": true, "gate_value": 3.75, "gate_applied": true}
```

### `summary.csv`
Aggregate metrics:
```csv
n,baseline_acc,chameleon_acc,delta_acc,mcnemar_b,mcnemar_c,p_value,valid_bl_rate,valid_ch_rate
500,0.7240,0.7480,0.0240,15,3,0.003,0.990,0.994
```

### `summary.md`
Human-readable report with statistical analysis:
```markdown
# LaMP-2 Benchmark Summary

**Results:**
- Baseline Accuracy = **0.7240**
- Chameleon Accuracy = **0.7480** (Δ = +0.0240)
- McNemar exact p = **0.003** (b=15, c=3)

🎯 **Result: STATISTICALLY SIGNIFICANT improvement**
```

## 📈 Interpreting Results

### Key Metrics

1. **Δ Accuracy**: Chameleon accuracy - Baseline accuracy
   - Positive = improvement, negative = degradation

2. **McNemar p-value**: Statistical significance of the difference
   - p < 0.05 = significant improvement
   - p ≥ 0.05 = not statistically significant

3. **McNemar counts (b, c)**:
   - b = cases where Baseline failed but Chameleon succeeded
   - c = cases where Baseline succeeded but Chameleon failed
   - Higher b/c ratio = stronger Chameleon advantage

4. **Valid-Tag Rate**: Proportion of outputs that map to allowed tags
   - Should be >0.98 for both baseline and Chameleon

### Result Interpretation

| Δ Accuracy | P-Value | Interpretation |
|------------|---------|----------------|
| +0.02+     | <0.05   | ✅ **Significant improvement** |
| +0.01+     | <0.10   | 📈 **Promising trend** |  
| +0.005+    | >0.10   | 📊 **Marginal, increase sample size** |
| ±0.000     | >0.50   | 📋 **No difference** |
| -0.01-     | <0.05   | ❌ **Significant degradation** |

## 🔧 Parameter Optimization

### Gate Threshold (τ) Selection

Use threshold sweep to find optimal τ for target application rate:

```bash
# Sweep thresholds to find 30% application rate
bash scripts/run_threshold_sweep.sh
```

Expected application rates by threshold:
- τ = 0.018 → ~50% application
- τ = 0.022 → ~30% application  
- τ = 0.030 → ~15% application
- τ = 5.0   → ~0% application (baseline)

### Alpha Parameter Tuning

Test different α combinations:
```bash
for ap in 2.0 2.5 3.0; do
  for ag in -0.5 -1.0 -1.5; do
    python tools/run_benchmark_lamp2.py \
      --alpha_personal $ap --alpha_general $ag \
      --limit 500 --out_dir results/bench/alpha_${ap}_${ag}
  done
done
```

## 🧪 Validation Checklist

Before running production benchmarks:

### ✅ Pre-flight Checks
- [ ] Model loaded successfully
- [ ] Dataset accessible at specified path
- [ ] TwoStepPrefixProcessor creating valid sequences  
- [ ] Tag normalization handling sci-fi correctly
- [ ] Reproducibility seeds set (PYTHONHASHSEED=42)

### ✅ Sanity Checks
- [ ] Baseline accuracy > 0.5 (better than random)
- [ ] Valid-tag rate > 0.95 (constraint system working)
- [ ] Gate values > 0 when applied=True
- [ ] Generated text length ≥ 1 (min_new_tokens working)

### ✅ Statistical Validity
- [ ] Sample size ≥ 500 for initial testing
- [ ] Sample size ≥ 2000 for publication-ready results
- [ ] Multiple runs with different seeds show consistent trends
- [ ] Per-tag analysis shows broad improvements (not single-tag artifacts)

## 🚀 Production Workflow

### 1. Development Testing (Fast)
```bash
# Quick smoke test with 100 samples
python tools/run_benchmark_lamp2.py --limit 100 --seed 42
```

### 2. Parameter Optimization (Thorough)
```bash
# Threshold sweep with 500 samples
LIMIT=500 bash scripts/run_threshold_sweep.sh

# Select best threshold from sweep_report.md
BEST_TAU=0.022  # Example from sweep

# Alpha optimization with best threshold
for ap in 2.0 2.5 3.0; do
  python tools/run_benchmark_lamp2.py \
    --alpha_personal $ap --edit_gate_threshold $BEST_TAU \
    --limit 500 --seed 42
done
```

### 3. Final Validation (Publication)
```bash
# Large-scale evaluation with optimal parameters
python tools/run_benchmark_lamp2.py \
  --alpha_personal 2.75 --alpha_general -1.0 \
  --edit_gate_threshold 0.022 \
  --limit -1 --seed 42 \  # Full dataset
  --out_dir results/bench/final_validation
```

### 4. Cross-Validation
```bash
# Multiple seeds for robustness
for seed in 42 123 456 789; do
  python tools/run_benchmark_lamp2.py \
    --seed $seed --limit 2000 \
    --out_dir results/bench/cv_seed_$seed
done
```

## 📋 Troubleshooting

### Empty Generation
```bash
# Check constraints are not too restrictive
grep "prefix_processor_sequences" results/*/predictions.jsonl
# Should show >0 sequences for most samples
```

### Low Valid-Tag Rate
```bash
# Check tag normalization
python -c "
from tools.run_benchmark_lamp2 import LaMP2Benchmarker
b = LaMP2Benchmarker()
print(b.normalize_tag('sci fi'))  # Should output 'sci-fi'
"
```

### No Significant Improvement
1. Check gate application rate: Should be 20-40%
2. Verify direction vectors are computed correctly
3. Increase sample size (>2000 for robust statistics)
4. Try different α parameters or target layers

## 🎯 Expected Performance

**Typical LaMP-2 Results** (500 samples):
- Baseline accuracy: 0.65-0.75
- Chameleon improvement: +0.01 to +0.05
- Statistical significance: p < 0.05 with n≥1000
- Valid-tag rate: >0.98 for both methods

Ready to benchmark! Start with the smoke test and scale up based on results.