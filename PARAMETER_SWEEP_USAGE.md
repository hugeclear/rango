# Parameter Sweep Usage Guide

## Overview

The parameter sweep functionality automatically optimizes β (prior_beta) and temperature (score_temp) parameters to balance accuracy improvement and distribution flattening. This addresses the challenge where optimal parameters vary by input difficulty and user preference diversity.

## Key Concepts

### Prior Correction Modes
- **none**: No prior correction (baseline PMI)
- **empty**: Empty context PMI correction  
- **global**: Global dataset distribution correction
- **user**: User-specific distribution with global fallback

### Distribution Metrics
- **Entropy**: Measures prediction diversity (higher = more diverse)
- **KL to uniform**: Distance from uniform distribution (lower = flatter)
- **Top-1 share**: Fraction of most frequent prediction (lower = flatter)

### Parameter Optimization
- **β (prior_beta)**: Weight for prior correction term (0.0 = no correction, 2.0 = strong correction)
- **Temperature**: Softmax temperature for flattening (1.0 = original, 5.0 = very flat)

## Usage Examples

### Basic Parameter Sweep

```bash
python tools/run_benchmark_lamp2.py \
  --data_path data \
  --split test \
  --limit 100 \
  --prior_mode global \
  --prior_beta_sweep "0.0,0.5,1.0,1.5,2.0" \
  --score_temp_sweep "1.0,2.0,3.0,5.0" \
  --target_entropy 2.3 \
  --max_kl_to_uniform 0.25 \
  --max_top1_share 0.35 \
  --select_by delta_acc \
  --out_dir results/sweep/lamp2_optimization
```

### Advanced Configuration

```bash
python tools/run_benchmark_lamp2.py \
  --data_path data \
  --split test \
  --limit 500 \
  --prior_mode user \
  --prior_alpha 1.5 \
  --prior_beta_sweep "0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0" \
  --score_temp_sweep "1.0,1.5,2.0,2.5,3.0,4.0,5.0" \
  --target_entropy 2.5 \
  --max_kl_to_uniform 0.2 \
  --max_top1_share 0.3 \
  --select_by ch_acc \
  --seed 42 \
  --out_dir results/sweep/comprehensive_tuning
```

### Quick Beta-Only Sweep

```bash
python tools/run_benchmark_lamp2.py \
  --data_path data \
  --split test \
  --limit 200 \
  --prior_mode global \
  --prior_beta_sweep "0.5,1.0,2.0" \
  --score_temp 2.0 \
  --target_entropy 2.0 \
  --out_dir results/sweep/beta_only
```

## Command Line Arguments

### Core Parameters
- `--data_path`: Dataset path (default: "data")
- `--split`: Dataset split to evaluate (default: "test") 
- `--limit`: Number of samples (-1 for all)
- `--prior_mode`: Prior correction mode (none/empty/global/user)

### Sweep Configuration
- `--prior_beta_sweep`: Comma-separated β values (e.g., "0.0,0.5,1.0,2.0")
- `--score_temp_sweep`: Comma-separated temperature values (e.g., "1.0,2.0,3.0")

### Constraint Settings
- `--target_entropy`: Minimum entropy requirement (default: 2.3)
- `--max_kl_to_uniform`: Maximum KL divergence to uniform (default: 0.25)
- `--max_top1_share`: Maximum top-1 label share (default: 0.35)

### Selection Criterion
- `--select_by`: Optimization criterion
  - `delta_acc`: Accuracy improvement (Chameleon - Baseline)
  - `ch_acc`: Absolute Chameleon accuracy
  - `entropy`: Distribution entropy

## Output Files

### 1. sweep.csv
Detailed results for all parameter combinations:
```csv
config,prior_beta,score_temp,baseline_acc,chameleon_acc,delta_acc,p_value,entropy,kl_to_uniform,top1_share,meets_entropy,meets_kl,meets_top1,meets_constraints,selection_score
beta0.5_temp1.0,0.50,1.00,0.1500,0.2000,0.0500,0.034,1.85,0.31,0.45,False,False,False,False,-inf
beta1.0_temp2.0,1.00,2.00,0.1500,0.2200,0.0700,0.019,2.41,0.18,0.28,True,True,True,True,0.0700
```

### 2. best_config.json
Best configuration and sweep summary:
```json
{
  "best_config": {
    "config": "beta1.0_temp2.0",
    "prior_beta": 1.0,
    "score_temp": 2.0,
    "baseline_acc": 0.15,
    "chameleon_acc": 0.22,
    "delta_acc": 0.07,
    "entropy": 2.41,
    "meets_constraints": true
  },
  "sweep_summary": {
    "total_configs": 20,
    "constraint_pass_count": 3
  }
}
```

### 3. summary_best.md
Human-readable report with best configuration and performance analysis.

### 4. Standard Benchmark Outputs
When sweep finds a valid configuration, also generates:
- `predictions.jsonl`: Per-sample predictions with best parameters
- `summary.csv`: Aggregate metrics with best parameters  
- `summary.md`: Detailed report with best parameters
- `summary_per_user.csv/md`: Per-user analysis with best parameters

## Constraint Guidelines

### Entropy Targets
- **Low diversity tasks**: 1.5-2.0 (fewer valid labels)
- **Medium diversity tasks**: 2.0-2.5 (balanced label set)
- **High diversity tasks**: 2.5+ (many equally valid labels)

### KL Divergence Limits
- **Strict flattening**: 0.1-0.2 (very uniform)
- **Moderate flattening**: 0.2-0.3 (balanced)
- **Mild flattening**: 0.3-0.5 (preserve some bias)

### Top-1 Share Limits
- **Prevent dominance**: 0.2-0.3 (no single label >30%)
- **Allow preference**: 0.3-0.4 (some preference OK)
- **Minimal constraint**: 0.4+ (mainly prevent 100% bias)

## Performance Considerations

### Computational Cost
- Each configuration requires full model inference
- Cost = β_values × temp_values × sample_count
- Recommend: Start with 3-5 values per parameter, 100-200 samples

### Parameter Ranges
- **Beta**: 0.0 (no correction) to 2.0 (strong correction)
- **Temperature**: 1.0 (original) to 5.0 (very flat)
- Values >2.0 (beta) or >5.0 (temp) rarely beneficial

### Optimization Strategy
1. **Quick scan**: 3×3 grid, 100 samples
2. **Focused search**: 5×5 grid around promising region, 200-500 samples  
3. **Final tuning**: 7×7 grid with best ranges, full dataset

## Troubleshooting

### No Valid Configurations
```bash
❌ No valid configuration found meeting constraints
   Consider relaxing constraints or expanding parameter ranges
```

**Solutions**:
- Increase `--target_entropy` threshold
- Increase `--max_kl_to_uniform` limit
- Increase `--max_top1_share` limit
- Expand parameter ranges (more β/temp values)

### All Configurations Fail Constraints
- Dataset may have inherent label imbalance
- Try `--prior_mode user` for personalization
- Relax constraints progressively
- Check if baseline predictions are too biased

### Long Runtime
- Reduce `--limit` for faster testing
- Use fewer sweep values initially
- Consider background execution with `nohup`

## Example Workflows

### Workflow 1: Quick Optimization
```bash
# Step 1: Quick 3×3 sweep with relaxed constraints
python tools/run_benchmark_lamp2.py \
  --prior_beta_sweep "0.5,1.0,1.5" \
  --score_temp_sweep "1.0,2.0,3.0" \
  --limit 100 \
  --target_entropy 2.0 \
  --max_top1_share 0.4

# Step 2: Check best_config.json and refine ranges
```

### Workflow 2: Comprehensive Search  
```bash
# Step 1: Broad sweep
python tools/run_benchmark_lamp2.py \
  --prior_beta_sweep "0.0,0.5,1.0,1.5,2.0" \
  --score_temp_sweep "1.0,1.5,2.0,3.0,5.0" \
  --limit 200

# Step 2: Focused sweep around best region
python tools/run_benchmark_lamp2.py \
  --prior_beta_sweep "0.8,1.0,1.2" \
  --score_temp_sweep "1.8,2.0,2.2" \
  --limit 500

# Step 3: Final validation with full dataset
python tools/run_benchmark_lamp2.py \
  --prior_beta 1.0 \
  --score_temp 2.0 \
  --limit -1
```

## Integration with Existing Workflows

The parameter sweep is fully compatible with existing LaMP-2 benchmark workflows:

- All existing CLI arguments remain functional
- Sweep mode automatically detected when `--prior_beta_sweep` or `--score_temp_sweep` provided
- Output format matches standard benchmark results
- Can be integrated into automated evaluation pipelines

## Next Steps After Parameter Sweep

1. **Apply Best Configuration**: Use optimal β and temperature in production runs
2. **Validate Generalization**: Test best parameters on different datasets/splits  
3. **Document Findings**: Record optimal parameters for similar task types
4. **Automate Pipeline**: Integrate sweep into regular model evaluation