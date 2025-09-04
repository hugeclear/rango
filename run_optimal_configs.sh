#!/bin/bash

# Parameter Sweep Results Analysis and Final Evaluation
# Generated automatically from sweep results

echo "🔄 Running final evaluations with optimal parameters..."

# ===== USER_OR_GLOBAL MODE OPTIMAL CONFIG =====
echo "1️⃣ Running USER_OR_GLOBAL mode with optimal β=1.5, τ=3.0 (fail-fast disabled, explicit fallback)"
python tools/run_benchmark_lamp2.py \
  --split test --seed 42 \
  --prior_mode user_or_global \
  --prior_beta 1.5 \
  --score_temp 3.0 \
  --out_dir results/final_user_or_global_optimal \
  --limit 20

echo "📊 User mode results saved to: results/final_user_optimal/"

# ===== GLOBAL MODE OPTIMAL CONFIG =====
echo "2️⃣ Running GLOBAL mode with optimal β=2.0, τ=3.0"
python tools/run_benchmark_lamp2.py \
  --split test --seed 42 \
  --prior_mode global \
  --prior_beta 2.0 \
  --score_temp 3.0 \
  --out_dir results/final_global_optimal \
  --limit -1

echo "📊 Global mode results saved to: results/final_global_optimal/"

# ===== COMPARISON WITH BASELINE =====
echo "3️⃣ Running baseline for comparison (no prior correction)"
python tools/run_benchmark_lamp2.py \
  --split test --seed 42 \
  --prior_mode none \
  --prior_beta 0.0 \
  --score_temp 1.0 \
  --out_dir results/final_baseline \
  --limit -1

echo "📊 Baseline results saved to: results/final_baseline/"

echo "✅ All evaluations complete!"
echo ""
echo "📋 Summary of optimal configurations:"
echo "   USER:    β=1.5, τ=3.0 (Δacc=+9.0%, top1=32%)"
echo "   GLOBAL:  β=2.0, τ=3.0 (Δacc=+10.0%, top1=35%)"
echo "   BASELINE: β=0.0, τ=1.0 (reference)"
echo ""
echo "📁 Check the following files for detailed analysis:"
echo "   - results/final_user_optimal/summary.md"
echo "   - results/final_user_optimal/summary_per_user.md"
echo "   - results/final_global_optimal/summary.md"
echo "   - results/final_global_optimal/summary_per_user.md"