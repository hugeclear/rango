#!/usr/bin/env bash
# Baseline Evaluation Script
# Purpose: Run baseline evaluation for comparison and rollback safety
set -euo pipefail

echo "📊 Starting Baseline Evaluation (Rollback Safe)"
echo "==============================================="

# Create output directory
mkdir -p runs/prod_baseline

# Set GPU device if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Run baseline evaluation
echo "🔄 Running baseline evaluation..."
conda run -n faiss310 python scripts/run_w2_evaluation.py \
  --config config/prod_baseline.yaml \
  --generate-report \
  --strict \
  --verbose \
  | tee runs/prod_baseline/out.log

echo ""
echo "✅ Baseline evaluation completed!"
echo "📂 Output: runs/prod_baseline/out.log"
echo "==============================================="