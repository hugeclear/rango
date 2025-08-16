#!/usr/bin/env bash
# Production RC Evaluation Script
# Purpose: Run V2 (Gate + Curriculum + Anti-Hub) evaluation with conservative settings
set -euo pipefail

echo "🚀 Starting Production RC Evaluation (V2 Conservative)"
echo "=================================================="

# Create output directory
mkdir -p runs/prod_rc

# Set GPU device if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Run evaluation
echo "📊 Running V2 evaluation with conservative curriculum..."
conda run -n faiss310 python scripts/run_w2_evaluation.py \
  --config config/prod_cfs_v2.yaml \
  --generate-report \
  --strict \
  --verbose \
  | tee runs/prod_rc/out.log

echo ""
echo "✅ Production RC evaluation completed!"
echo "📂 Output: runs/prod_rc/out.log"
echo "=================================================="