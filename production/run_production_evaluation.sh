#!/bin/bash
# Production Chameleon Evaluation Script
# Generated: 20250829_200821

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=0
TIMEOUT=1800  # 30 minutes
CONFIG="production/production_config.yaml"
MODE="full"

echo "🚀 Starting Production Chameleon Evaluation"
echo "========================================="
echo "Config: $CONFIG"
echo "Mode: $MODE"
echo "Timeout: ${TIMEOUT}s"
echo "========================================="

# Run evaluation with timeout
timeout $TIMEOUT python chameleon_evaluator.py \
    --config "$CONFIG" \
    --mode "$MODE" \
    --gen greedy \
    --data_path "./chameleon_prime_personalization/data"

if [ $? -eq 0 ]; then
    echo "✅ Production evaluation completed successfully!"
    echo "📊 Results saved in results/ directory"
else
    echo "❌ Production evaluation failed!"
    exit 1
fi
