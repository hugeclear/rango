#!/bin/bash

# LaMP-2 Benchmark with Auto Result Display
# Usage: ./run_with_results.sh [LIMIT] [OUTPUT_DIR]

LIMIT=${1:-100}
OUTPUT_DIR=${2:-"results/auto_$(date +%Y%m%d_%H%M%S)"}

echo "🚀 Starting LaMP-2 Benchmark (n=$LIMIT)..."
echo "📁 Output directory: $OUTPUT_DIR"
echo "⏰ Start time: $(date)"
echo

# Run benchmark
python tools/run_benchmark_lamp2.py \
  --config_path config/lamp2_eval_config.yaml \
  --data_path data --split test --limit $LIMIT --seed 42 \
  --prior_mode user --prior_fallback global \
  --prior_beta 1.5 --score_temp 3.0 --score_norm avg \
  --out_dir $OUTPUT_DIR

echo
echo "✅ Benchmark completed!"
echo "⏰ End time: $(date)"
echo

# Display results if successful
if [ -f "$OUTPUT_DIR/summary.md" ]; then
    echo "=== 📊 RESULTS ===" 
    cat $OUTPUT_DIR/summary.md
    echo
    echo "=== 🔍 EDITING EFFECTS ANALYSIS ===" 
    python tools/detect_editing_effects.py $OUTPUT_DIR/predictions.jsonl
    echo
    echo "📁 Full results saved to: $OUTPUT_DIR"
else
    echo "❌ Results not found. Check for errors above."
fi