#!/bin/bash
# 
# GraphRAG-CFS-Chameleon Week 2 Evaluation Script
# Runs comprehensive evaluation across multiple conditions
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/results/w2"
CONFIG_FILE="$PROJECT_ROOT/config/w2_evaluation.yaml"
PYTHON_ENV="$PROJECT_ROOT/chameleon_prime_personalization/.venv/bin/python"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/eval_${TIMESTAMP}.log"
mkdir -p "$OUTPUT_DIR"

echo "=== GraphRAG-CFS-Chameleon Week 2 Evaluation ===" | tee "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Project Root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

# Check environment
echo "Checking environment..." | tee -a "$LOG_FILE"
if [[ ! -f "$PYTHON_ENV" ]]; then
    echo "ERROR: Python environment not found at $PYTHON_ENV" | tee -a "$LOG_FILE"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "WARNING: Config file not found at $CONFIG_FILE, using default" | tee -a "$LOG_FILE"
    CONFIG_FILE=""
fi

# Set CUDA device if available
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
echo "Starting evaluation..." | tee -a "$LOG_FILE"

RUN_ID="w2_eval_${TIMESTAMP}"

# Execute Python evaluation script
$PYTHON_ENV "$PROJECT_ROOT/scripts/run_w2_evaluation.py" \
    --run-id "$RUN_ID" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --conditions "legacy_chameleon,graphrag_v1,graphrag_v1_diversity,cfs_enabled,cfs_disabled" \
    --include-bertscore \
    --significance-test \
    --generate-report \
    2>&1 | tee -a "$LOG_FILE"

EVAL_EXIT_CODE=$?

if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "Evaluation completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved to: $OUTPUT_DIR/$RUN_ID/" | tee -a "$LOG_FILE"
    
    # Display summary
    echo "" | tee -a "$LOG_FILE"
    echo "=== EVALUATION SUMMARY ===" | tee -a "$LOG_FILE"
    
    if [[ -f "$OUTPUT_DIR/$RUN_ID/ablation.csv" ]]; then
        echo "Results overview:" | tee -a "$LOG_FILE"
        head -6 "$OUTPUT_DIR/$RUN_ID/ablation.csv" | tee -a "$LOG_FILE"
    fi
    
    if [[ -f "$OUTPUT_DIR/$RUN_ID/evaluation_report.md" ]]; then
        echo "" | tee -a "$LOG_FILE"
        echo "Full report available at: $OUTPUT_DIR/$RUN_ID/evaluation_report.md" | tee -a "$LOG_FILE"
    fi
    
else
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EVAL_EXIT_CODE
fi

echo "=================================================" | tee -a "$LOG_FILE"
echo "Evaluation completed at: $(date)" | tee -a "$LOG_FILE"