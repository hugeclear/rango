# Production RC Quickstart Guide

**Version**: V2 Conservative Release Candidate  
**Created**: 2025-08-17  
**Purpose**: Production deployment guide for CFS-Chameleon V2 with conservative curriculum

## Overview

This document provides a quickstart guide for deploying the CFS-Chameleon V2 system with production-ready configurations. The RC (Release Candidate) includes conservative curriculum settings for stable deployment.

## Quick Start

### Prerequisites

- CUDA-compatible GPU (tested on A100)
- Python environment with required dependencies  
- Valid LaMP-2 dataset and artifacts
- Faiss310 conda environment (if using conda)

### Step 1: Configuration Files

Production configurations are available in `config/`:

```bash
# V2 Production RC (recommended)
config/prod_cfs_v2.yaml

# Baseline fallback
config/prod_baseline.yaml
```

### Step 2: Run Production Evaluation

```bash
# Option A: Direct execution (recommended)
bash scripts/run_prod_eval.sh

# Option B: Conda environment
conda run -n faiss310 python scripts/run_w2_evaluation.py \
  --config config/prod_cfs_v2.yaml \
  --generate-report --strict --verbose

# Baseline evaluation (for comparison)
bash scripts/run_baseline_eval.sh
```

### Step 3: Quality Assessment

```bash
# Run automated quality check
bash scripts/check_rc_quality.sh
```

Expected output:
- ✅ RC quality PASSED: Ready for production
- ⚠️ RC quality MARGINAL: Monitor closely  
- ❌ RC quality INSUFFICIENT: Optimization needed

## Configuration Details

### V2 Production RC Features

- **Conservative Curriculum**: `easy:1,medium:0,hard:0` (hard=0 for stability)
- **Composite Selector**: `cos+tags+ppr` with optimized weights
- **MMR Diversity**: λ=0.3 for balanced relevance/diversity
- **Adaptive K**: 1-5 range with τ=0.05 threshold
- **Anti-hub Sampling**: Enabled with degree cap 200
- **Strict Validation**: Regex-based output validation

### Baseline Configuration

- **Basic Selector**: No advanced features
- **Fixed K**: K=5 (no adaptation)
- **No Curriculum**: Empty curriculum learning
- **No Anti-hub**: Standard retrieval
- **Legacy Mode**: Compatibility fallback

## Monitoring & Rollback

### Health Checks

```bash
# Check evaluation logs
tail -f runs/prod_rc/out.log
tail -f runs/prod_baseline/out.log

# Monitor GPU usage
nvidia-smi -l 1
```

### Emergency Rollback

If V2 fails, immediately rollback to baseline:

```bash
# 1. Stop current evaluation
pkill -f run_w2_evaluation

# 2. Switch to baseline config
cp config/prod_baseline.yaml config/active_config.yaml

# 3. Run baseline evaluation
bash scripts/run_baseline_eval.sh

# 4. Verify baseline operation
bash scripts/check_rc_quality.sh
```

## Quality Thresholds

| Metric | Target | Status |
|--------|--------|--------|
| Improvement | ≥ +0.15 | PASS |
| Improvement | 0.0 - 0.15 | MARGINAL |
| Improvement | < 0.0 | INSUFFICIENT |

## Production Deployment Checklist

- [ ] V2 RC configuration tested
- [ ] Baseline configuration tested  
- [ ] Quality assessment PASS/MARGINAL
- [ ] GPU resources available
- [ ] Monitoring system ready
- [ ] Rollback procedure verified
- [ ] Team notified of deployment

## File Structure

```
config/
├── prod_cfs_v2.yaml        # V2 Production RC
└── prod_baseline.yaml      # Baseline fallback

scripts/
├── run_prod_eval.sh        # V2 evaluation
├── run_baseline_eval.sh    # Baseline evaluation
└── check_rc_quality.sh     # Quality assessment

runs/
├── prod_rc/                # V2 evaluation results
└── prod_baseline/          # Baseline results

docs/
└── prod_rc_quickstart.md   # This document
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   # Reduce batch size in config
   model:
     batch_size: 2  # From 4
   ```

2. **Model Loading Errors**
   ```bash
   # Check Hugging Face cache
   ls -la ~/.cache/huggingface/hub/
   
   # Clear cache if needed
   rm -rf ~/.cache/huggingface/hub/models--meta-llama*
   ```

3. **Asset Path Errors**
   ```bash
   # Verify artifact paths exist
   ls -la /home/nakata/master_thesis/rango/artifacts/20250810_053000/
   ```

4. **Zero Quality Metrics**
   - Model may not be generating valid "Answer: <TAG>" format
   - Check prediction samples in evaluation logs
   - Verify prompts and allowed tags configuration

### Support Contacts

- **Technical Issues**: Check logs and GitHub issues
- **Performance Issues**: Monitor GPU usage and batch sizes
- **Quality Issues**: Review prompts and model configuration

## Version History

- **2025-08-17**: Initial production RC setup
- **Conservative curriculum**: hard=0 for stability
- **Quality assessment**: Automated GO/HOLD decision system

---

**Next Steps**: After successful RC deployment, consider gradual curriculum progression and full-scale evaluation on complete LaMP-2 dataset.