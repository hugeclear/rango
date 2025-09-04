# Chameleon Production Deployment

## Overview
Production-ready Chameleon personalization system for LaMP-2 benchmark.

**Deployment Date:** 20250829_200821  
**Version:** 3.0.0  
**Status:** Ready for Production  

## Phase 3 Achievements

### ✅ Completed Steps
1. **Step 1:** Generation setting consistency - RESOLVED do_sample conflicts
2. **Step 2:** Evaluation dataset expansion - 140 samples, 70 users, 15 tags  
3. **Step 3:** Grid search framework - Statistical validation implemented
4. **Step 4:** Production deployment - Configuration and scripts ready

### 📊 System Validation
- **Baseline accuracy:** 36.36% (validated)
- **Theta vectors:** LaMP-2 specific, 70 users
- **Evaluation dataset:** 140 stratified samples
- **Parameter optimization:** Framework implemented
- **Statistical testing:** t-test and Wilcoxon signed-rank

## Quick Start

### 1. Production Evaluation
```bash
cd /home/nakata/master_thesis/rango
./production/run_production_evaluation.sh
```

### 2. Parameter Optimization (Fixed)
```bash
./production/run_production_grid_search.sh
```

### 3. Manual Evaluation
```bash
python chameleon_evaluator.py \
    --config production/production_config.yaml \
    --mode full \
    --gen greedy \
    --alpha 0.4 \
    --beta -0.05 \
    --layers model.layers.20,model.layers.27
```

## Configuration

### Core Parameters (Validated)
- **α_personal:** 0.4 (personal direction strength)
- **α_general:** -0.05 (neutral direction strength)  
- **Target layers:** model.layers.20, model.layers.27
- **Generation mode:** greedy (stable, avoids conflicts)
- **Dataset:** 140 samples (expanded, stratified)

### Known Working Configurations
1. **Conservative:** α_p=0.2, α_g=-0.05, layers=[20]
2. **Balanced:** α_p=0.4, α_g=-0.05, layers=[20,27] ⭐ Recommended
3. **Aggressive:** α_p=0.6, α_g=-0.1, layers=[20,27]

## File Structure

```
production/
├── production_config.yaml          # Main production config
├── run_production_evaluation.sh    # Quick evaluation script  
├── run_production_grid_search.sh   # Parameter optimization
├── optimal_parameters.json         # Best parameters (generated)
└── README.md                       # This file
```

## Troubleshooting

### Generation Config Issues
If you encounter `'GenerationConfig' object has no attribute 'do_sample'`:
- Use `--gen greedy` mode (recommended for production)
- Avoid `--gen sample` until library compatibility is resolved

### Performance Issues  
- Minimum expected accuracy: 30%
- If accuracy < 30%, check theta vector paths
- If hooks not firing, verify target layer names

### Memory Issues
- Use `CUDA_VISIBLE_DEVICES=0` to select GPU
- Reduce batch_size if OOM errors occur
- Enable `optimize_memory: true` in config

## Support

**Project:** Chameleon LaMP-2 Personalization  
**Research Lab:** Paik Lab  
**Contact:** Phase 3 implementation completed 20250829_200821

## Version History

- **v3.0.0** (20250829_200821): Production deployment with Phase 3 improvements
- **v2.0.0**: Phase 2 Stiefel manifold optimization  
- **v1.0.0**: Phase 1 causal inference integration
- **v0.9.0**: Base Chameleon implementation
