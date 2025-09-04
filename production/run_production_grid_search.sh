#!/bin/bash
# Production Grid Search Script (with generation config fix)
# Generated: 20250829_200821

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
TIMEOUT=3600  # 60 minutes

echo "🔍 Starting Production Grid Search"
echo "=================================="
echo "Using greedy mode to avoid generation config conflicts"
echo "Timeout: ${TIMEOUT}s"
echo "=================================="

# Run with greedy mode to avoid do_sample issues
timeout $TIMEOUT python -c "
import sys
sys.path.append('.')

from chameleon_evaluator import ChameleonEvaluator
import json
import time
import numpy as np
from scipy import stats

# Test multiple parameter combinations with greedy mode
configs = [
    (0.2, -0.05, ['model.layers.20']),
    (0.4, -0.05, ['model.layers.27']), 
    (0.4, -0.05, ['model.layers.20', 'model.layers.27']),
    (0.6, -0.1, ['model.layers.20', 'model.layers.27']),
    (0.3, -0.02, ['model.layers.20', 'model.layers.24', 'model.layers.27'])
]

print(f'🧪 Testing {len(configs)} configurations...')

results = []
for i, (alpha_p, alpha_g, layers) in enumerate(configs):
    print(f'📊 Configuration {i+1}/{len(configs)}: α_p={alpha_p}, α_g={alpha_g}, layers={len(layers)}')
    
    try:
        evaluator = ChameleonEvaluator(
            config_path='production/production_config.yaml',
            data_path='./chameleon_prime_personalization/data',
            decoding_mode='greedy'
        )
        
        result = evaluator.run_evaluation(
            mode='demo',  # Use smaller dataset for quick testing
            alpha_override=alpha_p,
            beta_override=alpha_g,
            layers_override=layers,
            max_users_override=10
        )
        
        if 'baseline' in result and 'chameleon' in result:
            baseline_acc = result['baseline'].accuracy
            chameleon_acc = result['chameleon'].accuracy
            improvement = chameleon_acc - baseline_acc
            improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
            
            results.append({
                'alpha_p': alpha_p,
                'alpha_g': alpha_g,
                'layers': layers,
                'baseline_accuracy': baseline_acc,
                'chameleon_accuracy': chameleon_acc,
                'improvement_pct': improvement_pct,
                'success': True
            })
            
            print(f'   Result: {improvement_pct:+.1f}% improvement')
        else:
            print(f'   Result: Failed to get complete results')
            
    except Exception as e:
        print(f'   Error: {e}')
        continue

# Find best configuration
if results:
    best_result = max(results, key=lambda x: x['improvement_pct'])
    print(f'\n🏆 Best Configuration:')
    print(f'   α_personal: {best_result["alpha_p"]}')
    print(f'   α_general: {best_result["alpha_g"]}')  
    print(f'   Target layers: {best_result["layers"]}')
    print(f'   Improvement: {best_result["improvement_pct"]:+.1f}%')
    
    # Save best configuration
    with open('production/optimal_parameters.json', 'w') as f:
        json.dump(best_result, f, indent=2)
    
    print(f'\n✅ Production grid search completed!')
    print(f'💾 Optimal parameters saved to: production/optimal_parameters.json')
else:
    print(f'\n❌ No successful configurations found')
    exit(1)
"

echo "🎉 Production grid search completed!"
