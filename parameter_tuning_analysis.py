#!/usr/bin/env python3
"""
Parameter Tuning Analysis
========================

Test different α/β parameter combinations to find ranges that produce measurable
performance differences while avoiding over-editing.

Based on findings:
- System is functional (hooks working, theta vectors loaded)
- Current α=0.4, β=-0.05 too conservative (edit ratio 0.82% vs target 2.5%)
- Need to find parameter ranges that produce meaningful differences
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.append('/home/nakata/master_thesis/rango')

from chameleon_evaluator import ChameleonEvaluator

def test_parameter_ranges():
    """Test different parameter combinations to find effective ranges"""
    print("🧪 PARAMETER TUNING ANALYSIS")
    print("=" * 60)
    
    # Parameter test matrix
    parameter_tests = [
        # Conservative range (current)
        (0.4, -0.05, "Conservative (current)"),
        
        # Moderate range
        (0.8, -0.1, "Moderate"),
        (1.0, -0.2, "Moderate+"),
        
        # Aggressive range
        (1.5, -0.3, "Aggressive"),
        (2.0, -0.5, "Very Aggressive"),
        
        # Single direction tests
        (1.0, 0.0, "Personal Only"),
        (0.0, -0.2, "Neutral Only"),
    ]
    
    results = []
    
    # Initialize evaluator once
    print("🚀 Initializing ChameleonEvaluator...")
    evaluator = ChameleonEvaluator('config.yaml', './chameleon_prime_personalization/data')
    
    if not (hasattr(evaluator, 'chameleon_editor') and evaluator.chameleon_editor):
        print("❌ ChameleonEditor not available")
        return
    
    editor = evaluator.chameleon_editor
    
    # Verify theta vectors
    if not (hasattr(editor, 'personal_direction') and editor.personal_direction is not None):
        print("❌ Personal direction vector not loaded")
        return
    
    print(f"✅ System ready - Personal norm: {editor.personal_direction.norm().item():.4f}")
    
    # Test each parameter combination
    for alpha, beta, description in parameter_tests:
        print(f"\n{'─'*60}")
        print(f"🧪 TESTING: {description}")
        print(f"   Parameters: α_personal={alpha}, β_neutral={beta}")
        print(f"{'─'*60}")
        
        test_result = {
            'description': description,
            'alpha_personal': alpha,
            'beta_neutral': beta,
            'responses': {},
            'edit_stats': {},
            'performance_impact': None
        }
        
        # Test with multiple prompts to assess consistency
        test_prompts = [
            "For the romantic movie 'Titanic', the tag is:",
            "For the action movie 'Die Hard', the tag is:", 
            "For the comedy movie 'Anchorman', the tag is:",
            "For the drama movie 'The Godfather', the tag is:"
        ]
        
        responses = []
        edit_ratios = []
        hook_calls_total = 0
        
        for i, prompt in enumerate(test_prompts):
            try:
                # Generate with specific parameters
                start_time = time.time()
                
                response = editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=alpha,
                    alpha_neutral=beta,
                    target_layers=['model.layers.20.mlp'],
                    gen_kwargs={'max_new_tokens': 5, 'do_sample': False}
                )
                
                gen_time = time.time() - start_time
                
                # Collect statistics
                hook_calls = getattr(editor, '_hook_calls_in_this_generate', 0)
                current_edit_ratios = getattr(editor, '_edit_ratios', [])
                avg_edit_ratio = sum(current_edit_ratios) / len(current_edit_ratios) if current_edit_ratios else 0.0
                
                responses.append(response.strip().split()[0] if response.strip() else "")
                edit_ratios.append(avg_edit_ratio)
                hook_calls_total += hook_calls
                
                print(f"   Prompt {i+1}: '{response[:20]}...' (edit: {avg_edit_ratio:.4f}, hooks: {hook_calls})")
                
            except Exception as e:
                print(f"   ❌ Prompt {i+1} failed: {e}")
                responses.append("ERROR")
                edit_ratios.append(0.0)
        
        # Analyze results
        valid_responses = [r for r in responses if r != "ERROR"]
        unique_responses = len(set(valid_responses))
        avg_edit_ratio = sum(edit_ratios) / len(edit_ratios) if edit_ratios else 0.0
        
        test_result['responses'] = {
            'generated': responses,
            'unique_count': unique_responses,
            'total_count': len(valid_responses)
        }
        
        test_result['edit_stats'] = {
            'avg_edit_ratio': avg_edit_ratio,
            'hook_calls_total': hook_calls_total,
            'edit_ratios': edit_ratios
        }
        
        # Assess parameter effectiveness
        if avg_edit_ratio > 0.02:  # > 2%
            if avg_edit_ratio > 0.10:  # > 10%
                effectiveness = "⚠️ OVER-EDITING (>10%)"
            else:
                effectiveness = "✅ EFFECTIVE RANGE (2-10%)"
        elif avg_edit_ratio > 0.005:  # > 0.5%
            effectiveness = "⚠️ WEAK (0.5-2%)"
        else:
            effectiveness = "❌ INEFFECTIVE (<0.5%)"
        
        print(f"\n   📊 Summary:")
        print(f"      Edit ratio: {avg_edit_ratio:.4f} ({avg_edit_ratio*100:.2f}%)")
        print(f"      Hook calls: {hook_calls_total}")
        print(f"      Response diversity: {unique_responses}/{len(valid_responses)}")
        print(f"      Effectiveness: {effectiveness}")
        
        test_result['effectiveness'] = effectiveness
        results.append(test_result)
    
    # Analysis and recommendations
    print(f"\n{'='*60}")
    print(f"📊 PARAMETER TUNING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Find optimal parameters
    effective_params = [r for r in results if "EFFECTIVE RANGE" in r.get('effectiveness', '')]
    weak_params = [r for r in results if "WEAK" in r.get('effectiveness', '')]
    over_editing = [r for r in results if "OVER-EDITING" in r.get('effectiveness', '')]
    
    print(f"\n🎯 PARAMETER EFFECTIVENESS CLASSIFICATION:")
    print(f"   ✅ Effective range (2-10% edit): {len(effective_params)} configurations")
    print(f"   ⚠️  Weak range (0.5-2% edit): {len(weak_params)} configurations")  
    print(f"   ⚠️  Over-editing (>10% edit): {len(over_editing)} configurations")
    print(f"   ❌ Ineffective (<0.5% edit): {len(results) - len(effective_params) - len(weak_params) - len(over_editing)} configurations")
    
    if effective_params:
        best_config = max(effective_params, key=lambda x: x['edit_stats']['avg_edit_ratio'])
        
        print(f"\n🏆 RECOMMENDED PARAMETERS:")
        print(f"   Configuration: {best_config['description']}")
        print(f"   α_personal: {best_config['alpha_personal']}")
        print(f"   β_neutral: {best_config['beta_neutral']}")
        print(f"   Edit ratio: {best_config['edit_stats']['avg_edit_ratio']:.4f} ({best_config['edit_stats']['avg_edit_ratio']*100:.2f}%)")
        
        # Test recommended parameters in actual evaluation
        print(f"\n🧪 VALIDATION TEST WITH RECOMMENDED PARAMETERS:")
        try:
            validation_start = time.time()
            
            # Update config for validation
            evaluator.config['chameleon']['alpha_personal'] = best_config['alpha_personal'] 
            evaluator.config['chameleon']['alpha_general'] = best_config['beta_neutral']
            
            # Run small evaluation
            eval_results = evaluator.run_evaluation(mode='demo')
            
            validation_time = time.time() - validation_start
            
            if eval_results and isinstance(eval_results, dict):
                baseline = eval_results.get('baseline')
                enhanced = eval_results.get('chameleon')
                
                if baseline and enhanced:
                    improvement = ((enhanced.accuracy - baseline.accuracy) / baseline.accuracy * 100) if baseline.accuracy > 0 else 0.0
                    
                    print(f"   📊 Validation Results:")
                    print(f"      Baseline: {baseline.accuracy:.4f}")
                    print(f"      Enhanced: {enhanced.accuracy:.4f}")
                    print(f"      Improvement: {improvement:+.2f}%")
                    print(f"      Time: {validation_time:.1f}s")
                    
                    if abs(improvement) > 0.5:  # > 0.5% difference
                        print(f"   ✅ SIGNIFICANT IMPROVEMENT DETECTED!")
                        print(f"   🎯 Ready for full Phase 3-A evaluation with these parameters")
                    else:
                        print(f"   ⚠️  Improvement still minimal - consider higher parameters")
                        
        except Exception as e:
            print(f"   ❌ Validation test failed: {e}")
    
    else:
        print(f"\n⚠️  NO EFFECTIVE PARAMETERS FOUND")
        print(f"   All tested configurations either too weak or over-editing")
        print(f"   Consider:")
        print(f"   • Testing intermediate values")
        print(f"   • Using different target layers")
        print(f"   • Switching to sampling mode (do_sample=True)")
    
    # Save results
    output_path = Path('results/fixed_phase3a_test/parameter_tuning_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Parameter Tuning Analysis...")
    
    results = test_parameter_ranges()
    
    print(f"\n✅ Parameter Tuning Analysis Complete!")
    print(f"📊 Use the recommended parameters for Phase 3-A evaluation")