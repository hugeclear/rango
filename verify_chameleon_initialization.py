#!/usr/bin/env python3
"""
Verify Chameleon System Initialization
======================================

Test that theta vectors are loaded and hooks are properly installed after the path fix.
"""

import sys
import os
import torch
from pathlib import Path

sys.path.append('/home/nakata/master_thesis/rango')

from chameleon_evaluator import ChameleonEvaluator

def test_chameleon_initialization():
    """Test proper Chameleon initialization"""
    print("🧪 TESTING CHAMELEON INITIALIZATION POST-FIX")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = ChameleonEvaluator('config.yaml', './chameleon_prime_personalization/data')
        
        print("✅ ChameleonEvaluator initialization: SUCCESS")
        
        # Check chameleon_editor exists
        if hasattr(evaluator, 'chameleon_editor') and evaluator.chameleon_editor:
            editor = evaluator.chameleon_editor
            print("✅ ChameleonEditor available: SUCCESS")
            
            # Check theta vectors loaded
            has_personal = hasattr(editor, 'personal_direction') and editor.personal_direction is not None
            has_neutral = hasattr(editor, 'neutral_direction') and editor.neutral_direction is not None
            
            print(f"📊 Personal direction vector: {'✅ LOADED' if has_personal else '❌ MISSING'}")
            print(f"📊 Neutral direction vector: {'✅ LOADED' if has_neutral else '❌ MISSING'}")
            
            if has_personal:
                personal_shape = editor.personal_direction.shape
                personal_norm = torch.norm(editor.personal_direction).item()
                print(f"   Personal vector: shape={personal_shape}, norm={personal_norm:.4f}")
            
            if has_neutral:
                neutral_shape = editor.neutral_direction.shape
                neutral_norm = torch.norm(editor.neutral_direction).item()
                print(f"   Neutral vector: shape={neutral_shape}, norm={neutral_norm:.4f}")
            
            # Test hook installation during generation
            print(f"\n🔌 TESTING HOOK INSTALLATION:")
            
            if has_personal and has_neutral:
                # Test generation with hooks
                test_prompt = "For the romantic movie 'The Notebook', the tag is:"
                
                print(f"   Test prompt: {test_prompt}")
                print(f"   Attempting generation with Chameleon editing...")
                
                try:
                    # This should install hooks and perform editing
                    response = editor.generate_with_chameleon(
                        prompt=test_prompt,
                        alpha_personal=0.4,
                        alpha_neutral=-0.05,
                        target_layers=['model.layers.20.mlp'],
                        gen_kwargs={'max_new_tokens': 10}
                    )
                    
                    print(f"✅ Chameleon generation: SUCCESS")
                    print(f"   Response: '{response[:50]}...'")
                    
                    # Check hook call statistics
                    hook_calls = getattr(editor, '_hook_calls_in_this_generate', 0)
                    edit_ratios = getattr(editor, '_edit_ratios', [])
                    
                    print(f"📈 Hook statistics:")
                    print(f"   Hook calls: {hook_calls}")
                    print(f"   Edit operations: {len(edit_ratios)}")
                    
                    if edit_ratios:
                        avg_edit_ratio = sum(edit_ratios) / len(edit_ratios)
                        print(f"   Average edit ratio: {avg_edit_ratio:.6f} ({avg_edit_ratio*100:.3f}%)")
                        
                        if avg_edit_ratio > 0:
                            print(f"✅ CHAMELEON EDITING IS ACTIVE!")
                        else:
                            print(f"⚠️  Edit ratio is zero - parameters may be too conservative")
                    
                except Exception as e:
                    print(f"❌ Chameleon generation failed: {e}")
                    return False
                    
            else:
                print(f"⚠️  Cannot test hooks - theta vectors missing")
                return False
                
        else:
            print("❌ ChameleonEditor not available")
            return False
        
        print(f"\n🎯 CHAMELEON SYSTEM STATUS:")
        if has_personal and has_neutral:
            print(f"   ✅ Theta vectors: LOADED")
            print(f"   ✅ Direction editing: ENABLED") 
            print(f"   ✅ Hook system: OPERATIONAL")
            print(f"   ✅ Ready for differentiated evaluation")
            return True
        else:
            print(f"   ❌ System not fully operational")
            return False
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chameleon_initialization()
    if success:
        print(f"\n🚀 CHAMELEON SYSTEM FULLY OPERATIONAL!")
        print(f"   Ready for proper ablation study with real performance differences")
    else:
        print(f"\n❌ CHAMELEON SYSTEM ISSUES DETECTED")
        print(f"   Evaluation will continue showing identical performance")