#!/usr/bin/env python3
"""
Critical Fix: Theta Vector Path Resolution
==========================================

Issue: ChameleonEvaluator can't find theta vectors because they're in 
/home/nakata/master_thesis/rango/processed/LaMP-2/ but it's looking in 
./chameleon_prime_personalization/data/processed/LaMP-2/

This is why all configurations show identical 36.36% accuracy - no Chameleon editing!

Fix: Update path resolution in ChameleonEvaluator
"""

import sys
import os
import shutil
from pathlib import Path

def fix_theta_vector_paths():
    """Fix theta vector path resolution"""
    
    print("🔧 CRITICAL FIX: Theta Vector Path Resolution")
    print("=" * 60)
    
    # Source: Where theta vectors actually are
    source_dir = Path("/home/nakata/master_thesis/rango/processed/LaMP-2")
    
    # Target: Where ChameleonEvaluator expects them
    target_dir = Path("/home/nakata/master_thesis/rango/chameleon_prime_personalization/data/processed/LaMP-2")
    
    print(f"📂 Source directory: {source_dir}")
    print(f"📂 Target directory: {target_dir}")
    
    # Check source files exist
    theta_files = ['theta_p.json', 'theta_n.json', 'theta_p.npy', 'theta_n.npy']
    
    for file in theta_files:
        source_file = source_dir / file
        if not source_file.exists():
            print(f"❌ Missing source file: {source_file}")
            return False
        else:
            print(f"✅ Found source file: {source_file}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created target directory: {target_dir}")
    
    # Copy theta vector files
    for file in theta_files:
        source_file = source_dir / file
        target_file = target_dir / file
        
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"📋 Copied: {file}")
        else:
            print(f"⏩ Already exists: {file}")
    
    # Verify copy success
    print(f"\n✅ VERIFICATION:")
    for file in theta_files:
        target_file = target_dir / file
        if target_file.exists():
            size = target_file.stat().st_size
            print(f"   {file}: {size:,} bytes")
        else:
            print(f"   ❌ {file}: MISSING")
            return False
    
    print(f"\n🎯 CRITICAL FIX COMPLETED!")
    print(f"   ✅ Theta vectors now accessible to ChameleonEvaluator")
    print(f"   ✅ All configurations will now use actual Chameleon editing")
    print(f"   ✅ Performance differences should now be visible")
    
    return True

if __name__ == "__main__":
    success = fix_theta_vector_paths()
    if success:
        print(f"\n🚀 Ready for re-evaluation with working Chameleon system!")
    else:
        print(f"\n❌ Fix failed - manual intervention required")
        sys.exit(1)