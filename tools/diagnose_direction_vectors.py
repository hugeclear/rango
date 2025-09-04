#!/usr/bin/env python3
"""
Direction Vector Implementation Diagnosis Tool

This script diagnoses the direction vector implementation by:
1. Extracting the final prompt from output_detail_analysis.md
2. Testing _compute_direction_vectors_strict method
3. Validating numerical health of results
4. Reporting status and detailed metrics
"""

import torch
import json
import re
import time
import traceback
import math
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

def parse_prompt_from_md(md_path: str) -> str:
    """Extract the first code block from output_detail_analysis.md"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for code blocks - first try ```text, then fall back to ```
        patterns = [
            r'```text\n(.*?)\n```',
            r'```\n(.*?)\n```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                prompt = matches[0].strip()
                if len(prompt) > 0:
                    return prompt
        
        raise ValueError("No valid code block found in markdown file")
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse prompt from {md_path}: {e}")

def diagnose():
    """Main diagnosis function"""
    t0 = time.time()
    status = "ok"
    err = None
    result = {}
    
    try:
        # Step 1: Extract prompt from markdown
        prompt = parse_prompt_from_md("output_detail_analysis.md")
        assert isinstance(prompt, str) and len(prompt) > 0
        result["prompt_length"] = len(prompt)
        result["prompt_extracted"] = True
        
        # Step 2: Initialize evaluator and editor
        from chameleon_evaluator import ChameleonEvaluator
        print("Initializing ChameleonEvaluator...")
        evaluator = ChameleonEvaluator("config.yaml", "./chameleon_prime_personalization/data")
        editor = evaluator.chameleon_editor
        
        # Step 3: Check if _compute_direction_vectors_strict exists
        has_strict = hasattr(editor, "_compute_direction_vectors_strict")
        result["has_strict"] = bool(has_strict)
        
        if not has_strict:
            status = "missing"  # Implementation physically absent
            result["missing_methods"] = ["_compute_direction_vectors_strict"]
        else:
            print("Testing _compute_direction_vectors_strict method...")
            
            # Step 4: Prepare sample and clear GPU memory
            sample = {"prompt": prompt}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 5: Execute strict direction vector computation
            with torch.no_grad():
                vecs = editor._compute_direction_vectors_strict(
                    sample, 
                    target_layers=None, 
                    norm_scale=1.0
                )
            
            # Step 6: Handle different return types
            if isinstance(vecs, tuple) and len(vecs) >= 2:
                v_p, v_g = vecs[0], vecs[1]
                result["return_type"] = "tuple"
            elif isinstance(vecs, dict):
                v_p, v_g = vecs.get("v_personal"), vecs.get("v_general")
                if v_p is None or v_g is None:
                    raise RuntimeError("Dict return type missing v_personal or v_general keys")
                result["return_type"] = "dict"
            else:
                raise RuntimeError(f"Unexpected return type from _compute_direction_vectors_strict: {type(vecs)}")
            
            # Step 7: Tensor validation and shape checks
            if not (hasattr(v_p, "shape") and hasattr(v_g, "shape")):
                raise RuntimeError("Returned vectors are not tensors with shape attribute")
            
            h_p = v_p.shape[-1]
            h_g = v_g.shape[-1]
            if h_p != h_g or h_p <= 0:
                raise RuntimeError(f"Invalid vector dimensions: v_p={v_p.shape}, v_g={v_g.shape}")
            
            result["vector_dimension"] = int(h_p)
            result["v_personal_shape"] = list(v_p.shape)
            result["v_general_shape"] = list(v_g.shape)
            
            # Step 8: Numerical health checks
            l2p = float(v_p.norm(p=2).item())
            l2g = float(v_g.norm(p=2).item())
            
            if not (math.isfinite(l2p) and math.isfinite(l2g) and l2p > 0 and l2g > 0):
                raise RuntimeError(f"Invalid L2 norms: l2_personal={l2p}, l2_general={l2g}")
            
            # Step 9: Cosine similarity validation
            cos_th = float(torch.dot(
                v_p / (l2p + 1e-8), 
                v_g / (l2g + 1e-8)
            ).item())
            
            if not (-1.0001 <= cos_th <= 1.0001):
                raise RuntimeError(f"Invalid cosine similarity: {cos_th}")
            
            # Step 10: Check for NaN/Inf in vectors
            if torch.isnan(v_p).any() or torch.isinf(v_p).any():
                raise RuntimeError("v_personal contains NaN or Inf values")
            if torch.isnan(v_g).any() or torch.isinf(v_g).any():
                raise RuntimeError("v_general contains NaN or Inf values")
            
            # Step 11: Store validated results
            result.update({
                "l2_personal": l2p,
                "l2_general": l2g,
                "cos_theta": cos_th,
                "numerical_health": "passed",
                "contains_nan_inf": False
            })
            
            print(f"✅ Direction vectors computed successfully:")
            print(f"   L2 norms: personal={l2p:.6f}, general={l2g:.6f}")
            print(f"   Cosine similarity: {cos_th:.6f}")
            print(f"   Vector dimension: {h_p}")
            
    except Exception as e:
        status = "fail"
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"❌ Diagnosis failed: {e}")
        
    t1 = time.time()
    
    # Step 12: Compile final payload
    payload = {
        "status": status,
        "error": err,
        "result": result,
        "elapsed_sec": round(t1 - t0, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Step 13: Save JSON results
    Path("results/trace").mkdir(parents=True, exist_ok=True)
    with open("results/trace/dirvec_diagnosis.json", "w", encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # Step 14: Generate Markdown report
    with open("results/trace/dirvec_diagnosis.md", "w", encoding='utf-8') as f:
        f.write("# Direction Vector Diagnosis\n\n")
        f.write(f"**Status:** **{status}**\n\n")
        
        if result:
            if "has_strict" in result:
                f.write(f"- has_strict: {result['has_strict']}\n")
            if "prompt_extracted" in result:
                f.write(f"- prompt_extracted: {result['prompt_extracted']}\n")
            if "vector_dimension" in result:
                f.write(f"- vector_dimension: {result['vector_dimension']}\n")
            if "l2_personal" in result:
                f.write(f"- l2_personal: {result['l2_personal']:.6f}\n")
                f.write(f"- l2_general: {result['l2_general']:.6f}\n")
                f.write(f"- cos_theta: {result['cos_theta']:.6f}\n")
            if "numerical_health" in result:
                f.write(f"- numerical_health: {result['numerical_health']}\n")
        
        if status == "missing":
            f.write(f"\n⚠️ **Issue:** Implementation missing - {result.get('missing_methods', [])}\n")
        elif status == "fail":
            f.write(f"\n❌ **Issue:** Implementation broken or failed validation\n")
        elif status == "ok":
            f.write(f"\n✅ **Result:** Implementation working correctly\n")
        
        if err:
            f.write(f"\n<details><summary>Error Details</summary>\n\n```\n{err}\n```\n</details>\n")
        
        f.write(f"\n- elapsed: {payload['elapsed_sec']} sec\n")
        f.write(f"- timestamp: {payload['timestamp']}\n")
    
    print(f"\n📄 Results saved to:")
    print(f"   📄 results/trace/dirvec_diagnosis.json")
    print(f"   📄 results/trace/dirvec_diagnosis.md")
    
    return payload

if __name__ == "__main__":
    diagnose()