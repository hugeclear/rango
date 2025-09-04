#!/usr/bin/env python3
"""
Preflight Prior Generator: Ensures zero missing user_id/priors before runtime.

Usage:
    python tools/preflight_priors.py \
        --dataset data/evaluation/lamp2_expanded_eval.jsonl \
        --labels data/id2tag.txt \
        --out data/user_priors.jsonl
        
Exits with non-zero if any user_id missing or profile empty.
"""

import json
import argparse
import os
import sys
from pathlib import Path

def iter_jsonl(path):
    """Iterate over JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_prior_prompt(profile, id2tag_lines):
    """
    Build deterministic prior prompt from user profile.
    Raises ValueError if profile is empty or unusable.
    
    Args:
        profile: User's preference history
        id2tag_lines: Pre-formatted ID->tag mapping lines
        
    Returns:
        Complete prior prompt string for ID-mode classification
    """
    lines = []
    for p in profile or []:
        tag = (p.get("tag") or "").strip()
        desc = (p.get("description") or "").strip()
        if tag and desc:
            lines.append(f"- {tag}: {desc}")
    
    if not lines:
        raise ValueError("empty profile - cannot generate prior")
    
    # Build content with user preferences only (no specific movie)
    content = "User's movie preferences:\n" + "\n".join(lines) + "\n\nMovie:\n\nTag:"
    
    # Complete prior prompt matching the ID-mode format
    prompt = (
        "You are a tag classifier.\n"
        "Choose exactly ONE tag ID from the list and output the ID ONLY (integer, no text).\n"
        f"Tags:\n{id2tag_lines}\n\n"
        f"Input:\n{content}\n\n"
        "Answer with the ID only on a single line."
    )
    
    return prompt

def main():
    parser = argparse.ArgumentParser(
        description="Generate user priors with strict validation (fail-fast on missing data)"
    )
    parser.add_argument("--dataset", required=True, 
                       help="LaMP-2 JSONL dataset (e.g., lamp2_expanded_eval.jsonl)")
    parser.add_argument("--labels", required=True,
                       help="ID->label mapping file with 'i: label' format per line")
    parser.add_argument("--out", required=True,
                       help="Output file: user_priors.jsonl")
    parser.add_argument("--fail_on_missing", action="store_true", default=True,
                       help="Exit with error if any user_id/profile missing (default: True)")
    
    args = parser.parse_args()
    
    # Read deterministic label ordering
    try:
        id2tag_lines = Path(args.labels).read_text().strip()
        if not id2tag_lines:
            print(f"[strict] ERROR: Empty labels file: {args.labels}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[strict] ERROR: Cannot read labels file {args.labels}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[strict] Loaded label mapping from: {args.labels}")
    print(f"[strict] Processing dataset: {args.dataset}")
    
    seen_users = set()
    user_priors = []
    total_samples = 0
    failed_users = []
    
    # Process all samples to extract unique users
    try:
        for sample in iter_jsonl(args.dataset):
            total_samples += 1
            
            # Extract user_id with multiple fallback keys
            user_id = (sample.get("user_id") or 
                      sample.get("uid") or 
                      sample.get("user") or "").strip()
            
            if not user_id:
                error_msg = f"sample id={sample.get('id', 'unknown')} has no user_id"
                print(f"[strict] ERROR: {error_msg}", file=sys.stderr)
                if args.fail_on_missing:
                    sys.exit(2)
                continue
            
            # Skip if already processed this user
            if user_id in seen_users:
                continue
            
            # Extract and validate profile
            profile = sample.get("profile", [])
            if not profile or not isinstance(profile, list):
                error_msg = f"user_id={user_id} has empty or invalid profile"
                print(f"[strict] ERROR: {error_msg}", file=sys.stderr)
                failed_users.append(user_id)
                if args.fail_on_missing:
                    sys.exit(3)
                continue
            
            # Generate prior prompt
            try:
                prior_prompt = build_prior_prompt(profile, id2tag_lines)
            except Exception as e:
                error_msg = f"user_id={user_id} prior generation failed: {e}"
                print(f"[strict] ERROR: {error_msg}", file=sys.stderr)
                failed_users.append(user_id)
                if args.fail_on_missing:
                    sys.exit(3)
                continue
            
            # Successfully generated prior
            user_priors.append({
                "user_id": user_id,
                "prior_prompt": prior_prompt,
                "profile_items": len(profile)
            })
            seen_users.add(user_id)
            
    except Exception as e:
        print(f"[strict] ERROR: Failed to process dataset {args.dataset}: {e}", file=sys.stderr)
        sys.exit(4)
    
    # Validation summary
    print(f"[strict] Processed {total_samples} total samples")
    print(f"[strict] Found {len(seen_users)} unique users")
    
    if failed_users:
        print(f"[strict] WARNING: {len(failed_users)} users failed prior generation")
        if args.fail_on_missing:
            print(f"[strict] ERROR: Cannot proceed with missing priors in strict mode", file=sys.stderr)
            sys.exit(5)
    
    if not user_priors:
        print(f"[strict] ERROR: No valid user priors generated", file=sys.stderr)
        sys.exit(6)
    
    # Write output
    try:
        output_content = "\n".join(
            json.dumps(prior, ensure_ascii=False) for prior in user_priors
        ) + "\n"
        
        Path(args.out).write_text(output_content, encoding='utf-8')
        print(f"[strict] SUCCESS: Wrote {len(user_priors)} user priors to {args.out}")
        
    except Exception as e:
        print(f"[strict] ERROR: Failed to write output {args.out}: {e}", file=sys.stderr)
        sys.exit(7)
    
    # Final validation
    print(f"[strict] ✅ All user_ids have valid priors - ready for strict runtime")
    print(f"[strict] Use: --strict true --prior_mode user --user_prior_path {args.out}")

if __name__ == "__main__":
    main()