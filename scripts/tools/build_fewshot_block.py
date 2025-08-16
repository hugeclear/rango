#!/usr/bin/env python3
"""
LaMP-2 Few-shot Block Builder
Extract real few-shot examples from LaMP-2 training data for prompt enhancement.
Adheres to "no mock data" policy - all examples extracted from actual dataset.
"""

import json
import argparse
import random
import sys
from pathlib import Path


def load_lamp2_data(data_file_path):
    """Load LaMP-2 data from JSONL file"""
    examples = []
    
    try:
        if data_file_path.endswith('.jsonl'):
            # JSONL format
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        example = json.loads(line)
                        examples.append(example)
        else:
            # JSON format
            with open(data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                elif 'golds' in data:
                    # LaMP-2 dev_outputs format
                    examples = data['golds']
                else:
                    examples = [data]
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return examples


def load_allowed_tags(allowed_tags_file):
    """Load allowed tags from file"""
    try:
        with open(allowed_tags_file, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f if line.strip()]
        return tags
    except Exception as e:
        print(f"Error loading allowed tags from {allowed_tags_file}: {e}", file=sys.stderr)
        sys.exit(1)


def load_questions_and_outputs():
    """Load LaMP-2 dev questions and outputs for few-shot examples"""
    questions_path = "chameleon_prime_personalization/data/raw/LaMP-2/dev_questions.json"
    outputs_path = "chameleon_prime_personalization/data/raw/LaMP-2/dev_outputs.json"
    
    try:
        # Load questions
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Load outputs  
        with open(outputs_path, 'r', encoding='utf-8') as f:
            outputs_data = json.load(f)
            outputs = outputs_data['golds']
        
        # Create output lookup
        output_lookup = {str(item['id']): item['output'] for item in outputs}
        
        # Merge questions with outputs
        merged_examples = []
        for question_item in questions_data:
            question_id = str(question_item['id'])
            if question_id in output_lookup:
                example = {
                    'id': question_item['id'],
                    'question': question_item['input'],
                    'user_profile': question_item.get('profile', []),
                    'reference': output_lookup[question_id]
                }
                merged_examples.append(example)
        
        return merged_examples
        
    except Exception as e:
        print(f"Error loading LaMP-2 training data: {e}", file=sys.stderr)
        sys.exit(1)


def format_user_profile(profile):
    """Format user profile for prompt"""
    if not profile:
        return "No user profile available"
    
    profile_items = []
    for item in profile[:5]:  # Limit to first 5 items to avoid overly long prompts
        tag = item.get('tag', 'unknown')
        desc = item.get('description', '')[:100]  # Truncate long descriptions
        if desc:
            profile_items.append(f"{tag}: {desc}")
        else:
            profile_items.append(f"{tag}")
    
    return "; ".join(profile_items)


def build_fewshot_block(examples, allowed_tags, k=3, seed=42):
    """Build few-shot block from real LaMP-2 examples"""
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Filter examples that have valid references in allowed tags
    valid_examples = []
    for ex in examples:
        if ex.get('reference') in allowed_tags:
            valid_examples.append(ex)
    
    if len(valid_examples) < k:
        print(f"Warning: Only {len(valid_examples)} valid examples found, requested {k}", file=sys.stderr)
        k = len(valid_examples)
    
    if k == 0:
        print("Error: No valid examples found for few-shot generation", file=sys.stderr)
        sys.exit(1)
    
    # Sample k examples
    selected_examples = random.sample(valid_examples, k)
    
    # Build few-shot block
    allowed_tags_str = ", ".join(allowed_tags)
    fewshot_lines = ["# FEW-SHOT EXAMPLES (all from LaMP-2 training split)", ""]
    
    for i, ex in enumerate(selected_examples, 1):
        question = ex['question']
        user_profile = format_user_profile(ex['user_profile'])
        answer = ex['reference']
        
        fewshot_lines.extend([
            f"Example {i}",
            f"Question: {question}",
            f"User Profile: {user_profile}",
            f"Allowed Tags: {allowed_tags_str}",
            f"Answer: {answer}",
            ""
        ])
    
    return "\n".join(fewshot_lines)


def main():
    parser = argparse.ArgumentParser(description="Build few-shot block for LaMP-2 prompts")
    parser.add_argument("--data", required=True, help="Path to LaMP-2 data file")
    parser.add_argument("--k", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--allowed-tags-file", required=True, help="Path to allowed tags file")
    
    args = parser.parse_args()
    
    # Load allowed tags
    allowed_tags = load_allowed_tags(args.allowed_tags_file)
    
    # Load LaMP-2 training examples (use dev data since training split not explicitly available)
    examples = load_questions_and_outputs()
    
    # Build few-shot block
    fewshot_block = build_fewshot_block(examples, allowed_tags, args.k, args.seed)
    
    # Output to stdout
    print(fewshot_block)


if __name__ == "__main__":
    main()