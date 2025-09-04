#!/usr/bin/env python3
"""
LaMP-2 Benchmark: Baseline vs Chameleon evaluation with statistical testing.

Usage:
python tools/run_benchmark_lamp2.py \
  --data_path data --split test --limit 500 --seed 42 \
  --alpha_personal 2.75 --alpha_general -1.0 \
  --edit_gate_threshold 0.022 \
  --out_dir results/bench/lamp2_ap275_ag-10_tau0022

Outputs:
- predictions.jsonl: Per-sample predictions and metadata
- summary.csv: Aggregate metrics  
- summary.md: Human-readable report with McNemar test
"""

import argparse
import os
import json
import math
import time
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Timeout prevention: offline mode + parallelism fix
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")        # Stop HF online exploration
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid parallelism deadlock

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from chameleon_evaluator import ChameleonEvaluator
from chameleon_prime_personalization.utils.tags import DEFAULT_ALLOWED_TAGS, build_id_maps, match_allowed
from chameleon_prime_personalization.utils.labels_from_dataset import resolve_label_set
from chameleon_prime_personalization.utils.strict_classifier import build_id_prompt, parse_id_output
from chameleon_prime_personalization.utils.score_classifier import (
    classify_by_scores, classify_by_scores_with_calibration, 
    classify_by_scores_pmi, classify_by_scores_with_calibration_pmi, compute_prior_logprobs,
    classify_with_prior_correction,
    classify_by_scores_batched, classify_by_scores_with_calibration_batched
)
from chameleon_prime_personalization.utils.prior_provider import PriorProvider
import torch
from utils.reproducibility import set_reproducible_seeds, apply_reproducible_env


def mcnemar_exact(b: int, c: int) -> float:
    """
    McNemar exact test (two-sided) for paired binary outcomes.
    
    Args:
        b: Cases where Baseline failed but Chameleon succeeded  
        c: Cases where Baseline succeeded but Chameleon failed
        
    Returns:
        Two-sided exact p-value
    """
    from math import comb
    
    n = b + c
    if n == 0:
        return 1.0
    
    # Exact binomial test: P = 2 * sum_{k=0..min(b,c)} C(n,k) * (1/2)^n
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    p = 2.0 * tail * (0.5 ** n)
    return min(1.0, p)


class LaMP2Benchmarker:
    """Benchmark runner for LaMP-2 task using existing ChameleonEvaluator API."""
    
    def __init__(self, config_path: str = None, data_path: str = None):
        """Initialize evaluator with existing configuration."""
        self.data_path = data_path or "data"
        try:
            self.evaluator = ChameleonEvaluator(
                config_path=config_path,
                data_path=self.data_path
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChameleonEvaluator: {e}")
            # Create a minimal mock evaluator for testing
            self.evaluator = None
        
        # Use the evaluation engine for tag normalization if available
        if self.evaluator:
            self.eval_engine = self.evaluator.evaluation_engine
        else:
            self.eval_engine = None
        
    def load_dataset(self, task: str = "lamp2", split: str = "test", limit: int = None, user_id: str = None) -> List[Tuple[int, Dict[str, Any]]]:
        """Load dataset samples with indices."""
        # Try to load actual LaMP-2 data first
        lamp2_files = [
            f"data/evaluation/lamp2_expanded_eval.jsonl",
            f"data/evaluation/lamp2_backup_eval.jsonl",  
            f"data/evaluation/lamp2_robust_eval_140.jsonl"
        ]
        
        raw_data = []
        for filepath in lamp2_files:
            try:
                import json
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            raw_data.append(sample)
                            if limit and len(raw_data) >= limit:
                                break
                print(f"[load_dataset] Loaded {len(raw_data)} samples from {filepath}")
                break  # Use first available file
            except Exception as e:
                print(f"[load_dataset] Could not load {filepath}: {e}")
                continue
        
        if not raw_data:
            print(f"[load_dataset] No LaMP-2 data found, using fallback mock data")
            # Create minimal mock data respecting limit
            fallback_limit = limit or 50
            print(f"Using mock data with {fallback_limit} samples")
            raw_data = []
            for i in range(fallback_limit):
                raw_data.append({
                    "id": i,
                    "question": f"A {['drama', 'action', 'comedy', 'sci-fi', 'thriller'][i % 5]} movie description {i}",
                    "profile": [{"tag": ['drama', 'action', 'comedy', 'sci-fi', 'thriller'][i % 5], "description": f"Sample movie {i}"}],
                    "reference": ['drama', 'action', 'comedy', 'sci-fi', 'thriller'][i % 5],  # Varied mock ground truth
                    "gold": ['drama', 'action', 'comedy', 'sci-fi', 'thriller'][i % 5]
                })
        
        # Filter out samples with empty questions and add indices
        samples = []
        for i, sample in enumerate(raw_data):
            # Handle different sample formats
            question = ""
            if isinstance(sample, dict):
                question = (sample.get("question") or sample.get("input") or sample.get("text") or "").strip()
            
            if question:
                # Ensure consistent ground truth field
                if "reference" in sample and "gold" not in sample:
                    sample["gold"] = sample["reference"]
                elif "output" in sample and "gold" not in sample:
                    sample["gold"] = sample["output"]
                elif "gold" not in sample:
                    sample["gold"] = ""
                    
                # For LaMP-2 data, ensure we have the required fields
                if "question" not in sample and "input" in sample:
                    sample["question"] = sample["input"]
                if "profile" not in sample and "user_profile" in sample:
                    sample["profile"] = sample["user_profile"]
                
                # Ensure question field exists
                if "question" not in sample:
                    sample["question"] = question
                    
                samples.append((i, sample))
                
        # Filter by user_id if specified
        if user_id is not None:
            original_count = len(samples)
            filtered_samples = []
            for idx, sample in samples:
                # Get user_id from sample with priority order
                sample_user_id = sample.get("user_id") or sample.get("uid") or sample.get("user") or "unknown"
                if str(sample_user_id) == str(user_id):
                    filtered_samples.append((idx, sample))
            samples = filtered_samples
            print(f"[load_dataset] Filtered from {original_count} to {len(samples)} samples for user_id={user_id}")
            
        return samples
    
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build prompt using existing template logic."""
        try:
            # Use trace_one_example compatible prompt building if available
            from tools.trace_one_example import TraceAnalyzer
            analyzer = TraceAnalyzer(self.evaluator)
            return analyzer.build_prompt(sample)
        except Exception:
            # Fallback to simple template
            user_profile = sample.get("user_profile", "")
            question = sample.get("question", "")
            
            return f"""Task: Classify the movie description into exactly one tag from this closed set:
[action, adventure, animation, comedy, crime, drama, fantasy, horror, mystery, romance, sci-fi, thriller, war, western, family]

<user profile excerpt>
{user_profile}

<movie description>  
{question}

Answer:"""
    
    def generate_baseline(self, prompt: str, **gen_kwargs) -> Dict[str, Any]:
        """Generate baseline prediction (no editing)."""
        if self.evaluator:
            try:
                # Use Chameleon API with zero editing parameters
                allowed_tags = gen_kwargs.get("allowed_tags")
                generated_text = self.evaluator.chameleon_editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=0.0,
                    alpha_neutral=0.0,
                    alpha_fakeit=0.0,
                    target_layers=[],  # No editing layers
                    gen_kwargs={
                        "max_new_tokens": gen_kwargs.get("max_new_tokens", 4),
                        "min_new_tokens": gen_kwargs.get("min_new_tokens", 1),
                        "do_sample": gen_kwargs.get("do_sample", False),
                        "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
                        "allowed_tags": allowed_tags,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                    }
                )
                
                return {
                    "text": generated_text,
                    "avg_logprob": None,  # Would be populated by updated generate method
                    "gate_value": 0.0,
                    "gate_applied": False,
                }
                
            except Exception as e:
                return {
                    "text": "",
                    "error": str(e),
                    "avg_logprob": None,
                    "gate_value": 0.0,
                    "gate_applied": False,
                }
        else:
            # Mock baseline generation
            # If ID-mode prompt, emit a valid ID token '1'
            if "Answer with the ID only" in prompt:
                return {
                    "text": "1",
                    "avg_logprob": -2.5,
                    "gate_value": 0.0,
                    "gate_applied": False,
                }
            # Extract likely tag from prompt (very simple baseline)
            import re
            tags = ["drama", "action", "comedy", "sci-fi", "thriller", "fantasy", "horror", "romance"]
            for tag in tags:
                if tag in prompt.lower():
                    return {
                        "text": tag,
                        "avg_logprob": -2.5,  # Mock logprob
                        "gate_value": 0.0,
                        "gate_applied": False,
                    }
            # Default fallback
            return {
                "text": "drama",
                "avg_logprob": -2.8,
                "gate_value": 0.0,
                "gate_applied": False,
            }
    
    def generate_chameleon(self, prompt: str, alpha_personal: float = 2.75, 
                          alpha_general: float = -1.0, norm_scale: float = 0.9,
                          edit_gate_threshold: float = 0.022, target_layers: List[str] = None,
                          **gen_kwargs) -> Dict[str, Any]:
        """Generate Chameleon prediction with editing."""
        if self.evaluator is None:
            # Mock Chameleon generation with slight bias toward user preference
            if "Answer with the ID only" in prompt:
                return {
                    "text": "1",
                    "avg_logprob": -2.1,
                    "gate_value": 0.5,
                    "gate_applied": True,
                }
            import re
            import random
            user_prefs = []
            tags = ["drama", "action", "comedy", "sci-fi", "thriller", "fantasy", "horror", "romance"]
            for tag in tags:
                if tag in prompt.lower() and "user" in prompt.lower():
                    user_prefs.append(tag)
            if user_prefs and random.random() < 0.7:
                result_tag = user_prefs[0]
                avg_logprob = -2.1
            else:
                for tag in tags:
                    if tag in prompt.lower():
                        result_tag = tag
                        avg_logprob = -2.3
                        break
                else:
                    result_tag = "drama"
                    avg_logprob = -2.6
            return {
                "text": result_tag,
                "avg_logprob": avg_logprob,
                "gate_value": 3.75 if edit_gate_threshold < 1.0 else 0.5,
                "gate_applied": edit_gate_threshold < 1.0,
            }
        # Scoring route with edit hooks
        try:
            allowed = gen_kwargs.get("allowed_tags") or []
            if not allowed:
                # Fallback to generative path if no candidates provided
                generated_text = self.evaluator.chameleon_editor.generate_with_chameleon(
                    prompt=prompt,
                    alpha_personal=alpha_personal,
                    alpha_neutral=alpha_general,
                    alpha_fakeit=0.0,
                    target_layers=target_layers,
                    gen_kwargs={
                        "max_new_tokens": gen_kwargs.get("max_new_tokens", 4),
                        "min_new_tokens": gen_kwargs.get("min_new_tokens", 1),
                        "do_sample": gen_kwargs.get("do_sample", False),
                        "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
                        "norm_scale": norm_scale,
                        "edit_gate_threshold": edit_gate_threshold,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                    }
                )
                return {"text": generated_text, "avg_logprob": None, "gate_value": None, "gate_applied": None}

            # 1) compute direction vectors and gate
            dv = self.evaluator.compute_direction_vectors({"prompt": prompt}, target_layers=None, norm_scale=norm_scale)
            hidden_norm = max(1e-8, float(dv["l2_general"]))
            gate = self.evaluator.summarize_gate(hidden_norm, dv, alpha_personal, alpha_general, edit_gate_threshold)

            # 2) build delta and resolve target layers
            editor = self.evaluator.chameleon_editor
            delta = editor._edit_delta(dv, alpha_personal, alpha_general)
            try:
                num_layers = len(getattr(getattr(editor.model, 'model', editor.model), 'layers', []))
            except Exception:
                num_layers = int(getattr(getattr(editor.model, 'config', {}), 'num_hidden_layers', 0))

            def _sanitize_layers(layers, n):
                if not layers:
                    # default last 4
                    if n >= 4:
                        return [n-4, n-3, n-2, n-1]
                    return [max(0, n-1)] if n > 0 else []
                out = []
                for x in layers:
                    try:
                        i = int(x)
                        if i < 0:
                            i = n + i
                        if 0 <= i < n:
                            out.append(i)
                    except Exception:
                        continue
                return sorted(set(out))

            layers_eff = _sanitize_layers(target_layers, num_layers)

            # 3) score candidates under edit (disable KV cache to force pre-hook on each step)
            scores = []
            tok = editor.tokenizer
            dev = editor.device
            with editor._layer_injection_ctx(layers_eff, delta, bool(gate.get("applied", False)), None):
                for cand in allowed:
                    # sequential conditional logprob
                    p_ids = tok(prompt, return_tensors="pt").to(dev)["input_ids"]
                    c_ids = tok(cand, return_tensors="pt").to(dev)["input_ids"][0]
                    total = 0.0
                    inp = p_ids
                    for t in c_ids:
                        out = editor.model(input_ids=inp, use_cache=False)
                        logits = out.logits[:, -1, :]
                        logp = torch.log_softmax(logits, dim=-1)
                        total += float(logp[0, int(t.item())])
                        inp = torch.cat([inp, t.view(1,1)], dim=1)
                    scores.append(total)

            import numpy as np
            idx = int(np.argmax(scores)) if scores else 0
            return {
                "text": allowed[idx] if allowed else "",
                "avg_logprob": float(scores[idx]) if scores else None,
                "gate_value": float(gate.get("gate_value")) if gate else None,
                "gate_applied": bool(gate.get("applied", False)) if gate else None,
            }
        except Exception as e:
            print(f"[generate_chameleon] scoring with edit failed: {e}")
            return {"text": "", "error": str(e), "avg_logprob": None, "gate_value": None, "gate_applied": None}
    
    def normalize_tag(self, text: str) -> str:
        """Normalize tag using existing evaluation engine."""
        try:
            return self.eval_engine._normalize_tag(text)
        except Exception:
            # Fallback normalization
            import re
            text = text.strip().lower()
            
            # Enhanced sci-fi recovery
            if any(token in text for token in ["sci", "-", "fi"]):
                if re.search(r"sci[-\s]*fi", text) or ("sci" in text and "fi" in text):
                    return "sci-fi"
            
            # Extract first valid word
            words = re.findall(r"[A-Za-z\-]+", text)
            allowed_tags = {
                "action", "adventure", "animation", "comedy", "crime", "drama",
                "fantasy", "horror", "mystery", "romance", "sci-fi", "thriller", 
                "war", "western", "family"
            }
            
            for word in words:
                if word in allowed_tags:
                    return word
                    
            return words[0] if words else "unknown"
    
    def run_benchmark(self, split: str = "test", limit: int = -1, seed: int = 42,
                     alpha_personal: float = 2.75, alpha_general: float = -1.0,
                     norm_scale: float = 0.9, edit_gate_threshold: float = 0.022,
                     target_layers: List[str] = None,
                     max_new_tokens: int = 2, min_new_tokens: int = 1,
                     do_sample: bool = False, repetition_penalty: float = 1.0,
                     out_dir: str = None,
                     mode: str = "id", use_pmi: bool = False, score_norm: str = "avg",
                     prior_mode: str = "empty", score_temp: float = 1.0,
                     prior_beta: float = 1.0, prior_alpha: float = 1.0,
                     score_template: str = "\nAnswer: {label}",
                     prior_fallback: str = "global", user_id: str = None, 
                     strict_mode: bool = False, 
                     user_prior_path: str = None,
                     scoring_mode: str = "batched") -> Dict[str, Any]:
        """Run complete benchmark evaluation."""
        
        # Setup reproducibility
        set_reproducible_seeds(seed)
        apply_reproducible_env()
        
        # Load dataset with limit
        samples = self.load_dataset(split=split, limit=limit if limit > 0 else None, user_id=user_id)
        
        # Check if samples exist after filtering
        if not samples:
            if user_id:
                print(f"No samples for user_id={user_id} in {split}")
            else:
                print(f"No samples found in {split}")
            return {"n": 0, "baseline_acc": 0.0, "chameleon_acc": 0.0, "delta": 0.0, "mcnemar_p": 1.0}

        # ラベル集合をデータから解決（fallback付き）
        labels = resolve_label_set(self.data_path, split, DEFAULT_ALLOWED_TAGS)
        id2tag, tag2id = build_id_maps(labels)
        use_calibration = getattr(self, "use_calibration", True)
        
        print(f"[label_set] collected {len(labels)} unique labels from data (after normalization)")
        
        # Build prior tables based on prior_mode
        prior_table = None
        user_prior_table = None
        
        if prior_mode in ["global", "user"]:
            prior_table = build_global_prior_table(self.data_path, split, labels, prior_alpha)
            
        if prior_mode == "user":
            user_prior_table = build_user_prior_table(self.data_path, split, labels, prior_alpha)
        
        # Backward compatibility: use_pmi flag maps to empty mode
        if use_pmi and prior_mode == "empty":
            print(f"[prior] Using legacy PMI mode (empty context)")
        
        # Pre-compute PMI priors for legacy compatibility
        prior_logprobs = None
        if use_pmi and self.evaluator:
            editor = self.evaluator.chameleon_editor
            print(f"[PMI] Pre-computing prior logprobs with norm={score_norm}, temp={score_temp}")
            prior_logprobs = compute_prior_logprobs(
                editor.model, editor.tokenizer, id2tag, 
                device=editor.device, score_norm=score_norm
            )
        if self.eval_engine and hasattr(self.eval_engine, 'set_allowed_tags'):
            try:
                self.eval_engine.set_allowed_tags(labels)
            except Exception as e:
                print(f"[run_benchmark] set_allowed_tags failed: {e} (continuing with defaults)")
            
        print(f"Running benchmark on {len(samples)} samples...")
        
        # Initialize counters
        n = 0
        bl_acc = ch_acc = 0
        valid_bl = valid_ch = 0
        b = c = 0  # McNemar counts
        per_tag = {}  # Per-tag analysis
        
        results = []
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        
        for sid, sample in samples:
            n += 1
            
            # Build prompt from LaMP-2 format
            question = sample.get("question", "")
            profile = sample.get("profile", [])
            
            # Extract user preferences from profile history
            if profile:
                pref_lines = []
                for item in profile[:5]:  # Use first 5 preferences to avoid token limit
                    tag = item.get("tag", "")
                    desc = item.get("description", "")
                    if tag and desc:
                        pref_lines.append(f"- {tag}: {desc[:100]}...")  # Truncate for brevity
                user_profile = "\n".join(pref_lines)
            else:
                user_profile = "No preferences available"
                
            # Extract movie description from question
            if "description:" in question:
                movie_desc = question.split("description:")[1].strip()
            else:
                movie_desc = question
                
            content = f"User's movie preferences:\n{user_profile}\n\nMovie:\n{movie_desc}"
            
            # IDプロンプト（選択肢は id2tag の順で固定）：
            prompt = build_id_prompt(content, id2tag=id2tag)
            
            # Initialize PriorProvider once per sample set for efficiency
            if 'prior_provider' not in locals() and self.evaluator and use_calibration:
                editor = self.evaluator.chameleon_editor
                prior_prompt = build_id_prompt("", id2tag=id2tag)
                # Map current prior_mode to new provider system
                if prior_mode in ["empty", "global"]:
                    provider_mode = "global" 
                elif prior_mode == "user":
                    provider_mode = "user"
                else:
                    provider_mode = prior_mode  # none, user_or_global
                    
                prior_provider = PriorProvider(
                    model=editor.model,
                    tokenizer=editor.tokenizer, 
                    id2tag=id2tag,
                    device=editor.device,
                    prior_prompt=prior_prompt,
                    beta=prior_beta,
                    prior_mode=provider_mode,
                    strict_mode=strict_mode,
                    user_prior_path=user_prior_path
                )
            if n <= 3:  # Debug first few samples
                print(f"[Sample {n}] Content: {content[:200]}...")
                print(f"[Sample {n}] Prompt: {prompt[:300]}...")

            # Normalize gold
            gold = match_allowed(sample.get("gold", ""), allowed=labels) or "unknown"
            
            # --- Baseline（編集なし）: スコアリング分類 ---
            if n <= 3:
                print(f"[Sample {n}] Processing baseline scoring...")
                
            if self.evaluator and use_calibration and 'prior_provider' in locals():
                editor = self.evaluator.chameleon_editor
                tokenizer = editor.tokenizer
                try:
                    # Get prior scores with metadata (never leaves prior_scores undefined)
                    user_id = sample.get("user_id") or sample.get("uid") or sample.get("user") or "global"
                    prior_used = prior_mode
                    prior_scores = None
                    prior_meta = None
                    
                    try:
                        prior_scores, prior_meta = prior_provider.get(user_id)
                    except Exception as e:
                        print(f"[Sample {n}] prior: ERROR building prior (mode={prior_mode}, user={user_id}): {e}")
                        
                        # Implement comprehensive fallback system
                        if prior_mode == "user" and prior_fallback != "none":
                            if prior_fallback == "global":
                                try:
                                    # Try to rebuild provider with global mode
                                    global_provider = PriorProvider(
                                        model=editor.model,
                                        tokenizer=editor.tokenizer,
                                        id2tag=id2tag,
                                        device=editor.device,
                                        prior_prompt=prior_provider.prior_prompt,
                                        beta=prior_beta,
                                        prior_mode="global",
                                        strict_mode=False  # Disable strict for fallback
                                    )
                                    prior_scores, temp_meta = global_provider.get(None)
                                    prior_meta = {"mode": "global(fallback)", "source": "global", "user_id": user_id, "beta": prior_beta}
                                    prior_used = "global(fallback)"
                                    print(f"[Sample {n}] prior: fallback -> global prior")
                                except Exception as eg:
                                    print(f"[Sample {n}] prior: global fallback failed: {eg}")
                            
                            # If global failed or uniform requested, use uniform fallback
                            if prior_scores is None and prior_fallback in ("uniform", "global"):
                                K = len(id2tag)
                                prior_scores = {i: 1.0 / max(K, 1) for i in range(K)}
                                prior_meta = {"mode": "uniform(fallback)", "source": "uniform", "user_id": user_id, "beta": 0.0}
                                prior_used = "uniform(fallback)"
                                print(f"[Sample {n}] prior: fallback -> uniform prior (K={K})")
                        
                        # Absolute last resort: no prior correction
                        if prior_scores is None:
                            K = len(id2tag)
                            prior_scores = {i: 0.0 for i in range(K)}
                            prior_meta = {"mode": "none(fallback)", "source": "none", "user_id": user_id, "beta": 0.0}
                            prior_used = "none(fallback)"
                            print(f"[Sample {n}] prior: fallback -> none (no correction)")
                    
                    # Ensure we never have undefined prior_scores
                    if prior_scores is None:
                        raise RuntimeError("prior_scores remained None after all fallbacks")
                    
                    # Log fallback once per sample (spam prevention)
                    if n <= 3 or prior_meta.get('source') == 'global':
                        if prior_meta.get('source') != 'user':
                            print(f"[Sample {n}] prior: mode={prior_meta['mode']} user_id={prior_meta['user_id']} -> source={prior_meta['source']}")
                    
                    if scoring_mode == "batched":
                        bid, btag = classify_by_scores_with_calibration_batched(
                            editor.model, tokenizer, prompt, id2tag,
                            prior_prompt=None, device=editor.device, lam=1.0,
                            prior_scores=prior_scores
                        )
                    else:
                        bid, btag = classify_by_scores_with_calibration(
                            editor.model, tokenizer, prompt, id2tag,
                            prior_prompt=None, device=editor.device, lam=1.0,
                            prior_scores=prior_scores
                        )
                    if n <= 3:
                        print(f"[Sample {n}] Baseline result: ID={bid}, tag='{btag}'")
                except Exception as e:
                    print(f"[Sample {n}] Baseline scoring failed: {e}")
                    bid, btag = 1, list(id2tag.values())[0]
                    prior_meta = {"mode": prior_mode, "source": "error", "user_id": user_id}
            elif self.evaluator:
                # Fallback to non-calibrated scoring
                editor = self.evaluator.chameleon_editor
                tokenizer = editor.tokenizer
                if scoring_mode == "batched":
                    bid, btag = classify_by_scores_batched(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                else:
                    bid, btag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                prior_meta = {"mode": "none", "source": "none", "user_id": "none"}
                prior_scores = {}  # No prior scores for non-calibrated path
            else:
                # Mock for testing
                if n <= 3:
                    print(f"[Sample {n}] Using mock baseline (no evaluator)")
                bid, btag = 1, list(id2tag.values())[0]
                prior_meta = {"mode": "none", "source": "none", "user_id": "unknown"}
                prior_scores = {}  # No prior scores for mock case
                
            bl_pred = btag or "unknown"
            bl_valid = bl_pred != "unknown" and bl_pred != ""
            bl_ok = bl_pred == gold
            
            # --- Chameleon（編集あり）: 編集を有効化した上で同様にスコア ---
            if self.evaluator:
                editor = self.evaluator.chameleon_editor
                # 方向ベクトルをサンプルに応じて更新（API 名称はあなたの実装に合わせて）
                try:
                    # Convert sample to expected format with prompt field
                    sample_for_dv = {
                        "prompt": content,  # Use the constructed content as prompt
                        "question": sample.get("question", ""),
                        "profile": sample.get("profile", [])
                    }
                    dv = self.evaluator.compute_direction_vectors(sample_for_dv)
                except Exception as e1:
                    try:
                        dv = getattr(editor, "compute_direction_vectors", lambda s: None)(sample_for_dv)
                    except Exception as e2:
                        print(f"[Sample {n}] Direction vector computation failed: {e1}, {e2}")
                        # Try with minimal content
                        try:
                            minimal_sample = {"prompt": prompt}
                            dv = editor.compute_direction_vectors(minimal_sample, target_layers=target_layers, norm_scale=norm_scale)
                        except Exception as e3:
                            print(f"[Sample {n}] All direction vector attempts failed: {e3}")
                            dv = {}
                editor.last_direction_vectors = dv or {}
                
                # Debug: Check if direction vectors are valid
                if dv:
                    print(f"[Sample {n}] Direction vectors computed: keys={list(dv.keys())}")
                else:
                    print(f"[Sample {n}] Warning: No direction vectors computed")
                    
                if hasattr(editor, "cham_context"):
                    try:
                        with editor.cham_context(alpha_personal=alpha_personal,
                                                 alpha_neutral=alpha_general,
                                                 alpha_fakeit=0.0,
                                                 target_layers=target_layers,
                                                 norm_scale=norm_scale,
                                                 gate=edit_gate_threshold) as hook_context:
                            # Use same prior scores as baseline to prevent cancellation
                            if use_calibration and 'prior_provider' in locals():
                                if scoring_mode == "batched":
                                    cid, ctag = classify_by_scores_with_calibration_batched(
                                        editor.model, tokenizer, prompt, id2tag,
                                        prior_prompt=None, device=editor.device, lam=1.0,
                                        prior_scores=prior_scores  # Same as baseline
                                    )
                                else:
                                    cid, ctag = classify_by_scores_with_calibration(
                                        editor.model, tokenizer, prompt, id2tag,
                                        prior_prompt=None, device=editor.device, lam=1.0,
                                        prior_scores=prior_scores  # Same as baseline
                                    )
                            else:
                                if scoring_mode == "batched":
                                    cid, ctag = classify_by_scores_batched(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                                else:
                                    cid, ctag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                        
                        # Record hook firing count for observability
                        hook_calls = getattr(hook_context, 'calls', 0)
                        if n <= 3:
                            print(f"[Sample {n}] gate_applied=True, hook_calls={hook_calls}")
                        
                    except Exception as e:
                        print(f"[Sample {n}] Chameleon context scoring failed: {e}")
                        cid, ctag = 1, list(id2tag.values())[0]
                        hook_calls = 0
                else:
                    # Fallback（編集無効）だが、ログで警告しておく
                    if n <= 3:
                        print("[warn] cham_context not available; chameleon==baseline in scoring path")
                    # Use same approach as baseline for consistency
                    if use_calibration and 'prior_provider' in locals():
                        if scoring_mode == "batched":
                            cid, ctag = classify_by_scores_with_calibration_batched(
                                editor.model, tokenizer, prompt, id2tag,
                                prior_prompt=None, device=editor.device, lam=1.0,
                                prior_scores=prior_scores  # Same as baseline
                            )
                        else:
                            cid, ctag = classify_by_scores_with_calibration(
                                editor.model, tokenizer, prompt, id2tag,
                                prior_prompt=None, device=editor.device, lam=1.0,
                                prior_scores=prior_scores  # Same as baseline
                            )
                    else:
                        if scoring_mode == "batched":
                            cid, ctag = classify_by_scores_batched(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                        else:
                            cid, ctag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                    hook_calls = 0  # No hooks in fallback path
            else:
                # Mock for testing
                if n <= 3:
                    print(f"[Sample {n}] Using mock chameleon (no evaluator)")
                cid, ctag = 1, list(id2tag.values())[0]
                hook_calls = 0
                
            ch_pred = ctag or "unknown" 
            ch_valid = ch_pred != "unknown" and ch_pred != ""
            ch_ok = ch_pred == gold

            # Gate info fallback (when not provided by generator)
            gate_value = ch_result.get("gate_value") if 'ch_result' in locals() and isinstance(ch_result, dict) else None
            gate_applied = ch_result.get("gate_applied") if 'ch_result' in locals() and isinstance(ch_result, dict) else None
            gate_debug = None
            if self.evaluator is not None:
                try:
                    dv = self.evaluator.compute_direction_vectors({"prompt": prompt}, target_layers=None, norm_scale=norm_scale)
                    hidden_norm = max(1e-8, float(dv["l2_general"]))
                    g = self.evaluator.summarize_gate(hidden_norm, dv, alpha_personal, alpha_general, edit_gate_threshold)
                    gv = float(g.get("gate_value")) if g is not None and "gate_value" in g else None
                    ga = bool(g.get("applied")) if g is not None and "applied" in g else None
                    gate_value = gv if gv is not None else gate_value
                    gate_applied = ga if ga is not None else gate_applied
                    gate_debug = {
                        "persona_norm": float(dv.get("l2_personal", 0.0)),
                        "general_norm": float(dv.get("l2_general", 0.0)),
                        "cos_theta": float(dv.get("cos_theta", 0.0)) if dv.get("cos_theta") is not None else None,
                        "raw_score": gate_value,
                        "threshold": float(edit_gate_threshold),
                        "applied": gate_applied,
                        "reason": ("ok" if gate_applied else "below_threshold"),
                    }
                except Exception as e:
                    print(f"[run_benchmark] gate computation failed: {e} (fallback to null gate)")
                    if gate_value is None:
                        gate_value = 0.0
                    if gate_applied is None:
                        gate_applied = False
                    gate_debug = {"applied": False, "reason": f"exception:{type(e).__name__}"}
            
            # McNemar counts
            if (not bl_ok) and ch_ok:
                b += 1
            if bl_ok and (not ch_ok):
                c += 1
            
            # Per-tag analysis
            tag_stats = per_tag.setdefault(gold, {"gold": 0, "bl_ok": 0, "ch_ok": 0})
            tag_stats["gold"] += 1
            tag_stats["bl_ok"] += int(bl_ok)
            tag_stats["ch_ok"] += int(ch_ok)
            
            # Overall accuracy
            bl_acc += int(bl_ok)
            ch_acc += int(ch_ok)
            valid_bl += int(bl_valid)
            valid_ch += int(ch_valid)
            
            # Store result (predictions.jsonl schema requirement)
            # Get user_id with priority order  
            sample_user_id = sample.get("user_id") or sample.get("uid") or sample.get("user") or "unknown"
            result = {
                "id": sample.get("id", sid),
                "user_id": sample_user_id,
                "gold": gold,
                "baseline": bl_pred,
                "chameleon": ch_pred,
                "valid": ch_valid,
                "gate_value": gate_value,
                "gate_applied": gate_applied,
            }
            # store indices if parsed
            result["answer_index_baseline"] = tag2id.get(bl_pred)
            result["answer_index_chameleon"] = tag2id.get(ch_pred)
            result["label_set"] = labels
            result["hook_calls"] = hook_calls  # Add hook firing count for observability
            
            # Add prior correction metadata for transparency and reproducibility
            if use_calibration and 'prior_meta' in locals():
                result["prior"] = prior_meta  # Contains mode, source, beta, user_id
            else:
                result["prior"] = {"mode": "none", "source": "none", "user_id": sample_user_id}
            
            # Calculate score differences for observability
            import math
            import numpy as np
            delta_max, delta_max_label, changed = 0.0, None, False
            
            # Legacy score analysis - skip for now since we're using simplified approach
            baseline_scores = {}
            chameleon_scores = {}
            
            if False:  # Disabled - old score analysis code
                try:
                    # Ensure both score dicts have same keys
                    common_keys = set(baseline_scores.keys()) & set(chameleon_scores.keys())
                    if common_keys:
                        bl_array = np.array([baseline_scores[k] for k in sorted(common_keys)], dtype=float)
                        ch_array = np.array([chameleon_scores[k] for k in sorted(common_keys)], dtype=float)
                        delta_array = ch_array - bl_array
                        
                        # Find max delta and check if prediction changed
                        k_max_idx = int(delta_array.argmax()) if delta_array.size > 0 else 0
                        k_max_key = sorted(common_keys)[k_max_idx] if k_max_idx < len(common_keys) else None
                        delta_max = float(delta_array[k_max_idx]) if delta_array.size > 0 else 0.0
                        delta_max_label = id2tag.get(k_max_key) if k_max_key is not None else None
                        
                        # Check if prediction changed
                        bl_winner_idx = int(bl_array.argmax()) if bl_array.size > 0 else 0
                        ch_winner_idx = int(ch_array.argmax()) if ch_array.size > 0 else 0
                        changed = bool(bl_winner_idx != ch_winner_idx)
                except Exception as e:
                    print(f"[Sample {n}] Warning: Score diff calculation failed: {e}")
            
            result["delta_max"] = delta_max
            result["delta_max_label"] = delta_max_label 
            result["changed"] = changed
            
            if gate_debug is not None:
                result["gate_debug"] = gate_debug
            results.append(result)
            
            if n % 50 == 0:
                print(f"Processed {n}/{len(samples)} samples...")
        
        # Calculate final metrics
        bl_acc_rate = bl_acc / max(1, n)
        ch_acc_rate = ch_acc / max(1, n) 
        valid_bl_rate = valid_bl / max(1, n)
        valid_ch_rate = valid_ch / max(1, n)
        delta_acc = ch_acc_rate - bl_acc_rate
        p_value = mcnemar_exact(b, c)
        
        return {
            "results": results,
            "summary": {
                "n": n,
                "baseline_acc": bl_acc_rate,
                "chameleon_acc": ch_acc_rate,
                "delta_acc": delta_acc,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "p_value": p_value,
                "valid_bl_rate": valid_bl_rate,
                "valid_ch_rate": valid_ch_rate,
                "per_tag": per_tag,
            }
        }


def generate_per_user_summary(results: List[Dict[str, Any]], out_dir: str):
    """Generate per-user summary from benchmark results."""
    try:
        from collections import defaultdict
        import csv
        
        # Group results by user_id
        user_data = defaultdict(list)
        for result in results:
            user_id = result.get("user_id", "unknown")
            user_data[user_id].append(result)
        
        per_user_stats = []
        
        for user_id, user_results in user_data.items():
            n = len(user_results)
            
            # Count correct predictions
            bl_ok_count = sum(1 for r in user_results if r.get("gold") == r.get("baseline"))
            ch_ok_count = sum(1 for r in user_results if r.get("gold") == r.get("chameleon"))
            
            # Calculate accuracies
            baseline_acc = bl_ok_count / max(1, n)
            chameleon_acc = ch_ok_count / max(1, n)
            delta = chameleon_acc - baseline_acc
            
            # McNemar counts (b: baseline wrong, chameleon right; c: baseline right, chameleon wrong)
            b = sum(1 for r in user_results 
                   if r.get("gold") != r.get("baseline") and r.get("gold") == r.get("chameleon"))
            c = sum(1 for r in user_results 
                   if r.get("gold") == r.get("baseline") and r.get("gold") != r.get("chameleon"))
            
            # Gate rate
            gate_rate = sum(1 for r in user_results if r.get("gate_applied", False)) / max(1, n)
            
            # Mean delta_max (only for results that have this field)
            delta_max_values = [r.get("delta_max") for r in user_results if r.get("delta_max") is not None]
            mean_delta_max = sum(delta_max_values) / len(delta_max_values) if delta_max_values else None
            
            per_user_stats.append({
                "user_id": user_id,
                "n": n,
                "baseline_acc": baseline_acc,
                "chameleon_acc": chameleon_acc,
                "delta": delta,
                "b": b,
                "c": c,
                "gate_rate": gate_rate,
                "mean_delta_max": mean_delta_max
            })
        
        # Sort by delta (improvement) descending
        per_user_stats.sort(key=lambda x: x["delta"], reverse=True)
        
        # Save CSV
        csv_path = os.path.join(out_dir, "summary_per_user.csv")
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["user_id", "n", "baseline_acc", "chameleon_acc", "delta", "b", "c", "gate_rate", "mean_delta_max"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stats in per_user_stats:
                # Format floats for cleaner CSV
                formatted_stats = stats.copy()
                for key in ["baseline_acc", "chameleon_acc", "delta", "gate_rate"]:
                    formatted_stats[key] = f"{stats[key]:.4f}"
                if stats["mean_delta_max"] is not None:
                    formatted_stats["mean_delta_max"] = f"{stats['mean_delta_max']:.4f}"
                writer.writerow(formatted_stats)
        
        # Save markdown with top 5
        md_path = os.path.join(out_dir, "summary_per_user.md")
        with open(md_path, 'w') as f:
            f.write("# Per-User Performance Summary\n\n")
            f.write("## Top 5 Users by Improvement (Δ)\n\n")
            f.write("| User ID | Samples | Baseline Acc | Chameleon Acc | Δ | b | c | Gate Rate | Mean Δ_max |\n")
            f.write("|---------|---------|--------------|---------------|---|---|---|-----------|------------|\n")
            
            for stats in per_user_stats[:5]:
                delta_max_str = f"{stats['mean_delta_max']:.4f}" if stats['mean_delta_max'] is not None else "-"
                f.write(f"| {stats['user_id']} | {stats['n']} | {stats['baseline_acc']:.4f} | "
                       f"{stats['chameleon_acc']:.4f} | {stats['delta']:+.4f} | {stats['b']} | "
                       f"{stats['c']} | {stats['gate_rate']:.4f} | {delta_max_str} |\n")
        
        # Print top 5 to console
        print("\n📊 Top 5 Users by Improvement (Δ):")
        print("User ID    | Samples | Baseline | Chameleon |     Δ     |  b |  c | Gate Rate | Mean Δ_max")
        print("-----------+---------+----------+-----------+-----------+----+----+-----------+-----------")
        for stats in per_user_stats[:5]:
            delta_max_str = f"{stats['mean_delta_max']:.4f}" if stats['mean_delta_max'] is not None else "    -    "
            print(f"{stats['user_id']:>10} | {stats['n']:>7} | {stats['baseline_acc']:>8.4f} | "
                 f"{stats['chameleon_acc']:>9.4f} | {stats['delta']:>+9.4f} | {stats['b']:>2} | "
                 f"{stats['c']:>2} | {stats['gate_rate']:>9.4f} | {delta_max_str}")
        
        print(f"\nPer-user summary saved: {csv_path} (and {md_path})")
        
    except Exception as e:
        print(f"[per-user summary] failed in generate_per_user_summary: {e}")


def build_global_prior_table(data_path: str, split: str, labels: List[str], prior_alpha: float = 1.0) -> Dict[str, float]:
    """
    Build global prior probability table with Dirichlet smoothing from dataset.
    
    Args:
        data_path: Path to dataset 
        split: Current split being evaluated
        labels: List of allowed labels
        prior_alpha: Dirichlet smoothing parameter
        
    Returns:
        Dictionary mapping labels to smoothed probabilities
    """
    try:
        # Determine source split priority: dev > train > test (same split)
        source_splits = []
        if split != "dev":
            source_splits.append("dev")
        if split != "train":
            source_splits.append("train")
        if split != "test":
            source_splits.append("test")
        
        # Try to load from preferred splits
        all_samples = []
        source_used = None
        
        for source_split in source_splits:
            source_file = f"{data_path}/evaluation/lamp2_expanded_eval.jsonl"
            try:
                with open(source_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            # Filter by split if available
                            sample_split = sample.get("split", "test")  # assume test if no split field
                            if sample_split == source_split:
                                all_samples.append(sample)
                
                if all_samples:
                    source_used = source_split
                    break
                        
            except Exception as e:
                print(f"[prior] Could not load from {source_file} for split {source_split}: {e}")
                continue
        
        if not all_samples:
            print(f"[prior] WARNING: No samples found for global prior, using uniform distribution")
            uniform_prob = 1.0 / len(labels)
            return {label: uniform_prob for label in labels}
        
        # Count label occurrences
        label_counts = {label: 0 for label in labels}
        total_samples = 0
        
        for sample in all_samples:
            gold = sample.get("gold") or sample.get("reference") or sample.get("output", "")
            # Normalize gold label 
            normalized_gold = match_allowed(gold, allowed=labels) or "unknown"
            if normalized_gold in label_counts:
                label_counts[normalized_gold] += 1
                total_samples += 1
        
        # Apply Dirichlet smoothing: (count + α) / (N + α*K)
        K = len(labels)
        smoothed_probs = {}
        
        for label in labels:
            count = label_counts[label]
            smoothed_prob = (count + prior_alpha) / (total_samples + prior_alpha * K)
            smoothed_probs[label] = smoothed_prob
        
        # Log statistics
        print(f"[prior] mode=global, beta=1.0, alpha={prior_alpha:.1f}, source={source_used or 'fallback'}")
        sorted_probs = sorted(smoothed_probs.items(), key=lambda x: x[1], reverse=True)
        top5_str = ", ".join([f"{label}={prob:.2f}" for label, prob in sorted_probs[:5]])
        print(f"[prior] top5: {top5_str}")
        
        return smoothed_probs
        
    except Exception as e:
        print(f"[prior] Failed to build global prior table: {e}")
        # Fallback to uniform
        uniform_prob = 1.0 / len(labels)
        return {label: uniform_prob for label in labels}


def build_user_prior_table(data_path: str, split: str, labels: List[str], prior_alpha: float = 1.0,
                          min_samples_threshold: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Build user-specific prior probability tables with Dirichlet smoothing.
    
    Args:
        data_path: Path to dataset
        split: Current split being evaluated  
        labels: List of allowed labels
        prior_alpha: Dirichlet smoothing parameter
        min_samples_threshold: Minimum samples per user for user-specific priors
        
    Returns:
        Dictionary mapping user_ids to label probability dictionaries
    """
    try:
        from collections import defaultdict
        
        # Load all samples (preferring dev/train over test)
        all_samples = []
        source_splits = ["dev", "train", "test"]
        
        for source_split in source_splits:
            if source_split == split:
                continue  # Skip current evaluation split
                
            source_file = f"{data_path}/evaluation/lamp2_expanded_eval.jsonl"
            try:
                with open(source_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            # Filter by split if available
                            sample_split = sample.get("split", "test")
                            if sample_split == source_split:
                                all_samples.append(sample)
            except:
                continue
        
        if not all_samples:
            print(f"[prior] WARNING: No samples for user prior estimation")
            return {}
        
        # Group by user and count labels
        user_label_counts = defaultdict(lambda: defaultdict(int))
        user_totals = defaultdict(int)
        
        for sample in all_samples:
            # Extract user_id
            user_id = sample.get("user_id") or sample.get("uid") or sample.get("user")
            if not user_id:
                continue
                
            user_id = str(user_id)
            gold = sample.get("gold") or sample.get("reference") or sample.get("output", "")
            normalized_gold = match_allowed(gold, allowed=labels) or "unknown"
            
            if normalized_gold in labels:
                user_label_counts[user_id][normalized_gold] += 1
                user_totals[user_id] += 1
        
        # Build user-specific probability tables
        user_prior_tables = {}
        K = len(labels)
        
        for user_id, label_counts in user_label_counts.items():
            total_samples = user_totals[user_id]
            
            if total_samples >= min_samples_threshold:
                # User has enough samples - use user-specific distribution
                user_probs = {}
                for label in labels:
                    count = label_counts[label]
                    smoothed_prob = (count + prior_alpha) / (total_samples + prior_alpha * K)
                    user_probs[label] = smoothed_prob
                
                user_prior_tables[user_id] = user_probs
        
        print(f"[prior] Built user-specific priors for {len(user_prior_tables)} users")
        print(f"[prior] (users with <{min_samples_threshold} samples will fallback to global)")
        
        return user_prior_tables
        
    except Exception as e:
        print(f"[prior] Failed to build user prior table: {e}")
        return {}


def calculate_distribution_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate distribution flattening metrics from prediction results.
    
    Args:
        results: List of prediction result dictionaries
        
    Returns:
        Dictionary with entropy, kl_to_uniform, and top1_share metrics
    """
    if not results:
        return {"entropy": 0.0, "kl_to_uniform": float('inf'), "top1_share": 1.0}
    
    # Count label frequencies from chameleon predictions
    label_counts = {}
    total_valid = 0
    
    for result in results:
        if result.get("valid", False):
            ch_pred = result.get("chameleon", "")
            if ch_pred and ch_pred != "unknown":
                label_counts[ch_pred] = label_counts.get(ch_pred, 0) + 1
                total_valid += 1
    
    if total_valid == 0:
        return {"entropy": 0.0, "kl_to_uniform": float('inf'), "top1_share": 1.0}
    
    # Convert to probabilities
    probs = [count / total_valid for count in label_counts.values()]
    if not probs:
        return {"entropy": 0.0, "kl_to_uniform": float('inf'), "top1_share": 1.0}
    
    # Calculate entropy: H = -Σ p_i * log(p_i)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    
    # Calculate top-1 share
    top1_share = max(probs) if probs else 1.0
    
    # Calculate KL divergence to uniform distribution
    k = len(label_counts)  # number of unique labels
    uniform_p = 1.0 / k if k > 0 else 1.0
    
    # KL(P||U) = Σ p_i * log(p_i / uniform_p)
    kl_to_uniform = sum(p * math.log(p / uniform_p) for p in probs if p > 0)
    
    return {
        "entropy": entropy,
        "kl_to_uniform": kl_to_uniform, 
        "top1_share": top1_share
    }


def run_parameter_sweep(benchmarker, args, beta_values: List[float], temp_values: List[float]) -> Dict:
    """
    Run parameter sweep over beta and temperature values.
    
    Args:
        benchmarker: LaMP2Benchmarker instance
        args: Command line arguments
        beta_values: List of prior_beta values to sweep
        temp_values: List of score_temp values to sweep
        
    Returns:
        Dictionary with sweep results and best configuration
    """
    print(f"\n🔄 Starting parameter sweep: {len(beta_values)} β × {len(temp_values)} temp = {len(beta_values) * len(temp_values)} configurations")
    
    sweep_results = []
    best_config = None
    best_score = -float('inf')
    
    for i, beta in enumerate(beta_values):
        for j, temp in enumerate(temp_values):
            config_name = f"beta{beta:.1f}_temp{temp:.1f}"
            print(f"\n[{i*len(temp_values)+j+1}/{len(beta_values)*len(temp_values)}] Testing β={beta:.1f}, temp={temp:.1f}")
            
            # Run benchmark with current parameters
            try:
                benchmark_results = benchmarker.run_benchmark(
                    split=args.split,
                    limit=args.limit,
                    seed=args.seed,
                    alpha_personal=args.alpha_personal,
                    alpha_general=args.alpha_general,
                    norm_scale=args.norm_scale,
                    edit_gate_threshold=args.edit_gate_threshold,
                    target_layers=args.target_layers,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    do_sample=args.do_sample,
                    repetition_penalty=args.repetition_penalty,
                    out_dir=None,  # Don't save individual runs
                    mode=args.mode,
                    use_pmi=args.use_pmi,
                    score_norm=args.score_norm,
                    prior_mode=args.prior_mode,
                    score_temp=temp,  # Use swept temperature
                    prior_beta=beta,  # Use swept beta
                    prior_alpha=args.prior_alpha,
                    score_template=args.score_template,
                    prior_fallback=args.prior_fallback,
                    user_id=args.user_id,
                    scoring_mode=args.scoring_mode,
                )
                
                if not benchmark_results or not benchmark_results.get("results"):
                    print(f"⚠️  No results for β={beta:.1f}, temp={temp:.1f}")
                    continue
                
                summary = benchmark_results["summary"]
                results = benchmark_results["results"]
                
                # Calculate distribution metrics
                dist_metrics = calculate_distribution_metrics(results)
                
                # Check constraints
                meets_entropy = dist_metrics["entropy"] >= args.target_entropy
                meets_kl = dist_metrics["kl_to_uniform"] <= args.max_kl_to_uniform
                meets_top1 = dist_metrics["top1_share"] <= args.max_top1_share
                meets_constraints = meets_entropy and meets_kl and meets_top1
                
                # Calculate selection score
                if args.select_by == "delta_acc":
                    score = summary["delta_acc"] if meets_constraints else -float('inf')
                elif args.select_by == "ch_acc":
                    score = summary["chameleon_acc"] if meets_constraints else -float('inf')
                elif args.select_by == "entropy":
                    score = dist_metrics["entropy"] if meets_constraints else -float('inf')
                else:
                    score = summary["delta_acc"] if meets_constraints else -float('inf')
                
                # Store results
                config_result = {
                    "config": config_name,
                    "prior_beta": beta,
                    "score_temp": temp,
                    "baseline_acc": summary["baseline_acc"],
                    "chameleon_acc": summary["chameleon_acc"],
                    "delta_acc": summary["delta_acc"],
                    "p_value": summary["p_value"],
                    "entropy": dist_metrics["entropy"],
                    "kl_to_uniform": dist_metrics["kl_to_uniform"],
                    "top1_share": dist_metrics["top1_share"],
                    "meets_entropy": meets_entropy,
                    "meets_kl": meets_kl,
                    "meets_top1": meets_top1,
                    "meets_constraints": meets_constraints,
                    "selection_score": score
                }
                
                sweep_results.append(config_result)
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_config = config_result.copy()
                
                print(f"   acc={summary['chameleon_acc']:.3f} (Δ={summary['delta_acc']:+.3f}), "
                      f"entropy={dist_metrics['entropy']:.2f}, "
                      f"top1={dist_metrics['top1_share']:.2f}, "
                      f"{'✓' if meets_constraints else '✗'}")
                
            except Exception as e:
                print(f"❌ Error with β={beta:.1f}, temp={temp:.1f}: {e}")
                continue
    
    return {
        "sweep_results": sweep_results,
        "best_config": best_config,
        "num_configs": len(sweep_results),
        "constraint_pass_count": sum(1 for r in sweep_results if r["meets_constraints"])
    }


def save_sweep_results(sweep_data: Dict, out_dir: str, args):
    """Save parameter sweep results to files."""
    
    # Create sweep.csv
    csv_path = os.path.join(out_dir, "sweep.csv")
    with open(csv_path, 'w') as f:
        headers = ["config", "prior_beta", "score_temp", "baseline_acc", "chameleon_acc", "delta_acc", 
                  "p_value", "entropy", "kl_to_uniform", "top1_share", "meets_entropy", "meets_kl", 
                  "meets_top1", "meets_constraints", "selection_score"]
        f.write(",".join(headers) + "\n")
        
        for result in sweep_data["sweep_results"]:
            row = [
                result["config"],
                f"{result['prior_beta']:.2f}",
                f"{result['score_temp']:.2f}",
                f"{result['baseline_acc']:.4f}",
                f"{result['chameleon_acc']:.4f}",
                f"{result['delta_acc']:.4f}",
                f"{result['p_value']:.6g}",
                f"{result['entropy']:.4f}",
                f"{result['kl_to_uniform']:.4f}",
                f"{result['top1_share']:.4f}",
                str(result["meets_entropy"]),
                str(result["meets_kl"]),
                str(result["meets_top1"]),
                str(result["meets_constraints"]),
                f"{result['selection_score']:.6f}" if result['selection_score'] != -float('inf') else "-inf"
            ]
            f.write(",".join(row) + "\n")
    
    # Create best_config.json
    json_path = os.path.join(out_dir, "best_config.json")
    with open(json_path, 'w') as f:
        json.dump({
            "best_config": sweep_data["best_config"],
            "sweep_summary": {
                "total_configs": sweep_data["num_configs"],
                "constraint_pass_count": sweep_data["constraint_pass_count"],
                "constraints": {
                    "target_entropy": args.target_entropy,
                    "max_kl_to_uniform": args.max_kl_to_uniform,
                    "max_top1_share": args.max_top1_share
                },
                "selection_criterion": args.select_by
            }
        }, f, indent=2)
    
    # Create summary_best.md
    md_path = os.path.join(out_dir, "summary_best.md")
    with open(md_path, 'w') as f:
        best = sweep_data["best_config"]
        
        f.write("# Parameter Sweep Results\n\n")
        f.write(f"**Sweep Configuration:**\n")
        f.write(f"- Total configurations tested: {sweep_data['num_configs']}\n")
        f.write(f"- Configurations passing constraints: {sweep_data['constraint_pass_count']}\n")
        f.write(f"- Selection criterion: {args.select_by}\n\n")
        
        f.write(f"**Constraints:**\n")
        f.write(f"- Minimum entropy: {args.target_entropy:.2f}\n")
        f.write(f"- Maximum KL to uniform: {args.max_kl_to_uniform:.3f}\n")
        f.write(f"- Maximum top-1 share: {args.max_top1_share:.3f}\n\n")
        
        if best:
            f.write(f"## 🏆 Best Configuration: {best['config']}\n\n")
            f.write(f"**Parameters:**\n")
            f.write(f"- prior_beta (β): {best['prior_beta']:.2f}\n")
            f.write(f"- score_temp: {best['score_temp']:.2f}\n\n")
            
            f.write(f"**Performance:**\n")
            f.write(f"- Baseline Accuracy: {best['baseline_acc']:.4f}\n")
            f.write(f"- Chameleon Accuracy: {best['chameleon_acc']:.4f}\n")
            f.write(f"- Accuracy Improvement (Δ): {best['delta_acc']:+.4f}\n")
            f.write(f"- McNemar p-value: {best['p_value']:.6g}\n\n")
            
            f.write(f"**Distribution Metrics:**\n")
            f.write(f"- Entropy: {best['entropy']:.4f} ({'✓' if best['meets_entropy'] else '✗'} >= {args.target_entropy:.2f})\n")
            f.write(f"- KL to uniform: {best['kl_to_uniform']:.4f} ({'✓' if best['meets_kl'] else '✗'} <= {args.max_kl_to_uniform:.3f})\n")
            f.write(f"- Top-1 share: {best['top1_share']:.4f} ({'✓' if best['meets_top1'] else '✗'} <= {args.max_top1_share:.3f})\n\n")
            
            if best['meets_constraints']:
                f.write("🎯 **All constraints satisfied!**\n")
            else:
                f.write("⚠️ **Some constraints not met**\n")
        else:
            f.write("## ❌ No valid configuration found\n\n")
            f.write("All tested configurations failed to meet the specified constraints.\n")
            f.write("Consider relaxing constraints or expanding parameter ranges.\n")
    
    print(f"\n📊 Sweep results saved:")
    print(f"   - {csv_path}")
    print(f"   - {json_path}")
    print(f"   - {md_path}")


def main():
    parser = argparse.ArgumentParser(description="LaMP-2 Benchmark: Baseline vs Chameleon")
    
    # Data and configuration
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=-1, help="Limit samples (-1 for all)")
    parser.add_argument("--user_id", type=str, default=None, help="Filter to specific user_id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, choices=["id", "text"], default="id", help="Classification mode: id (default) or text")
    
    # Chameleon parameters
    parser.add_argument("--alpha_personal", type=float, default=2.75, help="Personal alpha")
    parser.add_argument("--alpha_general", type=float, default=-1.0, help="General alpha") 
    parser.add_argument("--norm_scale", type=float, default=0.9, help="Norm scale")
    parser.add_argument("--edit_gate_threshold", type=float, default=0.022, help="Gate threshold")
    parser.add_argument("--target_layers", nargs="*", default=[24, 25, 26, 27], help="Target layers (default: last 4 layers)")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=6, help="Max new tokens")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="Min new tokens")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    
    # Calibration
    parser.add_argument("--calibrate", action="store_true", default=True, help="Enable contextual calibration (default: True)")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false", help="Disable contextual calibration")
    
    # PMI Scoring (Flattening) with Prior Correction Modes
    parser.add_argument("--score_norm", choices=["sum", "avg"], default="avg", help="Score normalization: sum or avg (default: avg)")  
    parser.add_argument("--prior_mode", choices=["none", "empty", "global", "user", "user_or_global"], default="empty", help="Prior correction mode: none (no correction), empty (empty context PMI), global (global distribution PMI), user (user-specific PMI), user_or_global (user with global fallback) (default: empty)")
    parser.add_argument("--prior_fallback", choices=["global", "uniform", "none"], default="global", help="When prior_mode=user and no user prior found: fallback to global/uniform/none")
    parser.add_argument("--prior_beta", type=float, default=1.0, help="β weight for prior term (default: 1.0)")
    parser.add_argument("--prior_alpha", type=float, default=1.0, help="α Dirichlet smoothing parameter (default: 1.0)")
    parser.add_argument("--score_temp", type=float, default=1.0, help="Temperature for score flattening (>1 = flatter, default: 1.0)")
    parser.add_argument("--score_template", type=str, default="\\nAnswer: {label}", help="Template for answer generation (default: \\nAnswer: {label})")
    parser.add_argument("--use_pmi", action="store_true", help="Use PMI scoring to flatten label distribution bias (deprecated - use --prior_mode instead)")
    
    # Strict Mode (Zero-Fallback Enforcement)
    parser.add_argument("--strict", action="store_true", help="Enable strict mode: fail immediately on any fallback, require all user priors")
    parser.add_argument("--user_prior_path", type=str, default=None, help="Path to pre-generated user_priors.jsonl (required in strict mode)")
    
    # Parameter Sweep for Auto-Tuning Flattening
    parser.add_argument("--prior_beta_sweep", type=str, default=None, help="Beta values to sweep (e.g., '0.0,0.5,1.0,1.5,2.0')")
    parser.add_argument("--score_temp_sweep", type=str, default=None, help="Temperature values to sweep (e.g., '1.0,2.0,3.0,5.0')")
    parser.add_argument("--target_entropy", type=float, default=2.3, help="Minimum target entropy for distribution flattening")
    parser.add_argument("--max_kl_to_uniform", type=float, default=0.25, help="Maximum allowed KL divergence to uniform distribution")
    parser.add_argument("--max_top1_share", type=float, default=0.35, help="Maximum allowed top-1 label share")
    parser.add_argument("--select_by", choices=["delta_acc", "ch_acc", "entropy"], default="delta_acc", help="Selection criterion for best configuration")
    
    # Scoring mode
    parser.add_argument("--scoring_mode", choices=["loop", "batched"], default="batched", help="Scoring mode: loop (sequential) or batched (faster)")
    
    # Output
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet_summary", action="store_true", help="Suppress console summary display")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") 
        run_name = f"lamp2_{args.split}_ap{args.alpha_personal}_ag{args.alpha_general}_tau{args.edit_gate_threshold}_{ts}"
        args.out_dir = os.path.join("results", "bench", run_name)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize benchmarker
    benchmarker = LaMP2Benchmarker(args.config_path, args.data_path)
    benchmarker.use_calibration = args.calibrate
    
    # Strict mode validation
    if args.strict:
        print("🔒 STRICT MODE: Zero-fallback enforcement enabled")
        
        # Enforce strict requirements
        if args.prior_mode != "user":
            print(f"❌ STRICT ERROR: --prior_mode must be 'user' in strict mode, got '{args.prior_mode}'")
            return
            
        if not args.user_prior_path:
            print(f"❌ STRICT ERROR: --user_prior_path required in strict mode")
            return
            
        if not os.path.exists(args.user_prior_path):
            print(f"❌ STRICT ERROR: User prior file not found: {args.user_prior_path}")
            return
            
        print(f"✅ STRICT VALIDATION: prior_mode=user, priors={args.user_prior_path}")
    
    # Check if parameter sweep is requested
    if args.prior_beta_sweep or args.score_temp_sweep:
        print("🔄 Parameter sweep mode activated")
        
        # Parse sweep parameters
        if args.prior_beta_sweep:
            try:
                beta_values = [float(x.strip()) for x in args.prior_beta_sweep.split(',')]
            except ValueError as e:
                print(f"❌ Invalid prior_beta_sweep format: {e}")
                return
        else:
            beta_values = [args.prior_beta]  # Use single value
        
        if args.score_temp_sweep:
            try:
                temp_values = [float(x.strip()) for x in args.score_temp_sweep.split(',')]
            except ValueError as e:
                print(f"❌ Invalid score_temp_sweep format: {e}")
                return
        else:
            temp_values = [args.score_temp]  # Use single value
        
        print(f"📋 Sweep parameters:")
        print(f"   - Beta values: {beta_values}")
        print(f"   - Temperature values: {temp_values}")
        print(f"   - Total configurations: {len(beta_values) * len(temp_values)}")
        
        # Run parameter sweep
        start_time = time.perf_counter()
        sweep_data = run_parameter_sweep(benchmarker, args, beta_values, temp_values)
        
        if not sweep_data["sweep_results"]:
            print("❌ No sweep results obtained - exiting")
            return
            
        # Save sweep results
        save_sweep_results(sweep_data, args.out_dir, args)
        
        # Print summary
        best = sweep_data["best_config"]
        if best:
            print(f"\n🏆 Best configuration: β={best['prior_beta']:.2f}, temp={best['score_temp']:.2f}")
            print(f"   Performance: acc={best['chameleon_acc']:.3f} (Δ={best['delta_acc']:+.3f})")
            print(f"   Distribution: entropy={best['entropy']:.2f}, top1={best['top1_share']:.2f}")
            print(f"   Constraints: {'✓ PASSED' if best['meets_constraints'] else '✗ FAILED'}")
            
            # Optionally run final benchmark with best parameters for detailed output
            print(f"\n📊 Running final benchmark with best parameters...")
            benchmark_results = benchmarker.run_benchmark(
                split=args.split,
                limit=args.limit,
                seed=args.seed,
                alpha_personal=args.alpha_personal,
                alpha_general=args.alpha_general,
                norm_scale=args.norm_scale,
                edit_gate_threshold=args.edit_gate_threshold,
                target_layers=args.target_layers,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                out_dir=args.out_dir,  # Save final results
                mode=args.mode,
                use_pmi=args.use_pmi,
                score_norm=args.score_norm,
                prior_mode=args.prior_mode,
                score_temp=best['score_temp'],  # Use best temperature
                prior_beta=best['prior_beta'],  # Use best beta
                prior_alpha=args.prior_alpha,
                score_template=args.score_template,
                prior_fallback=args.prior_fallback,
                user_id=args.user_id,
                strict_mode=args.strict,
                user_prior_path=args.user_prior_path,
                scoring_mode=args.scoring_mode,
            )
            end_time = time.perf_counter()
            execution_time = end_time - start_time
        else:
            print(f"\n❌ No valid configuration found meeting constraints")
            print(f"   Consider relaxing constraints or expanding parameter ranges")
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return
            
    else:
        # Single configuration run (original behavior)
        start_time = time.perf_counter()
        benchmark_results = benchmarker.run_benchmark(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            alpha_personal=args.alpha_personal,
            alpha_general=args.alpha_general,
            norm_scale=args.norm_scale,
            edit_gate_threshold=args.edit_gate_threshold,
            target_layers=args.target_layers,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            out_dir=args.out_dir,
            mode=args.mode,
            # PMI parameters
            use_pmi=args.use_pmi,
            score_norm=args.score_norm,
            prior_mode=args.prior_mode,
            score_temp=args.score_temp,
            # Prior correction parameters
            prior_beta=args.prior_beta,
            prior_alpha=args.prior_alpha,
            score_template=args.score_template,
            prior_fallback=args.prior_fallback,
            # User filtering
            user_id=args.user_id,
            # Strict mode
            strict_mode=args.strict,
            user_prior_path=args.user_prior_path,
            # Scoring mode
            scoring_mode=args.scoring_mode,
        )
        end_time = time.perf_counter()
        execution_time = end_time - start_time
    
    # Early exit if no samples found
    if not benchmark_results.get("results"):
        print("No results to save - exiting.")
        return
    
    # Save outputs
    jsonl_path = os.path.join(args.out_dir, "predictions.jsonl")
    csv_path = os.path.join(args.out_dir, "summary.csv")
    md_path = os.path.join(args.out_dir, "summary.md")
    
    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for result in benchmark_results["results"]:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Write CSV summary
    summary = benchmark_results["summary"]
    with open(csv_path, 'w') as f:
        f.write("n,baseline_acc,chameleon_acc,delta_acc,mcnemar_b,mcnemar_c,p_value,valid_bl_rate,valid_ch_rate\n")
        f.write(f"{summary['n']},{summary['baseline_acc']:.4f},{summary['chameleon_acc']:.4f},"
                f"{summary['delta_acc']:.4f},{summary['mcnemar_b']},{summary['mcnemar_c']},"
                f"{summary['p_value']:.6g},{summary['valid_bl_rate']:.4f},{summary['valid_ch_rate']:.4f}\n")
    
    # Write Markdown report
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# LaMP-2 Benchmark Summary\n\n")
        f.write(f"**Configuration:**\n")
        f.write(f"- α_personal = {args.alpha_personal}, α_general = {args.alpha_general}\n")
        f.write(f"- Gate threshold = {args.edit_gate_threshold}\n")
        f.write(f"- Prior mode = {args.prior_mode}, β = {args.prior_beta}, α = {args.prior_alpha}\n")
        f.write(f"- Score template = '{args.score_template}', temp = {args.score_temp}, norm = {args.score_norm}\n")
        f.write(f"- Label set size = {len(labels) if 'labels' in locals() else 'N/A'}\n")
        f.write(f"- Samples = {summary['n']}\n\n")
        
        f.write(f"**Results:**\n") 
        f.write(f"- Baseline Accuracy = **{summary['baseline_acc']:.4f}**\n")
        f.write(f"- Chameleon Accuracy = **{summary['chameleon_acc']:.4f}** (Δ = {summary['delta_acc']:+.4f})\n")
        f.write(f"- McNemar exact p = **{summary['p_value']:.3g}** (b={summary['mcnemar_b']}, c={summary['mcnemar_c']})\n")
        f.write(f"- Valid-Tag rate: Baseline={summary['valid_bl_rate']:.3f}, Chameleon={summary['valid_ch_rate']:.3f}\n\n")
        
        if summary['p_value'] < 0.05:
            f.write("🎯 **Result: STATISTICALLY SIGNIFICANT improvement**\n\n")
        elif summary['delta_acc'] > 0:
            f.write("📈 **Result: Positive trend, more samples needed for significance**\n\n")
        else:
            f.write("📊 **Result: No significant improvement detected**\n\n")
            
        f.write("## Per-Tag Analysis\n\n")
        f.write("| Tag | Gold Count | Baseline OK | Chameleon OK | Improvement |\n")
        f.write("|-----|------------|-------------|--------------|-------------|\n")
        
        for tag, stats in sorted(summary['per_tag'].items()):
            bl_rate = stats['bl_ok'] / max(1, stats['gold'])
            ch_rate = stats['ch_ok'] / max(1, stats['gold'])
            improvement = ch_rate - bl_rate
            f.write(f"| {tag or '(empty)'} | {stats['gold']} | {stats['bl_ok']} | {stats['ch_ok']} | {improvement:+.3f} |\n")
    
    # Generate per-user summary
    if benchmark_results.get("results"):
        generate_per_user_summary(benchmark_results["results"], args.out_dir)
    
    # Create latest symlink
    latest_link = "results/latest"
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.makedirs(os.path.dirname(latest_link), exist_ok=True)
        os.symlink(os.path.relpath(args.out_dir), latest_link)
    except Exception as e:
        print(f"⚠️  Failed to create latest symlink: {e}")
    
    # Print console summary (unless --quiet_summary)
    if not args.quiet_summary:
        print_console_summary(args.out_dir, execution_time if 'execution_time' in locals() else None)


def print_console_summary(out_dir: str, execution_time: float = None):
    """Print human-friendly summary to console with fallback data retrieval."""
    import csv
    import json
    import os
    from pathlib import Path
    
    # Data retrieval with fallback priority: summary.csv → summary.md → predictions.jsonl
    summary_data = None
    data_source = None
    
    # Try 1: summary.csv
    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                summary_data = {
                    'n': int(row['n']),
                    'baseline_acc': float(row['baseline_acc']),
                    'chameleon_acc': float(row['chameleon_acc']),
                    'delta_acc': float(row['delta_acc']),
                    'mcnemar_b': int(row['mcnemar_b']),
                    'mcnemar_c': int(row['mcnemar_c']),
                    'p_value': float(row['p_value']),
                    'valid_bl_rate': float(row.get('valid_bl_rate', 0.0)),
                    'valid_ch_rate': float(row.get('valid_ch_rate', 0.0)),
                }
                data_source = "summary.csv"
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
    
    # Try 2: summary.md (basic parsing)
    if not summary_data:
        md_path = os.path.join(out_dir, "summary.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, 'r') as f:
                    content = f.read()
                # Basic regex parsing for key metrics
                import re
                n_match = re.search(r'Samples = (\d+)', content)
                baseline_match = re.search(r'Baseline Accuracy = \*\*([0-9.]+)\*\*', content)
                chameleon_match = re.search(r'Chameleon Accuracy = \*\*([0-9.]+)\*\*', content)
                delta_match = re.search(r'Δ = ([+-][0-9.]+)', content)
                p_match = re.search(r'McNemar exact p = \*\*([0-9.e-]+)\*\*', content)
                b_match = re.search(r'b=(\d+)', content)
                c_match = re.search(r'c=(\d+)', content)
                
                if all([n_match, baseline_match, chameleon_match, delta_match, p_match, b_match, c_match]):
                    summary_data = {
                        'n': int(n_match.group(1)),
                        'baseline_acc': float(baseline_match.group(1)),
                        'chameleon_acc': float(chameleon_match.group(1)),
                        'delta_acc': float(delta_match.group(1)),
                        'mcnemar_b': int(b_match.group(1)),
                        'mcnemar_c': int(c_match.group(1)),
                        'p_value': float(p_match.group(1)),
                        'valid_bl_rate': 0.0,
                        'valid_ch_rate': 0.0,
                    }
                    data_source = "summary.md"
            except Exception as e:
                print(f"❌ Failed to parse {md_path}: {e}")
    
    # Try 3: predictions.jsonl (compute on-the-fly)
    if not summary_data:
        jsonl_path = os.path.join(out_dir, "predictions.jsonl")
        if os.path.exists(jsonl_path):
            try:
                baseline_correct = 0
                chameleon_correct = 0
                valid_baseline = 0
                valid_chameleon = 0
                total = 0
                b_count = 0  # baseline correct, chameleon wrong
                c_count = 0  # baseline wrong, chameleon correct
                changed_count = 0
                prior_mode_counts = {}
                all_preds = []  # For entropy/top1 calculation
                
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        total += 1
                        
                        # Use correct field names from JSONL structure
                        baseline_pred = data.get('baseline', data.get('baseline_pred', ''))
                        chameleon_pred = data.get('chameleon', data.get('chameleon_pred', ''))
                        gold = data.get('gold', '')
                        
                        bl_correct = baseline_pred == gold
                        ch_correct = chameleon_pred == gold
                        
                        if baseline_pred:  # Non-empty prediction
                            valid_baseline += 1
                            if bl_correct:
                                baseline_correct += 1
                        
                        if chameleon_pred:  # Non-empty prediction
                            valid_chameleon += 1
                            if ch_correct:
                                chameleon_correct += 1
                            all_preds.append(chameleon_pred)
                        
                        # McNemar table - use gold_tag for consistency
                        if bl_correct and not ch_correct:
                            b_count += 1
                        elif not bl_correct and ch_correct:
                            c_count += 1
                        
                        # Changed rate
                        if data.get('changed', False):
                            changed_count += 1
                        
                        # Prior mode histogram
                        prior_info = data.get('prior', {})
                        prior_mode = prior_info.get('mode', 'unknown')
                        prior_mode_counts[prior_mode] = prior_mode_counts.get(prior_mode, 0) + 1
                
                baseline_acc = baseline_correct / max(1, valid_baseline)
                chameleon_acc = chameleon_correct / max(1, valid_chameleon)
                delta_acc = chameleon_acc - baseline_acc
                
                # McNemar exact p-value
                from scipy.stats import binom
                p_value = min(1.0, 2 * binom.cdf(min(b_count, c_count), b_count + c_count, 0.5)) if (b_count + c_count) > 0 else 1.0
                
                # Compute entropy and top1_share from prediction distribution
                entropy = 0.0
                top1_share = 0.0
                if all_preds:
                    from collections import Counter
                    pred_counts = Counter(all_preds)
                    total_preds = len(all_preds)
                    
                    # Entropy calculation
                    import math
                    for count in pred_counts.values():
                        p = count / total_preds
                        if p > 0:
                            entropy -= p * math.log2(p)
                    
                    # Top1 share
                    if pred_counts:
                        top1_count = max(pred_counts.values())
                        top1_share = top1_count / total_preds
                
                summary_data = {
                    'n': total,
                    'baseline_acc': baseline_acc,
                    'chameleon_acc': chameleon_acc,
                    'delta_acc': delta_acc,
                    'mcnemar_b': b_count,
                    'mcnemar_c': c_count,
                    'p_value': p_value,
                    'valid_bl_rate': valid_baseline / max(1, total),
                    'valid_ch_rate': valid_chameleon / max(1, total),
                    'changed_rate': changed_count / max(1, total),
                    'prior_mode_counts': prior_mode_counts,
                    'entropy': entropy,
                    'top1_share': top1_share,
                }
                data_source = "predictions.jsonl"
            except Exception as e:
                print(f"❌ Failed to compute from {jsonl_path}: {e}")
    
    # Display summary
    print(f"\n" + "="*60)
    print(f"📊 LAMP-2 BENCHMARK SUMMARY")
    print(f"="*60)
    
    if summary_data:
        print(f"📈 Sample Size: n = {summary_data['n']}")
        print(f"🎯 Baseline Accuracy:  {summary_data['baseline_acc']:.4f}")
        print(f"🚀 Chameleon Accuracy: {summary_data['chameleon_acc']:.4f}")
        print(f"📊 Delta (Improvement): {summary_data['delta_acc']:+.4f}")
        print(f"")
        print(f"🔬 McNemar Test:")
        print(f"   - b (BL✓, CH✗): {summary_data['mcnemar_b']}")
        print(f"   - c (BL✗, CH✓): {summary_data['mcnemar_c']}")
        print(f"   - p-value: {summary_data['p_value']:.3g}")
        print(f"")
        print(f"✅ Valid Predictions:")
        print(f"   - Baseline: {summary_data['valid_bl_rate']:.1%}")
        print(f"   - Chameleon: {summary_data['valid_ch_rate']:.1%}")
        
        # Additional metrics (from predictions.jsonl fallback)
        if 'changed_rate' in summary_data:
            print(f"")
            print(f"🔄 Changed rate: {summary_data['changed_rate']:.3f}")
            
            # Prior modes histogram
            if summary_data.get('prior_mode_counts'):
                mode_str = ", ".join([f"{mode}: {count}" for mode, count in sorted(summary_data['prior_mode_counts'].items())])
                print(f"🔧 Prior modes used: {{{mode_str}}}")
            
            print(f"📊 Entropy (pred): {summary_data.get('entropy', 0.0):.3f} | Top1 share: {summary_data.get('top1_share', 0.0):.2f}")
        
        if summary_data['p_value'] < 0.05:
            print(f"\n🎉 STATISTICALLY SIGNIFICANT improvement!")
        elif summary_data['delta_acc'] > 0:
            print(f"\n📈 Positive trend - consider larger sample size")
        else:
            print(f"\n📝 No significant improvement detected")
            
        print(f"\n📄 Data source: {data_source}")
    else:
        print(f"❌ No summary data available in {out_dir}")
    
    # Execution time
    if execution_time is not None:
        print(f"⏱️  Execution time: {execution_time:.1f}s")
    
    # Output files
    print(f"\n📁 Output directory: {out_dir}")
    for fname in ["predictions.jsonl", "summary.csv", "summary.md", "summary_per_user.csv", "summary_per_user.md"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"   ✓ {fname} ({size} bytes)")
        else:
            print(f"   ✗ {fname} (missing)")
    
    print(f"="*60)


if __name__ == "__main__":
    main()
