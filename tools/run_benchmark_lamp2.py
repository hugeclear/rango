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

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from chameleon_evaluator import ChameleonEvaluator
from chameleon_prime_personalization.utils.tags import DEFAULT_ALLOWED_TAGS, build_id_maps, match_allowed
from chameleon_prime_personalization.utils.labels_from_dataset import resolve_label_set
from chameleon_prime_personalization.utils.strict_classifier import build_id_prompt, parse_id_output
from chameleon_prime_personalization.utils.score_classifier import (
    classify_by_scores, classify_by_scores_with_calibration, compute_prior_scores
)
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
        
    def load_dataset(self, task: str = "lamp2", split: str = "test", limit: int = None) -> List[Tuple[int, Dict[str, Any]]]:
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
                     mode: str = "id") -> Dict[str, Any]:
        """Run complete benchmark evaluation."""
        
        # Setup reproducibility
        set_reproducible_seeds(seed)
        apply_reproducible_env()
        
        # Load dataset with limit
        samples = self.load_dataset(split=split, limit=limit if limit > 0 else None)

        # ラベル集合をデータから解決（fallback付き）
        labels = resolve_label_set(self.data_path, split, DEFAULT_ALLOWED_TAGS)
        id2tag, tag2id = build_id_maps(labels)
        use_calibration = getattr(self, "use_calibration", True)
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
            prior_prompt = build_id_prompt("", id2tag=id2tag) if use_calibration else None
            # 重要：prior は baseline（編集なし）で固定して計算し、両者で使い回す
            prior_scores_base = None
            if use_calibration and self.evaluator:
                editor = self.evaluator.chameleon_editor
                tokenizer = editor.tokenizer
                prior_scores_base = compute_prior_scores(editor.model, tokenizer, prior_prompt, id2tag, device=editor.device)
            if n <= 3:  # Debug first few samples
                print(f"[Sample {n}] Content: {content[:200]}...")
                print(f"[Sample {n}] Prompt: {prompt[:300]}...")

            # Normalize gold
            gold = match_allowed(sample.get("gold", ""), allowed=labels) or "unknown"
            
            # --- Baseline（編集なし）: スコアリング分類 ---
            print(f"[Sample {n}] Processing baseline scoring...")
            if self.evaluator:
                editor = self.evaluator.chameleon_editor
                tokenizer = editor.tokenizer
                try:
                    print(f"[Sample {n}] Using actual model for scoring")
                    if use_calibration:
                        bid, btag = classify_by_scores_with_calibration(
                            editor.model, tokenizer, prompt, id2tag,
                            prior_prompt=None,  # ここは使わず
                            device=editor.device, lam=1.0,
                            prior_scores=prior_scores_base  # 事前に baseline で計算した prior を使う
                        )
                    else:
                        bid, btag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                    print(f"[Sample {n}] Baseline result: ID={bid}, tag='{btag}'")
                except Exception as e:
                    print(f"[Sample {n}] Baseline scoring failed: {e}")
                    bid, btag = 1, list(id2tag.values())[0]
            else:
                # Mock for testing
                print(f"[Sample {n}] Using mock baseline (no evaluator)")
                bid, btag = 1, list(id2tag.values())[0]
                
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
                                                 gate=edit_gate_threshold):
                            if use_calibration:
                                cid, ctag = classify_by_scores_with_calibration(
                                    editor.model, tokenizer, prompt, id2tag,
                                    prior_prompt=None,  # chameleon 側では prior を再計算しない
                                    device=editor.device, lam=1.0,
                                    prior_scores=prior_scores_base  # baseline 由来の prior を差し引く
                                )
                            else:
                                cid, ctag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
                    except Exception as e:
                        print(f"[Sample {n}] Chameleon context scoring failed: {e}")
                        cid, ctag = 1, list(id2tag.values())[0]
                else:
                    # Fallback（編集無効）だが、ログで警告しておく
                    print("[warn] cham_context not available; chameleon==baseline in scoring path")
                    if use_calibration:
                        cid, ctag = classify_by_scores_with_calibration(
                            editor.model, tokenizer, prompt, id2tag,
                            prior_prompt=None,  # fallback でも prior を再計算しない
                            device=editor.device, lam=1.0,
                            prior_scores=prior_scores_base  # baseline 由来の prior を差し引く
                        )
                    else:
                        cid, ctag = classify_by_scores(editor.model, tokenizer, prompt, id2tag, device=editor.device)
            else:
                cid, ctag = 1, list(id2tag.values())[0]
                
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
            result = {
                "id": sample.get("id", sid),
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


def main():
    parser = argparse.ArgumentParser(description="LaMP-2 Benchmark: Baseline vs Chameleon")
    
    # Data and configuration
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=-1, help="Limit samples (-1 for all)")
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
    
    # Output
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") 
        run_name = f"lamp2_{args.split}_ap{args.alpha_personal}_ag{args.alpha_general}_tau{args.edit_gate_threshold}_{ts}"
        args.out_dir = os.path.join("results", "bench", run_name)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Run benchmark
    benchmarker = LaMP2Benchmarker(args.config_path, args.data_path)
    benchmarker.use_calibration = args.calibrate
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
    )
    
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
    
    # Print summary
    print(f"\n🎯 LaMP-2 Benchmark Complete!")
    print(f"📊 Results: n={summary['n']}, Δacc={summary['delta_acc']:+.4f}, p={summary['p_value']:.3g}")
    print(f"📁 Output files:")
    print(f"   - {jsonl_path}")
    print(f"   - {csv_path}")
    print(f"   - {md_path}")
    
    if summary['p_value'] < 0.05:
        print(f"🎉 SIGNIFICANT improvement detected!")
    elif summary['delta_acc'] > 0:
        print(f"📈 Positive trend - consider larger sample size")
    else:
        print(f"📝 No improvement - check parameters or implementation")


if __name__ == "__main__":
    main()
