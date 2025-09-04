#!/usr/bin/env python3
"""
Trace One Example - Detailed Chameleon Pipeline Analysis Tool

This script provides comprehensive tracing of the Chameleon personalization pipeline
for a single example, capturing all intermediate states, transformations, and statistics.
"""

import sys
import os
import json
import argparse
import time
import math
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, Tuple, List

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GenerationConfig
import yaml

# Import Chameleon components
from chameleon_evaluator import ChameleonEvaluator


def _avg_logprob_from_scores(outs):
    import torch.nn.functional as F
    scores = getattr(outs, "scores", None)
    seq = getattr(outs, "sequences", None)
    if scores is None or seq is None or len(scores) == 0:
        return None
    seq = seq[0]             # [L]
    gen_ids = seq[-len(scores):]
    logps = []
    for t, logits in enumerate(scores):
        lp = F.log_softmax(logits[0], dim=-1)  # [V]
        tok_id = int(gen_ids[t].item())
        logps.append(float(lp[tok_id].item()))
    return sum(logps) / max(len(logps), 1)


def _safe_generate_with_scores(editor, prompt: str, **gkwargs):
    gen_kwargs = dict(gkwargs)
    try:
        gen_kwargs.setdefault("return_dict_in_generate", True)
        gen_kwargs.setdefault("output_scores", True)
        outs = editor.model.generate(**editor.tokenizer(prompt, return_tensors="pt").to(editor.device),
                                     **gen_kwargs)
        meta = {"scores_available": True,
                "num_scores": len(getattr(outs, "scores", []) or []),
                "avg_logprob": _avg_logprob_from_scores(outs)}
        return outs, meta
    except TypeError:
        # fallback: no scores
        gen_kwargs.pop("return_dict_in_generate", None)
        gen_kwargs.pop("output_scores", None)
        outs = editor.model.generate(**editor.tokenizer(prompt, return_tensors="pt").to(editor.device),
                                     **gen_kwargs)
        return outs, {"scores_available": False, "num_scores": 0, "avg_logprob": None}


class ChameleonTracer:
    """Comprehensive tracing system for Chameleon pipeline analysis."""
    
    def __init__(self, config_path: str = "config.yaml", data_path: str = "./chameleon_prime_personalization/data"):
        self.config_path = config_path
        self.data_path = data_path
        self.evaluator = None
        self.results_dir = Path("results/trace")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        self._initialize_evaluator()
        
        # Environment info
        self.env_info = self._collect_environment_info()
        
    def _initialize_evaluator(self):
        """Initialize the Chameleon evaluator with proper configuration."""
        try:
            self.evaluator = ChameleonEvaluator(self.config_path, self.data_path)
            print(f"✅ ChameleonEvaluator initialized successfully")
            print(f"   Model: {self.evaluator.config['model']['name']}")
            print(f"   Device: {self.evaluator.chameleon_editor.device}")
        except Exception as e:
            print(f"❌ Failed to initialize ChameleonEvaluator: {e}")
            sys.exit(1)
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "repo_root": str(repo_root),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            env_info["device_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        # Git info
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], 
                                               cwd=repo_root, text=True).strip()
            env_info["git_commit"] = git_commit
        except:
            env_info["git_commit"] = "unknown"
        
        # Version info
        try:
            import transformers
            env_info["transformers_version"] = transformers.__version__
        except:
            env_info["transformers_version"] = "unknown"
            
        try:
            import accelerate
            env_info["accelerate_version"] = accelerate.__version__
        except:
            env_info["accelerate_version"] = "unknown"
            
        return env_info
    
    def _safe_generate_with_scores(self, prompt: str, **generate_kwargs):
        """
        Try to use return_dict_in_generate/output_scores; if not supported, fallback to plain generate.
        Returns: (outputs, meta) where meta may have scores/logprobs or N/A.
        """
        editor = self.evaluator.chameleon_editor
        return _safe_generate_with_scores(editor, prompt, **generate_kwargs)

    def select_sample(self, sample_id: Optional[int] = None) -> Dict[str, Any]:
        """Select and load a sample for analysis."""
        # Get sample data from evaluator
        try:
            # Use evaluator's data loading methods
            test_data = self.evaluator.data_loader.load_test_data()
            
            if sample_id is None:
                sample_id = 0
            
            if sample_id >= len(test_data):
                print(f"⚠️ Sample {sample_id} not found, using sample 0")
                sample_id = 0
                
            sample = test_data[sample_id]
            
            # Add some sample metadata
            sample_info = {
                "dataset": "LaMP-2",
                "split": "test", 
                "sample_id": sample_id,
                "original_sample_id": sample.get("id", sample_id),
                "total_samples": len(test_data),
                "gold_label": sample.get("reference", "N/A"),
                "user_id": sample.get("user_id", "N/A"),
                "question": sample.get("question", sample.get("input", "")),
                "profile_length": len(sample.get("profile", []))
            }
            
        except Exception as e:
            print(f"❌ Error loading sample: {e}")
            # Create a default sample for testing
            sample = {
                "id": 0,
                "question": "x Overwhelmed by her suffocating schedule, touring European princess Ann takes off for a night while in Rome. When a sedative she took from her doctor kicks in, however, she falls asleep on a park bench and is found by an American reporter, Joe Bradley, who takes her back to his apartment for safety. At work the next morning, Joe finds out Ann's regal identity and bets his editor he can get exclusive interview with her, but romance soon gets in the way.",
                "user_id": "0",
                "profile": [
                    {"tag": "psychology", "description": "A petty criminal fakes insanity to serve his sentence in a mental ward rather than prison. He soon finds himself as a leader to the other patients—and an enemy to the cruel, domineering nurse who runs the ward."},
                    {"tag": "psychology", "description": "David Aames has it all: wealth, good looks and gorgeous women on his arm. But just as he begins falling for the warmhearted Sofia, his face is horribly disfigured in a car accident. That's just the beginning of his troubles as the lines between illusion and reality, between life and death, are blurred."},
                    {"tag": "action", "description": "When a virus leaks from a top-secret facility, turning all resident researchers into ravenous zombies and their lab animals into mutated hounds from hell, the government sends in an elite military task force to contain the outbreak."},
                    {"tag": "action", "description": "Former Special Forces officer, Frank Martin will deliver anything to anyone for the right price, and his no-questions-asked policy puts him in high demand. But when he realizes his latest cargo is alive, it sets in motion a dangerous chain of events. The bound and gagged Lai is being smuggled to France by a shady American businessman, and Frank works to save her as his own illegal activities are uncovered by a French detective."},
                    {"tag": "comedy", "description": "Viktor Navorski is a man without a country; his plane took off just as a coup d'etat exploded in his homeland, leaving it in shambles, and now he's stranded at Kennedy Airport, where he's holding a passport that nobody recognizes. While quarantined in the transit lounge until authorities can figure out what to do with him, Viktor simply goes on living – and courts romance with a beautiful flight attendant."}
                ],
                "reference": "romance"
            }
            
            sample_info = {
                "dataset": "LaMP-2",
                "split": "test",
                "sample_id": 0,
                "original_sample_id": 0,
                "total_samples": 46,
                "gold_label": "N/A",
                "user_id": "N/A",
                "question": sample["question"],
                "profile_length": len(sample["profile"])
            }
        
        return {"sample": sample, "info": sample_info}
    
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build the complete prompt with strict classification constraints."""
        # LaMP-2: 1ラベル強制 & 余計な 'x ' を除去
        movie_desc = (sample.get("movie_description") or sample.get("text") or sample.get("question") or sample.get("input") or "").lstrip("x ").strip()
        user_profile = (sample.get("user_profile") or "").strip()
        
        # Extract profile information
        profile = sample.get("profile", [])
        tags_from_profile = []
        for item in profile:
            if isinstance(item, dict):
                tag = item.get("tag", "").lower().strip()
                if tag:
                    tags_from_profile.append(tag)
        
        fallback_tags = ["romance", "drama"]
        allowed_tags = sorted(set(tags_from_profile + fallback_tags)) or ["action", "comedy", "drama", "romance"]
        
        # Build user context
        context_parts = []
        if profile:
            context_parts.append("User's movie preferences:")
            for item in profile[:5]:  # Limit to 5 items for clarity
                tag = item.get("tag", "").lower()
                desc = item.get("description", "")
                if tag and desc:
                    context_parts.append(f"- {tag}: {desc}")
        
        context_str = "\n".join(context_parts) if context_parts else "User's movie preferences:\nN/A"
        prompt = f"""You are a classifier.
Given the user's movie preferences, classify the following movie description with a single tag.

Allowed tags (choose EXACTLY one): {", ".join(allowed_tags)}
Output format: just one lowercase tag from the list above. No explanations, no punctuation.

{context_str}

Movie: {movie_desc}

Tag:"""
        
        # Store prompt in sample for direction vector computation
        sample["prompt"] = prompt
        return prompt
    
    def tokenize_input(self, prompt: str, norm_scale: float = 0.9) -> Dict[str, Any]:
        """Tokenize input and gather tokenization statistics."""
        tokenizer = self.evaluator.chameleon_editor.tokenizer
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        
        # Get token strings
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Identify special tokens
        special_tokens = []
        for i, (token_id, token_str) in enumerate(zip(input_ids[0], tokens)):
            if token_id.item() in tokenizer.all_special_ids:
                special_tokens.append({"position": i, "token": token_str, "id": token_id.item()})
        
        tokenization_info = {
            "input_length": input_ids.shape[1],
            "tokens_sample": [(tokens[i], input_ids[0][i].item()) for i in range(min(10, len(tokens)))],
            "tokens_sample_end": [(tokens[i], input_ids[0][i].item()) for i in range(max(0, len(tokens)-5), len(tokens))],
            "special_tokens": special_tokens,
            "norm_scale_applied": norm_scale != 1.0,
            "norm_scale_value": norm_scale
        }
        
        return {
            "inputs": inputs,
            "tokenization_info": tokenization_info
        }
    
    def generate_baseline(self, prompt: str, generation_args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate baseline output without Chameleon editing."""
        try:
            # Use Chameleon API with zero editing (alpha_personal=0, alpha_neutral=0)
            start_time = time.time()
            
            generated_text = self.evaluator.chameleon_editor.generate_with_chameleon(
                prompt=prompt,
                alpha_personal=0.0,
                alpha_neutral=0.0, 
                alpha_fakeit=0.0,
                target_layers=[],  # No editing layers
                gen_kwargs={
                    "max_new_tokens": max(2, generation_args.get("max_new_tokens", 6)),
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "eos_token_id": self.evaluator.chameleon_editor.tokenizer.eos_token_id
                }
            )
            
            generation_time = time.time() - start_time
            
            baseline_info = {
                "generation_time": generation_time,
                "full_output": prompt + generated_text,
                "generated_text": generated_text,
                "generated_length": len(generated_text.split()) if generated_text else 0,
                "generation_config": generation_args
            }
            
            # Try to get scores safely
            try:
                outs, score_meta = self._safe_generate_with_scores(
                    prompt, 
                    max_new_tokens=max(2, generation_args.get("max_new_tokens", 6)),
                    do_sample=False, 
                    repetition_penalty=1.0,
                    eos_token_id=self.evaluator.chameleon_editor.tokenizer.eos_token_id
                )
                # Recompute avg_logprob using editor's updated utility method
                try:
                    # Get input_ids for the prompt
                    tokenizer = self.evaluator.chameleon_editor.tokenizer
                    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
                    score_meta["avg_logprob"] = self.evaluator.chameleon_editor._avg_logprob_from_scores(
                        input_ids, outs, tokenizer
                    )
                except Exception as e:
                    score_meta["avg_logprob_error"] = str(e)
                baseline_info.update(score_meta)
            except Exception as e:
                baseline_info["scoring_error"] = str(e)
        
            return baseline_info
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_editing_vectors(self, sample: Dict[str, Any], alpha_personal: float, alpha_general: float, target_layers: List[str], norm_scale: float, edit_gate_threshold: float) -> Dict[str, Any]:
        """Analyze the editing vectors and gate decision using robust methods."""
        # 方向ベクトルと gate の記録
        try:
            dv = self.evaluator.compute_direction_vectors(sample, target_layers=target_layers, norm_scale=norm_scale)
            trace = {
                "direction_vectors": {
                    "l2_personal": dv["l2_personal"], 
                    "l2_general": dv["l2_general"], 
                    "cos_theta": dv["cos_theta"]
                }
            }
            # hidden_norm を映画側ベクトルの L2 で近似（安全かつ一貫）
            hidden_norm = max(1e-8, dv["l2_general"])
            gate_info = self.evaluator.summarize_gate(hidden_norm, dv, alpha_personal, alpha_general, edit_gate_threshold)
            trace["gate"] = gate_info
            return trace
        except Exception as e:
            return {"direction_vectors_error": str(e), "gate": {"gate_value": 0.0, "applied": False, "threshold": edit_gate_threshold}}
    
    def generate_personalized(self, sample: Dict[str, Any], prompt: str, generation_args: Dict[str, Any], 
                            alpha_personal: float, alpha_general: float,
                            norm_scale: float, edit_gate_threshold: float,
                            target_layers: List[str] | None = None) -> Dict[str, Any]:
        """Generate personalized output with Chameleon editing."""
        try:
            # Use evaluator's generate_with_chameleon method with fixed kwargs
            start_time = time.time()
            
            # Generate with Chameleon using gen_kwargs parameter
            generated_text = self.evaluator.chameleon_editor.generate_with_chameleon(
                prompt=prompt,
                alpha_personal=alpha_personal,
                alpha_neutral=alpha_general, 
                alpha_fakeit=0.0,
                target_layers=(target_layers or ['model.layers.20', 'model.layers.27']),
                gen_kwargs={
                    "max_new_tokens": max(2, generation_args.get("max_new_tokens", 6)),
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "eos_token_id": self.evaluator.chameleon_editor.tokenizer.eos_token_id
                }
            )
            
            generation_time = time.time() - start_time
            
            personalized_info = {
                "generation_time": generation_time,
                "generated_text": generated_text,
                "generated_length": len(generated_text.split()) if generated_text else 0,
                "parameters_used": {
                    "alpha_personal": alpha_personal,
                    "alpha_general": alpha_general
                }
            }
            
            # Try to get scores with injection active
            try:
                editor = self.evaluator.chameleon_editor
                # Prepare dv/gate/delta
                tgt_layers = (target_layers or ['model.layers.20', 'model.layers.27'])
                dv = self.evaluator.compute_direction_vectors(sample, target_layers=tgt_layers, norm_scale=norm_scale)
                hidden_norm = max(1e-8, float(dv.get("l2_general", 0.0)))
                gate = self.evaluator.summarize_gate(hidden_norm, dv, alpha_personal, alpha_general, edit_gate_threshold)
                delta_vec = editor._edit_delta(dv, alpha_personal, alpha_general)
                # Inputs and kwargs for scored generation
                tok = editor.tokenizer(prompt, return_tensors="pt").to(editor.device)
                gk = {
                    "max_new_tokens": max(2, generation_args.get("max_new_tokens", 6)),
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "eos_token_id": editor.tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                with editor._layer_injection_ctx(tgt_layers, delta_vec, bool(gate.get("applied", False)), tok["input_ids"].size(1)):
                    outs = editor.model.generate(**tok, **gk)
                # Use updated avg_logprob method signature
                personalized_info["avg_logprob"] = editor._avg_logprob_from_scores(
                    tok["input_ids"], outs, editor.tokenizer
                )
                personalized_info["num_scores"] = len(getattr(outs, "scores", []) or [])
            except Exception as e:
                personalized_info["avg_logprob"] = None
                personalized_info["scoring_error"] = str(e)

            return personalized_info
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_outputs(self, sample: Dict[str, Any], baseline_output: str, personalized_output: str) -> Dict[str, Any]:
        """Evaluate and compare the outputs with enhanced tag extraction."""
        gold_label = sample.get("reference", "").lower().strip()
        
        # Define allowed tags for validation
        allowed_tags = [
            "action", "adventure", "animation", "biography", "comedy", "crime", 
            "documentary", "drama", "family", "fantasy", "history", "horror", 
            "music", "mystery", "romance", "sci-fi", "sport", "thriller", "war", "western"
        ]
        
        def extract_tag(text: str) -> str:
            """Extract the most likely tag from generated text."""
            if not text:
                return "unknown"
            
            # Clean the text
            text = text.lower().strip()
            
            # Multi-strategy extraction
            strategies = [
                text.split()[0] if text.split() else "",  # First word
                text.split('\n')[0].strip() if '\n' in text else text,  # First line
                text.replace("tag:", "").replace("output:", "").strip()  # Remove prefixes
            ]
            
            # Try each strategy and check against allowed tags
            for candidate in strategies:
                candidate = candidate.strip('.,!?"\'-()[]{}').lower()
                if candidate in allowed_tags:
                    return candidate
            
            # If no match found, return the first word or "unknown"
            first_word = text.split()[0] if text.split() else "unknown"
            return first_word.strip('.,!?"\'-()[]{}').lower()
        
        # Extract predictions
        baseline_prediction = extract_tag(baseline_output)
        personalized_prediction = extract_tag(personalized_output)
        
        # Validation checks
        baseline_valid_tag = baseline_prediction in allowed_tags
        personalized_valid_tag = personalized_prediction in allowed_tags
        
        # Correctness (if we have gold label)
        baseline_correct = baseline_prediction == gold_label if gold_label else False
        personalized_correct = personalized_prediction == gold_label if gold_label else False
        
        # Word-level comparison
        baseline_words = baseline_output.split()
        personalized_words = personalized_output.split()
        
        word_differences = []
        max_len = max(len(baseline_words), len(personalized_words))
        
        for i in range(max_len):
            baseline_word = baseline_words[i] if i < len(baseline_words) else "<MISSING>"
            personalized_word = personalized_words[i] if i < len(personalized_words) else "<MISSING>"
            
            if baseline_word != personalized_word:
                word_differences.append({
                    "position": i,
                    "baseline": baseline_word,
                    "personalized": personalized_word
                })
        
        return {
            "gold_label": gold_label,
            "baseline_prediction": baseline_prediction,
            "personalized_prediction": personalized_prediction,
            "baseline_correct": baseline_correct,
            "personalized_correct": personalized_correct,
            "baseline_valid_tag": baseline_valid_tag,
            "personalized_valid_tag": personalized_valid_tag,
            "accuracy_improvement": personalized_correct and not baseline_correct,
            "accuracy_degradation": baseline_correct and not personalized_correct,
            "valid_improvement": personalized_valid_tag and not baseline_valid_tag,
            "word_differences": word_differences,
            "output_lengths": {
                "baseline": len(baseline_output),
                "personalized": len(personalized_output)
            },
            "allowed_tags": allowed_tags,
            "raw_outputs": {
                "baseline": baseline_output,
                "personalized": personalized_output
            }
        }
    
    def run_sensitivity_analysis(self, sample: Dict[str, Any], prompt: str, generation_args: Dict[str, Any],
                                base_alpha_p: float, base_alpha_g: float, base_norm_scale: float, 
                                base_gate: float) -> Dict[str, Any]:
        """Run mini ablation study for sensitivity analysis."""
        sensitivity_results = {
            "base_parameters": {
                "alpha_personal": base_alpha_p,
                "alpha_general": base_alpha_g,
                "norm_scale": base_norm_scale,
                "gate_threshold": base_gate
            },
            "gate_variations": {},
            "norm_scale_variations": {}
        }
        
        # Test gate variations
        for gate_val in [0.018, 0.026, 0.030]:
            try:
                # Simplified: just check if gate would be applied differently
                target_layers = ['model.layers.20', 'model.layers.27']
                editing_info = self.analyze_editing_vectors(sample, base_alpha_p, base_alpha_g, target_layers, base_norm_scale, gate_val)
                gate_value = editing_info.get("gate", {}).get("gate_value", 0.0)
                applied = gate_value > gate_val
                
                sensitivity_results["gate_variations"][str(gate_val)] = {
                    "gate_applied": applied,
                    "gate_value": gate_value
                }
            except Exception as e:
                sensitivity_results["gate_variations"][str(gate_val)] = {"error": str(e)}
        
        # Test norm_scale variations  
        for norm_val in [0.85, 0.90, 1.00]:
            try:
                # Generate with different norm scale (simplified)
                result = self.generate_personalized(sample, prompt, generation_args, base_alpha_p, base_alpha_g, 
                                                   norm_val, base_gate, ['model.layers.20', 'model.layers.27'])
                sensitivity_results["norm_scale_variations"][str(norm_val)] = {
                    "output_length": len(result.get("generated_text", "")),
                    "generation_time": result.get("generation_time", 0.0)
                }
            except Exception as e:
                sensitivity_results["norm_scale_variations"][str(norm_val)] = {"error": str(e)}
        
        return sensitivity_results
    
    def trace_single_example(self, sample_id: Optional[int] = None, 
                           alpha_personal: float = 2.75, alpha_general: float = -1.0,
                           norm_scale: float = 0.9, edit_gate_threshold: float = 0.022,
                           temperature: float = 0.0, top_p: float = 1.0, 
                           repetition_penalty: float = 1.0, do_sample: bool = False,
                           seed: int = 42, return_scores: bool = True,
                           max_new_tokens: int = 10) -> Dict[str, Any]:
        """Run complete tracing analysis for a single example."""
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"🔍 Starting trace analysis for sample {sample_id}")
        
        # 1. Sample Selection
        sample_data = self.select_sample(sample_id)
        sample = sample_data["sample"]
        sample_info = sample_data["info"]
        
        print(f"   Sample: {sample_info['sample_id']} (User: {sample_info['user_id']})")
        print(f"   Gold: {sample_info['gold_label']}")
        
        # 2. Prompt Building
        prompt = self.build_prompt(sample)
        
        # 3. Tokenization
        tokenization_data = self.tokenize_input(prompt, norm_scale)
        inputs = tokenization_data["inputs"]
        tokenization_info = tokenization_data["tokenization_info"]
        
        print(f"   Tokenized: {tokenization_info['input_length']} tokens")
        
        # 4. Generation arguments
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "return_scores": return_scores
        }
        
        # 5. Baseline Generation
        print("   Generating baseline...")
        baseline_info = self.generate_baseline(prompt, generation_args)
        
        # 6. Editing Vector Analysis
        print("   Analyzing editing vectors...")
        target_layers = ['model.layers.20', 'model.layers.27']
        editing_info = self.analyze_editing_vectors(sample, alpha_personal, alpha_general, target_layers, norm_scale, edit_gate_threshold)
        
        # 7. Personalized Generation
        print("   Generating personalized...")
        personalized_info = self.generate_personalized(sample, prompt, generation_args, 
                                                       alpha_personal, alpha_general,
                                                       norm_scale, edit_gate_threshold,
                                                       target_layers)
        
        # 8. Evaluation
        baseline_text = baseline_info.get("generated_text", "")
        personalized_text = personalized_info.get("generated_text", "")
        evaluation_info = self.evaluate_outputs(sample, baseline_text, personalized_text)
        
        # 8.5. Add edit information to editing_info
        if "gate" in editing_info:
            editing_info["edit"] = {
                "layers": target_layers,
                "alpha_personal": alpha_personal,
                "alpha_general": alpha_general,
                "applied": editing_info["gate"].get("applied", False)
            }
        
        print(f"   Baseline: '{baseline_text}' ({'✅' if evaluation_info['baseline_correct'] else '❌'})")
        print(f"   Personalized: '{personalized_text}' ({'✅' if evaluation_info['personalized_correct'] else '❌'})")
        
        # 9. Sensitivity Analysis
        print("   Running sensitivity analysis...")
        sensitivity_info = self.run_sensitivity_analysis(
            sample, prompt, generation_args, alpha_personal, alpha_general, 
            norm_scale, edit_gate_threshold
        )
        
        # 10. Compile Complete Results
        complete_results = {
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "sample_id": sample_info["sample_id"],
                "original_sample_id": sample_info.get("original_sample_id", sample_id),
                "parameters": {
                    "alpha_personal": alpha_personal,
                    "alpha_general": alpha_general,
                    "norm_scale": norm_scale,
                    "edit_gate_threshold": edit_gate_threshold,
                    "seed": seed
                },
                "generation_config": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "return_scores": return_scores
                },
                "model_dtype": self.evaluator.get_effective_model_dtype()
            },
            "environment_info": self.env_info,
            "sample_info": sample_info,
            "prompt": prompt,
            "tokenization": tokenization_info,
            "baseline_generation": baseline_info,
            "editing_analysis": editing_info,
            "personalized_generation": personalized_info,
            "evaluation": evaluation_info,
            "sensitivity_analysis": sensitivity_info
        }
        
        # Add delta_avg_logprob for editing contribution analysis
        try:
            baseline_logprob = complete_results.get("baseline_generation", {}).get("avg_logprob")
            personalized_logprob = complete_results.get("personalized_generation", {}).get("avg_logprob")
            
            if baseline_logprob is not None and personalized_logprob is not None:
                complete_results["delta_avg_logprob"] = personalized_logprob - baseline_logprob
                complete_results["editing_contribution"] = {
                    "avg_logprob_change": complete_results["delta_avg_logprob"],
                    "direction": "positive" if complete_results["delta_avg_logprob"] > 0 else "negative",
                    "magnitude": abs(complete_results["delta_avg_logprob"])
                }
            else:
                complete_results["delta_avg_logprob"] = None
                complete_results["editing_contribution"] = {
                    "avg_logprob_change": None,
                    "reason": f"Missing logprobs: baseline={baseline_logprob is not None}, personalized={personalized_logprob is not None}"
                }
        except Exception as e:
            complete_results["delta_avg_logprob"] = None
            complete_results["delta_avg_logprob_error"] = str(e)
        
        return complete_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        # Main results JSON
        results_file = self.results_dir / "trace_one_example.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Individual text files for easy inspection
        baseline_text = results.get("baseline_generation", {}).get("generated_text", "")
        personalized_text = results.get("personalized_generation", {}).get("generated_text", "")
        
        if baseline_text:
            with open(self.results_dir / "baseline.txt", 'w', encoding='utf-8') as f:
                f.write(baseline_text)
        
        if personalized_text:
            with open(self.results_dir / "personalized.txt", 'w', encoding='utf-8') as f:
                f.write(personalized_text)
        
        print(f"✅ Results saved to:")
        print(f"   📄 {results_file}")
        print(f"   📄 {self.results_dir / 'baseline.txt'}")
        print(f"   📄 {self.results_dir / 'personalized.txt'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Trace Chameleon Pipeline for Single Example")
    
    # Sample selection
    parser.add_argument("--sample_id", type=int, default=None,
                      help="Sample ID to analyze (default: 0)")
    
    # Chameleon parameters
    parser.add_argument("--alpha_personal", type=float, default=2.75,
                      help="Personal direction scaling factor")
    parser.add_argument("--alpha_general", type=float, default=-1.0,
                      help="General direction scaling factor")
    parser.add_argument("--norm_scale", type=float, default=0.9,
                      help="Normalization scale factor")
    parser.add_argument("--edit_gate_threshold", type=float, default=0.022,
                      help="Edit gate threshold")
    
    # Generation parameters (optimized for classification)
    parser.add_argument("--temperature", type=float, default=0.0,
                      help="Generation temperature (0.0 for greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0,
                      help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                      help="Repetition penalty")
    parser.add_argument("--do_sample", type=lambda x: x.lower() == 'true', default=False,
                      help="Whether to use sampling (False for greedy decoding)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--return_scores", type=lambda x: x.lower() == 'true', default=True,
                      help="Return generation scores")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                      help="Maximum new tokens to generate")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Configuration file path")
    parser.add_argument("--data_path", type=str, default="./chameleon_prime_personalization/data",
                      help="Data directory path")
    
    args = parser.parse_args()
    
    try:
        # Initialize tracer
        tracer = ChameleonTracer(args.config, args.data_path)
        
        # Run trace analysis
        results = tracer.trace_single_example(
            sample_id=args.sample_id,
            alpha_personal=args.alpha_personal,
            alpha_general=args.alpha_general,
            norm_scale=args.norm_scale,
            edit_gate_threshold=args.edit_gate_threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            seed=args.seed,
            return_scores=args.return_scores,
            max_new_tokens=args.max_new_tokens
        )
        
        # Save results
        tracer.save_results(results)
        
        print("✅ Trace analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Trace analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
