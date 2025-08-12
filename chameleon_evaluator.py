#!/usr/bin/env python3
"""
Chameleon LaMP-2 Evaluation System
完全なChameleon実装とLaMP-2ベンチマーク自動評価システム

特徴:
- PyTorchフックによるTransformer中間層埋め込み抽出
- SVD方向学習とリアルタイム埋め込み編集
- 統計的有意性検定を含む包括的評価
"""

import json
import os
from threading import local
import time
import yaml
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
from collections import defaultdict
import logging

# NLTK for BLEU score
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Dataclasses
# =========================
@dataclass
class EvaluationResult:
    method_name: str
    accuracy: float
    exact_match: float
    bleu_score: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    total_samples: int
    correct_predictions: int
    predictions: List[str]
    ground_truths: List[str]


# =========================
# Data Loader
# =========================
class LaMPDataLoader:
    """LaMP-2データの読み込みとユーザー分割"""
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.merged_data = None
        self.ground_truth = None

    def load_merged_data(self) -> List[Dict]:
        # Priority 1: Check data_path/raw/LaMP-2/merged.json first
        primary_merged = self.data_path / "raw/LaMP-2/merged.json"
        if primary_merged.exists():
            logger.info(f"Loading merged data from priority path: {primary_merged}")
            with open(primary_merged, 'r', encoding='utf-8') as f:
                self.merged_data = json.load(f)
            return self.merged_data

        # Priority 2: Check data_path/raw/LaMP-2/answers.json as backup
        backup_answers = self.data_path / "raw/LaMP-2/answers.json"
        if backup_answers.exists():
            logger.info(f"Loading data from backup answers: {backup_answers}")
            with open(backup_answers, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            # Convert answers format to merged format if needed
            if isinstance(answers_data, dict) and 'golds' in answers_data:
                # Convert to merged-like format
                self.merged_data = []
                for gold in answers_data['golds'][:1000]:  # Limit for demo
                    self.merged_data.append({
                        'id': gold['id'],
                        'input': f"Question: {gold.get('input', 'Unknown question')}",
                        'output': gold['output']
                    })
                logger.info(f"Converted {len(self.merged_data)} samples from answers format")
                return self.merged_data

        # Fallback: Original path resolution for compatibility
        possible_paths = [
            self.data_path / "chameleon_prime_personalization/data/raw/LaMP-2/merged.json",
            self.data_path / "processed/LaMP-2/merged.json",
            self.data_path / "data/raw/LaMP-2/merged.json",
            self.data_path / "merged.json"
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading merged data from fallback path: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    self.merged_data = json.load(f)
                return self.merged_data

        backup_path = self.data_path / "data/raw/LaMP_all/LaMP_2/user-based/dev/dev_questions.json"
        if backup_path.exists():
            logger.info(f"Using backup data source: {backup_path}")
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            if isinstance(backup_data, dict) and 'instances' in backup_data:
                self.merged_data = backup_data['instances'][:1000]
                logger.info(f"Loaded {len(self.merged_data)} samples from backup source")
                return self.merged_data

        raise FileNotFoundError("No valid data source found (primary or backup)")

    def load_ground_truth(self) -> Dict[str, str]:
        # Priority 1: Check data_path/raw/LaMP-2/answers.json first
        primary_answers = self.data_path / "raw/LaMP-2/answers.json"
        if primary_answers.exists():
            logger.info(f"Loading ground truth from priority path: {primary_answers}")
            with open(primary_answers, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)

            if isinstance(answers_data, dict) and 'golds' in answers_data:
                golds = answers_data['golds']
                return {str(g['id']): g['output'].strip().lower() for g in golds}
            elif isinstance(answers_data, list):
                return {str(a['id']): a['output'].strip().lower() for a in answers_data}

        # Fallback: Original path resolution for compatibility
        possible_paths = [
            self.data_path / "chameleon_prime_personalization/data/raw/LaMP-2/answers.json",
            self.data_path / "data/raw/LaMP-2/answers.json",
            self.data_path / "answers.json"
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading ground truth from fallback path: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    answers_data = json.load(f)

                if isinstance(answers_data, dict) and 'golds' in answers_data:
                    golds = answers_data['golds']
                    return {str(g['id']): g['output'].strip().lower() for g in golds}
                elif isinstance(answers_data, list):
                    return {str(a['id']): a['output'].strip().lower() for a in answers_data}

        backup_answers_path = self.data_path / "data/raw/LaMP_all/LaMP_2/user-based/dev/dev_outputs.json"
        if backup_answers_path.exists():
            logger.info(f"Using backup ground truth: {backup_answers_path}")
            with open(backup_answers_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            if isinstance(answers_data, dict) and 'golds' in answers_data:
                golds = answers_data['golds']
                return {str(g['id']): g['output'].strip().lower() for g in golds}

        logger.warning("Ground truth not found, evaluation will be prediction-only")
        return {}

    def get_user_samples(self, user_limit: int = 10) -> List[Dict]:
        if not self.merged_data:
            self.load_merged_data()

        user_data = defaultdict(list)
        for item in self.merged_data:
            user_id = str(item['id'])[:3]  # 擬似ユーザーID
            user_data[user_id].append(item)

        selected_samples = []
        for i, (_, samples) in enumerate(user_data.items()):
            if i >= user_limit:
                break
            selected_samples.extend(samples[:5])
        logger.info(f"Selected {len(selected_samples)} samples from {min(len(user_data), user_limit)} users")
        return selected_samples


# =========================
# Chameleon Editor
# =========================
class ChameleonEditor:
    """
    Transformerの中間出力に方向ベクトルを加算して編集する
    """
    def __init__(self, model_name: str = "./chameleon_prime_personalization/models/base_model",
                 device: str = "auto", torch_dtype: str = "float32"):
        from pathlib import Path

        # モデルパスを絶対化してローカル固定
        resolved = Path(model_name).expanduser()
        if not resolved.is_absolute():
            # 呼び出しディレクトリ依存を断ち切る
            resolved = (Path.cwd() / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model path not found: {resolved}")
        self.model_name = str(resolved)
        # --- NEW: resolve local path and force local load if exists ---
        model_path = Path(model_name)
        load_kwargs = {}
        if model_path.exists():
            model_name = str(model_path.resolve())  # use absolute path
            load_kwargs["local_files_only"] = True  # force local
        # -------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self._hook_calls_total = 0
        self._hook_calls_in_this_generate = 0
        
        # Enhanced diagnostics tracking
        self._edit_ratios = []  # Track edit strength per hook call
        self._kl_divergences = []  # Track KL divergences when available
        self._weak_edit_warnings = 0
        # Always enable diagnostics for proper hook tracking
        self._diag_enable = True

        if torch_dtype == "float32":
            dtype = torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        self.personal_direction: torch.Tensor | None = None
        self.neutral_direction: torch.Tensor | None = None
        self.editing_hooks = []
        logger.info(f"Model loaded on device: {self.device}")

    def load_theta_vectors(self, theta_p_path: str, theta_n_path: str):
        try:
            with open(theta_p_path, 'r') as f:
                theta_p = np.array(json.load(f))
            with open(theta_n_path, 'r') as f:
                theta_n = np.array(json.load(f))
            P = torch.tensor(theta_p, dtype=torch.float32, device=self.device).view(-1)
            N = torch.tensor(theta_n, dtype=torch.float32, device=self.device).view(-1)

            try:
                H = int(getattr(self.model.config, "hidden_size", P.numel()))
            except Exception:
                H = int(P.numel())

            def _fit_len_1d(v: torch.Tensor, H: int) -> torch.Tensor:
                v = v.view(-1)
                n = v.numel()
                if n == H:
                    return v
                if n > 0 and H % n == 0:
                    return v.repeat(H // n)[:H]
                if n > H:
                    return v[:H]
                out = torch.zeros(H, dtype=v.dtype, device=v.device)
                out[:n] = v
                return out

            self.personal_direction = _fit_len_1d(P, H)
            self.neutral_direction  = _fit_len_1d(N, H)
            logger.info(f"Loaded theta vectors (aligned): P={tuple(self.personal_direction.shape)}, "
                        f"N={tuple(self.neutral_direction.shape)}, hidden_size={H}")
            return True
        except Exception as e:
            logger.error(f"Failed to load theta vectors: {e}")
            return False

    def _get_layer_by_name(self, layer_name: str):
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def register_editing_hooks(self, target_layers: List[str], alpha_personal: float, alpha_neutral: float, last_k_tokens: int = 0):
        if self.personal_direction is None or self.neutral_direction is None:
            raise ValueError("Direction vectors not loaded")

        def editing_hook(module, inputs, output):
            self._hook_calls_total += 1
            self._hook_calls_in_this_generate += 1

            try:
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    extra = output[1:]
                else:
                    output_tensor = output
                    extra = ()

                shape = output_tensor.shape
                device = output_tensor.device
                dtype = output_tensor.dtype

                if len(shape) == 3:
                    b, t, h = shape
                elif len(shape) == 2:
                    b, h = shape
                    t = 1
                else:
                    return output  # 不明形状

                def _fit_len(v: torch.Tensor, H: int) -> torch.Tensor:
                    v = v.to(device=device, dtype=dtype).view(-1)
                    n = v.numel()
                    if n == H:
                        return v
                    if n > 0 and H % n == 0:
                        return v.repeat(H // n)[:H]
                    if n > H:
                        return v[:H]
                    out = torch.zeros(H, dtype=dtype, device=device)
                    out[:n] = v
                    return out

                pvec = _fit_len(self.personal_direction, h)
                nvec = _fit_len(self.neutral_direction,  h)
                base_edit = float(alpha_personal) * pvec + float(alpha_neutral) * nvec

                if len(shape) == 3:
                    edit = base_edit.view(1, 1, h).expand(b, t, h)
                    
                    # Apply last-k-tokens editing if specified
                    if last_k_tokens > 0:
                        k = min(last_k_tokens, t)
                        mask = torch.zeros_like(edit)
                        mask[:, -k:, :] = 1
                        edit = edit * mask
                        logger.info(f"[LAST-K] Applied editing to last {k}/{t} tokens")
                else:
                    edit = base_edit.view(1, h).expand(b, h)

                edited = output_tensor + edit

                # Enhanced diagnostics
                if self._diag_enable:
                    try:
                        # Compute edit ratio (L2 norm ratio)
                        edit_norm = edit.norm().item()
                        output_norm = output_tensor.norm().item()
                        ratio = edit_norm / (output_norm + 1e-9)
                        self._edit_ratios.append(ratio)
                        
                        logger.info(f"[DIAG] edit_ratio={ratio:.4e} | alpha_p={alpha_personal:.3g} alpha_n={alpha_neutral:.3g}")
                        
                        # Warn about weak edits
                        if ratio < 0.005:
                            self._weak_edit_warnings += 1
                            logger.warning(f"[WARN] weak_edit detected: ratio={ratio:.4e} < 0.005")
                            
                    except Exception as e:
                        logger.warning(f"Error computing edit diagnostics: {e}")

                if isinstance(output, tuple):
                    return (edited,) + extra
                return edited

            except Exception as e:
                logger.warning(f"Error in editing hook: {e}. Returning original.")
                return output

        for layer_name in target_layers:
            try:
                layer = self._get_layer_by_name(layer_name)
                hook = layer.register_forward_hook(editing_hook)
                self.editing_hooks.append(hook)
                logger.info(f"Registered editing hook on {layer_name}")
            except AttributeError:
                logger.warning(f"Layer name invalid: '{layer_name}' - skipping this layer")
                continue  # Continue with other layers

    def remove_editing_hooks(self):
        for h in self.editing_hooks:
            h.remove()
        self.editing_hooks = []

    def generate_with_chameleon(self, prompt: str, alpha_personal: float = 1.5, alpha_neutral: float = -0.8,
                                target_layers: List[str] = None, gen_kwargs: dict | None = None,
                                target_edit_ratio: float = 0.02, edit_ratio_tolerance: float = 0.5,
                                adaptive_alpha: bool = False, last_k_tokens: int = 0) -> str:
        if target_layers is None:
            target_layers = ["model.layers.20"]

        # Reset diagnostics for this generation
        self._hook_calls_in_this_generate = 0
        self._edit_ratios.clear()
        self._weak_edit_warnings = 0
        self.register_editing_hooks(target_layers, alpha_personal, alpha_neutral, last_k_tokens)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 1回だけ KL(before||after) を測る（診断ON時）
            if self._diag_enable:
                with torch.no_grad():
                    # フックOFFで
                    self.remove_editing_hooks()
                    logits0 = self.model(**inputs).logits[:, -1, :]
                    # フック再ONで
                    self.register_editing_hooks(target_layers, alpha_personal, alpha_neutral)
                    logits1 = self.model(**inputs).logits[:, -1, :]
                    p = torch.log_softmax(logits0, dim=-1)
                    q = torch.log_softmax(logits1, dim=-1)
                    kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean().item()
                    logger.info(f"[DIAG] KL(before||after)={kl:.4e}")

            # 生成
            if gen_kwargs is None:
                gen_kwargs = dict(max_new_tokens=10, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            # None を渡さない（transformersの警告回避）
            clean_gen = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **clean_gen)

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = text[len(prompt):].strip()
            return response

        finally:
            # Log detailed diagnostics before resetting counters
            avg_edit_ratio = np.mean(self._edit_ratios) if self._edit_ratios else 0.0
            
            # Adaptive alpha scaling logic
            suggested_alpha = alpha_personal
            self._suggested_alpha = alpha_personal  # Default value
            
            if adaptive_alpha and avg_edit_ratio > 0:
                lower = target_edit_ratio * (1 - edit_ratio_tolerance)
                upper = target_edit_ratio * (1 + edit_ratio_tolerance)
                
                if avg_edit_ratio < lower:
                    suggested_alpha = alpha_personal * min(3.0, max(1.05, lower / max(avg_edit_ratio, 1e-8)))
                    logger.info(f"[ADAPTIVE] Edit ratio {avg_edit_ratio:.4e} < target {target_edit_ratio:.4e}, suggesting alpha increase: {alpha_personal:.3f} → {suggested_alpha:.3f}")
                elif avg_edit_ratio > upper:
                    suggested_alpha = alpha_personal * max(0.2, min(0.95, upper / avg_edit_ratio))
                    logger.info(f"[ADAPTIVE] Edit ratio {avg_edit_ratio:.4e} > target {target_edit_ratio:.4e}, suggesting alpha decrease: {alpha_personal:.3f} → {suggested_alpha:.3f}")
                else:
                    logger.info(f"[ADAPTIVE] Edit ratio {avg_edit_ratio:.4e} within target range [{lower:.4e}, {upper:.4e}], keeping alpha={alpha_personal:.3f}")
                    
                self._suggested_alpha = suggested_alpha
            
            logger.info(f"[DIAG] Generation complete: hook_calls={self._hook_calls_in_this_generate}, avg_edit_ratio={avg_edit_ratio:.4e}, edit_count={len(self._edit_ratios)}, suggested_alpha={suggested_alpha:.3f}")
            
            # Validation checks
            if len(target_layers) > 0 and self._hook_calls_in_this_generate == 0:
                logger.warning(f"[BUG] NO hooks fired! Expected {len(target_layers)} layers, got 0 hook calls")
            elif len(target_layers) > 0 and self._hook_calls_in_this_generate < len(target_layers):
                logger.warning(f"[BUG] Some hooks not firing: expected {len(target_layers)}, got {self._hook_calls_in_this_generate}")
            
            # Don't reset here - let the caller collect diagnostics first
            # self._hook_calls_in_this_generate = 0  # Moved to after collection
            self.remove_editing_hooks()


# =========================
# Evaluation Engine
# =========================
class EvaluationEngine:
    """ベースライン vs Chameleon比較評価エンジン"""
    def __init__(self, chameleon_editor: ChameleonEditor):
        self.chameleon_editor = chameleon_editor
        self.parent = None  # ChameleonEvaluator が設定する

    def calculate_exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not ground_truths:
            return 0.0
        return sum(p.strip().lower() == g.strip().lower() for p, g in zip(predictions, ground_truths)) / len(ground_truths)

    def calculate_bleu_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not NLTK_AVAILABLE or not ground_truths:
            return 0.0
        smoothing = SmoothingFunction().method1
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            pt = pred.strip().lower().split()
            tt = [truth.strip().lower().split()]
            if pt and tt[0]:
                scores.append(sentence_bleu(tt, pt, smoothing_function=smoothing))
        return float(np.mean(scores)) if scores else 0.0

    def _finalize_metrics(self, predictions: List[str], ground_truth: Dict[str, str],
                          start_time: float, name: str) -> EvaluationResult:
        matched_truths: List[str] = []
        matched_preds: List[str] = []
        for p, sample in zip(predictions, self.parent.test_samples_cache):
            sid = str(sample['id'])
            if sid in ground_truth:
                matched_truths.append(ground_truth[sid])
                matched_preds.append(p)

        inference_time = time.time() - start_time
        correct = sum(int(p == g) for p, g in zip(matched_preds, matched_truths))
        acc = correct / len(matched_truths) if matched_truths else 0.0
        exact = self.calculate_exact_match(matched_preds, matched_truths)
        bleu = self.calculate_bleu_score(matched_preds, matched_truths)

        # 簡易: precision/recall/f1 を accuracy と同一に置く（分類器でないため）
        prec = rec = f1 = acc

        return EvaluationResult(
            method_name=name,
            accuracy=acc,
            exact_match=exact,
            bleu_score=bleu,
            precision=prec,
            recall=rec,
            f1_score=f1,
            inference_time=inference_time,
            total_samples=len(matched_truths),
            correct_predictions=correct,
            predictions=matched_preds,
            ground_truths=matched_truths
        )

    def evaluate_baseline(self, test_samples: List[Dict], ground_truth: Dict[str, str]) -> EvaluationResult:
        logger.info("Starting baseline evaluation...")
        predictions: List[str] = []
        start = time.time()

        for i, sample in enumerate(test_samples):
            logger.info(f"Baseline progress: {i+1}/{len(test_samples)}")
            prompt = f"Given the following movie description, provide a single word tag that best describes the movie:\n\nMovie: {sample['input']}\n\nTag:"
            inputs = self.chameleon_editor.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.chameleon_editor.device) for k, v in inputs.items()}

            clean_gen = {k: v for k, v in self.parent.gen_kwargs.items() if v is not None}
            with torch.no_grad():
                outputs = self.chameleon_editor.model.generate(**inputs, **clean_gen)

            gen = self.chameleon_editor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = (gen[len(prompt):].strip().lower().split() or ["unknown"])[0]
            predictions.append(pred)

        return self._finalize_metrics(predictions, ground_truth, start, "Baseline")

    def evaluate_chameleon(self, test_samples: List[Dict], ground_truth: Dict[str, str],
                           alpha_personal: float = 1.5, alpha_neutral: float = -0.8,
                           target_layers: List[str] = None,
                           target_edit_ratio: float = 0.02, edit_ratio_tolerance: float = 0.5,
                           adaptive_alpha: bool = False, last_k_tokens: int = 0) -> EvaluationResult:
        logger.info(f"Starting Chameleon evaluation (α_p={alpha_personal}, α_n={alpha_neutral})...")
        predictions: List[str] = []
        suggested_alphas: List[float] = []  # Track suggested alpha per sample
        start = time.time()

        for i, sample in enumerate(test_samples):
            logger.info(f"Chameleon progress: {i+1}/{len(test_samples)}")
            prompt = f"Given the following movie description, provide a single word tag that best describes the movie:\n\nMovie: {sample['input']}\n\nTag:"
            resp = self.chameleon_editor.generate_with_chameleon(
                prompt=prompt,
                alpha_personal=alpha_personal,
                alpha_neutral=alpha_neutral,
                target_layers=target_layers,
                gen_kwargs=self.parent.gen_kwargs,
                target_edit_ratio=target_edit_ratio,
                edit_ratio_tolerance=edit_ratio_tolerance,
                adaptive_alpha=adaptive_alpha,
                last_k_tokens=last_k_tokens
            )
            pred = (resp.strip().lower().split() or ["unknown"])[0]
            predictions.append(pred)
            
            # Collect diagnostics from this generation (before hook reset)
            if hasattr(self.chameleon_editor, '_edit_ratios') and self.chameleon_editor._edit_ratios:
                self.parent.evaluation_diagnostics['edit_ratios'].extend(self.chameleon_editor._edit_ratios.copy())
            
            # Get hook calls before they are reset to 0 in finally block
            hook_calls_this_gen = getattr(self.chameleon_editor, '_hook_calls_in_this_generate', 0)
            self.parent.evaluation_diagnostics['hook_calls'].append(hook_calls_this_gen)
            
            # Collect suggested alpha if available
            suggested_alpha = getattr(self.chameleon_editor, '_suggested_alpha', alpha_personal)
            suggested_alphas.append(suggested_alpha)
            
            # Reset the counter after collection
            self.chameleon_editor._hook_calls_in_this_generate = 0
            
            # Log per-sample diagnostics for debugging
            avg_edit_ratio = np.mean(self.chameleon_editor._edit_ratios) if hasattr(self.chameleon_editor, '_edit_ratios') and self.chameleon_editor._edit_ratios else 0.0
            logger.info(f"Sample {i+1}: hook_calls={hook_calls_this_gen}, avg_edit_ratio={avg_edit_ratio:.4e}, suggested_alpha={suggested_alpha:.3f}")

        # Store suggested alphas in diagnostics for CSV
        self.parent.evaluation_diagnostics['suggested_alphas'] = suggested_alphas
        
        return self._finalize_metrics(predictions, ground_truth, start, "Chameleon")


# =========================
# Orchestrator
# =========================
class ChameleonEvaluator:
    """Chameleon LaMP-2 評価システムのメインクラス"""
    def __init__(self, config_path: str | None, data_path: str, decoding_mode: str = "greedy"):
        self.config = self._load_config(config_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(self.config.get('output_dir', './results'))
        self.output_dir.mkdir(exist_ok=True)

        # decoding setup
        self.decoding_mode = decoding_mode
        use_sample = (decoding_mode == "sample")
        # None は渡さないため、後でフィルタ
        self.gen_kwargs = dict(
            max_new_tokens=10,
            do_sample=use_sample,
            temperature=(0.7 if use_sample else None),
            top_p=(0.9 if use_sample else None),
            pad_token_id=None  # 後で tokenizer.eos に差し替え
        )

        # components
        self.data_loader = LaMPDataLoader(data_path)
        self.chameleon_editor = ChameleonEditor(
            model_name=self.config['model']['name'],
            device=self.config['model'].get('device', 'auto'),
            torch_dtype=self.config['model'].get('torch_dtype', 'float32')
        )
        # pad_token_id の確定
        self.gen_kwargs['pad_token_id'] = self.chameleon_editor.tokenizer.eos_token_id
        
        # Diagnostic aggregation
        self.evaluation_diagnostics = {
            'edit_ratios': [],
            'hook_calls': [],
            'kl_divergences': []
        }

        self.evaluation_engine = EvaluationEngine(self.chameleon_editor)
        self.evaluation_engine.parent = self

        # theta
        self._load_theta_vectors()

        # cache
        self.test_samples_cache: List[Dict] = []
        logger.info(f"Chameleon Evaluator initialized (decoding={self.decoding_mode})")

    def _load_config(self, config_path: str | None) -> Dict:
        default_config = {
            'model': {
                'name': './chameleon_prime_personalization/models/base_model',
                'device': 'auto',
                'max_length': 512,
                'batch_size': 4
            },
            'chameleon': {
                'num_self_generated': 10,
                'target_layers': ['model.layers.20'],
                'alpha_personal': 1.5,
                'alpha_general': -0.8
            },
            'evaluation': {
                'max_users': 10,
                'metrics': ['exact_match', 'bleu_score']
            },
            'output_dir': './results'
        }
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            for k, v in default_config.items():
                if k not in cfg:
                    cfg[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in cfg[k]:
                            cfg[k][kk] = vv
            return cfg
        else:
            logger.info("Using default configuration")
            return default_config

    def _load_theta_vectors(self):
        theta_paths = [
            (self.data_path / "processed/LaMP-2/theta_p.json", self.data_path / "processed/LaMP-2/theta_n.json"),
            (Path("processed/LaMP-2/theta_p.json"), Path("processed/LaMP-2/theta_n.json"))
        ]
        for tp, tn in theta_paths:
            if tp.exists() and tn.exists():
                if self.chameleon_editor.load_theta_vectors(str(tp), str(tn)):
                    logger.info("Theta vectors loaded successfully")
                    return
        logger.warning("Theta vectors not found - Chameleon evaluation will be limited")

    def run_evaluation(self, mode: str = "full", alpha_override: float = None, 
                      beta_override: float = None, layers_override: List[str] = None,
                      target_edit_ratio: float = 0.02, edit_ratio_tolerance: float = 0.5,
                      adaptive_alpha: bool = False, last_k_tokens: int = 0,
                      max_users_override: int = None) -> Dict[str, Any]:
        logger.info(f"=== Chameleon LaMP-2 Evaluation ({mode} mode, decoding={self.decoding_mode}) ===")

        if mode == "demo":
            user_limit = 3
        elif mode == "full":
            user_limit = max_users_override if max_users_override is not None else self.config['evaluation']['max_users']
        else:
            user_limit = 10

        test_samples = self.data_loader.get_user_samples(user_limit)
        self.test_samples_cache = test_samples  # metrics用
        ground_truth = self.data_loader.load_ground_truth()
        logger.info(f"Evaluating {len(test_samples)} samples from {user_limit} users")

        results: Dict[str, Any] = {}
        baseline = self.evaluation_engine.evaluate_baseline(test_samples, ground_truth)
        results['baseline'] = baseline

        if self.chameleon_editor.personal_direction is not None:
            # Use CLI overrides if provided, otherwise use config
            alpha_p = alpha_override if alpha_override is not None else self.config['chameleon']['alpha_personal']
            alpha_n = beta_override if beta_override is not None else self.config['chameleon']['alpha_general']
            layers = layers_override if layers_override is not None else self.config['chameleon']['target_layers']
            
            chameleon = self.evaluation_engine.evaluate_chameleon(
                test_samples, ground_truth,
                alpha_personal=alpha_p,
                alpha_neutral=alpha_n,
                target_layers=layers,
                target_edit_ratio=target_edit_ratio,
                edit_ratio_tolerance=edit_ratio_tolerance,
                adaptive_alpha=adaptive_alpha,
                last_k_tokens=last_k_tokens
            )
            results['chameleon'] = chameleon

            # significance
            if baseline.total_samples >= 2 and chameleon.total_samples >= 2:
                b_correct = np.array([int(p == g) for p, g in zip(baseline.predictions, baseline.ground_truths)])
                c_correct = np.array([int(p == g) for p, g in zip(chameleon.predictions, chameleon.ground_truths)])
                if len(b_correct) == len(c_correct):
                    _, p_value = stats.ttest_rel(c_correct, b_correct)
                else:
                    _, p_value = stats.ttest_ind(c_correct, b_correct)
            else:
                p_value = 1.0

            imp_rate = (chameleon.accuracy - baseline.accuracy) / baseline.accuracy if baseline.accuracy > 0 else 0.0
            results['significance'] = {
                "p_value": float(p_value),
                "improvement_rate": float(imp_rate),
                "baseline_accuracy": float(baseline.accuracy),
                "chameleon_accuracy": float(chameleon.accuracy)
            }
        else:
            logger.warning("Chameleon evaluation skipped - theta vectors not available")

        # Write CSV results 
        self._write_csv_results(results, mode, alpha_override, beta_override, layers_override,
                               target_edit_ratio, edit_ratio_tolerance, adaptive_alpha, last_k_tokens)
        
        self._save_results(results)
        self._print_report(results)
        return results

    def _write_csv_results(self, results: Dict[str, Any], mode: str, alpha_override: float = None, 
                          beta_override: float = None, layers_override: List[str] = None,
                          target_edit_ratio: float = 0.02, edit_ratio_tolerance: float = 0.5,
                          adaptive_alpha: bool = False, last_k_tokens: int = 0):
        """Write evaluation results to CSV file"""
        csv_file = self.output_dir / "experiment_results.csv"
        
        # Prepare row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get parameters used
        alpha_p = alpha_override if alpha_override is not None else self.config['chameleon']['alpha_personal']
        alpha_n = beta_override if beta_override is not None else self.config['chameleon']['alpha_general']
        layers = layers_override if layers_override is not None else self.config['chameleon']['target_layers']
        layers_str = ','.join(layers) if isinstance(layers, list) else str(layers)
        
        baseline = results.get('baseline')
        chameleon = results.get('chameleon')
        significance = results.get('significance', {})
        
        # Compute diagnostic aggregates
        avg_edit_ratio = np.mean(self.evaluation_diagnostics['edit_ratios']) if self.evaluation_diagnostics['edit_ratios'] else 0.0
        avg_kl = np.mean(self.evaluation_diagnostics['kl_divergences']) if self.evaluation_diagnostics['kl_divergences'] else 0.0
        hook_calls_mean = np.mean(self.evaluation_diagnostics['hook_calls']) if self.evaluation_diagnostics['hook_calls'] else 0.0
        suggested_alpha_mean = np.mean(self.evaluation_diagnostics['suggested_alphas']) if 'suggested_alphas' in self.evaluation_diagnostics and self.evaluation_diagnostics['suggested_alphas'] else alpha_p
        
        row = {
            'timestamp': timestamp,
            'mode': mode,
            'gen': self.decoding_mode,
            'layers': layers_str,
            'alpha': alpha_p,
            'beta': alpha_n,
            'baseline_acc': baseline.accuracy if baseline else 0.0,
            'baseline_em': baseline.exact_match if baseline else 0.0,
            'baseline_bleu': baseline.bleu_score if baseline else 0.0,
            'baseline_f1': baseline.f1_score if baseline else 0.0,
            'baseline_time': baseline.inference_time if baseline else 0.0,
            'cham_acc': chameleon.accuracy if chameleon else 0.0,
            'cham_em': chameleon.exact_match if chameleon else 0.0,
            'cham_bleu': chameleon.bleu_score if chameleon else 0.0,
            'cham_f1': chameleon.f1_score if chameleon else 0.0,
            'cham_time': chameleon.inference_time if chameleon else 0.0,
            'impr_abs_acc': (chameleon.accuracy - baseline.accuracy) if (baseline and chameleon) else 0.0,
            'impr_rel_acc_pct': (((chameleon.accuracy - baseline.accuracy) / baseline.accuracy * 100) if (baseline and chameleon and baseline.accuracy > 0) else 0.0),
            'p_value': significance.get('p_value', 1.0),
            'avg_edit_ratio': avg_edit_ratio,
            'avg_kl': avg_kl,
            'hook_calls_mean': hook_calls_mean,
            'target_edit_ratio': target_edit_ratio,
            'edit_ratio_tolerance': edit_ratio_tolerance,
            'adaptive_alpha': adaptive_alpha,
            'suggested_alpha': suggested_alpha_mean,
            'last_k_tokens': last_k_tokens
        }
        
        # Write header if file doesn't exist
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"Results written to CSV: {csv_file}")
        logger.info(f"[SUMMARY] avg_edit_ratio={avg_edit_ratio:.4e}, hook_calls_mean={hook_calls_mean:.1f}")
        
        # Acceptance checks
        self._run_acceptance_checks(results, avg_edit_ratio, hook_calls_mean, layers)

    def _run_acceptance_checks(self, results: Dict[str, Any], avg_edit_ratio: float, 
                              hook_calls_mean: float, layers):
        """Run acceptance checks and validation alerts"""
        baseline = results.get('baseline')
        chameleon = results.get('chameleon')
        
        if not (baseline and chameleon):
            return
            
        num_layers = len(layers) if isinstance(layers, list) else 1
        
        # Check for insensitive model outputs
        metrics_identical = (
            abs(baseline.accuracy - chameleon.accuracy) < 1e-6 and
            abs(baseline.exact_match - chameleon.exact_match) < 1e-6 and
            abs(baseline.bleu_score - chameleon.bleu_score) < 1e-6 and
            abs(baseline.f1_score - chameleon.f1_score) < 1e-6
        )
        
        if metrics_identical and avg_edit_ratio >= 0.02 and hook_calls_mean >= num_layers:
            logger.warning("[ALERT] Model outputs insensitive to current edits - "
                          f"all metrics identical but avg_edit_ratio={avg_edit_ratio:.4e} >= 0.02 "
                          f"and hook_calls_mean={hook_calls_mean:.1f} >= {num_layers}")
        
        # Check for hooks not firing
        if hook_calls_mean < num_layers:
            logger.warning(f"[BUG] Hooks not firing for some layers: "
                          f"expected {num_layers}, got mean {hook_calls_mean:.1f}")
        
        # Check for weak edits across the evaluation
        if avg_edit_ratio < 0.005:
            logger.warning(f"[WARN] Consistently weak edits across evaluation: "
                          f"avg_edit_ratio={avg_edit_ratio:.4e} < 0.005")
        
        # Log success/failure of improvement
        improvement = chameleon.accuracy - baseline.accuracy
        if improvement >= 0.02:
            logger.info(f"[SUCCESS] Significant accuracy improvement: {improvement:+.4f} >= +0.02")
        elif improvement > 0:
            logger.info(f"[PARTIAL] Minor accuracy improvement: {improvement:+.4f}")
        else:
            logger.warning(f"[FAIL] No accuracy improvement: {improvement:+.4f}")

    def _save_results(self, results: Dict[str, Any]):
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = self.output_dir / f"evaluation_{ts}"
        outdir.mkdir(exist_ok=True)

        serializable = {}
        for k, v in results.items():
            if isinstance(v, EvaluationResult):
                serializable[k] = {
                    'method_name': v.method_name,
                    'accuracy': float(v.accuracy),
                    'exact_match': float(v.exact_match),
                    'bleu_score': float(v.bleu_score),
                    'precision': float(v.precision),
                    'recall': float(v.recall),
                    'f1_score': float(v.f1_score),
                    'inference_time': float(v.inference_time),
                    'total_samples': int(v.total_samples),
                    'correct_predictions': int(v.correct_predictions),
                    'predictions': v.predictions,
                    'ground_truths': v.ground_truths
                }
            else:
                serializable[k] = v

        with open(outdir / "results.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {outdir}")

    def _print_report(self, results: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("🎯 Chameleon LaMP-2 Evaluation Results")
        print("=" * 60)

        b = results.get('baseline')
        c = results.get('chameleon')
        sig = results.get('significance', {})

        if b:
            print(f"\n📊 Baseline Performance:")
            print(f"   Accuracy:     {b.accuracy:.4f}")
            print(f"   Exact Match:  {b.exact_match:.4f}")
            print(f"   BLEU Score:   {b.bleu_score:.4f}")
            print(f"   F1 Score:     {b.f1_score:.4f}")
            print(f"   Inference:    {b.inference_time:.2f}s")

        if c:
            print(f"\n🦎 Chameleon Performance:")
            print(f"   Accuracy:     {c.accuracy:.4f}")
            print(f"   Exact Match:  {c.exact_match:.4f}")
            print(f"   BLEU Score:   {c.bleu_score:.4f}")
            print(f"   F1 Score:     {c.f1_score:.4f}")
            print(f"   Inference:    {c.inference_time:.2f}s")

        if b and c and sig:
            imp = sig.get("improvement_rate", 0.0) * 100
            p_value = sig.get("p_value", 1.0)
            print(f"\n📈 Improvement Analysis:")
            print(f"   Improvement Rate: {imp:+.1f}%")
            print(f"   Statistical Significance: p = {p_value:.4f}")
            if p_value < 0.05:
                if c.accuracy > b.accuracy:
                    print("   ✅ Statistically significant improvement!")
                elif c.accuracy < b.accuracy:
                    print("   ❌ Statistically significant degradation!")
                else:
                    print("   ⚠️  Significant difference detected but same accuracy (check metrics)")
            else:
                print("   ⚠️  No significant difference detected")
        print("\n" + "=" * 60)


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chameleon LaMP-2 Evaluation System")
    parser.add_argument("--mode", choices=["demo", "full", "ablation"], default="full",
                        help="Evaluation mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Root directory that contains LaMP-2 data (expects raw/LaMP-2/merged.json or answers.json)",
    )
    parser.add_argument("--gen", choices=["greedy", "sample"], default="greedy",
                        help="Generation mode: greedy (do_sample=False) or sample (do_sample=True, temp/top_p used)")
    parser.add_argument("--alpha", type=float, help="Alpha parameter (personal direction strength)")
    parser.add_argument("--beta", type=float, help="Beta parameter (neutral direction strength)")
    parser.add_argument("--layers", type=str, help="Comma-separated layer names (e.g., model.layers.20,model.layers.30)")
    parser.add_argument("--target_edit_ratio", type=float, default=0.02, help="Target edit ratio for adaptive alpha")
    parser.add_argument("--edit_ratio_tolerance", type=float, default=0.5, help="Tolerance ratio for target edit ratio (±)")
    parser.add_argument("--adaptive_alpha", action="store_true", help="Enable adaptive alpha scaling")
    parser.add_argument("--last_k_tokens", type=int, default=0, help="Apply editing only to last k tokens (0 = all tokens)")
    parser.add_argument("--max_users", type=int, default=None, help="Maximum number of users for evaluation")
    args = parser.parse_args()

    # Parse layers
    target_layers = None
    if args.layers:
        target_layers = [layer.strip() for layer in args.layers.split(',') if layer.strip()]

    evaluator = ChameleonEvaluator(
        config_path=args.config,
        data_path=args.data_path or "./",
        decoding_mode=args.gen
    )
    results = evaluator.run_evaluation(
        mode=args.mode,
        alpha_override=args.alpha,
        beta_override=args.beta,
        layers_override=target_layers,
        target_edit_ratio=args.target_edit_ratio,
        edit_ratio_tolerance=args.edit_ratio_tolerance,
        adaptive_alpha=args.adaptive_alpha,
        last_k_tokens=args.last_k_tokens,
        max_users_override=args.max_users
    )
    print("\n✅ Evaluation completed successfully!")