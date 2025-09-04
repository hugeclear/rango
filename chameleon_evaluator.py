
'\nChameleon LaMP-2 Evaluation System\n完全なChameleon実装とLaMP-2ベンチマーク自動評価システム\n\n特徴:\n- PyTorchフックによるTransformer中間層埋め込み抽出\n- SVD方向学習とリアルタイム埋め込み編集\n- 統計的有意性検定を含む包括的評価\n'
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
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from scipy import stats
from collections import defaultdict
import logging
try:
    from causal_inference import CausalGraphBuilder, TemporalConstraintManager, DoCalculusEstimator
    from causal_inference.causal_graph_builder import integrate_with_chameleon_data_loader
    from causal_inference.temporal_constraints import integrate_temporal_constraints_with_chameleon
    from causal_inference.do_calculus import integrate_do_calculus_with_chameleon_evaluation
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    logging.warning('Causal inference modules not available. Basic functionality preserved.')
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult():
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


class TwoStepPrefixProcessor(LogitsProcessor):
    """
    2-stage prefix constraints for multi-token tags like "sci-fi".
    More robust than first-token-only constraints.
    """
    def __init__(self, tokenizer, tags, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        
        # Map each tag to sequence of up to 2 token IDs
        self.seq_map = []
        for tag in tags:
            try:
                ids = tokenizer(tag, add_special_tokens=False).input_ids
                if len(ids) > 0:
                    self.seq_map.append(ids[:2])  # Take first 2 tokens max
            except Exception:
                continue
        
        # Cache EOS token for convenience
        self.eos_token_id = tokenizer.eos_token_id
        if isinstance(self.eos_token_id, (list, tuple)):
            self.eos_token_id = self.eos_token_id[0]
            
    def __call__(self, input_ids, scores):
        # Only apply constraints during generation phase
        if input_ids.size(1) < self.prompt_len + 1:
            return scores
            
        step = input_ids.size(1) - self.prompt_len  # 1, 2, 3, ...
        vocab_size = scores.size(-1)
        
        # Create mask (default: block everything)
        mask = torch.full(
            (vocab_size,), 
            float("-inf"), 
            dtype=scores.dtype, 
            device=scores.device
        )
        
        candidates = set()
        
        if step == 1:
            # First generation step - allow first tokens of all sequences
            candidates = {seq[0] for seq in self.seq_map if len(seq) >= 1}
        elif step == 2:
            # Second generation step - constrain based on previous token
            prev_token = input_ids[0, self.prompt_len].item()
            
            # Find sequences that start with prev_token and get their second tokens
            for seq in self.seq_map:
                if len(seq) >= 2 and seq[0] == prev_token:
                    candidates.add(seq[1])
            
            # For single-token tags that matched in step 1, allow EOS
            single_token_complete = any(
                len(seq) == 1 and seq[0] == prev_token 
                for seq in self.seq_map
            )
            if single_token_complete and self.eos_token_id is not None:
                candidates.add(self.eos_token_id)
        else:
            # Step 3+ - no constraints (max_new_tokens=2 handles length)
            return scores
        
        # If no candidates found, don't apply constraints (safety fallback)
        if not candidates:
            return scores
        
        # Apply mask - allow only candidate tokens
        mask[list(candidates)] = 0.0
        return scores + mask


class AllowedFirstTokenProcessor(LogitsProcessor):
    """
    最初の生成トークンをタグ候補に制約するプロセッサ（LLaMAの空白付きサブワードを考慮）。
    - 空白付き/無し + 原形/小文字 の全組合せの「先頭トークンID」を許可集合に追加
    - 許可集合が空のときは何もしない（安全フォールバック）
    """
    def __init__(self, tokenizer, allowed_tags, prompt_len: int):
        self.prompt_len = int(prompt_len)
        self.allowed_first_ids = self._build_allowed_ids(tokenizer, allowed_tags)

    @staticmethod
    def _first_token_ids(tokenizer, text: str):
        enc = tokenizer(text, add_special_tokens=False)
        ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        if isinstance(ids, list) and len(ids) > 0:
            return [ids[0]]
        return []

    @classmethod
    def _build_allowed_ids(cls, tokenizer, allowed_tags):
        cand_ids = []
        for tag in allowed_tags:
            forms = {tag, tag.lower(), " " + tag, " " + tag.lower()}
            for t in forms:
                cand_ids.extend(cls._first_token_ids(tokenizer, t))
        if len(cand_ids) == 0:
            return None
        uniq = sorted(set(cand_ids))
        return torch.tensor(uniq, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 許可集合が空なら制約しない（安全）
        if self.allowed_first_ids is None:
            return scores
        # ちょうど最初の新規トークン生成タイミングのみ制約（use_cache=True 前提）
        if input_ids.size(1) == self.prompt_len:
            vocab = scores.size(-1)
            mask = torch.full((vocab,), float("-inf"), dtype=scores.dtype, device=scores.device)
            mask[self.allowed_first_ids.to(scores.device)] = 0.0
            scores = scores + mask
        return scores


class AllowedPhraseProcessor(LogitsProcessor):
    """
    Phrase-level constrained decoding using full-token sequences of allowed labels.
    Tracks prefix match and only allows next tokens that continue any allowed phrase.
    Safe fallback: if no candidates match, returns scores unchanged.
    """
    def __init__(self, tokenizer, phrases, prompt_len: int, eos_token_id):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        # Normalize eos into list
        if isinstance(eos_token_id, (list, tuple)):
            self.eos_ids = list(eos_token_id)
        elif eos_token_id is None:
            self.eos_ids = []
        else:
            self.eos_ids = [int(eos_token_id)]
        # Build sequences of token ids for each phrase (encode with leading space)
        self.seqs = []
        for p in (phrases or []):
            try:
                s = str(p).strip()
                if not s:
                    continue
                # For pure-numeric phrases (e.g., "1", "10"), avoid forcing leading space
                if s.isdigit():
                    ids = tokenizer.encode(s, add_special_tokens=False)
                else:
                    ids = tokenizer.encode(" " + s, add_special_tokens=False)
                if isinstance(ids, list) and len(ids) > 0:
                    self.seqs.append(ids)
            except Exception:
                continue

    def __call__(self, input_ids, scores):
        try:
            # Apply only after prompt; batch=1 expected but handle others safely
            if input_ids.dim() != 2 or input_ids.size(0) != 1:
                return scores
            if input_ids.size(1) <= self.prompt_len:
                return scores

            gen_suffix = input_ids[0, self.prompt_len:].tolist()

            # Determine allowed next tokens by prefix matching
            k = len(gen_suffix)
            if k == 0:
                allowed = {seq[0] for seq in self.seqs if len(seq) > 0}
            else:
                matched = [seq for seq in self.seqs if len(seq) >= k and seq[:k] == gen_suffix]
                if not matched:
                    return scores  # no constraint if nothing matches
                cont = {seq[k] for seq in matched if len(seq) > k}
                if cont:
                    allowed = cont
                else:
                    allowed = set(self.eos_ids)

            if not allowed:
                return scores
            vocab = scores.size(-1)
            mask = torch.full((vocab,), float("-inf"), dtype=scores.dtype, device=scores.device)
            mask[list(allowed)] = 0.0
            return scores + mask
        except Exception:
            return scores

class LaMPDataLoader():
    'LaMP-2データの読み込みとユーザー分割'

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.merged_data = None
        self.ground_truth = None

    def load_merged_data(self) -> List[Dict]:
        primary_merged = (self.data_path / 'raw/LaMP-2/merged.json')
        if primary_merged.exists():
            logger.info(f'Loading merged data from priority path: {primary_merged}')
            with open(primary_merged, 'r', encoding='utf-8') as f:
                self.merged_data = json.load(f)
            return self.merged_data
        backup_answers = (self.data_path / 'raw/LaMP-2/answers.json')
        if backup_answers.exists():
            logger.info(f'Loading data from backup answers: {backup_answers}')
            with open(backup_answers, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            if (isinstance(answers_data, dict) and ('golds' in answers_data)):
                self.merged_data = []
                for gold in answers_data['golds'][:1000]:
                    self.merged_data.append({'id': gold['id'], 'input': f"Question: {gold.get('input', 'Unknown question')}", 'output': gold['output']})
                logger.info(f'Converted {len(self.merged_data)} samples from answers format')
                return self.merged_data
        possible_paths = [(self.data_path / 'chameleon_prime_personalization/data/raw/LaMP-2/merged.json'), (self.data_path / 'processed/LaMP-2/merged.json'), (self.data_path / 'data/raw/LaMP-2/merged.json'), (self.data_path / 'merged.json')]
        for path in possible_paths:
            if path.exists():
                logger.info(f'Loading merged data from fallback path: {path}')
                with open(path, 'r', encoding='utf-8') as f:
                    self.merged_data = json.load(f)
                return self.merged_data
        backup_path = (self.data_path / 'data/raw/LaMP_all/LaMP_2/user-based/dev/dev_questions.json')
        if backup_path.exists():
            logger.info(f'Using backup data source: {backup_path}')
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            if (isinstance(backup_data, dict) and ('instances' in backup_data)):
                self.merged_data = backup_data['instances'][:1000]
                logger.info(f'Loaded {len(self.merged_data)} samples from backup source')
                return self.merged_data
        raise FileNotFoundError('No valid data source found (primary or backup)')

    def load_ground_truth(self) -> Dict[(str, str)]:
        primary_answers = (self.data_path / 'raw/LaMP-2/answers.json')
        if primary_answers.exists():
            logger.info(f'Loading ground truth from priority path: {primary_answers}')
            with open(primary_answers, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            if (isinstance(answers_data, dict) and ('golds' in answers_data)):
                golds = answers_data['golds']
                return {str(g['id']): g['output'].strip().lower() for g in golds}
            elif isinstance(answers_data, list):
                return {str(a['id']): a['output'].strip().lower() for a in answers_data}
        possible_paths = [(self.data_path / 'chameleon_prime_personalization/data/raw/LaMP-2/answers.json'), (self.data_path / 'data/raw/LaMP-2/answers.json'), (self.data_path / 'answers.json')]
        for path in possible_paths:
            if path.exists():
                logger.info(f'Loading ground truth from fallback path: {path}')
                with open(path, 'r', encoding='utf-8') as f:
                    answers_data = json.load(f)
                if (isinstance(answers_data, dict) and ('golds' in answers_data)):
                    golds = answers_data['golds']
                    return {str(g['id']): g['output'].strip().lower() for g in golds}
                elif isinstance(answers_data, list):
                    return {str(a['id']): a['output'].strip().lower() for a in answers_data}
        backup_answers_path = (self.data_path / 'data/raw/LaMP_all/LaMP_2/user-based/dev/dev_outputs.json')
        if backup_answers_path.exists():
            logger.info(f'Using backup ground truth: {backup_answers_path}')
            with open(backup_answers_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            if (isinstance(answers_data, dict) and ('golds' in answers_data)):
                golds = answers_data['golds']
                return {str(g['id']): g['output'].strip().lower() for g in golds}
        logger.warning('Ground truth not found, evaluation will be prediction-only')
        return {}

    def get_user_samples(self, user_limit: int=10) -> List[Dict]:
        if (not self.merged_data):
            self.load_merged_data()
        user_data = defaultdict(list)
        for item in self.merged_data:
            user_id = str(item['id'])[:3]
            user_data[user_id].append(item)
        selected_samples = []
        for (i, (_, samples)) in enumerate(user_data.items()):
            if (i >= user_limit):
                break
            selected_samples.extend(samples[:5])
        logger.info(f'Selected {len(selected_samples)} samples from {min(len(user_data), user_limit)} users')
        return selected_samples

    def load_split(self, split: str, limit: int = None) -> List[Dict]:
        """Load dataset split with optional limit."""
        if not self.merged_data:
            self.load_merged_data()
        
        # For now, return merged_data as test split (can be enhanced for train/dev splits)
        data = self.merged_data or []
        
        if limit is not None and limit > 0:
            data = data[:limit]
            
        return data
    
    def load_test_data(self, limit: int = None) -> List[Dict]:
        """Load test data with optional limit."""
        return self.load_split('test', limit=limit)
    
    def load_train_data(self, limit: int = None) -> List[Dict]:  
        """Load train data with optional limit."""
        return self.load_split('train', limit=limit)

class ChameleonEditor():
    '\n    Transformerの中間出力に方向ベクトルを加算して編集する\n    '

    def __init__(self, model_name: str='./chameleon_prime_personalization/models/base_model', device: str='auto', torch_dtype: str='float32'):
        from pathlib import Path
        
        # Check if model_name is a HuggingFace model identifier (contains '/')
        # or a local path
        if '/' in model_name and not Path(model_name).exists():
            # Treat as HuggingFace model identifier
            self.model_name = model_name
            load_kwargs = {'local_files_only': False}
        else:
            # Treat as local path
            resolved = Path(model_name).expanduser()
            if (not resolved.is_absolute()):
                resolved = (Path.cwd() / resolved).resolve()
            if (not resolved.exists()):
                raise FileNotFoundError(f'Model path not found: {resolved}')
            self.model_name = str(resolved)
            model_name = str(resolved)
            load_kwargs = {'local_files_only': True}
        self.device = torch.device(('cuda' if (torch.cuda.is_available() and (device == 'auto')) else device))
        self._hook_calls_total = 0
        self._hook_calls_in_this_generate = 0
        self._edit_ratios = []
        self._kl_divergences = []
        self._weak_edit_warnings = 0
        self._diag_enable = True
        self._hook_handles = []
        self._registered_layers = set()
        self._weak_streak = 0
        self._editing_disabled = False
        self._diag_token_level = False
        self._diag_kl = False
        self._tokenization_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        if (torch_dtype == 'float32'):
            dtype = torch.float32
        elif (torch_dtype == 'float16'):
            dtype = torch.float16
        else:
            dtype = torch.float32
        logger.info(f'Loading model: {model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=('auto' if (self.device.type == 'cuda') else None), **load_kwargs)
        if (self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if (hasattr(self.model.config, 'pad_token_id') and (self.model.config.pad_token_id is None)):
            self.model.config.pad_token_id = self.model.config.eos_token_id
        try:
            if hasattr(self.model.config, 'attn_implementation'):
                self.model.config.attn_implementation = 'flash_attention_2'
        except:
            pass
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id
            if hasattr(self.model.generation_config, 'do_sample'):
                delattr(self.model.generation_config, 'do_sample')
            if hasattr(self.model.generation_config, 'temperature'):
                delattr(self.model.generation_config, 'temperature')
            if hasattr(self.model.generation_config, 'top_p'):
                delattr(self.model.generation_config, 'top_p')
        self.model.eval()
        self.personal_direction: (torch.Tensor | None) = None
        self.neutral_direction: (torch.Tensor | None) = None
        self.fakeit_direction: (torch.Tensor | None) = None
        self.editing_hooks = self._hook_handles
        self.gate_params = {'h0': 10.0, 'a': 0.4, 'u0': 1.0, 'beta_min': 0.1, 'gamma_max': 0.5}
        self.fakeit_cache = {}
        self.causal_graph_builder = None
        self.temporal_constraint_manager = None
        self.causal_mask = None
        self.use_causal_constraints = False
        logger.info(f'Model loaded on device: {self.device}')
        if CAUSAL_INFERENCE_AVAILABLE:
            logger.info('Causal inference capabilities available')

    def initialize_causal_inference(self, user_profiles: List[Dict], enable_causal_constraints: bool=True, causality_radius: float=86400.0) -> bool:
        '\n        Initialize causal inference components for constrained editing\n        \n        Args:\n            user_profiles: User profiles from data loader\n            enable_causal_constraints: Whether to enable causal constraints\n            causality_radius: Temporal causality radius in seconds\n            \n        Returns:\n            True if initialization successful\n        '
        if (not CAUSAL_INFERENCE_AVAILABLE):
            logger.warning('Causal inference not available - continuing without constraints')
            return False
        try:
            self.causal_graph_builder = CausalGraphBuilder(alpha=0.05, max_parents=5, min_samples=10, cache_dir='./causal_cache')
            self.temporal_constraint_manager = integrate_temporal_constraints_with_chameleon(self, user_profiles, causality_radius)
            if (len(user_profiles) >= 10):
                causal_result = self.causal_graph_builder.build_causal_graph(user_profiles)
                if causal_result:
                    logger.info(f'Causal graph built with {np.sum((causal_result.adjacency_matrix != 0))} edges')
                    target_features = ['likes_action', 'likes_comedy', 'likes_drama', 'avg_rating']
                    self.causal_mask = self.causal_graph_builder.get_causal_mask(causal_result, target_features)
                    self.use_causal_constraints = enable_causal_constraints
            logger.info('Causal inference initialized successfully')
            return True
        except Exception as e:
            logger.error(f'Failed to initialize causal inference: {e}')
            return False

    def load_theta_vectors(self, theta_p_path: str, theta_n_path: str):
        try:
            with open(theta_p_path, 'r') as f:
                theta_p = np.array(json.load(f))
            with open(theta_n_path, 'r') as f:
                theta_n = np.array(json.load(f))
            P = torch.tensor(theta_p, dtype=torch.float32, device=self.device).view((- 1))
            N = torch.tensor(theta_n, dtype=torch.float32, device=self.device).view((- 1))
            try:
                H = int(getattr(self.model.config, 'hidden_size', P.numel()))
            except Exception:
                H = int(P.numel())

            def _fit_len_1d(v: torch.Tensor, H: int) -> torch.Tensor:
                v = v.view((- 1))
                n = v.numel()
                if (n == H):
                    return v
                if ((n > 0) and ((H % n) == 0)):
                    return v.repeat((H // n))[:H]
                if (n > H):
                    return v[:H]
                out = torch.zeros(H, dtype=v.dtype, device=v.device)
                out[:n] = v
                return out
            self.personal_direction = _fit_len_1d(P, H)
            self.neutral_direction = _fit_len_1d(N, H)
            logger.info(f'Loaded theta vectors (aligned): P={tuple(self.personal_direction.shape)}, N={tuple(self.neutral_direction.shape)}, hidden_size={H}')
            return True
        except Exception as e:
            logger.error(f'Failed to load theta vectors: {e}')
            return False

    def generate_personal_fakeit(self, user_context: str, question: str, user_id: str=None) -> str:
        'Generate Personal Fake-it text using LLM'
        cache_key = f'{user_id}_{(hash((user_context + question)) % 100000)}'
        if (cache_key in self.fakeit_cache):
            return self.fakeit_cache[cache_key]
        prompt = f'''You are writing a short, first-person preference blurb for a user.
Do NOT mention labels or categories explicitly. Do NOT guess the answer.
Write 2–3 sentences that capture stable tastes and style preferences
based on this (possibly noisy) history:

{user_context[:400]}

Constraints:
- No label leakage, no category names.
- Focus on recurring motifs (themes, tone, pace, director/author styles).
- Keep it generic and timeless.

Return only the blurb text:'''
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
            with torch.no_grad():
                gen = dict(max_new_tokens=80, temperature=0.7, do_sample=True, repetition_penalty=1.1, top_p=0.9)
                pad_id = self.tokenizer.eos_token_id
                if isinstance(pad_id, (list, tuple)):
                    pad_id = pad_id[0]
                gen['pad_token_id'] = pad_id
                outputs = self.model.generate(**inputs, **gen)
            fakeit_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            self.fakeit_cache[cache_key] = fakeit_text
            return fakeit_text
        except Exception as e:
            logger.warning(f'Fake-it generation failed: {e}')
            fallback = 'I appreciate well-crafted narratives with compelling character development and engaging themes that resonate with contemporary audiences.'
            self.fakeit_cache[cache_key] = fallback
            return fallback

    def encode_fakeit_to_direction(self, fakeit_text: str) -> torch.Tensor:
        'Convert Fake-it text to direction vector via embedding'
        try:
            inputs = self.tokenizer(fakeit_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[(- 1)]
                fakeit_embedding = last_hidden.mean(dim=1).squeeze(0)
                fakeit_embedding = F.normalize(fakeit_embedding, p=2, dim=0)
            return fakeit_embedding
        except Exception as e:
            logger.error(f'Fake-it encoding failed: {e}')
            hidden_size = getattr(self.model.config, 'hidden_size', 4096)
            return torch.zeros(hidden_size, device=self.device)

    def compute_gating_weights(self, history_length: int, uncertainty: float) -> tuple:
        'Compute α, β, γ weights for 3-component mixing'
        (h, u) = (float(history_length), float(uncertainty))
        params = self.gate_params
        alpha = torch.sigmoid(torch.tensor((params['a'] * (h - params['h0'])))).item()
        gamma_raw = ((1 - alpha) * min(1.0, (u / params['u0'])))
        gamma = min(gamma_raw, params['gamma_max'])
        beta_raw = ((1 - alpha) - gamma)
        beta = max(beta_raw, params['beta_min'])
        total = (((abs(alpha) + abs(beta)) + abs(gamma)) + 1e-06)
        (alpha, beta, gamma) = ((alpha / total), (beta / total), (gamma / total))
        return (alpha, beta, gamma)

    def _get_layer_by_name(self, layer_name: str):
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def register_editing_hooks(self, target_layers: List[str], alpha_personal: float, alpha_neutral: float, alpha_fakeit: float=0.0, last_k_tokens: int=0):
        if ((self.personal_direction is None) or (self.neutral_direction is None)):
            raise ValueError('Direction vectors not loaded')
        new_layers = []
        for layer_name in target_layers:
            if (layer_name not in self._registered_layers):
                new_layers.append(layer_name)
        if (not new_layers):
            return
        if (last_k_tokens == 0):
            last_k_tokens = 16
        if ((alpha_fakeit > 0) and (self.fakeit_direction is not None)):
            total_weight = ((alpha_personal + alpha_neutral) + alpha_fakeit)
            (alpha_personal, alpha_neutral, alpha_fakeit) = ((alpha_personal / total_weight), (alpha_neutral / total_weight), (alpha_fakeit / total_weight))

        def editing_hook(module, inputs, output):
            self._hook_calls_total += 1
            self._hook_calls_in_this_generate += 1
            if self._editing_disabled:
                return output
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
                if (len(shape) == 3):
                    (b, t, h) = shape
                elif (len(shape) == 2):
                    (b, h) = shape
                    t = 1
                else:
                    return output

                def _fit_len(v: torch.Tensor, H: int) -> torch.Tensor:
                    v = v.to(device=device, dtype=dtype).view((- 1))
                    n = v.numel()
                    if (n == H):
                        return v
                    if ((n > 0) and ((H % n) == 0)):
                        return v.repeat((H // n))[:H]
                    if (n > H):
                        return v[:H]
                    out = torch.zeros(H, dtype=dtype, device=device)
                    out[:n] = v
                    return out

                def _project(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    '論文準拠: 投影成分計算 ⟨x,v⟩/||v||² * v'
                    v_norm = (v.norm() + 1e-08)
                    v_normalized = (v / v_norm)
                    dot_product = torch.sum((x * v_normalized), dim=(- 1), keepdim=True)
                    projection = (dot_product * v_normalized)
                    return projection
                pvec = _fit_len(self.personal_direction, h)
                nvec = _fit_len(self.neutral_direction, h)
                if (self.use_causal_constraints and self.temporal_constraint_manager):
                    try:
                        output_tensor = self.temporal_constraint_manager.apply_temporal_constraints_to_editing(output_tensor, user_history=[], current_timestamp=None)
                    except Exception as e:
                        logger.debug(f'Temporal constraint application failed: {e}')
                proj_p = _project(output_tensor, pvec)
                proj_n = _project(output_tensor, nvec)
                if (self.use_causal_constraints and (self.causal_mask is not None)):
                    try:
                        causal_weight = torch.tensor(self.causal_mask.mean(), dtype=dtype, device=device)
                        proj_p = (proj_p * causal_weight)
                        proj_n = (proj_n * causal_weight)
                    except Exception as e:
                        logger.debug(f'Causal mask application failed: {e}')
                effective_alpha_p = float(alpha_personal)
                effective_alpha_n = float(alpha_neutral)
                effective_alpha_f = (float(alpha_fakeit) if (alpha_fakeit > 0) else 0.0)
                if hasattr(self, '_alpha_reduction_factor'):
                    effective_alpha_p *= self._alpha_reduction_factor
                    effective_alpha_n *= self._alpha_reduction_factor
                    effective_alpha_f *= self._alpha_reduction_factor
                base_edit = ((effective_alpha_p * proj_p) - (abs(effective_alpha_n) * proj_n))
                if ((alpha_fakeit > 0) and (self.fakeit_direction is not None)):
                    fvec = _fit_len(self.fakeit_direction, h)
                    proj_f = _project(output_tensor, fvec)
                    base_edit = (base_edit + (effective_alpha_f * proj_f))
                edit = base_edit
                if ((len(shape) == 3) and (last_k_tokens > 0)):
                    k = min(last_k_tokens, t)
                    mask = torch.zeros_like(edit)
                    mask[:, -k:, :] = 1
                    edit = (edit * mask)
                edited = (output_tensor + edit)
                if self._diag_enable:
                    try:
                        edit_norm = edit.norm().item()
                        output_norm = output_tensor.norm().item()
                        ratio = (edit_norm / (output_norm + 1e-09))
                        self._edit_ratios.append(ratio)
                        if self._diag_token_level:
                            if (alpha_fakeit > 0):
                                logger.info(f'[DIAG] edit_ratio={ratio:.4e} | α={alpha_personal:.3g} β={alpha_neutral:.3g} γ={alpha_fakeit:.3g}')
                            else:
                                logger.info(f'[DIAG] edit_ratio={ratio:.4e} | alpha_p={alpha_personal:.3g} alpha_n={alpha_neutral:.3g}')
                        weak_threshold = 0.001
                        excessive_threshold = 0.25
                        if (ratio < weak_threshold):
                            self._weak_streak += 1
                            if self._diag_token_level:
                                logger.warning(f'[WARN] weak_edit detected: ratio={ratio:.4e} < {weak_threshold}')
                            if (self._weak_streak >= 2):
                                self._editing_disabled = True
                                if self._diag_token_level:
                                    logger.info(f'[EARLY-STOP] Disabling editing due to {self._weak_streak} consecutive weak edits')
                        elif (ratio > excessive_threshold):
                            if self._diag_token_level:
                                logger.warning(f'[WARN] excessive_edit detected: ratio={ratio:.4e} > {excessive_threshold} - reducing α')
                            if hasattr(self, '_alpha_reduction_factor'):
                                self._alpha_reduction_factor *= 0.8
                            else:
                                self._alpha_reduction_factor = 0.8
                            self._weak_streak = 0
                        else:
                            self._weak_streak = 0
                            if hasattr(self, '_alpha_reduction_factor'):
                                self._alpha_reduction_factor = min((self._alpha_reduction_factor * 1.05), 1.0)
                    except Exception as e:
                        logger.warning(f'Error computing edit diagnostics: {e}')
                if isinstance(output, tuple):
                    return ((edited,) + extra)
                return edited
            except Exception as e:
                logger.warning(f'Error in editing hook: {e}. Returning original.')
                return output
        for layer_name in new_layers:
            try:
                layer = self._get_layer_by_name(layer_name)
                hook = layer.register_forward_hook(editing_hook)
                self._hook_handles.append(hook)
                self._registered_layers.add(layer_name)
                logger.info(f'Registered editing hook on {layer_name}')
            except AttributeError:
                logger.warning(f"Layer name invalid: '{layer_name}' - skipping this layer")
                continue

    def log_fakeit_execution(self, user_id: str, pairs_count: int, insights_count: int, svd_success: bool, ccs_success: bool):
        'Log Fake it pipeline execution summary (single line format)'
        logger = logging.getLogger(__name__)
        logger.info(f'[DIAG] FAKEIT user={user_id} pairs={pairs_count} insights={insights_count} svd_ok={svd_success} ccs_ok={ccs_success}')

    def remove_editing_hooks(self):
        removed_count = len(self._hook_handles)
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        self._registered_layers.clear()
        self._editing_disabled = False
        self._weak_streak = 0
        if (removed_count > 0):
            logger.info(f'Removed editing hooks: {removed_count}')

    def _edit_delta(self, dv, alpha_personal, alpha_general):
        """
        Compute edit delta vector: ap*v_personal + ag*v_general
        
        Args:
            dv: Direction vectors dict with v_personal, v_general keys
            alpha_personal: Personal direction scaling factor
            alpha_general: General direction scaling factor
            
        Returns:
            torch.Tensor: Edit delta vector [H] with proper dtype/device
        """
        import torch
        
        v_p = dv["v_personal"]
        v_g = dv["v_general"]
        
        # Ensure tensors are on the correct device and dtype
        if not isinstance(v_p, torch.Tensor):
            v_p = torch.tensor(v_p, device=self.device)
        else:
            v_p = v_p.to(self.device)
        if not isinstance(v_g, torch.Tensor):
            v_g = torch.tensor(v_g, device=self.device)
        else:
            v_g = v_g.to(self.device)
        
        # Compute weighted combination
        delta_vec = float(alpha_personal) * v_p + float(alpha_general) * v_g
        
        # Ensure delta_vec is on the correct device and dtype as the model
        delta_vec = delta_vec.to(device=self.device)
        if hasattr(self.model, 'dtype'):
            delta_vec = delta_vec.to(dtype=self.model.dtype)
        
        return delta_vec

    def _layer_injection_ctx(self, target_layers, delta_vec, gate_applied, prompt_len=None):
        """
        Context manager for layer injection using forward pre-hooks
        
        Args:
            target_layers: List of layer names/indices to inject into
            delta_vec: Edit vector to inject [H]
            gate_applied: Whether to actually apply the edit
            prompt_len: Length of prompt tokens (to separate prompt/generation phases)
        """
        class LayerInjectionContext:
            def __init__(self, editor, target_layers, delta_vec, gate_applied, prompt_len):
                self.editor = editor
                self.target_layers = target_layers or []
                self.delta_vec = delta_vec
                self.gate_applied = gate_applied
                self.prompt_len = prompt_len
                self.hook_handles = []
                self.fire_count = 0  # count hook firings for visibility

            def _resolve_layer(self, ref):
                """
                Resolve layer reference to a module and its index.
                ref may be int or str like "24", "layers.24", "model.layers.24".
                Returns (module, idx) or (None, None).
                """
                try:
                    base = getattr(self.editor.model, "model", self.editor.model)
                    layers = getattr(base, "layers", None)
                    if layers is None:
                        return (None, None)
                    n = len(layers)
                    # int index
                    if isinstance(ref, int):
                        idx = ref if ref >= 0 else (n + ref)
                        if 0 <= idx < n:
                            return (layers[idx], idx)
                        return (None, None)
                    # string index
                    s = str(ref)
                    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                        idx = int(s)
                        idx = idx if idx >= 0 else (n + idx)
                        if 0 <= idx < n:
                            return (layers[idx], idx)
                        return (None, None)
                    import re
                    m = re.search(r"(?:^|\.)layers\.(\-?\d+)$", s)
                    if m:
                        idx = int(m.group(1))
                        idx = idx if idx >= 0 else (n + idx)
                        if 0 <= idx < n:
                            return (layers[idx], idx)
                    return (None, None)
                except Exception:
                    return (None, None)
                
            def __enter__(self):
                if not self.gate_applied or not self.target_layers:
                    return self
                
                def hook_fn(module, inputs):
                    if not inputs:
                        return inputs
                    hs = inputs[0]             # [B, T, H]
                    rest = inputs[1:] if len(inputs) > 1 else ()
                    
                    # Check if we're still in prompt phase - if so, don't apply edits
                    if self.prompt_len is not None and hs.shape[1] <= int(self.prompt_len):
                        return (hs,) + rest  # Still in prompt phase → no edits
                    
                    if self.gate_applied and self.delta_vec is not None:
                        try:
                            # ★ ここで dtype/device を層入力に合わせる（安全）
                            delta = self.delta_vec.to(hs.device, dtype=hs.dtype)
                            # 生成トークンに加算 (only during generation phase)
                            hs = hs + delta.view(1, 1, -1)
                            self.fire_count += 1
                        except Exception:
                            # エラー時はそのまま通す
                            pass

                    return (hs,) + rest
                
                # Register hooks on target layers (int/str reference supported)
                resolved = []
                for ref in self.target_layers:
                    mod, idx = self._resolve_layer(ref)
                    if mod is None:
                        logger.warning(f"[inject] Skip invalid layer ref: {ref}")
                        continue
                    try:
                        handle = mod.register_forward_pre_hook(hook_fn)
                        self.hook_handles.append(handle)
                        resolved.append(idx)
                    except Exception as e:
                        logger.warning(f"[inject] Failed to register hook on layer {idx}: {e}")
                
                logger.info(f"[inject] Registered {len(self.hook_handles)} hooks on layers: {resolved}")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Remove all hooks
                for handle in self.hook_handles:
                    handle.remove()
                logger.info(f"[inject] Removed {len(self.hook_handles)} hooks; fired={self.fire_count}")
                self.hook_handles.clear()
        
        return LayerInjectionContext(self, target_layers, delta_vec, gate_applied, prompt_len)

    def _avg_logprob_from_scores(self, input_ids, outputs, tokenizer):
        """
        Compute average log-prob of generated tokens from HF output_scores.
        
        Args:
            input_ids: Original input token IDs
            outputs: HF generate output with scores and sequences
            tokenizer: Tokenizer instance
            
        Returns:
            float or None if scores unavailable
        """
        # outputs.scores: List[Tensor[B, V]], len = #new_tokens
        # outputs.sequences: Tensor[B, prompt+new]
        if not (hasattr(outputs, "scores") and hasattr(outputs, "sequences")):
            return None
            
        scores = outputs.scores
        if not scores:
            return None
            
        seq = outputs.sequences[0]
        new_len = len(scores)
        prompt_len = seq.size(0) - new_len
        
        logps = []
        for t, step_scores in enumerate(scores):
            y = seq[prompt_len + t].item()
            try:
                logp = torch.log_softmax(step_scores[0], dim=-1)[y]
                logps.append(float(logp.item()))
            except (IndexError, RuntimeError):
                # Skip invalid tokens
                continue
                
        return sum(logps) / max(1, len(logps)) if logps else None

    def generate_with_chameleon(self, prompt: str, alpha_personal: float=1.5, alpha_neutral: float=(- 0.8), alpha_fakeit: float=0.0, target_layers: List[str]=None, gen_kwargs: (dict | None)=None, target_edit_ratio: float=0.02, edit_ratio_tolerance: float=0.5, adaptive_alpha: bool=False, last_k_tokens: int=0, **generate_kwargs) -> str:
        """
        Minimal injection path using direction vectors + gate.
        - Merges generation kwargs and cleans sampling flags when greedy.
        - Defaults target layer to last transformer block.
        - Injects a single delta on forward-pre-hook if gate applies.
        """
        # Merge/clean kwargs
        gen_args = {}
        if gen_kwargs:
            gen_args.update(gen_kwargs)
        gen_args.update(generate_kwargs)
        # Greedy default for LaMP-2; remove sampling-only flags if not sampling
        if not gen_args.get('do_sample', False):
            gen_args.pop('temperature', None)
            gen_args.pop('top_p', None)
        # Default pad token id
        pad_id = self.tokenizer.eos_token_id
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0]
        gen_args.setdefault('pad_token_id', pad_id)
        gen_args.setdefault('max_new_tokens', 2)  # Allow 2 tokens for multi-word tags like "sci-fi"
        gen_args.setdefault('min_new_tokens', 1)  # 空出力の物理防止 - guarantee at least 1 token

        # Compute direction vectors strictly from prompt spans
        sample = {'prompt': prompt}
        dv = self._compute_direction_vectors_strict(sample, target_layers=gen_args.get('target_layers'), norm_scale=gen_args.get('norm_scale', 1.0))
        hidden_norm = max(1e-8, float(dv.get('l2_general', 0.0)))

        # alpha_general alias: prefer explicit arg if provided
        ag = gen_args.get('alpha_general', gen_args.get('alpha_neutral', alpha_neutral))

        gate_info = self.compute_gate(hidden_norm,
                                      float(dv.get('l2_personal', 0.0)),
                                      float(dv.get('l2_general', 0.0)),
                                      alpha_personal,
                                      ag,
                                      float(gen_args.get('edit_gate_threshold', 0.022)))

        # Default target layers: last 4 layers if unspecified
        target_layers_eff = gen_args.get('target_layers')
        if not target_layers_eff:
            try:
                base = getattr(self.model, 'model', self.model)
                n_layers = len(getattr(base, 'layers', []))
            except Exception:
                n_layers = 0
            if n_layers >= 4:
                target_layers_eff = [n_layers-4, n_layers-3, n_layers-2, n_layers-1]
            elif n_layers > 0:
                target_layers_eff = [n_layers-1]
            else:
                target_layers_eff = []

        # Build edit delta
        delta = self._edit_delta(dv, alpha_personal, ag)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device, non_blocking=True) for (k, v) in inputs.items()}

        # Exclude custom keys before generate
        filtered = {k: v for (k, v) in gen_args.items() if k not in ['target_layers', 'norm_scale', 'alpha_neutral', 'alpha_general', 'edit_gate_threshold']}

        # Add phrase-level constraints using dataset-driven closed set if provided
        allowed_tags = None
        try:
            allowed_tags = gen_args.get('allowed_tags') or filtered.get('allowed_tags')
        except Exception:
            allowed_tags = None
        lp = filtered.get("logits_processor", None)
        if not isinstance(lp, LogitsProcessorList):
            lp = LogitsProcessorList()
        
        prompt_len = inputs["input_ids"].size(1)
        
        # Prefer phrase-level constraint when allowed_tags provided; otherwise skip constraints
        phrase_proc = None
        if allowed_tags:
            phrase_proc = AllowedPhraseProcessor(self.tokenizer, allowed_tags, prompt_len, self.tokenizer.eos_token_id)
            # Only append if we have at least one phrase sequence
            if getattr(phrase_proc, 'seqs', None):
                lp.append(phrase_proc)
        
        # Optional: Keep AllowedFirstTokenProcessor as fallback 
        # (comment out if TwoStepPrefixProcessor proves sufficient)
        # first_token_proc = AllowedFirstTokenProcessor(self.tokenizer, allowed_tags, prompt_len)
        # if getattr(first_token_proc, "allowed_first_ids", None) is not None and first_token_proc.allowed_first_ids.numel() > 0:
        #     lp.append(first_token_proc)
        
        filtered["logits_processor"] = lp

        # Enhanced observability logging - capture all critical metrics for regression detection
        dbg = {
            "prompt_len": int(inputs["input_ids"].size(1)),
            "prefix_processor_sequences": (len(getattr(phrase_proc, 'seqs', [])) if phrase_proc else 0),
            "prefix_processor_tags": (len(allowed_tags) if allowed_tags else 0),
            "delta_norm": float(delta.norm(p=2).item()) if delta is not None else 0.0,
            "hook_layers": target_layers_eff,
            "gate": gate_info,
        }
        
        # Log warning signals for regression detection  
        if phrase_proc and dbg["prefix_processor_sequences"] == 0:
            logger.warning(f"⚠️ No valid tag sequences for prefix processor - may cause empty generation")
        if not dbg["hook_layers"] and gate_info.get('applied', False):
            logger.warning(f"⚠️ Gate applied but no hook layers - edit will have no effect")
            
        logger.info(f"[edit-debug] {dbg}")

        # Enable score tracking for avg_logprob calculation
        filtered.update({
            "output_scores": True,
            "return_dict_in_generate": True,
        })

        # Adjust max/min new tokens to fit longest phrase tokens if provided
        if allowed_tags:
            try:
                max_phrase_len = 0
                for t in allowed_tags:
                    ids = self.tokenizer.encode(" " + str(t), add_special_tokens=False)
                    max_phrase_len = max(max_phrase_len, len(ids))
                if max_phrase_len > 0:
                    prev = int(filtered.get('max_new_tokens', 0) or 0)
                    if max_phrase_len > prev:
                        filtered['max_new_tokens'] = max_phrase_len
                filtered['min_new_tokens'] = max(1, int(filtered.get('min_new_tokens', 1) or 1))
            except Exception:
                pass

        # Store prompt length for safety (future code paths)
        self._last_prompt_len = prompt_len

        # Force gate applied when threshold explicitly negative
        try:
            if float(gate_info.get('threshold', gen_args.get('edit_gate_threshold', 0.022))) < 0:
                gate_info['applied'] = True
        except Exception:
            pass

        # Run generation with minimal one-shot injection
        bf16_supported = (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
        with self._layer_injection_ctx(target_layers_eff, delta, gate_info.get('applied', False), prompt_len) as _inj_ctx:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16_supported):
                outputs = self.model.generate(**inputs, **filtered)
        logger.info(f"[inject] Summary: layers={target_layers_eff}, gate_applied={gate_info.get('applied', False)}")
        
        # Extract sequences from the generation output dict
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        response = text[len(prompt):].strip()
        
        # Calculate avg_logprob using the improved method
        avg_logprob = self._avg_logprob_from_scores(inputs["input_ids"], outputs, self.tokenizer)
        
        # Post-generation logging for complete observability
        generated_length = len(sequences[0]) - inputs["input_ids"].size(1)
        post_dbg = {
            "generated_length": generated_length,
            "response_length": len(response),
            "response_preview": response[:50] if len(response) > 50 else response,
            "avg_logprob": avg_logprob,
        }
        
        # Critical failure signal: empty output with empty constraints
        if generated_length == 0 and dbg["prefix_processor_sequences"] == 0:
            logger.warning(f"⚠️ REGRESSION: prefix_processor_sequences==0 AND generated_length==0")
            
        logger.info(f"[edit-result] {post_dbg}")
        return response

    def compute_direction_vectors(self, sample, target_layers=None, norm_scale=1.0):
        """
        Robust direction vectors:
        1) 既存実装があればそれを優先（_compute_direction_vectors_strict）
        2) 失敗時は hidden_states の単純平均（persona span vs movie span）にフォールバック
        戻り値 keys: v_personal, v_general, l2_personal, l2_general, cos_theta
        """
        import torch
        import torch.nn.functional as F
        try:
            return self._compute_direction_vectors_strict(sample, target_layers, norm_scale)
        except Exception:
            prompt = sample.get("prompt") or sample.get("text") or ""
            if not prompt:
                raise RuntimeError("No prompt text for direction vectors.")
            p_idx = prompt.find("User's movie preferences:")
            m_idx = prompt.find("Movie:")
            if p_idx == -1 or m_idx == -1:
                raise RuntimeError("Cannot locate persona/movie spans in prompt.")
            persona_text = prompt[p_idx:m_idx]
            movie_text = prompt[m_idx:]

            def _embed(text: str) -> torch.Tensor:
                toks = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outs = self.model(**toks, output_hidden_states=True, use_cache=False)
                vec = outs.hidden_states[-1].mean(dim=1).squeeze(0)  # [H]
                return vec

            v_p, v_g = _embed(persona_text), _embed(movie_text)
            if norm_scale is not None:
                v_p = v_p / (v_p.norm(p=2) + 1e-8) * float(norm_scale)
                v_g = v_g / (v_g.norm(p=2) + 1e-8) * float(norm_scale)
            cos_th = float(torch.cosine_similarity(v_p.unsqueeze(0), v_g.unsqueeze(0)).item())
            result = {
                "v_personal": v_p, "v_general": v_g,
                "l2_personal": float(v_p.norm(p=2).item()),
                "l2_general": float(v_g.norm(p=2).item()),
                "cos_theta": cos_th,
            }
            # Store direction vectors for hook usage
            self.personal_direction = v_p
            self.neutral_direction = v_g  # Using general as neutral
            return result
    
    def _compute_direction_vectors_strict(self, sample, target_layers=None, norm_scale=1.0):
        """
        Strict direction vector computation using prompt-based span extraction
        
        Args:
            sample: Dict with 'prompt' field containing full inference prompt
            target_layers: List of layer indices or None for last layer
            norm_scale: Normalization scale factor
            
        Returns:
            Dict with v_personal, v_general, l2_personal, l2_general, cos_theta
        """
        import torch
        import torch.nn.functional as F
        
        prompt = sample.get("prompt") or ""
        if not prompt:
            raise RuntimeError("sample['prompt'] is empty")

        p_idx = prompt.find("User's movie preferences:")
        m_idx = prompt.find("Movie:")
        if p_idx == -1 or m_idx == -1 or m_idx <= p_idx:
            raise RuntimeError("Cannot locate persona/movie spans in prompt.")
        
        persona_text = prompt[p_idx:m_idx]
        movie_text = prompt[m_idx:]

        def _embed(text: str) -> torch.Tensor:
            tok = self.tokenizer(text, return_tensors="pt").to(self.device)
            self.model.eval()
            with torch.no_grad():
                outs = self.model(**tok, output_hidden_states=True, return_dict=True, use_cache=False)
            hs_all = outs.hidden_states  # tuple of [B,T,H]
            
            # choose layers
            if target_layers:
                sel = [i if i >= 0 else len(hs_all)+i for i in target_layers]
                sel = [i for i in sel if 0 <= i < len(hs_all)]
                if not sel:
                    raise RuntimeError(f"Invalid target_layers: {target_layers}")
                hs = torch.stack([hs_all[i] for i in sel]).mean(dim=0)  # [B,T,H]
            else:
                hs = hs_all[-1]  # last layer [B,T,H]
            vec = hs.mean(dim=1).squeeze(0)  # [H]
            return vec

        v_p = _embed(persona_text)
        v_g = _embed(movie_text)

        # normalize
        l2p = v_p.norm(p=2)
        l2g = v_g.norm(p=2)
        v_p = v_p / (l2p + 1e-8)
        v_g = v_g / (l2g + 1e-8)
        if norm_scale is not None:
            v_p = v_p * float(norm_scale)
            v_g = v_g * float(norm_scale)
        
        l2p = float(v_p.norm(p=2).item())
        l2g = float(v_g.norm(p=2).item())
        cos_th = float(F.cosine_similarity(v_p.unsqueeze(0), v_g.unsqueeze(0)).item())
        
        result = {
            "v_personal": v_p, "v_general": v_g,
            "l2_personal": l2p, "l2_general": l2g, 
            "cos_theta": cos_th
        }
        # Store direction vectors for hook usage
        self.personal_direction = v_p
        self.neutral_direction = v_g  # Using general as neutral
        return result
    
    def _strict_direction_vectors(self, sample, target_layers=None, norm_scale=1.0):
        """Strict direction vector computation using SVD"""
        profile = sample['profile']
        question = sample['question']
        
        # Build personal and neutral prompts
        personal_items = [item for item in profile if item.get('tag') and item.get('description')]
        
        if len(personal_items) < 2:
            raise ValueError("Insufficient personal items for direction computation")
        
        # Create positive and negative examples
        personal_texts = []
        neutral_texts = []
        
        for item in personal_items[:3]:  # Use first 3 for stability
            personal_texts.append(f"I love {item['tag']} movies like: {item['description']}")
            neutral_texts.append(f"This is a {item['tag']} movie: {item['description']}")
        
        # Generate embeddings
        personal_embeddings = self._get_text_embeddings(personal_texts)
        neutral_embeddings = self._get_text_embeddings(neutral_texts)
        
        # Compute direction vectors using SVD
        personal_mean = torch.mean(personal_embeddings, dim=0)
        neutral_mean = torch.mean(neutral_embeddings, dim=0)
        
        # Direction = normalized difference
        direction_p = F.normalize(personal_mean - neutral_mean, p=2, dim=0) * norm_scale
        direction_n = F.normalize(neutral_mean - personal_mean, p=2, dim=0) * norm_scale
        
        return direction_p, direction_n
    
    def _fallback_span_averaging(self, sample, target_layers=None, norm_scale=1.0):
        """Fallback: Simple span averaging on hidden states"""
        try:
            profile = sample['profile']
            question = sample['question']
            
            # Create simple personal preference text
            tags = [item.get('tag', '') for item in profile[:3] if item.get('tag')]
            if not tags:
                tags = ['action', 'comedy', 'drama']  # Safe fallback
                
            personal_text = f"I prefer {', '.join(tags)} movies"
            neutral_text = "This is a movie"
            
            # Get embeddings using simple encoding
            personal_emb = self._encode_text_to_hidden(personal_text)
            neutral_emb = self._encode_text_to_hidden(neutral_text)
            
            # Simple direction computation
            direction_p = F.normalize(personal_emb - neutral_emb, p=2, dim=0) * norm_scale
            direction_n = F.normalize(neutral_emb - personal_emb, p=2, dim=0) * norm_scale
            
            logger.info(f"Fallback direction vectors computed: P={direction_p.shape}, N={direction_n.shape}")
            return direction_p, direction_n
            
        except Exception as e:
            logger.error(f"Fallback direction computation also failed: {e}")
            # Last resort: return zero vectors
            hidden_size = getattr(self.model.config, 'hidden_size', 3072)
            zero_p = torch.zeros(hidden_size, device=self.device) * norm_scale
            zero_n = torch.zeros(hidden_size, device=self.device) * norm_scale
            logger.warning("Returning zero direction vectors as last resort")
            return zero_p, zero_n
    
    def _get_text_embeddings(self, texts):
        """Extract embeddings from text list"""
        embeddings = []
        for text in texts:
            emb = self._encode_text_to_hidden(text)
            embeddings.append(emb)
        return torch.stack(embeddings)
    
    def _encode_text_to_hidden(self, text):
        """Encode single text to hidden representation"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean of last hidden layer
            hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            return hidden

    def compute_gate(self, hidden_norm, l2_personal, l2_general,
                     alpha_personal, alpha_general, edit_gate_threshold):
        denom = max(float(hidden_norm), 1e-8)
        edit_mag = abs(float(alpha_personal)) * float(l2_personal) + \
                   abs(float(alpha_general))  * float(l2_general)
        gate_value = edit_mag / denom
        return {"gate_value": float(gate_value),
                "applied": bool(gate_value >= float(edit_gate_threshold)),
                "threshold": float(edit_gate_threshold)}
    
    def compute_gate_value(self, direction_p, direction_n, threshold=0.022):
        """
        Explicit gate computation method
        
        Computes gate value based on direction vector properties
        instead of returning fixed 0.0
        """
        if direction_p is None or direction_n is None:
            return 0.0
            
        try:
            # Compute cosine similarity
            p_norm = F.normalize(direction_p, p=2, dim=0)
            n_norm = F.normalize(direction_n, p=2, dim=0)
            cosine_sim = torch.dot(p_norm, n_norm).item()
            
            # Compute magnitude ratio  
            p_magnitude = torch.norm(direction_p).item()
            n_magnitude = torch.norm(direction_n).item()
            magnitude_ratio = p_magnitude / max(n_magnitude, 1e-8)
            
            # Gate value based on orthogonality and magnitude balance
            orthogonality = 1.0 - abs(cosine_sim)  # Higher when vectors are orthogonal
            balance = min(magnitude_ratio, 1.0/magnitude_ratio)  # Higher when magnitudes are balanced
            
            gate_value = orthogonality * balance
            
            logger.debug(f"Gate computation: cos={cosine_sim:.4f}, p_mag={p_magnitude:.4f}, n_mag={n_magnitude:.4f}, gate={gate_value:.4f}")
            
            return gate_value
            
        except Exception as e:
            logger.warning(f"Gate computation failed: {e}")
            return 0.0

    def set_diagnostics_config(self, token_level: bool=False, kl_computation: bool=False):
        'Configure diagnostic verbosity levels for performance optimization'
        self._diag_token_level = token_level
        self._diag_kl = kl_computation

    def clear_tokenization_cache(self):
        'Clear tokenization cache to free memory'
        self._tokenization_cache.clear()

    def _should_apply_gate(self, gate_value: float, threshold: float) -> bool:
        try:
            return float(gate_value) >= float(threshold)
        except Exception:
            return False

    def _compute_gate_value(self, dv) -> float:
        # Use existing persona/general vec from direction vectors
        pv = dv.get("v_personal", None)
        if pv is None:
            return 0.0
        return float(torch.linalg.vector_norm(pv).item())

    def _compute_delta(self, hidden_states, dv, alpha_personal, alpha_neutral, norm_scale):
        # Compute delta using existing direction vectors
        pv = dv.get("v_personal", None)
        gv = dv.get("v_general", None)
        delta = 0.0
        if pv is not None:
            delta = alpha_personal * (pv / (torch.linalg.vector_norm(pv) + 1e-12)) * norm_scale
        if gv is not None:
            delta = delta + alpha_neutral * (gv / (torch.linalg.vector_norm(gv) + 1e-12)) * norm_scale
        # Broadcast delta to match hidden_states shape
        if isinstance(hidden_states, torch.Tensor) and isinstance(delta, torch.Tensor):
            # Expand delta to match hidden_states dimensions
            while delta.dim() < hidden_states.dim():
                delta = delta.unsqueeze(0)
            try:
                # Broadcast to the last dimension (hidden_size)
                if delta.size(-1) == hidden_states.size(-1):
                    return hidden_states + delta.expand_as(hidden_states)
                else:
                    return hidden_states
            except Exception:
                return hidden_states
        return hidden_states

    def _compute_gate_value_simple(self):
        """Simple gate computation using current direction vectors."""
        if hasattr(self, 'personal_direction') and self.personal_direction is not None:
            if hasattr(self, 'neutral_direction') and self.neutral_direction is not None:
                # Simple cosine similarity as gate value
                cos_sim = torch.cosine_similarity(
                    self.personal_direction.flatten(), 
                    self.neutral_direction.flatten(), 
                    dim=0
                )
                return float(cos_sim) + 1.0  # Ensure positive gate value
        return 1.0  # Default gate value

    @contextmanager
    def cham_context(self, *, alpha_personal, alpha_neutral, alpha_fakeit,
                     target_layers, norm_scale, gate):
        """
        Context manager for applying Chameleon edits during scoring.
        Temporarily registers forward hooks on specified layers.
        Returns context object with 'calls' attribute for observability.
        """
        import torch
        
        # Context object to track hook firing
        class HookContext:
            def __init__(self):
                self.calls = 0  # Hook firing counter
        
        context = HookContext()
        handles = []
        
        # Resolve direction vectors - use current instance variables
        personal_dir = getattr(self, 'personal_direction', None)
        neutral_dir = getattr(self, 'neutral_direction', None)
        
        if personal_dir is None or neutral_dir is None:
            print(f"[DEBUG] Missing direction vectors: personal={personal_dir is not None}, neutral={neutral_dir is not None}")
            yield context  # Return context even if no editing
            return

        def _hook_fn(module, inputs, outputs):
            context.calls += 1  # Track hook firing
            
            # Compute gate value using current direction vectors
            if hasattr(self, '_compute_gate_value_simple'):
                gate_value = self._compute_gate_value_simple()
            else:
                gate_value = gate + 0.1  # Simple fallback to ensure gate passes
                
            if gate_value < gate:
                return outputs
                
            # Handle HF layer outputs (tuple or tensor)
            if isinstance(outputs, tuple):
                hs = outputs[0]  # Hidden states
                # Apply direction editing
                if personal_dir is not None:
                    print(f"[DEBUG] hs shape: {hs.shape}, personal_dir shape: {personal_dir.shape}")
                    
                    # Handle dimension mismatch
                    if personal_dir.shape[-1] != hs.shape[-1]:
                        if personal_dir.shape[-1] < hs.shape[-1]:
                            # Pad direction vector
                            pad_size = hs.shape[-1] - personal_dir.shape[-1]
                            personal_dir_padded = torch.cat([personal_dir, torch.zeros(pad_size, device=personal_dir.device)], dim=0)
                            print(f"[DEBUG] Padded personal direction from {personal_dir.shape} to {personal_dir_padded.shape}")
                        else:
                            # Truncate direction vector
                            personal_dir_padded = personal_dir[:hs.shape[-1]]
                            print(f"[DEBUG] Truncated personal direction from {personal_dir.shape} to {personal_dir_padded.shape}")
                    else:
                        personal_dir_padded = personal_dir
                    
                    delta = alpha_personal * personal_dir_padded.to(hs.device)
                    
                    if neutral_dir is not None:
                        # Handle neutral direction similarly
                        if neutral_dir.shape[-1] != hs.shape[-1]:
                            if neutral_dir.shape[-1] < hs.shape[-1]:
                                pad_size = hs.shape[-1] - neutral_dir.shape[-1]
                                neutral_dir_padded = torch.cat([neutral_dir, torch.zeros(pad_size, device=neutral_dir.device)], dim=0)
                            else:
                                neutral_dir_padded = neutral_dir[:hs.shape[-1]]
                        else:
                            neutral_dir_padded = neutral_dir
                        delta += alpha_neutral * neutral_dir_padded.to(hs.device)
                    
                    # Apply normalization
                    if norm_scale and torch.norm(delta) > 0:
                        delta = delta / torch.norm(delta) * norm_scale
                    hs_edited = hs + delta
                    print(f"[DEBUG] Applied editing: delta norm = {torch.norm(delta):.4f}")
                else:
                    hs_edited = hs
                return (hs_edited,) + outputs[1:]
            elif isinstance(outputs, torch.Tensor):
                # Apply direction editing to tensor directly
                if personal_dir is not None:
                    delta = alpha_personal * personal_dir.to(outputs.device)
                    if neutral_dir is not None:
                        delta += alpha_neutral * neutral_dir.to(outputs.device)
                    # Apply normalization
                    if norm_scale and torch.norm(delta) > 0:
                        delta = delta / torch.norm(delta) * norm_scale
                    return outputs + delta
            return outputs

        try:
            # Resolve target layers with multiple fallback paths
            target_layers = target_layers or [24, 25, 26, 27]  # Default for 28-layer model
            # Ensure target_layers are integers
            target_layers = [int(li) for li in target_layers]
            
            print(f"[DEBUG] Trying to register hooks on layers: {target_layers}")
            print(f"[DEBUG] Model structure: {type(self.model).__name__}")
            print(f"[DEBUG] Model has 'model' attr: {hasattr(self.model, 'model')}")
            if hasattr(self.model, 'model'):
                print(f"[DEBUG] model.model has 'layers': {hasattr(self.model.model, 'layers')}")
                if hasattr(self.model.model, 'layers'):
                    print(f"[DEBUG] Total layers: {len(self.model.model.layers)}")
            
            for li in target_layers:
                layer = None
                # Try multiple layer access patterns
                try:
                    # Pattern 1: model.model.layers[i] (Llama-style)
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                        if li < len(self.model.model.layers):
                            layer = self.model.model.layers[li]
                            print(f"[DEBUG] Found layer {li} via model.model.layers")
                        else:
                            print(f"[DEBUG] Layer {li} out of range (max: {len(self.model.model.layers)-1})")
                except (IndexError, AttributeError) as e:
                    print(f"[DEBUG] Pattern 1 failed for layer {li}: {e}")
                    try:
                        # Pattern 2: model.layers[i] (direct access)
                        if hasattr(self.model, 'layers'):
                            if li < len(self.model.layers):
                                layer = self.model.layers[li]
                                print(f"[DEBUG] Found layer {li} via model.layers")
                    except (IndexError, AttributeError) as e2:
                        print(f"[DEBUG] Pattern 2 failed for layer {li}: {e2}")
                        try:
                            # Pattern 3: model.transformer.h[i] (GPT-style)
                            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                                if li < len(self.model.transformer.h):
                                    layer = self.model.transformer.h[li]
                                    print(f"[DEBUG] Found layer {li} via model.transformer.h")
                        except (IndexError, AttributeError) as e3:
                            print(f"[DEBUG] Pattern 3 failed for layer {li}: {e3}")
                
                if layer is not None:
                    h = layer.register_forward_hook(_hook_fn)
                    handles.append(h)
                    print(f"[DEBUG] Successfully registered hook on layer {li}")
                else:
                    print(f"[DEBUG] Could not access layer {li} via any pattern")
                    
            print(f"[DEBUG] Total hooks registered: {len(handles)}")
            yield context
        finally:
            for h in handles:
                try: 
                    h.remove()
                except Exception: 
                    pass
            print(f"[DEBUG] Removed {len(handles)} hooks (total calls: {context.calls})")

class EvaluationEngine():
    'ベースライン vs Chameleon比較評価エンジン'

    def __init__(self, chameleon_editor: ChameleonEditor):
        self.chameleon_editor = chameleon_editor
        self.parent = None

    def calculate_exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        if (not ground_truths):
            return 0.0
        return (sum(((p.strip().lower() == g.strip().lower()) for (p, g) in zip(predictions, ground_truths))) / len(ground_truths))

    def calculate_bleu_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        if ((not NLTK_AVAILABLE) or (not ground_truths)):
            return 0.0
        smoothing = SmoothingFunction().method1
        scores = []
        for (pred, truth) in zip(predictions, ground_truths):
            pt = pred.strip().lower().split()
            tt = [truth.strip().lower().split()]
            if (pt and tt[0]):
                scores.append(sentence_bleu(tt, pt, smoothing_function=smoothing))
        return (float(np.mean(scores)) if scores else 0.0)

    # LaMP-2 tag normalization
    ALLOWED_TAGS = {
        "action","adventure","animation","comedy","crime","drama","fantasy","horror",
        "mystery","romance","sci-fi","thriller","war","western","family"
    }

    # Runtime-override allowed tags (dataset-driven). None means use defaults.
    _allowed_tags_runtime = None

    def set_allowed_tags(self, tags):
        """Set dynamic allowed tag set (case-insensitive). Pass falsy to reset."""
        try:
            if not tags:
                self._allowed_tags_runtime = None
                return
            self._allowed_tags_runtime = {str(t).strip().lower() for t in tags if str(t).strip()}
        except Exception as e:
            logger.warning(f"set_allowed_tags failed: {e}")
            self._allowed_tags_runtime = None

    def get_allowed_tags(self):
        """Get active allowed tag set (runtime if set, else static defaults)."""
        if self._allowed_tags_runtime:
            return set(self._allowed_tags_runtime)
        return set(self.ALLOWED_TAGS)

    def _normalize_tag(self, text: str) -> str:
        import re
        if not text:
            return "unknown"
        
        # Clean text: strip whitespace and punctuation
        text = text.strip().lower()
        text = text.strip(" \n\r\t.,:;!?\"'()[]{}")
        # Collapse multiple spaces
        text = " ".join(text.split())
        
        # Enhanced sci-fi recovery - handle 2-token concatenation patterns
        toks = text.strip().lower().split()
        
        # Check token pairs for sci-fi patterns
        if len(toks) >= 2 and toks[:2] in [["sci", "fi"], ["sci", "-"], ["sci-fi"]]:
            return "sci-fi"
        
        # Check space-removed variants
        no_space = text.replace(" ", "").replace("_", "").replace("—", "-")
        if no_space in {"scifi", "sci-fi", "sci—fi", "sci_fi"}:
            return "sci-fi"
            
        # Original pattern matching for fallback
        if any(token in text for token in ["sci", "-", "fi"]):
            # Look for various sci-fi patterns
            if re.search(r"sci[-\s]*fi", text) or ("sci" in text and any(x in text for x in ["-", "fi"])):
                return "sci-fi"
        
        # Direct phrase match with active allowed set
        active_allowed = self.get_allowed_tags()
        if text in active_allowed:
            return text
        # Hyphen-space variation
        if '-' in text and text.replace('-', ' ') in active_allowed:
            return text.replace('-', ' ')

        # Extract words and hyphenated terms
        words = re.findall(r"[A-Za-z\-]+", text)
        
        # Direct match with allowed tags (case-insensitive)
        allowed_tags_lower = {tag.lower(): tag for tag in active_allowed}
        for w in words:
            w_clean = w.lower()
            if w_clean in allowed_tags_lower:
                return allowed_tags_lower[w_clean]
        
        # Handle synonym mapping
        synonyms = {
            "scifi": "sci-fi",
            "science-fiction": "sci-fi", 
            "sciencefiction": "sci-fi",
        }
        for w in words:
            w_clean = w.lower()
            if w_clean in synonyms:
                return synonyms[w_clean]
                
        # If no match found, return first word or "unknown"
        return (words[0] if words else "unknown")

    def _finalize_metrics(self, predictions: List[str], ground_truth: Dict[(str, str)], start_time: float, name: str) -> EvaluationResult:
        matched_truths: List[str] = []
        matched_preds: List[str] = []
        for (p, sample) in zip(predictions, self.parent.test_samples_cache):
            sid = str(sample['id'])
            if (sid in ground_truth):
                matched_truths.append(ground_truth[sid])
                matched_preds.append(p)
        inference_time = (time.time() - start_time)
        correct = sum((int((p == g)) for (p, g) in zip(matched_preds, matched_truths)))
        acc = ((correct / len(matched_truths)) if matched_truths else 0.0)
        exact = self.calculate_exact_match(matched_preds, matched_truths)
        bleu = self.calculate_bleu_score(matched_preds, matched_truths)
        prec = rec = f1 = acc
        return EvaluationResult(method_name=name, accuracy=acc, exact_match=exact, bleu_score=bleu, precision=prec, recall=rec, f1_score=f1, inference_time=inference_time, total_samples=len(matched_truths), correct_predictions=correct, predictions=matched_preds, ground_truths=matched_truths)

    def evaluate_baseline(self, test_samples: List[Dict], ground_truth: Dict[(str, str)]) -> EvaluationResult:
        logger.info('Starting baseline evaluation...')
        predictions: List[str] = []
        start = time.time()
        for (i, sample) in enumerate(test_samples):
            logger.info(f'Baseline progress: {(i + 1)}/{len(test_samples)}')
            prompt = f'''Given the following movie description, provide a single word tag that best describes the movie:

Movie: {sample['input']}

Tag:'''
            inputs = self.chameleon_editor.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.chameleon_editor.device) for (k, v) in inputs.items()}
            # Minimal, robust generation kwargs
            pad_id = self.chameleon_editor.tokenizer.eos_token_id
            if isinstance(pad_id, (list, tuple)):
                pad_id = pad_id[0]
            gen = dict(max_new_tokens=10, pad_token_id=pad_id)
            with torch.no_grad():
                outputs = self.chameleon_editor.model.generate(**inputs, **gen)
            gen = self.chameleon_editor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = self._normalize_tag(gen[len(prompt):])
            predictions.append(pred)
        return self._finalize_metrics(predictions, ground_truth, start, 'Baseline')

    def evaluate_chameleon(self, test_samples: List[Dict], ground_truth: Dict[(str, str)], alpha_personal: float=1.5, alpha_neutral: float=(- 0.8), alpha_fakeit: float=0.0, target_layers: List[str]=None, name: str='Chameleon', target_edit_ratio: float=0.02, edit_ratio_tolerance: float=0.5, adaptive_alpha: bool=False, last_k_tokens: int=0) -> EvaluationResult:
        logger.info(f'Starting Chameleon evaluation (α_p={alpha_personal}, α_n={alpha_neutral})...')
        predictions: List[str] = []
        suggested_alphas: List[float] = []
        start = time.time()
        for (i, sample) in enumerate(test_samples):
            logger.info(f'Chameleon progress: {(i + 1)}/{len(test_samples)}')
            prompt = f'''Given the following movie description, provide a single word tag that best describes the movie:

Movie: {sample['input']}

Tag:'''
            resp = self.chameleon_editor.generate_with_chameleon(prompt=prompt, alpha_personal=alpha_personal, alpha_neutral=alpha_neutral, alpha_fakeit=alpha_fakeit, target_layers=target_layers, gen_kwargs=self.parent.gen_kwargs, target_edit_ratio=target_edit_ratio, edit_ratio_tolerance=edit_ratio_tolerance, adaptive_alpha=adaptive_alpha, last_k_tokens=last_k_tokens)
            pred = self._normalize_tag(resp)
            predictions.append(pred)
            if (hasattr(self.chameleon_editor, '_edit_ratios') and self.chameleon_editor._edit_ratios):
                self.parent.evaluation_diagnostics['edit_ratios'].extend(self.chameleon_editor._edit_ratios.copy())
            hook_calls_this_gen = getattr(self.chameleon_editor, '_hook_calls_in_this_generate', 0)
            self.parent.evaluation_diagnostics['hook_calls'].append(hook_calls_this_gen)
            suggested_alpha = getattr(self.chameleon_editor, '_suggested_alpha', alpha_personal)
            suggested_alphas.append(suggested_alpha)
            self.chameleon_editor._hook_calls_in_this_generate = 0
            avg_edit_ratio = (np.mean(self.chameleon_editor._edit_ratios) if (hasattr(self.chameleon_editor, '_edit_ratios') and self.chameleon_editor._edit_ratios) else 0.0)
            logger.info(f'Sample {(i + 1)}: hook_calls={hook_calls_this_gen}, avg_edit_ratio={avg_edit_ratio:.4e}, suggested_alpha={suggested_alpha:.3f}')
        self.parent.evaluation_diagnostics['suggested_alphas'] = suggested_alphas
        return self._finalize_metrics(predictions, ground_truth, start, name)

    def _estimate_uncertainty(self, input_text: str, user_profile: List[Dict]) -> float:
        '不確実性推定: エントロピーベース計算'
        try:
            tags_list = 'sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story'
            simple_prompt = f'''Movie: {input_text[:150]}
From tags: [{tags_list}]
Most likely tag:'''
            inputs = self.chameleon_editor.tokenizer(simple_prompt, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.chameleon_editor.device) for (k, v) in inputs.items()}
            with torch.no_grad():
                outputs = self.chameleon_editor.model(**inputs)
                logits = outputs.logits[0, -1, :]
                T = 1.5
                probs = F.softmax((logits / T), dim=(- 1))
                entropy = (- (probs * torch.log((probs + 1e-08))).sum().item())
                normalized_uncertainty = min((entropy / 10.0), 1.0)
                return normalized_uncertainty
        except Exception as e:
            logger.debug(f'Uncertainty estimation failed: {e}')
            return 0.5

    def evaluate_chameleon_full(self, test_samples: List[Dict], ground_truth: Dict[(str, str)], target_layers: List[str]=None, name: str='Chameleon(full)') -> EvaluationResult:
        'Evaluate Chameleon(full) with dynamic gating and Personal Fake-it'
        logger.info('Starting Chameleon(full) evaluation with dynamic gating...')
        predictions: List[str] = []
        gating_weights: List[Dict] = []
        start = time.time()
        for (i, sample) in enumerate(test_samples):
            logger.info(f'Chameleon(full) progress: {(i + 1)}/{len(test_samples)}')
            user_profile = sample.get('profile', [])
            history_length = len(user_profile)
            user_context = (str(user_profile)[:500] if user_profile else 'No previous history available')
            question = sample['input'][:200]
            fakeit_text = self.chameleon_editor.generate_personal_fakeit(user_context=user_context, question=question, user_id=str(sample.get('id', i)))
            self.chameleon_editor.fakeit_direction = self.chameleon_editor.encode_fakeit_to_direction(fakeit_text)
            uncertainty = self._estimate_uncertainty(sample['input'], user_profile)
            (alpha, beta, gamma) = self.chameleon_editor.compute_gating_weights(history_length, uncertainty)
            gating_weights.append({'sample_id': sample.get('id', i), 'history_length': history_length, 'uncertainty': uncertainty, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'fakeit_text': ((fakeit_text[:100] + '...') if (len(fakeit_text) > 100) else fakeit_text)})
            prompt = f'''Given the following movie description, provide a single word tag that best describes the movie:

Movie: {sample['input']}

Tag:'''
            resp = self.chameleon_editor.generate_with_chameleon(prompt=prompt, alpha_personal=alpha, alpha_neutral=beta, alpha_fakeit=gamma, target_layers=target_layers, gen_kwargs=self.parent.gen_kwargs)
            pred = (resp.strip().lower().split() or ['unknown'])[0]
            predictions.append(pred)
            logger.info(f'[GATING] α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f} | h={history_length}, u={uncertainty:.3f}')
        self.parent.gating_analysis = gating_weights
        return self._finalize_metrics(predictions, ground_truth, start, name)

class ChameleonEvaluator():
    'Chameleon LaMP-2 評価システムのメインクラス'

    def __init__(self, config_path: (str | None), data_path: str, decoding_mode: str='greedy'):
        self.config = self._load_config(config_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(self.config.get('output_dir', './results'))
        self.output_dir.mkdir(exist_ok=True)
        self.decoding_mode = decoding_mode
        use_sample = (decoding_mode == 'sample')
        self.gen_kwargs = dict(max_new_tokens=10, do_sample=use_sample, temperature=(0.7 if use_sample else None), top_p=(0.9 if use_sample else None), pad_token_id=None)
        self.data_loader = LaMPDataLoader(data_path)
        self.chameleon_editor = ChameleonEditor(model_name=self.config['model']['name'], device=self.config['model'].get('device', 'auto'), torch_dtype=self.config['model'].get('torch_dtype', 'float32'))
        eos = self.chameleon_editor.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        self.gen_kwargs['pad_token_id'] = eos
        # Ensure generation_config has required attributes for HF validate()
        try:
            gc = self.chameleon_editor.model.generation_config
            if not hasattr(gc, 'do_sample'):
                gc.do_sample = True
            if not hasattr(gc, 'temperature'):
                gc.temperature = 0.7
            if not hasattr(gc, 'top_p'):
                gc.top_p = 0.9
            if (getattr(gc, 'pad_token_id', None) is None):
                gc.pad_token_id = eos
        except Exception:
            pass
        self.evaluation_diagnostics = {'edit_ratios': [], 'hook_calls': [], 'kl_divergences': []}
        self.evaluation_engine = EvaluationEngine(self.chameleon_editor)
        self.evaluation_engine.parent = self
        self._load_theta_vectors()
        self.test_samples_cache: List[Dict] = []
        logger.info(f'Chameleon Evaluator initialized (decoding={self.decoding_mode})')

    def _load_config(self, config_path: (str | None)) -> Dict:
        default_config = {'model': {'name': './chameleon_prime_personalization/models/base_model', 'device': 'auto', 'max_length': 512, 'batch_size': 4}, 'chameleon': {'num_self_generated': 10, 'target_layers': ['model.layers.20'], 'alpha_personal': 1.5, 'alpha_general': (- 0.8)}, 'evaluation': {'max_users': 10, 'metrics': ['exact_match', 'bleu_score']}, 'output_dir': './results'}
        if (config_path and Path(config_path).exists()):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            for (k, v) in default_config.items():
                if (k not in cfg):
                    cfg[k] = v
                elif isinstance(v, dict):
                    for (kk, vv) in v.items():
                        if (kk not in cfg[k]):
                            cfg[k][kk] = vv
            return cfg
        else:
            logger.info('Using default configuration')
            return default_config

    # Compatibility shim: allow callers that expect this on Evaluator to work.
    def compute_direction_vectors(self, *args, **kwargs):
        """Delegate to ChameleonEditor for compatibility"""
        if hasattr(self.chameleon_editor, 'compute_direction_vectors'):
            return self.chameleon_editor.compute_direction_vectors(*args, **kwargs)
        else:
            return None

    def get_effective_model_dtype(self):
        """Get actual model dtype from parameters for accurate reporting"""
        try:
            return str(next(self.chameleon_editor.model.parameters()).dtype)
        except Exception:
            return str(getattr(self.chameleon_editor.model, "dtype", "unknown"))

    def summarize_gate(self, hidden_norm, dv, alpha_personal, alpha_general, edit_gate_threshold):
        return self.chameleon_editor.compute_gate(hidden_norm,
                                                  dv["l2_personal"], dv["l2_general"],
                                                  alpha_personal, alpha_general,
                                                  edit_gate_threshold)

    def get_gate_summary(self, sample, target_layers=None, norm_scale=1.0, threshold=0.022):
        """
        Get gate summary with direction vector computation and gate value
        
        Returns comprehensive gate analysis including direction vectors,
        gate values, and threshold comparison results.
        """
        gate_summary = {
            'direction_vectors_computed': False,
            'gate_value': 0.0,
            'threshold': threshold,
            'gate_active': False,
            'error_message': None,
            'direction_p_norm': None,
            'direction_n_norm': None,
            'cosine_similarity': None
        }
        
        try:
            # Compute direction vectors using the robust method
            direction_result = self.chameleon_editor.compute_direction_vectors(
                sample, target_layers, norm_scale
            )
            
            if direction_result is None:
                gate_summary['error_message'] = "Direction vector computation returned None"
                return gate_summary
                
            direction_p, direction_n = direction_result
            gate_summary['direction_vectors_computed'] = True
            
            # Store direction vector norms
            if direction_p is not None:
                gate_summary['direction_p_norm'] = torch.norm(direction_p).item()
            if direction_n is not None:
                gate_summary['direction_n_norm'] = torch.norm(direction_n).item()
            
            # Compute gate value using the explicit method
            gate_value = self.chameleon_editor.compute_gate_value(
                direction_p, direction_n, threshold
            )
            
            gate_summary['gate_value'] = gate_value
            gate_summary['gate_active'] = gate_value >= threshold
            
            # Additional statistics
            if direction_p is not None and direction_n is not None:
                p_norm = F.normalize(direction_p, p=2, dim=0)
                n_norm = F.normalize(direction_n, p=2, dim=0)
                cosine_sim = torch.dot(p_norm, n_norm).item()
                gate_summary['cosine_similarity'] = cosine_sim
            
        except Exception as e:
            gate_summary['error_message'] = str(e)
            logger.error(f"Gate summary computation failed: {e}")
        
        return gate_summary

    def _load_theta_vectors(self):
        theta_paths = [((self.data_path / 'processed/LaMP-2/theta_p.json'), (self.data_path / 'processed/LaMP-2/theta_n.json')), (Path('processed/LaMP-2/theta_p.json'), Path('processed/LaMP-2/theta_n.json'))]
        for (tp, tn) in theta_paths:
            if (tp.exists() and tn.exists()):
                if self.chameleon_editor.load_theta_vectors(str(tp), str(tn)):
                    logger.info('Theta vectors loaded successfully')
                    return
        logger.warning('Theta vectors not found - Chameleon evaluation will be limited')

    def run_evaluation(self, mode: str='full', alpha_override: float=None, beta_override: float=None, layers_override: List[str]=None, target_edit_ratio: float=0.02, edit_ratio_tolerance: float=0.5, adaptive_alpha: bool=False, last_k_tokens: int=0, max_users_override: int=None) -> Dict[(str, Any)]:
        logger.info(f'=== Chameleon LaMP-2 Evaluation ({mode} mode, decoding={self.decoding_mode}) ===')
        if (mode == 'demo'):
            user_limit = 3
        elif (mode == 'full'):
            user_limit = (max_users_override if (max_users_override is not None) else self.config['evaluation']['max_users'])
        else:
            user_limit = 10
        test_samples = self.data_loader.get_user_samples(user_limit)
        self.test_samples_cache = test_samples
        ground_truth = self.data_loader.load_ground_truth()
        logger.info(f'Evaluating {len(test_samples)} samples from {user_limit} users')
        results: Dict[(str, Any)] = {}
        baseline = self.evaluation_engine.evaluate_baseline(test_samples, ground_truth)
        results['baseline'] = baseline
        if (self.chameleon_editor.personal_direction is not None):
            alpha_p = (alpha_override if (alpha_override is not None) else self.config['chameleon']['alpha_personal'])
            alpha_n = (beta_override if (beta_override is not None) else self.config['chameleon']['alpha_general'])
            layers = (layers_override if (layers_override is not None) else self.config['chameleon']['target_layers'])
            chameleon = self.evaluation_engine.evaluate_chameleon(test_samples, ground_truth, alpha_personal=alpha_p, alpha_neutral=alpha_n, target_layers=layers, target_edit_ratio=target_edit_ratio, edit_ratio_tolerance=edit_ratio_tolerance, adaptive_alpha=adaptive_alpha, last_k_tokens=last_k_tokens)
            results['chameleon'] = chameleon
            if ((baseline.total_samples >= 2) and (chameleon.total_samples >= 2)):
                b_correct = np.array([int((p == g)) for (p, g) in zip(baseline.predictions, baseline.ground_truths)])
                c_correct = np.array([int((p == g)) for (p, g) in zip(chameleon.predictions, chameleon.ground_truths)])
                if (len(b_correct) == len(c_correct)):
                    (_, p_value) = stats.ttest_rel(c_correct, b_correct)
                else:
                    (_, p_value) = stats.ttest_ind(c_correct, b_correct)
            else:
                p_value = 1.0
            imp_rate = (((chameleon.accuracy - baseline.accuracy) / baseline.accuracy) if (baseline.accuracy > 0) else 0.0)
            results['significance'] = {'p_value': float(p_value), 'improvement_rate': float(imp_rate), 'baseline_accuracy': float(baseline.accuracy), 'chameleon_accuracy': float(chameleon.accuracy)}
        else:
            logger.warning('Chameleon evaluation skipped - theta vectors not available')
        self._write_csv_results(results, mode, alpha_override, beta_override, layers_override, target_edit_ratio, edit_ratio_tolerance, adaptive_alpha, last_k_tokens)
        self._save_results(results)
        self._print_report(results)
        return results

    def run_3condition_ablation(self, mode: str='demo') -> Dict[(str, Any)]:
        'Run 3-condition ablation: User-only, Chameleon(-FI), Chameleon(full)'
        logger.info(f'=== 3-Condition Ablation Study ({mode} mode) ===')
        if (mode == 'demo'):
            user_limit = 3
        elif (mode == 'full'):
            user_limit = self.config['evaluation']['max_users']
        else:
            user_limit = 8
        test_samples = self.data_loader.get_user_samples(user_limit)
        ground_truth = self.data_loader.load_ground_truth()
        logger.info(f'Evaluating {len(test_samples)} samples from {user_limit} users')
        results = {}
        logger.info('--- Condition 1: User-only (UO) ---')
        if (self.chameleon_editor.personal_direction is not None):
            uo_result = self.evaluation_engine.evaluate_chameleon(test_samples, ground_truth, target_layers=self.config['chameleon']['target_layers'], alpha_personal=1.0, alpha_neutral=0.0, alpha_fakeit=0.0, name='User-only')
            results['user_only'] = uo_result
            logger.info(f'User-only: acc={uo_result.accuracy:.3f}')
        logger.info('--- Condition 2: Chameleon(-FI) ---')
        alpha_p = self.config['chameleon']['alpha_personal']
        alpha_n = self.config['chameleon']['alpha_general']
        total = (alpha_p + alpha_n)
        (alpha_p_norm, alpha_n_norm) = ((alpha_p / total), (alpha_n / total))
        chameleon_nfi_result = self.evaluation_engine.evaluate_chameleon(test_samples, ground_truth, target_layers=self.config['chameleon']['target_layers'], alpha_personal=alpha_p_norm, alpha_neutral=alpha_n_norm, alpha_fakeit=0.0, name='Chameleon(-FI)')
        results['chameleon_no_fakeit'] = chameleon_nfi_result
        logger.info(f'Chameleon(-FI): acc={chameleon_nfi_result.accuracy:.3f}')
        logger.info('--- Condition 3: Chameleon(full) ---')
        try:
            chameleon_full_result = self.evaluation_engine.evaluate_chameleon_full(test_samples, ground_truth, target_layers=self.config['chameleon']['target_layers'], name='Chameleon(full)')
            results['chameleon_full'] = chameleon_full_result
            logger.info(f'Chameleon(full): acc={chameleon_full_result.accuracy:.3f}')
        except AttributeError:
            logger.warning('evaluate_chameleon_full not implemented yet - using placeholder')
            results['chameleon_full'] = chameleon_nfi_result
        if (len(results) >= 2):
            logger.info('--- Statistical Significance ---')
            conditions = list(results.keys())
            for (i, cond1) in enumerate(conditions):
                for (j, cond2) in enumerate(conditions[(i + 1):], (i + 1)):
                    acc1 = results[cond1].accuracy
                    acc2 = results[cond2].accuracy
                    improvement = ((((acc2 - acc1) / acc1) * 100) if (acc1 > 0) else 0)
                    logger.info(f'{conditions[j]} vs {cond1}: Δacc={improvement:+.1f}%')
        results['ablation_summary'] = {'conditions': len(results), 'best_condition': (max(results.keys(), key=(lambda k: results[k].accuracy)) if results else None), 'best_accuracy': (max((r.accuracy for r in results.values())) if results else 0.0)}
        return results

    def _write_csv_results(self, results: Dict[(str, Any)], mode: str, alpha_override: float=None, beta_override: float=None, layers_override: List[str]=None, target_edit_ratio: float=0.02, edit_ratio_tolerance: float=0.5, adaptive_alpha: bool=False, last_k_tokens: int=0):
        'Write evaluation results to CSV file'
        csv_file = (self.output_dir / 'experiment_results.csv')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alpha_p = (alpha_override if (alpha_override is not None) else self.config['chameleon']['alpha_personal'])
        alpha_n = (beta_override if (beta_override is not None) else self.config['chameleon']['alpha_general'])
        layers = (layers_override if (layers_override is not None) else self.config['chameleon']['target_layers'])
        layers_str = (','.join(layers) if isinstance(layers, list) else str(layers))
        baseline = results.get('baseline')
        chameleon = results.get('chameleon')
        significance = results.get('significance', {})
        avg_edit_ratio = (np.mean(self.evaluation_diagnostics['edit_ratios']) if self.evaluation_diagnostics['edit_ratios'] else 0.0)
        avg_kl = (np.mean(self.evaluation_diagnostics['kl_divergences']) if self.evaluation_diagnostics['kl_divergences'] else 0.0)
        hook_calls_mean = (np.mean(self.evaluation_diagnostics['hook_calls']) if self.evaluation_diagnostics['hook_calls'] else 0.0)
        suggested_alpha_mean = (np.mean(self.evaluation_diagnostics['suggested_alphas']) if (('suggested_alphas' in self.evaluation_diagnostics) and self.evaluation_diagnostics['suggested_alphas']) else alpha_p)
        row = {'timestamp': timestamp, 'mode': mode, 'gen': self.decoding_mode, 'layers': layers_str, 'alpha': alpha_p, 'beta': alpha_n, 'baseline_acc': (baseline.accuracy if baseline else 0.0), 'baseline_em': (baseline.exact_match if baseline else 0.0), 'baseline_bleu': (baseline.bleu_score if baseline else 0.0), 'baseline_f1': (baseline.f1_score if baseline else 0.0), 'baseline_time': (baseline.inference_time if baseline else 0.0), 'cham_acc': (chameleon.accuracy if chameleon else 0.0), 'cham_em': (chameleon.exact_match if chameleon else 0.0), 'cham_bleu': (chameleon.bleu_score if chameleon else 0.0), 'cham_f1': (chameleon.f1_score if chameleon else 0.0), 'cham_time': (chameleon.inference_time if chameleon else 0.0), 'impr_abs_acc': ((chameleon.accuracy - baseline.accuracy) if (baseline and chameleon) else 0.0), 'impr_rel_acc_pct': ((((chameleon.accuracy - baseline.accuracy) / baseline.accuracy) * 100) if (baseline and chameleon and (baseline.accuracy > 0)) else 0.0), 'p_value': significance.get('p_value', 1.0), 'avg_edit_ratio': avg_edit_ratio, 'avg_kl': avg_kl, 'hook_calls_mean': hook_calls_mean, 'target_edit_ratio': target_edit_ratio, 'edit_ratio_tolerance': edit_ratio_tolerance, 'adaptive_alpha': adaptive_alpha, 'suggested_alpha': suggested_alpha_mean, 'last_k_tokens': last_k_tokens}
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if (not file_exists):
                writer.writeheader()
            writer.writerow(row)
        logger.info(f'Results written to CSV: {csv_file}')
        logger.info(f'[SUMMARY] avg_edit_ratio={avg_edit_ratio:.4e}, hook_calls_mean={hook_calls_mean:.1f}')
        self._run_acceptance_checks(results, avg_edit_ratio, hook_calls_mean, layers)

    def _run_acceptance_checks(self, results: Dict[(str, Any)], avg_edit_ratio: float, hook_calls_mean: float, layers):
        'Run acceptance checks and validation alerts'
        baseline = results.get('baseline')
        chameleon = results.get('chameleon')
        if (not (baseline and chameleon)):
            return
        num_layers = (len(layers) if isinstance(layers, list) else 1)
        metrics_identical = ((abs((baseline.accuracy - chameleon.accuracy)) < 1e-06) and (abs((baseline.exact_match - chameleon.exact_match)) < 1e-06) and (abs((baseline.bleu_score - chameleon.bleu_score)) < 1e-06) and (abs((baseline.f1_score - chameleon.f1_score)) < 1e-06))
        if (metrics_identical and (avg_edit_ratio >= 0.02) and (hook_calls_mean >= num_layers)):
            logger.warning(f'[ALERT] Model outputs insensitive to current edits - all metrics identical but avg_edit_ratio={avg_edit_ratio:.4e} >= 0.02 and hook_calls_mean={hook_calls_mean:.1f} >= {num_layers}')
        if (hook_calls_mean < num_layers):
            logger.warning(f'[BUG] Hooks not firing for some layers: expected {num_layers}, got mean {hook_calls_mean:.1f}')
        if (avg_edit_ratio < 0.005):
            logger.warning(f'[WARN] Consistently weak edits across evaluation: avg_edit_ratio={avg_edit_ratio:.4e} < 0.005')
        improvement = (chameleon.accuracy - baseline.accuracy)
        if (improvement >= 0.02):
            logger.info(f'[SUCCESS] Significant accuracy improvement: {improvement:+.4f} >= +0.02')
        elif (improvement > 0):
            logger.info(f'[PARTIAL] Minor accuracy improvement: {improvement:+.4f}')
        else:
            logger.warning(f'[FAIL] No accuracy improvement: {improvement:+.4f}')

    def _save_results(self, results: Dict[(str, Any)]):
        ts = time.strftime('%Y%m%d_%H%M%S')
        outdir = (self.output_dir / f'evaluation_{ts}')
        outdir.mkdir(exist_ok=True)
        serializable = {}
        for (k, v) in results.items():
            if isinstance(v, EvaluationResult):
                serializable[k] = {'method_name': v.method_name, 'accuracy': float(v.accuracy), 'exact_match': float(v.exact_match), 'bleu_score': float(v.bleu_score), 'precision': float(v.precision), 'recall': float(v.recall), 'f1_score': float(v.f1_score), 'inference_time': float(v.inference_time), 'total_samples': int(v.total_samples), 'correct_predictions': int(v.correct_predictions), 'predictions': v.predictions, 'ground_truths': v.ground_truths}
            else:
                serializable[k] = v
        with open((outdir / 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f'Results saved to: {outdir}')

    def _print_report(self, results: Dict[(str, Any)]):
        print(('\n' + ('=' * 60)))
        print('🎯 Chameleon LaMP-2 Evaluation Results')
        print(('=' * 60))
        b = results.get('baseline')
        c = results.get('chameleon')
        sig = results.get('significance', {})
        if b:
            print(f'''
📊 Baseline Performance:''')
            print(f'   Accuracy:     {b.accuracy:.4f}')
            print(f'   Exact Match:  {b.exact_match:.4f}')
            print(f'   BLEU Score:   {b.bleu_score:.4f}')
            print(f'   F1 Score:     {b.f1_score:.4f}')
            print(f'   Inference:    {b.inference_time:.2f}s')
        if c:
            print(f'''
🦎 Chameleon Performance:''')
            print(f'   Accuracy:     {c.accuracy:.4f}')
            print(f'   Exact Match:  {c.exact_match:.4f}')
            print(f'   BLEU Score:   {c.bleu_score:.4f}')
            print(f'   F1 Score:     {c.f1_score:.4f}')
            print(f'   Inference:    {c.inference_time:.2f}s')
        if (b and c and sig):
            imp = (sig.get('improvement_rate', 0.0) * 100)
            p_value = sig.get('p_value', 1.0)
            print(f'''
📈 Improvement Analysis:''')
            print(f'   Improvement Rate: {imp:+.1f}%')
            print(f'   Statistical Significance: p = {p_value:.4f}')
            if (p_value < 0.05):
                if (c.accuracy > b.accuracy):
                    print('   ✅ Statistically significant improvement!')
                elif (c.accuracy < b.accuracy):
                    print('   ❌ Statistically significant degradation!')
                else:
                    print('   ⚠️  Significant difference detected but same accuracy (check metrics)')
            else:
                print('   ⚠️  No significant difference detected')
        print(('\n' + ('=' * 60)))
if (__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='Chameleon LaMP-2 Evaluation System')
    parser.add_argument('--mode', choices=['demo', 'full', 'ablation'], default='full', help='Evaluation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data_path', type=str, default=None, help='Root directory that contains LaMP-2 data (expects raw/LaMP-2/merged.json or answers.json)')
    parser.add_argument('--gen', choices=['greedy', 'sample'], default='greedy', help='Generation mode: greedy (do_sample=False) or sample (do_sample=True, temp/top_p used)')
    parser.add_argument('--alpha', type=float, help='Alpha parameter (personal direction strength)')
    parser.add_argument('--beta', type=float, help='Beta parameter (neutral direction strength)')
    parser.add_argument('--layers', type=str, help='Comma-separated layer names (e.g., model.layers.20,model.layers.30)')
    parser.add_argument('--target_edit_ratio', type=float, default=0.02, help='Target edit ratio for adaptive alpha')
    parser.add_argument('--edit_ratio_tolerance', type=float, default=0.5, help='Tolerance ratio for target edit ratio (±)')
    parser.add_argument('--adaptive_alpha', action='store_true', help='Enable adaptive alpha scaling')
    parser.add_argument('--last_k_tokens', type=int, default=0, help='Apply editing only to last k tokens (0 = all tokens)')
    parser.add_argument('--max_users', type=int, default=None, help='Maximum number of users for evaluation')
    args = parser.parse_args()
    target_layers = None
    if args.layers:
        target_layers = [layer.strip() for layer in args.layers.split(',') if layer.strip()]
    evaluator = ChameleonEvaluator(config_path=args.config, data_path=(args.data_path or './'), decoding_mode=args.gen)
    results = evaluator.run_evaluation(mode=args.mode, alpha_override=args.alpha, beta_override=args.beta, layers_override=target_layers, target_edit_ratio=args.target_edit_ratio, edit_ratio_tolerance=args.edit_ratio_tolerance, adaptive_alpha=args.adaptive_alpha, last_k_tokens=args.last_k_tokens, max_users_override=args.max_users)
    print('\n✅ Evaluation completed successfully!')
