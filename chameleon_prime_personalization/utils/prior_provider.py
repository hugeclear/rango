"""
Prior Provider: Unified prior score management with explicit fallback policies.

Handles prior_mode policies:
- 'none': No calibration (prior unused)
- 'global': Global prior only  
- 'user': User prior only (fail if missing)
- 'user_or_global': User prior with explicit global fallback
"""

import typing as T
from .score_classifier import compute_prior_scores


class PriorProvider:
    """
    Centralized prior score provider with explicit fallback handling.
    
    Args:
        model: Language model for scoring
        tokenizer: Model tokenizer
        id2tag: Label ID to tag mapping
        device: Compute device
        prior_prompt: Base prompt for prior computation
        beta: User/global mixing weight (0=global only, 1=user only)
        prior_mode: Policy for prior selection
    """
    
    def __init__(self, model, tokenizer, id2tag: dict, device="cuda",
                 prior_prompt: str = "", beta: float = 1.0, prior_mode: str = "global",
                 strict_mode: bool = False, user_prior_path: T.Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.id2tag = id2tag
        self.device = device
        self.prior_prompt = prior_prompt
        self.beta = float(beta)
        self.mode = prior_mode
        self.strict_mode = strict_mode
        self.user_prior_path = user_prior_path
        self.cache_global: T.Optional[dict] = None
        self.cache_user: dict[str, dict] = {}
        self._fallback_logged: set = set()  # Track logged fallbacks to prevent spam
        
        # STRICT mode enforcement
        if self.strict_mode:
            if self.mode != "user":
                raise RuntimeError("[STRICT] prior_mode must be 'user' under --strict")
            if not self.user_prior_path:
                raise RuntimeError("[STRICT] --user_prior_path is required under --strict")
            
            # Load user priors and enforce completeness
            loaded_count = self.load_user_priors_from_file(self.user_prior_path)
            if loaded_count == 0:
                raise RuntimeError("[STRICT] loaded user priors are empty")
            
            print(f"[STRICT] Loaded {loaded_count} user priors, fallback disabled")
            
        # Only prepare global/uniform priors when NOT strict
        if not self.strict_mode:
            # Global prior will be computed lazily via ensure_global()
            pass
        else:
            # In strict mode, never prepare global priors
            self.cache_global = None
        
    def ensure_global(self):
        """Lazily compute and cache global prior scores."""
        if self.strict_mode:
            # In strict mode, never compute global priors
            raise RuntimeError("[STRICT] Global prior access forbidden in strict mode")
        
        if self.cache_global is None:
            if self.mode == "none":
                self.cache_global = {}
            else:
                self.cache_global = compute_prior_scores(
                    self.model, self.tokenizer, self.prior_prompt, self.id2tag, self.device
                )
    
    def has_user(self, user_id: T.Union[str, int]) -> bool:
        """Check if user-specific prior is available."""
        return str(user_id) in self.cache_user
    
    def put_user_prior(self, user_id: T.Union[str, int], prompt: str):
        """Compute and cache user-specific prior scores."""
        uid = str(user_id)
        self.cache_user[uid] = compute_prior_scores(
            self.model, self.tokenizer, prompt, self.id2tag, self.device
        )
    
    def load_user_priors_from_file(self, user_priors_path: str):
        """Load pre-computed user priors from JSONL file (for strict mode)."""
        import json
        
        loaded_count = 0
        with open(user_priors_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    user_id = str(data["user_id"])
                    prior_prompt = data["prior_prompt"]
                    
                    # Compute prior scores for this user
                    prior_scores = compute_prior_scores(
                        self.model, self.tokenizer, prior_prompt, self.id2tag, self.device
                    )
                    self.cache_user[user_id] = prior_scores
                    loaded_count += 1
        
        print(f"[strict] Loaded {loaded_count} user priors from {user_priors_path}")
        return loaded_count
    
    def get(self, user_id: T.Union[str, int]) -> tuple[dict, dict]:
        """
        Get prior scores with metadata.
        
        Returns:
            (prior_scores, meta): Prior scores dict and metadata dict
            
        Metadata contains:
            - mode: Prior mode used
            - source: 'user'|'global'|'none'|'mixed'
            - beta: Mixing weight
            - user_id: User identifier
        """
        uid = str(user_id)
        meta = {
            'mode': self.mode, 
            'beta': self.beta, 
            'user_id': uid
        }
        
        # STRICT mode: Only user priors allowed, no fallback
        if self.strict_mode:
            if self.mode != "user":
                raise RuntimeError(f"[STRICT] Invalid mode '{self.mode}' in strict mode")
            if uid not in self.cache_user:
                raise RuntimeError(f"[STRICT] Missing user prior for user_id={uid}")
            return self.cache_user[uid], {**meta, 'source': 'user'}
        
        # Non-strict mode: Original behavior
        if self.mode == "none":
            return {}, {**meta, 'source': 'none'}
        
        self.ensure_global()
        global_scores = self.cache_global or {}
        
        if self.mode == "global":
            return global_scores, {**meta, 'source': 'global'}
        
        if self.mode == "user":
            if uid not in self.cache_user:
                raise RuntimeError(
                    f"[prior] user {uid} has no prior; mode=user prohibits fallback"
                )
            return self.cache_user[uid], {**meta, 'source': 'user'}
        
        if self.mode == "user_or_global":
            if uid in self.cache_user:
                user_scores = self.cache_user[uid]
                
                if self.beta >= 1.0:
                    # Pure user prior
                    return user_scores, {**meta, 'source': 'user'}
                elif self.beta <= 0.0:
                    # Pure global prior  
                    return global_scores, {**meta, 'source': 'global'}
                else:
                    # Mixed user/global prior (linear combination)
                    mixed_scores = {}
                    for label_id in self.id2tag.keys():
                        global_score = global_scores.get(label_id, 0.0)
                        user_score = user_scores.get(label_id, 0.0)
                        mixed_scores[label_id] = (1.0 - self.beta) * global_score + self.beta * user_score
                    return mixed_scores, {**meta, 'source': 'mixed'}
            else:
                # Explicit fallback to global (non-strict mode only)
                # Log fallback once per user to prevent spam
                if uid not in self._fallback_logged:
                    print(f"[prior] user {uid} missing -> fallback to global")
                    self._fallback_logged.add(uid)
                return global_scores, {**meta, 'source': 'global'}
        
        raise ValueError(f"[prior] unknown mode: {self.mode}")
    
    def get_stats(self) -> dict:
        """Get provider statistics for debugging."""
        return {
            'mode': self.mode,
            'beta': self.beta,
            'global_cached': self.cache_global is not None,
            'user_count': len(self.cache_user),
            'fallback_logged': len(self._fallback_logged)
        }