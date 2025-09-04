import torch
import math
from typing import List, Tuple, Optional, Dict

@torch.no_grad()
def conditional_logprob(model, tokenizer, prompt: str, continuation: str, device="cuda", return_length=False):
    """
    P(continuation | prompt) のトークン対数確率総和を返す（decoder-only）。
    Optimized version for single-token continuations.
    
    Args:
        return_length: If True, returns (logprob, token_length) tuple for PMI calculation
    
    Returns:
        float or tuple: logprob or (logprob, length) if return_length=True
    """
    # For single digit continuations ("1", "2", etc.), use more efficient approach
    if len(continuation) == 1 and continuation.isdigit():
        # Single token approach
        p_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        c_token_id = tokenizer.encode(continuation, add_special_tokens=False)[0]
        
        # Single forward pass - force no cache to ensure hooks fire
        out = model(input_ids=p_ids, use_cache=False)
        logits = out.logits[:, -1, :]  # Last position logits
        logprobs = torch.log_softmax(logits, dim=-1)
        
        logprob = float(logprobs[0, c_token_id])
        return (logprob, 1) if return_length else logprob
    else:
        # Fallback to sequential approach for multi-token continuations
        p_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        c_ids = tokenizer(continuation, return_tensors="pt").to(device)["input_ids"]
        # 逐次的に continuation を条件付きで評価
        total = 0.0
        inp = p_ids
        token_count = 0
        for t in c_ids[0]:
            out = model(input_ids=inp, use_cache=False)
            logits = out.logits[:, -1, :]
            logp = torch.log_softmax(logits, dim=-1)
            total += float(logp[0, t])
            inp = torch.cat([inp, t.view(1,1)], dim=1)
            token_count += 1
        
        return (total, token_count) if return_length else total

@torch.no_grad()
def conditional_logprob_batched(model, tokenizer, prompt: str, continuations: List[str], device="cuda"):
    """
    Batched version of conditional_logprob for faster scoring of multiple candidates.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer  
        prompt: Input prompt string
        continuations: List of continuation strings to score
        device: Computing device
        
    Returns:
        List[float]: Log probabilities for each continuation
    """
    if not continuations:
        return []
    
    # Tokenize prompt once
    p_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]  # [1, seq_len]
    
    # Check if all continuations are single digits (optimization case)
    all_single_digits = all(len(cont) == 1 and cont.isdigit() for cont in continuations)
    
    if all_single_digits:
        # Single token batched scoring - most efficient
        # Get token IDs for all continuations
        token_ids = []
        for cont in continuations:
            token_id = tokenizer.encode(cont, add_special_tokens=False)[0]
            token_ids.append(token_id)
        
        # Single forward pass
        out = model(input_ids=p_ids, use_cache=False)
        logits = out.logits[:, -1, :]  # [1, vocab_size]
        logprobs = torch.log_softmax(logits, dim=-1)
        
        # Extract scores for all candidates
        scores = []
        for token_id in token_ids:
            scores.append(float(logprobs[0, token_id]))
        return scores
    
    else:
        # Multi-token case: still process sequentially but with shared prefix
        # Get past_key_values from prompt processing
        with torch.no_grad():
            prompt_out = model(input_ids=p_ids, use_cache=True)
            past_key_values = prompt_out.past_key_values
        
        scores = []
        for continuation in continuations:
            c_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").to(device)["input_ids"]
            
            if c_ids.size(1) == 0:  # Empty continuation
                scores.append(0.0)
                continue
            
            # Use cached prefix for faster processing
            total_logprob = 0.0
            current_past = past_key_values
            
            for i, token_id in enumerate(c_ids[0]):
                if i == 0:
                    # First token: use cached past
                    token_input = token_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
                    out = model(input_ids=token_input, past_key_values=current_past, use_cache=True)
                else:
                    # Subsequent tokens: continue with updated cache
                    token_input = token_id.unsqueeze(0).unsqueeze(0)  # [1, 1] 
                    out = model(input_ids=token_input, past_key_values=current_past, use_cache=True)
                
                logits = out.logits[:, -1, :]  # [1, vocab_size]
                logprobs = torch.log_softmax(logits, dim=-1)
                total_logprob += float(logprobs[0, token_id])
                current_past = out.past_key_values
            
            scores.append(total_logprob)
        
        return scores

@torch.no_grad()
def safe_score_first_token(model, tokenizer, prompt: str, label: str, device="cuda"):
    """Fallback scorer used only when conditional_logprob fails for a candidate."""
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    y_ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=p_ids, use_cache=False)
    logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)
    return float(logp[0, y_ids[0, 0]].item())

@torch.no_grad()
def compute_prior_logprobs(model, tokenizer, id2tag: Dict[int, str], device="cuda", 
                          score_norm="avg", null_prompts=None) -> Dict[int, float]:
    """
    Compute prior log probabilities P(y|∅) for PMI scoring.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        id2tag: Mapping from IDs to tag names
        device: Computing device
        score_norm: "sum" or "avg" for length normalization
        null_prompts: List of neutral prompts, or None for default
        
    Returns:
        Dict mapping tag IDs to prior log probabilities
    """
    if null_prompts is None:
        # Default neutral prompts - multiple variations for stability
        label_list = list(id2tag.values())
        null_prompts = [
            f"You are a tag classifier.\nAnswer with exactly one tag from this closed set:\n[{', '.join(label_list)}]\nAnswer:",
            f"Choose one tag from: [{', '.join(label_list)}].\nAnswer:",
            f"Single tag from [{', '.join(label_list)}]:"
        ]
    
    prior_logprobs = {}
    
    for tag_id, tag_name in id2tag.items():
        # Compute average prior across multiple neutral prompts for stability
        tag_priors = []
        
        for null_prompt in null_prompts:
            logprob, length = conditional_logprob(
                model, tokenizer, null_prompt, str(tag_id), 
                device=device, return_length=True
            )
            
            # Apply length normalization
            normalized_logprob = logprob / max(length, 1) if score_norm == "avg" else logprob
            tag_priors.append(normalized_logprob)
        
        # Average across prompts for stability
        prior_logprobs[tag_id] = sum(tag_priors) / len(tag_priors)
    
    # Log prior statistics
    prior_values = list(prior_logprobs.values())
    print(f"[PMI] Computed priors: mean={sum(prior_values)/len(prior_values):.4f}, "
          f"std={torch.std(torch.tensor(prior_values)):.4f}")
    
    # Show top prior labels (most biased)
    sorted_priors = sorted(prior_logprobs.items(), key=lambda x: x[1], reverse=True)[:3]
    top_labels = [(id2tag[pid], pval) for pid, pval in sorted_priors]
    print(f"[PMI] Top prior labels: {top_labels}")
    
    return prior_logprobs

@torch.no_grad()
def classify_by_scores_pmi(model, tokenizer, prompt: str, id2tag: Dict[int, str], 
                          prior_logprobs: Dict[int, float], device="cuda", 
                          score_norm="avg", score_temp=1.0, return_scores=False):
    """
    Classify using PMI scores: log P(y|x) - log P(y|∅)
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompt: Input prompt
        id2tag: Mapping from IDs to tag names  
        prior_logprobs: Pre-computed prior log probabilities
        device: Computing device
        score_norm: "sum" or "avg" for length normalization
        score_temp: Temperature for softmax smoothing (>1 = flatter)
        return_scores: Whether to return full score dict
        
    Returns:
        (best_id, best_tag) or (best_id, best_tag, scores) if return_scores=True
    """
    print(f"[PMI] Evaluating {len(id2tag)} candidates with PMI scoring")
    scores = {}
    best_id, best_tag, best_score = None, None, float('-inf')
    
    for tag_id, tag_name in id2tag.items():
        # Compute conditional log probability
        logprob, length = conditional_logprob(
            model, tokenizer, prompt, str(tag_id), 
            device=device, return_length=True
        )
        
        # Apply length normalization
        normalized_logprob = logprob / max(length, 1) if score_norm == "avg" else logprob
        
        # PMI score: conditional - prior
        prior_logprob = prior_logprobs.get(tag_id, 0.0)
        pmi_score = normalized_logprob - prior_logprob
        
        # Apply temperature smoothing
        final_score = pmi_score / max(score_temp, 1e-6)
        
        scores[tag_id] = final_score
        print(f"[PMI] ID {tag_id} ('{tag_name}'): cond={normalized_logprob:.4f}, "
              f"prior={prior_logprob:.4f}, pmi={pmi_score:.4f}, final={final_score:.4f}")
        
        if final_score > best_score:
            best_score, best_id, best_tag = final_score, tag_id, tag_name
    
    print(f"[PMI] Selected: ID {best_id} ('{best_tag}') with PMI score {best_score:.4f}")
    
    if return_scores:
        return best_id, best_tag, scores
    return best_id, best_tag


@torch.no_grad()
def classify_with_prior_correction(model, tokenizer, prompt: str, id2tag: Dict[int, str], 
                                  device="cuda", prior_mode="empty", prior_beta=1.0, 
                                  prior_alpha=1.0, score_norm="avg", score_temp=1.0,
                                  score_template="\nAnswer: {label}",
                                  prior_table=None, user_prior_table=None, user_id=None,
                                  return_scores=False, return_prior_info=False):
    """
    Unified scoring function supporting multiple prior correction modes.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompt: Input prompt  
        id2tag: Mapping from IDs to tag names
        device: Computing device
        prior_mode: {"none", "empty", "global", "user"} - prior correction mode
        prior_beta: β weight for prior term (default 1.0)
        prior_alpha: α Dirichlet smoothing parameter (default 1.0)
        score_norm: "sum" or "avg" for length normalization
        score_temp: Temperature for distribution flattening
        score_template: Template for generating answer format (e.g., "\nAnswer: {label}")
        prior_table: Pre-computed global priors {label: prob}
        user_prior_table: Pre-computed user priors {user_id: {label: prob}}
        user_id: Current user ID (for user mode)
        return_scores: Whether to return full score dict
        return_prior_info: Whether to return prior information
        
    Returns:
        (best_id, best_tag) or extended tuple if return_scores/return_prior_info=True
    """
    print(f"[scoring] prior_mode={prior_mode}, beta={prior_beta:.2f}, alpha={prior_alpha:.2f}")
    print(f"[scoring] template='{score_template}', temp={score_temp:.1f}, len_norm={score_norm}")
    
    scores = {}
    best_id, best_tag, best_score = None, None, float('-inf')
    prior_info = {}
    
    # Step 1: Compute conditional probabilities for all candidates
    for tag_id, tag_name in id2tag.items():
        # Generate full prompt with template
        answer_text = score_template.format(label=tag_name)
        
        # Compute conditional log probability P(T(y)|x)
        try:
            logprob, length = conditional_logprob(
                model, tokenizer, prompt, answer_text, 
                device=device, return_length=True
            )
        except Exception as e:
            print(f"[scoring-fallback] conditional_logprob failed for {tag_name}: {e} -> first-token fallback")
            logprob = safe_score_first_token(model, tokenizer, prompt, tag_name, device)
            length = 1
        
        # Apply length normalization
        score_cond = logprob / max(length, 1) if score_norm == "avg" else logprob
        
        # Step 2: Compute prior term based on mode
        prior_term = 0.0
        
        if prior_mode == "none":
            prior_term = 0.0
            
        elif prior_mode == "empty":
            # Empty context PMI: log P(T(y)|∅)
            try:
                empty_logprob, empty_length = conditional_logprob(
                    model, tokenizer, "", answer_text,
                    device=device, return_length=True
                )
            except Exception as e:
                print(f"[scoring-fallback] empty PMI failed for {tag_name}: {e} -> first-token fallback")
                empty_logprob = safe_score_first_token(model, tokenizer, "", tag_name, device)
                empty_length = 1
            empty_normalized = empty_logprob / max(empty_length, 1) if score_norm == "avg" else empty_logprob
            prior_term = prior_beta * empty_normalized
            
        elif prior_mode == "global":
            # Global PMI: β * log P_global(y)
            if prior_table and tag_name in prior_table:
                global_prob = prior_table[tag_name]
                prior_term = prior_beta * math.log(max(global_prob, 1e-8))
            else:
                print(f"[scoring] WARNING: missing global prior for '{tag_name}', using 0")
                prior_term = 0.0
                
        elif prior_mode == "user":
            # User PMI: β * log P_user(y) with fallback to global
            user_prob = None
            
            if user_prior_table and user_id and user_id in user_prior_table:
                user_priors = user_prior_table[user_id]
                if tag_name in user_priors:
                    user_prob = user_priors[tag_name]
                    prior_term = prior_beta * math.log(max(user_prob, 1e-8))
                else:
                    print(f"[scoring] missing user prior for user={user_id}, label='{tag_name}', fallback to global")
                    if prior_table and tag_name in prior_table:
                        global_prob = prior_table[tag_name]
                        prior_term = prior_beta * math.log(max(global_prob, 1e-8))
                    else:
                        prior_term = 0.0
            else:
                print(f"[scoring] user_id={user_id} not found, fallback to global for '{tag_name}'")
                if prior_table and tag_name in prior_table:
                    global_prob = prior_table[tag_name]
                    prior_term = prior_beta * math.log(max(global_prob, 1e-8))
                else:
                    prior_term = 0.0
        
        # Step 3: Final score = conditional - prior_term
        corrected_score = score_cond - prior_term
        
        # Apply temperature smoothing
        final_score = corrected_score / max(score_temp, 1e-6)
        
        scores[tag_id] = final_score
        
        if final_score > best_score:
            best_score, best_id, best_tag = final_score, tag_id, tag_name
    
    print(f"[scoring] Selected: ID {best_id} ('{best_tag}') with score {best_score:.4f}")
    
    # Prepare return values
    result = [best_id, best_tag]
    if return_scores:
        result.append(scores)
    if return_prior_info:
        # Add top-5 priors for logging
        if prior_mode == "global" and prior_table:
            sorted_priors = sorted(prior_table.items(), key=lambda x: x[1], reverse=True)
            prior_info["prior_top5"] = sorted_priors[:5]
        elif prior_mode == "user" and user_prior_table and user_id in user_prior_table:
            user_priors = user_prior_table[user_id]
            sorted_user_priors = sorted(user_priors.items(), key=lambda x: x[1], reverse=True)
            prior_info["user_prior_top5"] = sorted_user_priors[:5]
        result.append(prior_info)
    
    return tuple(result) if len(result) > 2 else (result[0], result[1])


@torch.no_grad()
def classify_by_scores(model, tokenizer, prompt: str, id2tag: dict, device="cuda", return_scores=False):
    """
    各候補ID文字列("1","2",...)の条件付き対数尤度を比較し argmax を返す。
    """
    print(f"[classify_by_scores] Evaluating {len(id2tag)} candidates")
    scores = {}
    best_id, best_tag, best_score = None, None, -1e30
    for i, tag in id2tag.items():
        s = conditional_logprob(model, tokenizer, prompt, str(i), device)
        scores[i] = s
        print(f"[classify_by_scores] ID {i} ('{tag}'): {s:.4f}")
        if s > best_score:
            best_score, best_id, best_tag = s, i, tag
    print(f"[classify_by_scores] Selected: ID {best_id} ('{best_tag}') with score {best_score:.4f}")
    
    if return_scores:
        return best_id, best_tag, scores
    return best_id, best_tag

@torch.no_grad()
def compute_prior_scores(model, tokenizer, prior_prompt: str, id2tag: dict, device="cuda"):
    pri = {}
    for i in id2tag:
        pri[i] = conditional_logprob(model, tokenizer, prior_prompt, str(i), device)
    return pri

@torch.no_grad()
def classify_by_scores_with_calibration(model, tokenizer, prompt: str, id2tag: dict,
                                        prior_prompt: Optional[str] = None,
                                        device="cuda", lam: float = 1.0,
                                        prior_scores: Optional[dict] = None,
                                        return_scores: bool = False):
    # prior_scores が与えられていればそれを使い、無ければ current model で prior を計算
    pri = None
    if prior_scores is not None:
        pri = prior_scores
    elif prior_prompt:
        pri = compute_prior_scores(model, tokenizer, prior_prompt, id2tag, device)
    else:
        if return_scores:
            return classify_by_scores(model, tokenizer, prompt, id2tag, device, return_scores=True)
        return classify_by_scores(model, tokenizer, prompt, id2tag, device)
    
    scores = {}
    best_id, best_tag, best_score = None, None, -1e30
    for i, tag in id2tag.items():
        try:
            s = conditional_logprob(model, tokenizer, prompt, str(i), device)
        except Exception as e:
            print(f"[scoring-fallback] conditional_logprob failed for ID {i}: {e} -> first-token fallback")
            s = safe_score_first_token(model, tokenizer, prompt, str(i), device)
        s_adj = s - lam * pri[i]
        scores[i] = s_adj  # Store calibrated score
        if s_adj > best_score:
            best_score, best_id, best_tag = s_adj, i, tag
    
    if return_scores:
        return best_id, best_tag, scores
    return best_id, best_tag

@torch.no_grad()
def classify_by_scores_with_calibration_pmi(model, tokenizer, prompt: str, id2tag: Dict[int, str],
                                           prior_logprobs: Dict[int, float],
                                           prior_prompt: Optional[str] = None,
                                           device="cuda", lam: float = 1.0,
                                           prior_scores: Optional[dict] = None,
                                           score_norm="avg", score_temp=1.0,
                                           return_scores: bool = False):
    """
    PMI scoring with calibration support.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer  
        prompt: Input prompt
        id2tag: Mapping from IDs to tag names
        prior_logprobs: PMI prior log probabilities P(y|∅)
        prior_prompt: Legacy calibration prompt (unused with PMI)
        device: Computing device
        lam: Legacy calibration weight (unused with PMI)
        prior_scores: Legacy prior scores (unused with PMI)
        score_norm: "sum" or "avg" for length normalization
        score_temp: Temperature for smoothing
        return_scores: Whether to return full score dict
        
    Returns:
        (best_id, best_tag) or (best_id, best_tag, scores) if return_scores=True
    """
    # PMI already handles "calibration" via prior subtraction
    # Just use PMI scoring directly
    return classify_by_scores_pmi(
        model, tokenizer, prompt, id2tag, prior_logprobs,
        device=device, score_norm=score_norm, score_temp=score_temp,
        return_scores=return_scores
    )


# Batched scoring functions for improved performance

@torch.no_grad()
def classify_by_scores_batched(model, tokenizer, prompt: str, id2tag: dict, device="cuda", return_scores=False):
    """Batched version of classify_by_scores for faster inference."""
    continuations = [str(i) for i in id2tag.keys()]
    scores_list = conditional_logprob_batched(model, tokenizer, prompt, continuations, device)
    
    scores = {}
    best_id, best_tag, best_score = None, None, -1e30
    
    for i, (tag_id, tag_name) in enumerate(id2tag.items()):
        s = scores_list[i]
        scores[tag_id] = s
        if s > best_score:
            best_score, best_id, best_tag = s, tag_id, tag_name
    
    if return_scores:
        return best_id, best_tag, scores
    return best_id, best_tag


@torch.no_grad() 
def classify_by_scores_with_calibration_batched(model, tokenizer, prompt: str, id2tag: dict,
                                               prior_prompt: Optional[str] = None,
                                               device="cuda", lam: float = 1.0,
                                               prior_scores: Optional[dict] = None,
                                               return_scores: bool = False):
    """Batched version of classify_by_scores_with_calibration for faster inference."""
    # Handle prior scores
    pri = None
    if prior_scores is not None:
        pri = prior_scores
    elif prior_prompt:
        pri = compute_prior_scores(model, tokenizer, prior_prompt, id2tag, device)
    else:
        if return_scores:
            return classify_by_scores_batched(model, tokenizer, prompt, id2tag, device, return_scores=True)
        return classify_by_scores_batched(model, tokenizer, prompt, id2tag, device)
    
    # Batched scoring
    continuations = [str(i) for i in id2tag.keys()]
    scores_list = conditional_logprob_batched(model, tokenizer, prompt, continuations, device)
    
    scores = {}
    best_id, best_tag, best_score = None, None, -1e30
    
    for i, (tag_id, tag_name) in enumerate(id2tag.items()):
        s = scores_list[i]
        s_adj = s - lam * pri[tag_id]
        scores[tag_id] = s_adj  # Store calibrated score
        if s_adj > best_score:
            best_score, best_id, best_tag = s_adj, tag_id, tag_name
    
    if return_scores:
        return best_id, best_tag, scores
    return best_id, best_tag