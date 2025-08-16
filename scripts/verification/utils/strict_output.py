"""
Strict Output Format Utilities
==============================

Provides utilities for enforcing strict "Answer: <label>" output format with:
- Pattern-based extraction with regex
- Format repair prompting for retry scenarios  
- JSON serialization helpers for numpy types
"""

import re
import json
from typing import Tuple, List, Optional, Any
from difflib import get_close_matches


def force_singleline_answer(text: str) -> str:
    """
    Force text to single line by extracting first Answer: line.
    
    Args:
        text: Input text potentially with multiple lines
        
    Returns:
        Single line starting with Answer: or empty string if not found
    """
    try:
        # Extract the first line matching ^Answer:\s*(.*)$
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^Answer:\s*.*$', line):
                return line
        return ""
    except Exception:
        return ""


def fuzzy_match_label(label: str, allowed_labels: List[str], max_distance: int = 1) -> Optional[str]:
    """
    Find fuzzy match for label in allowed list with case-insensitive and distance-based matching.
    
    Args:
        label: Input label to match
        allowed_labels: List of allowed labels
        max_distance: Maximum edit distance for matching (default: 1)
        
    Returns:
        Matched label from allowed list or None if no match found
    """
    if not allowed_labels:
        return label
    
    # Exact match (case-insensitive)
    for allowed in allowed_labels:
        if label.lower() == allowed.lower():
            return allowed
    
    # Fuzzy match with distance ≤ max_distance
    close_matches = get_close_matches(label.lower(), 
                                     [a.lower() for a in allowed_labels], 
                                     n=1, cutoff=0.6)
    
    if close_matches:
        # Find the original case version
        matched_lower = close_matches[0]
        for allowed in allowed_labels:
            if allowed.lower() == matched_lower:
                return allowed
    
    return None


def extract_raw_answer_only(text: str, pattern: str) -> Tuple[str, bool]:
    """
    Extract answer using raw regex matching only - NO fuzzy matching or corrections.
    Use this for metrics (accuracy/quality) to ensure unbiased evaluation.
    
    Args:
        text: Input text to search
        pattern: Regex pattern with one capture group for the answer
        
    Returns:
        Tuple of (extracted_answer, success_flag)
        - If exact regex match found: (answer, True)
        - If no match: ("", False)
    """
    try:
        regex = re.compile(pattern, re.MULTILINE)
        match = regex.search(text)
        if not match:
            return ("", False)
        return (match.group(1).strip(), True)
    except Exception:
        return ("", False)


def extract_strict_answer(text: str, pattern: str, allowed: Optional[List[str]] = None, allow_fuzzy: bool = False) -> Tuple[str, bool]:
    """
    Extract answer from text using regex pattern with optional fuzzy matching.
    Use this for format compliance checking (strict by default).
    
    Args:
        text: Input text to search
        pattern: Regex pattern with one capture group for the answer
        allowed: Optional list of allowed labels for validation
        allow_fuzzy: Whether to allow fuzzy matching (default: False for strict compliance)
        
    Returns:
        Tuple of (extracted_answer, success_flag)
        - If match found and valid: (answer, True)
        - If no match or invalid: ("", False)
    """
    try:
        # First, force to single line format
        single_line_text = force_singleline_answer(text)
        if not single_line_text:
            # If no Answer: line found, try original text
            single_line_text = text
        
        # Compile pattern with MULTILINE flag
        regex = re.compile(pattern, re.MULTILINE)
        match = regex.search(single_line_text)
        
        if not match:
            return ("", False)
            
        # Extract first capture group
        answer = match.group(1).strip()
        
        # Strict validation with optional fuzzy matching
        if allowed is not None:
            # First try exact match
            if answer in allowed:
                return (answer, True)
            
            # Try fuzzy matching only if explicitly allowed
            if allow_fuzzy:
                fuzzy_match = fuzzy_match_label(answer, allowed, max_distance=1)
                if fuzzy_match:
                    return (fuzzy_match, True)
            
            # No valid match found (strict mode or fuzzy failed)
            return ("", False)
                
        return (answer, True)
        
    except (re.error, IndexError, AttributeError):
        return ("", False)


def format_repair_prompt(base_prompt: str, allowed: Optional[List[str]] = None) -> str:
    """
    Enhance prompt with strict format requirements for repair attempts.
    
    Args:
        base_prompt: Original prompt text
        allowed: Optional list of allowed labels
        
    Returns:
        Enhanced prompt with strict format instructions
    """
    format_instruction = "\n\n絶対に次の形式のみで返答してください: `Answer: <label>`。前後に一切の説明文を付けないこと。"
    
    if allowed:
        labels_str = ", ".join(f"'{label}'" for label in allowed)
        format_instruction += f"\nラベルは {{{labels_str}}} のいずれかを使用してください。"
    
    return base_prompt + format_instruction


def json_default(obj: Any) -> Any:
    """
    JSON serialization helper for numpy and custom types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serializable representation of the object
        
    Raises:
        TypeError: If object cannot be serialized
    """
    # Handle numpy types
    import numpy as np
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle custom types
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '_asdict'):  # namedtuple
        return obj._asdict()
    
    # Fallback to string representation
    return str(obj)


class StrictOutputValidator:
    """
    Validator for strict output format compliance.
    """
    
    def __init__(self, pattern: str, allowed_labels: Optional[List[str]] = None):
        """
        Initialize validator.
        
        Args:
            pattern: Regex pattern for extraction
            allowed_labels: Optional list of valid labels
        """
        self.pattern = pattern
        self.allowed_labels = allowed_labels
        self.total_attempts = 0
        self.successful_extractions = 0
        
    def validate(self, text: str) -> Tuple[str, bool]:
        """
        Validate and extract answer from text.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (answer, is_valid)
        """
        self.total_attempts += 1
        answer, is_valid = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        if is_valid:
            self.successful_extractions += 1
            
        return answer, is_valid
    
    def get_compliance_rate(self) -> float:
        """
        Get current format compliance rate.
        
        Returns:
            Compliance rate as float between 0.0 and 1.0
        """
        if self.total_attempts == 0:
            return 0.0
        return self.successful_extractions / self.total_attempts
    
    def get_stats(self) -> dict:
        """
        Get detailed compliance statistics.
        
        Returns:
            Dictionary with compliance metrics
        """
        return {
            'format_compliance_total': self.total_attempts,
            'format_compliance_ok': self.successful_extractions,
            'format_compliance_rate': self.get_compliance_rate()
        }


# Default pattern for LaMP-2 Answer format
DEFAULT_ANSWER_PATTERN = r"^Answer:\s*([A-Za-z0-9_\- ]+)\s*$"