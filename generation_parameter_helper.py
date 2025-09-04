#!/usr/bin/env python3
"""
Generation Parameter Helper

Provides consistent generation parameter handling across the system
to eliminate do_sample conflicts and ensure proper observation of editing effects.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GenerationParameterHelper:
    """
    Unified generation parameter management
    """
    
    @staticmethod
    def get_clean_generation_params(
        mode: str = "sampling",  # "greedy" or "sampling" 
        temperature: Optional[float] = None,
        max_new_tokens: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get clean generation parameters without conflicts
        
        Args:
            mode: Generation mode ("greedy" for deterministic, "sampling" for stochastic)
            temperature: Override temperature (None to use mode default)
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Clean generation parameters dictionary
        """
        if mode == "greedy" or (temperature is not None and temperature == 0.0):
            # Deterministic generation
            params = {
                'do_sample': False,
                'max_new_tokens': max_new_tokens,
                'use_cache': True
            }
            # Don't include sampling parameters to avoid warnings
            
        elif mode == "sampling" or (temperature is not None and temperature > 0.0):
            # Stochastic generation for observing editing effects
            params = {
                'do_sample': True,
                'temperature': temperature if temperature is not None else 0.7,
                'top_p': kwargs.get('top_p', 0.9),
                'max_new_tokens': max_new_tokens,
                'use_cache': True
            }
        else:
            raise ValueError(f"Invalid generation mode: {mode}")
        
        # Add pad token handling
        if 'pad_token_id' in kwargs:
            params['pad_token_id'] = kwargs['pad_token_id']
        
        # Remove None values to avoid conflicts
        params = {k: v for k, v in params.items() if v is not None}
        
        logger.debug(f"Generated clean params for mode '{mode}': {params}")
        return params
    
    @staticmethod
    def validate_generation_config(params: Dict[str, Any]) -> bool:
        """
        Validate generation parameters for consistency
        
        Args:
            params: Generation parameters to validate
            
        Returns:
            True if parameters are consistent, False otherwise
        """
        do_sample = params.get('do_sample', False)
        temperature = params.get('temperature')
        
        if do_sample == False:
            # Deterministic generation - should not have sampling parameters
            if temperature is not None and temperature > 0.0:
                logger.warning("Inconsistent: do_sample=False but temperature>0")
                return False
            if 'top_p' in params:
                logger.warning("Inconsistent: do_sample=False but top_p present")
                return False
        elif do_sample == True:
            # Stochastic generation - should have temperature
            if temperature is None or temperature <= 0.0:
                logger.warning("Inconsistent: do_sample=True but temperature<=0 or None")
                return False
        
        return True


def get_generation_params_for_editing(
    enable_sampling: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 10
) -> Dict[str, Any]:
    """
    Get generation parameters optimized for observing Chameleon editing effects
    
    Args:
        enable_sampling: Whether to enable sampling (required to observe editing)
        temperature: Temperature for sampling (>0 required for editing observation)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generation parameters dictionary
    """
    helper = GenerationParameterHelper()
    
    if enable_sampling:
        return helper.get_clean_generation_params(
            mode="sampling",
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    else:
        return helper.get_clean_generation_params(
            mode="greedy",
            max_new_tokens=max_new_tokens
        )
