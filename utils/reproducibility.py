#!/usr/bin/env python3
"""
Reproducibility utilities for Chameleon experiments.

Usage:
    from utils.reproducibility import set_reproducible_seeds
    set_reproducible_seeds(42)
"""

import random
import os
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int = 42):
    """
    Set all random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set deterministic mode for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For additional reproducibility in some PyTorch operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Optional: Set deterministic algorithms (can be slow)
    # Uncomment if maximum reproducibility is needed
    # torch.use_deterministic_algorithms(True)
    
    logger.info(f"Set reproducible seeds: {seed}")


def set_generation_deterministic():
    """
    Set generation-specific deterministic settings.
    Ensures do_sample=False and removes stochastic parameters.
    
    Returns:
        dict: Deterministic generation config
    """
    config = {
        "do_sample": False,          # Greedy decoding
        "temperature": None,         # Remove temperature
        "top_p": None,              # Remove top_p  
        "top_k": None,              # Remove top_k
        "repetition_penalty": 1.0,   # Neutral repetition penalty
        "length_penalty": 1.0,       # Neutral length penalty
    }
    
    # Remove None values to avoid passing them to generation
    return {k: v for k, v in config.items() if v is not None}


def get_reproducible_env_vars():
    """
    Get environment variables for maximum reproducibility.
    
    Returns:
        dict: Environment variables to set
    """
    return {
        "PYTHONHASHSEED": "42",
        "CUDA_LAUNCH_BLOCKING": "1",  # Synchronous CUDA operations
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # Deterministic CUBLAS
    }


def apply_reproducible_env():
    """Apply reproducible environment variables to current process."""
    env_vars = get_reproducible_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.debug(f"Set {key}={value}")


def check_reproducibility_status():
    """
    Check and report current reproducibility status.
    
    Returns:
        dict: Status of various reproducibility settings
    """
    status = {
        "torch_deterministic": torch.backends.cudnn.deterministic,
        "torch_benchmark": torch.backends.cudnn.benchmark,
        "cuda_available": torch.cuda.is_available(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }
    
    if torch.cuda.is_available():
        status["cuda_device_count"] = torch.cuda.device_count()
        status["current_device"] = torch.cuda.current_device()
    
    try:
        # Check if deterministic algorithms are enabled
        # This might not be available in all PyTorch versions
        status["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        status["deterministic_algorithms"] = "not_available"
    
    return status


def log_reproducibility_status():
    """Log current reproducibility status for debugging."""
    status = check_reproducibility_status()
    logger.info("Reproducibility status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Test reproducibility setup
    print("Testing reproducibility setup...")
    
    set_reproducible_seeds(42)
    apply_reproducible_env()
    
    status = check_reproducibility_status()
    print("\nReproducibility Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test deterministic generation
    gen_config = set_generation_deterministic()
    print(f"\nDeterministic generation config: {gen_config}")
    
    print("\n✅ Reproducibility setup complete!")