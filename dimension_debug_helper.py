#!/usr/bin/env python3
"""
Dimension Helper for Chameleon Vector Fitting
"""
import torch
import torch.nn.functional as F

def fit_to_hidden(vec, hidden_dim, device, dtype):
    """
    Universal dimension fitter for direction vectors
    Handles both torch tensors and numpy arrays
    """
    # Convert to tensor
    v = torch.as_tensor(vec, device=device, dtype=dtype).view(-1)
    
    if v.numel() == hidden_dim:
        return v
    
    # If longer, truncate
    if v.numel() > hidden_dim:
        return v[:hidden_dim]
    
    # If shorter, pad with zeros
    return F.pad(v, (0, hidden_dim - v.numel()))