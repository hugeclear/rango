"""
Manifold Optimization Module for Chameleon
Implements Stiefel manifold optimization for orthogonal direction vectors

Components:
- stiefel_optimizer: Stiefel manifold projection and geodesic updates
- convergence_monitor: Convergence guarantees and monitoring
- manifold_evaluator: Extended evaluator with manifold optimization

Author: Phase 2 Implementation - Stiefel Manifold Integration
Date: 2025-08-27
"""

from .stiefel_optimizer import StiefelProjector, StiefelOptimizer
from .convergence_monitor import ConvergenceGuarantee, ConvergenceMonitor

__all__ = [
    'StiefelProjector',
    'StiefelOptimizer', 
    'ConvergenceGuarantee',
    'ConvergenceMonitor'
]

__version__ = '2.0.0'