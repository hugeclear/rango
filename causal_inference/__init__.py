"""
Causal Inference Integration for Chameleon
Adds causal constraints to personalized LLM editing

Components:
- causal_graph_builder: PC algorithm for causal discovery
- temporal_constraints: Light cone temporal causality
- do_calculus: Average Treatment Effect (ATE) estimation
"""

from .causal_graph_builder import CausalGraphBuilder
from .temporal_constraints import TemporalConstraintManager
from .do_calculus import DoCalculusEstimator

__all__ = [
    'CausalGraphBuilder',
    'TemporalConstraintManager', 
    'DoCalculusEstimator'
]

__version__ = '1.0.0'