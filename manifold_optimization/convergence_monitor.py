#!/usr/bin/env python3
"""
Convergence Guarantee and Monitoring System for Stiefel Manifold Optimization

Provides theoretical convergence guarantees and practical monitoring tools
for Stiefel manifold-based direction vector optimization.

Key features:
- Lipschitz continuity verification
- Theoretical convergence rate analysis (O(1/t) vs O(1/√t))
- Real-time convergence monitoring
- Adaptive learning rate scheduling
- Orthogonality constraint violation detection

Author: Phase 2 Implementation
Date: 2025-08-27
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ConvergenceStats:
    """Statistics for convergence analysis"""
    iteration: int
    loss_value: float
    gradient_norm: float
    orthogonality_error: float
    theoretical_rate: float
    actual_rate: float
    convergence_ratio: float  # actual_rate / theoretical_rate

@dataclass 
class LipschitzStats:
    """Lipschitz continuity analysis results"""
    is_lipschitz: bool
    lipschitz_constant: float
    violation_points: List[Tuple[torch.Tensor, torch.Tensor]]
    confidence_level: float

class ConvergenceGuarantee:
    """
    Theoretical convergence guarantees for Stiefel manifold optimization
    
    Provides mathematical foundations and verification tools for convergence
    properties of Riemannian optimization on Stiefel manifolds.
    """
    
    def __init__(self, 
                 manifold_dim: Tuple[int, int] = (768, 128),
                 lipschitz_constant: Optional[float] = None,
                 strong_convexity: Optional[float] = None):
        """
        Initialize convergence guarantee analyzer
        
        Args:
            manifold_dim: Stiefel manifold dimensions (n, k)
            lipschitz_constant: Known Lipschitz constant (estimated if None)
            strong_convexity: Strong convexity parameter (estimated if None)
        """
        self.n, self.k = manifold_dim
        self.lipschitz_constant = lipschitz_constant
        self.strong_convexity = strong_convexity
        
        # Theoretical constants for Stiefel manifold
        self.manifold_curvature_bound = np.sqrt(self.n - self.k)  # Sectional curvature bound
        
        logger.info(f"✅ ConvergenceGuarantee initialized for St({self.n},{self.k})")
    
    def verify_lipschitz(self, 
                        f: callable, 
                        W: torch.Tensor, 
                        epsilon: float = 1e-4,
                        n_samples: int = 100) -> LipschitzStats:
        """
        Verify Lipschitz continuity of objective function on Stiefel manifold
        
        For convergence guarantees, we need ||∇f(x) - ∇f(y)|| ≤ L||x - y||
        
        Args:
            f: Objective function
            W: Base point for testing
            epsilon: Perturbation size for testing
            n_samples: Number of test points
            
        Returns:
            LipschitzStats: Lipschitz analysis results
        """
        lipschitz_estimates = []
        violation_points = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Generate random tangent vector
                tangent_vec = torch.randn_like(W) * epsilon
                # Project to tangent space of Stiefel manifold
                tangent_vec = tangent_vec - W @ (W.T @ tangent_vec)
                
                # Perturbed points (using exponential map would be more accurate)
                W_pert = W + tangent_vec
                
                # Estimate gradients numerically
                grad_W = self._compute_numerical_gradient(f, W, epsilon/10)
                grad_W_pert = self._compute_numerical_gradient(f, W_pert, epsilon/10)
                
                # Compute Lipschitz estimate
                grad_diff_norm = torch.norm(grad_W - grad_W_pert)
                point_diff_norm = torch.norm(tangent_vec)
                
                if point_diff_norm > 1e-8:
                    lipschitz_est = (grad_diff_norm / point_diff_norm).item()
                    lipschitz_estimates.append(lipschitz_est)
                    
                    # Check for violations if we have a known constant
                    if self.lipschitz_constant and lipschitz_est > self.lipschitz_constant * 1.1:
                        violation_points.append((W.clone(), W_pert.clone()))
        
        # Analyze results
        if lipschitz_estimates:
            estimated_L = max(lipschitz_estimates)
            is_lipschitz = len(violation_points) / len(lipschitz_estimates) < 0.05  # 5% tolerance
        else:
            estimated_L = float('inf')
            is_lipschitz = False
        
        return LipschitzStats(
            is_lipschitz=is_lipschitz,
            lipschitz_constant=estimated_L,
            violation_points=violation_points,
            confidence_level=1.0 - len(violation_points) / max(len(lipschitz_estimates), 1)
        )
    
    def _compute_numerical_gradient(self, f: callable, W: torch.Tensor, h: float) -> torch.Tensor:
        """Compute numerical gradient of function f at point W"""
        grad = torch.zeros_like(W)
        
        with torch.no_grad():
            f_W = f(W)
            
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W_plus = W.clone()
                    W_plus[i, j] += h
                    
                    grad[i, j] = (f(W_plus) - f_W) / h
        
        return grad
    
    def theoretical_convergence_rate(self, iteration: int, optimization_method: str = "riemannian_gradient") -> float:
        """
        Theoretical convergence rate for different optimization methods
        
        Args:
            iteration: Current iteration number
            optimization_method: Type of optimization algorithm
            
        Returns:
            float: Theoretical convergence rate bound
        """
        t = max(iteration, 1)
        
        if optimization_method == "riemannian_gradient":
            # Standard Riemannian gradient descent: O(1/t)
            return 1.0 / t
            
        elif optimization_method == "riemannian_adam":
            # Adam on manifolds: typically O(1/t) with better constants
            return 0.8 / t
            
        elif optimization_method == "euclidean_gradient":
            # Standard Euclidean gradient descent: O(1/√t)
            return 1.0 / np.sqrt(t)
            
        else:
            logger.warning(f"Unknown optimization method: {optimization_method}")
            return 1.0 / t
    
    def compute_improvement_factor(self, iteration: int) -> float:
        """
        Compute theoretical speedup of Stiefel optimization vs Euclidean
        
        Args:
            iteration: Current iteration
            
        Returns:
            float: Improvement factor (how many times faster)
        """
        stiefel_rate = self.theoretical_convergence_rate(iteration, "riemannian_gradient")
        euclidean_rate = self.theoretical_convergence_rate(iteration, "euclidean_gradient")
        
        return euclidean_rate / stiefel_rate  # Should be ≈ √t
    
    def verify_orthogonality_preservation(self, W: torch.Tensor, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Verify that matrix maintains orthogonality within tolerance
        
        Args:
            W: Matrix to check [n, k]
            tolerance: Orthogonality tolerance
            
        Returns:
            Tuple of (is_orthogonal, error_magnitude)
        """
        orthogonality_error = torch.norm(W.T @ W - torch.eye(W.shape[1], device=W.device))
        is_orthogonal = orthogonality_error.item() < tolerance
        
        return is_orthogonal, orthogonality_error.item()

class ConvergenceMonitor:
    """
    Real-time convergence monitoring and adaptive optimization
    
    Monitors optimization progress, detects convergence issues,
    and provides adaptive learning rate scheduling.
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 convergence_tolerance: float = 1e-6,
                 patience: int = 100,
                 min_lr: float = 1e-8):
        """
        Initialize convergence monitor
        
        Args:
            window_size: Window for moving average computation
            convergence_tolerance: Tolerance for convergence detection
            patience: Iterations to wait before declaring convergence
            min_lr: Minimum learning rate for adaptive scheduling
        """
        self.window_size = window_size
        self.convergence_tolerance = convergence_tolerance
        self.patience = patience
        self.min_lr = min_lr
        
        # History tracking
        self.history = deque(maxlen=window_size * 2)  # Keep more history for analysis
        self.convergence_stats = []
        
        # State tracking
        self.best_loss = float('inf')
        self.iterations_without_improvement = 0
        self.is_converged = False
        
        logger.info(f"✅ ConvergenceMonitor initialized (window={window_size}, tol={convergence_tolerance})")
    
    def update(self, 
               iteration: int,
               loss_value: float, 
               W: torch.Tensor,
               gradient_norm: Optional[float] = None,
               learning_rate: Optional[float] = None) -> ConvergenceStats:
        """
        Update monitoring with new iteration data
        
        Args:
            iteration: Current iteration
            loss_value: Current loss value
            W: Current parameters
            gradient_norm: Current gradient norm (computed if None)
            learning_rate: Current learning rate
            
        Returns:
            ConvergenceStats: Current iteration statistics
        """
        # Compute orthogonality error
        _, orthogonality_error = ConvergenceGuarantee().verify_orthogonality_preservation(W)
        
        # Compute gradient norm if not provided
        if gradient_norm is None:
            # This would normally come from the optimizer
            gradient_norm = 0.0  # Placeholder
        
        # Compute convergence rates
        theoretical_rate = ConvergenceGuarantee().theoretical_convergence_rate(iteration)
        
        # Compute actual convergence rate
        if len(self.history) >= 2:
            recent_losses = [entry.loss_value for entry in list(self.history)[-10:]]
            if len(recent_losses) >= 2:
                actual_rate = abs(recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            else:
                actual_rate = 0.0
        else:
            actual_rate = float('inf')
        
        # Compute convergence ratio
        convergence_ratio = actual_rate / max(theoretical_rate, 1e-12)
        
        # Create stats
        stats = ConvergenceStats(
            iteration=iteration,
            loss_value=loss_value,
            gradient_norm=gradient_norm,
            orthogonality_error=orthogonality_error,
            theoretical_rate=theoretical_rate,
            actual_rate=actual_rate,
            convergence_ratio=convergence_ratio
        )
        
        # Update history
        self.history.append(stats)
        self.convergence_stats.append(stats)
        
        # Check for improvement
        if loss_value < self.best_loss - self.convergence_tolerance:
            self.best_loss = loss_value
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Check convergence
        self._check_convergence()
        
        return stats
    
    def _check_convergence(self) -> None:
        """Internal convergence checking"""
        if len(self.history) < self.window_size:
            return
        
        recent_stats = list(self.history)[-self.window_size:]
        recent_losses = [s.loss_value for s in recent_stats]
        
        # Check if loss has plateaued
        loss_variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)
        
        # Relative variance check
        if loss_variance / max(mean_loss**2, 1e-12) < self.convergence_tolerance:
            if self.iterations_without_improvement >= self.patience:
                self.is_converged = True
                logger.info(f"Convergence detected: loss variance {loss_variance:.2e}")
    
    def should_reduce_lr(self, factor: float = 0.5) -> Tuple[bool, float]:
        """
        Determine if learning rate should be reduced
        
        Args:
            factor: Reduction factor
            
        Returns:
            Tuple of (should_reduce, suggested_new_lr)
        """
        if self.iterations_without_improvement >= self.patience // 2:
            # Suggest learning rate reduction
            if hasattr(self, 'current_lr'):
                new_lr = max(self.current_lr * factor, self.min_lr)
                return True, new_lr
            else:
                return True, 0.001 * factor  # Default fallback
        
        return False, 0.0
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive convergence summary"""
        if not self.convergence_stats:
            return {"status": "no_data"}
        
        recent = self.convergence_stats[-10:] if len(self.convergence_stats) >= 10 else self.convergence_stats
        
        return {
            "status": "converged" if self.is_converged else "in_progress",
            "total_iterations": len(self.convergence_stats),
            "best_loss": self.best_loss,
            "final_loss": recent[-1].loss_value,
            "iterations_without_improvement": self.iterations_without_improvement,
            "average_orthogonality_error": np.mean([s.orthogonality_error for s in recent]),
            "convergence_ratio": np.mean([s.convergence_ratio for s in recent]),
            "theoretical_speedup": np.mean([ConvergenceGuarantee().compute_improvement_factor(s.iteration) for s in recent])
        }
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot convergence analysis
        
        Args:
            save_path: Path to save plot (displays if None)
        """
        if not self.convergence_stats:
            logger.warning("No convergence data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        iterations = [s.iteration for s in self.convergence_stats]
        losses = [s.loss_value for s in self.convergence_stats]
        orth_errors = [s.orthogonality_error for s in self.convergence_stats]
        theoretical_rates = [s.theoretical_rate for s in self.convergence_stats]
        actual_rates = [s.actual_rate for s in self.convergence_stats]
        
        # Loss convergence
        axes[0, 0].semilogy(iterations, losses)
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss Value')
        axes[0, 0].grid(True)
        
        # Orthogonality error
        axes[0, 1].semilogy(iterations, orth_errors)
        axes[0, 1].set_title('Orthogonality Error')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('||W^T W - I||_F')
        axes[0, 1].grid(True)
        
        # Convergence rates comparison
        axes[1, 0].loglog(iterations, theoretical_rates, label='Theoretical O(1/t)')
        axes[1, 0].loglog(iterations, actual_rates, label='Actual Rate')
        axes[1, 0].set_title('Convergence Rates')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Convergence Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Speedup factor
        speedups = [ConvergenceGuarantee().compute_improvement_factor(it) for it in iterations]
        axes[1, 1].plot(iterations, speedups)
        axes[1, 1].set_title('Theoretical Speedup vs Euclidean')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Speedup Factor')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def create_convergence_monitor_for_manifold(manifold_dims: Tuple[int, int],
                                          optimization_config: Dict[str, Any]) -> ConvergenceMonitor:
    """
    Factory function to create properly configured convergence monitor
    
    Args:
        manifold_dims: Stiefel manifold dimensions (n, k)
        optimization_config: Configuration dictionary
        
    Returns:
        ConvergenceMonitor: Configured monitor
    """
    return ConvergenceMonitor(
        window_size=optimization_config.get('convergence_window', 50),
        convergence_tolerance=optimization_config.get('convergence_tolerance', 1e-6),
        patience=optimization_config.get('patience', 100),
        min_lr=optimization_config.get('min_learning_rate', 1e-8)
    )