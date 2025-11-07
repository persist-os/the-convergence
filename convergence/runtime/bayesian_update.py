"""Bayesian update computation for Beta distribution (Thompson Sampling).

This module provides the single source of truth for all Bayesian update calculations
used in multi-armed bandit learning. All storage backends receive pre-computed values
from this module, eliminating duplication across different storage implementations.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from convergence.types.runtime import RuntimeArm


def compute_bayesian_update(arm: RuntimeArm, reward: float) -> Dict[str, Any]:
    """
    Compute Bayesian update for Beta distribution given an arm and observed reward.
    
    This is the single source of truth for all Bayesian update calculations.
    Storage backends should use the values returned by this function rather than
    recomputing the math, ensuring consistency across all implementations.
    
    Formula:
    - Beta distribution: Beta(alpha, beta)
    - Update: alpha_new = alpha_old + reward, beta_new = beta_old + (1 - reward)
    - Mean: mean = alpha / (alpha + beta)
    - Variance: var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    - 95% CI: mean ± 1.96 * std_dev
    
    Args:
        arm: RuntimeArm with current alpha, beta, total_pulls, total_reward
        reward: Observed reward value (will be clamped to [0, 1])
        
    Returns:
        Dictionary containing all computed update values:
        {
            "alpha": float,
            "beta": float,
            "total_pulls": int,
            "total_reward": float,
            "avg_reward": float,
            "mean_estimate": float,
            "confidence_interval": {
                "lower": float,
                "upper": float,
            },
        }
    """
    # Input validation: clamp reward to [0, 1]
    reward = max(0.0, min(1.0, reward))
    
    # Bayesian update (Beta distribution)
    success_weight = reward
    failure_weight = 1.0 - reward
    
    new_alpha = arm.alpha + success_weight
    new_beta = arm.beta + failure_weight
    new_total_pulls = arm.total_pulls + 1
    new_total_reward = arm.total_reward + reward
    new_avg_reward = new_total_reward / new_total_pulls
    new_mean_estimate = new_alpha / (new_alpha + new_beta)
    
    # Confidence interval (95% CI for Beta distribution)
    # Variance formula: var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    variance = (new_alpha * new_beta) / ((new_alpha + new_beta) ** 2 * (new_alpha + new_beta + 1))
    std_dev = math.sqrt(variance)
    
    confidence_interval = {
        "lower": max(0.0, new_mean_estimate - 1.96 * std_dev),
        "upper": min(1.0, new_mean_estimate + 1.96 * std_dev),
    }
    
    return {
        "alpha": new_alpha,
        "beta": new_beta,
        "total_pulls": new_total_pulls,
        "total_reward": new_total_reward,
        "avg_reward": new_avg_reward,
        "mean_estimate": new_mean_estimate,
        "confidence_interval": confidence_interval,
    }

