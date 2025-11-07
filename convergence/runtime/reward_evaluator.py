"""
Runtime reward evaluation system for multi-armed bandit learning.

Provides abstract reward computation from multiple metric signals,
similar to the optimization Evaluator pattern but designed for runtime MAB.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field


class RewardMetricConfig(BaseModel):
    """Configuration for a single reward metric."""
    
    name: str = Field(..., description="Metric name (e.g., 'explicit_rating', 'engagement_score')")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in reward aggregation (0-1)")
    normalize: bool = Field(True, description="Whether to normalize signal to [0, 1]")


class CustomRewardEvaluatorConfig(BaseModel):
    """Configuration for custom reward evaluator function."""
    
    enabled: bool = Field(False, description="Whether custom evaluator is enabled")
    module: Optional[str] = Field(None, description="Module name containing evaluator function")
    function: Optional[str] = Field(None, description="Function name to call")


class RewardEvaluatorConfig(BaseModel):
    """Configuration for reward evaluation."""
    
    metrics: Dict[str, RewardMetricConfig] = Field(..., description="Metric name -> config mapping")
    custom_evaluator: Optional[CustomRewardEvaluatorConfig] = Field(
        None, description="Optional custom evaluator for complex reward logic"
    )


class RuntimeRewardEvaluator:
    """
    Evaluates reward from normalized metric signals using weighted aggregation.
    
    Similar to optimization Evaluator but designed for runtime MAB reward computation.
    Accepts normalized signals (0-1 scores) and applies configured weights.
    """
    
    def __init__(
        self,
        config: RewardEvaluatorConfig,
        custom_evaluator_callable: Optional[Callable] = None
    ):
        """
        Initialize reward evaluator.
        
        Args:
            config: Reward evaluator configuration with metrics and weights
            custom_evaluator_callable: Optional callable evaluator function (programmatic mode)
        """
        self.config = config
        self.custom_evaluator = custom_evaluator_callable
        
        # Load custom evaluator from config if enabled and no callable provided
        if config.custom_evaluator and config.custom_evaluator.enabled and not custom_evaluator_callable:
            self._load_custom_evaluator()
    
    def _load_custom_evaluator(self) -> None:
        """Load custom evaluator function from module (similar to optimization Evaluator)."""
        if not self.config.custom_evaluator:
            return
        
        import importlib
        import importlib.util
        from pathlib import Path
        
        module_name = self.config.custom_evaluator.module
        function_name = self.config.custom_evaluator.function
        
        if not module_name or not function_name:
            return
        
        # Try built-in evaluators
        try:
            module = importlib.import_module(f"convergence.evaluators.{module_name}")
            self.custom_evaluator = getattr(module, function_name)
            return
        except (ImportError, AttributeError):
            pass
        
        # Try standard module import
        try:
            module = importlib.import_module(module_name)
            self.custom_evaluator = getattr(module, function_name)
            return
        except (ImportError, AttributeError):
            pass
    
    def evaluate(self, signals: Dict[str, float]) -> float:
        """
        Evaluate reward from normalized metric signals.
        
        Applies weighted aggregation: sum(metric_score * metric_weight) / sum(weights)
        
        Args:
            signals: Dict of metric_name -> normalized score (0-1)
            
        Returns:
            Aggregated reward score (0-1)
        """
        # If custom evaluator exists, use it
        if self.custom_evaluator:
            try:
                # Custom evaluator receives signals dict and returns reward
                result = self.custom_evaluator(signals)
                if isinstance(result, (int, float)):
                    return max(0.0, min(1.0, float(result)))
                # If dict, extract reward field or aggregate
                if isinstance(result, dict):
                    return max(0.0, min(1.0, float(result.get("reward", 0.0))))
            except Exception as e:
                # Fall back to weighted aggregation on error
                pass
        
        # Weighted aggregation (same pattern as optimization Evaluator)
        total_weight = sum(metric.weight for metric in self.config.metrics.values())
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = 0.0
        for metric_name, metric_config in self.config.metrics.items():
            if metric_name in signals:
                score = signals[metric_name]
                
                # Normalize if needed
                if metric_config.normalize:
                    score = max(0.0, min(1.0, score))
                
                weighted_sum += score * metric_config.weight
        
        return max(0.0, min(1.0, weighted_sum / total_weight))

