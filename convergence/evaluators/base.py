"""
Base evaluator class for custom evaluators.

Provides a standard interface and utilities for building custom evaluation functions.
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.
    
    Custom evaluators should inherit from this class and implement the evaluate() method.
    
    Example:
        class MyCustomEvaluator(BaseEvaluator):
            @staticmethod
            def evaluate(result: Any, expected: Any, params: Dict, metric: Optional[str] = None) -> float:
                # Your evaluation logic here
                return 0.85
    """
    
    @staticmethod
    @abstractmethod
    def evaluate(
        result: Any,
        expected: Any,
        params: Dict[str, Any],
        metric: Optional[str] = None
    ) -> float:
        """
        Evaluate the API response result.
        
        Args:
            result: The API response result to evaluate
            expected: Expected output/criteria from test case
            params: API parameters used for this call
            metric: Optional specific metric being evaluated
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


def score_wrapper(func):
    """
    Decorator to ensure scores are clamped between 0 and 1.
    
    Usage:
        @score_wrapper
        def my_evaluator(result, expected, params, metric=None):
            return some_score
    """
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return max(0.0, min(1.0, float(score)))
    return wrapper


