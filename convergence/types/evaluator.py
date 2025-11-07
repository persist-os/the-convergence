"""
Evaluator protocol for Convergence SDK.
"""

from typing import Protocol, Dict, Optional


class Evaluator(Protocol):
    """Protocol for evaluation functions."""
    
    def __call__(
        self,
        prediction: Dict,
        expected: Dict,
        *,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Evaluate a prediction against expected results.
        
        Args:
            prediction: Model/API prediction
            expected: Expected results
            context: Optional context information
            
        Returns:
            Dictionary of metric names to scores
        """
        ...

