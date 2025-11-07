"""
Result types for Convergence SDK.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel


class OptimizationRunResult(BaseModel):
    """Result of optimization run."""
    success: bool
    best_config: Dict[str, Any]
    best_score: float
    configs_generated: int
    generations_run: int
    optimization_run_id: str
    timestamp: datetime
    events: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

