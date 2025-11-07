"""
Type definitions for Convergence SDK.
"""

from .config import (
    ConvergenceConfig,
    ApiConfig,
    SearchSpaceConfig,
    RunnerConfig,
    EvaluationConfig,
    StorageConfig,
    AdaptersConfig,
    AgentConfig,
)
from .results import OptimizationRunResult
from .evaluator import Evaluator
from .runtime import (
    RuntimeArm,
    RuntimeArmState,
    RuntimeArmTemplate,
    RuntimeConfig,
    RuntimeDecision,
    RuntimeSelection,
    SelectionStrategyConfig,
)

# Rebuild RuntimeConfig to resolve RewardEvaluatorConfig forward reference (Pydantic v2)
try:
    from convergence.runtime.reward_evaluator import RewardEvaluatorConfig
    RuntimeConfig.model_rebuild()
except ImportError:
    pass  # Will be rebuilt when RewardEvaluatorConfig is imported

__all__ = [
    "ConvergenceConfig",
    "ApiConfig",
    "SearchSpaceConfig",
    "RunnerConfig",
    "EvaluationConfig",
    "StorageConfig",
    "AdaptersConfig",
    "AgentConfig",
    "OptimizationRunResult",
    "Evaluator",
    "RuntimeArm",
    "RuntimeArmState",
    "RuntimeArmTemplate",
    "RuntimeConfig",
    "RuntimeDecision",
    "RuntimeSelection",
    "SelectionStrategyConfig",
]

