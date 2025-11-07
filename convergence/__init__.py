"""
The Convergence: API Optimization Framework.

Finds optimal API configurations through evolutionary optimization powered by
an agent society using RLP (reasoning), SAO (self-improvement), MAB (exploration),
and hierarchical learning.

Usage:
    CLI: convergence optimize config.yaml
    SDK: from convergence import run_optimization
"""

__version__ = "0.1.3"

from convergence.core.protocols import (
    LLMProvider,
    MABStrategy,
    MemorySystem,
    Agent,
    Plugin,
)
from convergence.core.config import ConvergenceConfig
from convergence.core.registry import PluginRegistry

# Optimization components
from convergence.optimization.config_loader import ConfigLoader
from convergence.optimization.runner import OptimizationRunner

# SDK interface (for programmatic use)
from convergence.sdk import run_optimization

# Runtime interface
from convergence.runtime.online import (
    configure as configure_runtime,
    select as runtime_select,
    update as runtime_update,
    get_decision as runtime_get_decision,
)
from convergence.runtime.evolution import evolve_arms as runtime_evolve_arms

# Type definitions
from convergence.types import (
    ConvergenceConfig as ConvergenceConfigSDK,
    RuntimeConfig as RuntimeConfigSDK,
    RuntimeSelection,
    RuntimeDecision,
    SelectionStrategyConfig,
    RuntimeArmTemplate,
)
from convergence.runtime.reward_evaluator import (
    RuntimeRewardEvaluator,
    RewardEvaluatorConfig,
    RewardMetricConfig,
)

# Rebuild RuntimeConfig to resolve RewardEvaluatorConfig forward reference (Pydantic v2)
# This ensures RuntimeConfig can be instantiated after RewardEvaluatorConfig is imported
from convergence.types import RuntimeConfig
RuntimeConfig.model_rebuild()

__all__ = [
    # Core protocols
    "LLMProvider",
    "MABStrategy",
    "MemorySystem",
    "Agent",
    "Plugin",
    "ConvergenceConfig",  # From core.config
    "ConvergenceConfigSDK",  # From types (programmatic SDK config)
    "PluginRegistry",
    # Optimization
    "ConfigLoader",
    "OptimizationRunner",
    # SDK
    "run_optimization",
    # Runtime
    "configure_runtime",
    "runtime_select",
    "runtime_update",
    "runtime_get_decision",
    "runtime_evolve_arms",
    "RuntimeConfigSDK",
    "RuntimeSelection",
    "RuntimeDecision",
    "SelectionStrategyConfig",
    "RuntimeRewardEvaluator",
    "RewardEvaluatorConfig",
    "RewardMetricConfig",
    "RuntimeArmTemplate",
]
