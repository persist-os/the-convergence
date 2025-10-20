"""
The Convergence: API Optimization Framework.

Finds optimal API configurations through evolutionary optimization powered by
an agent society using RLP (reasoning), SAO (self-improvement), MAB (exploration),
and hierarchical learning.

Usage:
    CLI: convergence optimize config.yaml
    SDK: from convergence import run_optimization
"""

__version__ = "0.1.0"

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
from convergence.sdk import run_optimization, load_config_from_file

__all__ = [
    # Core protocols
    "LLMProvider",
    "MABStrategy",
    "MemorySystem",
    "Agent",
    "Plugin",
    "ConvergenceConfig",
    "PluginRegistry",
    # Optimization
    "ConfigLoader",
    "OptimizationRunner",
    # SDK
    "run_optimization",
    "load_config_from_file",
]
