"""
The Convergence: API Optimization Framework.

Finds optimal API configurations through evolutionary optimization powered by
an agent society using RLP (reasoning), SAO (self-improvement), MAB (exploration),
and hierarchical learning.

Usage:
    convergence optimize config.yaml
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
]
