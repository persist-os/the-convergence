"""
Legacy System - Permanent optimization history and winner tracking.

This module provides session-based tracking of optimization runs,
allowing results to build on each other across multiple sessions.

Key Features:
- Session-based organization (never expires)
- Winner tracking per test case
- Full audit trail of all decisions
- Works with any API type
- No RL dependency
- Optional external trackers (MLflow, Aim, Weave)

Usage:
    from convergence.legacy import LegacyStore, LegacyConfig
    
    config = LegacyConfig(enabled=True, tracking_backend="builtin")
    store = LegacyStore(config)
    
    await store.record_run(run_data)
    winner = await store.get_winner("bedtime_story", "openai")
"""

from convergence.legacy.models import (
    LegacyConfig,
    Session,
    OptimizationRun,
    TestCaseResult,
    TestCaseWinner,
    RunLineage,
    DecisionLog,
    TrackingBackend,
)
from convergence.legacy.store import LegacyStore

__all__ = [
    "LegacyConfig",
    "Session",
    "OptimizationRun",
    "TestCaseResult",
    "TestCaseWinner",
    "RunLineage",
    "DecisionLog",
    "TrackingBackend",
    "LegacyStore",
]

