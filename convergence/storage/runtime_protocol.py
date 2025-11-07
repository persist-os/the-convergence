"""Storage abstraction for Convergence runtime loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class RuntimeStorageProtocol(Protocol):
    """Backend-facing storage contract for runtime decisions and arms."""

    async def get_arms(self, *, user_id: str, agent_type: str) -> List[Any]:
        """Return the current arm snapshots for the user/agent pair."""

    async def initialize_arms(
        self,
        *,
        user_id: str,
        agent_type: str,
        arms: List[Dict[str, Any]],
    ) -> Any:
        """Seed arms for cold-start users (idempotent)."""

    async def create_decision(
        self,
        *,
        user_id: str,
        agent_type: str,
        arm_pulled: str,
        strategy_params: Dict[str, Any],
        arms_snapshot: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist a decision event and return its identifier (if available)."""

    async def update_performance(
        self,
        *,
        user_id: str,
        agent_type: str,
        decision_id: str,
        reward: float,
        engagement: Optional[float] = None,
        grading: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        computed_update: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Apply a reward update to the selected arm.
        
        If `computed_update` is provided, use those pre-computed values instead of
        computing them. This allows the SDK to centralize all Bayesian update
        computation logic, while storage backends only handle persistence.
        
        Args:
            computed_update: Optional pre-computed Bayesian update values from SDK.
                If provided, should contain: alpha, beta, total_pulls, total_reward,
                avg_reward, mean_estimate, confidence_interval.
        """

    async def get_decision(
        self,
        *,
        user_id: str,
        decision_id: str,
    ) -> Dict[str, Any]:
        """Fetch a recorded decision for auditing or reward calculation."""



