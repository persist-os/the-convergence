"""Fallback runtime storage implementation that raises configuration errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .runtime_protocol import RuntimeStorageProtocol


class UnconfiguredRuntimeStorage(RuntimeStorageProtocol):
    """Runtime storage that raises helpful errors when not configured."""

    def __init__(self, *, system: str):
        self.system = system

    def _error(self) -> RuntimeError:
        return RuntimeError(
            "Runtime storage is not configured for system "
            f"'{self.system}'. Provide a RuntimeStorageProtocol implementation "
            "via configure_runtime(..., storage=your_adapter)."
        )

    async def get_arms(self, *, user_id: str, agent_type: str) -> List[Any]:
        raise self._error()

    async def initialize_arms(
        self,
        *,
        user_id: str,
        agent_type: str,
        arms: List[Dict[str, Any]],
    ) -> Any:
        raise self._error()

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
        raise self._error()

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
    ) -> Any:
        raise self._error()

    async def get_decision(
        self,
        *,
        user_id: str,
        decision_id: str,
    ) -> Dict[str, Any]:
        raise self._error()


__all__ = ["UnconfiguredRuntimeStorage"]



