"""Runtime-specific data models shared across SDK consumers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from convergence.runtime.reward_evaluator import RewardEvaluatorConfig

from pydantic import BaseModel, Field


class SelectionStrategyConfig(BaseModel):
    """Configuration for MAB selection strategy."""
    
    exploration_bonus: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Bonus added to under-explored arms (0-1)"
    )
    exploration_min_pulls: int = Field(
        5, ge=0,
        description="Minimum pulls required before removing exploration bonus"
    )
    stability_confidence_threshold: float = Field(
        0.2, ge=0.0, le=1.0,
        description="CI width threshold for considering arm stable"
    )
    stability_improvement_threshold: float = Field(
        0.1, ge=0.0, le=1.0,
        description="Required improvement (absolute) to switch from stable arm"
    )
    stability_min_pulls: int = Field(
        10, ge=0,
        description="Minimum pulls required before applying stability check"
    )
    use_stability: bool = Field(
        True,
        description="Enable/disable arm stability mechanisms"
    )


class RuntimeArm(BaseModel):
    """Represents an arm and its learned statistics."""

    arm_id: str = Field(..., description="Unique identifier for the arm")
    name: Optional[str] = Field(None, description="Human-readable name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameter payload")
    alpha: float = Field(1.0, description="Success prior (Beta alpha)")
    beta: float = Field(1.0, description="Failure prior (Beta beta)")
    total_pulls: int = Field(0, description="Number of pulls")
    total_reward: float = Field(0.0, description="Cumulative reward")
    mean_estimate: Optional[float] = Field(None, description="Posterior mean estimate")
    avg_reward: Optional[float] = Field(None, description="Average observed reward")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class RuntimeArmState(BaseModel):
    """Snapshot of an arm at decision time."""

    arm_id: str
    name: Optional[str] = None
    alpha: float
    beta: float
    sampled_value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class RuntimeArmTemplate(BaseModel):
    """Template used to seed fallback arms."""

    arm_id: str
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class RuntimeConfig(BaseModel):
    """Runtime configuration for a system."""

    system: str
    agent_type: Optional[str] = Field(None, description="Optional agent type tag")
    min_arms: int = Field(1, ge=0)
    cache_ttl_seconds: int = Field(30, ge=0)
    default_arms: List[RuntimeArmTemplate] = Field(default_factory=list)
    selection_strategy: Optional[SelectionStrategyConfig] = Field(
        None,
        description="Selection strategy configuration (exploration/exploitation, stability)"
    )
    reward_evaluator: Optional["RewardEvaluatorConfig"] = Field(
        None,
        description="Reward evaluator configuration for multi-signal reward computation"
    )


class RuntimeDecision(BaseModel):
    """Decision record used for auditing and reward updates."""

    decision_id: Optional[str] = Field(None, description="Identifier of the decision")
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    arm_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    arms_snapshot: List[RuntimeArmState] = Field(default_factory=list)
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class RuntimeSelection(BaseModel):
    """Selection result returned to runtime callers."""

    decision_id: Optional[str]
    arm_id: str
    params: Dict[str, Any]
    sampled_value: float
    arms_state: List[RuntimeArmState]
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "RuntimeArm",
    "RuntimeArmState",
    "RuntimeArmTemplate",
    "RuntimeConfig",
    "RuntimeDecision",
    "RuntimeSelection",
    "SelectionStrategyConfig",
]


# Rebuild RuntimeConfig after RewardEvaluatorConfig is available (Pydantic v2 forward reference fix)
def _rebuild_runtime_config():
    """Rebuild RuntimeConfig model to resolve RewardEvaluatorConfig forward reference."""
    try:
        from convergence.runtime.reward_evaluator import RewardEvaluatorConfig
        RuntimeConfig.model_rebuild()
    except ImportError:
        # RewardEvaluatorConfig not available yet, will be rebuilt on first use
        pass


# Auto-rebuild if RewardEvaluatorConfig is already imported
_rebuild_runtime_config()



