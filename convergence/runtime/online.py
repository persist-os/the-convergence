"""Backend-agnostic runtime loop for multi-armed bandit selection."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional

try:  # Optional dependency for reproducible sampling
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

from convergence.storage.runtime_protocol import RuntimeStorageProtocol
from convergence.storage.runtime_stub import UnconfiguredRuntimeStorage
from convergence.types import (
    RuntimeArm,
    RuntimeArmState,
    RuntimeArmTemplate,
    RuntimeConfig,
    RuntimeDecision,
    RuntimeSelection,
    SelectionStrategyConfig,
)
from convergence.runtime.bayesian_update import compute_bayesian_update

logger = logging.getLogger(__name__)

_runtime_managers: Dict[str, "RuntimeManager"] = {}
_manager_lock = asyncio.Lock()


class RuntimeManager:
    """Maintains runtime selection state for a single system."""

    def __init__(self, *, config: RuntimeConfig, storage: RuntimeStorageProtocol):
        self.config = config
        self.storage = storage
        self._arms_cache: Dict[tuple[str, str], tuple[float, List[RuntimeArm]]] = {}
        self._lock = asyncio.Lock()

    async def select(
        self,
        *,
        user_id: str,
        agent_type: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> RuntimeSelection:
        agent = agent_type or self.config.agent_type or "default"
        async with self._lock:
            arms = await self._load_arms(user_id=user_id, agent_type=agent)
            if not arms:
                logger.debug(
                    "[Runtime] No arms found for system=%s user=%s; falling back to defaults",
                    self.config.system,
                    user_id,
                )
                return self._fallback_selection()

            samples = self._sample(arms)
            
            # Apply stability check if configured
            strategy = self.config.selection_strategy
            if strategy and strategy.use_stability:
                # Find current best arm (highest mean_estimate with sufficient pulls)
                current_best = None
                for arm in arms:
                    if arm.total_pulls >= strategy.stability_min_pulls:
                        if current_best is None or (arm.mean_estimate or 0.0) > (current_best.mean_estimate or 0.0):
                            current_best = arm
                
                # If we have a stable current best, check if we should stick with it
                if current_best:
                    # Get CI width from metadata or compute it
                    ci_width = self._get_confidence_interval_width(current_best)
                    
                    if ci_width < strategy.stability_confidence_threshold:
                        # High confidence in current arm, check if candidate is significantly better
                        candidate_id = max(samples.items(), key=lambda item: item[1])[0]
                        candidate_arm = next(arm for arm in arms if arm.arm_id == candidate_id)
                        
                        improvement = (candidate_arm.mean_estimate or 0.0) - (current_best.mean_estimate or 0.0)
                        
                        if improvement < strategy.stability_improvement_threshold:
                            # Stick with current best (stability)
                            logger.debug(
                                f"[Runtime] Staying with stable arm {current_best.arm_id} "
                                f"(improvement {improvement:.3f} < threshold {strategy.stability_improvement_threshold})"
                            )
                            selected_id = current_best.arm_id
                            selected_arm = current_best
                        else:
                            selected_id = candidate_id
                            selected_arm = candidate_arm
                    else:
                        # Not stable enough, allow switching
                        selected_id = max(samples.items(), key=lambda item: item[1])[0]
                        selected_arm = next(arm for arm in arms if arm.arm_id == selected_id)
                else:
                    # No stable arm yet, use highest sample
                    selected_id = max(samples.items(), key=lambda item: item[1])[0]
                    selected_arm = next(arm for arm in arms if arm.arm_id == selected_id)
            else:
                # No stability check, use highest sample
                selected_id = max(samples.items(), key=lambda item: item[1])[0]
                selected_arm = next(arm for arm in arms if arm.arm_id == selected_id)

            arms_state = [
                RuntimeArmState(
                    arm_id=arm.arm_id,
                    name=arm.name,
                    alpha=arm.alpha,
                    beta=arm.beta,
                    sampled_value=samples[arm.arm_id],
                    metadata=arm.metadata,
                )
                for arm in arms
            ]

            decision_id = await self._persist_decision(
                user_id=user_id,
                agent_type=agent,
                arm=selected_arm,
                arms_state=arms_state,
                context=context,
            )

            metadata = {
                "system": self.config.system,
                "agent_type": agent,
                "decision_id": decision_id,
                "samples": samples,
            }

            return RuntimeSelection(
                decision_id=decision_id,
                arm_id=selected_arm.arm_id,
                params=selected_arm.params,
                sampled_value=samples[selected_arm.arm_id],
                arms_state=arms_state,
                metadata=metadata,
            )

    async def update(
        self,
        *,
        user_id: str,
        decision_id: str,
        reward: Optional[float] = None,
        signals: Optional[Dict[str, float]] = None,
        agent_type: Optional[str] = None,
        engagement_score: Optional[float] = None,
        grading_score: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        agent = agent_type or self.config.agent_type or "default"
        
        # Compute reward using evaluator if configured, otherwise use raw reward
        computed_reward = None
        
        # If reward_evaluator configured and signals provided, use evaluator
        if self.config.reward_evaluator and signals:
            from convergence.runtime.reward_evaluator import RuntimeRewardEvaluator
            evaluator = RuntimeRewardEvaluator(self.config.reward_evaluator)
            computed_reward = evaluator.evaluate(signals)
            logger.debug(f"[Runtime] Evaluated reward from signals: {computed_reward:.3f}")
        elif reward is not None:
            computed_reward = reward
        elif engagement_score is not None:
            computed_reward = engagement_score
        else:
            computed_reward = 0.5  # Default neutral reward
        
        reward_clamped = _clamp(computed_reward)
        engagement = _clamp(engagement_score) if engagement_score is not None else reward_clamped
        grading = _clamp(grading_score) if grading_score is not None else None

        # Load decision to get arm_id
        decision = await self.get_decision(user_id=user_id, decision_id=decision_id)
        if not decision or not decision.arm_id:
            logger.error(
                f"[Runtime] Decision {decision_id} not found or missing arm_id for user {user_id}"
            )
            return {
                "success": False,
                "error": f"Decision {decision_id} not found or missing arm_id",
                "reward": reward_clamped,
                "engagement_score": engagement,
                "grading_score": grading,
            }

        # Load current arms to get the selected arm's current state
        arms = await self._load_arms(user_id=user_id, agent_type=agent)
        arm = next((a for a in arms if a.arm_id == decision.arm_id), None)
        if not arm:
            logger.error(
                f"[Runtime] Arm {decision.arm_id} not found for user {user_id}, agent {agent}"
            )
            return {
                "success": False,
                "error": f"Arm {decision.arm_id} not found",
                "reward": reward_clamped,
                "engagement_score": engagement,
                "grading_score": grading,
            }

        # Compute Bayesian update using centralized function
        computed_update = compute_bayesian_update(arm, reward_clamped)

        # Pass computed values to storage (maintain backward compatibility with reward parameter)
        response = await self.storage.update_performance(
            user_id=user_id,
            agent_type=agent,
            decision_id=decision_id,
            reward=reward_clamped,  # Keep for backward compatibility
            engagement=engagement,
            grading=grading,
            metadata=metadata,
            computed_update=computed_update,  # NEW: Pre-computed values
        )

        self._invalidate_cache(user_id=user_id, agent_type=agent)
        success = True
        if isinstance(response, dict):
            success = bool(response.get("success", True))
        return {
            "success": success,
            "reward": reward_clamped,
            "engagement_score": engagement,
            "grading_score": grading,
            "computed_update": computed_update,  # NEW: Include computed values in response
        }

    async def get_decision(
        self,
        *,
        user_id: str,
        decision_id: str,
    ) -> RuntimeDecision:
        payload = await self.storage.get_decision(user_id=user_id, decision_id=decision_id)
        return RuntimeDecision(**_coerce_decision_payload(payload))

    async def _load_arms(self, *, user_id: str, agent_type: str) -> List[RuntimeArm]:
        cache_key = (user_id, agent_type)
        cached = self._arms_cache.get(cache_key)
        now = time.monotonic()
        if cached and (now - cached[0]) < self.config.cache_ttl_seconds:
            return cached[1]

        raw_arms = await self.storage.get_arms(user_id=user_id, agent_type=agent_type)
        arms = [_coerce_arm_payload(raw) for raw in raw_arms if raw is not None]

        if not arms and self.config.default_arms:
            await self._initialize_defaults(user_id=user_id, agent_type=agent_type)
            raw_arms = await self.storage.get_arms(user_id=user_id, agent_type=agent_type)
            arms = [_coerce_arm_payload(raw) for raw in raw_arms if raw is not None]

        self._arms_cache[cache_key] = (now, arms)
        return arms

    async def _initialize_defaults(self, *, user_id: str, agent_type: str) -> None:
        if not self.config.default_arms:
            return
        payload = [template.dict() for template in self.config.default_arms]
        try:
            await self.storage.initialize_arms(user_id=user_id, agent_type=agent_type, arms=payload)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "[Runtime] Failed to initialize default arms | system=%s user=%s error=%s",
                self.config.system,
                user_id,
                exc,
                exc_info=True,
            )

    async def _persist_decision(
        self,
        *,
        user_id: str,
        agent_type: str,
        arm: RuntimeArm,
        arms_state: List[RuntimeArmState],
        context: Optional[Dict[str, object]],
    ) -> Optional[str]:
        metadata = context.copy() if context else {}
        try:
            return await self.storage.create_decision(
                user_id=user_id,
                agent_type=agent_type,
                arm_pulled=arm.arm_id,
                strategy_params=arm.params,
                arms_snapshot=[state.dict() for state in arms_state],
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover
            logger.error(
                "[Runtime] Failed to persist decision | system=%s user=%s error=%s",
                self.config.system,
                user_id,
                exc,
                exc_info=True,
            )
            return None

    def _fallback_selection(self) -> RuntimeSelection:
        if self.config.default_arms:
            template = self.config.default_arms[0]
            fallback_state = RuntimeArmState(
                arm_id=template.arm_id,
                name=template.name,
                alpha=1.0,
                beta=1.0,
                sampled_value=0.5,
            )
            return RuntimeSelection(
                decision_id=None,
                arm_id=template.arm_id,
                params=template.params,
                sampled_value=0.5,
                arms_state=[fallback_state],
                metadata={"system": self.config.system, "fallback": True},
            )

        return RuntimeSelection(
            decision_id=None,
            arm_id="default",
            params={},
            sampled_value=0.0,
            arms_state=[],
            metadata={"system": self.config.system, "fallback": True},
        )

    def _sample(self, arms: List[RuntimeArm]) -> Dict[str, float]:
        """Sample from arms with optional exploration bonus."""
        samples: Dict[str, float] = {}
        strategy = self.config.selection_strategy
        
        for arm in arms:
            # Thompson Sampling: sample from Beta(alpha, beta)
            alpha = max(arm.alpha, 1e-6)
            beta = max(arm.beta, 1e-6)
            if _np is not None:
                sample = float(_np.random.beta(alpha, beta))  # type: ignore[attr-defined]
            else:
                sample = random.betavariate(alpha, beta)
            
            # Apply exploration bonus if configured
            if strategy and strategy.exploration_bonus > 0:
                if arm.total_pulls < strategy.exploration_min_pulls:
                    sample += strategy.exploration_bonus
                    # Clamp to [0, 1] after bonus
                    sample = min(1.0, sample)
            
            samples[arm.arm_id] = sample
        return samples

    def _invalidate_cache(self, *, user_id: str, agent_type: str) -> None:
        self._arms_cache.pop((user_id, agent_type), None)
    
    def _get_confidence_interval_width(self, arm: RuntimeArm) -> float:
        """Get confidence interval width for an arm."""
        # Try to get from metadata first
        if arm.metadata and "confidence_interval" in arm.metadata:
            ci = arm.metadata["confidence_interval"]
            if isinstance(ci, dict) and "lower" in ci and "upper" in ci:
                return ci["upper"] - ci["lower"]
        
        # Compute from alpha/beta if mean_estimate exists
        if arm.mean_estimate is not None and arm.alpha > 0 and arm.beta > 0:
            import math
            variance = (arm.alpha * arm.beta) / ((arm.alpha + arm.beta) ** 2 * (arm.alpha + arm.beta + 1))
            std_dev = math.sqrt(variance)
            return 2 * 1.96 * std_dev  # 95% CI width
        
        # Fallback: return large width (no confidence)
        return 1.0


def _coerce_arm_payload(payload: object) -> RuntimeArm:
    if isinstance(payload, RuntimeArm):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Arm payload must be a dict or RuntimeArm")

    data = dict(payload)
    data.setdefault("arm_id", data.get("armId") or data.get("arm_id"))
    if not data.get("arm_id"):
        raise ValueError("Arm payload missing 'arm_id' field")
    data.setdefault("name", data.get("armName"))
    params = data.get("params")
    if params is None and "strategy_params" in data:
        data["params"] = data["strategy_params"]
    data.setdefault("alpha", data.get("alpha", 1.0))
    data.setdefault("beta", data.get("beta", 1.0))
    data.setdefault("total_pulls", data.get("total_pulls", data.get("totalPulls", 0)))
    data.setdefault("total_reward", data.get("total_reward", data.get("totalReward", 0.0)))
    data.setdefault("mean_estimate", data.get("mean_estimate", data.get("meanEstimate")))
    data.setdefault("avg_reward", data.get("avg_reward", data.get("avgReward")))
    data.setdefault("metadata", data.get("metadata", {}))
    return RuntimeArm(**data)


def _coerce_decision_payload(payload: Dict[str, object]) -> Dict[str, object]:
    data = dict(payload)
    if "arm_id" not in data and "armPulled" in data:
        data["arm_id"] = data["armPulled"]
    if "params" not in data and "strategyUsed" in data:
        data["params"] = data["strategyUsed"]
    if "arms_snapshot" not in data:
        snapshot = data.get("arms_state") or data.get("armsSnapshot")
        if snapshot:
            data["arms_snapshot"] = [RuntimeArmState(**_coerce_arm_state(raw)) for raw in snapshot]  # type: ignore[arg-type]
    if "metadata" not in data:
        meta = data.get("metadata") or {}
        data["metadata"] = meta
    if "decision_id" not in data:
        data["decision_id"] = data.get("decisionId") or data.get("_id")
    if "created_at" not in data:
        data["created_at"] = data.get("decisionAt")
    return data


def _coerce_arm_state(payload: object) -> Dict[str, object]:
    if isinstance(payload, RuntimeArmState):
        return payload.dict()
    if not isinstance(payload, dict):
        raise TypeError("Arm state payload must be a dict or RuntimeArmState")
    data = dict(payload)
    data.setdefault("arm_id", data.get("armId") or data.get("arm_id"))
    data.setdefault("name", data.get("armName"))
    data.setdefault("alpha", data.get("alpha", 1.0))
    data.setdefault("beta", data.get("beta", 1.0))
    data.setdefault("sampled_value", data.get("sampled_value", data.get("sampledValue", 0.0)))
    data.setdefault("metadata", data.get("metadata", {}))
    return data


def _clamp(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


async def configure(
    system: str,
    *,
    config: RuntimeConfig,
    storage: RuntimeStorageProtocol,
) -> None:
    """Register runtime configuration and storage for a system."""

    async with _manager_lock:
        _runtime_managers[system] = RuntimeManager(config=config, storage=storage)


async def select(
    system: str,
    *,
    user_id: str,
    agent_type: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
) -> RuntimeSelection:
    manager = await _get_manager(system)
    return await manager.select(user_id=user_id, agent_type=agent_type, context=context)


async def update(
    system: str,
    *,
    user_id: str,
    decision_id: str,
    reward: Optional[float] = None,
    signals: Optional[Dict[str, float]] = None,
    agent_type: Optional[str] = None,
    engagement_score: Optional[float] = None,
    grading_score: Optional[float] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    manager = await _get_manager(system)
    return await manager.update(
        user_id=user_id,
        decision_id=decision_id,
        reward=reward,
        signals=signals,
        agent_type=agent_type,
        engagement_score=engagement_score,
        grading_score=grading_score,
        metadata=metadata,
    )


async def get_decision(system: str, *, user_id: str, decision_id: str) -> RuntimeDecision:
    manager = await _get_manager(system)
    return await manager.get_decision(user_id=user_id, decision_id=decision_id)


async def _get_manager(system: str) -> RuntimeManager:
    manager = _runtime_managers.get(system)
    if manager is not None:
        return manager

    async with _manager_lock:
        manager = _runtime_managers.get(system)
        if manager is None:
            manager = RuntimeManager(
                config=RuntimeConfig(system=system),
                storage=UnconfiguredRuntimeStorage(system=system),
            )
            _runtime_managers[system] = manager
        return manager


__all__ = [
    "configure",
    "select",
    "update",
    "get_decision",
    "RuntimeConfig",
    "_get_manager",  # Export for evolution module
]


