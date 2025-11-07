"""
Runtime arm evolution using Convergence Evolution Engine.

Evolves top-performing arms into new configurations through mutation and crossover,
enabling automatic discovery of better-performing arm configurations.
"""
from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional

from convergence.types.runtime import RuntimeArm, RuntimeArmTemplate
from convergence.optimization.evolution import EvolutionEngine
from convergence.optimization.models import SearchSpaceConfig, SearchSpaceParameter

logger = logging.getLogger(__name__)


def infer_search_space_from_arms(arms: List[RuntimeArm]) -> Optional[SearchSpaceConfig]:
    """
    Infer search space from arm parameters (helper for convenience).
    
    Automatically detects parameter types and ranges from existing arm params.
    Less precise than explicit definition but useful for quick setup.
    
    Args:
        arms: List of arms to infer search space from
        
    Returns:
        SearchSpaceConfig if inference successful, None otherwise
    """
    if not arms or not arms[0].params:
        return None
    
    # Sample first arm's params to infer structure
    sample_params = arms[0].params
    parameters: Dict[str, SearchSpaceParameter] = {}
    
    # Collect all unique values for each param across all arms
    param_values: Dict[str, List[Any]] = {}
    for arm in arms:
        for param_name, param_value in arm.params.items():
            if param_name not in param_values:
                param_values[param_name] = []
            if param_value not in param_values[param_name]:
                param_values[param_name].append(param_value)
    
    # Infer type and range for each parameter
    for param_name, param_value in sample_params.items():
        unique_values = param_values.get(param_name, [param_value])
        
        # Determine parameter type
        if isinstance(param_value, bool):
            # Categorical: True/False
            parameters[param_name] = SearchSpaceParameter(
                type="categorical",
                values=[True, False]
            )
        elif isinstance(param_value, str):
            # Categorical: Use all unique string values found
            if len(unique_values) <= 10:  # Reasonable for categorical
                parameters[param_name] = SearchSpaceParameter(
                    type="categorical",
                    values=unique_values
                )
            else:
                # Too many unique values, skip (can't reasonably mutate strings)
                logger.warning(
                    f"[Evolution] Skipping parameter {param_name}: "
                    f"too many unique string values ({len(unique_values)})"
                )
                continue
        elif isinstance(param_value, (int, float)):
            # Numeric: Infer as discrete or continuous
            numeric_values = [v for v in unique_values if isinstance(v, (int, float))]
            if not numeric_values:
                continue
            
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            
            # If values look like discrete steps, use discrete
            if len(numeric_values) <= 10 and all(isinstance(v, int) for v in numeric_values):
                # Discrete: infer step size
                if len(numeric_values) > 1:
                    # Estimate step from common differences
                    sorted_vals = sorted(numeric_values)
                    diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
                    step = min(diffs) if diffs else 1
                else:
                    step = 1
                
                # Expand range slightly for mutation
                expanded_min = max(0, min_val - 2 * step)
                expanded_max = max_val + 2 * step
                
                parameters[param_name] = SearchSpaceParameter(
                    type="discrete",
                    min=expanded_min,
                    max=expanded_max,
                    step=step
                )
            else:
                # Continuous: use min/max with reasonable step
                expanded_min = min_val * 0.5  # Allow 50% reduction
                expanded_max = max_val * 2.0  # Allow 100% increase
                step = (expanded_max - expanded_min) / 100  # 100 steps
                
                parameters[param_name] = SearchSpaceParameter(
                    type="continuous",
                    min=expanded_min,
                    max=expanded_max,
                    step=step
                )
        else:
            # Unknown type, skip
            logger.warning(
                f"[Evolution] Skipping parameter {param_name}: "
                f"unknown type {type(param_value)}"
            )
            continue
    
    if not parameters:
        logger.warning("[Evolution] Could not infer any parameters from arms")
        return None
    
    return SearchSpaceConfig(parameters=parameters)


async def evolve_arms(
    system: str,
    *,
    user_id: str,
    agent_type: str,
    top_n: int = 3,
    search_space: Optional[SearchSpaceConfig] = None,
    evolution_config: Optional[Dict[str, Any]] = None,
    storage: Optional[Any] = None,  # RuntimeStorageProtocol, but avoiding circular import
) -> List[RuntimeArmTemplate]:
    """
    Evolve top-performing arms into new configurations.
    
    Uses Convergence EvolutionEngine to mutate/crossover best arms, creating
    new arm templates that can be initialized in the MAB system.
    
    Args:
        system: System name (e.g., "chat_model_selection")
        user_id: User ID
        agent_type: Agent type
        top_n: Number of top arms to evolve from
        search_space: Explicit search space definition (recommended)
        evolution_config: Evolution parameters (mutation_rate, crossover_rate, elite_size)
        storage: Storage adapter to load arms (optional, will use manager's storage if not provided)
    
    Returns:
        List of evolved arm templates ready for initialization
    """
    from convergence.runtime.online import _get_manager
    
    # Get manager to access arms
    manager = await _get_manager(system)
    arms_storage = storage or manager.storage
    
    # Load arms for user/agent
    arms = await arms_storage.get_arms(user_id=user_id, agent_type=agent_type)
    
    if not arms:
        logger.warning(f"[Evolution] No arms found for user {user_id}, agent {agent_type}")
        return []
    
    # Convert to RuntimeArm objects if needed
    runtime_arms = []
    for arm_data in arms:
        if isinstance(arm_data, RuntimeArm):
            runtime_arms.append(arm_data)
        else:
            runtime_arms.append(RuntimeArm(**arm_data))
    
    # Sort by performance (mean_estimate)
    sorted_arms = sorted(
        runtime_arms,
        key=lambda a: a.mean_estimate or 0.0,
        reverse=True
    )
    top_arms = sorted_arms[:top_n]
    
    if not top_arms:
        logger.warning("[Evolution] No arms to evolve")
        return []
    
    # Infer search space if not provided
    if not search_space:
        logger.info("[Evolution] Inferring search space from arm parameters")
        search_space = infer_search_space_from_arms(top_arms)
        if not search_space:
            logger.error("[Evolution] Failed to infer search space, cannot evolve")
            return []
    else:
        logger.debug("[Evolution] Using explicit search space definition")
    
    # Create evolution engine
    mutation_rate = evolution_config.get("mutation_rate", 0.2) if evolution_config else 0.2
    crossover_rate = evolution_config.get("crossover_rate", 0.7) if evolution_config else 0.7
    elite_size = evolution_config.get("elite_size", 1) if evolution_config else 1
    
    evolution_engine = EvolutionEngine(
        search_space=search_space,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_size=elite_size,
    )
    
    # Prepare configs and fitness scores
    top_configs = [arm.params for arm in top_arms]
    fitness_scores = [arm.mean_estimate or 0.0 for arm in top_arms]
    
    # Evolve
    evolved_configs = evolution_engine.evolve_population(top_configs, fitness_scores)
    
    # Create new arm templates (skip elite - first config)
    new_arms = []
    for i, evolved_config in enumerate(evolved_configs[elite_size:], 1):
        new_arm = RuntimeArmTemplate(
            arm_id=f"evolved_{agent_type}_{int(time.time())}_{i}",
            name=f"Evolved Arm {i}",
            params=evolved_config,
            description=f"Evolved from top performer (mean_estimate: {fitness_scores[0]:.3f})"
        )
        new_arms.append(new_arm)
    
    logger.info(
        f"[Evolution] Evolved {len(new_arms)} new arms from top {top_n} performers "
        f"(system={system}, user={user_id}, agent={agent_type})"
    )
    
    return new_arms

