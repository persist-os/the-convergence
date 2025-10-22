"""
Convex Storage Backend for The Convergence

Generic bridge that routes Convergence storage calls to any Convex implementation.
Supports custom Convex clients or auto-imports from your backend environment.

Usage:
    # Option 1: Auto-import from your backend
    storage = ConvexStorage()
    
    # Option 2: Pass your own Convex client/toolkit
    from my_app.convex import my_convex_client
    storage = ConvexStorage(client=my_convex_client)
    
    # Option 3: Pass async functions directly
    storage = ConvexStorage(
        save_rl_data_fn=my_save_rl_data,
        save_experiment_fn=my_save_experiment,
        # ... other functions
    )
"""

from typing import Any, List, Optional, Callable, Awaitable
import asyncio


class ConvexStorage:
    """
    Convex storage backend - routes Convergence data to Convex.
    
    Flexible initialization:
    1. Auto-import: ConvexStorage() - imports from app.convex_toolkit.api
    2. Client injection: ConvexStorage(client=my_client)
    3. Function injection: ConvexStorage(save_rl_data_fn=my_fn, ...)
    """
    
    def __init__(
        self,
        client: Optional[Any] = None,
        save_rl_data_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        save_experiment_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        start_run_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        complete_run_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        get_rl_data_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        get_run_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    ):
        """
        Initialize Convex storage.
        
        Args:
            client: Optional Convex client/toolkit object with methods like
                   save_rl_data(), save_experiment(), etc.
            save_rl_data_fn: Optional custom function for saving RL data
            save_experiment_fn: Optional custom function for saving experiments
            start_run_fn: Optional custom function for starting runs
            complete_run_fn: Optional custom function for completing runs
            get_rl_data_fn: Optional custom function for loading RL data
            get_run_fn: Optional custom function for loading runs
        
        If no arguments provided, attempts to auto-import from:
        - app.convex_toolkit.api.convergence_storage_convex (if available)
        
        Examples:
            # Auto-import (for apps with standard structure)
            storage = ConvexStorage()
            
            # Inject client
            from my_app.convex import convex_client
            storage = ConvexStorage(client=convex_client)
            
            # Inject custom functions
            storage = ConvexStorage(
                save_rl_data_fn=my_custom_save,
                save_experiment_fn=my_custom_experiment_save
            )
        """
        # Option 1: Use provided client
        if client is not None:
            self.storage = client
            self._custom_fns = None
        # Option 2: Use provided functions
        elif any([save_rl_data_fn, save_experiment_fn, start_run_fn, complete_run_fn]):
            self.storage = None
            self._custom_fns = {
                'save_rl_data': save_rl_data_fn,
                'save_experiment': save_experiment_fn,
                'start_run': start_run_fn,
                'complete_run': complete_run_fn,
                'get_rl_data': get_rl_data_fn,
                'get_run': get_run_fn,
            }
        # Option 3: Auto-import from standard location
        else:
            try:
                from app.convex_toolkit.api import convergence_storage_convex
                self.storage = convergence_storage_convex
                self._custom_fns = None
            except ImportError as e:
                raise RuntimeError(
                    "ConvexStorage requires either:\n"
                    "1. A client parameter: ConvexStorage(client=my_client)\n"
                    "2. Custom functions: ConvexStorage(save_rl_data_fn=my_fn, ...)\n"
                    "3. Standard backend structure with app.convex_toolkit.api available\n\n"
                    "See documentation for examples."
                ) from e
    
    async def save(self, key: str, value: Any) -> None:
        """
        Save data to Convex, routing by key prefix.
        
        Key Patterns:
        - episode:* → RL training data
        - experiment:* → Optimization experiment
        - run:* → Optimization run
        - audit:* → Audit trail
        """
        try:
            # Route based on key prefix
            if key.startswith("episode:") or key.startswith("trajectory:") or key.startswith("agent_legacy:") or key.startswith("training_run:"):
                # RL training data
                await self._save_rl_data(key, value)
            elif key.startswith("experiment:"):
                # Optimization experiment
                await self._save_experiment(key, value)
            elif key.startswith("run:"):
                # Optimization run
                await self._save_run(key, value)
            else:
                # Unknown key pattern - log warning but don't fail
                print(f"⚠️ Unknown key pattern for Convex storage: {key}")
        except Exception as e:
            # Log error but don't crash optimization
            print(f"❌ Convex save failed for {key}: {e}")
            raise
    
    async def _save_rl_data(self, key: str, value: Any) -> None:
        """Save RL training data."""
        # Determine record type from key prefix
        if key.startswith("episode:"):
            record_type = "episode"
        elif key.startswith("trajectory:"):
            record_type = "trajectory"
        elif key.startswith("agent_legacy:"):
            record_type = "agent_legacy"
        elif key.startswith("training_run:"):
            record_type = "training_run"
        else:
            record_type = "episode"  # Default
        
        # Use custom function if provided, otherwise use client
        if self._custom_fns and self._custom_fns.get('save_rl_data'):
            response = await self._custom_fns['save_rl_data'](
                rl_key=key,
                rl_record_type=record_type,
                agent_id=value.get("agent_id", "unknown"),
                episode_timestamp=value.get("timestamp", 0),
                rl_episode_data=value,
                civilization_id=value.get("civilization_id"),
                station=value.get("station"),
                reward_score=value.get("reward"),
                fitness_score=value.get("fitness_score"),
                success=value.get("success"),
            )
        else:
            # Extract metadata from value
            response = await self.storage.save_rl_data(
            rl_key=key,
            rl_record_type=record_type,
            agent_id=value.get("agent_id", "unknown"),
            episode_timestamp=value.get("timestamp", 0),
            rl_episode_data=value,
            civilization_id=value.get("civilization_id"),
            station=value.get("station"),
            reward_score=value.get("reward"),
            fitness_score=value.get("fitness_score"),
            success=value.get("success"),
        )
        
        if not response.get("success"):
            raise RuntimeError(f"Failed to save RL data: {response.get('error')}")
    
    async def _save_experiment(self, key: str, value: Any) -> None:
        """Save optimization experiment."""
        # Use custom function if provided, otherwise use client
        if self._custom_fns and self._custom_fns.get('save_experiment'):
            response = await self._custom_fns['save_experiment'](
                experiment_id=value.get("experiment_id", key),
                optimization_run_id=value.get("optimization_run_id", "unknown"),
                system_name=value.get("system_name", "unknown"),
                algorithm_name=value.get("algorithm_name", "unknown"),
                test_case_id=value.get("test_case_id", "unknown"),
                tested_config=value.get("config", {}),
                experiment_score=value.get("score", 0.0),
                test_passed=value.get("passed", False),
                experiment_timestamp=value.get("timestamp", 0),
                generation_number=value.get("generation"),
                latency_ms=value.get("latency_ms"),
                cost_usd=value.get("cost_usd"),
                full_metrics=value.get("metrics"),
                session_id=value.get("session_id"),
            )
        else:
            response = await self.storage.save_experiment(
            experiment_id=value.get("experiment_id", key),
            optimization_run_id=value.get("optimization_run_id", "unknown"),
            system_name=value.get("system_name", "unknown"),
            algorithm_name=value.get("algorithm_name", "unknown"),
            test_case_id=value.get("test_case_id", "unknown"),
            tested_config=value.get("config", {}),
            experiment_score=value.get("score", 0.0),
            test_passed=value.get("passed", False),
            experiment_timestamp=value.get("timestamp", 0),
            generation_number=value.get("generation"),
            latency_ms=value.get("latency_ms"),
            cost_usd=value.get("cost_usd"),
            full_metrics=value.get("metrics"),
            session_id=value.get("session_id"),
        )
        
        if not response.get("success"):
            raise RuntimeError(f"Failed to save experiment: {response.get('error')}")
    
    async def _save_run(self, key: str, value: Any) -> None:
        """Save or update optimization run."""
        # Check if this is start or complete
        if value.get("status") == "started":
            if self._custom_fns and self._custom_fns.get('start_run'):
                response = await self._custom_fns['start_run'](
                    run_id=value.get("run_id", key),
                    system_name=value.get("system_name", "unknown"),
                    algorithm_name=value.get("algorithm_name", "unknown"),
                )
            else:
                response = await self.storage.start_optimization_run(
                    run_id=value.get("run_id", key),
                    system_name=value.get("system_name", "unknown"),
                    algorithm_name=value.get("algorithm_name", "unknown"),
                )
        elif value.get("status") == "completed":
            if self._custom_fns and self._custom_fns.get('complete_run'):
                response = await self._custom_fns['complete_run'](
                    run_id=value.get("run_id", key),
                    winning_config_id=value.get("winning_config_id"),
                    winning_config_snapshot=value.get("winning_config"),
                    total_generations=value.get("total_generations"),
                    convergence_achieved=value.get("convergence_achieved"),
                )
            else:
                response = await self.storage.complete_optimization_run(
                    run_id=value.get("run_id", key),
                    winning_config_id=value.get("winning_config_id"),
                    winning_config_snapshot=value.get("winning_config"),
                    total_generations=value.get("total_generations"),
                    convergence_achieved=value.get("convergence_achieved"),
                )
        else:
            # Default to start
            if self._custom_fns and self._custom_fns.get('start_run'):
                response = await self._custom_fns['start_run'](
                    run_id=value.get("run_id", key),
                    system_name=value.get("system_name", "unknown"),
                    algorithm_name=value.get("algorithm_name", "unknown"),
                )
            else:
                response = await self.storage.start_optimization_run(
                    run_id=value.get("run_id", key),
                    system_name=value.get("system_name", "unknown"),
                    algorithm_name=value.get("algorithm_name", "unknown"),
                )
        
        if not response.get("success"):
            raise RuntimeError(f"Failed to save run: {response.get('error')}")
    
    async def load(self, key: str) -> Any:
        """Load data from Convex by key."""
        try:
            # Route based on key prefix
            if key.startswith("episode:") or key.startswith("trajectory:") or key.startswith("agent_legacy:") or key.startswith("training_run:"):
                if self._custom_fns and self._custom_fns.get('get_rl_data'):
                    response = await self._custom_fns['get_rl_data'](key)
                else:
                    response = await self.storage.get_rl_data_by_key(key)
            elif key.startswith("run:"):
                run_id = key.split(":", 1)[1]
                if self._custom_fns and self._custom_fns.get('get_run'):
                    response = await self._custom_fns['get_run'](run_id)
                else:
                    response = await self.storage.get_optimization_run(run_id)
            else:
                raise KeyError(f"Unknown key pattern: {key}")
            
            if response.get("success"):
                data = response.get("data")
                if data is None:
                    raise KeyError(f"Key not found: {key}")
                return data.get("rl_episode_data") if "rl_episode_data" in data else data
            else:
                raise KeyError(f"Key not found: {key}")
        except Exception as e:
            raise KeyError(f"Failed to load {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Convex."""
        try:
            await self.load(key)
            return True
        except KeyError:
            return False
    
    async def delete(self, key: str) -> None:
        """Delete from Convex (not implemented - Convex handles retention)."""
        # Convex manages its own data retention
        # We don't need to implement delete for optimization data
        pass
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys by prefix.
        
        Note: Limited implementation - Convex doesn't have a generic list_keys API.
        Use specific queries instead (e.g., query_rl_episodes_for_training).
        """
        # This would require custom queries per prefix type
        # For now, return empty list - users should use specific query functions
        print(f"⚠️ list_keys not fully implemented for Convex storage. Use specific queries instead.")
        return []
    
    async def count_keys(self, prefix: str = "") -> int:
        """Count keys by prefix."""
        keys = await self.list_keys(prefix)
        return len(keys)
    
    async def clear(self, prefix: str = "") -> int:
        """Clear keys by prefix (not implemented - use Convex dashboard)."""
        print(f"⚠️ clear not implemented for Convex storage. Use Convex dashboard for data management.")
        return 0
    
    async def close(self) -> None:
        """Close storage (no-op for Convex)."""
        pass
