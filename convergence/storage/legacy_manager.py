"""
Legacy Manager - The Heart of The Convergence Memory System.

This is the critical infrastructure that ensures:
1. Nothing is ever lost
2. Best methods are preserved
3. Knowledge passes between generations
4. RL training has optimal data
5. Continuous improvement across runs

WHY THIS MATTERS:
Without this, each civilization starts from scratch. With this,
every generation builds on the last. This is the difference between
random exploration and true evolution.

Usage:
    legacy = LegacyManager()
    
    # Record an episode
    await legacy.record_episode(episode)
    
    # Get best strategies for an agent
    strategies = await legacy.get_best_strategies(agent_id)
    
    # Query RL training data
    training_data = await legacy.query_episodes_for_training(
        agent_id=agent_id,
        min_reward=0.7,
        limit=100
    )
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

from convergence.storage.multi_backend import MultiBackendStorage, get_legacy_storage
from convergence.storage.rl_models import (
    RLEpisode,
    RLTrajectory,
    AgentLegacy,
    CivilizationLegacy,
    RLTrainingRun,
    generate_episode_key,
    generate_trajectory_key,
    generate_legacy_key,
    generate_civilization_key,
    generate_training_run_key,
)


class LegacyManager:
    """
    Manages preservation and retrieval of agent/civilization knowledge.
    
    Features:
    - Continuous episode recording
    - Automatic best-method tracking
    - RL training data queries
    - Multi-generational knowledge transfer
    - Data integrity validation
    
    This is the "institutional memory" of The Convergence.
    """
    
    def __init__(
        self,
        storage: Optional[MultiBackendStorage] = None,
        auto_update_legacy: bool = True
    ):
        """
        Initialize legacy manager.
        
        Args:
            storage: Storage backend (default: multi-backend with SQLite + File)
            auto_update_legacy: Automatically update legacies on episode record
        """
        self.storage = storage or get_legacy_storage()
        self.auto_update_legacy = auto_update_legacy
        
        # Episode sequence counters (per agent)
        self._episode_counters: Dict[str, int] = {}
        self._trajectory_counters: Dict[str, int] = {}
        
        print("📚 Legacy Manager initialized")
        print("   Storage: Multi-backend (SQLite + File)")
        print("   Auto-update: Enabled" if auto_update_legacy else "   Auto-update: Disabled")
    
    # =================================================================
    # EPISODE RECORDING
    # =================================================================
    
    async def record_episode(
        self,
        episode: RLEpisode,
        update_legacy: Optional[bool] = None
    ) -> str:
        """
        Record an episode to persistent storage.
        
        This is the critical path - every agent interaction flows through here.
        
        Args:
            episode: Episode to record
            update_legacy: Override auto_update_legacy setting
            
        Returns:
            Episode key
        """
        # Generate key
        if not episode.episode_id:
            sequence = self._get_next_episode_sequence(episode.agent_id)
            episode.episode_id = generate_episode_key(
                episode.agent_id,
                episode.station,
                sequence
            )
        
        key = f"episode:{episode.episode_id}"
        
        # Save to storage (written to ALL backends)
        await self.storage.save(key, episode.dict())
        
        # Update agent legacy if enabled
        should_update = update_legacy if update_legacy is not None else self.auto_update_legacy
        if should_update:
            await self._update_agent_legacy(episode)
        
        # Update civilization legacy if part of a civilization
        if episode.civilization_id:
            await self._update_civilization_legacy(episode)
        
        return key
    
    async def record_episodes_batch(
        self,
        episodes: List[RLEpisode]
    ) -> List[str]:
        """
        Record multiple episodes efficiently.
        
        Args:
            episodes: List of episodes to record
            
        Returns:
            List of episode keys
        """
        import asyncio
        tasks = [self.record_episode(ep) for ep in episodes]
        return await asyncio.gather(*tasks)
    
    # =================================================================
    # TRAJECTORY MANAGEMENT
    # =================================================================
    
    async def create_trajectory(
        self,
        agent_id: str,
        episodes: List[RLEpisode],
        civilization_id: Optional[str] = None
    ) -> str:
        """
        Create a trajectory from episodes.
        
        Trajectories are used for multi-step RL algorithms.
        
        Args:
            agent_id: Agent ID
            episodes: Sequence of episodes
            civilization_id: Optional civilization ID
            
        Returns:
            Trajectory key
        """
        sequence = self._get_next_trajectory_sequence(agent_id)
        trajectory_id = generate_trajectory_key(agent_id, sequence)
        
        trajectory = RLTrajectory(
            trajectory_id=trajectory_id,
            agent_id=agent_id,
            civilization_id=civilization_id,
            episodes=episodes,
            total_reward=sum(ep.reward for ep in episodes),
            average_reward=sum(ep.reward for ep in episodes) / len(episodes) if episodes else 0.0
        )
        
        key = f"trajectory:{trajectory_id}"
        await self.storage.save(key, trajectory.dict())
        
        return key
    
    # =================================================================
    # LEGACY RETRIEVAL
    # =================================================================
    
    async def get_agent_legacy(self, agent_id: str) -> Optional[AgentLegacy]:
        """
        Get an agent's legacy.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            AgentLegacy or None if doesn't exist
        """
        key = generate_legacy_key(agent_id)
        
        try:
            data = await self.storage.load(key)
            return AgentLegacy(**data)
        except KeyError:
            return None
    
    async def get_best_strategies(
        self,
        agent_id: str,
        station: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get best strategies for an agent.
        
        Args:
            agent_id: Agent ID
            station: Optional station filter
            
        Returns:
            Dict mapping station to best strategy info
        """
        legacy = await self.get_agent_legacy(agent_id)
        
        if not legacy:
            return {}
        
        if station:
            return {station: legacy.best_strategies.get(station, {})}
        
        return legacy.best_strategies
    
    async def get_civilization_legacy(
        self,
        civilization_id: str
    ) -> Optional[CivilizationLegacy]:
        """
        Get a civilization's legacy.
        
        Args:
            civilization_id: Civilization ID
            
        Returns:
            CivilizationLegacy or None if doesn't exist
        """
        key = generate_civilization_key(civilization_id)
        
        try:
            data = await self.storage.load(key)
            return CivilizationLegacy(**data)
        except KeyError:
            return None
    
    # =================================================================
    # RL TRAINING DATA QUERIES
    # =================================================================
    
    async def query_episodes_for_training(
        self,
        agent_id: Optional[str] = None,
        civilization_id: Optional[str] = None,
        station: Optional[str] = None,
        min_reward: float = 0.0,
        min_timestamp: Optional[datetime] = None,
        limit: int = 1000,
        strategy: Optional[str] = None
    ) -> List[RLEpisode]:
        """
        Query episodes optimized for RL training.
        
        This is the key method for preparing training data.
        
        Args:
            agent_id: Filter by agent
            civilization_id: Filter by civilization
            station: Filter by station
            min_reward: Minimum reward threshold
            min_timestamp: Only episodes after this time
            limit: Maximum episodes to return
            strategy: Filter by strategy
            
        Returns:
            List of episodes matching criteria
        """
        # Build prefix for efficient filtering
        if agent_id:
            prefix = f"episode:{agent_id}"
        else:
            prefix = "episode:"
        
        # Get all matching keys
        keys = await self.storage.list_keys(prefix)
        
        # Load and filter episodes
        episodes = []
        
        for key in keys[:limit * 2]:  # Load more than needed for filtering
            try:
                data = await self.storage.load(key)
                episode = RLEpisode(**data)
                
                # Apply filters
                if civilization_id and episode.civilization_id != civilization_id:
                    continue
                
                if station and episode.station != station:
                    continue
                
                if episode.reward < min_reward:
                    continue
                
                if min_timestamp and episode.timestamp < min_timestamp:
                    continue
                
                if strategy and episode.action.strategy != strategy:
                    continue
                
                episodes.append(episode)
                
                if len(episodes) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Failed to load episode {key}: {e}")
                continue
        
        return episodes[:limit]
    
    async def get_training_dataset(
        self,
        agent_id: str,
        min_episodes: int = 50,
        quality_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get a complete training dataset for RL.
        
        This prepares data in the format needed for training_gym.py.
        
        Args:
            agent_id: Agent to get data for
            min_episodes: Minimum episodes required
            quality_threshold: Minimum average reward
            
        Returns:
            Training dataset with episodes and metadata
        """
        episodes = await self.query_episodes_for_training(
            agent_id=agent_id,
            limit=1000
        )
        
        if len(episodes) < min_episodes:
            raise ValueError(
                f"Insufficient episodes: need {min_episodes}, have {len(episodes)}"
            )
        
        # Calculate dataset quality
        avg_reward = sum(ep.reward for ep in episodes) / len(episodes)
        
        if avg_reward < quality_threshold:
            print(f"⚠️ Low quality dataset: avg reward {avg_reward:.3f}")
        
        # Group by station for GRPO
        station_groups = {}
        for ep in episodes:
            if ep.station not in station_groups:
                station_groups[ep.station] = []
            station_groups[ep.station].append(ep)
        
        return {
            "agent_id": agent_id,
            "num_episodes": len(episodes),
            "stations_covered": list(station_groups.keys()),
            "avg_reward": avg_reward,
            "episodes": [ep.dict() for ep in episodes],
            "station_groups": {
                station: [ep.dict() for ep in eps]
                for station, eps in station_groups.items()
            },
            "quality": "good" if avg_reward >= quality_threshold else "fair",
            "ready_for_training": len(episodes) >= min_episodes
        }
    
    # =================================================================
    # RL TRAINING RUNS
    # =================================================================
    
    async def record_training_run(
        self,
        training_run: RLTrainingRun
    ) -> str:
        """
        Record an RL training run.
        
        Args:
            training_run: Training run to record
            
        Returns:
            Training run key
        """
        key = generate_training_run_key(training_run.run_id)
        await self.storage.save(key, training_run.dict())
        
        # Update agent legacy with new policy
        await self._apply_trained_policy(training_run)
        
        # Update civilization legacy
        if training_run.civilization_id:
            await self._record_civilization_training(training_run)
        
        return key
    
    # =================================================================
    # INTERNAL METHODS
    # =================================================================
    
    async def _update_agent_legacy(self, episode: RLEpisode) -> None:
        """Update agent's legacy from an episode."""
        legacy_key = generate_legacy_key(episode.agent_id)
        
        try:
            data = await self.storage.load(legacy_key)
            legacy = AgentLegacy(**data)
        except KeyError:
            # Create new legacy
            legacy = AgentLegacy(
                agent_id=episode.agent_id,
                civilization_id=episode.civilization_id,
                generation=0
            )
        
        # Update with episode
        legacy.update_from_episode(episode)
        
        # Save back
        await self.storage.save(legacy_key, legacy.dict())
    
    async def _update_civilization_legacy(self, episode: RLEpisode) -> None:
        """Update civilization's legacy from an episode."""
        if not episode.civilization_id:
            return
        
        civ_key = generate_civilization_key(episode.civilization_id)
        
        try:
            data = await self.storage.load(civ_key)
            civ_legacy = CivilizationLegacy(**data)
        except KeyError:
            # Create new civilization legacy
            civ_legacy = CivilizationLegacy(
                civilization_id=episode.civilization_id,
                name=episode.civilization_id,
                generation=0
            )
        
        # Update metrics
        civ_legacy.last_updated = datetime.utcnow()
        
        # Save back
        await self.storage.save(civ_key, civ_legacy.dict())
    
    async def _apply_trained_policy(self, training_run: RLTrainingRun) -> None:
        """Apply trained policy to agent's legacy."""
        legacy_key = generate_legacy_key(training_run.agent_id)
        
        try:
            data = await self.storage.load(legacy_key)
            legacy = AgentLegacy(**data)
        except KeyError:
            print(f"⚠️ No legacy found for {training_run.agent_id}")
            return
        
        # Update policy
        legacy.rl_policy = training_run.policy
        legacy.policy_version = training_run.policy_version
        legacy.policy_training_episodes = training_run.num_episodes
        legacy.last_updated = datetime.utcnow()
        
        # Save back
        await self.storage.save(legacy_key, legacy.dict())
    
    async def _record_civilization_training(self, training_run: RLTrainingRun) -> None:
        """Record training run in civilization legacy."""
        if not training_run.civilization_id:
            return
        
        civ_key = generate_civilization_key(training_run.civilization_id)
        
        try:
            data = await self.storage.load(civ_key)
            civ_legacy = CivilizationLegacy(**data)
        except KeyError:
            return
        
        # Add training run
        civ_legacy.rl_training_runs.append({
            "run_id": training_run.run_id,
            "agent_id": training_run.agent_id,
            "improvement": training_run.improvement_percentage,
            "timestamp": training_run.started_at.isoformat()
        })
        
        civ_legacy.last_updated = datetime.utcnow()
        
        # Save back
        await self.storage.save(civ_key, civ_legacy.dict())
    
    def _get_next_episode_sequence(self, agent_id: str) -> int:
        """Get next episode sequence number for agent."""
        if agent_id not in self._episode_counters:
            self._episode_counters[agent_id] = 0
        
        self._episode_counters[agent_id] += 1
        return self._episode_counters[agent_id]
    
    def _get_next_trajectory_sequence(self, agent_id: str) -> int:
        """Get next trajectory sequence number for agent."""
        if agent_id not in self._trajectory_counters:
            self._trajectory_counters[agent_id] = 0
        
        self._trajectory_counters[agent_id] += 1
        return self._trajectory_counters[agent_id]
    
    # =================================================================
    # ANALYTICS & INSIGHTS
    # =================================================================
    
    async def get_performance_trends(
        self,
        agent_id: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            agent_id: Agent to analyze
            window_size: Moving average window
            
        Returns:
            Performance trends analysis
        """
        episodes = await self.query_episodes_for_training(
            agent_id=agent_id,
            limit=1000
        )
        
        if not episodes:
            return {"error": "No episodes found"}
        
        # Sort by timestamp
        episodes.sort(key=lambda ep: ep.timestamp)
        
        # Calculate moving average
        rewards = [ep.reward for ep in episodes]
        moving_avg = []
        
        for i in range(len(rewards)):
            start = max(0, i - window_size + 1)
            window = rewards[start:i+1]
            moving_avg.append(sum(window) / len(window))
        
        # Detect trend
        if len(moving_avg) >= 2:
            early_avg = sum(moving_avg[:len(moving_avg)//3]) / (len(moving_avg)//3)
            late_avg = sum(moving_avg[-len(moving_avg)//3:]) / (len(moving_avg)//3)
            trend = "improving" if late_avg > early_avg else "declining"
            improvement = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
        else:
            trend = "insufficient_data"
            improvement = 0
        
        return {
            "agent_id": agent_id,
            "total_episodes": len(episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "trend": trend,
            "improvement_percentage": improvement,
            "moving_average": moving_avg,
            "raw_rewards": rewards
        }
    
    async def close(self) -> None:
        """Close storage backends."""
        await self.storage.close()


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

async def create_agent_from_legacy(
    parent_agent_id: str,
    new_agent_id: str,
    legacy_manager: LegacyManager
) -> AgentLegacy:
    """
    Create a new agent inheriting from parent's legacy.
    
    This is how knowledge passes between generations.
    
    Args:
        parent_agent_id: Parent agent ID
        new_agent_id: New agent ID
        legacy_manager: Legacy manager instance
        
    Returns:
        New agent's legacy (initialized with parent's knowledge)
    """
    parent_legacy = await legacy_manager.get_agent_legacy(parent_agent_id)
    
    if not parent_legacy:
        raise ValueError(f"Parent legacy not found: {parent_agent_id}")
    
    # Create new legacy inheriting parent's knowledge
    new_legacy = AgentLegacy(
        agent_id=new_agent_id,
        civilization_id=parent_legacy.civilization_id,
        generation=parent_legacy.generation + 1,
        best_strategies=parent_legacy.best_strategies.copy(),
        meta_insights=parent_legacy.meta_insights.copy(),
        rl_policy=parent_legacy.rl_policy,
        policy_version=parent_legacy.policy_version,
        parent_agent_ids=[parent_agent_id]
    )
    
    # Save new legacy
    key = generate_legacy_key(new_agent_id)
    await legacy_manager.storage.save(key, new_legacy.dict())
    
    # Update parent's children
    parent_legacy.child_agent_ids.append(new_agent_id)
    parent_key = generate_legacy_key(parent_agent_id)
    await legacy_manager.storage.save(parent_key, parent_legacy.dict())
    
    print(f"🧬 Created agent {new_agent_id} from parent {parent_agent_id}")
    print(f"   Generation: {new_legacy.generation}")
    print(f"   Inherited strategies: {len(new_legacy.best_strategies)}")
    
    return new_legacy

