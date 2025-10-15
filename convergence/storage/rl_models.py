"""
RL-Optimized Data Models for The Convergence.

These models are designed specifically for reinforcement learning training.
They match the patterns used in training_gym.py for GRPO and meta-learning.

Why These Models?
- Structured for RL algorithms (state-action-reward-next_state)
- Optimized for time-series queries (trajectories over time)
- Support for meta-learning (cross-station patterns)
- Compatible with W&B Training / ART SDK
- Efficient serialization for storage

Usage:
    episode = RLEpisode(
        episode_id="ep_001",
        agent_id="agent_123",
        station="web_playground",
        strategy="systematic",
        state={"level": 5, "mastery": 0.7},
        action={"strategy_selected": "systematic"},
        reward=0.85,
        next_state={"level": 5, "mastery": 0.75}
    )
    
    await storage.save(f"episode:{episode.episode_id}", episode.dict())
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class StationType(str, Enum):
    """Station types in The Convergence."""
    WEB_PLAYGROUND = "web_playground"
    RESEARCH_LIBRARY = "research_library"
    TOOL_WORKSHOP = "tool_workshop"
    SAFE_SANDBOX = "safe_sandbox"
    CLOUD_CLASSROOM = "cloud_classroom"
    COLLAB_SPACE = "collab_space"
    DEBUG_DOJO = "debug_dojo"
    TRAINING_GYM = "training_gym"


class StrategyType(str, Enum):
    """MAB strategy types."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"
    QUICK_THINKER = "quick_thinker"
    DEEP_PLANNER = "deep_planner"
    ADAPTIVE = "adaptive"
    STRUCTURED_FORMATTER = "structured_formatter"
    ITERATIVE = "iterative"
    BALANCED = "balanced"


class RLState(BaseModel):
    """
    State representation for RL.
    
    Captures agent's current situation at a point in time.
    This is the "S" in (S, A, R, S').
    """
    agent_level: int = Field(ge=1, le=10)
    station_mastery: Dict[str, float] = Field(
        default_factory=dict,
        description="Mastery level for each station [0.0, 1.0]"
    )
    overall_mastery: float = Field(ge=0.0, le=1.0, default=0.0)
    current_station: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional context that might help RL
    total_challenges_completed: int = 0
    recent_performance: List[float] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RLAction(BaseModel):
    """
    Action representation for RL.
    
    Captures what the agent decided to do.
    This is the "A" in (S, A, R, S').
    """
    strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    mab_arm_index: Optional[int] = None
    exploration_bonus: float = 0.0
    
    # Meta-information
    decision_method: str = "mab"  # "mab", "rl_policy", "random", "heuristic"
    action_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RLEpisode(BaseModel):
    """
    Complete RL episode (one interaction).
    
    This is the fundamental unit of RL training data.
    Represents: S -> A -> R -> S'
    
    Storage key pattern: "episode:{episode_id}"
    Example: "episode:agent_123_station_web_playground_001"
    """
    # Identification
    episode_id: str = Field(description="Unique episode ID")
    agent_id: str
    civilization_id: Optional[str] = None
    
    # Episode context
    station: str
    challenge_level: int = Field(ge=1, le=10)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # RL tuple: (S, A, R, S')
    state: RLState
    action: RLAction
    reward: float = Field(ge=0.0, le=1.0, description="Normalized reward [0, 1]")
    next_state: RLState
    
    # Episode outcome
    success: bool
    fitness_score: float = Field(ge=0.0, le=1.0)
    duration_seconds: float
    
    # Meta-learning information
    strategy_effectiveness: Dict[str, float] = Field(
        default_factory=dict,
        description="How well different strategies worked"
    )
    insights: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_rl_tuple(self) -> tuple:
        """Convert to classic RL tuple for training."""
        return (
            self.state.dict(),
            self.action.dict(),
            self.reward,
            self.next_state.dict()
        )


class RLTrajectory(BaseModel):
    """
    Sequence of episodes forming a trajectory.
    
    Used for:
    - GRPO training (requires trajectory groups)
    - Multi-step RL algorithms
    - Credit assignment over time
    
    Storage key pattern: "trajectory:{trajectory_id}"
    """
    trajectory_id: str
    agent_id: str
    civilization_id: Optional[str] = None
    
    # Sequence of episodes
    episodes: List[RLEpisode] = Field(default_factory=list)
    
    # Trajectory metadata
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_reward: float = 0.0
    average_reward: float = 0.0
    
    # Learning metrics
    improvement_over_baseline: float = 0.0
    meta_patterns_discovered: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_episode(self, episode: RLEpisode) -> None:
        """Add an episode to the trajectory."""
        self.episodes.append(episode)
        self.total_reward += episode.reward
        self.average_reward = self.total_reward / len(self.episodes)
        self.end_time = datetime.utcnow()


class AgentLegacy(BaseModel):
    """
    Preserves an agent's best methods and knowledge.
    
    This is the "memory" that gets passed down:
    - Best strategies per station
    - Successful tool patterns
    - Effective reasoning chains
    - Meta-learning insights
    
    Storage key pattern: "legacy:agent:{agent_id}"
    """
    agent_id: str
    civilization_id: Optional[str] = None
    generation: int = Field(ge=0, description="Which generation (0 = original)")
    
    # Performance history
    total_episodes: int = 0
    total_reward: float = 0.0
    average_performance: float = 0.0
    peak_performance: float = 0.0
    
    # Best methods per station
    best_strategies: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Best strategy for each station with metadata"
    )
    
    # Learned patterns
    effective_tool_sequences: List[Dict[str, Any]] = Field(default_factory=list)
    successful_reasoning_patterns: List[str] = Field(default_factory=list)
    meta_insights: List[str] = Field(default_factory=list)
    
    # RL policy (if trained)
    rl_policy: Optional[Dict[str, Any]] = None
    policy_version: Optional[str] = None
    policy_training_episodes: int = 0
    
    # Lineage tracking
    parent_agent_ids: List[str] = Field(default_factory=list)
    child_agent_ids: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_from_episode(self, episode: RLEpisode) -> None:
        """Update legacy from a new episode."""
        self.total_episodes += 1
        self.total_reward += episode.reward
        self.average_performance = self.total_reward / self.total_episodes
        self.peak_performance = max(self.peak_performance, episode.reward)
        
        # Update best strategy for station if this one is better
        station = episode.station
        strategy = episode.action.strategy
        
        if station not in self.best_strategies:
            self.best_strategies[station] = {
                "strategy": strategy,
                "avg_reward": episode.reward,
                "uses": 1,
                "confidence": episode.action.confidence
            }
        else:
            current_best = self.best_strategies[station]
            # Exponential moving average
            current_best["avg_reward"] = (
                0.9 * current_best["avg_reward"] + 0.1 * episode.reward
            )
            
            if episode.reward > current_best["avg_reward"]:
                current_best["strategy"] = strategy
                current_best["confidence"] = episode.action.confidence
            
            current_best["uses"] += 1
        
        # Add insights
        if episode.insights:
            self.meta_insights.extend(episode.insights)
            # Keep only unique insights
            self.meta_insights = list(set(self.meta_insights))
        
        self.last_updated = datetime.utcnow()


class CivilizationLegacy(BaseModel):
    """
    Preserves a civilization's collective knowledge.
    
    This is the society-level memory that enables
    true multi-generational learning.
    
    Storage key pattern: "legacy:civilization:{civilization_id}"
    """
    civilization_id: str
    name: str
    generation: int = Field(ge=0)
    
    # Population history
    total_agents_created: int = 0
    active_agents: int = 0
    generations_completed: int = 0
    
    # Collective knowledge
    best_agent_legacies: List[str] = Field(
        default_factory=list,
        description="Agent IDs of top performers"
    )
    
    collective_strategies: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Strategies that work well across agents"
    )
    
    discovered_meta_patterns: List[str] = Field(default_factory=list)
    
    # Evolution metrics
    avg_performance_by_generation: Dict[int, float] = Field(default_factory=dict)
    best_performance_by_generation: Dict[int, float] = Field(default_factory=dict)
    
    # RL training history
    rl_training_runs: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RLTrainingRun(BaseModel):
    """
    Record of an RL training session.
    
    Preserves training metadata for auditing and improvement.
    
    Storage key pattern: "rl_training:{run_id}"
    """
    run_id: str
    agent_id: str
    civilization_id: Optional[str] = None
    
    # Training configuration
    algorithm: str = "grpo"  # GRPO, PPO, SAC, etc.
    num_episodes: int
    num_iterations: int
    learning_rate: float
    
    # Training data
    episodes_used: List[str] = Field(
        default_factory=list,
        description="Episode IDs used for training"
    )
    
    # Results
    baseline_performance: float
    final_performance: float
    improvement_percentage: float
    
    # Learned policy
    policy: Dict[str, Any] = Field(default_factory=dict)
    policy_version: str
    
    # Training metrics
    training_duration_seconds: float
    compute_backend: str = "local"  # "local", "coreweave", "aws", etc.
    
    # Validation
    validation_results: Optional[Dict[str, Any]] = None
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Helper functions for querying RL data

def generate_episode_key(agent_id: str, station: str, sequence: int) -> str:
    """Generate consistent episode key."""
    return f"episode:{agent_id}:{station}:{sequence:06d}"


def generate_trajectory_key(agent_id: str, sequence: int) -> str:
    """Generate consistent trajectory key."""
    return f"trajectory:{agent_id}:{sequence:06d}"


def generate_legacy_key(agent_id: str) -> str:
    """Generate consistent legacy key."""
    return f"legacy:agent:{agent_id}"


def generate_civilization_key(civilization_id: str) -> str:
    """Generate consistent civilization key."""
    return f"legacy:civilization:{civilization_id}"


def generate_training_run_key(run_id: str) -> str:
    """Generate consistent training run key."""
    return f"rl_training:{run_id}"

