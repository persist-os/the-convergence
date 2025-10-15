"""
Core protocols defining the interfaces for The Convergence framework.

All components implement these protocols for maximum flexibility and composability.
Uses Python's Protocol for structural subtyping (PEP 544).
"""

from typing import Protocol, Any, Dict, List, Optional, runtime_checkable
from pydantic import BaseModel


# ============================================================================
# PROVIDER PROTOCOLS
# ============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers - any implementation that matches this works."""
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Returns:
            Dict with 'content' and optional 'metadata'
        """
        ...
    
    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any
    ) -> BaseModel:
        """Generate structured output matching Pydantic schema."""
        ...


# ============================================================================
# LEARNING PROTOCOLS
# ============================================================================

@runtime_checkable
class MABStrategy(Protocol):
    """Protocol for Multi-Armed Bandit algorithms."""
    
    def select_arm(self, arms: List[str], state: Dict[str, Any]) -> str:
        """Select which arm to pull given current state."""
        ...
    
    def update(self, arm: str, reward: float, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state after observing reward."""
        ...


@runtime_checkable
class RLPLearner(Protocol):
    """Protocol for Reinforcement Learning Pretraining (NVIDIA research)."""
    
    async def generate_internal_reasoning(self, state: Dict[str, Any]) -> str:
        """
        Generate chain-of-thought before making prediction.
        
        Core of RLP: model thinks before acting.
        """
        ...
    
    def information_gain_reward(
        self,
        thought: str,
        prediction: str,
        outcome: str
    ) -> float:
        """
        Calculate reward based on whether thought improved prediction.
        
        Reward = improvement in log-likelihood with thought vs without.
        """
        ...


@runtime_checkable
class SAOGenerator(Protocol):
    """Protocol for Self-Alignment Optimization (Hugging Face research)."""
    
    async def generate_synthetic_prompts(
        self,
        n_samples: int,
        persona_templates: List[str]
    ) -> List[str]:
        """Generate diverse prompts via persona role-play."""
        ...
    
    async def generate_response_pairs(self, prompt: str) -> tuple[str, str]:
        """Generate two responses for comparison."""
        ...
    
    async def self_judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> tuple[str, str]:
        """
        Self-evaluate responses to create preference pairs.
        
        Returns: (winning_response, losing_response)
        """
        ...


# ============================================================================
# MEMORY PROTOCOLS
# ============================================================================

@runtime_checkable
class MemorySystem(Protocol):
    """Base protocol for memory systems."""
    
    async def store(
        self,
        agent_id: str,
        experience: Dict[str, Any]
    ) -> None:
        """Store experience in memory."""
        ...
    
    async def retrieve(
        self,
        agent_id: str,
        query: Dict[str, Any],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories."""
        ...


@runtime_checkable
class ProceduralMemory(Protocol):
    """Protocol for procedural memory (Memp research)."""
    
    async def distill_strategy(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Distill past trajectories into reusable strategies.
        
        Creates both step-by-step and script-level abstractions.
        """
        ...
    
    async def transfer_to_agent(
        self,
        strategy: Dict[str, Any],
        target_agent_id: str
    ) -> bool:
        """Transfer strategy from strong model to weaker one."""
        ...


@runtime_checkable
class SemanticMemory(Protocol):
    """Protocol for semantic/graph memory (Mem0 research)."""
    
    async def extract_entities(
        self,
        text: str
    ) -> List[tuple[str, str]]:
        """Extract entities as (entity, type) pairs."""
        ...
    
    async def extract_relations(
        self,
        text: str
    ) -> List[tuple[str, str, str]]:
        """Extract relations as (entity1, relation, entity2) triples."""
        ...
    
    async def update_graph(
        self,
        entities: List[tuple[str, str]],
        relations: List[tuple[str, str, str]],
        operation: str  # ADD, UPDATE, DELETE, NOOP
    ) -> None:
        """Update knowledge graph with conflict resolution."""
        ...


@runtime_checkable
class EpisodicMemory(Protocol):
    """Protocol for episodic memory (Nemori research)."""
    
    async def segment_by_semantics(
        self,
        conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Segment conversation into coherent episodes.
        
        Uses semantic boundaries, not arbitrary chunking.
        """
        ...
    
    async def predict_and_calibrate(
        self,
        episode: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from prediction gaps (Predict-Calibrate Principle).
        
        Proactively learns from what it gets wrong.
        """
        ...


# ============================================================================
# AGENT PROTOCOL
# ============================================================================

@runtime_checkable
class Agent(Protocol):
    """Protocol for agents in The Convergence."""
    
    agent_id: str
    config: Dict[str, Any]
    
    async def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Take action given current state."""
        ...
    
    async def learn(self, experience: Dict[str, Any]) -> None:
        """Learn from experience."""
        ...
    
    async def evolve(self, fitness: float) -> Optional["Agent"]:
        """Evolve based on fitness (may create offspring)."""
        ...


# ============================================================================
# PLUGIN PROTOCOL
# ============================================================================

@runtime_checkable
class Plugin(Protocol):
    """Protocol for plugins in The Convergence."""
    
    name: str
    version: str
    description: str
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        ...
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        ...


# ============================================================================
# STATION PROTOCOL (for challenges/tasks)
# ============================================================================

@runtime_checkable
class Station(Protocol):
    """Protocol for stations (challenges/tasks)."""
    
    def get_challenge(self, level: int) -> Dict[str, Any]:
        """Get challenge for given level."""
        ...
    
    async def execute(
        self,
        agent: Agent,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agent's action and return result.
        
        Returns: Dict with 'success', 'reward', 'next_state'
        """
        ...


# ============================================================================
# EVOLUTION PROTOCOL
# ============================================================================

@runtime_checkable
class EvolutionEngine(Protocol):
    """Protocol for evolution/selection mechanisms."""
    
    def spawn_variants(
        self,
        agent: Agent,
        n_variants: int,
        mutation_rate: float
    ) -> List[Agent]:
        """Create variants of agent with mutations."""
        ...
    
    def select_survivors(
        self,
        agents: List[Agent],
        fitness_scores: List[float],
        survival_rate: float
    ) -> List[Agent]:
        """Select survivors based on fitness."""
        ...
    
    def crossover(self, agent_a: Agent, agent_b: Agent) -> Agent:
        """Create offspring by combining two agents."""
        ...

