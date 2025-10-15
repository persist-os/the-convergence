"""
Civilization Runtime Engine - Executes AI civilizations with RLP, SAO, MAB, and Evolution.

This is where agents actually run, learn, and evolve in a simulation loop.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import weave

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from convergence.core.weave_logger import init_weave_logger, get_weave_logger

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

# Suppress litellm spam
logging.getLogger("litellm").setLevel(logging.ERROR)
os.environ["LITELLM_LOG"] = "ERROR"

console = Console()


@dataclass
class AgentMetrics:
    """Track agent performance metrics."""
    
    agent_id: str
    total_actions: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    rlp_thoughts_generated: int = 0
    mab_explorations: int = 0
    mab_exploitations: int = 0
    learning_episodes: int = 0
    fitness_score: float = 0.0


@dataclass
class CivilizationState:
    """Current state of the civilization."""
    
    iteration: int = 0
    total_actions: int = 0
    total_rewards: float = 0.0
    agents: List[Any] = field(default_factory=list)
    metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class Environment:
    """
    Simple environment for agents to interact with.
    
    Provides tasks, rewards agents for good performance.
    """
    
    def __init__(self):
        """Initialize environment."""
        self.tasks = [
            {"id": "task_1", "difficulty": 0.3, "type": "exploration"},
            {"id": "task_2", "difficulty": 0.5, "type": "problem_solving"},
            {"id": "task_3", "difficulty": 0.7, "type": "collaboration"},
            {"id": "task_4", "difficulty": 0.4, "type": "learning"},
            {"id": "task_5", "difficulty": 0.6, "type": "optimization"},
        ]
        self.task_history = []
    
    def get_task(self) -> Dict[str, Any]:
        """Get a random task for agent."""
        return random.choice(self.tasks)
    
    def evaluate_action(
        self,
        task: Dict[str, Any],
        action: Dict[str, Any],
        agent_thought: str
    ) -> float:
        """
        Evaluate agent's action and return reward.
        
        Higher reward for:
        - Good reasoning (longer, more detailed thoughts)
        - Appropriate strategy selection
        - Task difficulty completion
        """
        reward = 0.0
        
        # Base reward from task difficulty
        difficulty = task.get("difficulty", 0.5)
        reward += difficulty * 0.5
        
        # Bonus for thoughtful reasoning (RLP)
        if agent_thought and len(agent_thought) > 50:
            reward += 0.2
        if agent_thought and len(agent_thought) > 100:
            reward += 0.1
        
        # Bonus for strategy selection (MAB)
        strategy = action.get("strategy", "")
        task_type = task.get("type", "")
        
        # Reward matching strategy to task type
        strategy_match = {
            "exploration": "explore",
            "problem_solving": "exploit",
            "collaboration": "cooperate",
            "learning": "explore",
            "optimization": "exploit",
        }
        
        if strategy == strategy_match.get(task_type):
            reward += 0.3
        
        # Add some randomness (environment uncertainty)
        reward += random.uniform(-0.1, 0.1)
        
        # Clip to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Store in history
        self.task_history.append({
            "task": task,
            "action": action,
            "thought": agent_thought,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })
        
        return reward


class CivilizationRuntime:
    """
    Runtime engine for AI civilizations.
    
    Manages:
    - Agent execution loops
    - Learning and evolution
    - Metrics and observability
    - Survival selection (Darwin)
    """
    
    def __init__(
        self,
        agents: List[Any],
        max_iterations: int = 100,
        evolution_enabled: bool = True,
        evolution_frequency: int = 20,
        verbose: bool = False,
    ):
        """
        Initialize civilization runtime.
        
        Args:
            agents: List of agent instances
            max_iterations: Max iterations to run
            evolution_enabled: Enable Darwin-style evolution
            evolution_frequency: How often to trigger evolution
            verbose: If True, show detailed LLM outputs
        """
        self.agents = agents
        self.max_iterations = max_iterations
        self.evolution_enabled = evolution_enabled
        self.evolution_frequency = evolution_frequency
        self.verbose = verbose
        
        self.environment = Environment()
        self.state = CivilizationState(agents=agents)
        
        # Initialize Weave logger (reads organization/project from env variables)
        # Env vars: WANDB_ENTITY (or WEAVE_ORGANIZATION) and WANDB_PROJECT (or WEAVE_PROJECT)
        self.weave_logger = init_weave_logger(enabled=True)
        
        if self.weave_logger and self.weave_logger.enabled:
            console.print(f"[dim]📊 Weave tracing enabled - {self.weave_logger.project}[/dim]")
        else:
            console.print(f"[dim yellow]⚠️  Weave tracing disabled[/dim yellow]")
            if not os.getenv("WANDB_ENTITY") and not os.getenv("WEAVE_ORGANIZATION"):
                console.print(f"[dim yellow]   Set WANDB_ENTITY in .env to enable tracing[/dim yellow]")
        
        # Initialize metrics for each agent
        for agent in agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            self.state.metrics[agent_id] = AgentMetrics(agent_id=agent_id)
    
    @weave.op()
    async def run(self) -> CivilizationState:
        """
        Run the civilization simulation.
        
        Main execution loop where agents:
        1. Receive tasks from environment
        2. Think (RLP) and select strategies (MAB)
        3. Take actions
        4. Learn from rewards
        5. Evolve over time
        """
        
        console.print(Panel.fit(
            "[bold cyan]Civilization Running[/bold cyan]\n"
            f"Agents: {len(self.agents)} | "
            f"Max Iterations: {self.max_iterations} | "
            f"Evolution: {'Enabled' if self.evolution_enabled else 'Disabled'}",
            border_style="cyan"
        ))
        
        # Main simulation loop
        for iteration in range(self.max_iterations):
            self.state.iteration = iteration + 1
            
            # Each agent acts
            for agent in self.agents:
                await self._agent_step(agent)
            
            # Evolution check
            if (self.evolution_enabled and 
                iteration > 0 and 
                iteration % self.evolution_frequency == 0):
                await self._evolve_population()
            
            # Display progress
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                self._display_progress()
                
                # Log civilization metrics periodically
                if self.weave_logger and self.weave_logger.enabled:
                    agent_metrics_dict = {
                        agent_id: {
                            'total_actions': m.total_actions,
                            'avg_reward': m.avg_reward,
                            'fitness_score': m.fitness_score,
                            'rlp_thoughts': m.rlp_thoughts_generated,
                            'explorations': m.mab_explorations,
                            'exploitations': m.mab_exploitations
                        }
                        for agent_id, m in self.state.metrics.items()
                    }
                    self.weave_logger.log_civilization_metrics(
                        iteration=self.state.iteration,
                        total_actions=self.state.total_actions,
                        total_rewards=self.state.total_rewards,
                        avg_reward=self.state.total_rewards / max(self.state.total_actions, 1),
                        agent_metrics=agent_metrics_dict
                    )
            
            # Small delay for readability
            await asyncio.sleep(0.1)
        
        # Final summary
        self._display_final_summary()
        
        return self.state
    
    @weave.op()
    async def _agent_step(self, agent: Any) -> None:
        """Execute one step for an agent."""
        
        agent_id = getattr(agent, 'agent_id', str(agent))
        metrics = self.state.metrics[agent_id]
        
        # Get task from environment
        task = self.environment.get_task()
        
        # Agent acts (with RLP thinking and MAB strategy selection)
        state = {
            "task": task,
            "iteration": self.state.iteration,
            "goal": f"complete_{task['type']}"
        }
        
        try:
            # Agent thinks and acts
            result = await agent.act(state)
            
            # Extract information
            thought = result.get("thought", "")
            strategy = result.get("strategy", "explore")
            action = result.get("action", "completed")
            
            # Verbose output: show actual LLM-generated thoughts
            if self.verbose and thought:
                console.print(f"\n[dim cyan]💭 {agent_id} thinking:[/dim cyan]")
                console.print(f"[dim]{thought[:200]}{'...' if len(thought) > 200 else ''}[/dim]")
            
            # Environment evaluates and gives reward
            reward = self.environment.evaluate_action(task, result, thought)
            
            # Log to Weave
            if self.weave_logger and self.weave_logger.enabled:
                self.weave_logger.log_agent_action(
                    agent_id=agent_id,
                    iteration=self.state.iteration,
                    state=state,
                    thought=thought,
                    strategy=strategy,
                    action=result,
                    reward=reward
                )
            
            # Agent learns from experience
            experience = {
                "state": state,
                "thought": thought,
                "strategy": strategy,
                "action": action,
                "reward": reward,
                "task": task
            }
            
            await agent.learn(experience)
            
            # Update metrics
            metrics.total_actions += 1
            metrics.total_reward += reward
            metrics.avg_reward = metrics.total_reward / metrics.total_actions
            metrics.learning_episodes += 1
            
            if thought:
                metrics.rlp_thoughts_generated += 1
            
            if strategy == "explore":
                metrics.mab_explorations += 1
            else:
                metrics.mab_exploitations += 1
            
            # Update fitness (used for evolution)
            metrics.fitness_score = self._calculate_fitness(metrics)
            
            # Update state
            self.state.total_actions += 1
            self.state.total_rewards += reward
            
        except Exception as e:
            console.print(f"[red]Error in agent {agent_id}: {e}[/red]")
    
    def _calculate_fitness(self, metrics: AgentMetrics) -> float:
        """
        Calculate agent fitness for evolution.
        
        Fitness based on:
        - Average reward (performance)
        - Learning episodes (experience)
        - Exploration/exploitation balance
        """
        fitness = 0.0
        
        # Average reward is primary factor
        fitness += metrics.avg_reward * 0.6
        
        # Experience bonus (but with diminishing returns)
        experience_bonus = min(metrics.learning_episodes / 100, 0.2)
        fitness += experience_bonus
        
        # Balance bonus (good to explore AND exploit)
        if metrics.total_actions > 0:
            explore_ratio = metrics.mab_explorations / metrics.total_actions
            # Reward ~30-70% exploration rate
            if 0.3 <= explore_ratio <= 0.7:
                fitness += 0.2
        
        return fitness
    
    @weave.op()
    async def _evolve_population(self) -> None:
        """
        Apply Darwin-style evolution to agent population.
        
        Selection based on fitness:
        - Top performers survive
        - Weak performers are replaced
        - Creates selection pressure for better strategies
        """
        
        console.print("\n[yellow]🧬 Evolution Event![/yellow]")
        
        # Calculate fitness for all agents
        fitnesses = [
            (agent, self.state.metrics[getattr(agent, 'agent_id', str(agent))].fitness_score)
            for agent in self.agents
        ]
        
        # Sort by fitness
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        # Show fitness rankings
        console.print("[dim]Fitness Rankings:[/dim]")
        for i, (agent, fitness) in enumerate(fitnesses[:3], 1):
            agent_id = getattr(agent, 'agent_id', str(agent))
            console.print(f"  {i}. {agent_id}: {fitness:.3f}")
        
        # Log to Weave
        if self.weave_logger and self.weave_logger.enabled:
            fitness_rankings = [
                {
                    'agent_id': getattr(agent, 'agent_id', str(agent)),
                    'fitness': fitness
                }
                for agent, fitness in fitnesses
            ]
            self.weave_logger.log_evolution_event(
                iteration=self.state.iteration,
                generation=self.state.iteration // 20,  # Based on evolution_frequency
                fitness_rankings=fitness_rankings,
                selection_stats={
                    'num_agents': len(self.agents),
                    'top_fitness': fitnesses[0][1] if fitnesses else 0.0,
                    'avg_fitness': sum(f[1] for f in fitnesses) / len(fitnesses) if fitnesses else 0.0
                }
            )
        
        # In a full implementation, we'd:
        # - Keep top 50% of agents
        # - Create variants of top performers
        # - Replace bottom 50% with variants
        # For now, just log the event
        
        console.print("[green]Selection complete![/green]\n")
    
    def _display_progress(self) -> None:
        """Display current progress."""
        
        table = Table(title=f"Iteration {self.state.iteration}/{self.max_iterations}")
        
        table.add_column("Agent", style="cyan")
        table.add_column("Actions", justify="right")
        table.add_column("Avg Reward", justify="right", style="green")
        table.add_column("Thoughts", justify="right")
        table.add_column("Fitness", justify="right", style="yellow")
        
        for agent in self.agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            metrics = self.state.metrics[agent_id]
            
            table.add_row(
                agent_id,
                str(metrics.total_actions),
                f"{metrics.avg_reward:.3f}",
                str(metrics.rlp_thoughts_generated),
                f"{metrics.fitness_score:.3f}"
            )
        
        console.print(table)
        console.print()
    
    def _display_final_summary(self) -> None:
        """Display final summary of civilization run."""
        
        runtime = (datetime.now() - self.state.start_time).total_seconds()
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            f"[bold green]Civilization Complete![/bold green]\n\n"
            f"Total Iterations: {self.state.iteration}\n"
            f"Total Actions: {self.state.total_actions}\n"
            f"Total Rewards: {self.state.total_rewards:.2f}\n"
            f"Avg Reward: {self.state.total_rewards/self.state.total_actions:.3f}\n"
            f"Runtime: {runtime:.1f}s",
            border_style="green"
        ))
        
        # Show top performer
        if self.state.metrics:
            top_agent = max(
                self.state.metrics.values(),
                key=lambda m: m.fitness_score
            )
            
            console.print(f"\n[bold]🏆 Top Performer: {top_agent.agent_id}[/bold]")
            console.print(f"  Fitness: {top_agent.fitness_score:.3f}")
            console.print(f"  Avg Reward: {top_agent.avg_reward:.3f}")
            console.print(f"  Thoughts Generated: {top_agent.rlp_thoughts_generated}")
            console.print(f"  Explorations: {top_agent.mab_explorations}")
            console.print(f"  Exploitations: {top_agent.mab_exploitations}")


__all__ = ['CivilizationRuntime', 'Environment', 'CivilizationState', 'AgentMetrics']

