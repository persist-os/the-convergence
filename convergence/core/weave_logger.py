"""
Weave Logger - Observability and Metrics Tracking for The Convergence

Provides comprehensive tracing and metrics tracking for:
- Agent performance and learning metrics
- LLM calls and token usage
- MAB strategy convergence
- RLP/SAO learning progress
- Evolution and fitness tracking
"""

import os
import weave
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WeaveLogger:
    """
    Weave-based observability for The Convergence civilizations.
    
    Tracks:
    - Agent actions and thoughts (RLP)
    - Strategy selection and rewards (MAB)
    - Synthetic data generation (SAO)
    - Evolution events and fitness scores
    - LLM token usage and costs
    """
    
    def __init__(
        self, 
        organization: Optional[str] = None,
        project: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Weave logger.
        
        Args:
            organization: Weave organization/entity (reads from WANDB_ENTITY or WEAVE_ORGANIZATION env var)
            project: Weave project name (reads from WANDB_PROJECT or WEAVE_PROJECT env var)
            enabled: Whether to enable Weave logging
        
        Environment Variables:
            WANDB_ENTITY or WEAVE_ORGANIZATION: Organization name (REQUIRED)
            WANDB_PROJECT or WEAVE_PROJECT: Project name (default: "learning-society")
        """
        # Read organization from environment variables
        self.organization = organization or os.getenv("WEAVE_ORGANIZATION") or os.getenv("WANDB_ENTITY")
        self.project_name = project or os.getenv("WEAVE_PROJECT") or os.getenv("WANDB_PROJECT", "learning-society")
        
        # Check if organization is set
        if not self.organization:
            logger.warning("[Weave] No organization specified. Set WANDB_ENTITY or WEAVE_ORGANIZATION environment variable.")
            logger.warning("[Weave] Disabling Weave tracing.")
            self.enabled = False
            self.project = None
            self.run_start_time = datetime.now()
            return
        
        # Construct full project path: organization/project
        self.project = f"{self.organization}/{self.project_name}"
        
        self.enabled = enabled
        self.run_start_time = datetime.now()
        
        if self.enabled:
            try:
                # Initialize Weave for this project
                weave.init(self.project)
                logger.info(f"[Weave] Initialized and ready for project: {self.project}")
            except Exception as e:
                logger.warning(f"[Weave] Initialization failed: {e}")
                logger.warning(f"[Weave] Make sure '{self.organization}' is a valid W&B entity/organization")
                logger.warning("[Weave] Continuing without tracing")
                self.enabled = False
    
    @weave.op()
    def log_agent_action(
        self,
        agent_id: str,
        iteration: int,
        state: Dict[str, Any],
        thought: str,
        strategy: str,
        action: Dict[str, Any],
        reward: float
    ):
        """
        Log a complete agent action cycle.
        
        Args:
            agent_id: Agent identifier
            iteration: Current iteration number
            state: Agent state at time of action
            thought: Internal reasoning generated (RLP)
            strategy: Strategy selected (MAB)
            action: Action taken
            reward: Reward received
        """
        if not self.enabled:
            return
        
        try:
            # Return structured data for Weave to track
            return {
                'agent_id': agent_id,
                'iteration': iteration,
                'thought_length': len(thought) if thought else 0,
                'thought_preview': thought[:100] if thought else "",
                'strategy': strategy,
                'action_type': action.get('action', 'unknown'),
                'reward': reward,
                'state_snapshot': {
                    'goal': state.get('goal', 'unknown'),
                    'iteration': state.get('iteration', iteration)
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging agent action: {e}")
            return {}
    
    @weave.op()
    def log_agent_learning(
        self,
        agent_id: str,
        iteration: int,
        experience: Dict[str, Any],
        rlp_metrics: Optional[Dict[str, Any]] = None,
        mab_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Log agent learning update.
        
        Args:
            agent_id: Agent identifier
            iteration: Current iteration
            experience: Learning experience
            rlp_metrics: RLP learning metrics
            mab_stats: MAB arm statistics
        """
        if not self.enabled:
            return
        
        try:
            return {
                'agent_id': agent_id,
                'iteration': iteration,
                'reward': experience.get('reward', 0.0),
                'rlp_metrics': rlp_metrics or {},
                'mab_stats': mab_stats or {},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging learning: {e}")
            return {}
    
    @weave.op()
    def log_mab_selection(
        self,
        agent_id: str,
        iteration: int,
        available_strategies: List[str],
        selected_strategy: str,
        arm_statistics: Dict[str, Any]
    ):
        """
        Log MAB arm selection.
        
        Args:
            agent_id: Agent identifier
            iteration: Current iteration
            available_strategies: All available strategies
            selected_strategy: Strategy that was selected
            arm_statistics: Current statistics for all arms
        """
        if not self.enabled:
            return
        
        try:
            return {
                'agent_id': agent_id,
                'iteration': iteration,
                'available_strategies': available_strategies,
                'selected_strategy': selected_strategy,
                'arm_statistics': arm_statistics,
                'exploration': selected_strategy == 'explore',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging MAB selection: {e}")
            return {}
    
    @weave.op()
    def log_evolution_event(
        self,
        iteration: int,
        generation: int,
        fitness_rankings: List[Dict[str, Any]],
        selection_stats: Dict[str, Any]
    ):
        """
        Log evolution/selection event.
        
        Args:
            iteration: Current iteration
            generation: Generation number
            fitness_rankings: Ranked list of agents by fitness
            selection_stats: Statistics about the selection process
        """
        if not self.enabled:
            return
        
        try:
            return {
                'iteration': iteration,
                'generation': generation,
                'fitness_rankings': fitness_rankings,
                'top_fitness': fitness_rankings[0]['fitness'] if fitness_rankings else 0.0,
                'avg_fitness': sum(r['fitness'] for r in fitness_rankings) / len(fitness_rankings) if fitness_rankings else 0.0,
                'selection_stats': selection_stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging evolution: {e}")
            return {}
    
    @weave.op()
    def log_llm_usage(
        self,
        agent_id: str,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        duration: float
    ):
        """
        Log LLM API usage.
        
        Args:
            agent_id: Agent that made the call
            operation: Type of operation (reasoning, prompt_gen, etc.)
            model: Model used
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            total_tokens: Total tokens
            duration: Call duration in seconds
        """
        if not self.enabled:
            return
        
        try:
            return {
                'agent_id': agent_id,
                'operation': operation,
                'model': model,
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                },
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging LLM usage: {e}")
            return {}
    
    @weave.op()
    def log_civilization_metrics(
        self,
        iteration: int,
        total_actions: int,
        total_rewards: float,
        avg_reward: float,
        agent_metrics: Dict[str, Dict[str, Any]]
    ):
        """
        Log overall civilization metrics.
        
        Args:
            iteration: Current iteration
            total_actions: Total actions taken across all agents
            total_rewards: Sum of all rewards
            avg_reward: Average reward per action
            agent_metrics: Per-agent metrics
        """
        if not self.enabled:
            return
        
        try:
            return {
                'iteration': iteration,
                'total_actions': total_actions,
                'total_rewards': total_rewards,
                'avg_reward': avg_reward,
                'num_agents': len(agent_metrics),
                'agent_metrics': agent_metrics,
                'runtime_seconds': (datetime.now() - self.run_start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[Weave] Error logging civilization metrics: {e}")
            return {}
    
    def get_dashboard_url(self) -> Optional[str]:
        """
        Get Weave dashboard URL for this run.
        
        Returns:
            Dashboard URL or None if not available
        """
        if not self.enabled:
            return None
        
        try:
            # Construct Weave UI URL
            # Format: https://wandb.ai/entity/project/weave
            return f"https://wandb.ai/{self.project}/weave"
        except Exception as e:
            logger.error(f"[Weave] Error getting dashboard URL: {e}")
            return None


# Global instance
_weave_logger: Optional[WeaveLogger] = None


def get_weave_logger() -> Optional[WeaveLogger]:
    """Get global Weave logger instance."""
    return _weave_logger


def init_weave_logger(
    organization: Optional[str] = None,
    project: Optional[str] = None,
    enabled: bool = True
) -> WeaveLogger:
    """
    Initialize global Weave logger.
    
    Args:
        organization: Weave organization/entity (reads from env if not provided)
        project: Weave project name (reads from env if not provided)
        enabled: Whether to enable Weave logging
    
    Environment Variables:
        WANDB_ENTITY or WEAVE_ORGANIZATION: Organization name (default: "the-convergence")
        WANDB_PROJECT or WEAVE_PROJECT: Project name (default: "learning-society")
    """
    global _weave_logger
    _weave_logger = WeaveLogger(organization=organization, project=project, enabled=enabled)
    return _weave_logger


__all__ = ['WeaveLogger', 'get_weave_logger', 'init_weave_logger']

