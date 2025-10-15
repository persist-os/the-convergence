"""
Thompson Sampling MAB Strategy

A Bayesian approach to the multi-armed bandit problem.
Balances exploration and exploitation naturally through posterior sampling.
"""

from typing import Any, Dict, List
import random
from pydantic import BaseModel, Field
import weave


class ThompsonSamplingConfig(BaseModel):
    """Configuration for Thompson Sampling."""
    
    alpha_prior: float = Field(
        default=1.0,
        description="Prior alpha (successes) for Beta distribution"
    )
    beta_prior: float = Field(
        default=1.0,
        description="Prior beta (failures) for Beta distribution"
    )


class ThompsonSamplingStrategy:
    """
    Thompson Sampling for Multi-Armed Bandits.
    
    Maintains Beta(alpha, beta) distribution for each arm.
    Samples from each distribution and selects the arm with highest sample.
    
    Benefits:
    - Naturally balances exploration/exploitation
    - Bayesian interpretation
    - Strong theoretical guarantees
    """
    
    def __init__(self, config: ThompsonSamplingConfig | None = None):
        """Initialize Thompson Sampling strategy."""
        self.config = config or ThompsonSamplingConfig()
        
        # State: Beta distribution parameters for each arm
        self.arm_stats: Dict[str, Dict[str, float]] = {}
    
    def _ensure_arm_exists(self, arm: str) -> None:
        """Ensure arm is initialized in our stats."""
        if arm not in self.arm_stats:
            self.arm_stats[arm] = {
                'alpha': self.config.alpha_prior,
                'beta': self.config.beta_prior,
                'pulls': 0,
                'total_reward': 0.0,
            }
    
    @weave.op()
    def select_arm(self, arms: List[str], state: Dict[str, Any]) -> str:
        """
        Select arm using Thompson Sampling.
        
        Args:
            arms: List of available arms
            state: Current state (not used in basic Thompson Sampling)
            
        Returns:
            Selected arm name
        """
        if not arms:
            raise ValueError("No arms available for selection")
        
        # Ensure all arms are initialized
        for arm in arms:
            self._ensure_arm_exists(arm)
        
        # Sample from each arm's Beta distribution
        samples = {}
        for arm in arms:
            alpha = self.arm_stats[arm]['alpha']
            beta = self.arm_stats[arm]['beta']
            
            # Sample from Beta(alpha, beta)
            try:
                import numpy as np
                sample = np.random.beta(alpha, beta)
            except ImportError:
                # Fallback if numpy not available
                sample = random.betavariate(alpha, beta)
            
            samples[arm] = sample
        
        # Select arm with highest sample
        selected_arm = max(samples.items(), key=lambda x: x[1])[0]
        
        return selected_arm
    
    @weave.op()
    def update(
        self,
        arm: str,
        reward: float,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update arm statistics after observing reward.
        
        For Thompson Sampling with Beta distributions:
        - Treat reward in [0,1] as success probability
        - Update: alpha += reward, beta += (1 - reward)
        
        Args:
            arm: Arm that was pulled
            reward: Observed reward (should be in [0, 1])
            state: Current state
            
        Returns:
            Updated state with statistics
        """
        self._ensure_arm_exists(arm)
        
        # Clip reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Update Beta distribution parameters
        self.arm_stats[arm]['alpha'] += reward
        self.arm_stats[arm]['beta'] += (1.0 - reward)
        self.arm_stats[arm]['pulls'] += 1
        self.arm_stats[arm]['total_reward'] += reward
        
        # Add MAB statistics to state
        state['mab_stats'] = {
            'arm': arm,
            'reward': reward,
            'pulls': self.arm_stats[arm]['pulls'],
            'estimated_mean': self._get_estimated_mean(arm),
            'total_arms': len(self.arm_stats),
        }
        
        return state
    
    def _get_estimated_mean(self, arm: str) -> float:
        """Get estimated mean reward for an arm."""
        if arm not in self.arm_stats:
            return 0.0
        
        stats = self.arm_stats[arm]
        # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
        return stats['alpha'] / (stats['alpha'] + stats['beta'])
    
    def get_arm_statistics(self, arm: str) -> Dict[str, Any]:
        """Get detailed statistics for an arm."""
        if arm not in self.arm_stats:
            return {}
        
        stats = self.arm_stats[arm]
        return {
            'arm': arm,
            'pulls': stats['pulls'],
            'total_reward': stats['total_reward'],
            'mean_reward': stats['total_reward'] / stats['pulls'] if stats['pulls'] > 0 else 0.0,
            'estimated_mean': self._get_estimated_mean(arm),
            'alpha': stats['alpha'],
            'beta': stats['beta'],
        }
    
    def get_all_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all arms."""
        return [
            self.get_arm_statistics(arm)
            for arm in self.arm_stats.keys()
        ]


class ThompsonSamplingPlugin:
    """
    Plugin wrapper for Thompson Sampling MAB strategy.
    
    Usage:
        from convergence.plugins.mab.thompson_sampling import ThompsonSamplingPlugin
        
        plugin = ThompsonSamplingPlugin()
        registry.register_plugin(plugin)
    """
    
    name = "thompson_sampling"
    version = "0.1.0"
    description = "Thompson Sampling for Multi-Armed Bandits"
    
    def __init__(self, config: ThompsonSamplingConfig | None = None):
        """Initialize the plugin."""
        self.config = config or ThompsonSamplingConfig()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        self.config = ThompsonSamplingConfig(**config)
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return [
            "arm_selection",
            "bayesian_learning",
            "exploration_exploitation",
            "posterior_sampling"
        ]
    
    def create_strategy(self) -> ThompsonSamplingStrategy:
        """Create Thompson Sampling strategy instance."""
        return ThompsonSamplingStrategy(config=self.config)


__all__ = ['ThompsonSamplingStrategy', 'ThompsonSamplingPlugin', 'ThompsonSamplingConfig']

