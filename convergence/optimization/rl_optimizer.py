"""
RL Meta-Optimizer for API Configuration Optimization.

Implements hierarchical learning: MAB (Layer 1) + RL (Layer 2)
- Layer 1 (MAB): Fast tactical decisions
- Layer 2 (RL): Strategic meta-patterns

This module learns "which parameter regions are promising" from
optimization history and biases future exploration accordingly.
"""
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import statistics

from .models import SearchSpaceConfig, SearchSpaceParameter


class RLMetaOptimizer:
    """
    RL-based meta-optimizer that learns from optimization history.
    
    Unlike pure MAB (which treats each configuration independently),
    the RL layer discovers:
    - Which parameter values consistently perform well
    - Which parameter combinations work together
    - Which search space regions to prioritize
    
    This hierarchical approach (MAB + RL) provides:
    - Fast decisions (MAB milliseconds)
    - Strategic improvement (RL discovers patterns)
    - Continuous learning (improves over time)
    """
    
    def __init__(
        self,
        search_space: SearchSpaceConfig,
        min_episodes_for_training: int = 50
    ):
        """
        Initialize RL meta-optimizer.
        
        Args:
            search_space: Search space configuration
            min_episodes_for_training: Minimum optimization runs needed
        """
        self.search_space = search_space
        self.min_episodes = min_episodes_for_training
        
        # Optimization history
        self.history: List[Dict[str, Any]] = []
        
        # Learned parameter preferences
        self.parameter_preferences: Dict[str, Dict[Any, float]] = {}
        
        # Meta-patterns discovered
        self.meta_patterns: List[str] = []
        
        # Policy (parameter biases)
        self.policy: Optional[Dict[str, Any]] = None
        self.policy_version: int = 0
    
    def record_episode(
        self,
        config: Dict[str, Any],
        score: float,
        metrics: Dict[str, float],
        generation: int
    ) -> None:
        """
        Record an optimization episode.
        
        Args:
            config: Configuration tested
            score: Aggregate fitness score
            metrics: Individual metric scores
            generation: Generation number
        """
        episode = {
            "config": config,
            "score": score,
            "metrics": metrics,
            "generation": generation
        }
        self.history.append(episode)
    
    def is_ready_for_training(self) -> bool:
        """Check if enough data exists for RL training."""
        return len(self.history) >= self.min_episodes
    
    def train_policy(self) -> Dict[str, Any]:
        """
        Train RL policy from optimization history.
        
        Discovers:
        1. Parameter value preferences (which values work well)
        2. Parameter importance (which params matter most)
        3. Meta-patterns (insights about search space)
        
        Returns:
            Policy dictionary with learned preferences
        """
        if not self.is_ready_for_training():
            raise ValueError(f"Need at least {self.min_episodes} episodes, have {len(self.history)}")
        
        print(f"\n🧠 RL META-OPTIMIZER TRAINING")
        print(f"━" * 60)
        print(f"📊 Analyzing {len(self.history)} optimization episodes...")
        
        # 1. Learn parameter value preferences
        self._learn_parameter_preferences()
        
        # 2. Calculate parameter importance
        param_importance = self._calculate_parameter_importance()
        
        # 3. Discover meta-patterns
        self._discover_meta_patterns()
        
        # 4. Create policy
        self.policy = {
            "version": self.policy_version + 1,
            "param_preferences": self.parameter_preferences,
            "param_importance": param_importance,
            "meta_patterns": self.meta_patterns,
            "bias_strength": 0.3  # How much to bias MAB (0-1)
        }
        self.policy_version += 1
        
        # Log insights
        print(f"\n✅ Policy v{self.policy_version} trained!")
        print(f"   📈 Parameter Importance:")
        for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"      • {param}: {importance:.3f}")
        
        print(f"\n   🔍 Meta-Patterns Discovered:")
        for i, pattern in enumerate(self.meta_patterns, 1):
            print(f"      {i}. {pattern}")
        
        return self.policy
    
    def _learn_parameter_preferences(self) -> None:
        """
        Learn which parameter values lead to high scores.
        
        For each parameter, track average score for each value.
        """
        # Group episodes by parameter values
        value_scores = defaultdict(lambda: defaultdict(list))
        
        for episode in self.history:
            config = episode["config"]
            score = episode["score"]
            
            for param_name, value in config.items():
                # Convert value to hashable type
                hashable_value = str(value) if isinstance(value, (dict, list)) else value
                value_scores[param_name][hashable_value].append(score)
        
        # Calculate average score for each value
        self.parameter_preferences = {}
        for param_name, values in value_scores.items():
            self.parameter_preferences[param_name] = {}
            for value, scores in values.items():
                avg_score = statistics.mean(scores) if scores else 0.0
                self.parameter_preferences[param_name][value] = avg_score
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate which parameters have the biggest impact on score.
        
        Uses variance in average scores across parameter values.
        """
        importance = {}
        
        for param_name, value_scores in self.parameter_preferences.items():
            if len(value_scores) < 2:
                importance[param_name] = 0.0
                continue
            
            # Calculate variance in average scores
            avg_scores = list(value_scores.values())
            mean = statistics.mean(avg_scores)
            variance = statistics.variance(avg_scores) if len(avg_scores) > 1 else 0.0
            
            # Normalize to [0, 1]
            importance[param_name] = min(1.0, variance * 10)  # Scale factor
        
        return importance
    
    def _discover_meta_patterns(self) -> None:
        """
        Discover high-level patterns in optimization history.
        
        Generates human-readable insights about what works.
        """
        self.meta_patterns = []
        
        # Pattern 1: Best parameter values
        for param_name, value_scores in self.parameter_preferences.items():
            if not value_scores:
                continue
            
            best_value = max(value_scores.items(), key=lambda x: x[1])
            if best_value[1] > 0.7:  # High score
                self.meta_patterns.append(
                    f"Parameter '{param_name}' works best with value '{best_value[0]}' (avg score: {best_value[1]:.3f})"
                )
        
        # Pattern 2: Score improvement over time
        if len(self.history) >= 10:
            early_avg = statistics.mean([ep["score"] for ep in self.history[:10]])
            recent_avg = statistics.mean([ep["score"] for ep in self.history[-10:]])
            improvement = recent_avg - early_avg
            
            if improvement > 0.05:
                self.meta_patterns.append(
                    f"Optimization is improving: +{improvement:.2%} from early to recent episodes"
                )
        
        # Pattern 3: Parameter combinations
        # (Simplified - could be expanded to discover interactions)
        self.meta_patterns.append(
            f"Explored {len(self.history)} configurations across search space"
        )
    
    def bias_config_sampling(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Bias a configuration toward learned preferences.
        
        With probability `bias_strength`, replaces parameter values
        with preferred values from policy.
        
        Args:
            config: Configuration from MAB
            
        Returns:
            Biased configuration
        """
        if not self.policy:
            return config
        
        biased = config.copy()
        bias_strength = self.policy.get("bias_strength", 0.3)
        preferences = self.policy.get("param_preferences", {})
        
        import random
        
        for param_name, value in config.items():
            # With probability bias_strength, use preferred value
            if random.random() < bias_strength:
                if param_name in preferences and preferences[param_name]:
                    # Pick best value for this parameter
                    best_value = max(
                        preferences[param_name].items(),
                        key=lambda x: x[1]
                    )[0]
                    
                    # Convert back from string if needed
                    param_spec = self.search_space.parameters.get(param_name)
                    if param_spec:
                        biased[param_name] = self._convert_value(
                            best_value,
                            param_spec
                        )
        
        return biased
    
    def _convert_value(
        self,
        value_str: str,
        param_spec: SearchSpaceParameter
    ) -> Any:
        """Convert string value back to proper type."""
        if param_spec.type == "continuous":
            return float(value_str)
        elif param_spec.type in ["discrete", "categorical"]:
            # Try to match to original value
            for orig_value in param_spec.values:
                if str(orig_value) == value_str:
                    return orig_value
            # Fallback: return first value
            return param_spec.values[0] if param_spec.values else value_str
        else:
            return value_str
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RL optimizer statistics."""
        return {
            "episodes_recorded": len(self.history),
            "ready_for_training": self.is_ready_for_training(),
            "policy_version": self.policy_version,
            "policy_active": self.policy is not None,
            "meta_patterns_count": len(self.meta_patterns),
            "avg_score": statistics.mean([ep["score"] for ep in self.history]) if self.history else 0.0,
            "best_score": max([ep["score"] for ep in self.history]) if self.history else 0.0
        }
    
    def suggest_next_parameters(self) -> List[Dict[str, Any]]:
        """
        Suggest promising parameter configurations based on learned policy.
        
        Uses policy to generate configs likely to perform well.
        
        Returns:
            List of suggested configurations
        """
        if not self.policy:
            return []
        
        suggestions = []
        preferences = self.policy.get("param_preferences", {})
        
        # Suggestion 1: All best values
        best_config = {}
        for param_name, param_spec in self.search_space.parameters.items():
            if param_name in preferences and preferences[param_name]:
                best_value_str = max(
                    preferences[param_name].items(),
                    key=lambda x: x[1]
                )[0]
                best_config[param_name] = self._convert_value(best_value_str, param_spec)
        
        if best_config:
            suggestions.append(best_config)
        
        # Could generate more sophisticated suggestions here
        # (e.g., variations on best config, exploring promising regions)
        
        return suggestions

