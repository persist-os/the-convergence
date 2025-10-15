"""
Evolution engine for API configuration optimization.

Implements mutation, crossover, and natural selection for finding
optimal API configurations through genetic algorithms.
"""
import random
import copy
from typing import List, Dict, Any, Tuple, Optional

from .models import SearchSpaceParameter, SearchSpaceConfig


class EvolutionEngine:
    """
    Evolutionary optimization for API configurations.
    
    Uses genetic algorithm principles:
    - Mutation: Random parameter changes within search space
    - Crossover: Combine two successful configurations  
    - Selection: Keep top performers (elitism + tournament selection)
    """
    
    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 2
    ):
        """
        Initialize evolution engine.
        
        Args:
            search_space: Search space defining parameters to optimize
            mutation_rate: Probability of mutating each parameter
            crossover_rate: Probability of crossover vs mutation
            elite_size: Number of top configs to preserve unchanged
        """
        self.search_space = search_space
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def create_initial_population(self, size: int) -> List[Dict[str, Any]]:
        """
        Create initial random population of configurations.
        
        Args:
            size: Population size
            
        Returns:
            List of random configurations
        """
        population = []
        for _ in range(size):
            config = self._create_random_config()
            population.append(config)
        return population
    
    def _create_random_config(self) -> Dict[str, Any]:
        """Create a single random configuration."""
        config = {}
        
        for param_name, param_spec in self.search_space.parameters.items():
            config[param_name] = self._sample_parameter(param_spec)
        
        return config
    
    def _sample_parameter(self, param_spec: SearchSpaceParameter) -> Any:
        """Sample a random value from parameter specification."""
        if param_spec.type == "categorical":
            return random.choice(param_spec.values)
        
        elif param_spec.type == "discrete":
            return random.choice(param_spec.values)
        
        elif param_spec.type == "continuous":
            # Sample in range [min, max] with step granularity
            min_val = param_spec.min
            max_val = param_spec.max
            step = param_spec.step if param_spec.step else 0.01
            
            # Generate value within range
            n_steps = int((max_val - min_val) / step)
            random_step = random.randint(0, n_steps)
            value = min_val + (random_step * step)
            
            return round(value, 4)  # Round to avoid floating point errors
        
        else:
            raise ValueError(f"Unknown parameter type: {param_spec.type}")
    
    def evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        reasoning: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evolve population to create next generation.
        
        Args:
            population: Current population configs
            fitness_scores: Fitness score for each config
            reasoning: RLP reasoning to guide evolution (optional)
            
        Returns:
            Next generation population
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Elite: Keep top N unchanged
        next_gen = []
        for i in range(min(self.elite_size, len(sorted_pop))):
            elite_config = copy.deepcopy(sorted_pop[i][0])
            next_gen.append(elite_config)
        
        # Fill rest with evolved configs
        while len(next_gen) < len(population):
            if random.random() < self.crossover_rate:
                # Crossover: Combine two parents
                parent1, parent2 = self._select_parents(sorted_pop)
                child = self.crossover(parent1[0], parent2[0])
            else:
                # Mutation: Mutate a parent (guided by reasoning if available)
                parent = self._select_parent(sorted_pop)
                child = self.mutate(parent[0], reasoning=reasoning)
            
            next_gen.append(child)
        
        return next_gen
    
    def _select_parents(
        self,
        sorted_population: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[Tuple[Dict[str, Any], float], Tuple[Dict[str, Any], float]]:
        """Select two parents using tournament selection."""
        parent1 = self._select_parent(sorted_population)
        parent2 = self._select_parent(sorted_population)
        return parent1, parent2
    
    def _select_parent(
        self,
        sorted_population: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[Dict[str, Any], float]:
        """Select one parent using tournament selection (top 50%)."""
        # Tournament: Pick best from random subset
        tournament_size = max(2, len(sorted_population) // 4)
        tournament = random.sample(sorted_population, tournament_size)
        return max(tournament, key=lambda x: x[1])
    
    def mutate(self, config: Dict[str, Any], reasoning: Optional[str] = None) -> Dict[str, Any]:
        """
        Mutate a configuration.
        
        Randomly changes parameters within search space constraints.
        Can be guided by RLP reasoning for smarter mutations.
        
        Args:
            config: Configuration to mutate
            reasoning: RLP reasoning to guide mutation (optional)
            
        Returns:
            Mutated configuration
        """
        mutated = copy.deepcopy(config)
        
        # Apply reasoning-guided mutation if reasoning is provided
        reasoning_guided = False
        if reasoning:
            reasoning_guided = self._apply_reasoning_guided_mutation(mutated, reasoning)
        
        # Standard random mutation for remaining parameters
        for param_name, param_spec in self.search_space.parameters.items():
            # Mutate each parameter with probability mutation_rate
            if random.random() < self.mutation_rate:
                mutated[param_name] = self._mutate_parameter(
                    config[param_name],
                    param_spec
                )
        
        return mutated
    
    def _apply_reasoning_guided_mutation(self, config: Dict[str, Any], reasoning: str) -> bool:
        """
        Apply reasoning-guided mutations based on RLP insights.
        
        Args:
            config: Configuration to modify
            reasoning: RLP reasoning text
            
        Returns:
            True if any guided mutations were applied
        """
        reasoning_lower = reasoning.lower()
        mutations_applied = False
        
        # Guide mutations based on reasoning content
        if 'chromium' in reasoning_lower and 'browser_type' in config:
            if config['browser_type'] != 'chromium':
                config['browser_type'] = 'chromium'
                mutations_applied = True
        
        if 'headless' in reasoning_lower and 'headless' in config:
            if not config['headless']:
                config['headless'] = True
                mutations_applied = True
        
        if '1920' in reasoning_lower and 'viewport_width' in config:
            if config['viewport_width'] != 1920:
                config['viewport_width'] = 1920
                mutations_applied = True
        
        if '1080' in reasoning_lower and 'viewport_height' in config:
            if config['viewport_height'] != 1080:
                config['viewport_height'] = 1080
                mutations_applied = True
        
        if ('timeout' in reasoning_lower or '5-10' in reasoning_lower) and 'timeout_ms' in config:
            if config['timeout_ms'] not in [5000, 10000]:
                config['timeout_ms'] = random.choice([5000, 10000])
                mutations_applied = True
        
        if 'network_idle' in reasoning_lower and 'wait_strategy' in config:
            if config['wait_strategy'] != 'network_idle':
                config['wait_strategy'] = 'network_idle'
                mutations_applied = True
        
        return mutations_applied
    
    def _mutate_parameter(
        self,
        current_value: Any,
        param_spec: SearchSpaceParameter
    ) -> Any:
        """Mutate a single parameter value."""
        if param_spec.type == "categorical":
            # Pick a different category
            values = [v for v in param_spec.values if v != current_value]
            return random.choice(values) if values else current_value
        
        elif param_spec.type == "discrete":
            # Pick a different discrete value
            values = [v for v in param_spec.values if v != current_value]
            return random.choice(values) if values else current_value
        
        elif param_spec.type == "continuous":
            # Gaussian mutation around current value
            min_val = param_spec.min
            max_val = param_spec.max
            step = param_spec.step if param_spec.step else 0.01
            
            # Add Gaussian noise (σ = 10% of range)
            sigma = (max_val - min_val) * 0.1
            new_value = current_value + random.gauss(0, sigma)
            
            # Clip to bounds
            new_value = max(min_val, min(max_val, new_value))
            
            # Round to step granularity
            new_value = round(new_value / step) * step
            
            return round(new_value, 4)
        
        else:
            return current_value
    
    def crossover(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform crossover between two configurations.
        
        Uses uniform crossover: randomly pick parameters from each parent.
        
        Args:
            config1: First parent configuration
            config2: Second parent configuration
            
        Returns:
            Child configuration
        """
        child = {}
        
        for param_name in self.search_space.parameters.keys():
            # Randomly choose from parent1 or parent2
            if random.random() < 0.5:
                child[param_name] = config1.get(param_name)
            else:
                child[param_name] = config2.get(param_name)
        
        return child
    
    def select_survivors(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        survival_rate: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Select survivors using natural selection.
        
        Keeps top N percent of population based on fitness.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            survival_rate: Fraction to keep (default 0.4 = 40%)
            
        Returns:
            List of surviving configurations
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep top survival_rate percent
        n_survivors = max(1, int(len(sorted_pop) * survival_rate))
        survivors = [config for config, _ in sorted_pop[:n_survivors]]
        
        return survivors


class ConfigurationAnalyzer:
    """Analyze configuration space and optimization progress."""
    
    @staticmethod
    def analyze_diversity(population: List[Dict[str, Any]]) -> float:
        """
        Calculate diversity of population.
        
        Returns value between 0 (all identical) and 1 (maximum diversity).
        """
        if len(population) < 2:
            return 0.0
        
        # Convert configs to hashable format (handle lists/dicts)
        def make_hashable(config: Dict[str, Any]) -> frozenset:
            """Convert config to hashable format, handling lists and dicts."""
            hashable_items = []
            for key, value in config.items():
                if isinstance(value, list):
                    # Convert list to tuple (hashable)
                    hashable_value = tuple(value)
                elif isinstance(value, dict):
                    # Convert dict to frozenset of items
                    hashable_value = frozenset(value.items())
                else:
                    hashable_value = value
                hashable_items.append((key, hashable_value))
            return frozenset(hashable_items)
        
        # Count unique configs
        unique_configs = len({
            make_hashable(config) for config in population
        })
        
        diversity = unique_configs / len(population)
        return diversity
    
    @staticmethod
    def identify_important_parameters(
        configs: List[Dict[str, Any]],
        scores: List[float]
    ) -> Dict[str, float]:
        """
        Identify which parameters correlate most with high scores.
        
        Returns parameter importance scores (0-1).
        """
        if len(configs) < 5:
            return {}
        
        # Simple correlation analysis
        # For each parameter, check if certain values lead to better scores
        importance = {}
        
        for param_name in configs[0].keys():
            # Group by parameter value
            value_scores = {}
            for config, score in zip(configs, scores):
                value = str(config[param_name])  # Convert to string for grouping
                if value not in value_scores:
                    value_scores[value] = []
                value_scores[value].append(score)
            
            # Calculate variance in average scores across values
            avg_scores = [sum(scores) / len(scores) for scores in value_scores.values()]
            if len(avg_scores) > 1:
                variance = sum((s - sum(avg_scores) / len(avg_scores)) ** 2 for s in avg_scores)
                importance[param_name] = min(1.0, variance)
            else:
                importance[param_name] = 0.0
        
        return importance

