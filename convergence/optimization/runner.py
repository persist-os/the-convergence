"""
Main optimization runner - orchestrates the entire optimization process.

This is the core engine that coordinates MAB, Evolution, RL, and evaluation
to find optimal API configurations.
"""
import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from functools import partial

from .models import OptimizationSchema, APIResponse, OptimizationResult
from .config_loader import ConfigLoader
from .api_caller import APICaller
from .evaluator import Evaluator
from .config_validator import ConfigValidator
from .evolution import EvolutionEngine, ConfigurationAnalyzer
from .test_case_evolution import TestCaseEvolutionEngine, TestCaseAnalyzer
from .rl_optimizer import RLMetaOptimizer
from .response_utils import extract_response_text
from ..plugins.mab.thompson_sampling import ThompsonSamplingStrategy
from ..storage import get_storage_registry
from rich.console import Console

console = Console()

# API Adapters
from .adapters.browserbase import BrowserBaseAdapter

# API adapters for provider-specific request/response transformations
try:
    from .adapters.azure import AzureOpenAIAdapter
    from .adapters.gemini import GeminiAdapter
    from .adapters.browserbase import BrowserBaseAdapter
    # Universal agent adapter - handles ALL Agno agents (Discord, Gmail, Reddit, future agents)
    from .adapters.agno_agent import UniversalAgentAdapter
    # Local function adapter - for optimizing internal Python functions
    from .adapters.local_function import LocalFunctionAdapter
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False
    AzureOpenAIAdapter = None
    GeminiAdapter = None
    BrowserBaseAdapter = None
    UniversalAgentAdapter = None
    LocalFunctionAdapter = None

# Weave integration for observability
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None


class OptimizationRunner:
    """
    Main optimization orchestration engine.
    
    Coordinates:
    - MAB exploration (Thompson Sampling)
    - Evolution refinement (mutation + crossover)
    - RL meta-optimization (hierarchical learning)
    - Parallel execution (multiprocessing)
    - Progress tracking and early stopping
    """
    
    def __init__(
        self, 
        config: OptimizationSchema, 
        config_file_path: Optional[Path] = None,
        custom_evaluator_callable: Optional[Callable] = None
    ):
        """
        Initialize optimization runner.
        
        Args:
            config: Complete optimization configuration
            config_file_path: Path to the config file (for loading local evaluators)
            custom_evaluator_callable: Optional callable evaluator (SDK programmatic mode)
        """
        self.config = config
        self.config_file_path = config_file_path
        
        # Validate configuration early
        try:
            ConfigValidator.validate_and_suggest_fixes(
                config.dict(), 
                str(config_file_path) if config_file_path else None
            )
        except Exception as e:
            console.print(f"[red]❌ Configuration validation failed:[/red]")
            console.print(f"[red]{e}[/red]")
            raise
        
        # Initialize components
        self.api_caller = APICaller(timeout=config.api.request.timeout_seconds)
        self.evaluator = Evaluator(
            config.evaluation, 
            config_file_path=config_file_path,
            custom_evaluator_callable=custom_evaluator_callable
        )
        
        # Initialize API adapter based on provider
        # OpenAI-compatible providers (OpenAI, Groq, Anthropic) use default behavior
        # Only providers that deviate from OpenAI format need adapters
        self.adapter = self._detect_adapter(config.api.name, config_file_path)
        if self.adapter:
            adapter_name = self.adapter.__class__.__name__
            console.print(f"   ✨ Using {adapter_name} for request/response transformation")
        
        # MAB: Thompson Sampling for config selection
        self.mab = ThompsonSamplingStrategy()
        
        # Evolution: Genetic algorithm for refinement
        self.evolution = EvolutionEngine(
            search_space=config.search_space,
            mutation_rate=config.optimization.evolution.mutation_rate,
            crossover_rate=config.optimization.evolution.crossover_rate,
            elite_size=config.optimization.evolution.elite_size
        )
        
        # RL: Meta-optimizer (optional, for hierarchical learning)
        self.rl_optimizer = None
        if config.society and config.society.enabled:
            self.rl_optimizer = RLMetaOptimizer(
                search_space=config.search_space,
                min_episodes_for_training=15  # Lower threshold for faster learning
            )
        
        # Agent society components (RLP + SAO)
        self.rlp_agent = None
        self.sao_agent = None
        self.llm_provider = None
        
        if config.society and config.society.enabled:
            # Initialize LLM provider for reasoning (config-driven via env vars)
            from ..core.llm_provider import get_llm_provider
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                self.llm_provider = get_llm_provider()
                logger.info(f"[AGENT SOCIETY] LLM Provider initialized: {self.llm_provider.config.model}")
            except Exception as e:
                logger.warning(f"[AGENT SOCIETY] Failed to initialize LLM provider: {e}")
                logger.warning("[AGENT SOCIETY] Falling back to heuristic reasoning")
                self.llm_provider = None
            
            if config.society.learning.rlp_enabled:
                from ..plugins.learning.rlp import RLPMixin, RLPConfig
                self.rlp_agent = RLPMixin(
                    config=RLPConfig(),
                    llm_provider=self.llm_provider
                )
                logger.info("[AGENT SOCIETY] RLP (Reinforcement Learning on Policy) enabled")
            
            if config.society.learning.sao_enabled:
                from ..plugins.learning.sao import SAOMixin, SAOConfig
                self.sao_agent = SAOMixin(
                    config=SAOConfig(),
                    llm_provider=self.llm_provider
                )
                logger.info("[AGENT SOCIETY] SAO (Self-Alignment Optimization) enabled")
        
        # Legacy API Adapter code - now handled by _detect_adapter() above
        # Keeping this for reference but it's no longer used
        self.api_adapter = self.adapter  # Point to the same adapter for backwards compatibility
        
        # Storage
        self._init_storage()
        
        # Legacy tracking (enabled by default)
        self.legacy_store = None
        self.legacy_session = None
        if config.legacy.enabled:
            self._init_legacy()
        
        # Test cases
        self.test_cases = self._load_test_cases()
        
        # Tracking
        self.history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        self.generation_best_scores: List[float] = []
        
        # Early stopping
        self.patience = config.optimization.execution.early_stopping.patience
        self.min_improvement = config.optimization.execution.early_stopping.min_improvement
        self.no_improvement_count = 0
    
    def _init_storage(self) -> None:
        """Initialize storage backend."""
        society = self.config.society
        if society and society.enabled:
            backend = society.storage.backend
            path = society.storage.path
        else:
            # Default: multi-backend (SQLite + File)
            backend = "multi"
            path = "./data/optimization"
        
        registry = get_storage_registry()
        
        # Pass appropriate parameters based on backend type
        if backend == "convex":
            # Convex storage - no file paths needed, uses auto-import
            self.storage = registry.get(backend)
        elif backend in ["multi", "sqlite", "file"]:
            # File-based storage - needs paths
            self.storage = registry.get(
                backend,
                sqlite_path=f"{path}/optimization.db",
                file_base_dir=f"{path}/files"
            )
        else:
            # Generic - let registry handle it
            self.storage = registry.get(backend)
    
    def _detect_adapter(self, api_name: str, config_file_path: Optional[Path] = None):
        """Detect and return appropriate API adapter based on API name."""
        if not ADAPTERS_AVAILABLE:
            return None
        
        api_name_lower = api_name.lower()
        
        # Local function adapter - for internal function optimization
        if api_name_lower.startswith("local_"):
            if LocalFunctionAdapter:
                # Extract function from config metadata or name
                # Config should have func or func_path in metadata
                func = getattr(self.config, 'local_function', None)
                func_path = getattr(self.config, 'local_function_path', None)
                
                if func or func_path:
                    return LocalFunctionAdapter(func=func, func_path=func_path)
                else:
                    console.print("[yellow]⚠️  LocalFunctionAdapter needs func or func_path[/yellow]")
                    return None
            else:
                console.print("[yellow]⚠️  LocalFunctionAdapter not available[/yellow]")
                return None
        
        # Universal agent adapter - handles ALL Agno agents (Discord, Gmail, Reddit, future agents)
        if "agno" in api_name_lower:
            if UniversalAgentAdapter:
                return UniversalAgentAdapter(self.config.dict(), config_file_path)
            else:
                console.print("[yellow]⚠️  UniversalAgentAdapter not available[/yellow]")
                return None
        
        # Standard API adapters
        if "azure" in api_name_lower:
            return AzureOpenAIAdapter(self.config.dict())
        elif "browserbase" in api_name_lower:
            return BrowserBaseAdapter()
        else:
            return None
    
    def _init_legacy(self) -> None:
        """Initialize legacy tracking system."""
        from convergence.legacy import LegacyStore, LegacyConfig
        
        legacy_cfg = self.config.legacy
        
        # Create legacy config
        legacy_config = LegacyConfig(
            enabled=True,
            session_id=legacy_cfg.session_id,
            tracking_backend=legacy_cfg.tracking_backend,
            sqlite_path=legacy_cfg.sqlite_path,
            export_dir=legacy_cfg.export_dir,
            export_formats=legacy_cfg.export_formats,
            mlflow_config=legacy_cfg.mlflow_config,
            aim_config=legacy_cfg.aim_config,
            weave_config=legacy_cfg.weave_config
        )
        
        # Initialize store
        self.legacy_store = LegacyStore(legacy_config)
        
        print(f"📚 Legacy tracking enabled")
        print(f"   Backend: {legacy_cfg.tracking_backend}")
        print(f"   Database: {legacy_cfg.sqlite_path}")
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from file or inline config, with optional augmentation."""
        test_cases_config = self.config.evaluation.test_cases
        
        if not test_cases_config:
            raise ValueError("No test cases configured")
        
        # Load base test cases
        base_test_cases = []
        
        # Load from file
        if test_cases_config.path:
            path = Path(test_cases_config.path)
            
            # If path is relative, resolve it relative to the config file's location
            if not path.is_absolute() and self.config_file_path:
                path = self.config_file_path.parent / path
            
            if not path.exists():
                raise FileNotFoundError(f"Test cases file not found: {path}")
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # Direct list of test cases
                base_test_cases = data
            elif isinstance(data, dict):
                # Check for "test_cases" key
                if "test_cases" in data:
                    base_test_cases = data["test_cases"]
                else:
                    # Single test case as dict
                    base_test_cases = [data]
            else:
                raise ValueError(f"Invalid test cases format in {path}")
        
        # Use inline test cases
        elif test_cases_config.inline:
            base_test_cases = test_cases_config.inline
        
        else:
            raise ValueError("Test cases must be provided via path or inline")
        
        # Apply test case augmentation if enabled
        if test_cases_config.augmentation and test_cases_config.augmentation.enabled:
            console.print(f"\n🧬 [bold cyan]Test Case Augmentation Enabled[/bold cyan]")
            console.print(f"   Original test cases: {len(base_test_cases)}")
            
            # Create evolution engine
            augmentation_engine = TestCaseEvolutionEngine(
                mutation_rate=test_cases_config.augmentation.mutation_rate,
                crossover_rate=test_cases_config.augmentation.crossover_rate,
                augmentation_factor=test_cases_config.augmentation.augmentation_factor,
                preserve_originals=test_cases_config.augmentation.preserve_originals
            )
            
            # Generate augmented test cases
            augmented_test_cases = augmentation_engine.augment_test_cases(base_test_cases)
            
            console.print(f"   Augmented test cases: {len(augmented_test_cases)}")
            console.print(f"   Mutation rate: {test_cases_config.augmentation.mutation_rate}")
            console.print(f"   Crossover rate: {test_cases_config.augmentation.crossover_rate}")
            console.print(f"   Variants per test: {test_cases_config.augmentation.augmentation_factor}")
            
            # Analyze diversity
            diversity_metrics = TestCaseAnalyzer.analyze_diversity(augmented_test_cases)
            console.print(f"   Diversity score: {diversity_metrics['diversity']:.2f}")
            console.print(f"   Unique categories: {diversity_metrics['unique_categories']}")
            console.print(f"   Unique difficulties: {diversity_metrics['unique_difficulties']}\n")
            
            return augmented_test_cases
        
        return base_test_cases
    
    async def _load_warm_start_configs(self) -> List[Dict[str, Any]]:
        """
        Load proven configs from previous runs for warm-start.
        
        Returns:
            List of winning configs from legacy database
        """
        if not self.legacy_store or not self.config.legacy.enabled:
            return []
        
        try:
            # Get session info first to use correct session_id
            from convergence.legacy.models import Session
            import hashlib
            
            # Calculate same fingerprint as in _record_to_legacy
            config_str = json.dumps({
                "search_space": self.config.search_space.dict(),
                "evaluation": {
                    "metrics": list(self.config.evaluation.metrics.keys()),
                    "test_cases": len(self.test_cases),
                    "custom_evaluator": self.config.evaluation.custom_evaluator.dict() if self.config.evaluation.custom_evaluator else None
                }
            }, sort_keys=True)
            config_fingerprint = hashlib.md5(config_str.encode()).hexdigest()[:12]
            
            # Generate session_id same way as recording
            if self.config.legacy.session_id:
                session_id = f"{self.config.legacy.session_id}_{config_fingerprint}"
            else:
                session_id = f"{self.config.api.name}_{config_fingerprint}"
            
            # Connect and query winners
            async with self.legacy_store:
                winners = await self.legacy_store.get_top_winners(
                    session_id=session_id,
                    limit=10  # Get top 10 winners
                )
            
            if winners:
                print(f"\n🔄 WARM-START: Loaded {len(winners)} proven configs from previous optimization runs")
                print(f"   └─ Note: Scores shown are each config's BEST performance on a specific test")
                print(f"   └─ Overall scores (averaged across all tests) will be calculated below\n")
                for i, winner in enumerate(winners[:3], 1):
                    print(f"   Config #{i}:")
                    print(f"      • Best at test '{winner.test_case_id}': {winner.best_score:.4f}")
                    print(f"      • Settings: {winner.best_config}")
                
                return [w.best_config for w in winners]
            else:
                return []
        
        except Exception as e:
            print(f"⚠️ Failed to load warm-start configs: {e}")
            return []
    
    async def run(self) -> OptimizationResult:
        """
        Run complete optimization process.
        
        Returns:
            OptimizationResult with best config and all results
        """
        print("🚀 STARTING API OPTIMIZATION")
        print("=" * 70)
        print(f"API: {self.config.api.name}")
        if self.config.api.endpoint:
            print(f"Endpoint: {self.config.api.endpoint}")
        elif self.config.api.models:
            model_names = list(self.config.api.models.keys())
            print(f"Models: {', '.join(model_names)} ({len(model_names)} models)")
        print(f"Search Space: {len(self.config.search_space.parameters)} parameters")
        print(f"Test Cases: {len(self.test_cases)}")
        print(f"Algorithm: {self.config.optimization.algorithm}")
        print(f"Generations: {self.config.optimization.evolution.generations}")
        print(f"Population Size: {self.config.optimization.evolution.population_size}")
        print("=" * 70)
        
        # Load warm-start configs from previous runs (if legacy enabled)
        warm_configs = await self._load_warm_start_configs()
        
        # Initialize population: mix warm-start (proven) + random (exploration)
        pop_size = self.config.optimization.evolution.population_size
        num_warm = min(len(warm_configs), pop_size // 2)  # Max 50% from warm-start
        num_random = pop_size - num_warm
        
        if num_warm > 0:
            print(f"\n📊 POPULATION COMPOSITION:")
            print(f"   ├─ {num_warm} configs reused from previous runs (🏆 proven performers)")
            print(f"   └─ {num_random} new configs from random exploration (🎲 for diversity)\n")
            print(f"   💡 Tip: Mixing proven + new configs helps find better solutions faster!")
            population = warm_configs[:num_warm] + self.evolution.create_initial_population(size=num_random)
        else:
            print(f"\n📊 POPULATION: All {pop_size} configs are randomly generated")
            print(f"   └─ No previous data found - starting fresh exploration\n")
            population = self.evolution.create_initial_population(size=pop_size)
        
        # Main optimization loop
        for generation in range(self.config.optimization.evolution.generations):
            print(f"\n{'━' * 70}")
            print(f"🧬 GENERATION {generation + 1}/{self.config.optimization.evolution.generations}")
            print(f"{'━' * 70}")
            
            # Initialize variables for this generation
            thought = None
            
            # Apply RLP reasoning if enabled (think before selecting configs)
            if self.rlp_agent:
                print("🧠 RLP: Internal reasoning active...")
                # Use RLP to reason about promising parameter regions
                # This helps guide the search more intelligently
                state = {
                    "generation": generation,
                    "best_score": self.best_score,
                    "history_size": len(self.history),
                    "goal": f"Find optimal configuration for {self.config.api.name}",
                    "constraints": f"Maximize score (current best: {self.best_score:.4f})"
                }
                
                # Generate reasoning using LLM (configured via env vars)
                context = f"Optimizing {self.config.api.name} API. Current best score: {self.best_score:.4f}"
                thought_result = await self.rlp_agent.generate_internal_reasoning(
                    state=state,
                    context=context
                )
                
                # Extract thought (handle both tuple and string returns)
                if isinstance(thought_result, tuple):
                    thought, logprobs = thought_result
                else:
                    thought = thought_result
                
                if thought:
                    # Show full thought for first 3 generations, then abbreviated
                    if generation < 3:
                        print(f"   💭 Agent reasoning:\n      {thought}")
                    elif generation % 5 == 0:
                        print(f"   💭 Agent thought: {thought[:100]}...")
            
            # Apply RL bias if available
            if self.rl_optimizer and self.rl_optimizer.policy:
                print("🎯 Applying RL meta-policy bias...")
                population = [
                    self.rl_optimizer.bias_config_sampling(config)
                    for config in population
                ]
            
            # Evaluate generation
            scores, generation_history = await self._evaluate_generation(
                population,
                generation
            )
            
            # Track history
            self.history.extend(generation_history)
            
            # RLP Training: Update policy based on results
            if self.rlp_agent and thought:
                print("🧠 RLP: Training policy from generation results...")
                
                # Calculate information gain reward for the reasoning
                # Reward = normalized improvement in best score from reasoning
                previous_best = self.best_score
                current_best = max(scores) if scores else previous_best
                
                # Use normalized improvement to avoid zero rewards
                if previous_best > 0:
                    information_gain = (current_best - previous_best) / previous_best
                else:
                    information_gain = current_best  # First generation gets raw score
                
                # Update RLP policy with reward
                updated_state = self.rlp_agent.update_rlp_policy(
                    thought=thought,
                    reward=information_gain,
                    state=state,
                    action=f"generation_{generation}_reasoning",
                    next_state={
                        "generation": generation + 1,
                        "best_score": current_best,
                        "history_size": len(self.history) + len(generation_history)
                    },
                    done=(generation == self.config.optimization.evolution.generations - 1)
                )
                
                # Log detailed RLP metrics to W&B
                if WEAVE_AVAILABLE and weave:
                    @weave.op()
                    def log_rlp_training(
                        generation: int,
                        thought: str,
                        reward: float,
                        normalized_reward: float,
                        rlp_stats: Dict[str, Any],
                        best_score: float,
                        config_count: int,
                        information_gain: float
                    ):
                        return {
                            "generation": generation,
                            "thought_length": len(thought),
                            "thought_preview": thought[:100],
                            "raw_reward": reward,
                            "normalized_reward": normalized_reward,
                            "information_gain": information_gain,
                            "mean_reward": rlp_stats.get('mean_reward', 0),
                            "reward_trend": rlp_stats.get('reward_trend', 'stable'),
                            "total_episodes": rlp_stats.get('total_episodes', 0),
                            "buffer_size": rlp_stats.get('buffer_size', 0),
                            "best_score": best_score,
                            "configs_tested": config_count,
                            "rlp_active": True
                        }
                    
                    # Call the Weave-tracked function
                    log_rlp_training(
                        generation=generation,
                        thought=thought,
                        reward=information_gain,
                        normalized_reward=updated_state.get('rlp_stats', {}).get('mean_normalized_reward', 0),
                        rlp_stats=updated_state.get('rlp_stats', {}),
                        best_score=current_best,
                        config_count=len(population),
                        information_gain=information_gain
                    )
                
                # Log learning metrics
                if 'rlp_stats' in updated_state:
                    stats = updated_state['rlp_stats']
                    print(f"   📊 RLP Stats: Mean reward={stats['mean_reward']:.4f}, Episodes={stats['total_episodes']}")
            
            # Update best
            gen_best_score = max(scores)
            gen_best_config = population[scores.index(gen_best_score)]
            
            if gen_best_score > self.best_score:
                improvement = gen_best_score - self.best_score
                self.best_score = gen_best_score
                self.best_config = gen_best_config
                self.no_improvement_count = 0
                print(f"✨ NEW BEST! Score: {self.best_score:.4f} (+{improvement:.4f})")
            else:
                self.no_improvement_count += 1
                print(f"📊 Best: {self.best_score:.4f} (no improvement for {self.no_improvement_count} gen)")
            
            self.generation_best_scores.append(gen_best_score)
            
            # Record RL episodes
            if self.rl_optimizer:
                for config, score, history_entry in zip(population, scores, generation_history):
                    self.rl_optimizer.record_episode(
                        config=config,
                        score=score,
                        metrics=history_entry["metrics"],
                        generation=generation
                    )
            
            # Train RL policy if ready
            if self.rl_optimizer and not self.rl_optimizer.policy:
                if self.rl_optimizer.is_ready_for_training():
                    print("\n" + "=" * 70)
                    print("🎓 RL TRAINING TRIGGERED - Enough data collected!")
                    print("=" * 70)
                    self.rl_optimizer.train_policy()
            
            # SAO: Generate synthetic training data from optimization history
            if self.sao_agent and len(self.history) >= 15 and generation % 3 == 0:
                print("🔄 SAO: Self-improvement active...")
                
                # Generate synthetic preference pairs from optimization history
                try:
                    # Create preference pairs from successful vs failed configs
                    successful_configs = [h for h in self.history[-20:] if h.get('score', 0) > 0.7]
                    failed_configs = [h for h in self.history[-20:] if h.get('score', 0) < 0.5]
                    
                    if len(successful_configs) >= 2 and len(failed_configs) >= 2:
                        print("   📊 Running iterative SAO refinement...")

                        # Use iterative refinement for better quality data
                        synthetic_data = await self.sao_agent.iterative_sao_refinement(
                            n_samples_per_round=2
                        )
                        
                        if synthetic_data:
                            print(f"   ✅ Generated {len(synthetic_data)} preference pairs")
                            
                            # Store in SAO dataset
                            sao_stats = self.sao_agent.get_generation_stats()
                            print(f"   📈 SAO Dataset: {sao_stats['dataset_size']} total samples")
                            
                            # Log detailed SAO metrics to W&B
                            if WEAVE_AVAILABLE and weave:
                                @weave.op()
                                def log_sao_generation(
                                    generation: int,
                                    synthetic_samples: int,
                                    sao_stats: Dict[str, Any],
                                    optimization_history_size: int,
                                    best_score: float
                                ):
                                    return {
                                        "generation": generation,
                                        "synthetic_samples_generated": synthetic_samples,
                                        "total_dataset_size": sao_stats.get('dataset_size', 0),
                                        "unique_prompts": sao_stats.get('unique_prompts', 0),
                                        "diversity_score": sao_stats.get('diversity_score', 0),
                                        "quality_filtered": sao_stats.get('quality_filtered', 0),
                                        "duplicates_filtered": sao_stats.get('duplicates_filtered', 0),
                                        "rounds_completed": sao_stats.get('rounds_completed', 0),
                                        "optimization_history_size": optimization_history_size,
                                        "best_score": best_score,
                                        "sao_active": True
                                    }
                                
                                # Call the Weave-tracked function
                                log_sao_generation(
                                    generation=generation,
                                    synthetic_samples=len(synthetic_data),
                                    sao_stats=sao_stats,
                                    optimization_history_size=len(self.history),
                                    best_score=self.best_score
                                )
                        else:
                            print("   ⚠️ SAO generation failed")
                    else:
                        print("   ⚠️ Insufficient data for SAO (need more history)")
                        
                except Exception as e:
                    print(f"   ❌ SAO generation error: {e}")
            
            # Evolve population for next generation
            if generation < self.config.optimization.evolution.generations - 1:
                # Pass RLP reasoning to evolution engine for smarter mutations
                population = self.evolution.evolve_population(
                    population, 
                    scores,
                    reasoning=thought if thought else None
                )
                
                # Analyze diversity
                diversity = ConfigurationAnalyzer.analyze_diversity(population)
                print(f"🔬 Population Diversity: {diversity:.2%}")
            
            # Early stopping check (AFTER evolution to allow improvement)
            if self._should_stop_early():
                print(f"\n⚠️ EARLY STOPPING: No improvement for {self.no_improvement_count} generations")
                break
        
        # Create final result
        result = OptimizationResult(
            best_config=self.best_config,
            best_score=self.best_score,
            all_results=self.history,
            generations_run=len(self.generation_best_scores)
        )
        
        # Save to storage
        await self._save_results(result)
        
        # Record to legacy system if enabled
        if self.legacy_store:
            await self._record_to_legacy(result)
        
        print("\n" + "=" * 70)
        print("✅ OPTIMIZATION COMPLETE!")
        print("=" * 70)
        
        # Show optimization results
        print(f"\n📊 RESULTS:")
        print(f"   ├─ Best Score: {self.best_score:.4f} (averaged across all {len(self.test_cases)} test cases)")
        print(f"   ├─ Total Experiments: {len(self.history)}")
        print(f"   └─ Generations: {result.generations_run}\n")
        
        print(f"🏆 BEST CONFIGURATION FOUND:")
        for key, value in self.best_config.items():
            print(f"   • {key}: {value}")
        
        print(f"\n💾 DATA SAVED TO:")
        print(f"   ├─ Latest Results: {self.config.output.save_path}/")
        if self.config.legacy and self.config.legacy.enabled:
            print(f"   └─ Historical Database: {self.config.legacy.sqlite_path}\n")
            print(f"   💡 Next Run: Will load these winning configs as warm-start!")
        else:
            print(f"   └─ (Legacy tracking disabled - no warm-start available)\n")
        
        # Show agent society contributions
        if self.config.society and self.config.society.enabled:
            print(f"\n🤖 AGENT SOCIETY CONTRIBUTIONS:")
            if self.rlp_agent:
                rlp_stats = getattr(self.rlp_agent, 'experience_buffer', None)
                episodes = len(rlp_stats.buffer) if rlp_stats else 0
                print(f"   • RLP (Reasoning): Active - {episodes} experiences learned")
            if self.sao_agent:
                sao_stats = self.sao_agent.get_generation_stats()
                print(f"   • SAO (Self-Improvement): Active - {sao_stats['dataset_size']} synthetic samples")
            if self.rl_optimizer and self.rl_optimizer.policy:
                stats = self.rl_optimizer.get_statistics()
                print(f"   • RL Meta-Policy: Trained (v{stats['policy_version']})")
        
        print("\n" + "=" * 70)
        
        # Cleanup resources before returning
        await self._cleanup()
        
        return result
    
    async def _evaluate_generation(
        self,
        population: List[Dict[str, Any]],
        generation: int
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate all configs in a generation.
        
        Returns:
            (scores, history_entries)
        """
        print(f"📊 Evaluating {len(population)} configurations...")
        
        # Parallel evaluation
        parallel_workers = self.config.optimization.execution.parallel_workers
        
        if parallel_workers > 1:
            # Use multiprocessing pool for true parallelism
            scores, history = await self._evaluate_parallel(
                population,
                generation,
                parallel_workers
            )
        else:
            # Sequential evaluation
            scores, history = await self._evaluate_sequential(
                population,
                generation
            )
        
        return scores, history
    
    async def _evaluate_sequential(
        self,
        population: List[Dict[str, Any]],
        generation: int
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """Evaluate configs sequentially."""
        scores = []
        history = []
        
        for i, config in enumerate(population, 1):

            # Show which config is being evaluated
            config_params = ", ".join([f"{k}={v}" for k, v in config.items()])
            console.print(f"\n🔬 Config [{i}/{len(population)}]: [cyan]{config_params}[/cyan]")
            
            try:
                score, entry = await self._evaluate_single_config(config, generation)
                scores.append(score)
                history.append(entry)
                console.print(f"   ✅ Aggregate Score: [bold green]{score:.4f}[/bold green]\n")
            except Exception as e:
                console.print(f"   [red]❌ ERROR: {e}[/red]\n")
                scores.append(0.0)
                history.append({
                    "config": config,
                    "score": 0.0,
                    "metrics": {},  # Empty metrics for failed evaluations
                    "generation": generation,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return scores, history
    
    async def _evaluate_parallel(
        self,
        population: List[Dict[str, Any]],
        generation: int,
        workers: int
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """Evaluate configs in parallel using multiprocessing."""
        # Use asyncio.gather for concurrent API calls
        tasks = [
            self._evaluate_single_config(config, generation)
            for config in population
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores = []
        history = []
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"  [{i}/{len(population)}] ERROR: {result}")
                scores.append(0.0)
                history.append({
                    "config": population[i-1],
                    "score": 0.0,
                    "metrics": {},  # Empty metrics for failed evaluations
                    "generation": generation,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                score, entry = result
                scores.append(score)
                history.append(entry)
                print(f"  [{i}/{len(population)}] Score: {score:.4f}")
        
        return scores, history
    
    async def _evaluate_single_config(
        self,
        config: Dict[str, Any],
        generation: int
    ) -> tuple[float, Dict[str, Any]]:
        """
        Evaluate a single configuration against all test cases.
        
        Returns:
            (aggregate_score, history_entry)
        """
        test_scores = []
        test_metrics = []
        test_results = []  # Store per-test-case details
        
        # Run on all test cases
        for idx, test_case in enumerate(self.test_cases, 1):
            # Show which test case is being run
            test_id = test_case.get("id", f"test_{idx}")
            test_desc = test_case.get("description", "")
            console.print(f"\n   📝 Test {idx}/{len(self.test_cases)}: {test_id}")
            if test_desc:
                console.print(f"      [dim]{test_desc}[/dim]")
            
            # Show input preview
            test_input = test_case.get("input", {})
            if "input" in test_input:
                input_text = str(test_input["input"])[:80]
                console.print(f"      Input: [dim]{input_text}{'...' if len(str(test_input['input'])) > 80 else ''}[/dim]")
            
            # Make API call
            response = await self._call_api(config, test_case)
            
            # Evaluate response
            metrics, score = await self.evaluator.evaluate_with_aggregate(
                response,
                test_case,
                config
            )
            
            # Show response preview
            if response.success and response.result:
                response_text = extract_response_text(response.result)
                if response_text:
                    preview = response_text[:100].replace('\n', ' ')
                    console.print(f"      Output: [dim]{preview}{'...' if len(response_text) > 100 else ''}[/dim]")
                console.print(f"      Score: [bold cyan]{score:.3f}[/bold cyan] | Latency: [yellow]{response.latency_ms:.0f}ms[/yellow]")
            else:
                console.print(f"      [red]❌ Failed: {response.error}[/red]")
            
            # NOTE: BrowserBase sessions cannot be deleted via API
            # Sessions auto-expire after 5 minutes of inactivity
            # Cleanup code removed as DELETE endpoint doesn't exist
            
            test_scores.append(score)
            test_metrics.append(metrics)
            
            # Extract response text for display
            response_text = extract_response_text(response.result) if response.success else None
            
            # Store detailed per-test-case result
            test_results.append({
                "test_case_id": test_case.get("id", "unknown"),
                "input": test_case.get("input", {}),
                "response_text": response_text,  # Human-readable text
                "response": response.result if response.success else None,  # Full response
                "score": score,
                "metrics": metrics,
                "latency_ms": response.latency_ms,
                "cost_usd": response.estimated_cost_usd,
                "success": response.success,
                "error": response.error
            })
        
        # Average across test cases
        avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
        avg_metrics = self._average_metrics(test_metrics)
        
        # Create history entry with detailed per-test results
        history_entry = {
            "config": config,
            "score": avg_score,
            "metrics": avg_metrics,
            "generation": generation,
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": test_results  # Include detailed per-test data
        }
        
        return avg_score, history_entry
    
    async def _call_api(
        self,
        config: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> APIResponse:
        """Make API call with retry logic."""
        max_retries = self.config.optimization.execution.max_retries
        
        # Universal handling for local function adapters
        is_local_function_adapter = (
            LocalFunctionAdapter and isinstance(self.adapter, LocalFunctionAdapter)
        ) if self.adapter else False
        
        # Universal handling for ALL agent adapters
        is_agent_adapter = (
            UniversalAgentAdapter and isinstance(self.adapter, UniversalAgentAdapter)
        ) if self.adapter else False
        
        if is_local_function_adapter or is_agent_adapter:
            try:
                # Adapter executes function and returns result directly
                result = self.adapter.transform_request(config, test_case)
                
                # Transform response format
                result = self.adapter.transform_response(result)
                
                # Convert to APIResponse format
                latency_ms = result.get('latency_seconds', 0.0) * 1000
                
                return APIResponse(
                    success=result.get('success', True),
                    result=result.get('result'),
                    latency_ms=latency_ms,
                    error=result.get('error')
                )
            except Exception as e:
                error_msg = f"{'Local function' if is_local_function_adapter else 'Agent'} execution failed: {str(e)}"
                console.print(f"   [red]❌ {error_msg}[/red]")
                return APIResponse(
                    success=False,
                    result=None,
                    latency_ms=0.0,
                    error=error_msg
                )
        
        # Standard HTTP API call flow
        # Prepare request payload (merge test case input with config parameters)
        payload = {**test_case.get("input", {}), **config}
        
        # Transform payload using adapter if available
        if self.api_adapter:
            try:
                payload = self.api_adapter.transform_request(config, test_case)
            except Exception as e:
                console.print(f"   [yellow]⚠️  Adapter transform failed: {e}[/yellow]")
                # Continue with original payload if adapter fails
        
        # Get endpoint - use adapter if available, otherwise use hardcoded endpoint
        if self.adapter and hasattr(self.adapter, 'get_endpoint_for_model'):
            # Azure-style: dynamic endpoint selection
            model_key = config.get('model')
            if not model_key:
                raise ValueError("Model key required in config parameters")
            
            endpoint = self.adapter.get_endpoint_for_model(model_key)
            if not endpoint:
                raise ValueError(f"No endpoint found for model '{model_key}' in model registry")
        else:
            # Standard providers: use hardcoded endpoint, model goes in request body
            if not self.config.api.endpoint:
                raise ValueError("No endpoint or model registry configured. Provide either api.endpoint or api.models")
            endpoint = self.config.api.endpoint
        
        # Get auth configuration
        auth_config = {}
        if self.config.api.auth.type == "api_key":
            # Check for per-model API key if using adapter
            if self.adapter and hasattr(self.adapter, 'get_api_key_for_model'):
                token_env = self.adapter.get_api_key_for_model(config.get('model', ''))
                if token_env:
                    auth_config = {
                        "type": "api_key",
                        "token_env": token_env,  # Pass env var name, APICaller will look it up
                        "header_name": getattr(self.config.api.auth, 'header_name', None) or 'x-api-key'
                    }
            
            # Fallback to main auth config
            if not auth_config:
                auth_config = {
                    "type": "api_key",
                    "token_env": self.config.api.auth.token_env,
                    "header_name": getattr(self.config.api.auth, 'header_name', None) or 'x-api-key'
                }
        elif self.config.api.auth.type == "bearer":
            token_env = self.config.api.auth.token_env
            token = os.getenv(token_env) if token_env else None
            if token:
                auth_config = {"type": "bearer", "token_env": token_env, "token_env": token_env}
        
        # Prepare headers
        headers = self.config.api.request.headers.copy()
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                # Make actual API call with dynamic endpoint
                response = await self.api_caller.call(
                    endpoint=endpoint,
                    method=self.config.api.request.method,
                    params=payload,
                    auth=auth_config if auth_config else None,
                    headers=headers,
                    timeout=self.config.api.request.timeout_seconds
                )
                
                # Transform response using adapter if available
                if self.api_adapter:
                    try:
                        response = self.api_adapter.transform_response(response, config)
                    except Exception as e:
                        console.print(f"   [yellow]⚠️  Adapter response transform failed: {e}[/yellow]")
                        # Continue with original response if adapter fails
                
                return response
            
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # All retries failed - FAIL HARD
                    error_msg = f"API call failed after {max_retries} attempts: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    return APIResponse(
                        success=False,
                        result=None,
                        latency_ms=0.0,
                        error=error_msg
                    )
        
        # Should never reach here
        return APIResponse(success=False, result=None, latency_ms=0.0, error="Unknown error")
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across test cases."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0
        
        return avg_metrics
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria met."""
        if not self.config.optimization.execution.early_stopping.enabled:
            return False
        
        # Check if patience exceeded (use > not >= to allow full patience period)
        if self.no_improvement_count > self.patience:
            return True
        
        # Check if recent improvement is too small
        if len(self.generation_best_scores) >= 2:
            recent_improvement = (
                self.generation_best_scores[-1] -
                self.generation_best_scores[-2]
            )
            if abs(recent_improvement) < self.min_improvement:
                return True
        
        return False
    
    async def _save_results(self, result: OptimizationResult) -> None:
        """Save results to storage."""
        try:
            # Save best config
            await self.storage.save(
                f"optimization:best_config:{self.config.api.name}",
                result.best_config
            )
            
            # Save full history
            await self.storage.save(
                f"optimization:history:{self.config.api.name}:{result.timestamp.isoformat()}",
                result.all_results
            )
            
            print("💾 Results saved to storage")
        
        except Exception as e:
            print(f"⚠️ Failed to save to storage: {e}")
    
    async def _record_to_legacy(self, result: OptimizationResult) -> None:
        """Record optimization results to legacy system."""
        from convergence.legacy.models import Session, OptimizationRun, TestCaseResult
        import hashlib
        import uuid
        
        try:
            # Connect to legacy store
            async with self.legacy_store:
                # Create config fingerprint from search space + evaluation config
                config_str = json.dumps({
                    "search_space": self.config.search_space.dict(),
                    "evaluation": {
                        "metrics": list(self.config.evaluation.metrics.keys()),
                        "test_cases": len(self.test_cases),
                        "custom_evaluator": self.config.evaluation.custom_evaluator.dict() if self.config.evaluation.custom_evaluator else None
                    }
                }, sort_keys=True)
                config_fingerprint = hashlib.md5(config_str.encode()).hexdigest()[:12]
                
                # Auto-generate session_id from API name + fingerprint
                # This ensures config changes create new sessions (no overwrites)
                if self.config.legacy.session_id:
                    # Use provided session_id but append fingerprint for versioning
                    session_id = f"{self.config.legacy.session_id}_{config_fingerprint}"
                else:
                    # Auto-generate from API name
                    session_id = f"{self.config.api.name}_{config_fingerprint}"
                
                # Create or get session
                self.legacy_session = await self.legacy_store.create_or_get_session(
                    session_id=session_id,
                    api_name=self.config.api.name,
                    api_endpoint=self.config.api.endpoint or "model-registry",
                    config_fingerprint=config_fingerprint,
                    name=self.config.legacy.session_id or self.config.api.name
                )
                
                print(f"\n💾 SAVING TO LEGACY DATABASE")
                print(f"   ├─ Session: {self.legacy_session.session_id}")
                print(f"   ├─ Database: {self.config.legacy.sqlite_path}")
                print(f"   └─ Status: Accumulating history across all runs\n")
                
                # Get the highest generation number from existing runs in this session
                # This allows generation to increment across multiple optimization runs
                max_generation = await self.legacy_store.get_max_generation(self.legacy_session.session_id)
                base_generation = max_generation + 1 if max_generation is not None else 0
                
                # Get total experiment count for this session
                total_experiments = await self.legacy_store.get_experiment_count(self.legacy_session.session_id)
                
                print(f"📈 CUMULATIVE SESSION STATS:")
                print(f"   ├─ Total experiments so far: {total_experiments}")
                print(f"   ├─ Previous max generation: {max_generation}")
                print(f"   └─ Starting new experiments at generation: {base_generation}\n")
                
                # Record each run in history
                for idx, history_entry in enumerate(result.all_results):
                    run_id = f"run_{uuid.uuid4().hex[:12]}"
                    
                    # Extract test results if available
                    test_results = []
                    test_case_ids = []
                    if "test_results" in history_entry:
                        for test_result in history_entry["test_results"]:
                            result_id = f"result_{uuid.uuid4().hex[:12]}"
                            test_case_id = test_result.get("test_case_id", f"test_{idx}")
                            test_case_ids.append(test_case_id)
                            
                            test_results.append(TestCaseResult(
                                result_id=result_id,
                                run_id=run_id,
                                test_case_id=test_case_id,
                                config=history_entry["config"],
                                score=test_result.get("score", 0.0),
                                metrics=test_result.get("metrics", {}),
                                latency_ms=test_result.get("latency_ms", 0.0),
                                cost_usd=test_result.get("cost_usd", 0.0),
                                response_text=test_result.get("response_text"),
                                full_response=test_result.get("response"),
                                success=test_result.get("success", True),
                                error=test_result.get("error")
                            ))
                    
                    # Create optimization run
                    # Combine base generation (from previous runs) with current run's generation
                    current_generation = history_entry.get("generation", 0)
                    total_generation = base_generation + current_generation
                    
                    run = OptimizationRun(
                        run_id=run_id,
                        session_id=self.legacy_session.session_id,
                        timestamp=datetime.fromisoformat(history_entry.get("timestamp", datetime.utcnow().isoformat())),
                        api_name=self.config.api.name,
                        api_endpoint=self.config.api.endpoint or "model-registry",
                        config=history_entry["config"],
                        test_case_ids=test_case_ids,
                        test_results=test_results,
                        aggregate_score=history_entry["score"],
                        aggregate_metrics=history_entry.get("metrics", {}),
                        duration_ms=0.0,  # Not tracked in current system
                        cost_usd=0.0,  # Not tracked in current system
                        generation=total_generation
                    )
                    
                    # Record run
                    await self.legacy_store.record_run(run)
                
                # Export if configured
                if "winners_only" in self.config.legacy.export_formats:
                    csv_path = await self.legacy_store.export_winners_csv(self.config.api.name)
                    print(f"📄 Exported winners: {csv_path}")
                
                if "full_audit" in self.config.legacy.export_formats:
                    audit_path = await self.legacy_store.export_audit_csv(self.legacy_session.session_id)
                    print(f"📄 Exported audit trail: {audit_path}")
                
                # Show cumulative stats
                new_total = await self.legacy_store.get_experiment_count(self.legacy_session.session_id)
                new_max_gen = await self.legacy_store.get_max_generation(self.legacy_session.session_id)
                print(f"✅ Legacy tracking complete")
                print(f"   📊 Session '{self.legacy_session.session_id}':")
                print(f"      • Total experiments: {new_total}")
                print(f"      • Generations: 0 to {new_max_gen}")
                print(f"      • Database: {self.config.legacy.sqlite_path}")
        
        except Exception as e:
            print(f"⚠️ Failed to record to legacy: {e}")
            import traceback
            traceback.print_exc()
    
    async def _cleanup(self) -> None:
        """Cleanup resources (close HTTP client, storage connections, etc)."""
        try:
            # Close API caller's HTTP client
            if hasattr(self, 'api_caller') and self.api_caller:
                await self.api_caller.close()
            
            # Close storage connections if storage has a close method
            if hasattr(self, 'storage') and self.storage:
                if hasattr(self.storage, 'close'):
                    await self.storage.close()
        
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

