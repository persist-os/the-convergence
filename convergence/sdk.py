"""
Convergence SDK - Programmatic interface for optimization runs.

Provides a clean, type-safe API for backend services to run Convergence optimizations
without YAML configuration files.
"""

import asyncio
import importlib
from typing import Dict, Any, Optional, Iterable, Iterator, Callable, Literal, Union, List
from datetime import datetime
from pathlib import Path

from convergence.types import (
    ConvergenceConfig,
    AdaptersConfig,
    OptimizationRunResult,
    Evaluator,
)
from convergence.optimization.runner import OptimizationRunner
from convergence.optimization.models import OptimizationSchema


def resolve_callable(value: Optional[Union[Callable, str]]) -> Optional[Callable]:
    """
    Resolve a callable from direct function or dotted path string.
    
    Args:
        value: Callable function or dotted path string (e.g., "module.function")
        
    Returns:
        Resolved callable or None
    """
    if value is None:
        return None
    
    if callable(value):
        return value
    
    if isinstance(value, str):
        try:
            module_path, func_name = value.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except Exception as e:
            raise ValueError(f"Failed to import callable from '{value}': {e}")
    
    raise TypeError(f"Expected callable or string, got {type(value)}")


class TestCase:
    """Test case structure."""
    
    def __init__(self, input: Dict, expected: Dict, meta: Optional[Dict] = None):
        self.input = input
        self.expected = expected
        self.meta = meta or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input": self.input,
            "expected": self.expected,
            "meta": self.meta
        }


def normalize_test_cases(
    test_cases: Optional[Union[Iterable[Dict], Iterator[Dict], Callable[[], Iterator[Dict]]]]
) -> Iterator[TestCase]:
    """
    Normalize test cases into consistent TestCase iterator.
    
    Args:
        test_cases: List, iterator, or callable returning iterator of test cases
        
    Yields:
        TestCase instances
    """
    if test_cases is None:
        return
    
    # Handle callable that returns iterator
    if callable(test_cases):
        test_cases = test_cases()
    
    # Handle iterable
    for case in test_cases:
        if isinstance(case, dict):
            yield TestCase(
                input=case.get("input", {}),
                expected=case.get("expected", {}),
                meta=case.get("meta", {})
            )
        elif isinstance(case, TestCase):
            yield case
        else:
            raise ValueError(f"Invalid test case format: {type(case)}")


def _convert_to_optimization_schema(config: ConvergenceConfig, test_cases: Optional[List[Dict]] = None) -> OptimizationSchema:
    """
    Convert ConvergenceConfig to internal OptimizationSchema.
    
    This is a temporary bridge until OptimizationRunner accepts ConvergenceConfig directly.
    
    Args:
        config: ConvergenceConfig to convert
        test_cases: Optional inline test cases to include in config
    """
    # Convert search space parameters
    search_space_params = {}
    for param_name, param_def in config.search_space.parameters.items():
        if param_def.get("type") == "categorical":
            choices_value = param_def.get("choices")
            search_space_params[param_name] = {
                "type": "categorical",
                "values": choices_value
            }
        elif param_def.get("type") == "float":
            search_space_params[param_name] = {
                "type": "continuous",
                "min": param_def.get("min"),
                "max": param_def.get("max"),
                "step": param_def.get("step", 0.1)
            }
        elif param_def.get("type") == "int":
            search_space_params[param_name] = {
                "type": "discrete",
                "min": param_def.get("min"),
                "max": param_def.get("max"),
                "step": param_def.get("step", 1)
            }
        else:
            search_space_params[param_name] = param_def
    
    # Convert evaluation metrics
    metrics = {}
    for metric_name in config.evaluation.required_metrics:
        weight = config.evaluation.weights.get(metric_name, 1.0)
        threshold = config.evaluation.thresholds.get(metric_name) if config.evaluation.thresholds else None
        
        metric_config = {
            "weight": weight,
            "type": "higher_is_better"
        }
        if threshold is not None:
            metric_config["threshold"] = threshold
        
        metrics[metric_name] = metric_config
    
    # Build OptimizationSchema
    from convergence.optimization.models import (
        APIConfig as InternalApiConfig,
        AuthConfig,
        RequestConfig,
        ResponseConfig,
        SearchSpaceConfig as InternalSearchSpaceConfig,
        SearchSpaceParameter,
        EvaluationConfig as InternalEvaluationConfig,
        TestCasesConfig,
        MetricConfig,
        OptimizationAlgorithmConfig,
        EvolutionConfig,
        ExecutionConfig,
        EarlyStoppingConfig,
        MABConfig,
        OutputConfig,
        LegacyTrackingConfig
    )
    
    # Build auth config
    auth_config_data = {}
    if config.api.auth_type:
        auth_config_data["type"] = config.api.auth_type
    if config.api.auth_token_env:
        auth_config_data["token_env"] = config.api.auth_token_env
    if config.api.auth_header_name:
        auth_config_data["header_name"] = config.api.auth_header_name
    
    internal_auth = AuthConfig(**auth_config_data) if auth_config_data else AuthConfig()
    
    # Enable adapter for Agno agents
    adapter_enabled = "agno" in config.api.name.lower() or config.agent is not None
    
    internal_api = InternalApiConfig(
        name=config.api.name,
        endpoint=config.api.endpoint,
        auth=internal_auth,
        request=RequestConfig(timeout_seconds=int(config.api.request_timeout or 30)),
        response=ResponseConfig(),
        adapter_enabled=adapter_enabled,
        mock_mode=False
    )
    
    internal_search_space = InternalSearchSpaceConfig(
        parameters={k: SearchSpaceParameter(**v) for k, v in search_space_params.items()}
    )
    
    # Create test cases config if provided
    test_cases_config = None
    if test_cases:
        test_cases_config = TestCasesConfig(inline=test_cases)
    
    internal_evaluation = InternalEvaluationConfig(
        test_cases=test_cases_config,
        metrics={k: MetricConfig(**v) for k, v in metrics.items()}
    )
    
    # Build early stopping config
    early_stopping_config = None
    if config.runner.early_stopping:
        early_stopping_config = EarlyStoppingConfig(**config.runner.early_stopping)
    
    optimization_config = OptimizationAlgorithmConfig(
        algorithm="mab_evolution",
        mab=MABConfig(),
        evolution=EvolutionConfig(
            population_size=config.runner.population,
            generations=config.runner.generations
        ),
        execution=ExecutionConfig(early_stopping=early_stopping_config) if early_stopping_config else ExecutionConfig()
    )
    
    output_config = OutputConfig()
    legacy_config = LegacyTrackingConfig(enabled=False)
    
    # Convert agent config if provided
    agent_config = None
    if config.agent:
        from convergence.optimization.models import DiscordAuthConfig, ModelConfig
        agent_config_data = {"models": {}}
        
        # Convert models
        for model_key, model_data in config.agent.models.items():
            agent_config_data["models"][model_key] = ModelConfig(**model_data)
        
        # Convert discord_auth if present
        if config.agent.discord_auth:
            agent_config_data["discord_auth"] = DiscordAuthConfig(**config.agent.discord_auth)
        
        from convergence.optimization.models import AgentConfig as InternalAgentConfig
        agent_config = InternalAgentConfig(**agent_config_data)
    
    return OptimizationSchema(
        api=internal_api,
        search_space=internal_search_space,
        evaluation=internal_evaluation,
        optimization=optimization_config,
        output=output_config,
        agent=agent_config,
        legacy=legacy_config
    )


async def run_optimization(
    config: ConvergenceConfig,
    *,
    evaluator: Optional[Union[Callable, str]] = None,
    test_cases: Optional[Union[Iterable[Dict], Iterator[Dict], Callable[[], Iterator[Dict]]]] = None,
    adapters: Optional[AdaptersConfig] = None,
    local_function: Optional[Union[Callable, str]] = None,
    logging_mode: Literal["silent", "summary", "verbose"] = "silent",
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> OptimizationRunResult:
    """
    Run Convergence optimization programmatically.
    
    Args:
        config: Convergence configuration object
        evaluator: Optional callable or dotted path string for evaluation
        test_cases: Optional iterable/iterator/callable returning test cases
        adapters: Optional adapters configuration
        local_function: Optional callable or dotted path string for local function optimization (API name must start with "local_")
        logging_mode: Logging verbosity ("silent", "summary", "verbose")
        on_event: Optional callback for real-time events
        
    Returns:
        OptimizationRunResult with best config, scores, and metadata
        
    Example:
        >>> from convergence.types import ConvergenceConfig, ApiConfig, SearchSpaceConfig
        >>> config = ConvergenceConfig(
        ...     api=ApiConfig(name="test", endpoint="http://localhost:8000/test"),
        ...     search_space=SearchSpaceConfig(parameters={...}),
        ...     runner=RunnerConfig(generations=10, population=20),
        ...     evaluation=EvaluationConfig(required_metrics=["score"], weights={"score": 1.0})
        ... )
        >>> result = await run_optimization(
        ...     config=config,
        ...     evaluator=my_evaluator_function,
        ...     test_cases=[{"input": {...}, "expected": {...}}],
        ...     logging_mode="summary"
        ... )
    """
    # Resolve callables
    evaluator_fn = resolve_callable(evaluator)
    local_function_fn = resolve_callable(local_function)
    
    if adapters:
        input_adapter = resolve_callable(adapters.input_adapter)
        output_adapter = resolve_callable(adapters.output_adapter)
        case_adapter = resolve_callable(adapters.case_adapter)
        batch_adapter = resolve_callable(adapters.batch_adapter)
    else:
        input_adapter = None
        output_adapter = None
        case_adapter = None
        batch_adapter = None
    
    # Normalize test cases to list
    test_cases_list = None
    if test_cases:
        test_cases_list = [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in normalize_test_cases(test_cases)]
    
    # Convert to internal schema (temporary bridge)
    internal_config = _convert_to_optimization_schema(config, test_cases=test_cases_list)
    
    # Attach local function to config for adapter to use
    if local_function_fn:
        internal_config.local_function = local_function_fn
    elif local_function and isinstance(local_function, str):
        # If it's a string path, store it for adapter to resolve
        internal_config.local_function_path = local_function
    
    # Create runner with programmatic evaluator
    runner = OptimizationRunner(
        internal_config, 
        config_file_path=None,
        custom_evaluator_callable=evaluator_fn
    )
    
    # Run optimization
    # TODO: Pass adapters, test_cases to runner when it supports them
    result = await runner.run()
    
    # Generate run ID
    import time
    import uuid
    optimization_run_id = f"{config.api.name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Build result
    return OptimizationRunResult(
        success=True,
        best_config=result.best_config,
        best_score=result.best_score,
        configs_generated=len(result.all_results),
        generations_run=result.generations_run,
        optimization_run_id=optimization_run_id,
        timestamp=result.timestamp,
        events=[],
        error=None
    )


def run_optimization_sync(
    config: ConvergenceConfig,
    **kwargs
) -> OptimizationRunResult:
    """
    Synchronous wrapper for run_optimization.
    
    Safely wraps async run_optimization in event loop if needed.
    
    Args:
        config: Convergence configuration object
        **kwargs: Passed to run_optimization
        
    Returns:
        OptimizationRunResult
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, must call from async context
            raise RuntimeError(
                "run_optimization_sync called from running event loop. "
                "Use await run_optimization() instead."
            )
        else:
            return loop.run_until_complete(run_optimization(config, **kwargs))
    except RuntimeError as e:
        if "no current event loop" in str(e).lower():
            # No event loop, create one
            return asyncio.run(run_optimization(config, **kwargs))
        else:
            raise


__all__ = [
    "run_optimization",
    "run_optimization_sync",
    "resolve_callable",
    "normalize_test_cases",
    "TestCase",
]
