"""
Evaluation system for scoring API responses against defined metrics.

Supports built-in evaluation functions and custom user-provided evaluators.
"""
from typing import Dict, Any, Optional, Callable, List
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

from .models import APIResponse, EvaluationConfig, MetricConfig

# Weave integration for observability
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None


class Evaluator:
    """
    Evaluates API responses against test cases and metrics.
    
    Supports:
    - Built-in functions (exact_match, similarity)
    - Custom Python evaluator functions (built-in or local)
    - API-provided evaluation results
    - Weighted metric aggregation
    - Threshold validation
    """
    
    def __init__(self, eval_config: EvaluationConfig, config_file_path: Optional[Path] = None):
        """
        Initialize evaluator.
        
        Args:
            eval_config: Evaluation configuration from optimization schema
            config_file_path: Path to the config file (for loading local evaluators)
        """
        self.config = eval_config
        self.config_file_path = config_file_path
        self.custom_evaluator = None
        
        # Load custom evaluator if specified
        if eval_config.custom_evaluator.enabled:
            self._load_custom_evaluator()
    
    def _load_custom_evaluator(self) -> None:
        """
        Load custom evaluator function from module.
        
        Supports three loading strategies:
        1. Built-in evaluators from convergence.evaluators.* package
        2. Installed Python modules (standard import)
        3. Local evaluator files in the same directory as the config file
        """
        module_name = self.config.custom_evaluator.module
        function_name = self.config.custom_evaluator.function
        
        if not module_name or not function_name:
            raise ValueError("Custom evaluator requires both module and function names")
        
        # Strategy 1: Try built-in evaluators
        try:
            module = importlib.import_module(f"convergence.evaluators.{module_name}")
            self.custom_evaluator = getattr(module, function_name)
            return
        except (ImportError, AttributeError):
            pass
        
        # Strategy 2: Try standard module import
        try:
            module = importlib.import_module(module_name)
            self.custom_evaluator = getattr(module, function_name)
            return
        except (ImportError, AttributeError):
            pass
        
        # Strategy 3: Try loading from config file's directory
        if self.config_file_path:
            config_dir = self.config_file_path.parent
            evaluator_path = config_dir / f"{module_name}.py"
            
            if evaluator_path.exists():
                try:
                    # Load module from file path
                    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        self.custom_evaluator = getattr(module, function_name)
                        return
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Failed to load local evaluator from {evaluator_path}: {e}"
                    ) from e
        
        # If we get here, all strategies failed
        raise ValueError(
            f"Failed to load custom evaluator {module_name}.{function_name}\n"
            f"Tried:\n"
            f"  1. Built-in: convergence.evaluators.{module_name}\n"
            f"  2. Installed module: {module_name}\n"
            f"  3. Local file: {module_name}.py in config directory\n"
            f"\nTo add a custom evaluator:\n"
            f"  - Place {module_name}.py in the same directory as your config file, OR\n"
            f"  - Install it as a Python package, OR\n"
            f"  - Use a built-in evaluator from convergence.evaluators"
        )
    
    async def evaluate(
        self,
        response: APIResponse,
        test_case: Dict[str, Any],
        config_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate an API response against metrics.
        
        Args:
            response: API response to evaluate
            test_case: Test case with input and expected output
            config_params: Configuration parameters used for this call
            
        Returns:
            Dict of metric_name -> score pairs
        """
        scores = {}
        
        # If API call failed, FAIL HARD - don't return zeros
        if not response.success:
            error_msg = response.error or "Unknown API error"
            raise RuntimeError(
                f"API call failed: {error_msg}\n"
                f"This optimization requires successful API calls. "
                f"Please check your API configuration, credentials, and endpoint."
            )
        
        # Calculate each metric
        for metric_name, metric_config in self.config.metrics.items():
            score = await self._calculate_metric(
                metric_name,
                metric_config,
                response,
                test_case,
                config_params
            )
            scores[metric_name] = score
        
        return scores
    
    async def _calculate_metric(
        self,
        metric_name: str,
        metric_config: MetricConfig,
        response: APIResponse,
        test_case: Dict[str, Any],
        config_params: Dict[str, Any]
    ) -> float:
        """Calculate a single metric score."""
        
        # Built-in metrics
        if metric_name == "latency_ms":
            return self._score_latency(response.latency_ms, metric_config)
        
        elif metric_name == "cost_usd":
            return self._score_cost(response.estimated_cost_usd, metric_config)
        
        elif metric_name == "success_rate":
            return 1.0 if response.success else 0.0
        
        # Accuracy metrics require expected output
        elif metric_name == "accuracy":
            if "expected" not in test_case:
                raise ValueError(f"Test case missing 'expected' field for accuracy metric")
            
            if metric_config.function == "exact_match":
                return self._exact_match(response.result, test_case["expected"])
            
            elif metric_config.function == "similarity":
                return self._similarity(response.result, test_case["expected"])
            
            elif metric_config.function == "custom":
                if self.custom_evaluator:
                    return await self._call_custom_evaluator(
                        response.result,
                        test_case["expected"],
                        config_params
                    )
                else:
                    raise ValueError("Custom evaluator not configured")
        
        # Custom metric
        else:
            if self.custom_evaluator:
                return await self._call_custom_evaluator(
                    response.result,
                    test_case.get("expected"),
                    config_params,
                    metric_name=metric_name
                )
            else:
                raise ValueError(f"Unknown metric {metric_name} and no custom evaluator configured")
    
    def _score_latency(self, latency_ms: float, metric_config: MetricConfig) -> float:
        """
        Score latency (lower is better).
        
        Returns 0 if above threshold, otherwise normalized score.
        """
        threshold = metric_config.threshold
        
        if threshold and latency_ms > threshold:
            return 0.0
        
        # Normalize: faster = higher score
        # Assume reasonable range 0-5000ms
        max_latency = threshold if threshold else 5000.0
        normalized = 1.0 - (latency_ms / max_latency)
        return max(0.0, min(1.0, normalized))
    
    def _score_cost(self, cost_usd: float, metric_config: MetricConfig) -> float:
        """
        Score cost (lower is better).
        
        Returns 0 if above budget, otherwise normalized score.
        """
        budget = metric_config.budget_per_call
        
        if budget and cost_usd > budget:
            return 0.0
        
        # Normalize: cheaper = higher score
        max_cost = budget if budget else 1.0
        normalized = 1.0 - (cost_usd / max_cost)
        return max(0.0, min(1.0, normalized))
    
    def _exact_match(self, result: Any, expected: Any) -> float:
        """Check if result exactly matches expected output."""
        # Convert to strings for comparison
        result_str = json.dumps(result, sort_keys=True) if isinstance(result, (dict, list)) else str(result)
        expected_str = json.dumps(expected, sort_keys=True) if isinstance(expected, (dict, list)) else str(expected)
        
        return 1.0 if result_str == expected_str else 0.0
    
    def _similarity(self, result: Any, expected: Any) -> float:
        """Calculate similarity between result and expected output."""
        # Convert to strings for comparison
        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        expected_str = json.dumps(expected) if isinstance(expected, (dict, list)) else str(expected)
        
        # Use SequenceMatcher for string similarity
        ratio = SequenceMatcher(None, result_str, expected_str).ratio()
        return ratio
    
    async def _call_custom_evaluator(
        self,
        result: Any,
        expected: Any,
        config_params: Dict[str, Any],
        metric_name: Optional[str] = None
    ) -> float:
        """Call user-provided custom evaluator function."""
        if not self.custom_evaluator:
            raise ValueError("Custom evaluator not loaded")
        
        try:
            # Call custom function with standard signature
            score = self.custom_evaluator(
                result=result,
                expected=expected,
                params=config_params,
                metric=metric_name
            )
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(score)))
        
        except Exception as e:
            raise ValueError(f"Custom evaluator failed: {e}")
    
    def aggregate_scores(self, metric_scores: Dict[str, float]) -> float:
        """
        Aggregate multiple metric scores into a single weighted score.
        
        Args:
            metric_scores: Dict of metric_name -> score
            
        Returns:
            Weighted aggregate score between 0 and 1
        """
        total_weight = sum(m.weight for m in self.config.metrics.values())
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = 0.0
        for metric_name, score in metric_scores.items():
            if metric_name in self.config.metrics:
                weight = self.config.metrics[metric_name].weight
                weighted_sum += score * weight
        
        return weighted_sum / total_weight
    
    async def evaluate_with_aggregate(
        self,
        response: APIResponse,
        test_case: Dict[str, Any],
        config_params: Dict[str, Any]
    ) -> tuple[Dict[str, float], float]:
        """
        Evaluate and return both individual scores and aggregate.
        
        Returns:
            (metric_scores, aggregate_score)
        """
        # Use Weave-tracked version if available
        if WEAVE_AVAILABLE and weave:
            @weave.op()
            async def tracked_evaluation():
                metric_scores = await self.evaluate(response, test_case, config_params)
                aggregate = self.aggregate_scores(metric_scores)
                return metric_scores, aggregate
            
            return await tracked_evaluation()
        else:
            metric_scores = await self.evaluate(response, test_case, config_params)
            aggregate = self.aggregate_scores(metric_scores)
            return metric_scores, aggregate
    
    def passes_thresholds(self, metric_scores: Dict[str, float]) -> bool:
        """
        Check if all metric scores pass their thresholds.
        
        Args:
            metric_scores: Dict of metric_name -> score
            
        Returns:
            True if all thresholds are met
        """
        for metric_name, score in metric_scores.items():
            if metric_name in self.config.metrics:
                metric_config = self.config.metrics[metric_name]
                threshold = metric_config.threshold
                
                if threshold is not None:
                    if metric_config.type == "higher_is_better":
                        if score < threshold:
                            return False
                    elif metric_config.type == "lower_is_better":
                        if score > threshold:
                            return False
        
        return True

