"""
Local Function Adapter for Convergence.

Allows optimizing internal Python functions without HTTP endpoints.
Uses same pattern as UniversalAgentAdapter but for arbitrary callables.
"""

import time
import logging
from typing import Dict, Any, Callable, Optional
import importlib
from ..models import APIResponse

logger = logging.getLogger(__name__)


class LocalFunctionAdapter:
    """
    Adapter for optimizing local Python functions.
    
    Executes callable functions directly instead of making HTTP calls.
    Useful for optimizing internal services and parameters.
    """
    
    def __init__(self, func: Optional[Callable] = None, func_path: Optional[str] = None):
        """
        Initialize local function adapter.
        
        Args:
            func: Direct callable function (programmatic mode)
            func_path: Dotted path to function (e.g., "module.function_name")
        """
        self.func = func
        self.func_path = func_path
        
        # If func_path provided, resolve it
        if not self.func and self.func_path:
            self.func = self._resolve_function(self.func_path)
        
        if not self.func:
            raise ValueError("Either func or func_path must be provided")
    
    def _resolve_function(self, func_path: str) -> Callable:
        """Resolve function from dotted path string."""
        module_path, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    
    def transform_request(self, config_params: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute local function with config params and test case input.
        
        Args:
            config_params: Optimization parameters (e.g., {"threshold": 0.35, "limit": 5})
            test_case: Test case with input data (e.g., {"input": {"user_id": "...", "message": "..."}})
        
        Returns:
            Dict with function execution results in APIResponse format
        """
        try:
            start_time = time.time()
            
            # Extract test case input
            test_input = test_case.get("input", {})
            
            # Merge config params with test input
            # Function should receive merged params: {**config_params, **test_input}
            func_params = {**config_params, **test_input}
            
            # Execute function (handle both sync and async)
            logger.debug(f"Executing local function with params: {func_params.keys()}")
            
            import asyncio
            import inspect
            
            if inspect.iscoroutinefunction(self.func):
                # Async function - run in event loop
                # Since transform_request is sync but called from async context,
                # we check if there's a running loop
                try:
                    # Try to get running loop (will raise RuntimeError if none)
                    loop = asyncio.get_running_loop()
                    # If we have a running loop, create task and wait
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.func(**func_params))
                        result = future.result(timeout=30)
                except RuntimeError:
                    # No running loop, create one
                    result = asyncio.run(self.func(**func_params))
            else:
                # Sync function - execute directly
                result = self.func(**func_params)
            
            latency_seconds = time.time() - start_time
            
            # Convert result to expected format
            if isinstance(result, dict):
                execution_result = {
                    "success": True,
                    "result": result,
                    "latency_seconds": latency_seconds,
                    "error": None
                }
            else:
                # Wrap non-dict results
                execution_result = {
                    "success": True,
                    "result": {"output": result},
                    "latency_seconds": latency_seconds,
                    "error": None
                }
            
            logger.debug(f"Local function executed successfully in {latency_seconds:.3f}s")
            return execution_result
            
        except Exception as e:
            latency_seconds = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Local function execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": None,
                "latency_seconds": latency_seconds,
                "error": str(e)
            }
    
    def transform_response(self, response: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform response to standardize format (no-op for local functions).
        
        Args:
            response: Function execution response
            config: Optional config params (unused)
        
        Returns:
            Standardized response dict
        """
        # Response is already in correct format from transform_request
        return response
    
    @staticmethod
    def is_compatible(config: Dict[str, Any]) -> bool:
        """
        Check if this adapter is compatible with the given config.
        
        Args:
            config: API configuration
        
        Returns:
            True if this is a local function configuration
        """
        api_name = config.get("api", {}).get("name", "").lower()
        return api_name.startswith("local_")

