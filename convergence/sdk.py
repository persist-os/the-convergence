"""
Convergence SDK - Simple programmatic interface for optimization runs.

Provides a clean API for backend services to run Convergence optimizations
without dealing with CLI or complex configuration objects.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile
import asyncio

from convergence.optimization.config_loader import ConfigLoader
from convergence.optimization.runner import OptimizationRunner
from convergence.optimization.models import OptimizationResult


async def run_optimization(
    config_dict: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Convergence optimization programmatically.
    
    Simple interface for backend services to run optimization without CLI.
    Takes either a config dictionary or path to YAML config file.
    
    Args:
        config_dict: Configuration as dictionary (mutually exclusive with yaml_path)
        yaml_path: Path to YAML configuration file (mutually exclusive with config_dict)
    
    Returns:
        Dictionary with optimization results:
        {
            "success": bool,
            "best_config": dict,          # Winning parameter configuration
            "best_score": float,          # Score achieved (0.0-1.0)
            "configs_generated": int,     # Total configurations tested
            "optimization_run_id": str,   # Unique run identifier
            "configs_saved": int,         # Number of configs saved to storage
            "error": str (optional)       # Error message if failed
        }
    
    Example:
        >>> result = await run_optimization(
        ...     config_dict={
        ...         "api": {
        ...             "name": "context_enrichment",
        ...             "endpoint": "http://localhost:8000/enrich",
        ...             "mock_mode": True
        ...         },
        ...         "search_space": {
        ...             "parameters": {
        ...                 "threshold": {"type": "float", "min": 0.1, "max": 0.5},
        ...                 "limit": {"type": "int", "min": 5, "max": 20}
        ...             }
        ...         },
        ...         "evaluation": {
        ...             "test_cases": {"inline": [{"input": {...}, "expected": {...}}]},
        ...             "metrics": {"accuracy": {"weight": 1.0}}
        ...         }
        ...     }
        ... )
        >>> print(result["best_config"])
        {"threshold": 0.35, "limit": 15}
    """
    
    if config_dict is None and yaml_path is None:
        return {
            "success": False,
            "error": "Must provide either config_dict or yaml_path",
            "configs_generated": 0,
            "configs_saved": 0
        }
    
    if config_dict is not None and yaml_path is not None:
        return {
            "success": False,
            "error": "Provide only one of config_dict or yaml_path, not both",
            "configs_generated": 0,
            "configs_saved": 0
        }
    
    try:
        # Load configuration
        if yaml_path:
            config_path = Path(yaml_path)
            if not config_path.exists():
                return {
                    "success": False,
                    "error": f"Config file not found: {yaml_path}",
                    "configs_generated": 0,
                    "configs_saved": 0
                }
            config = ConfigLoader.load(config_path)
            config_file_path = config_path
        else:
            # Create temporary YAML file from dict
            # (ConfigLoader expects a file path for relative path resolution)
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.yaml',
                delete=False
            ) as f:
                import yaml
                yaml.dump(config_dict, f)
                temp_path = Path(f.name)
            
            try:
                config = ConfigLoader.load(temp_path)
                config_file_path = temp_path
            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
        
        # Create optimization runner
        runner = OptimizationRunner(config, config_file_path=config_file_path)
        
        # Run optimization
        result: OptimizationResult = await runner.run()
        
        # Generate run ID (timestamp + api name)
        import time
        import uuid
        optimization_run_id = f"{config.api.name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Return simplified result
        return {
            "success": True,
            "best_config": result.best_config,
            "best_score": result.best_score,
            "configs_generated": len(result.all_results),
            "configs_saved": len(result.all_results),  # All are saved by OptimizationRunner
            "optimization_run_id": optimization_run_id,
            "generations_run": result.generations_run,
            "timestamp": result.timestamp.isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "configs_generated": 0,
            "configs_saved": 0
        }


def load_config_from_file(yaml_path: str) -> Dict[str, Any]:
    """
    Load and validate Convergence config from YAML file.
    
    Useful for validating configs before running optimization.
    
    Args:
        yaml_path: Path to YAML configuration file
    
    Returns:
        Validated configuration as dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(yaml_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    config = ConfigLoader.load(config_path)
    return config.dict()


__all__ = ["run_optimization", "load_config_from_file"]

