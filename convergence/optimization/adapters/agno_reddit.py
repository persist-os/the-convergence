"""
Agno Reddit Agent Adapter for The Convergence.

Bridges the convergence optimization framework with Agno Reddit agents,
allowing agent-based tool execution instead of direct HTTP calls.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..models import APIResponse

logger = logging.getLogger(__name__)


class AgnoRedditAdapter:
    """
    Adapter for Agno Reddit agents.
    
    Executes Agno agents with Reddit tools instead of making HTTP calls.
    Dynamically loads RedditAgentRunner from the config directory.
    """
    
    def __init__(self, config: Dict[str, Any], config_file_path: Optional[Path] = None):
        """
        Initialize the Agno Reddit adapter.
        
        Args:
            config: Full optimization configuration
            config_file_path: Path to the config YAML file
        """
        self.config = config
        self.config_file_path = config_file_path
        self.runner = None
        
        # Load the runner from the config directory
        self._load_runner()
    
    def _load_runner(self):
        """Dynamically import RedditAgentRunner from the config directory."""
        if not self.config_file_path:
            raise RuntimeError("config_file_path required for AgnoRedditAdapter")
        
        # Config directory is where the YAML file is located
        config_dir = self.config_file_path.parent
        runner_path = config_dir / "reddit_agent_runner.py"
        
        logger.info(f"Loading runner from: {runner_path}")
        
        if not runner_path.exists():
            raise FileNotFoundError(
                f"RedditAgentRunner not found at {runner_path}. "
                f"Ensure reddit_agent_runner.py exists in the config directory."
            )
        
        try:
            # Add config directory to Python path
            sys.path.insert(0, str(config_dir))
            logger.debug(f"Added {config_dir} to Python path")
            
            # Import the runner module
            logger.debug("Creating module spec...")
            spec = importlib.util.spec_from_file_location("reddit_agent_runner", runner_path)
            if spec and spec.loader:
                logger.debug("Loading module...")
                module = importlib.util.module_from_spec(spec)
                
                logger.debug("Executing module...")
                spec.loader.exec_module(module)
                
                logger.debug("Getting RedditAgentRunner class...")
                # Get the RedditAgentRunner class
                runner_class = getattr(module, "RedditAgentRunner", None)
                if not runner_class:
                    raise AttributeError("RedditAgentRunner class not found in module")
                
                logger.debug(f"Initializing runner with config keys: {list(self.config.keys())}")
                # Initialize the runner with config
                self.runner = runner_class(self.config)
                logger.info("✅ RedditAgentRunner initialized successfully")
            else:
                raise ImportError(f"Could not create module spec from {runner_path}")
                
        except Exception as e:
            logger.error(f"Failed to load RedditAgentRunner: {type(e).__name__}: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot import RedditAgentRunner from {runner_path}. "
                f"Error: {type(e).__name__}: {e}"
            ) from e
    
    def transform_request(self, config_params: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Agno agent instead of making an HTTP request.
        
        This replaces the standard HTTP call flow with agent execution.
        
        Args:
            config_params: Configuration parameters (model, temperature, etc.)
            test_case: Test case with input and expected output
            
        Returns:
            Dict containing agent execution results
        """
        if not self.runner:
            raise RuntimeError("RedditAgentRunner not initialized")
        
        try:
            logger.info(f"🤖 Executing Agno agent for test: {test_case.get('name', 'unknown')}")
            
            # Execute the agent
            result = self.runner.run_test(test_case, config_params)
            
            # Check if the result contains an error
            if result.get('error'):
                logger.error(f"❌ Agent execution failed: {result.get('error')}")
            else:
                logger.info("✅ Agent execution successful")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            return {
                "error": str(e),
                "final_response": None,
                "tool_calls": [],
                "tool_results": [],
                "latency_seconds": 0.0,
                "tokens_used": 0
            }
    
    def transform_response(self, response: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform agent response into standardized format.
        
        Ensures the response format is compatible with the evaluator.
        
        Args:
            response: Raw agent execution results
            config: Optional config parameters (unused for Agno)
            
        Returns:
            Standardized response format
        """
        # Response from runner is already in correct format
        # Just ensure required fields exist
        return {
            "final_response": response.get("final_response"),
            "tool_calls": response.get("tool_calls", []),
            "tool_results": response.get("tool_results", []),
            "latency_seconds": response.get("latency_seconds", 0.0),
            "tokens_used": response.get("tokens_used", 0),
            "error": response.get("error")
        }
    
    @staticmethod
    def is_compatible(config: Dict[str, Any]) -> bool:
        """
        Check if this adapter is compatible with the given config.
        
        Args:
            config: API configuration
            
        Returns:
            True if this adapter should be used
        """
        api_name = config.get("api", {}).get("name", "").lower()
        return "agno" in api_name and "reddit" in api_name
