"""
Universal Agno Agent Adapter for The Convergence.

Consolidates all Agno agent adapters (Discord, Gmail, Reddit, and future agents)
into a single universal adapter that auto-discovers the appropriate agent runner.

Bridges the convergence optimization framework with any Agno agent implementation,
allowing agent-based tool execution instead of direct HTTP calls.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..models import APIResponse

logger = logging.getLogger(__name__)


class UniversalAgentAdapter:
    """
    Universal adapter for all Agno agent implementations.
    
    Auto-discovers and loads the appropriate agent runner from the config directory.
    Works with any agent that follows the BaseAgentRunner pattern:
    - DiscordAgentRunner
    - GmailAgentRunner  
    - RedditAgentRunner
    - Any future agent runners
    
    Dynamically loads the runner based on config file location.
    """
    
    # Agent runner patterns to search for
    AGENT_RUNNER_PATTERNS = [
        "discord_agent_runner.py",
        "gmail_agent_runner.py",
        "reddit_agent_runner.py",
        # Can be extended for future agents:
        # "slack_agent_runner.py",
        # "jira_agent_runner.py",
        # etc.
    ]
    
    AGENT_CLASS_NAMES = {
        "discord_agent_runner.py": "DiscordAgentRunner",
        "gmail_agent_runner.py": "GmailAgentRunner",
        "reddit_agent_runner.py": "RedditAgentRunner",
        # Can be extended for future agents:
        # "slack_agent_runner.py": "SlackAgentRunner",
    }
    
    def __init__(self, config: Dict[str, Any], config_file_path: Optional[Path] = None):
        """
        Initialize the universal agent adapter.
        
        Args:
            config: Full optimization configuration
            config_file_path: Path to the config YAML file
        """
        self.config = config
        self.config_file_path = config_file_path
        self.runner = None
        self.runner_class_name = None
        
        # Load the appropriate runner from the config directory
        self._load_runner()
    
    def _load_runner(self):
        """Auto-discover and load the agent runner from the config directory."""
        if not self.config_file_path:
            raise RuntimeError("config_file_path required for UniversalAgentAdapter")
        
        config_dir = self.config_file_path.parent
        runner_path = None
        runner_class_name = None
        
        # Determine which service we're optimizing from config
        api_name = self.config.get('api', {}).get('name', '').lower()
        service_type = None
        
        # Infer service type from API name or config
        if 'reddit' in api_name:
            service_type = 'reddit'
        elif 'gmail' in api_name or 'email' in api_name:
            service_type = 'gmail'
        elif 'discord' in api_name:
            service_type = 'discord'
        
        # Try to find the specific agent runner for this service
        if service_type:
            expected_pattern = f"{service_type}_agent_runner.py"
            candidate = config_dir / expected_pattern
            if candidate.exists():
                runner_path = candidate
                runner_class_name = self.AGENT_CLASS_NAMES.get(expected_pattern)
                logger.info(f"Found {service_type} agent runner: {expected_pattern}")
        
        # Fallback: search all patterns
        if not runner_path:
            for pattern in self.AGENT_RUNNER_PATTERNS:
                candidate = config_dir / pattern
                if candidate.exists():
                    runner_path = candidate
                    runner_class_name = self.AGENT_CLASS_NAMES.get(pattern)
                    logger.info(f"Found agent runner: {pattern}")
                    break
        
        if not runner_path or not runner_path.exists():
            # Try to find any _agent_runner.py file as fallback
            for file in config_dir.glob("*_agent_runner.py"):
                runner_path = file
                # Try to extract class name from filename
                runner_class_name = file.stem.replace("_", " ").title().replace(" ", "")
                logger.info(f"Found agent runner: {file.name}")
                break
        
        if not runner_path or not runner_path.exists():
            available_files = list(config_dir.glob("*.py"))
            raise FileNotFoundError(
                f"No agent runner found in {config_dir}. "
                f"Expected one of: {self.AGENT_RUNNER_PATTERNS}\n"
                f"Available Python files: {[f.name for f in available_files]}"
            )
        
        logger.info(f"Loading runner from: {runner_path}")
        
        try:
            # Add config directory to Python path
            sys.path.insert(0, str(config_dir))
            logger.debug(f"Added {config_dir} to Python path")
            
            # Import the runner module
            spec = importlib.util.spec_from_file_location(
                runner_path.stem, runner_path
            )
            
            if not spec or not spec.loader:
                raise ImportError(f"Could not create module spec from {runner_path}")
            
            logger.debug("Loading module...")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the runner class by name
            logger.debug(f"Looking for runner class: {runner_class_name}")
            runner_class = getattr(module, runner_class_name, None)
            
            if not runner_class:
                raise AttributeError(
                    f"{runner_class_name} class not found in {runner_path}. "
                    f"Available classes: {[name for name in dir(module) if 'Agent' in name]}"
                )
            
            logger.debug(f"Initializing {runner_class_name} with config")
            self.runner = runner_class(self.config)
            self.runner_class_name = runner_class_name
            logger.info(f"✅ {runner_class_name} initialized successfully")
            
        except Exception as e:
            logger.error(
                f"Failed to load agent runner: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Cannot import {runner_class_name} from {runner_path}. "
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
            raise RuntimeError(f"{self.runner_class_name} not initialized")
        
        try:
            logger.info(f"🤖 Executing {self.runner_class_name} for test: {test_case.get('id', 'unknown')}")
            
            # Execute the agent
            result = self.runner.run_test(test_case, config_params)
            
            # Check if the result contains an error
            if result.get('error'):
                logger.error(f"❌ Agent execution failed: {result.get('error')}")
            else:
                logger.info("✅ Agent execution successful")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "final_response": None,
                "tool_calls": [],
                "tool_results": [],
                "latency_seconds": 0.0,
                "tokens_used": {}
            }
    
    def transform_response(self, response: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform agent response into standardized format.
        
        Ensures the response format is compatible with the evaluator.
        
        Args:
            response: Raw agent execution results
            config: Optional config parameters (unused for agents)
            
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
            "tokens_used": response.get("tokens_used", {}),
            "error": response.get("error")
        }
    
    @staticmethod
    def is_compatible(config: Dict[str, Any]) -> bool:
        """
        Check if this adapter is compatible with the given config.
        
        Args:
            config: API configuration
            
        Returns:
            True if this is an Agno agent configuration
        """
        api_name = config.get("api", {}).get("name", "").lower()
        return "agno" in api_name

